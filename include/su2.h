/* su2.h -- shared types and entry points.
 *
 * Ground truth: paper.tex (arxiv 2605.23923), summarised in
 * notes/section_2_3_background.md and notes/section_4_algorithm.md.
 *
 * Conventions (paper lines in parentheses):
 *   - Euler-angle grid (line 684):
 *       phi[j1] = -pi + j1 * 2pi/(N-1),    j1 in [0, N-1]
 *       psi[j2] = -pi + j2 * 2pi/(N-1),    j2 in [0, N-1]
 *       theta[k] = k * pi/(N-1),           k in [0, N-1]
 *   - Bandlimit N: f-hat(l)_{mn} = 0 for l >= N.  Integer l only here.
 *   - Sample storage: f[j1*N*N + k*N + j2] (row-major in (j1, k, j2)).
 *   - Coefficient storage: fhat[l] is a (2l+1) x (2l+1) complex matrix,
 *       fhat[l][(m+l) * (2l+1) + (n+l)], indices m,n in [-l, l].
 *   - Haar prefactor: 1/(8 pi^2)  applied after the triple sum.
 *   - Quadrature: Riemann sum, weight sin(theta_k) * dphi * dtheta * dpsi
 *       with dphi = dpsi = 2pi/(N-1) and dtheta = pi/(N-1).
 */
#ifndef SU2_H
#define SU2_H

#include <complex.h>
#include <stddef.h>

/* ------- Coefficient layout helpers ------- */
/* Number of independent matrix entries at degree l. */
static inline size_t su2_dim_l(int l) { return (size_t)(2*l + 1) * (size_t)(2*l + 1); }

/* Offset of fhat[l] in a flat array fhat[0..total_coeffs(N)-1]. */
size_t su2_coeff_offset(int l);

/* Total number of coefficients across l = 0..N-1. */
size_t su2_total_coeffs(int N);

/* Index into the (2l+1) x (2l+1) matrix for f-hat(l)_{mn}, m,n in [-l, l]. */
static inline size_t su2_mn_index(int l, int m, int n)
{
    return (size_t)(m + l) * (size_t)(2*l + 1) + (size_t)(n + l);
}

/* Sample index: f stored as f[j1*N*N + k*N + j2]. */
static inline size_t su2_sample_index(int N, int j1, int k, int j2)
{
    return (size_t)j1 * (size_t)N * (size_t)N + (size_t)k * (size_t)N + (size_t)j2;
}

/* ------- Grid ------- */
/* Allocate and fill the three Euler-angle grids (length N each).
 * Caller frees with free(). */
double *su2_grid_phi(int N);
double *su2_grid_theta(int N);
double *su2_grid_psi(int N);

/* ------- Gauss-Legendre quadrature on [-1, 1] -------
 *
 * Compute N-point GL nodes and weights for exact integration of polynomials
 * up to degree 2N-1.  Used by su2_fft_gl / su2_fft_inv_gl (bead su2fft-ega)
 * to integrate the theta direction exactly under the substitution
 * x = cos(theta).
 *
 * @param[in]  N  Number of nodes (>= 1).
 * @param[out] x  Length-N array of nodes, ascending in [-1, 1].
 * @param[out] w  Length-N array of weights.
 *
 * See notes/gauss_legendre.md §3.
 */
void su2_gl_nodes_weights(int N, double *x, double *w);

/* ------- Wigner small-d / matrix coefficient P^l_{mn}(cos theta) -------
 *
 * Returns the value P^l_{mn}(cos theta_k) defined by the paper at line 537.
 * For integer l with -l <= m,n <= l this equals the standard Wigner small-d
 * function d^l_{mn}(theta) up to a real normalisation factor that depends
 * only on (l, m, n) -- chosen so that t^l_{nm}(g) is unitary across SU(2).
 *
 * Internally this evaluates the closed-form sum (Wigner's formula), suitable
 * for the O(N^6) reference path.  Stable for small l (l <= 50 or so).
 */
double _Complex su2_wigner_d(int l, int n, int m, double theta);

/* ------- Half-integer-compatible Wigner-d (bead su2fft-n8e, Tier 1) -------
 *
 * Same as su2_wigner_d but arguments are 2l, 2n, 2m (integers) so that
 * half-integer l (2l odd) is representable without floating-point
 * comparison.  Physical (l, n, m) = (two_l/2, two_n/2, two_m/2).
 *
 * Constraints: two_l >= 0, |two_n| <= two_l, |two_m| <= two_l, and all of
 * two_l, two_n, two_m must share parity (all even = integer l; all odd =
 * half-integer l).  For integer l (two_l even) the value matches
 * su2_wigner_d(l, n, m, theta) up to tgamma-vs-factorial-table FP.
 *
 * @par Reference paper.tex line 537; notes/half_integer.md.
 *
 * NOTE: this is Tier 1 of bead `su2fft-n8e` -- evaluation only.  The
 * half-integer FFT itself (forward/inverse on a 4pi-period phi/psi grid)
 * is deferred to a follow-up bead.
 */
double _Complex su2_wigner_d_half(int two_l, int two_n, int two_m, double theta);

/* ------- Wigner small-d sequence via three-term recurrence -------
 *
 * Fills out_d[0..l_max-l_min] with the REAL Sakurai-convention Wigner
 * small-d values d^l_{n,m}(theta) for l = l_min .. l_max, using the
 * forward (ascending-l) three-term recurrence derived from the Jacobi
 * polynomial recurrence (DLMF 18.9.1/18.9.2 lifted to d^l via the
 * normalisation R(l)).  See notes/wigner_recurrence.md.
 *
 * Convention reminder: the paper's P^l_{n,m} = i^{m-n} * d^l_{n,m};
 * this routine returns the real d.  The caller applies the phase.
 *
 * Requirements:
 *   - l_min >= max(|n|, |m|).  Below max(|n|,|m|) the value is zero by
 *     definition; the caller should not request those.
 *   - l_max >= l_min.
 *   - out_d points to a buffer of length (l_max - l_min + 1) doubles.
 *
 * Cost: O(l_max - l_min) flops.  Seeds two values via wigner_d_phys
 * (each O(1) terms in the de Moivre sum at l = l_min, l_min+1) then
 * O(1) per recurrence step.
 */
void su2_wigner_d_seq(int l_min, int l_max, int n, int m, double theta,
                      double *out_d);

/* ------- Direct (reference) Fourier transform on SU(2) -------
 * Cost: O(N^6).  Used as ground truth for the fast algorithm.
 *
 * f:     length N*N*N complex samples on the Euler grid.
 * fhat:  length su2_total_coeffs(N) complex coefficients, filled on return.
 */
void su2_ft_direct(int N,
                   const double _Complex *f,
                   double _Complex *fhat);

/* ------- Fast Fourier transform on SU(2) -------
 * Cost: O(N^4)  -- Stage 1 (2D FFT via FFTW) + Stage 2 (beta recursion).
 *
 * Produces (within floating-point tolerance) the same fhat as su2_ft_direct.
 */
void su2_fft(int N,
             const double _Complex *f,
             double _Complex *fhat);

/* ------- Inverse FFT (Peter-Weyl synthesis) -------
 * Cost: O(N^4) -- mirrors su2_fft structurally.
 *
 * Computes f from fhat via the Peter-Weyl synthesis:
 *   f(g) = sum_l (2l+1) sum_{m,n} fhat(l)_{m,n} * t^l_{n,m}(g)
 *
 * Stage 2-inv: per (m,n), Wigner recurrence sweeps l producing
 *   G[k,n,m] = i^{m-n} * sum_l (2l+1) * fhat(l)_{m,n} * d^l_{n,m}(theta_k)
 * Stage 1-inv: per theta slice, 2D FORWARD FFTW (size (N-1)x(N-1))
 *   with the closed-grid fold trick mirrored from Stage 1.
 *
 * Tolerance: this is the exact discrete synthesis. The roundtrip
 *   ||fft_inv(fft(f)) - f||
 * is bounded BELOW by the closed-grid Riemann theta quadrature error
 * (~(N/(N-1))^2 at l=0; higher for higher l). For exact roundtrip,
 * use Gauss-Legendre theta nodes (bead su2fft-ega).
 *
 * @param[in]  N     Bandlimit.
 * @param[in]  fhat  Length-su2_total_coeffs(N) complex coefficient array.
 * @param[out] f     Length-N^3 complex sample array, row-major (j1, k, j2).
 */
void su2_fft_inv(int N,
                 const double _Complex *fhat,
                 double _Complex *f);

/* ------- Gauss-Legendre variant FFTs (bead su2fft-ega) -------
 *
 * Same algorithm as su2_fft / su2_fft_inv but uses N-point Gauss-Legendre
 * quadrature in theta instead of the closed-grid Riemann sum. The theta
 * sample points are theta_k = arccos(x_k) where x_k is the k-th GL node;
 * phi and psi grids are unchanged (closed grid).
 *
 * Spec: notes/gauss_legendre.md.
 *
 * Pair `su2_fft_gl` and `su2_fft_inv_gl` together (the sample layout for f
 * MUST use GL theta nodes for both directions).
 *
 * @param[in]  N     Bandlimit.
 * @param[in]  f / fhat  Length-N^3 / length-su2_total_coeffs(N) complex array.
 * @param[out] fhat / f  Output of the corresponding size.
 */
void su2_fft_gl(int N, const double _Complex *f, double _Complex *fhat);
void su2_fft_inv_gl(int N, const double _Complex *fhat, double _Complex *f);

/* ------- Resolved-grid forward/inverse (bead su2fft-0t1) -------
 *
 * The default `su2_fft` and `su2_fft_gl` use an N-point CLOSED uniform
 * phi/psi grid; modes |m|, |n| > (N-1)/2 alias.  The resolved variant uses
 * the OPEN P-point grid with P = 2N-1, exactly matching the SU(2) bandlimit
 * of 2N-1 modes per axis.  Combined with Gauss-Legendre theta nodes, this
 * gives exact spectrum roundtrip at working precision.
 *
 * Sample storage:  f[j1 * N * P + k * P + j2], P = 2N-1.
 *                  j1 in [0, P-1], k in [0, N-1], j2 in [0, P-1].
 *                  phi[j1] = -pi + j1 * 2pi/P,  psi[j2] = -pi + j2 * 2pi/P,
 *                  theta[k] = arccos(x_k), x_k the k-th N-point GL node.
 *
 * Coefficient storage: same as su2_total_coeffs(N) layout.
 *
 * See notes/0t1_resolved_grid_design.md.
 */
static inline int su2_resolved_P(int N) { return 2*N - 1; }

static inline size_t su2_resolved_total_samples(int N)
{
    size_t P = (size_t)(2*N - 1);
    return P * P * (size_t)N;
}

static inline size_t su2_resolved_sample_index(int N, int j1, int k, int j2)
{
    int P = 2*N - 1;
    return (size_t)j1 * (size_t)N * (size_t)P + (size_t)k * (size_t)P + (size_t)j2;
}

/* O(N^6) reference FT on the resolved grid.  Ground truth for the cross-check
 * test of su2_fft_resolved.  Mirrors su2_ft_direct but on the new grid. */
void su2_ft_direct_resolved(int N,
                            const double _Complex *f,
                            double _Complex *fhat);

/* O(N^4) forward / inverse FFT on the resolved grid.  Pair these together --
 * the sample layout f[j1*N*P + k*P + j2] (P = 2N-1) is shared.  Inverse is the
 * exact Peter-Weyl synthesis at the resolved phi/psi + GL theta points; forward
 * is the (exact, for bandlimited inputs) analysis with norm = 1/(2 P^2).
 * See notes/0t1_resolved_grid_design.md §4-§5. */
void su2_fft_resolved(int N,
                      const double _Complex *f,
                      double _Complex *fhat);
void su2_fft_resolved_inv(int N,
                          const double _Complex *fhat,
                          double _Complex *f);

/* ------- Convolution via the spectrum (bead d7v) -------
 *
 * SU(2) convolution f * g via the Peter-Weyl convolution theorem:
 * the spectrum of f*g is the per-l matrix product of the spectra.
 *
 * fghat(l)_{mn} = sum_p fhat(l)_{mp} * ghat(l)_{pn}
 *
 * @param[in]  N      Bandlimit (l in [0, N-1]).
 * @param[in]  fhat   Length su2_total_coeffs(N).
 * @param[in]  ghat   Length su2_total_coeffs(N).
 * @param[out] fghat  Length su2_total_coeffs(N). May alias fhat or ghat.
 *
 * @par Complexity O(N^4).
 */
void su2_convolve(int N,
                  const double _Complex *fhat,
                  const double _Complex *ghat,
                  double _Complex *fghat);

/* ------- Spherical-harmonic FFT on S^2 (bead 5fb) -------
 *
 * S^2 = SU(2)/U(1). Functions on the sphere extend to psi-independent
 * functions on SU(2); their spectrum lives entirely on the m=0 row of
 * each fhat(l) block.
 *
 * Storage:
 *   f_sph[j1*N + k]       row-major (phi_index, theta_index)
 *   fhat_sph[ sum_{l'<l}(2l'+1) + (n+l) ]   l in [0, N-1], n in [-l, l]
 *   Total spectrum entries: N^2.
 *
 * Thin wrapper over su2_fft / su2_fft_inv; same closed-grid Riemann
 * tolerance and phi/psi aliasing floor (bead su2fft-0t1).
 */
size_t su2_sphere_total_coeffs(int N);

void su2_fft_sphere(int N,
                    const double _Complex *f_sph,
                    double _Complex *fhat_sph);

void su2_fft_sphere_inv(int N,
                        const double _Complex *fhat_sph,
                        double _Complex *f_sph);

/* ------- Resolved-grid spherical-harmonic FFT (bead su2fft-9qk) -------
 *
 * Sphere-FFT variant built over `su2_fft_resolved` instead of `su2_fft`.
 * Same spectrum layout (N^2 entries, m=0 row per fhat(l) block), but the
 * sample grid is the OPEN P-point phi grid (P = 2N-1) and the N-point
 * Gauss-Legendre theta grid:
 *
 *   f_sph_resolved[j1 * N + k]   row-major (phi_index, theta_index)
 *   j1 in [0, P-1],  k in [0, N-1].
 *   phi[j1]   = -pi + j1 * 2pi/P
 *   theta[k]  = arccos(x_k), x_k the k-th N-point GL node.
 *
 * Total sample count: P * N = (2N-1) * N (vs N*N for the closed wrapper).
 *
 * Inherits the exact spectrum roundtrip property of `su2_fft_resolved`
 * (bead su2fft-0t1): forward(inverse(fhat)) = fhat to working precision.
 *
 * See notes/0t1_resolved_grid_design.md §3-§5.
 */
size_t su2_sphere_resolved_total_samples(int N);

void su2_fft_sphere_resolved(int N,
                             const double _Complex *f_sph,
                             double _Complex *fhat_sph);

void su2_fft_sphere_inv_resolved(int N,
                                 const double _Complex *fhat_sph,
                                 double _Complex *f_sph);

#endif /* SU2_H */
