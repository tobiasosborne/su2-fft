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

#endif /* SU2_H */
