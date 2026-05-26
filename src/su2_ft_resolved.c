/* su2_ft_resolved.c -- Direct O(N^6) Fourier transform on the resolved grid.
 *
 * Bead: su2fft-0t1 (Step 3).  Binding spec: notes/0t1_resolved_grid_design.md.
 *
 * Mirrors src/su2_ft.c verbatim except for the grid and the resulting
 * quadrature weights:
 *
 *   phi[j1]  = -pi + j1 * 2pi/P,   j1 in [0, P-1],   P = 2N-1   (OPEN grid)
 *   psi[j2]  = -pi + j2 * 2pi/P,   j2 in [0, P-1]              (OPEN grid)
 *   theta[k] = arccos(x_k),        x_k the k-th N-point GL node on [-1, 1]
 *
 * Sample storage: f[j1*N*P + k*P + j2]  (via su2_resolved_sample_index).
 * Total samples:  P*P*N = (2N-1)^2 * N.
 *
 * Paper.tex line 1316 (verbatim, ground truth):
 *
 *   fhat(l)_{mn} ~= sum_{i=1..N^3} f(g_i) * conj(t^l_{mn}(g_i))
 *
 * (Each "i" is a (j1, k, j2) triple; the quadrature weight per sample is
 * absorbed into a single global "norm" below.)  With
 *
 *   t^l_{n,m}(phi, theta, psi) = P^l_{n,m}(cos theta) * exp(-i(n phi + m psi))
 *
 * the discrete sum unfolds as
 *
 *   fhat(l)_{m,n}
 *     = norm * sum_{j1, k, j2}
 *           f(g_{j1,k,j2})
 *         * conj(P^l_{n,m}(cos theta_k))
 *         * exp(+i n phi[j1])
 *         * exp(+i m psi[j2])
 *         * w_k
 *
 * with w_k the N-point Gauss-Legendre weight at x_k = cos(theta_k).  The
 * substitution x = cos(theta) absorbs the sin(theta) Jacobian of the Haar
 * measure into w_k, so NO explicit sin(theta_k) factor appears here (this
 * is the one structural difference from src/su2_ft.c).
 *
 * Normalisation derivation (notes/0t1_resolved_grid_design.md §5):
 *
 *   For constant input f == 1:
 *     fhat(0)_{0,0} = norm * sum_{j1, k, j2} 1 * 1 * 1 * w_k
 *                   = norm * P * P * sum_k w_k
 *                   = norm * P^2 * 2          (GL weights sum to 2 on [-1,1])
 *     -> require fhat(0)_{0,0} = 1
 *     -> norm = 1 / (2 * P^2).
 *
 *   Equivalently: norm = (1/(8 pi^2)) * dphi * dpsi * 1
 *                      = (1/(8 pi^2)) * (2pi/P)^2
 *                      = 1 / (2 * P^2),
 *   where the trailing "* 1" is the [-1,1]-measure factor (the GL sum
 *   already integrates `1 dx` to 2, matching the Haar `sin(theta) dtheta`
 *   integral 2 over [0, pi]).
 *
 * O(N^6) brute force; reference path only.  Cross-checked against
 * su2_fft_resolved (Step 4) in tests/test_resolved.c (Step 5).
 */
#include "su2.h"

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>

/**
 * @brief Direct O(N^6) Fourier transform on SU(2) -- resolved grid.
 *
 * Brute-force triple sum on the OPEN P-point phi/psi grid (P = 2N-1) and
 * the N-point Gauss-Legendre theta grid.  Used as ground truth for the
 * cross-check of su2_fft_resolved (bead su2fft-0t1).
 *
 * @param[in]  N     Bandlimit; sample grid is P x N x P, P = 2N-1.
 * @param[in]  f     Length-(P*P*N) complex sample array, row-major
 *                   (j1, k, j2); index via su2_resolved_sample_index.
 * @param[out] fhat  Length-su2_total_coeffs(N) complex coefficient array.
 * @par Complexity O(N^6) flops; O(N^2) auxiliary memory.
 * @par Reference paper.tex line 1316; notes/0t1_resolved_grid_design.md.
 */
void su2_ft_direct_resolved(int N,
                            const double _Complex *f,
                            double _Complex *fhat)
{
    assert(N >= 2 && "su2_ft_direct_resolved: N must be >= 2");
    assert(f && "su2_ft_direct_resolved: f must be non-NULL");
    assert(fhat && "su2_ft_direct_resolved: fhat must be non-NULL");

    const int    P        = 2 * N - 1;
    const double dphi     = 2.0 * M_PI / (double)P;
    const double dpsi     = 2.0 * M_PI / (double)P;
    /* norm = 1/(2 P^2); see file header derivation.
     * Equivalently (1/(8 pi^2)) * dphi * dpsi when paired with the GL
     * weights w_k summing to 2 on [-1, 1] (= Haar theta integral). */
    const double norm     = 1.0 / (2.0 * (double)P * (double)P);
    /* Suppress -Wunused-variable on dphi/dpsi while keeping them in the
     * source as a documented sanity-check that norm = (1/(8 pi^2))*dphi*dpsi. */
    (void)dphi; (void)dpsi;

    /* ----- Gauss-Legendre nodes + weights; theta_k = arccos(x_k). ----- */
    double *x_gl  = malloc((size_t)N * sizeof(double));
    double *w_gl  = malloc((size_t)N * sizeof(double));
    double *theta = malloc((size_t)N * sizeof(double));
    assert(x_gl && w_gl && theta && "su2_ft_direct_resolved: alloc");
    su2_gl_nodes_weights(N, x_gl, w_gl);
    for (int k = 0; k < N; ++k) theta[k] = acos(x_gl[k]);

    /* ----- Open phi / psi grids: phi[j] = -pi + j * 2pi/P, j in [0, P-1].
     * Note: pin j=0 to the exact value -pi (mirrors grid_uniform pinning
     * in src/su2_grid.c) to suppress float drift.                      */
    double *phi = malloc((size_t)P * sizeof(double));
    double *psi = malloc((size_t)P * sizeof(double));
    assert(phi && psi && "su2_ft_direct_resolved: alloc");
    const double dphi_step = 2.0 * M_PI / (double)P;
    for (int j = 0; j < P; ++j) {
        phi[j] = -M_PI + (double)j * dphi_step;
        psi[j] = -M_PI + (double)j * dphi_step;
    }
    phi[0] = -M_PI;
    psi[0] = -M_PI;

    /* ----- Precomputed exponential tables.
     * exp_phi[(nn + N-1) * P + j1] = exp(+i nn phi[j1]),  nn in [-(N-1), N-1].
     * exp_psi[(mm + N-1) * P + j2] = exp(+i mm psi[j2]),  mm in [-(N-1), N-1].
     */
    const int    nrange   = 2 * N - 1;            /* = P, coincidentally. */
    const size_t etab_len = (size_t)nrange * (size_t)P;
    double _Complex *exp_phi = malloc(etab_len * sizeof(double _Complex));
    double _Complex *exp_psi = malloc(etab_len * sizeof(double _Complex));
    assert(exp_phi && exp_psi && "su2_ft_direct_resolved: alloc");
    for (int nn = -(N - 1); nn <= N - 1; ++nn) {
        size_t row = (size_t)(nn + N - 1) * (size_t)P;
        for (int j = 0; j < P; ++j) {
            exp_phi[row + (size_t)j] = cexp(I * (double)nn * phi[j]);
            exp_psi[row + (size_t)j] = cexp(I * (double)nn * psi[j]);
        }
    }

    /* ----- Per-(l, m, n) Wigner kernel, length N.
     * P_k[k] = conj(P^l_{n,m}(cos theta_k)) * w_k.   NO sin(theta_k) factor:
     * the Jacobian is absorbed in w_k via the x = cos(theta) substitution.
     */
    double _Complex *P_k = malloc((size_t)N * sizeof(double _Complex));
    assert(P_k && "su2_ft_direct_resolved: alloc");

    /* ----- Brute force triple sum -- paper.tex line 1316.
     * fhat(l)_{m,n} = norm * sum_{j1, k, j2}
     *                    f(g) * conj(P^l_{n,m}(cos theta_k))
     *                         * exp(+i n phi[j1])
     *                         * exp(+i m psi[j2])
     *                         * w_k
     */
    for (int l = 0; l < N; ++l) {
        for (int m = -l; m <= l; ++m) {
            const size_t row_m = (size_t)(m + N - 1) * (size_t)P;
            for (int n = -l; n <= l; ++n) {
                const size_t row_n = (size_t)(n + N - 1) * (size_t)P;

                for (int k = 0; k < N; ++k) {
                    P_k[k] = conj(su2_wigner_d(l, n, m, theta[k])) * w_gl[k];
                }

                double _Complex acc = 0.0 + 0.0*I;
                for (int k = 0; k < N; ++k) {
                    double _Complex pk = P_k[k];
                    if (pk == 0.0) continue;
                    for (int j1 = 0; j1 < P; ++j1) {
                        double _Complex en = exp_phi[row_n + (size_t)j1];
                        for (int j2 = 0; j2 < P; ++j2) {
                            double _Complex em = exp_psi[row_m + (size_t)j2];
                            acc += f[su2_resolved_sample_index(N, j1, k, j2)]
                                 * pk * en * em;
                        }
                    }
                }

                fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)] = norm * acc;
            }
        }
    }

    free(P_k);
    free(exp_phi); free(exp_psi);
    free(phi); free(psi);
    free(theta); free(w_gl); free(x_gl);
}
