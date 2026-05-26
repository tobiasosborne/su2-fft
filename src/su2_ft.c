/* su2_ft.c -- Direct O(N^6) Fourier transform on SU(2).
 *
 * Computes the discrete sum from paper.tex line 1316 verbatim:
 *
 *   fhat(l)_{m,n}
 *     = (dphi * dtheta * dpsi / (8 pi^2)) *
 *       sum_{j1, k, j2}
 *           f(g_{j1,k,j2})
 *         * conj( t^l_{n,m}(g_{j1,k,j2}) )
 *         * sin(theta_k)
 *
 *   with  t^l_{n,m}(phi, theta, psi) = P^l_{n,m}(cos theta) * exp(-i(n phi + m psi))
 *   so    conj(t^l_{n,m})            = conj(P^l_{n,m}(cos theta)) * exp(+i(n phi + m psi))
 *
 * Brute force across all (l, m, n) (O(N^3) coefficients) times all (j1, k, j2)
 * grid points (O(N^3)) = O(N^6) flops.
 *
 * Per-coefficient inner loop is kept as the literal triple sum -- no Stage-1
 * separation -- so that su2_fft.c's claim of O(N^4) is a real speedup over
 * something honestly O(N^6).
 *
 * Pre-computations:
 *   sin_theta[k]         length N
 *   exp_phi[n + N-1, j1] complex, length (2N-1) * N    -- exp(+i n phi[j1])
 *   exp_psi[m + N-1, j2] complex, length (2N-1) * N    -- exp(+i m psi[j2])
 *   P_kernel[k]          per (l,m,n), length N         -- conj(P^l_{n,m}(cos theta_k)) * sin_theta[k]
 *
 * These are O(N^4) memory at most (the P_kernel is rebuilt per (l,m,n) so only
 * O(N) at a time).  Negligible next to the O(N^6) flops.
 */
#include "su2.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>

/**
 * @brief Direct O(N^6) Fourier transform on SU(2).
 *
 * Implements the discrete Fourier transform on SU(2) by brute-force triple
 * summation over the Euler-angle grid, as written verbatim in paper.tex
 * line 1316.  For each coefficient (l, m, n) with l in [0, N-1] and
 * m, n in [-l, l], computes:
 *   fhat(l)_{m,n} = (dphi * dtheta * dpsi / (8 pi^2)) *
 *                  sum_{j1, k, j2} f(g_{j1,k,j2})
 *                                  * conj(P^l_{n,m}(cos theta_k))
 *                                  * exp(+i(n phi[j1] + m psi[j2]))
 *                                  * sin(theta_k).
 * Exponential and sin(theta) tables are precomputed; the Wigner kernel
 * P_k is rebuilt per (l, m, n).  Used as a correctness reference only.
 *
 * @param[in]  N     Bandlimit; grid is N x N x N, coefficients span l < N.
 * @param[in]  f     Length-N^3 complex sample array, row-major (j1, k, j2).
 * @param[out] fhat  Length-su2_total_coeffs(N) complex coefficient array,
 *                   zeroed on entry is not required (values are overwritten).
 * @par Complexity O(N^6) flops; O(N^2) auxiliary memory.
 * @par Reference paper.tex line 1316; ALGORITHM.md Section 2.1.
 */
void su2_ft_direct(int N,
                   const double _Complex *f,
                   double _Complex *fhat)
{
    if (N < 2 || !f || !fhat) return;

    double *phi   = su2_grid_phi(N);
    double *theta = su2_grid_theta(N);
    double *psi   = su2_grid_psi(N);

    double *sin_th = malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N; ++k) sin_th[k] = sin(theta[k]);

    const int    nrange   = 2 * N - 1;   /* n, m in [-(N-1), N-1] */
    const size_t etab_len = (size_t)nrange * (size_t)N;
    double _Complex *exp_phi = malloc(etab_len * sizeof(double _Complex));
    double _Complex *exp_psi = malloc(etab_len * sizeof(double _Complex));
    for (int nn = -(N - 1); nn <= N - 1; ++nn) {
        size_t row = (size_t)(nn + N - 1) * (size_t)N;
        for (int j = 0; j < N; ++j) {
            exp_phi[row + (size_t)j] = cexp(I * nn * phi[j]);
            exp_psi[row + (size_t)j] = cexp(I * nn * psi[j]);
        }
    }

    const double dphi   = 2.0 * M_PI / (double)(N - 1);
    const double dtheta =       M_PI / (double)(N - 1);
    const double dpsi   = 2.0 * M_PI / (double)(N - 1);
    const double norm   = dphi * dtheta * dpsi / (8.0 * M_PI * M_PI);

    double _Complex *P_k = malloc((size_t)N * sizeof(double _Complex));

    for (int l = 0; l < N; ++l) {
        for (int m = -l; m <= l; ++m) {
            const size_t row_m = (size_t)(m + N - 1) * (size_t)N;
            for (int n = -l; n <= l; ++n) {
                const size_t row_n = (size_t)(n + N - 1) * (size_t)N;

                for (int k = 0; k < N; ++k) {
                    P_k[k] = conj(su2_wigner_d(l, n, m, theta[k])) * sin_th[k];
                }

                double _Complex acc = 0.0 + 0.0*I;
                for (int k = 0; k < N; ++k) {
                    double _Complex pk = P_k[k];
                    if (pk == 0.0) continue;
                    for (int j1 = 0; j1 < N; ++j1) {
                        double _Complex en = exp_phi[row_n + (size_t)j1];
                        for (int j2 = 0; j2 < N; ++j2) {
                            double _Complex em = exp_psi[row_m + (size_t)j2];
                            acc += f[su2_sample_index(N, j1, k, j2)] * pk * en * em;
                        }
                    }
                }

                fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)] = norm * acc;
            }
        }
    }

    free(P_k);
    free(exp_phi); free(exp_psi);
    free(sin_th);
    free(phi); free(theta); free(psi);
}
