/* su2_fft.c -- Fast Fourier transform on SU(2).  O(N^4) total cost.
 *
 * Strategy (paper.tex, Section 4, summarised in notes/section_4_algorithm.md):
 *
 *   Stage 1: For each theta-slice k, compute the 2-D DFT
 *              F2[k, n, m] = sum_{j1, j2} f(g_{j1,k,j2})
 *                                       * exp(+i n phi[j1])
 *                                       * exp(+i m psi[j2])
 *            via FFTW.  Per slice O(N^2 log N); total O(N^3 log N).
 *
 *   Stage 2: For each (l, m, n), evaluate
 *              fhat(l)_{m,n} = norm * sum_{k} F2[k, n, m]
 *                                          * conj(P^l_{n,m}(cos theta_k))
 *                                          * sin(theta_k)
 *            (paper line 1361, eq EQ_OP_1).  Per coefficient O(N); over O(N^3)
 *            coefficients = O(N^4) -- the headline complexity at split k=1
 *            (paper Theorem TEO_FIN, line 1455).
 *
 * Closed-grid handling for Stage 1
 * --------------------------------
 * The Euler grid has phi_{j1} = -pi + j1 * 2pi/(N-1), j1 in [0, N-1].  Both
 * endpoints j1=0 and j1=N-1 sit at the same point on the torus (mod 2pi), so
 * the N samples carry only N-1 independent Fourier modes.  Writing
 *
 *   exp(+i n phi[j1]) = (-1)^n * exp(+i n j1 * 2pi/(N-1))
 *
 * the sum over j1 in [0, N-1] equals (-1)^n times the (N-1)-point DFT of the
 * folded sequence  g[0] = f[0] + f[N-1],  g[j] = f[j]  for j in [1, N-2].
 * We do the analogous fold in j2 and use one FFTW 2-D backward plan of size
 * (N-1) x (N-1).
 */
#include "su2.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Fast O(N^4) Fourier transform on SU(2) via 2-D FFTW + Wigner inner product.
 *
 * Produces the same fhat as su2_ft_direct() (to floating-point tolerance) by
 * separating the phi/psi and theta summations into two stages:
 *
 *   Stage 1 (O(N^3 log N)): For each theta-slice k, fold the closed grid
 *     endpoints and apply a (N-1)x(N-1) backward 2-D FFTW plan to obtain
 *     F2[k, n, m] (paper.tex line 1347).  A (-1)^{n+m} factor restores the
 *     half-shift from the closed grid; see ALGORITHM.md Section 2.2.
 *
 *   Stage 2 (O(N^4)): For each coefficient (l, m, n), evaluate the length-N
 *     dot product against conj(P^l_{n,m}(cos theta_k)) * sin(theta_k)
 *     (paper.tex line 1361, eq EQ_OP_1).  O(N^3) coefficients x O(N) per dot
 *     product gives the headline O(N^4) (paper Theorem TEO_FIN, line 1455).
 *
 * @param[in]  N     Bandlimit; grid is N x N x N, coefficients span l < N.
 * @param[in]  f     Length-N^3 complex sample array, row-major (j1, k, j2).
 * @param[out] fhat  Length-su2_total_coeffs(N) complex coefficient array.
 * @par Complexity O(N^4) flops; O(N^3) auxiliary memory for F2.
 * @par Reference paper.tex lines 1347, 1361, 1455; ALGORITHM.md Section 2.2.
 */
void su2_fft(int N,
             const double _Complex *f,
             double _Complex *fhat)
{
    if (N < 2 || !f || !fhat) return;

    const int M = N - 1;
    double *theta = su2_grid_theta(N);

    /* -------- Stage 1 -------- */
    fftw_complex *g = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_complex *G = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_plan plan  = fftw_plan_dft_2d(M, M, g, G, FFTW_BACKWARD, FFTW_ESTIMATE);

    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;
    double _Complex *F2 = malloc((size_t)N * stride_k * sizeof(double _Complex));

    for (int k = 0; k < N; ++k) {
        memset(g, 0, sizeof(fftw_complex) * (size_t)M * (size_t)M);

        for (int j1 = 0; j1 < N; ++j1) {
            int j1m = (j1 == N - 1) ? 0 : j1;
            for (int j2 = 0; j2 < N; ++j2) {
                int j2m = (j2 == N - 1) ? 0 : j2;
                g[(size_t)j1m * (size_t)M + (size_t)j2m] +=
                    f[su2_sample_index(N, j1, k, j2)];
            }
        }
        fftw_execute(plan);

        for (int n = -(N - 1); n <= N - 1; ++n) {
            int n_mod = ((n % M) + M) % M;
            double sn = (n & 1) ? -1.0 : 1.0;   /* (-1)^n, valid for two's comp */
            for (int m = -(N - 1); m <= N - 1; ++m) {
                int m_mod = ((m % M) + M) % M;
                double sm = (m & 1) ? -1.0 : 1.0;
                F2[(size_t)k * stride_k
                   + (size_t)(n + N - 1) * stride_n
                   + (size_t)(m + N - 1)]
                    = sn * sm * G[(size_t)n_mod * (size_t)M + (size_t)m_mod];
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(g);
    fftw_free(G);

    /* -------- Stage 2 -------- */
    const double dphi   = 2.0 * M_PI / (double)(N - 1);
    const double dtheta =       M_PI / (double)(N - 1);
    const double dpsi   = 2.0 * M_PI / (double)(N - 1);
    const double norm   = dphi * dtheta * dpsi / (8.0 * M_PI * M_PI);

    double *sin_th = malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N; ++k) sin_th[k] = sin(theta[k]);

    for (int l = 0; l < N; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int n = -l; n <= l; ++n) {
                double _Complex acc = 0.0 + 0.0*I;
                for (int k = 0; k < N; ++k) {
                    double _Complex P = conj(su2_wigner_d(l, n, m, theta[k]));
                    double _Complex f2 = F2[(size_t)k * stride_k
                                            + (size_t)(n + N - 1) * stride_n
                                            + (size_t)(m + N - 1)];
                    acc += P * f2 * sin_th[k];
                }
                fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)] = norm * acc;
            }
        }
    }

    free(sin_th);
    free(F2);
    free(theta);
}
