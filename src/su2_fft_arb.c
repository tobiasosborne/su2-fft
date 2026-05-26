/* su2_fft_arb.c -- Arbitrary-precision O(N^4) fast FFT on SU(2).
 *
 * Mirror of src/su2_fft.c with acb_t arithmetic.
 *
 * Stage 1: For each theta-slice fold endpoints into a (N-1)x(N-1) array,
 *          take elementwise conjugate, run acb_dft_prod (forward DFT),
 *          conjugate again to get the backward 2-D DFT result.  The
 *          (-1)^{n+m} half-shift factor restores the closed-grid phase.
 *
 * Stage 2: For each (l, m, n) take the length-N inner product
 *          fhat = norm * sum_k F2[k, n, m] * conj(P^l_{n,m}(cos theta_k))
 *                                          * sin(theta_k).
 */
#include "su2_arb.h"

#include <flint/arb.h>
#include <flint/acb.h>
#include <flint/acb_dft.h>
#include <stdlib.h>
#include <string.h>

static void fill_grid_theta(arb_ptr out, int N, slong prec)
{
    arb_t pi, step;
    arb_init(pi); arb_init(step);
    arb_const_pi(pi, prec);
    arb_div_ui(step, pi, (ulong)(N - 1), prec);
    for (int k = 0; k < N; ++k) arb_mul_ui(out + k, step, (ulong)k, prec);
    arb_clear(pi); arb_clear(step);
}

/**
 * @brief Arbitrary-precision O(N^4) fast Fourier transform on SU(2).
 *
 * Mirrors su2_fft() with FLINT acb arithmetic throughout, providing certified
 * error balls on every output coefficient.  Uses the same two-stage strategy:
 *
 *   Stage 1 (O(N^3 log N)): Per theta-slice k, fold endpoints into an
 *     (N-1)x(N-1) array, compute conj(forward-DFT(conj(input))) via
 *     acb_dft_prod to obtain the backward 2-D DFT, then apply the (-1)^{n+m}
 *     closed-grid phase correction to recover F2[k, n, m] (paper.tex line 1347).
 *
 *   Stage 2 (O(N^4)): For each (l, m, n), accumulate the length-N dot product
 *     of F2[*, n, m] against conj(P^l_{n,m}(cos theta_k)) * sin(theta_k)
 *     (paper.tex line 1361).  O(N^3) coefficients x O(N) per dot = O(N^4).
 *
 * @param[in]  N     Bandlimit; grid is N x N x N, coefficients span l < N.
 * @param[in]  f     acb_srcptr of length N^3, row-major (j1, k, j2).
 * @param[out] fhat  acb_ptr of length su2_total_coeffs(N); overwritten.
 * @param[in]  prec  Working precision in bits (e.g. 53, 128, 256, 512).
 * @par Complexity O(N^4) acb multiplications; O(N^3) auxiliary acb memory.
 * @par Reference paper.tex lines 1347, 1361, 1455; ALGORITHM.md Section 2.2.
 */
void su2_fft_arb(int N, acb_srcptr f, acb_ptr fhat, slong prec)
{
    if (N < 2 || !f || !fhat) return;

    const int M = N - 1;
    arb_ptr theta = _arb_vec_init(N);
    arb_ptr sin_th = _arb_vec_init(N);
    fill_grid_theta(theta, N, prec);
    for (int k = 0; k < N; ++k) arb_sin(sin_th + k, theta + k, prec);

    /* -------- Stage 1 -------- */

    const int    nrange = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;

    slong cyc[2] = { (slong)M, (slong)M };
    acb_ptr g = _acb_vec_init((size_t)M * (size_t)M);
    acb_ptr G = _acb_vec_init((size_t)M * (size_t)M);
    acb_ptr F2 = _acb_vec_init((size_t)N * stride_k);

    for (int k = 0; k < N; ++k) {
        _acb_vec_zero(g, (size_t)M * (size_t)M);

        for (int j1 = 0; j1 < N; ++j1) {
            int j1m = (j1 == N - 1) ? 0 : j1;
            for (int j2 = 0; j2 < N; ++j2) {
                int j2m = (j2 == N - 1) ? 0 : j2;
                acb_add(g + (size_t)j1m * (size_t)M + (size_t)j2m,
                        g + (size_t)j1m * (size_t)M + (size_t)j2m,
                        f + su2_sample_index(N, j1, k, j2),
                        prec);
            }
        }

        /* backward DFT = conj(forward(conj(input))) */
        for (slong i = 0; i < (slong)M * M; ++i) acb_conj(g + i, g + i);
        acb_dft_prod(G, g, cyc, 2, prec);
        for (slong i = 0; i < (slong)M * M; ++i) acb_conj(G + i, G + i);

        for (int n = -(N - 1); n <= N - 1; ++n) {
            int n_mod = ((n % M) + M) % M;
            int sn    = (n & 1) ? -1 : 1;
            for (int m = -(N - 1); m <= N - 1; ++m) {
                int m_mod = ((m % M) + M) % M;
                int sm    = (m & 1) ? -1 : 1;
                acb_ptr dst = F2 + (size_t)k * stride_k
                               + (size_t)(n + N - 1) * stride_n
                               + (size_t)(m + N - 1);
                acb_srcptr src = G + (size_t)n_mod * (size_t)M + (size_t)m_mod;
                if (sn * sm > 0) acb_set(dst, src);
                else             acb_neg(dst, src);
            }
        }
    }
    _acb_vec_clear(g, (size_t)M * (size_t)M);
    _acb_vec_clear(G, (size_t)M * (size_t)M);

    /* -------- Stage 2 -------- */
    arb_t norm;
    arb_init(norm);
    arb_const_pi(norm, prec);
    arb_div_ui(norm, norm, 2 * (ulong)M * (ulong)M * (ulong)M, prec);

    acb_t acc, P_lnm, term, weighted_P;
    acb_init(acc); acb_init(P_lnm); acb_init(term); acb_init(weighted_P);

    for (int l = 0; l < N; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int n = -l; n <= l; ++n) {
                acb_zero(acc);
                for (int k = 0; k < N; ++k) {
                    su2_wigner_d_arb(P_lnm, l, n, m, theta + k, prec);
                    acb_conj(P_lnm, P_lnm);
                    acb_mul_arb(weighted_P, P_lnm, sin_th + k, prec);
                    acb_mul(term, weighted_P,
                            F2 + (size_t)k * stride_k
                               + (size_t)(n + N - 1) * stride_n
                               + (size_t)(m + N - 1),
                            prec);
                    acb_add(acc, acc, term, prec);
                }
                acb_mul_arb(fhat + su2_coeff_offset(l) + su2_mn_index(l, m, n),
                            acc, norm, prec);
            }
        }
    }

    acb_clear(acc); acb_clear(P_lnm); acb_clear(term); acb_clear(weighted_P);
    arb_clear(norm);
    _acb_vec_clear(F2, (size_t)N * stride_k);
    _arb_vec_clear(sin_th, N);
    _arb_vec_clear(theta,  N);
}
