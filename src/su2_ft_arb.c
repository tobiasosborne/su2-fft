/* su2_ft_arb.c -- Arbitrary-precision O(N^6) reference Fourier transform.
 *
 * Same discrete sum as src/su2_ft.c (paper line 1316) but every arithmetic
 * op is an arb/acb call at user-specified precision.
 *
 * Pre-computed scratch:
 *   theta[k], sin(theta[k])           -- arb_t
 *   phi[j], psi[j]                    -- arb_t
 *   exp_phi[(n+N-1)*N + j] = exp(+i n phi[j])   -- acb_t
 *   exp_psi[(m+N-1)*N + j] = exp(+i m psi[j])   -- acb_t
 */
#include "su2_arb.h"

#include <flint/arb.h>
#include <flint/acb.h>
#include <flint/arf.h>
#include <stdlib.h>
#include <string.h>

static void fill_grid_phi(arb_ptr out, int N, slong prec)
{
    /* phi[j] = -pi + j * 2 pi / (N-1) */
    arb_t pi, step;
    arb_init(pi); arb_init(step);
    arb_const_pi(pi, prec);
    arb_mul_2exp_si(step, pi, 1);                /* 2 pi */
    arb_div_ui(step, step, (ulong)(N - 1), prec);
    for (int j = 0; j < N; ++j) {
        arb_mul_ui(out + j, step, (ulong)j, prec);
        arb_sub(out + j, out + j, pi, prec);
    }
    arb_clear(pi); arb_clear(step);
}

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
 * @brief Arbitrary-precision O(N^6) direct Fourier transform on SU(2).
 *
 * Mirrors su2_ft_direct() using FLINT acb arithmetic throughout.  Implements
 * the same discrete sum (paper.tex line 1316) with every operation carried at
 * `prec` bits, yielding certified acb_t ball-valued coefficients.  The norm
 * factor is pi / (2*(N-1)^3) derived from dphi*dtheta*dpsi/(8*pi^2).
 * Precomputed tables for exp(+i*n*phi[j]) and sin(theta[k]) are held as
 * arb/acb vectors; the Wigner kernel is rebuilt per (l, m, n) via
 * su2_wigner_d_arb().
 *
 * @param[in]  N     Bandlimit; grid is N x N x N, coefficients span l < N.
 * @param[in]  f     acb_srcptr of length N^3, row-major (j1, k, j2).
 * @param[out] fhat  acb_ptr of length su2_total_coeffs(N); overwritten.
 * @param[in]  prec  Working precision in bits (e.g. 53 = IEEE double,
 *                   256 = high precision).
 * @par Complexity O(N^6) acb multiplications; O(N^2) auxiliary acb memory.
 * @par Reference paper.tex line 1316; ALGORITHM.md Section 2.1.
 */
void su2_ft_direct_arb(int N, acb_srcptr f, acb_ptr fhat, slong prec)
{
    if (N < 2 || !f || !fhat) return;

    arb_ptr theta = _arb_vec_init(N);
    arb_ptr phi   = _arb_vec_init(N);
    arb_ptr sin_th = _arb_vec_init(N);
    fill_grid_phi(phi,   N, prec);
    fill_grid_theta(theta, N, prec);
    for (int k = 0; k < N; ++k) arb_sin(sin_th + k, theta + k, prec);

    /* exp tables: exp_phi[(n+N-1)*N + j] = exp(+i n phi[j]) */
    const int    nrange = 2 * N - 1;
    const size_t etab_n = (size_t)nrange * (size_t)N;
    acb_ptr exp_phi = _acb_vec_init(etab_n);
    acb_ptr exp_psi = _acb_vec_init(etab_n);

    acb_t tmp_iz;
    acb_init(tmp_iz);
    for (int nn = -(N - 1); nn <= N - 1; ++nn) {
        size_t row = (size_t)(nn + N - 1) * (size_t)N;
        for (int j = 0; j < N; ++j) {
            /* exp(+i nn phi[j]) -- real=cos, imag=sin of (nn * phi[j]) */
            arb_t angle;
            arb_init(angle);
            arb_mul_si(angle, phi + j, nn, prec);
            arb_cos(acb_realref(exp_phi + row + (size_t)j), angle, prec);
            arb_sin(acb_imagref(exp_phi + row + (size_t)j), angle, prec);
            /* psi grid coincides with phi grid (both -pi..pi in N steps). */
            acb_set(exp_psi + row + (size_t)j, exp_phi + row + (size_t)j);
            arb_clear(angle);
        }
    }
    acb_clear(tmp_iz);

    /* norm = pi / (2 (N-1)^3) */
    arb_t norm;
    arb_init(norm);
    arb_const_pi(norm, prec);
    {
        ulong M = (ulong)(N - 1);
        arb_div_ui(norm, norm, 2 * M * M * M, prec);
    }

    acb_t acc, P_lnm, term, en, em, pk;
    acb_init(acc); acb_init(P_lnm); acb_init(term);
    acb_init(en); acb_init(em); acb_init(pk);
    acb_ptr P_k = _acb_vec_init(N);

    for (int l = 0; l < N; ++l) {
        for (int m = -l; m <= l; ++m) {
            size_t row_m = (size_t)(m + N - 1) * (size_t)N;
            for (int n = -l; n <= l; ++n) {
                size_t row_n = (size_t)(n + N - 1) * (size_t)N;

                /* Precompute conj(P^l_{n,m}(cos theta_k)) * sin(theta_k) for each k. */
                for (int k = 0; k < N; ++k) {
                    su2_wigner_d_arb(P_lnm, l, n, m, theta + k, prec);
                    acb_conj(P_lnm, P_lnm);
                    acb_mul_arb(P_k + k, P_lnm, sin_th + k, prec);
                }

                acb_zero(acc);
                for (int k = 0; k < N; ++k) {
                    for (int j1 = 0; j1 < N; ++j1) {
                        acb_set(en, exp_phi + row_n + (size_t)j1);
                        for (int j2 = 0; j2 < N; ++j2) {
                            acb_set(em, exp_psi + row_m + (size_t)j2);
                            acb_mul(term, en, em, prec);
                            acb_mul(term, term, P_k + k, prec);
                            acb_mul(term, term,
                                    f + su2_sample_index(N, j1, k, j2),
                                    prec);
                            acb_add(acc, acc, term, prec);
                        }
                    }
                }

                acb_mul_arb(fhat + su2_coeff_offset(l) + su2_mn_index(l, m, n),
                            acc, norm, prec);
            }
        }
    }

    _acb_vec_clear(P_k, N);
    acb_clear(acc); acb_clear(P_lnm); acb_clear(term);
    acb_clear(en);  acb_clear(em);    acb_clear(pk);
    arb_clear(norm);

    _acb_vec_clear(exp_phi, etab_n);
    _acb_vec_clear(exp_psi, etab_n);
    _arb_vec_clear(sin_th, N);
    _arb_vec_clear(phi,    N);
    _arb_vec_clear(theta,  N);
}
