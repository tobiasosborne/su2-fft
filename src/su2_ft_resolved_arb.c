/* su2_ft_resolved_arb.c -- Arbitrary-precision O(N^6) reference Fourier
 * transform on the resolved grid.
 *
 * Bead: su2fft-rrx (Step 8, Deliverable B).
 * Spec: notes/0t1_resolved_grid_design.md §5 (norm) and §8 (FLINT capability).
 *
 * Arb port of src/su2_ft_resolved.c, structurally identical (same triple sum,
 * same kernel) but every arithmetic operation runs at `prec` bits via FLINT
 * arb/acb.  GL nodes come from the arb wrapper su2_gl_nodes_weights_arb
 * (Deliverable A).
 *
 * Paper.tex line 1316 (verbatim, ground truth):
 *
 *   fhat(l)_{mn} ~= sum_{i=1..N^3} f(g_i) * conj(t^l_{mn}(g_i))
 *
 * with t^l_{n,m}(phi, theta, psi) = P^l_{n,m}(cos theta) * exp(-i(n phi + m psi)),
 * yielding on the resolved grid (notes/0t1_resolved_grid_design.md §5):
 *
 *   fhat(l)_{m,n}
 *     = norm * sum_{j1, k, j2}
 *           f(g_{j1,k,j2})
 *         * conj(P^l_{n,m}(cos theta_k))
 *         * exp(+i n phi[j1])
 *         * exp(+i m psi[j2])
 *         * w_k
 *
 *   norm = 1 / (2 P^2),     P = 2N - 1.
 *
 * The sin(theta) Jacobian of the Haar measure is absorbed into w_k via the
 * substitution x = cos(theta); no explicit sin(theta_k) factor here (the
 * one structural difference from the legacy src/su2_ft_arb.c).
 */
#include "su2.h"
#include "su2_arb.h"

#include <flint/arb.h>
#include <flint/acb.h>
#include <assert.h>

/* phi[j] = -pi + j * 2 pi / P, j in [0, P-1].  Pin j=0 to exact -pi via
 * arb_neg(pi) to mirror the float-pinning discipline of src/su2_ft_resolved.c
 * (no FP drift in arb anyway, but documented as such).                   */
static void fill_grid_phi_resolved(arb_ptr out, int P, slong prec)
{
    arb_t pi, step;
    arb_init(pi); arb_init(step);
    arb_const_pi(pi, prec);
    arb_mul_2exp_si(step, pi, 1);                /* 2 pi */
    arb_div_ui(step, step, (ulong)P, prec);
    for (int j = 0; j < P; ++j) {
        arb_mul_ui(out + j, step, (ulong)j, prec);
        arb_sub(out + j, out + j, pi, prec);
    }
    arb_neg(out + 0, pi);                        /* phi[0] = -pi exactly */
    arb_clear(pi); arb_clear(step);
}

/**
 * @brief Arbitrary-precision O(N^6) direct Fourier transform on SU(2)
 *        on the resolved grid (P = 2N-1 in phi/psi; N GL nodes in theta).
 *
 * Mirrors su2_ft_direct_resolved() using FLINT acb arithmetic throughout.
 * Implements the same discrete sum (paper.tex line 1316) with every operation
 * carried at `prec` bits, yielding certified acb_t ball-valued coefficients.
 * Norm: 1 / (2 P^2) (notes/0t1_resolved_grid_design.md §5).
 *
 * @param[in]  N     Bandlimit; sample grid is P x N x P, P = 2N-1.
 * @param[in]  f     acb_srcptr of length P*P*N, row-major (j1, k, j2);
 *                   index via su2_resolved_sample_index.
 * @param[out] fhat  acb_ptr of length su2_total_coeffs(N); overwritten.
 * @param[in]  prec  Working precision in bits (e.g. 53 = IEEE double,
 *                   128 = high precision).
 * @par Complexity O(N^6) acb multiplications; O(N^2) auxiliary acb memory.
 * @par Reference paper.tex line 1316; notes/0t1_resolved_grid_design.md §5;
 *                src/su2_ft_resolved.c (double-precision twin).
 */
void su2_ft_direct_resolved_arb(int N, acb_srcptr f, acb_ptr fhat, slong prec)
{
    assert(N >= 2 && "su2_ft_direct_resolved_arb: N must be >= 2");
    assert(f != NULL && "su2_ft_direct_resolved_arb: f must be non-NULL");
    assert(fhat != NULL && "su2_ft_direct_resolved_arb: fhat must be non-NULL");

    const int P = 2 * N - 1;

    /* ----- GL nodes (ascending) and weights at `prec` bits. ----- */
    arb_ptr x_gl  = _arb_vec_init(N);
    arb_ptr w_gl  = _arb_vec_init(N);
    arb_ptr theta = _arb_vec_init(N);
    su2_gl_nodes_weights_arb(N, x_gl, w_gl, prec);
    for (int k = 0; k < N; ++k) arb_acos(theta + k, x_gl + k, prec);

    /* ----- Open phi / psi grids (coincident here): phi[j] = -pi + j * 2pi/P. */
    arb_ptr phi = _arb_vec_init(P);
    arb_ptr psi = _arb_vec_init(P);
    fill_grid_phi_resolved(phi, P, prec);
    fill_grid_phi_resolved(psi, P, prec);

    /* ----- Exponential tables.
     * exp_phi[(nn + N-1)*P + j1] = exp(+i nn phi[j1]),  nn in [-(N-1), N-1].
     * exp_psi[(mm + N-1)*P + j2] = exp(+i mm psi[j2]),  mm in [-(N-1), N-1].
     * Note: phi and psi grids are identical here, but we keep two tables
     * to mirror src/su2_ft_arb.c structure and for clarity.              */
    const int    nrange   = 2 * N - 1;            /* = P, coincidentally. */
    const size_t etab_len = (size_t)nrange * (size_t)P;
    acb_ptr exp_phi = _acb_vec_init(etab_len);
    acb_ptr exp_psi = _acb_vec_init(etab_len);

    {
        arb_t angle;
        arb_init(angle);
        for (int nn = -(N - 1); nn <= N - 1; ++nn) {
            size_t row = (size_t)(nn + N - 1) * (size_t)P;
            for (int j = 0; j < P; ++j) {
                arb_mul_si(angle, phi + j, nn, prec);
                arb_cos(acb_realref(exp_phi + row + (size_t)j), angle, prec);
                arb_sin(acb_imagref(exp_phi + row + (size_t)j), angle, prec);
                acb_set(exp_psi + row + (size_t)j,
                        exp_phi + row + (size_t)j);
            }
        }
        arb_clear(angle);
    }

    /* ----- norm = 1 / (2 P^2)  (notes/0t1_resolved_grid_design.md §5).
     * Sanity (kept as a comment, matching src/su2_ft_resolved.c):
     *   norm = (1/(8 pi^2)) * dphi * dpsi
     *        = (1/(8 pi^2)) * (2 pi / P)^2
     *        = 1 / (2 P^2).
     * The trailing "* 1" is the [-1,1]-measure factor (GL weights sum to 2,
     * matching the Haar sin(theta) dtheta integral 2 over [0, pi]).      */
    arb_t norm;
    arb_init(norm);
    arb_one(norm);
    arb_div_ui(norm, norm, 2u * (ulong)P * (ulong)P, prec);

    /* ----- Per-(l, m, n) Wigner kernel:
     * P_k[k] = conj(P^l_{n,m}(cos theta_k)) * w_k.  NO sin(theta_k) factor:
     * Jacobian is absorbed in w_k via x = cos(theta).                    */
    acb_ptr P_k = _acb_vec_init(N);

    acb_t acc, P_lnm, term, en, em;
    acb_init(acc); acb_init(P_lnm); acb_init(term);
    acb_init(en); acb_init(em);

    /* ----- Brute force triple sum -- paper.tex line 1316. */
    for (int l = 0; l < N; ++l) {
        for (int m = -l; m <= l; ++m) {
            const size_t row_m = (size_t)(m + N - 1) * (size_t)P;
            for (int n = -l; n <= l; ++n) {
                const size_t row_n = (size_t)(n + N - 1) * (size_t)P;

                for (int k = 0; k < N; ++k) {
                    su2_wigner_d_arb(P_lnm, l, n, m, theta + k, prec);
                    acb_conj(P_lnm, P_lnm);
                    acb_mul_arb(P_k + k, P_lnm, w_gl + k, prec);
                }

                acb_zero(acc);
                for (int k = 0; k < N; ++k) {
                    for (int j1 = 0; j1 < P; ++j1) {
                        acb_set(en, exp_phi + row_n + (size_t)j1);
                        for (int j2 = 0; j2 < P; ++j2) {
                            acb_set(em, exp_psi + row_m + (size_t)j2);
                            acb_mul(term, en, em, prec);
                            acb_mul(term, term, P_k + k, prec);
                            acb_mul(term, term,
                                    f + su2_resolved_sample_index(N, j1, k, j2),
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
    acb_clear(en);  acb_clear(em);
    arb_clear(norm);

    _acb_vec_clear(exp_phi, etab_len);
    _acb_vec_clear(exp_psi, etab_len);
    _arb_vec_clear(phi, P);
    _arb_vec_clear(psi, P);
    _arb_vec_clear(theta, N);
    _arb_vec_clear(w_gl,  N);
    _arb_vec_clear(x_gl,  N);
}
