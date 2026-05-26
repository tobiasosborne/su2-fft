/* su2_fft_resolved_arb.c -- Arbitrary-precision O(N^4) SU(2) FFT on the
 * resolved (open P-point) grid.
 *
 * Bead: su2fft-rrx (Step 9).  Arb port of src/su2_fft_resolved.c (the
 * double-precision twin).  Mirrors the structure of src/su2_fft_arb.c but
 * differs in three places:
 *
 *   1. Stage 1 plan size is P = 2N-1 (no fold), not M = N-1.  Each FFTW
 *      output bin is touched once per (n, m).
 *   2. Stage 2 norm is 1 / (2 P^2) (notes/0t1_resolved_grid_design.md §5),
 *      not pi / (2 (N-1)^3).
 *   3. Theta nodes are N-point Gauss-Legendre on [-1, 1]; the sin(theta)
 *      Jacobian of the Haar measure is absorbed in the GL weight w_k
 *      via x = cos(theta) (same as src/su2_ft_resolved_arb.c).
 *
 * References:
 *   paper.tex line 1316 -- discrete forward analysis
 *     fhat(l)_{mn} ~= sum_i f(g_i) * conj(t^l_{mn}(g_i))
 *     with t^l_{n,m}(g) = P^l_{n,m}(cos theta) * exp(-i(n phi + m psi)).
 *   paper.tex line 554  -- Peter-Weyl synthesis (used by the inverse)
 *     f(g) = sum_l (2l+1) sum_{m,n} fhat(l)_{m,n} * t^l_{n,m}(g).
 *   notes/0t1_resolved_grid_design.md §4 (Stage 1, no fold) and §5 (norm).
 *   src/su2_fft_arb.c (legacy closed-grid arb FFT) -- the structural template.
 *   src/su2_fft_resolved.c (double twin) -- the math template.
 *   src/su2_ft_resolved_arb.c (arb direct) -- the cross-check target.
 */
#include "su2.h"
#include "su2_arb.h"

#include <flint/arb.h>
#include <flint/acb.h>
#include <flint/acb_dft.h>
#include <assert.h>
#include <stdlib.h>

/**
 * @brief Arbitrary-precision O(N^4) forward FFT on SU(2), resolved grid.
 *
 * Arb twin of su2_fft_resolved().  Sample layout, grid, and norm match the
 * double-precision routine (notes/0t1_resolved_grid_design.md §5):
 *     f[su2_resolved_sample_index(N, j1, k, j2)],  length P*P*N, P = 2N-1.
 *     phi[j] = -pi + j * 2pi/P,  theta_k = acos(x_k) (GL),
 *     norm   = 1 / (2 P^2).
 *
 * Stage 1 (P x P backward DFT via the acb_dft_prod conj trick, paper.tex:1347):
 *   acb_dft_prod is forward-only; conj(forward(conj(input))) = backward(input).
 *   The (-1)^{n+m} phase from the -pi origin shift maps each
 *   n in [-(N-1), N-1] to a unique bin (n mod P) since P = 2N-1.
 *
 * Stage 2 (GL theta inner product, paper.tex:1316):
 *   fhat(l)_{m,n} = norm * sum_k w_k * conj(P^l_{n,m}(cos theta_k)) * F2[k,n,m].
 *   No sin(theta_k) factor here: the Jacobian is absorbed in w_k via
 *   x = cos(theta), same as src/su2_ft_resolved_arb.c.
 *
 * @param[in]  N     Bandlimit; sample grid is P x N x P, P = 2N-1.
 * @param[in]  f     acb_srcptr of length P*P*N, row-major (j1, k, j2);
 *                   index via su2_resolved_sample_index.
 * @param[out] fhat  acb_ptr of length su2_total_coeffs(N); overwritten.
 * @param[in]  prec  Working precision in bits.
 * @par Complexity O(N^4) acb multiplications; O(N^3) auxiliary acb memory.
 * @par Reference paper.tex line 1316; notes/0t1_resolved_grid_design.md §4-§5.
 */
void su2_fft_resolved_arb(int N, acb_srcptr f, acb_ptr fhat, slong prec)
{
    assert(N >= 2 && "su2_fft_resolved_arb: N must be >= 2");
    assert(f != NULL && "su2_fft_resolved_arb: f must be non-NULL");
    assert(fhat != NULL && "su2_fft_resolved_arb: fhat must be non-NULL");

    const int P = 2 * N - 1;

    /* ----- GL nodes (ascending) and weights at `prec` bits;
     * theta_k = arccos(x_k) (notes/0t1_resolved_grid_design.md §4). */
    arb_ptr x_gl  = _arb_vec_init(N);
    arb_ptr w_gl  = _arb_vec_init(N);
    arb_ptr theta = _arb_vec_init(N);
    su2_gl_nodes_weights_arb(N, x_gl, w_gl, prec);
    for (int k = 0; k < N; ++k) arb_acos(theta + k, x_gl + k, prec);

    /* -------- Stage 1: per-theta P x P backward DFT, NO fold.
     *
     * acb_dft_prod is forward; we need backward, hence the conj trick
     * (HANDOFF.md §2 item 4, mirrored from src/su2_fft_arb.c):
     *     backward(g) = conj(forward(conj(g))).
     * The (-1)^{n+m} phase factor from the -pi origin shift restores
     * the closed-form DFT result on the open P-point grid; see
     * notes/0t1_resolved_grid_design.md §4. */
    const int    nrange   = 2 * N - 1;         /* = P, by construction. */
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;

    slong cyc[2] = { (slong)P, (slong)P };
    acb_ptr g  = _acb_vec_init((size_t)P * (size_t)P);
    acb_ptr G  = _acb_vec_init((size_t)P * (size_t)P);
    acb_ptr F2 = _acb_vec_init((size_t)N * stride_k);

    for (int k = 0; k < N; ++k) {
        _acb_vec_zero(g, (size_t)P * (size_t)P);

        /* Direct copy: no fold, no accumulation.  Each (j1, k, j2) maps
         * to exactly one input bin (j1, j2) of the P x P DFT plan. */
        for (int j1 = 0; j1 < P; ++j1) {
            for (int j2 = 0; j2 < P; ++j2) {
                acb_set(g + (size_t)j1 * (size_t)P + (size_t)j2,
                        f + su2_resolved_sample_index(N, j1, k, j2));
            }
        }

        /* backward DFT = conj(forward(conj(input))) */
        for (slong i = 0; i < (slong)P * (slong)P; ++i) acb_conj(g + i, g + i);
        acb_dft_prod(G, g, cyc, 2, prec);
        for (slong i = 0; i < (slong)P * (slong)P; ++i) acb_conj(G + i, G + i);

        for (int n = -(N - 1); n <= N - 1; ++n) {
            int n_mod = ((n % P) + P) % P;
            int sn    = (n & 1) ? -1 : 1;       /* (-1)^n */
            for (int m = -(N - 1); m <= N - 1; ++m) {
                int m_mod = ((m % P) + P) % P;
                int sm    = (m & 1) ? -1 : 1;
                acb_ptr    dst = F2 + (size_t)k * stride_k
                                    + (size_t)(n + N - 1) * stride_n
                                    + (size_t)(m + N - 1);
                acb_srcptr src = G + (size_t)n_mod * (size_t)P + (size_t)m_mod;
                if (sn * sm > 0) acb_set(dst, src);
                else             acb_neg(dst, src);
            }
        }
    }
    _acb_vec_clear(g, (size_t)P * (size_t)P);
    _acb_vec_clear(G, (size_t)P * (size_t)P);

    /* -------- Stage 2: GL theta inner product, norm = 1 / (2 P^2)
     * (notes/0t1_resolved_grid_design.md §5).  Sanity:
     *   norm = (1/(8 pi^2)) * dphi * dpsi   (Haar prefactor x phi/psi step)
     *        = (1/(8 pi^2)) * (2 pi / P)^2
     *        = 1 / (2 P^2).
     *
     * Per (l, m, n) with l >= max(|m|, |n|):
     *   fhat(l)_{m,n} = norm * sum_k w_k * conj(P^l_{n,m}(cos theta_k))
     *                                    * F2[k, n+N-1, m+N-1].
     * We call su2_wigner_d_arb (which returns the COMPLEX P^l_{n,m} =
     * i^{m-n} * d^l_{n,m}) and conjugate, mirroring src/su2_ft_resolved_arb.c
     * Steps 7-8. */
    arb_t norm;
    arb_init(norm);
    arb_one(norm);
    arb_div_ui(norm, norm, 2u * (ulong)P * (ulong)P, prec);

    acb_t acc, P_lnm, term, weighted_P;
    acb_init(acc); acb_init(P_lnm); acb_init(term); acb_init(weighted_P);

    for (int l = 0; l < N; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int n = -l; n <= l; ++n) {
                acb_zero(acc);
                for (int k = 0; k < N; ++k) {
                    su2_wigner_d_arb(P_lnm, l, n, m, theta + k, prec);
                    acb_conj(P_lnm, P_lnm);
                    acb_mul_arb(weighted_P, P_lnm, w_gl + k, prec);
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
    _arb_vec_clear(theta, N);
    _arb_vec_clear(w_gl,  N);
    _arb_vec_clear(x_gl,  N);
}

/**
 * @brief Arbitrary-precision O(N^4) inverse FFT on SU(2), resolved grid.
 *
 * Arb twin of su2_fft_resolved_inv().  Peter-Weyl synthesis (paper.tex line 554):
 *     f(g) = sum_l (2l+1) sum_{m,n} fhat(l)_{m,n} * t^l_{n,m}(g)
 * with t^l_{n,m}(g) = P^l_{n,m}(cos theta) * exp(-i(n phi + m psi)).
 *
 * Stage 2-inv (Wigner synthesis at theta_k):
 *   G[k, n, m] = sum_l (2l+1) * fhat(l)_{m,n} * P^l_{n,m}(cos theta_k).
 *   This is the EXACT discrete synthesis -- no quadrature weight, no
 *   sin(theta_k), no global norm.  We call su2_wigner_d_arb directly
 *   (no conjugate: this is synthesis, not analysis).
 *
 * Stage 1-inv (P x P FORWARD DFT per theta slice, no fold):
 *   acb_dft_prod IS forward, so we call it directly (no conj trick).
 *   Each n in [-(N-1), N-1] maps to a unique residue mod P, so the
 *   scatter into the DFT input is an assignment (not +=), mirroring
 *   src/su2_fft_resolved.c.  No closed-grid edge fill: the open grid
 *   has no duplicated j = P-1 sample.
 *
 * @param[in]  N     Bandlimit; output sample grid is P x N x P, P = 2N-1.
 * @param[in]  fhat  acb_srcptr of length su2_total_coeffs(N).
 * @param[out] f     acb_ptr of length P*P*N, row-major (j1, k, j2).
 * @param[in]  prec  Working precision in bits.
 * @par Complexity O(N^4) acb multiplications; O(N^3) auxiliary acb memory.
 * @par Reference paper.tex line 554; notes/0t1_resolved_grid_design.md §4;
 *                src/su2_fft_resolved.c (double-precision twin).
 */
void su2_fft_resolved_inv_arb(int N, acb_srcptr fhat, acb_ptr f, slong prec)
{
    assert(N >= 2 && "su2_fft_resolved_inv_arb: N must be >= 2");
    assert(fhat != NULL && "su2_fft_resolved_inv_arb: fhat must be non-NULL");
    assert(f != NULL && "su2_fft_resolved_inv_arb: f must be non-NULL");

    const int P = 2 * N - 1;

    /* ----- GL nodes; weights unused by the synthesis (Stage 2-inv carries
     * NO quadrature weight -- paper.tex line 554 is an exact pointwise sum). */
    arb_ptr x_gl  = _arb_vec_init(N);
    arb_ptr w_gl  = _arb_vec_init(N);   /* allocated for the wrapper API; unused. */
    arb_ptr theta = _arb_vec_init(N);
    su2_gl_nodes_weights_arb(N, x_gl, w_gl, prec);
    for (int k = 0; k < N; ++k) arb_acos(theta + k, x_gl + k, prec);

    /* -------- Stage 2-inv: Wigner synthesis at theta_k.
     * G[k, n+N-1, m+N-1] = sum_l (2l+1) * fhat(l)_{m,n} * P^l_{n,m}(cos theta_k).
     * Use the FULL complex P^l_{n,m} (i^{m-n} * d^l_{n,m}) at each iteration;
     * the factor-out-the-phase optimisation (cf. src/su2_fft_resolved.c) is
     * deferred -- this simpler form is identical mathematically and easier
     * to verify against the direct path. */
    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;
    acb_ptr G = _acb_vec_init((size_t)N * stride_k);

    acb_t acc, P_lnm, term;
    acb_init(acc); acb_init(P_lnm); acb_init(term);

    for (int m = -(N - 1); m <= N - 1; ++m) {
        for (int n = -(N - 1); n <= N - 1; ++n) {
            int l_min = (abs(m) > abs(n)) ? abs(m) : abs(n);
            if (l_min > N - 1) continue;

            for (int k = 0; k < N; ++k) {
                acb_zero(acc);
                for (int l = l_min; l < N; ++l) {
                    su2_wigner_d_arb(P_lnm, l, n, m, theta + k, prec);
                    acb_mul_si(term, P_lnm, 2 * l + 1, prec);
                    acb_mul(term, term,
                            fhat + su2_coeff_offset(l) + su2_mn_index(l, m, n),
                            prec);
                    acb_add(acc, acc, term, prec);
                }
                acb_set(G + (size_t)k * stride_k
                          + (size_t)(n + N - 1) * stride_n
                          + (size_t)(m + N - 1),
                        acc);
            }
        }
    }

    acb_clear(acc); acb_clear(P_lnm); acb_clear(term);

    /* -------- Stage 1-inv: P x P FORWARD DFT per theta slice.
     * For each k: scatter G[k, n, m] into G_slice[n_mod, m_mod] with the
     * (-1)^{n+m} phase, then call acb_dft_prod (forward) DIRECTLY -- no
     * conj trick needed for this direction.  The scatter is an assignment
     * (not +=): each (n_mod, m_mod) bin is touched by exactly one (n, m).
     * No closed-grid edge fill (the open grid has no duplicated endpoint).
     * notes/0t1_resolved_grid_design.md §4 (Stage 1-inv). */
    slong cyc[2] = { (slong)P, (slong)P };
    acb_ptr G_slice = _acb_vec_init((size_t)P * (size_t)P);
    acb_ptr f_slice = _acb_vec_init((size_t)P * (size_t)P);

    for (int k = 0; k < N; ++k) {
        _acb_vec_zero(G_slice, (size_t)P * (size_t)P);

        for (int n = -(N - 1); n <= N - 1; ++n) {
            int n_mod = ((n % P) + P) % P;
            int sn    = (n & 1) ? -1 : 1;
            for (int m = -(N - 1); m <= N - 1; ++m) {
                int m_mod = ((m % P) + P) % P;
                int sm    = (m & 1) ? -1 : 1;
                acb_srcptr src = G + (size_t)k * stride_k
                                   + (size_t)(n + N - 1) * stride_n
                                   + (size_t)(m + N - 1);
                acb_ptr    dst = G_slice + (size_t)n_mod * (size_t)P
                                         + (size_t)m_mod;
                if (sn * sm > 0) acb_set(dst, src);
                else             acb_neg(dst, src);
            }
        }

        /* acb_dft_prod is FORWARD; that is the direction we need here. */
        acb_dft_prod(f_slice, G_slice, cyc, 2, prec);

        for (int j1 = 0; j1 < P; ++j1) {
            for (int j2 = 0; j2 < P; ++j2) {
                acb_set(f + su2_resolved_sample_index(N, j1, k, j2),
                        f_slice + (size_t)j1 * (size_t)P + (size_t)j2);
            }
        }
    }

    _acb_vec_clear(G_slice, (size_t)P * (size_t)P);
    _acb_vec_clear(f_slice, (size_t)P * (size_t)P);
    _acb_vec_clear(G, (size_t)N * stride_k);
    _arb_vec_clear(theta, N);
    _arb_vec_clear(w_gl,  N);
    _arb_vec_clear(x_gl,  N);
}
