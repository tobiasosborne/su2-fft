/* su2_arb.h -- Arbitrary-precision SU(2) Fourier transform via FLINT.
 *
 * Mirrors include/su2.h but uses FLINT's acb_t (complex ball) and arb_t
 * (real ball) types.  Every routine takes a precision argument `prec` in bits
 * (e.g. prec = 53 matches IEEE double; prec = 256 is "high precision").
 *
 * Layout: same as su2.h.  fhat[ su2_coeff_offset(l) + su2_mn_index(l,m,n) ]
 * is now an acb_t.
 */
#ifndef SU2_ARB_H
#define SU2_ARB_H

#include "su2.h"
#include <flint/acb.h>
#include <flint/arb.h>

/* Allocate / free a vector of acb_t initialised to zero (FLINT _acb_vec_*
 * idioms are exposed for convenience; users may call them directly). */

/* Paper's P^l_{n,m}(cos theta) at precision `prec`. */
void su2_wigner_d_arb(acb_t out, int l, int n, int m,
                      const arb_t theta, slong prec);

/* Direct O(N^6) Fourier transform.  f and fhat are FLINT acb_ptr arrays of
 * length N*N*N and su2_total_coeffs(N), respectively. */
void su2_ft_direct_arb(int N, acb_srcptr f, acb_ptr fhat, slong prec);

/* Fast O(N^4) Fourier transform via acb_dft_prod for Stage 1. */
void su2_fft_arb(int N, acb_srcptr f, acb_ptr fhat, slong prec);

/* ------- Gauss-Legendre at arbitrary precision (bead su2fft-rrx) -------
 *
 * N-point Gauss-Legendre nodes (ascending) and weights on [-1, 1] in arb.
 * Wraps FLINT's arb_hypgeom_legendre_p_ui_root.  Twin of the double-
 * precision su2_gl_nodes_weights().  See src/su2_gauss_legendre_arb.c.
 *
 * @param[in]  N     Number of nodes (>= 1).
 * @param[out] x     arb_ptr of length N, ascending; x[0] = -x[N-1].
 * @param[out] w     arb_ptr of length N, the GL weights.
 * @param[in]  prec  Working precision in bits.
 */
void su2_gl_nodes_weights_arb(int N, arb_ptr x, arb_ptr w, slong prec);

/* ------- Direct O(N^6) FT on the resolved grid (bead su2fft-rrx) -------
 *
 * Arb port of su2_ft_direct_resolved.  Sample layout, grid, and norm match
 * the double-precision routine (notes/0t1_resolved_grid_design.md §5):
 *     f[su2_resolved_sample_index(N, j1, k, j2)],  length P*P*N, P = 2N-1.
 *     phi[j] = -pi + j * 2pi/P,  theta_k = acos(x_k) (GL),
 *     norm = 1 / (2 P^2).
 *
 * @param[in]  N     Bandlimit; sample grid is P x N x P, P = 2N-1.
 * @param[in]  f     acb_srcptr of length P*P*N.
 * @param[out] fhat  acb_ptr of length su2_total_coeffs(N); overwritten.
 * @param[in]  prec  Working precision in bits.
 */
void su2_ft_direct_resolved_arb(int N, acb_srcptr f, acb_ptr fhat, slong prec);

/* ------- Fast O(N^4) FFT on the resolved grid (bead su2fft-rrx, Step 9) -----
 *
 * Arb twins of su2_fft_resolved / su2_fft_resolved_inv.  Same sample layout
 * and norm as the direct arb routine (notes/0t1_resolved_grid_design.md §4-§5):
 *     f[su2_resolved_sample_index(N, j1, k, j2)], length P*P*N, P = 2N-1.
 *     phi[j] = -pi + j * 2pi/P,  theta_k = acos(x_k) (GL),
 *     norm   = 1 / (2 P^2).
 *
 * Stage 1 uses acb_dft_prod (length P x P, no fold) with the conj trick for
 * the backward direction in the forward routine; the inverse uses
 * acb_dft_prod directly (forward direction).  See src/su2_fft_resolved_arb.c.
 *
 * @param[in]  N     Bandlimit; sample grid is P x N x P, P = 2N-1.
 * @param[in]  f / fhat   acb_srcptr input of the appropriate length.
 * @param[out] fhat / f   acb_ptr output; overwritten.
 * @param[in]  prec  Working precision in bits.
 */
void su2_fft_resolved_arb(int N, acb_srcptr f, acb_ptr fhat, slong prec);
void su2_fft_resolved_inv_arb(int N, acb_srcptr fhat, acb_ptr f, slong prec);

#endif /* SU2_ARB_H */
