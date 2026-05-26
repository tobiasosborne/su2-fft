/* su2_gauss_legendre_arb.c -- Arbitrary-precision N-point Gauss-Legendre nodes
 * and weights on [-1, 1] via FLINT.
 *
 * Bead: su2fft-rrx (Step 8, Deliverable A).
 * Spec: notes/0t1_resolved_grid_design.md §8.
 *
 * Wraps FLINT's `arb_hypgeom_legendre_p_ui_root(res, weight, n, k, prec)`.
 *
 * FLINT root-indexing convention (verbatim, /usr/include/flint/arb_hypgeom.h
 * and flint/doc/source/arb_hypgeom.rst):
 *
 *     "Sets res to the k-th root of the Legendre polynomial P_n(x).
 *      We index the roots in decreasing order
 *          1 > x_0 > x_1 > ... > x_{n-1} > -1
 *      (which corresponds to ordering the roots of P_n(cos(theta))
 *      in order of increasing theta).
 *      If weight is non-NULL, it is set to the weight corresponding
 *      to the node x_k for Gaussian quadrature on [-1, 1]."
 *
 * So FLINT returns roots in DESCENDING order (k = 0 is the largest positive
 * root, k = N-1 is the most negative).  We need ASCENDING output, so
 *     x_ascending[i] = root(N - 1 - i),    i in [0, N-1].
 * The symmetry  x_k = -x_{N-1-k}  is built into FLINT but we still call it
 * for each k to obtain rigorous balls (no manual mirroring).
 *
 * Cross-checks (smoke run at prec=128, N=8):
 *   sum_k w_k = 2 to ball width ~2^-127.
 *   x[0] + x[N-1] = 0 to ball width ~2^-127.
 * Mid-points agree with the double Newton routine (src/su2_gauss_legendre.c)
 * to ~1e-15 at prec=53.
 */
#include "su2.h"
#include "su2_arb.h"

#include <flint/arb.h>
#include <flint/arb_hypgeom.h>
#include <assert.h>

/**
 * @brief N-point Gauss-Legendre nodes (ascending) and weights on [-1, 1] in arb.
 *
 * Wraps FLINT's arb_hypgeom_legendre_p_ui_root(res, weight, n, k, prec):
 * returns the k-th root of P_n and the associated GL weight at `prec` bits.
 * FLINT indexes roots in DECREASING order (x_0 = largest positive root,
 * x_{N-1} = most negative); this routine reorders to ASCENDING.
 *
 * @param[in]  N     Number of nodes (>= 1).
 * @param[out] x     arb_ptr of length N, ascending; x[0] = -x[N-1] (most negative).
 * @param[out] w     arb_ptr of length N, the GL weights (w_k = w_{N-1-k}).
 * @param[in]  prec  Working precision in bits.
 *
 * @par Complexity O(N) FLINT root computations; each is asymptotic guess +
 *      a few Newton iterations at `prec` bits.
 * @par Reference /usr/include/flint/arb_hypgeom.h;
 *                notes/0t1_resolved_grid_design.md §8;
 *                src/su2_gauss_legendre.c (double-precision twin).
 */
void su2_gl_nodes_weights_arb(int N, arb_ptr x, arb_ptr w, slong prec)
{
    assert(N >= 1 && "su2_gl_nodes_weights_arb: N must be >= 1");
    assert(x != NULL && "su2_gl_nodes_weights_arb: x must be non-NULL");
    assert(w != NULL && "su2_gl_nodes_weights_arb: w must be non-NULL");

    /* FLINT returns roots in DECREASING order: x_flint(0) > x_flint(1) > ...
     * For ascending output, x_ascending[i] = x_flint(N - 1 - i).
     * Same mapping for the weights (which are symmetric anyway).         */
    for (int i = 0; i < N; ++i) {
        ulong k_flint = (ulong)(N - 1 - i);
        arb_hypgeom_legendre_p_ui_root(x + i, w + i,
                                       (ulong)N, k_flint, prec);
    }
}
