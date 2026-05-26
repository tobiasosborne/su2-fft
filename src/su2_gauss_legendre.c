/* su2_gauss_legendre.c -- N-point Gauss-Legendre nodes and weights on [-1, 1].
 *
 * Standard Newton iteration on the Legendre polynomial (Press et al.,
 * "Numerical Recipes in C", 3rd ed., §4.6).  Nodes come in symmetric pairs;
 * the loop exploits symmetry by computing only the lower half.
 *
 * Used by su2_fft_gl and su2_fft_inv_gl (bead su2fft-ega) to integrate the
 * theta direction exactly for polynomials in cos(theta) of degree <= 2N-1.
 *
 * Spec: notes/gauss_legendre.md §3, §4.
 */
#include "su2.h"

#include <assert.h>
#include <math.h>

/**
 * @brief Compute N-point Gauss-Legendre nodes (ascending) and weights on [-1, 1].
 *
 * Nodes are the N roots of the Legendre polynomial P_N(x); weights are
 *   w_k = 2 / ((1 - x_k^2) * (P_N'(x_k))^2).
 * Newton iteration converges to 1e-15 in 4-6 steps for N up to several hundred.
 *
 * @param[in]  N  Number of nodes (>= 1).
 * @param[out] x  Length-N array of nodes, ascending order (x[0] = -x[N-1] most negative).
 * @param[out] w  Length-N array of weights.
 *
 * @par Reference Press et al. "Numerical Recipes in C" §4.6; DLMF §3.5(v).
 */
void su2_gl_nodes_weights(int N, double *x, double *w)
{
    assert(N >= 1);
    assert(x != NULL);
    assert(w != NULL);

    /* Roots come in symmetric pairs; compute only the lower half (i <= (N+1)/2). */
    int m = (N + 1) / 2;
    for (int i = 1; i <= m; ++i) {
        /* Initial guess (Press §4.6 eq. 4.6.4). */
        double xi = cos(M_PI * ((double)i - 0.25) / ((double)N + 0.5));
        double pp, p1, p2, p3;
        double xi_prev;
        int iter = 0;
        const int MAX_ITER = 50;
        do {
            /* Evaluate P_N(xi) and P_{N-1}(xi) via the three-term recurrence:
             *   j P_j(x) = (2j-1) x P_{j-1}(x) - (j-1) P_{j-2}(x).
             * After the loop:  p1 = P_N(xi);  p2 = P_{N-1}(xi).
             */
            p1 = 1.0;
            p2 = 0.0;
            for (int j = 1; j <= N; ++j) {
                p3 = p2;
                p2 = p1;
                p1 = ((2.0 * (double)j - 1.0) * xi * p2 - ((double)j - 1.0) * p3) / (double)j;
            }
            /* Derivative: P_N'(x) = N (x P_N(x) - P_{N-1}(x)) / (x^2 - 1). */
            pp = (double)N * (xi * p1 - p2) / (xi * xi - 1.0);
            xi_prev = xi;
            xi -= p1 / pp;
            ++iter;
        } while (fabs(xi - xi_prev) > 1e-15 && iter < MAX_ITER);
        assert(iter < MAX_ITER);  /* Newton must converge for valid input. */

        /* Symmetric pair: x_i is positive (initial guess gives the i-th positive root).
         * For ascending output, the i-th positive root (counted from 1) goes at
         * index (N - i) (zero-based); its negative goes at index (i - 1). */
        x[i - 1]     = -xi;
        x[N - i]     =  xi;
        w[i - 1]     =  2.0 / ((1.0 - xi * xi) * pp * pp);
        w[N - i]     =  w[i - 1];
    }
}
