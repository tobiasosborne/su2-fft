/* su2_grid.c -- Euler-angle grid and coefficient layout helpers.
 *
 * Paper reference: line 684 of paper.tex (Sampling-grid definition).
 *     phi[j1]  = -pi + j1 * 2pi/(N-1)
 *     psi[j2]  = -pi + j2 * 2pi/(N-1)
 *     theta[k] =       k *  pi/(N-1)
 *
 * Layout reference: see comments in include/su2.h.
 */
#include "su2.h"

#include <math.h>
#include <stdlib.h>

static double *grid_uniform(double start, double end, int N)
{
    if (N < 2) return NULL;
    double *g = malloc((size_t)N * sizeof(double));
    if (!g) return NULL;
    double step = (end - start) / (double)(N - 1);
    for (int i = 0; i < N; ++i) g[i] = start + (double)i * step;
    /* Pin endpoints to exact values to kill float drift at i = N-1. */
    g[0]     = start;
    g[N - 1] = end;
    return g;
}

/**
 * @brief Allocate and fill the phi Euler-angle grid.
 *
 * Returns a heap-allocated array of length N holding the azimuthal angles
 * phi[j] = -pi + j * 2*pi/(N-1) for j in [0, N-1], as defined at paper.tex
 * line 684.  Both endpoints are pinned to exact values to suppress float drift.
 *
 * @param[in] N Bandlimit / number of grid points.  Must be >= 2.
 * @return Pointer to a length-N array (caller frees with free()), or NULL on
 *         allocation failure or N < 2.
 * @par Complexity O(N) time and space.
 * @par Reference paper.tex line 684; ALGORITHM.md Section 1.
 */
double *su2_grid_phi  (int N) { return grid_uniform(-M_PI, M_PI, N); }

/**
 * @brief Allocate and fill the psi Euler-angle grid.
 *
 * Identical layout to su2_grid_phi(): psi[j] = -pi + j * 2*pi/(N-1),
 * j in [0, N-1] (paper.tex line 684).  The two grids coincide numerically;
 * separate functions are provided for clarity at call sites.
 *
 * @param[in] N Bandlimit / number of grid points.  Must be >= 2.
 * @return Pointer to a length-N array (caller frees with free()), or NULL on
 *         allocation failure or N < 2.
 * @par Complexity O(N) time and space.
 * @par Reference paper.tex line 684; ALGORITHM.md Section 1.
 */
double *su2_grid_psi  (int N) { return grid_uniform(-M_PI, M_PI, N); }

/**
 * @brief Allocate and fill the theta (polar) Euler-angle grid.
 *
 * Returns a heap-allocated array of length N holding theta[k] = k*pi/(N-1)
 * for k in [0, N-1], as defined at paper.tex line 684.
 *
 * @param[in] N Bandlimit / number of grid points.  Must be >= 2.
 * @return Pointer to a length-N array (caller frees with free()), or NULL on
 *         allocation failure or N < 2.
 * @par Complexity O(N) time and space.
 * @par Reference paper.tex line 684; ALGORITHM.md Section 1.
 */
double *su2_grid_theta(int N) { return grid_uniform(  0.0, M_PI, N); }

/**
 * @brief Total number of Fourier coefficients for bandlimit N.
 *
 * Counts all independent matrix entries fhat(l)_{m,n} across l = 0..N-1:
 *   sum_{l=0}^{N-1} (2l+1)^2 = N(2N-1)(2N+1)/3.
 * This equals the flat-array length needed to store the full coefficient
 * block as laid out by su2_coeff_offset() + su2_mn_index().
 *
 * @param[in] N Bandlimit (number of degree levels).  Returns 0 for N <= 0.
 * @return Total coefficient count as a size_t.
 * @par Complexity O(1).
 * @par Reference include/su2.h layout comment; ALGORITHM.md Section 1.
 */
/* su2_total_coeffs(N) = sum_{l=0..N-1} (2l+1)^2
 *                    = N(2N-1)(2N+1)/3                                 */
size_t su2_total_coeffs(int N)
{
    if (N <= 0) return 0;
    size_t Nu = (size_t)N;
    return Nu * (2*Nu - 1) * (2*Nu + 1) / 3;
}

/**
 * @brief Offset of degree-l coefficient block in the flat fhat array.
 *
 * Returns the index into the flat coefficient array where the (2l+1)x(2l+1)
 * matrix for degree l begins:
 *   offset(l) = sum_{j=0}^{l-1} (2j+1)^2 = l(2l-1)(2l+1)/3.
 * Combine with su2_mn_index(l, m, n) to address a single coefficient.
 *
 * @param[in] l Degree.  Returns 0 for l <= 0.
 * @return Byte offset (in units of double _Complex elements) to the start of
 *         the degree-l block.
 * @par Complexity O(1).
 * @par Reference include/su2.h layout comment; ALGORITHM.md Section 1.
 */
/* Offset of fhat[l] in the flat coefficient array. */
size_t su2_coeff_offset(int l)
{
    if (l <= 0) return 0;
    /* sum_{j=0..l-1} (2j+1)^2 = l(2l-1)(2l+1)/3 */
    size_t lu = (size_t)l;
    return lu * (2*lu - 1) * (2*lu + 1) / 3;
}
