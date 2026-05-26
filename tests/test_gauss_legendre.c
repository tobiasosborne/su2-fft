/* test_gauss_legendre.c -- Verify su2_gl_nodes_weights properties.
 *
 * Six properties checked:
 * 1. Nodes ascending and in (-1, 1).
 * 2. Symmetry: x[k] = -x[N-1-k].
 * 3. Weight sum = 2 (integral of 1 over [-1, 1]).
 * 4. Integral of x^p over [-1, 1] matches GL for p = 0..2N-1 (degree exactness).
 *    Closed form: int_{-1}^{1} x^p dx = 0 for odd p, 2/(p+1) for even p.
 * 5. Weights positive.
 * 6. Specific N=4 values vs. published table (CRC standard).
 */
#include "test_framework.h"
#include "su2.h"

#include <math.h>
#include <stdlib.h>

static void test_gl_basic_properties(void)
{
    /* For N in {1, 2, 5, 8, 16}, verify ascending order, symmetry, weights > 0. */
    int Ns[] = {1, 2, 5, 8, 16};
    for (int idx = 0; idx < 5; ++idx) {
        int N = Ns[idx];
        double *x = malloc((size_t)N * sizeof(double));
        double *w = malloc((size_t)N * sizeof(double));
        su2_gl_nodes_weights(N, x, w);

        /* Ascending. */
        for (int k = 1; k < N; ++k) {
            if (!(x[k-1] < x[k])) {
                fprintf(stderr, "  ! ascending fail at N=%d, k=%d: x[%d]=%g x[%d]=%g\n",
                        N, k, k-1, x[k-1], k, x[k]);
            }
            ASSERT_NEAR(x[k-1] < x[k] ? 1.0 : 0.0, 1.0, 1e-15);
        }
        /* In open (-1, 1). */
        for (int k = 0; k < N; ++k) {
            ASSERT_NEAR(fabs(x[k]) < 1.0 ? 1.0 : 0.0, 1.0, 1e-15);
        }
        /* Symmetry. */
        for (int k = 0; k < N; ++k) {
            ASSERT_NEAR(x[k] + x[N - 1 - k], 0.0, 1e-14);
        }
        /* Weights positive. */
        for (int k = 0; k < N; ++k) {
            ASSERT_NEAR(w[k] > 0.0 ? 1.0 : 0.0, 1.0, 1e-15);
        }
        /* Weight sum = 2. */
        double sw = 0.0;
        for (int k = 0; k < N; ++k) sw += w[k];
        ASSERT_NEAR(sw, 2.0, 1e-13);

        free(x);
        free(w);
    }
}

static void test_gl_degree_exactness(void)
{
    /* For N-point GL: exact for polynomials x^p, p <= 2N-1.
     * int_{-1}^{1} x^p dx = 0 (odd p) or 2/(p+1) (even p).
     * Test all p in [0, 2N-1] for N = 4, 8.
     */
    int Ns[] = {4, 8};
    for (int idx = 0; idx < 2; ++idx) {
        int N = Ns[idx];
        double *x = malloc((size_t)N * sizeof(double));
        double *w = malloc((size_t)N * sizeof(double));
        su2_gl_nodes_weights(N, x, w);

        for (int p = 0; p <= 2 * N - 1; ++p) {
            double approx = 0.0;
            for (int k = 0; k < N; ++k) approx += w[k] * pow(x[k], (double)p);
            double exact = (p % 2) ? 0.0 : 2.0 / ((double)p + 1.0);
            /* Tolerance: degree-exact, so machine precision. */
            ASSERT_NEAR(approx, exact, 1e-12);
        }

        free(x); free(w);
    }
}

static void test_gl_specific_n4(void)
{
    /* N=4: nodes are +/- sqrt((3 - 2*sqrt(6/5))/7), +/- sqrt((3 + 2*sqrt(6/5))/7).
     * Weights: (18 +/- sqrt(30)) / 36. */
    int N = 4;
    double x[4], w[4];
    su2_gl_nodes_weights(N, x, w);

    double x_outer = sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    double x_inner = sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0);

    /* Ascending: x[0] = -x_outer, x[1] = -x_inner, x[2] = x_inner, x[3] = x_outer. */
    ASSERT_NEAR(x[0], -x_outer, 1e-14);
    ASSERT_NEAR(x[1], -x_inner, 1e-14);
    ASSERT_NEAR(x[2],  x_inner, 1e-14);
    ASSERT_NEAR(x[3],  x_outer, 1e-14);

    double w_outer = (18.0 - sqrt(30.0)) / 36.0;
    double w_inner = (18.0 + sqrt(30.0)) / 36.0;
    ASSERT_NEAR(w[0], w_outer, 1e-14);
    ASSERT_NEAR(w[1], w_inner, 1e-14);
    ASSERT_NEAR(w[2], w_inner, 1e-14);
    ASSERT_NEAR(w[3], w_outer, 1e-14);
}

int main(void)
{
    RUN(test_gl_basic_properties);
    RUN(test_gl_degree_exactness);
    RUN(test_gl_specific_n4);
    TEST_REPORT_AND_EXIT();
}
