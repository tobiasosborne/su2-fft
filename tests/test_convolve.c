/* test_convolve.c -- Tests for SU(2) convolution via the spectrum. */
#include "test_framework.h"
#include "su2.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

static double urand(void) { return (double)rand() / (double)RAND_MAX; }

static void test_convolve_identity_constant(void)
{
    /* fhat = identity-in-l=0 (constant 1's spectrum):
     *   fhat(0)_{0,0} = 1, all other entries = 0.
     * Convolve with arbitrary ghat:
     *   fghat(0)_{0,0} = fhat(0)_{0,0} * ghat(0)_{0,0} = ghat(0)_{0,0}.
     *   fghat(l>0) = 0  (since fhat(l>0) is zero matrix).
     */
    int N = 6;
    size_t nc = su2_total_coeffs(N);
    double _Complex *fhat = calloc(nc, sizeof(double _Complex));
    double _Complex *ghat = malloc(nc * sizeof(double _Complex));
    double _Complex *fghat = malloc(nc * sizeof(double _Complex));

    fhat[0] = 1.0 + 0.0*I;
    srand(20260526);
    for (size_t i = 0; i < nc; ++i) ghat[i] = (urand() - 0.5) + (urand() - 0.5)*I;

    su2_convolve(N, fhat, ghat, fghat);

    ASSERT_CNEAR(fghat[0], ghat[0], 1e-13);
    /* All other entries should be 0. */
    for (size_t i = 1; i < nc; ++i) {
        ASSERT_NEAR(cabs(fghat[i]) < 1e-13 ? 1.0 : 0.0, 1.0, 1e-15);
    }

    free(fhat); free(ghat); free(fghat);
}

static void test_convolve_zero_input(void)
{
    int N = 5;
    size_t nc = su2_total_coeffs(N);
    double _Complex *zeros = calloc(nc, sizeof(double _Complex));
    double _Complex *ghat = malloc(nc * sizeof(double _Complex));
    double _Complex *fghat = malloc(nc * sizeof(double _Complex));
    srand(1);
    for (size_t i = 0; i < nc; ++i) ghat[i] = (urand() - 0.5) + (urand() - 0.5)*I;

    su2_convolve(N, zeros, ghat, fghat);
    for (size_t i = 0; i < nc; ++i) ASSERT_NEAR(cabs(fghat[i]), 0.0, 1e-14);

    su2_convolve(N, ghat, zeros, fghat);
    for (size_t i = 0; i < nc; ++i) ASSERT_NEAR(cabs(fghat[i]), 0.0, 1e-14);

    free(zeros); free(ghat); free(fghat);
}

static void test_convolve_linearity_first_arg(void)
{
    /* convolve(a*f1 + b*f2, g) == a*convolve(f1,g) + b*convolve(f2,g) */
    int N = 5;
    size_t nc = su2_total_coeffs(N);
    double _Complex *f1 = malloc(nc * sizeof(double _Complex));
    double _Complex *f2 = malloc(nc * sizeof(double _Complex));
    double _Complex *g  = malloc(nc * sizeof(double _Complex));
    double _Complex *combo  = malloc(nc * sizeof(double _Complex));
    double _Complex *cv1 = malloc(nc * sizeof(double _Complex));
    double _Complex *cv2 = malloc(nc * sizeof(double _Complex));
    double _Complex *cv_combo = malloc(nc * sizeof(double _Complex));

    srand(2);
    for (size_t i = 0; i < nc; ++i) {
        f1[i] = (urand() - 0.5) + (urand() - 0.5)*I;
        f2[i] = (urand() - 0.5) + (urand() - 0.5)*I;
        g[i]  = (urand() - 0.5) + (urand() - 0.5)*I;
    }
    double _Complex a = 2.0 + 1.5*I;
    double _Complex b = -0.7 + 0.4*I;
    for (size_t i = 0; i < nc; ++i) combo[i] = a*f1[i] + b*f2[i];

    su2_convolve(N, f1, g, cv1);
    su2_convolve(N, f2, g, cv2);
    su2_convolve(N, combo, g, cv_combo);

    for (size_t i = 0; i < nc; ++i) {
        double _Complex expected = a*cv1[i] + b*cv2[i];
        ASSERT_CNEAR(cv_combo[i], expected, 1e-12);
    }

    free(f1); free(f2); free(g); free(combo); free(cv1); free(cv2); free(cv_combo);
}

static void test_convolve_aliasing_first_arg(void)
{
    /* convolve(f, g, f) should give same result as convolve(f, g, out) */
    int N = 5;
    size_t nc = su2_total_coeffs(N);
    double _Complex *f = malloc(nc * sizeof(double _Complex));
    double _Complex *g = malloc(nc * sizeof(double _Complex));
    double _Complex *f_copy = malloc(nc * sizeof(double _Complex));
    double _Complex *out_separate = malloc(nc * sizeof(double _Complex));

    srand(3);
    for (size_t i = 0; i < nc; ++i) {
        f[i] = (urand() - 0.5) + (urand() - 0.5)*I;
        g[i] = (urand() - 0.5) + (urand() - 0.5)*I;
    }
    memcpy(f_copy, f, nc * sizeof(double _Complex));

    /* In-place: write into f. */
    su2_convolve(N, f, g, f);

    /* Separate: write into out_separate. */
    su2_convolve(N, f_copy, g, out_separate);

    for (size_t i = 0; i < nc; ++i) {
        ASSERT_CNEAR(f[i], out_separate[i], 1e-13);
    }

    free(f); free(g); free(f_copy); free(out_separate);
}

static void test_convolve_l1_explicit(void)
{
    /* At l=1: 3x3 matrix product. Use an explicit case to anchor the math.
     * Set fhat(1) = diag(1, 2, 3) and ghat(1) = diag(4, 5, 6).
     * Then (fhat * ghat)(1) = diag(4, 10, 18). */
    int N = 2;  /* covers l = 0, 1 */
    size_t nc = su2_total_coeffs(N);
    double _Complex *fhat = calloc(nc, sizeof(double _Complex));
    double _Complex *ghat = calloc(nc, sizeof(double _Complex));
    double _Complex *fghat = calloc(nc, sizeof(double _Complex));

    /* l=0 block: just one entry. Set both to 1 for simplicity. */
    fhat[0] = 1.0; ghat[0] = 1.0;

    /* l=1 block: 3x3, indices (m+1, n+1) with m, n in {-1, 0, 1}. */
    size_t off1 = su2_coeff_offset(1);
    /* Diagonal: (m+1, n+1) = (0,0), (1,1), (2,2). */
    for (int idx = 0; idx < 3; ++idx) {
        size_t flat = off1 + (size_t)idx * 3 + (size_t)idx;
        fhat[flat] = (double)(idx + 1);              /* 1, 2, 3 */
        ghat[flat] = (double)(idx + 4);              /* 4, 5, 6 */
    }

    su2_convolve(N, fhat, ghat, fghat);

    /* Check diagonal of l=1 block. */
    double expected[3] = {4.0, 10.0, 18.0};
    for (int idx = 0; idx < 3; ++idx) {
        size_t flat = off1 + (size_t)idx * 3 + (size_t)idx;
        ASSERT_CNEAR(fghat[flat], expected[idx] + 0.0*I, 1e-13);
    }
    /* Check off-diagonal of l=1 block is zero. */
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j) continue;
            ASSERT_NEAR(cabs(fghat[off1 + (size_t)i * 3 + (size_t)j]), 0.0, 1e-13);
        }
    }

    /* Check l=0 block. */
    ASSERT_CNEAR(fghat[0], 1.0 + 0.0*I, 1e-13);

    free(fhat); free(ghat); free(fghat);
}

int main(void)
{
    RUN(test_convolve_zero_input);
    RUN(test_convolve_identity_constant);
    RUN(test_convolve_l1_explicit);
    RUN(test_convolve_linearity_first_arg);
    RUN(test_convolve_aliasing_first_arg);
    TEST_REPORT_AND_EXIT();
}
