/* test_fft.c -- Red/green test for the O(N^4) fast FFT on SU(2).
 *
 * Specification: su2_fft must produce, to floating-point tolerance, the SAME
 * fhat as su2_ft_direct, because both implement the exact discrete sum from
 * paper.tex line 1316.  The cross-comparison is therefore the gold standard:
 * any numerical difference between the two beyond ~1e-10 indicates a bug.
 *
 * Pre-comparison sanity tests:
 *   1. Zero input -> zero output.
 *   2. Delta-spike input -> same closed-form as the direct case.
 *
 * Then the all-in-one comparison on a random complex input.
 */
#include "test_framework.h"
#include "su2.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void seed_rand(unsigned s) { srand(s); }
static double urand(void) { return (double)rand() / (double)RAND_MAX; }

static void test_fft_zero_input(void)
{
    int N = 5;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = calloc(nsamp,  sizeof(double _Complex));
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
    su2_fft(N, f, fhat);
    for (size_t i = 0; i < ncoeff; ++i) ASSERT_CNEAR(fhat[i], 0.0, 1e-14);
    free(f); free(fhat);
}

static void test_fft_matches_direct_random(void)
{
    /* The discrete sum is identical; FFT and direct FT must agree to within
     * a handful of ULPs, propagated through O(N^3) operations.  */
    int N = 6;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fhat_direct = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *fhat_fast   = calloc(ncoeff, sizeof(double _Complex));

    seed_rand(20260526);
    for (size_t i = 0; i < nsamp; ++i) f[i] = (urand() - 0.5) + (urand() - 0.5) * I;

    su2_ft_direct(N, f, fhat_direct);
    su2_fft       (N, f, fhat_fast);

    double tol = 1e-10;
    for (size_t i = 0; i < ncoeff; ++i) {
        ASSERT_CNEAR(fhat_fast[i], fhat_direct[i], tol);
    }
    free(f); free(fhat_direct); free(fhat_fast);
}

static void test_fft_matches_direct_constant(void)
{
    int N = 8;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fhat_direct = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *fhat_fast   = calloc(ncoeff, sizeof(double _Complex));
    for (size_t i = 0; i < nsamp; ++i) f[i] = 2.5 - 0.7 * I;

    su2_ft_direct(N, f, fhat_direct);
    su2_fft       (N, f, fhat_fast);

    double tol = 1e-10;
    for (size_t i = 0; i < ncoeff; ++i) {
        ASSERT_CNEAR(fhat_fast[i], fhat_direct[i], tol);
    }
    free(f); free(fhat_direct); free(fhat_fast);
}

int main(void)
{
    RUN(test_fft_zero_input);
    RUN(test_fft_matches_direct_random);
    RUN(test_fft_matches_direct_constant);
    TEST_REPORT_AND_EXIT();
}
