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

static void test_fft_real_matches_complex(void)
{
    /* su2_fft_real(N, f, .) must equal su2_fft(N, f + 0i, .) on the full
     * coefficient array.  It only differs by exploiting the Hermitian dual
     * symmetry fhat(l)_{m,n} = (-1)^{m-n} conj(fhat(l)_{-m,-n}) (bead 4v7);
     * the produced spectrum is byte-for-byte the same up to FP roundoff. */
    seed_rand(20260528);
    int Ns[] = {5, 6, 8};
    for (size_t t = 0; t < sizeof(Ns) / sizeof(Ns[0]); ++t) {
        int N = Ns[t];
        size_t nsamp  = (size_t)N * N * N;
        size_t ncoeff = su2_total_coeffs(N);

        double          *f         = malloc(nsamp  * sizeof(double));
        double _Complex *fc        = malloc(nsamp  * sizeof(double _Complex));
        double _Complex *fhat_real = calloc(ncoeff, sizeof(double _Complex));
        double _Complex *fhat_full = calloc(ncoeff, sizeof(double _Complex));

        for (size_t i = 0; i < nsamp; ++i) {
            f[i]  = urand() - 0.5;
            fc[i] = f[i] + 0.0 * I;
        }

        su2_fft_real(N, f,  fhat_real);
        su2_fft     (N, fc, fhat_full);

        double max_err = 0.0;
        for (size_t i = 0; i < ncoeff; ++i) {
            double d = cabs(fhat_real[i] - fhat_full[i]);
            if (d > max_err) max_err = d;
            ASSERT_CNEAR(fhat_real[i], fhat_full[i], 1e-13);
        }
        fprintf(stderr, "[N=%d max_err=%.2e] ", N, max_err);

        /* fhat(l)_{0,0} is self-paired, hence real: |imag| ~ 0. */
        for (int l = 0; l < N && l < 3; ++l) {
            double _Complex c00 = fhat_real[su2_coeff_offset(l) + su2_mn_index(l, 0, 0)];
            ASSERT_NEAR(cimag(c00), 0.0, 1e-13);
        }

        free(f); free(fc); free(fhat_real); free(fhat_full);
    }
}

int main(void)
{
    RUN(test_fft_zero_input);
    RUN(test_fft_matches_direct_random);
    RUN(test_fft_matches_direct_constant);
    RUN(test_fft_real_matches_complex);
    TEST_REPORT_AND_EXIT();
}
