/* test_resolved.c -- Strict correctness tests for the resolved-grid FFT
 * on SU(2) (bead `su2fft-0t1`).
 *
 * What bead 0t1 delivers
 * ----------------------
 * The closed-N phi/psi grid used by `su2_fft` / `su2_fft_gl` resolves only
 * N-1 distinct modes per axis even though the SU(2) bandlimit at degree N
 * needs 2N-1 modes per axis.  Modes with |n| > (N-1)/2 alias.  Combined
 * with Gauss-Legendre theta nodes this leaves the phi/psi aliasing as the
 * sole remaining quadrature defect (notes/0t1_resolved_grid_design.md §1).
 *
 * The resolved variant uses the OPEN P-point uniform grid with P = 2N-1
 * in phi and psi, and N-point Gauss-Legendre nodes in theta.  Forward
 * analysis becomes an exact composite quadrature for bandlimited inputs;
 * the spectrum roundtrip forward(inverse(fhat)) = fhat holds to working
 * precision (notes/0t1_resolved_grid_design.md §2, §4, §5).
 *
 * Tests below (per design brief §6)
 * ---------------------------------
 *   1. test_resolved_zero_forward          -- forward(0) = 0      < 1e-14
 *   2. test_resolved_zero_inverse          -- inverse(0) = 0      < 1e-14
 *   3. test_resolved_constant_forward      -- forward(A) = A * delta_{l=0}
 *                                                                 < 1e-12
 *   4. test_resolved_inv_delta_l1_m0_n0    -- inverse(fhat=delta@(1,0,0))
 *                                              = 3 cos(theta_k)   < 1e-12
 *   5. test_resolved_fft_matches_direct_random -- fast vs direct  < 1e-10
 *      (N = 5 and N = 6; two N to catch parity-dependent bugs)
 *   6. test_resolved_spectrum_roundtrip    -- HEADLINE.  Random fhat;
 *      forward(inverse(fhat)) recovers fhat to < 1e-12 relative at
 *      N = 4, 8, 16.  Per-N achieved tolerance printed.
 *   7. test_resolved_sample_roundtrip      -- f = inverse(fhat);
 *      h = inverse(forward(f)); h ~ f to < 1e-12 relative.  N = 4, 8.
 *   8. test_resolved_linearity_forward     -- forward is linear   < 1e-13
 *   9. test_resolved_linearity_inverse     -- inverse is linear   < 1e-13
 *
 * All tolerances are STRICT.  No "documented floor" assertions are
 * permitted on the resolved path -- if a test fails, fix the math, not
 * the tolerance.
 */
#include "test_framework.h"
#include "su2.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void seed_rand(unsigned s) { srand(s); }
static double urand(void) { return (double)rand() / (double)RAND_MAX; }

/* Infinity norm of a complex vector. */
static double cabs_inf(const double _Complex *a, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double v = cabs(a[i]);
        if (v > m) m = v;
    }
    return m;
}

/* Infinity norm of (a - b). */
static double cabs_inf_diff(const double _Complex *a,
                            const double _Complex *b,
                            size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double v = cabs(a[i] - b[i]);
        if (v > m) m = v;
    }
    return m;
}

/* ---- 1. forward(0) == 0 ---- */
static void test_resolved_zero_forward(void)
{
    int N = 6;
    size_t nsamp  = su2_resolved_total_samples(N);
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = calloc(nsamp,  sizeof(double _Complex));
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));

    su2_fft_resolved(N, f, fhat);
    for (size_t i = 0; i < ncoeff; ++i) ASSERT_CNEAR(fhat[i], 0.0, 1e-14);

    free(f); free(fhat);
}

/* ---- 2. inverse(0) == 0 ---- */
static void test_resolved_zero_inverse(void)
{
    int N = 6;
    size_t nsamp  = su2_resolved_total_samples(N);
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *f    = calloc(nsamp,  sizeof(double _Complex));

    su2_fft_resolved_inv(N, fhat, f);
    for (size_t i = 0; i < nsamp; ++i) ASSERT_CNEAR(f[i], 0.0, 1e-14);

    free(f); free(fhat);
}

/* ---- 3. forward(constant A) -> only fhat(0)_{0,0} = A nonzero ----
 * On the resolved grid the analysis quadrature is exact, so a constant
 * f = A produces fhat(0)_{0,0} = A and zero in every other slot.  The
 * Riemann constant 1/(2 P^2) with P = 2N-1 is the only normalisation
 * change vs su2_fft_gl (notes/0t1_resolved_grid_design.md §5).
 */
static void test_resolved_constant_forward(void)
{
    int N = 8;
    size_t nsamp  = su2_resolved_total_samples(N);
    size_t ncoeff = su2_total_coeffs(N);

    const double _Complex As[] = { 1.0 + 0.0 * I, 2.5 - 0.7 * I };
    for (int idx = 0; idx < 2; ++idx) {
        double _Complex A = As[idx];
        double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));
        double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
        for (size_t i = 0; i < nsamp; ++i) f[i] = A;

        su2_fft_resolved(N, f, fhat);

        size_t dc = su2_coeff_offset(0) + su2_mn_index(0, 0, 0);
        ASSERT_CNEAR(fhat[dc], A, 1e-12);
        for (size_t i = 0; i < ncoeff; ++i) {
            if (i == dc) continue;
            ASSERT_CNEAR(fhat[i], 0.0, 1e-12);
        }
        free(f); free(fhat);
    }
}

/* ---- 4. inverse(delta@(l=1, m=0, n=0)) = 3 cos(theta_k) on GL nodes ----
 * Peter-Weyl synthesis with a single coefficient fhat(1)_{0,0} = 1 yields
 *   f(g) = (2*1+1) * exp(-i*(0*phi + 0*psi)) * P^1_{0,0}(cos theta)
 *        = 3 * cos(theta).
 * The resolved path uses GL theta nodes; the expected value at slice k is
 * 3 * x_k where x_k is the k-th GL node (= cos(theta_k)).
 */
static void test_resolved_inv_delta_l1_m0_n0(void)
{
    int N = 8;
    int P = su2_resolved_P(N);
    size_t nsamp  = su2_resolved_total_samples(N);
    size_t ncoeff = su2_total_coeffs(N);

    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));

    fhat[su2_coeff_offset(1) + su2_mn_index(1, 0, 0)] = 1.0 + 0.0 * I;
    su2_fft_resolved_inv(N, fhat, f);

    double *x_gl = malloc((size_t)N * sizeof(double));
    double *w_gl = malloc((size_t)N * sizeof(double));
    su2_gl_nodes_weights(N, x_gl, w_gl);

    for (int j1 = 0; j1 < P; ++j1) {
        for (int k = 0; k < N; ++k) {
            double theta_k = acos(x_gl[k]);
            double _Complex expected = 3.0 * cos(theta_k) + 0.0 * I;
            for (int j2 = 0; j2 < P; ++j2) {
                double _Complex got =
                    f[su2_resolved_sample_index(N, j1, k, j2)];
                ASSERT_CNEAR(got, expected, 1e-12);
            }
        }
    }
    free(x_gl); free(w_gl);
    free(fhat); free(f);
}

/* ---- 5. fast vs direct on random complex input ----
 * Two N values (one odd, one even) to catch parity-dependent bugs at
 * P = 2N-1.
 */
static void test_resolved_fft_matches_direct_random(void)
{
    const int Ns[] = { 5, 6 };
    seed_rand(20260526);

    for (int idx = 0; idx < 2; ++idx) {
        int N = Ns[idx];
        size_t nsamp  = su2_resolved_total_samples(N);
        size_t ncoeff = su2_total_coeffs(N);

        double _Complex *f           = malloc(nsamp  * sizeof(double _Complex));
        double _Complex *fhat_direct = calloc(ncoeff, sizeof(double _Complex));
        double _Complex *fhat_fast   = calloc(ncoeff, sizeof(double _Complex));

        for (size_t i = 0; i < nsamp; ++i)
            f[i] = (urand() - 0.5) + (urand() - 0.5) * I;

        su2_ft_direct_resolved(N, f, fhat_direct);
        su2_fft_resolved      (N, f, fhat_fast);

        double err = cabs_inf_diff(fhat_fast, fhat_direct, ncoeff);
        printf("    [resolved fast-vs-direct] N=%d max_abs_err=%.4e\n",
               N, err);
        fflush(stdout);

        const double tol = 1e-10;
        for (size_t i = 0; i < ncoeff; ++i) {
            ASSERT_CNEAR(fhat_fast[i], fhat_direct[i], tol);
        }

        free(f); free(fhat_direct); free(fhat_fast);
    }
}

/* ---- 6. HEADLINE: spectrum roundtrip ----
 * fhat random; f = inverse(fhat); fhat2 = forward(f); compare.
 *
 * On the resolved grid the forward analysis is exact for bandlimited
 * inputs, so this should reach floating-point noise relative to the input
 * amplitude.  Strict 1e-12 normalised infinity-norm tolerance.
 */
static void test_resolved_spectrum_roundtrip(void)
{
    const int Ns[] = { 4, 8, 16 };
    seed_rand(20260526);

    for (int idx = 0; idx < 3; ++idx) {
        int N = Ns[idx];
        size_t nsamp  = su2_resolved_total_samples(N);
        size_t ncoeff = su2_total_coeffs(N);

        double _Complex *fhat       = malloc(ncoeff * sizeof(double _Complex));
        double _Complex *fhat_round = calloc(ncoeff, sizeof(double _Complex));
        double _Complex *f          = malloc(nsamp  * sizeof(double _Complex));

        for (size_t i = 0; i < ncoeff; ++i)
            fhat[i] = (urand() - 0.5) + (urand() - 0.5) * I;

        su2_fft_resolved_inv(N, fhat,       f);
        su2_fft_resolved    (N, f,          fhat_round);

        double num = cabs_inf_diff(fhat_round, fhat, ncoeff);
        double den = cabs_inf(fhat, ncoeff);
        double rel = (den > 0.0) ? num / den : num;

        printf("  [resolved spectrum roundtrip] N=%-2d max_relerr=%.4e\n",
               N, rel);
        fflush(stdout);

        ASSERT_TRUE(rel < 1e-12);

        free(fhat); free(fhat_round); free(f);
    }
}

/* ---- 7. sample roundtrip on a bandlimited f ---- */
static void test_resolved_sample_roundtrip(void)
{
    const int Ns[] = { 4, 8 };
    seed_rand(20260526);

    for (int idx = 0; idx < 2; ++idx) {
        int N = Ns[idx];
        size_t nsamp  = su2_resolved_total_samples(N);
        size_t ncoeff = su2_total_coeffs(N);

        double _Complex *fhat   = malloc(ncoeff * sizeof(double _Complex));
        double _Complex *ghat   = calloc(ncoeff, sizeof(double _Complex));
        double _Complex *f      = malloc(nsamp  * sizeof(double _Complex));
        double _Complex *h      = malloc(nsamp  * sizeof(double _Complex));

        for (size_t i = 0; i < ncoeff; ++i)
            fhat[i] = (urand() - 0.5) + (urand() - 0.5) * I;

        /* Build bandlimited f, then roundtrip through forward/inverse. */
        su2_fft_resolved_inv(N, fhat, f);
        su2_fft_resolved    (N, f,    ghat);
        su2_fft_resolved_inv(N, ghat, h);

        double num = cabs_inf_diff(h, f, nsamp);
        double den = cabs_inf(f, nsamp);
        double rel = (den > 0.0) ? num / den : num;

        printf("    [resolved sample roundtrip] N=%-2d max_relerr=%.4e\n",
               N, rel);
        fflush(stdout);

        ASSERT_TRUE(rel < 1e-12);

        free(fhat); free(ghat); free(f); free(h);
    }
}

/* ---- 8. forward is linear ---- */
static void test_resolved_linearity_forward(void)
{
    int N = 6;
    size_t nsamp  = su2_resolved_total_samples(N);
    size_t ncoeff = su2_total_coeffs(N);

    double _Complex *f1     = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *f2     = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fab    = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fh1    = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *fh2    = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *fh_ab  = calloc(ncoeff, sizeof(double _Complex));

    seed_rand(20260526);
    for (size_t i = 0; i < nsamp; ++i) {
        f1[i] = (urand() - 0.5) + (urand() - 0.5) * I;
        f2[i] = (urand() - 0.5) + (urand() - 0.5) * I;
    }
    const double _Complex alpha =  2.0 + 1.5 * I;
    const double _Complex beta  = -0.7 + 0.4 * I;
    for (size_t i = 0; i < nsamp; ++i)
        fab[i] = alpha * f1[i] + beta * f2[i];

    su2_fft_resolved(N, f1,  fh1);
    su2_fft_resolved(N, f2,  fh2);
    su2_fft_resolved(N, fab, fh_ab);

    for (size_t i = 0; i < ncoeff; ++i) {
        double _Complex combined = alpha * fh1[i] + beta * fh2[i];
        ASSERT_CNEAR(fh_ab[i], combined, 1e-13);
    }

    free(f1); free(f2); free(fab);
    free(fh1); free(fh2); free(fh_ab);
}

/* ---- 9. inverse is linear ---- */
static void test_resolved_linearity_inverse(void)
{
    int N = 6;
    size_t nsamp  = su2_resolved_total_samples(N);
    size_t ncoeff = su2_total_coeffs(N);

    double _Complex *fh1    = malloc(ncoeff * sizeof(double _Complex));
    double _Complex *fh2    = malloc(ncoeff * sizeof(double _Complex));
    double _Complex *fh_ab  = malloc(ncoeff * sizeof(double _Complex));
    double _Complex *f1     = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *f2     = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *f_ab   = malloc(nsamp  * sizeof(double _Complex));

    seed_rand(20260526);
    for (size_t i = 0; i < ncoeff; ++i) {
        fh1[i] = (urand() - 0.5) + (urand() - 0.5) * I;
        fh2[i] = (urand() - 0.5) + (urand() - 0.5) * I;
    }
    const double _Complex alpha =  2.0 + 1.5 * I;
    const double _Complex beta  = -0.7 + 0.4 * I;
    for (size_t i = 0; i < ncoeff; ++i)
        fh_ab[i] = alpha * fh1[i] + beta * fh2[i];

    su2_fft_resolved_inv(N, fh1,   f1);
    su2_fft_resolved_inv(N, fh2,   f2);
    su2_fft_resolved_inv(N, fh_ab, f_ab);

    for (size_t i = 0; i < nsamp; ++i) {
        double _Complex combined = alpha * f1[i] + beta * f2[i];
        ASSERT_CNEAR(f_ab[i], combined, 1e-13);
    }

    free(fh1); free(fh2); free(fh_ab);
    free(f1); free(f2); free(f_ab);
}

int main(void)
{
    RUN(test_resolved_zero_forward);
    RUN(test_resolved_zero_inverse);
    RUN(test_resolved_constant_forward);
    RUN(test_resolved_inv_delta_l1_m0_n0);
    RUN(test_resolved_fft_matches_direct_random);
    RUN(test_resolved_spectrum_roundtrip);
    RUN(test_resolved_sample_roundtrip);
    RUN(test_resolved_linearity_forward);
    RUN(test_resolved_linearity_inverse);
    TEST_REPORT_AND_EXIT();
}
