/* test_sphere.c -- Spherical-harmonic FFT tests (bead 5fb).
 *
 * Resolved-grid sphere FFT tests (bead 9qk) appended below the closed
 * variants; strict 1e-12 tolerances on the resolved path, mirroring
 * tests/test_resolved.c. */
#include "test_framework.h"
#include "su2.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double urand(void) { return (double)rand() / (double)RAND_MAX; }

static void test_sphere_total_coeffs(void)
{
    /* sum_{l=0..N-1}(2l+1) = N^2. */
    for (int N = 1; N <= 8; ++N) {
        size_t expected = (size_t)N * (size_t)N;
        ASSERT_NEAR((double)su2_sphere_total_coeffs(N), (double)expected, 0.5);
    }
}

static void test_sphere_zero_input(void)
{
    int N = 6;
    size_t ns = (size_t)N * (size_t)N;
    size_t nc = su2_sphere_total_coeffs(N);
    double _Complex *f_sph = calloc(ns, sizeof(double _Complex));
    double _Complex *fh_sph = malloc(nc * sizeof(double _Complex));
    su2_fft_sphere(N, f_sph, fh_sph);
    for (size_t i = 0; i < nc; ++i) ASSERT_NEAR(cabs(fh_sph[i]), 0.0, 1e-13);

    double _Complex *fhi_sph = calloc(nc, sizeof(double _Complex));
    double _Complex *f_back = malloc(ns * sizeof(double _Complex));
    su2_fft_sphere_inv(N, fhi_sph, f_back);
    for (size_t i = 0; i < ns; ++i) ASSERT_NEAR(cabs(f_back[i]), 0.0, 1e-13);

    free(f_sph); free(fh_sph); free(fhi_sph); free(f_back);
}

static void test_sphere_inv_delta_Y00_gives_constant(void)
{
    /* fhat_sph(0, 0) = 1; rest zero. Synthesis: f_sph = 1 everywhere. */
    int N = 8;
    size_t ns = (size_t)N * (size_t)N;
    size_t nc = su2_sphere_total_coeffs(N);
    double _Complex *fhat = calloc(nc, sizeof(double _Complex));
    double _Complex *f = malloc(ns * sizeof(double _Complex));
    fhat[0] = 1.0 + 0.0*I;  /* (l=0, n=0) is index 0 */
    su2_fft_sphere_inv(N, fhat, f);
    for (size_t i = 0; i < ns; ++i) {
        ASSERT_CNEAR(f[i], 1.0 + 0.0*I, 1e-13);
    }
    free(fhat); free(f);
}

static void test_sphere_inv_delta_Y10_gives_3cos_theta(void)
{
    /* fhat_sph(1, 0) = 1; rest zero. Synthesis: f_sph(theta, phi) = 3*cos(theta). */
    int N = 8;
    size_t ns = (size_t)N * (size_t)N;
    size_t nc = su2_sphere_total_coeffs(N);
    double _Complex *fhat = calloc(nc, sizeof(double _Complex));
    double _Complex *f = malloc(ns * sizeof(double _Complex));
    /* Index of (l=1, n=0): offset = sum_{l'<1}(2l'+1) = 1; then n+l = 0+1 = 1.
     * So index = 1 + 1 = 2. */
    fhat[2] = 1.0 + 0.0*I;
    su2_fft_sphere_inv(N, fhat, f);

    double *theta = su2_grid_theta(N);
    double max_err = 0.0;
    for (int j1 = 0; j1 < N; ++j1) {
        for (int k = 0; k < N; ++k) {
            double _Complex expected = 3.0 * cos(theta[k]) + 0.0*I;
            double _Complex got = f[(size_t)j1 * (size_t)N + (size_t)k];
            double e = cabs(got - expected);
            if (e > max_err) max_err = e;
        }
    }
    ASSERT_NEAR(max_err < 1e-12 ? 1.0 : 0.0, 1.0, 1e-15);
    free(theta); free(fhat); free(f);
}

static void test_sphere_constant_forward_bounded(void)
{
    /* f_sph = 1 -> fhat_sph(0,0) is approximately a constant, bounded.
     * Documents the closed-grid Riemann tolerance (bead su2fft-0t1). */
    int N = 8;
    size_t ns = (size_t)N * (size_t)N;
    size_t nc = su2_sphere_total_coeffs(N);
    double _Complex *f = malloc(ns * sizeof(double _Complex));
    double _Complex *fhat = malloc(nc * sizeof(double _Complex));
    for (size_t i = 0; i < ns; ++i) f[i] = 1.0 + 0.0*I;
    su2_fft_sphere(N, f, fhat);
    /* fhat[0] should be near (N/(N-1))^2 = 1.31 at N=8 under closed-grid forward. */
    /* Bound it loosely; the test documents that the result is consistent
     * with the underlying SU(2) FFT behaviour. */
    ASSERT_NEAR(cabs(fhat[0]) < 2.0 && cabs(fhat[0]) > 0.5 ? 1.0 : 0.0, 1.0, 1e-15);
    free(f); free(fhat);
}

static void test_sphere_inv_linearity(void)
{
    int N = 6;
    size_t ns = (size_t)N * (size_t)N;
    size_t nc = su2_sphere_total_coeffs(N);
    double _Complex *fh1 = malloc(nc * sizeof(double _Complex));
    double _Complex *fh2 = malloc(nc * sizeof(double _Complex));
    double _Complex *fhc = malloc(nc * sizeof(double _Complex));
    double _Complex *f1 = malloc(ns * sizeof(double _Complex));
    double _Complex *f2 = malloc(ns * sizeof(double _Complex));
    double _Complex *fc = malloc(ns * sizeof(double _Complex));
    srand(20260526);
    for (size_t i = 0; i < nc; ++i) {
        fh1[i] = (urand()-0.5) + (urand()-0.5)*I;
        fh2[i] = (urand()-0.5) + (urand()-0.5)*I;
    }
    double _Complex a = 2.0 + 1.5*I;
    double _Complex b = -0.7 + 0.4*I;
    for (size_t i = 0; i < nc; ++i) fhc[i] = a*fh1[i] + b*fh2[i];
    su2_fft_sphere_inv(N, fh1, f1);
    su2_fft_sphere_inv(N, fh2, f2);
    su2_fft_sphere_inv(N, fhc, fc);
    double max_err = 0;
    for (size_t i = 0; i < ns; ++i) {
        double _Complex want = a*f1[i] + b*f2[i];
        double e = cabs(fc[i] - want);
        if (e > max_err) max_err = e;
    }
    ASSERT_NEAR(max_err < 1e-12 ? 1.0 : 0.0, 1.0, 1e-15);
    free(fh1); free(fh2); free(fhc); free(f1); free(f2); free(fc);
}

/* =====================================================================
 * Resolved-grid sphere FFT tests (bead 9qk).  Strict 1e-12 tolerances --
 * the resolved path has no closed-grid Riemann floor.
 * ===================================================================== */

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

/* ---- 1. total samples == (2N-1)*N for N = 2..16 ---- */
static void test_sphere_resolved_total_samples(void)
{
    for (int N = 2; N <= 16; ++N) {
        size_t expected = (size_t)(2 * N - 1) * (size_t)N;
        ASSERT_TRUE(su2_sphere_resolved_total_samples(N) == expected);
    }
}

/* ---- 2. forward(0) == 0 and inverse(0) == 0 ---- */
static void test_sphere_resolved_zero(void)
{
    int N = 6;
    size_t ns = su2_sphere_resolved_total_samples(N);
    size_t nc = su2_sphere_total_coeffs(N);

    double _Complex *f_sph  = calloc(ns, sizeof(double _Complex));
    double _Complex *fh_sph = malloc(nc * sizeof(double _Complex));
    su2_fft_sphere_resolved(N, f_sph, fh_sph);
    for (size_t i = 0; i < nc; ++i) ASSERT_CNEAR(fh_sph[i], 0.0, 1e-14);

    double _Complex *fhi_sph = calloc(nc, sizeof(double _Complex));
    double _Complex *f_back  = malloc(ns * sizeof(double _Complex));
    su2_fft_sphere_inv_resolved(N, fhi_sph, f_back);
    for (size_t i = 0; i < ns; ++i) ASSERT_CNEAR(f_back[i], 0.0, 1e-14);

    free(f_sph); free(fh_sph); free(fhi_sph); free(f_back);
}

/* ---- 3. forward(constant A) -> fhat_sph[0] = A, rest 0 ----
 * On the resolved grid the analysis quadrature is exact for bandlimited
 * inputs (notes/0t1_resolved_grid_design.md §5).  Constant f extends to
 * the SU(2) constant, whose spectrum is the single coefficient
 * fhat(0)_{0,0} = A; the sphere wrapper extracts the m=0 row, so the
 * only nonzero entry of fhat_sph is the (l=0, n=0) slot at index 0. */
static void test_sphere_resolved_constant_forward(void)
{
    int N = 8;
    size_t ns = su2_sphere_resolved_total_samples(N);
    size_t nc = su2_sphere_total_coeffs(N);

    double _Complex A = 2.5 - 0.7 * I;
    double _Complex *f    = malloc(ns * sizeof(double _Complex));
    double _Complex *fhat = calloc(nc, sizeof(double _Complex));
    for (size_t i = 0; i < ns; ++i) f[i] = A;

    su2_fft_sphere_resolved(N, f, fhat);

    ASSERT_CNEAR(fhat[0], A, 1e-12);
    for (size_t i = 1; i < nc; ++i) ASSERT_CNEAR(fhat[i], 0.0, 1e-12);

    free(f); free(fhat);
}

/* ---- 4. inverse(delta @ (l=0, n=0)) = 1 everywhere ---- */
static void test_sphere_resolved_inv_Y00_gives_constant(void)
{
    int N = 8;
    size_t ns = su2_sphere_resolved_total_samples(N);
    size_t nc = su2_sphere_total_coeffs(N);

    double _Complex *fhat = calloc(nc, sizeof(double _Complex));
    double _Complex *f    = malloc(ns * sizeof(double _Complex));
    fhat[0] = 1.0 + 0.0 * I;
    su2_fft_sphere_inv_resolved(N, fhat, f);
    for (size_t i = 0; i < ns; ++i) ASSERT_CNEAR(f[i], 1.0 + 0.0 * I, 1e-13);
    free(fhat); free(f);
}

/* ---- 5. inverse(delta @ (l=1, n=0)) = 3 cos(theta_k) on GL nodes ----
 * (l=1, n=0) flat index: sum_{l'<1}(2l'+1) = 1, plus (n+l) = 1 = 2.
 * Peter-Weyl synthesis: f = 3 * P^1_{0,0}(cos theta) = 3 cos theta. */
static void test_sphere_resolved_inv_Y10_gives_3cos_theta(void)
{
    int N = 8;
    int P = su2_resolved_P(N);
    size_t ns = su2_sphere_resolved_total_samples(N);
    size_t nc = su2_sphere_total_coeffs(N);

    double _Complex *fhat = calloc(nc, sizeof(double _Complex));
    double _Complex *f    = malloc(ns * sizeof(double _Complex));
    fhat[2] = 1.0 + 0.0 * I;
    su2_fft_sphere_inv_resolved(N, fhat, f);

    double *x_gl = malloc((size_t)N * sizeof(double));
    double *w_gl = malloc((size_t)N * sizeof(double));
    su2_gl_nodes_weights(N, x_gl, w_gl);

    for (int j1 = 0; j1 < P; ++j1) {
        for (int k = 0; k < N; ++k) {
            double theta_k = acos(x_gl[k]);
            double _Complex expected = 3.0 * cos(theta_k) + 0.0 * I;
            double _Complex got = f[(size_t)j1 * (size_t)N + (size_t)k];
            ASSERT_CNEAR(got, expected, 1e-12);
        }
    }
    free(x_gl); free(w_gl);
    free(fhat); free(f);
}

/* ---- 6. HEADLINE: spectrum roundtrip ----
 * Random fhat_sph; f = inverse(fhat); fhat2 = forward(f); compare.
 * Inherits exact roundtrip from `su2_fft_resolved` (bead 0t1). */
static void test_sphere_resolved_spectrum_roundtrip(void)
{
    const int Ns[] = { 4, 8, 16 };
    srand(20260526);

    for (int idx = 0; idx < 3; ++idx) {
        int N = Ns[idx];
        size_t ns = su2_sphere_resolved_total_samples(N);
        size_t nc = su2_sphere_total_coeffs(N);

        double _Complex *fhat       = malloc(nc * sizeof(double _Complex));
        double _Complex *fhat_round = calloc(nc, sizeof(double _Complex));
        double _Complex *f          = malloc(ns * sizeof(double _Complex));

        for (size_t i = 0; i < nc; ++i)
            fhat[i] = (urand() - 0.5) + (urand() - 0.5) * I;

        su2_fft_sphere_inv_resolved(N, fhat,       f);
        su2_fft_sphere_resolved    (N, f,          fhat_round);

        double num = cabs_inf_diff(fhat_round, fhat, nc);
        double den = cabs_inf(fhat, nc);
        double rel = (den > 0.0) ? num / den : num;

        printf("  [sphere resolved spectrum roundtrip] N=%-2d max_relerr=%.4e\n",
               N, rel);
        fflush(stdout);

        ASSERT_TRUE(rel < 1e-12);

        free(fhat); free(fhat_round); free(f);
    }
}

int main(void)
{
    RUN(test_sphere_total_coeffs);
    RUN(test_sphere_zero_input);
    RUN(test_sphere_inv_delta_Y00_gives_constant);
    RUN(test_sphere_inv_delta_Y10_gives_3cos_theta);
    RUN(test_sphere_constant_forward_bounded);
    RUN(test_sphere_inv_linearity);
    RUN(test_sphere_resolved_total_samples);
    RUN(test_sphere_resolved_zero);
    RUN(test_sphere_resolved_constant_forward);
    RUN(test_sphere_resolved_inv_Y00_gives_constant);
    RUN(test_sphere_resolved_inv_Y10_gives_3cos_theta);
    RUN(test_sphere_resolved_spectrum_roundtrip);
    TEST_REPORT_AND_EXIT();
}
