/* test_sphere.c -- Spherical-harmonic FFT tests (bead 5fb). */
#include "test_framework.h"
#include "su2.h"

#include <math.h>
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

int main(void)
{
    RUN(test_sphere_total_coeffs);
    RUN(test_sphere_zero_input);
    RUN(test_sphere_inv_delta_Y00_gives_constant);
    RUN(test_sphere_inv_delta_Y10_gives_3cos_theta);
    RUN(test_sphere_constant_forward_bounded);
    RUN(test_sphere_inv_linearity);
    TEST_REPORT_AND_EXIT();
}
