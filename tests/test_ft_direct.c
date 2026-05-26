/* test_ft_direct.c -- Red/green test for the O(N^6) reference Fourier transform.
 *
 * Ground truth: paper.tex line 1316 (the discrete-sum definition that the
 * direct algorithm is required to compute exactly):
 *
 *   fhat(l)_{m,n} = (sin(theta_k) * dphi * dtheta * dpsi / (8 pi^2)) *
 *                   sum_{j1,k,j2} f(g_{j1,k,j2}) * conj(t^l_{n,m}(g_{j1,k,j2}))
 *
 * Tests:
 *   1. Zero input -> all-zero output (exact).
 *   2. Constant input f=1 -> fhat(0)[0,0] approx 1 within Riemann error;
 *      other coefficients small.
 *   3. Sample of f(g) = t^1_{0,0}(g) = cos(theta) -> fhat(1)[0,0] approx 1/3
 *      and other coefficients small (by Peter-Weyl orthogonality,
 *      ||t^l_{n,m}||^2 = 1/(2l+1), paper line 564).
 *
 * Tolerances are loose to absorb O(1/N^2) Riemann error.  The tight check is
 * left to test_compare (FFT vs direct FT, to machine precision).
 */
#include "test_framework.h"
#include "su2.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void test_ft_direct_zero_input(void)
{
    int N = 4;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = calloc(nsamp,  sizeof(double _Complex));
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
    su2_ft_direct(N, f, fhat);
    for (size_t i = 0; i < ncoeff; ++i) ASSERT_CNEAR(fhat[i], 0.0, 1e-14);
    free(f); free(fhat);
}

static void test_ft_direct_constant(void)
{
    /* For the closed grid (endpoints -pi and +pi both included), the discrete
     * formula gives fhat(0,0,0) = (N/(N-1))^2 * 2 / 2 = (N/(N-1))^2 (up to
     * a small trapezoidal error in the theta integral).  This factor goes to
     * 1 as N grows; it is intrinsic to the paper's grid choice. */
    int N = 16;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fhat = calloc(ncoeff,  sizeof(double _Complex));
    for (size_t i = 0; i < nsamp; ++i) f[i] = 1.0 + 0.0*I;

    su2_ft_direct(N, f, fhat);

    double expected_ratio = (double)N * N / ((double)(N - 1) * (N - 1));
    double tol = 5e-3;
    ASSERT_CNEAR(fhat[su2_coeff_offset(0) + su2_mn_index(0, 0, 0)],
                 expected_ratio, tol);

    /* Higher-degree coefficients should be small (no other Fourier content). */
    for (int l = 1; l <= 3; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int n = -l; n <= l; ++n) {
                size_t idx = su2_coeff_offset(l) + su2_mn_index(l, m, n);
                ASSERT_CNEAR(fhat[idx], 0.0, 5e-2);
            }
        }
    }
    free(f); free(fhat);
}

static void test_ft_direct_single_point(void)
{
    /* Hard discrete identity: a delta-spike input gives fhat(l)_{m,n}
     * exactly equal to  norm * conj(t^l_{n,m}(g_*)) * sin(theta_{k*})
     * where g_* is the spike location.  No quadrature error.        */
    int N = 5;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = calloc(nsamp,  sizeof(double _Complex));
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));

    int j1s = 1, ks = 2, j2s = 3;
    f[su2_sample_index(N, j1s, ks, j2s)] = 1.0 + 0.0*I;

    su2_ft_direct(N, f, fhat);

    double *phi   = su2_grid_phi(N);
    double *theta = su2_grid_theta(N);
    double *psi   = su2_grid_psi(N);
    double dphi   = 2.0 * M_PI / (N - 1);
    double dtheta =       M_PI / (N - 1);
    double dpsi   = 2.0 * M_PI / (N - 1);
    double norm   = dphi * dtheta * dpsi / (8.0 * M_PI * M_PI);

    for (int l = 0; l < 3; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int n = -l; n <= l; ++n) {
                double _Complex tlnm = su2_wigner_d(l, n, m, theta[ks])
                                       * cexp(-I * (n * phi[j1s] + m * psi[j2s]));
                double _Complex want = norm * conj(tlnm) * sin(theta[ks]);
                double _Complex got  = fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)];
                ASSERT_CNEAR(got, want, 1e-13);
            }
        }
    }
    free(phi); free(theta); free(psi);
    free(f); free(fhat);
}

static void test_ft_direct_single_mode(void)
{
    /* f(g) = t^1_{0,0}(g) = cos(theta).  Expected fhat(1)[0,0] = 1/3. */
    int N = 24;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fhat = calloc(ncoeff,  sizeof(double _Complex));

    double *theta = su2_grid_theta(N);
    for (int j1 = 0; j1 < N; ++j1) {
        for (int k = 0; k < N; ++k) {
            for (int j2 = 0; j2 < N; ++j2) {
                f[su2_sample_index(N, j1, k, j2)] = cos(theta[k]);
            }
        }
    }
    free(theta);

    su2_ft_direct(N, f, fhat);

    double tol = 5e-2;
    ASSERT_CNEAR(fhat[su2_coeff_offset(1) + su2_mn_index(1, 0, 0)],
                 1.0 / 3.0, tol);

    /* Diagnostic: spot-check a few "should be zero" coefficients. */
    ASSERT_CNEAR(fhat[su2_coeff_offset(0) + su2_mn_index(0, 0, 0)], 0.0, tol);
    ASSERT_CNEAR(fhat[su2_coeff_offset(2) + su2_mn_index(2, 0, 0)], 0.0, tol);
    ASSERT_CNEAR(fhat[su2_coeff_offset(1) + su2_mn_index(1, 1, 1)], 0.0, tol);

    free(f); free(fhat);
}

int main(void)
{
    RUN(test_ft_direct_zero_input);
    RUN(test_ft_direct_constant);
    RUN(test_ft_direct_single_mode);
    RUN(test_ft_direct_single_point);
    TEST_REPORT_AND_EXIT();
}
