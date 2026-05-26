/* test_grid.c -- Red/green test for the Euler-angle grid (paper line 684).
 *
 * Ground truth, transcribed from paper.tex line 684 verbatim:
 *     phi[j1]   = -pi + j1 * 2pi/(N-1)
 *     psi[j2]   = -pi + j2 * 2pi/(N-1)
 *     theta[k]  =  0  + k  *  pi/(N-1)
 * N must be >= 2 (so N-1 != 0).  Endpoints +pi and pi appear at the top index.
 */
#include "test_framework.h"
#include "su2.h"

#include <math.h>
#include <stdlib.h>

static void test_grid_endpoints_N5(void)
{
    int N = 5;
    double *phi   = su2_grid_phi(N);
    double *theta = su2_grid_theta(N);
    double *psi   = su2_grid_psi(N);
    ASSERT_TRUE(phi && theta && psi);

    ASSERT_NEAR(phi[0],     -M_PI, 1e-15);
    ASSERT_NEAR(phi[N - 1],  M_PI, 1e-15);
    ASSERT_NEAR(psi[0],     -M_PI, 1e-15);
    ASSERT_NEAR(psi[N - 1],  M_PI, 1e-15);
    ASSERT_NEAR(theta[0],   0.0,   1e-15);
    ASSERT_NEAR(theta[N - 1], M_PI, 1e-15);

    free(phi); free(theta); free(psi);
}

static void test_grid_spacing_uniform(void)
{
    int N = 9;
    double *phi   = su2_grid_phi(N);
    double *theta = su2_grid_theta(N);

    double dphi   = 2.0 * M_PI / (N - 1);
    double dtheta =        M_PI / (N - 1);

    for (int i = 1; i < N; ++i) {
        ASSERT_NEAR(phi[i]   - phi[i - 1],   dphi,   1e-14);
        ASSERT_NEAR(theta[i] - theta[i - 1], dtheta, 1e-14);
    }

    free(phi); free(theta);
}

static void test_grid_indices_helpers(void)
{
    /* Sample-array layout: f[j1*N*N + k*N + j2]. */
    int N = 4;
    ASSERT_TRUE(su2_sample_index(N, 0, 0, 0) == 0);
    ASSERT_TRUE(su2_sample_index(N, 0, 0, 1) == 1);
    ASSERT_TRUE(su2_sample_index(N, 0, 1, 0) == 4);
    ASSERT_TRUE(su2_sample_index(N, 1, 0, 0) == 16);
    ASSERT_TRUE(su2_sample_index(N, 3, 3, 3) == 63);

    /* fhat layout per l: (2l+1) x (2l+1) row-major in (m+l, n+l). */
    ASSERT_TRUE(su2_mn_index(0, 0, 0) == 0);
    ASSERT_TRUE(su2_mn_index(1, -1, -1) == 0);
    ASSERT_TRUE(su2_mn_index(1,  0,  0) == 4);
    ASSERT_TRUE(su2_mn_index(1,  1,  1) == 8);

    /* Total coefficient count: sum_{l=0..N-1} (2l+1)^2 . */
    /* N=1: 1.  N=2: 1+9=10.  N=3: 1+9+25=35.  N=4: 1+9+25+49=84. */
    ASSERT_TRUE(su2_total_coeffs(1) == 1);
    ASSERT_TRUE(su2_total_coeffs(2) == 10);
    ASSERT_TRUE(su2_total_coeffs(3) == 35);
    ASSERT_TRUE(su2_total_coeffs(4) == 84);

    /* Offset of fhat[l]: cumulative count of degrees below l. */
    ASSERT_TRUE(su2_coeff_offset(0) == 0);
    ASSERT_TRUE(su2_coeff_offset(1) == 1);
    ASSERT_TRUE(su2_coeff_offset(2) == 10);
    ASSERT_TRUE(su2_coeff_offset(3) == 35);
}

int main(void)
{
    RUN(test_grid_endpoints_N5);
    RUN(test_grid_spacing_uniform);
    RUN(test_grid_indices_helpers);
    TEST_REPORT_AND_EXIT();
}
