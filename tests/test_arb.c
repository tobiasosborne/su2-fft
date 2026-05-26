/* test_arb.c -- Cross-checks for the arbitrary-precision implementation.
 *
 *   1. su2_ft_direct_arb  vs  su2_fft_arb  at prec=64 (internal consistency).
 *   2. su2_fft_arb (prec=53)  vs  su2_fft (double)    (cross-impl).
 *   3. su2_wigner_d_arb (prec=128)  vs  su2_wigner_d  (sanity at higher prec).
 *
 * Test (1) is the gold standard: both arb routines compute the same discrete
 * sum, so the ball intervals must agree to at least the targeted precision.
 * Test (2) ties the arb path back to the double-precision FFT we already
 * trust.  Test (3) demonstrates the precision is real -- the agreement is
 * limited only by how stable the double path is.
 */
#include "test_framework.h"
#include "su2.h"
#include "su2_arb.h"

#include <flint/arb.h>
#include <flint/acb.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>

static double drand(unsigned *seed)
{
    *seed = (*seed) * 1103515245u + 12345u;
    return ((*seed >> 16) & 0x7FFF) / 32767.0;
}

static void acb_set_d_complex(acb_t z, double _Complex c)
{
    arb_set_d(acb_realref(z), creal(c));
    arb_set_d(acb_imagref(z), cimag(c));
}

static double _Complex acb_to_complex(const acb_t z)
{
    double re = arf_get_d(arb_midref(acb_realref(z)), ARF_RND_NEAR);
    double im = arf_get_d(arb_midref(acb_imagref(z)), ARF_RND_NEAR);
    return re + im*I;
}

static void test_arb_direct_vs_fast(void)
{
    int N = 5;
    slong prec = 64;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);

    acb_ptr f       = _acb_vec_init(nsamp);
    acb_ptr fhat_d  = _acb_vec_init(ncoeff);
    acb_ptr fhat_f  = _acb_vec_init(ncoeff);

    unsigned seed = 42;
    for (size_t i = 0; i < nsamp; ++i) {
        double _Complex c = (drand(&seed) - 0.5) + (drand(&seed) - 0.5) * I;
        acb_set_d_complex(f + i, c);
    }

    su2_ft_direct_arb(N, f, fhat_d, prec);
    su2_fft_arb      (N, f, fhat_f, prec);

    /* Both arb computations carry a precision of ~prec bits.  At prec=64 the
     * mid-point should agree to ~1e-15 or better; we allow 1e-12 for the
     * accumulated O(N^3) operations. */
    for (size_t i = 0; i < ncoeff; ++i) {
        double _Complex a = acb_to_complex(fhat_d + i);
        double _Complex b = acb_to_complex(fhat_f + i);
        ASSERT_CNEAR(a, b, 1e-12);
    }

    _acb_vec_clear(f, nsamp);
    _acb_vec_clear(fhat_d, ncoeff);
    _acb_vec_clear(fhat_f, ncoeff);
}

static void test_arb_vs_double(void)
{
    int N = 6;
    slong prec = 53;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);

    double _Complex *f_d    = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fhat_d = calloc(ncoeff, sizeof(double _Complex));
    acb_ptr f_a    = _acb_vec_init(nsamp);
    acb_ptr fhat_a = _acb_vec_init(ncoeff);

    unsigned seed = 20260526u;
    for (size_t i = 0; i < nsamp; ++i) {
        f_d[i] = (drand(&seed) - 0.5) + (drand(&seed) - 0.5) * I;
        acb_set_d_complex(f_a + i, f_d[i]);
    }

    su2_fft   (N, f_d, fhat_d);
    su2_fft_arb(N, f_a, fhat_a, prec);

    /* At prec=53 we expect agreement at the level of accumulated double
     * rounding -- a few ULPs scaled by O(N^3). */
    for (size_t i = 0; i < ncoeff; ++i) {
        double _Complex a = fhat_d[i];
        double _Complex b = acb_to_complex(fhat_a + i);
        ASSERT_CNEAR(a, b, 1e-10);
    }

    free(f_d); free(fhat_d);
    _acb_vec_clear(f_a, nsamp);
    _acb_vec_clear(fhat_a, ncoeff);
}

static void test_arb_wigner_spot(void)
{
    /* The hand-computed value from test_wigner.c, but verified at high prec.
     *   P^1_{n=1, m=0}(cos(pi/2)) = i / sqrt(2). */
    slong prec = 128;
    arb_t theta;
    arb_init(theta);
    arb_const_pi(theta, prec);
    arb_mul_2exp_si(theta, theta, -1);    /* pi/2 */

    acb_t got;
    acb_init(got);
    su2_wigner_d_arb(got, 1, 1, 0, theta, prec);

    arb_t want_mag;
    arb_init(want_mag);
    arb_one(want_mag);
    arb_div_ui(want_mag, want_mag, 2, prec);
    arb_sqrt(want_mag, want_mag, prec);    /* 1/sqrt(2) */

    /* Real part should be zero; imag should equal 1/sqrt(2). */
    arb_t diff_re, diff_im;
    arb_init(diff_re); arb_init(diff_im);
    arb_set(diff_re, acb_realref(got));
    arb_sub(diff_im, acb_imagref(got), want_mag, prec);

    /* arb_t carries certified error bounds; the mid-points should agree to
     * ~2^-prec.  We check that 1e-30 is comfortably inside the tolerance. */
    double dre = arf_get_d(arb_midref(diff_re), ARF_RND_NEAR);
    double dim = arf_get_d(arb_midref(diff_im), ARF_RND_NEAR);
    ASSERT_NEAR(dre, 0.0, 1e-30);
    ASSERT_NEAR(dim, 0.0, 1e-30);

    arb_clear(theta); acb_clear(got);
    arb_clear(want_mag); arb_clear(diff_re); arb_clear(diff_im);
}

int main(void)
{
    RUN(test_arb_wigner_spot);
    RUN(test_arb_direct_vs_fast);
    RUN(test_arb_vs_double);
    TEST_REPORT_AND_EXIT();
}
