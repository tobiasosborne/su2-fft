/* test_resolved_arb.c -- Strict cross-checks for the arbitrary-precision
 * resolved-grid SU(2) FFT (bead `su2fft-rrx`, Step 10).
 *
 * What this file verifies
 * -----------------------
 * The arb resolved-grid forward / inverse routines defined in
 *   src/su2_fft_resolved_arb.c
 *   src/su2_ft_resolved_arb.c
 *   src/su2_gauss_legendre_arb.c
 * are the FLINT-precision twins of the double-precision resolved-grid path
 * (bead `su2fft-0t1`, tests/test_resolved.c).  At prec=128 we expect the
 * forward/inverse pair to reach the 1e-30 floor; at prec=256 we expect
 * 1e-50 -- the "headline" certificate quoted in the bead.
 *
 * Tests (target tolerances after the colon)
 * -----------------------------------------
 *   1. test_resolved_arb_constant                : 1e-30  (prec=128)
 *   2. test_resolved_arb_fast_vs_direct          : 1e-30  (prec=128, N=4,5)
 *   3. test_resolved_arb_inv_delta               : 1e-15  (cos via double acos)
 *   4. test_resolved_arb_spectrum_roundtrip_p128 : 1e-30  (prec=128, N=4,8)
 *   5. test_resolved_arb_spectrum_roundtrip_p256 : 1e-50  (prec=256, N=4,8)  <- HEADLINE
 *   6. test_resolved_arb_vs_double               : 1e-10  (prec=53 vs double)
 *
 * All ratios / diffs are extracted as |z| via acb_abs and then arf_get_d at
 * ARF_RND_NEAR.  acb_abs gives an arb_t whose mid-point's exponent reaches
 * down to ~1e-308 in IEEE double, well below the 1e-50 target -- no precision
 * is lost in the assert step.
 */
#include "test_framework.h"
#include "su2.h"
#include "su2_arb.h"

#include <flint/arb.h>
#include <flint/acb.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* --------- Local helpers (mirrors tests/test_arb.c) --------- */

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
    return re + im * I;
}

/* |a - b| as a double; the subtraction and abs are performed in arb at the
 * given precision so the result keeps its exponent down to ~1e-308. */
static double acb_abs_diff(const acb_t a, const acb_t b, slong prec)
{
    acb_t d;
    arb_t r;
    acb_init(d); arb_init(r);
    acb_sub(d, a, b, prec);
    acb_abs(r, d, prec);
    double out = arf_get_d(arb_midref(r), ARF_RND_NEAR);
    acb_clear(d); arb_clear(r);
    return out;
}

/* |a| via acb_abs -> double mid-point. */
static double acb_abs_to_d(const acb_t a, slong prec)
{
    arb_t r;
    arb_init(r);
    acb_abs(r, a, prec);
    double out = arf_get_d(arb_midref(r), ARF_RND_NEAR);
    arb_clear(r);
    return out;
}

/* Infinity norm of an acb vector. */
static double acb_vec_inf_norm(acb_srcptr v, size_t n, slong prec)
{
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double x = acb_abs_to_d(v + i, prec);
        if (x > m) m = x;
    }
    return m;
}

/* Infinity norm of (a - b) over a pair of acb vectors. */
static double acb_vec_inf_diff(acb_srcptr a, acb_srcptr b,
                               size_t n, slong prec)
{
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double x = acb_abs_diff(a + i, b + i, prec);
        if (x > m) m = x;
    }
    return m;
}

/* --------- 1. forward(constant) at prec=128 ---------
 *
 * Constant input f = 1 on the resolved grid is exactly bandlimited at l=0,
 * so the analysis quadrature is exact -- fhat(0)_{0,0} = 1 and every other
 * coefficient must vanish.  Smoke at N=4 gave 1.18e-38; we assert 1e-30 to
 * leave headroom for accumulated O(N^4) arb operations.
 */
static void test_resolved_arb_constant(void)
{
    const int   N    = 4;
    const slong prec = 128;
    const size_t nsamp  = su2_resolved_total_samples(N);
    const size_t ncoeff = su2_total_coeffs(N);

    acb_ptr f    = _acb_vec_init(nsamp);
    acb_ptr fhat = _acb_vec_init(ncoeff);

    for (size_t i = 0; i < nsamp; ++i) acb_one(f + i);

    su2_fft_resolved_arb(N, f, fhat, prec);

    /* fhat(0)_{0,0} sits at offset 0. */
    acb_t one_acb, diff_acb;
    acb_init(one_acb); acb_init(diff_acb);
    acb_one(one_acb);
    acb_sub(diff_acb, fhat + 0, one_acb, prec);
    double err_dc = acb_abs_to_d(diff_acb, prec);

    /* Max |fhat[i]| over i != 0. */
    double leak = 0.0;
    for (size_t i = 1; i < ncoeff; ++i) {
        double x = acb_abs_to_d(fhat + i, prec);
        if (x > leak) leak = x;
    }

    printf("  [arb constant N=%d prec=%ld] fhat(0,0,0) - 1 = %.4e, max leak = %.4e\n",
           N, (long)prec, err_dc, leak);
    fflush(stdout);

    ASSERT_TRUE(err_dc < 1e-30);
    ASSERT_TRUE(leak   < 1e-30);

    acb_clear(one_acb); acb_clear(diff_acb);
    _acb_vec_clear(f,    nsamp);
    _acb_vec_clear(fhat, ncoeff);
}

/* --------- 2. arb fast vs arb direct on random complex input ---------
 *
 * Both routines compute the same discrete sum at the same precision -- the
 * ball mid-points must agree to ~2^-prec.  We assert 1e-30 at prec=128; the
 * smoke run reported 6.28e-39.  Two N (4 and 5) exercise even/odd P parity.
 */
static void test_resolved_arb_fast_vs_direct(void)
{
    const slong prec = 128;
    const int   Ns[] = { 4, 5 };

    for (int idx = 0; idx < 2; ++idx) {
        int N = Ns[idx];
        size_t nsamp  = su2_resolved_total_samples(N);
        size_t ncoeff = su2_total_coeffs(N);

        acb_ptr f      = _acb_vec_init(nsamp);
        acb_ptr fhat_d = _acb_vec_init(ncoeff);
        acb_ptr fhat_f = _acb_vec_init(ncoeff);

        unsigned seed = 20260526u + (unsigned)N;
        for (size_t i = 0; i < nsamp; ++i) {
            double _Complex c = (drand(&seed) - 0.5)
                              + (drand(&seed) - 0.5) * I;
            acb_set_d_complex(f + i, c);
        }

        su2_ft_direct_resolved_arb(N, f, fhat_d, prec);
        su2_fft_resolved_arb     (N, f, fhat_f, prec);

        double max_diff = acb_vec_inf_diff(fhat_f, fhat_d, ncoeff, prec);
        printf("  [arb fast vs direct] N=%d prec=%ld max_diff = %.4e\n",
               N, (long)prec, max_diff);
        fflush(stdout);

        ASSERT_TRUE(max_diff < 1e-30);

        _acb_vec_clear(f,      nsamp);
        _acb_vec_clear(fhat_d, ncoeff);
        _acb_vec_clear(fhat_f, ncoeff);
    }
}

/* --------- 3. inverse(delta@(l=1, m=0, n=0)) = 3 cos(theta_k) ---------
 *
 * Peter-Weyl with fhat(1)_{0,0} = 1 yields f(g) = 3 P^1_{0,0}(cos theta)
 * = 3 cos theta, independent of phi and psi.  The expected value uses the
 * arb GL nodes' mid-point reduced to a double via acos; that conversion
 * caps the achievable tolerance at ~1e-15.
 */
static void test_resolved_arb_inv_delta(void)
{
    const int   N    = 4;
    const slong prec = 128;
    const int   P    = su2_resolved_P(N);
    const size_t nsamp  = su2_resolved_total_samples(N);
    const size_t ncoeff = su2_total_coeffs(N);

    acb_ptr fhat = _acb_vec_init(ncoeff);
    acb_ptr f    = _acb_vec_init(nsamp);

    acb_one(fhat + su2_coeff_offset(1) + su2_mn_index(1, 0, 0));
    su2_fft_resolved_inv_arb(N, fhat, f, prec);

    /* GL nodes at the same precision; reduce each mid-point to double for
     * the cos comparison. */
    arb_ptr x_gl = _arb_vec_init(N);
    arb_ptr w_gl = _arb_vec_init(N);
    su2_gl_nodes_weights_arb(N, x_gl, w_gl, prec);

    double max_im = 0.0, max_re = 0.0;
    for (int j1 = 0; j1 < P; ++j1) {
        for (int k = 0; k < N; ++k) {
            double x_k     = arf_get_d(arb_midref(x_gl + k), ARF_RND_NEAR);
            double theta_k = acos(x_k);
            double expected_re = 3.0 * cos(theta_k);
            for (int j2 = 0; j2 < P; ++j2) {
                size_t idx = su2_resolved_sample_index(N, j1, k, j2);
                double _Complex got = acb_to_complex(f + idx);
                double dre = fabs(creal(got) - expected_re);
                double dim = fabs(cimag(got));
                if (dre > max_re) max_re = dre;
                if (dim > max_im) max_im = dim;
            }
        }
    }
    printf("  [arb inv delta N=%d prec=%ld] max |Re-3cos| = %.4e, max |Im| = %.4e\n",
           N, (long)prec, max_re, max_im);
    fflush(stdout);

    /* Imag part is real-zero up to working precision (well below 1e-30); Re
     * is limited by the double acos used for the reference. */
    ASSERT_TRUE(max_im < 1e-30);
    ASSERT_TRUE(max_re < 1e-15);

    _arb_vec_clear(x_gl, N);
    _arb_vec_clear(w_gl, N);
    _acb_vec_clear(fhat, ncoeff);
    _acb_vec_clear(f,    nsamp);
}

/* Shared body for tests 4 and 5: random fhat, inverse, then forward, check
 * fhat2 ~ fhat in relative infinity norm. */
static void run_spectrum_roundtrip(int N, slong prec, double tol,
                                   const char *label)
{
    size_t nsamp  = su2_resolved_total_samples(N);
    size_t ncoeff = su2_total_coeffs(N);

    acb_ptr fhat  = _acb_vec_init(ncoeff);
    acb_ptr fhat2 = _acb_vec_init(ncoeff);
    acb_ptr f     = _acb_vec_init(nsamp);

    unsigned seed = 20260526u + (unsigned)N + (unsigned)prec;
    for (size_t i = 0; i < ncoeff; ++i) {
        double _Complex c = (drand(&seed) - 0.5)
                          + (drand(&seed) - 0.5) * I;
        acb_set_d_complex(fhat + i, c);
    }

    su2_fft_resolved_inv_arb(N, fhat,  f,     prec);
    su2_fft_resolved_arb    (N, f,     fhat2, prec);

    double num    = acb_vec_inf_diff(fhat2, fhat, ncoeff, prec);
    double den    = acb_vec_inf_norm(fhat, ncoeff, prec);
    double relerr = (den > 0.0) ? num / den : num;

    printf("  [arb spectrum roundtrip %s] N=%d prec=%ld max_relerr = %.4e\n",
           label, N, (long)prec, relerr);
    fflush(stdout);

    ASSERT_TRUE(relerr < tol);

    _acb_vec_clear(fhat,  ncoeff);
    _acb_vec_clear(fhat2, ncoeff);
    _acb_vec_clear(f,     nsamp);
}

/* --------- 4. Spectrum roundtrip at prec=128 (N=4 and N=8) --------- */
static void test_resolved_arb_spectrum_roundtrip_prec128(void)
{
    run_spectrum_roundtrip(4, 128, 1e-30, "p128");
    run_spectrum_roundtrip(8, 128, 1e-30, "p128");
}

/* --------- 5. HEADLINE: spectrum roundtrip at prec=256 (N=4 and N=8) ---
 *
 * 1e-50 relative is well within 2^-256 ~ 8.6e-78.  This is the certificate
 * we quote in the bead.  The N=8 invocation dominates runtime in the suite
 * (acb arithmetic at 256 bits, O(N^4) operations).
 */
static void test_resolved_arb_spectrum_roundtrip_prec256(void)
{
    run_spectrum_roundtrip(4, 256, 1e-50, "p256");
    run_spectrum_roundtrip(8, 256, 1e-50, "p256");
}

/* --------- 6. arb at prec=53 vs the double-precision path ---------
 *
 * The arb routines, run at IEEE-double precision, must agree with the
 * double path to a few ULPs scaled by O(N^3).  Analog of test_arb_vs_double
 * in tests/test_arb.c.
 */
static void test_resolved_arb_vs_double(void)
{
    const int   N    = 5;
    const slong prec = 53;
    const size_t nsamp  = su2_resolved_total_samples(N);
    const size_t ncoeff = su2_total_coeffs(N);

    double _Complex *f_d        = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fhat_d_dbl = calloc(ncoeff, sizeof(double _Complex));
    acb_ptr f_a    = _acb_vec_init(nsamp);
    acb_ptr fhat_a = _acb_vec_init(ncoeff);

    unsigned seed = 20260526u;
    for (size_t i = 0; i < nsamp; ++i) {
        f_d[i] = (drand(&seed) - 0.5) + (drand(&seed) - 0.5) * I;
        acb_set_d_complex(f_a + i, f_d[i]);
    }

    su2_fft_resolved    (N, f_d, fhat_d_dbl);
    su2_fft_resolved_arb(N, f_a, fhat_a, prec);

    double max_diff = 0.0;
    for (size_t i = 0; i < ncoeff; ++i) {
        double _Complex a = fhat_d_dbl[i];
        double _Complex b = acb_to_complex(fhat_a + i);
        double d = cabs(a - b);
        if (d > max_diff) max_diff = d;
    }
    printf("  [arb vs double] N=%d prec=%ld max_diff = %.4e\n",
           N, (long)prec, max_diff);
    fflush(stdout);

    for (size_t i = 0; i < ncoeff; ++i) {
        double _Complex a = fhat_d_dbl[i];
        double _Complex b = acb_to_complex(fhat_a + i);
        ASSERT_CNEAR(a, b, 1e-10);
    }

    free(f_d); free(fhat_d_dbl);
    _acb_vec_clear(f_a,    nsamp);
    _acb_vec_clear(fhat_a, ncoeff);
}

int main(void)
{
    RUN(test_resolved_arb_constant);
    RUN(test_resolved_arb_fast_vs_direct);
    RUN(test_resolved_arb_inv_delta);
    RUN(test_resolved_arb_spectrum_roundtrip_prec128);
    RUN(test_resolved_arb_spectrum_roundtrip_prec256);
    RUN(test_resolved_arb_vs_double);
    TEST_REPORT_AND_EXIT();
}
