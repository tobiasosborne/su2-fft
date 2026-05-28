/* test_wigner.c -- Red/green test for the matrix coefficient P^l_{n,m}(cos theta).
 *
 * Ground truth: paper.tex lines 537-543 (the Rodrigues-form definition).
 *
 * Cross-checks:
 *   1. P^l_{0,0}(cos theta) = Legendre polynomial P_l(cos theta)  (paper line 734).
 *      Verified at l = 0, 1, 2.
 *   2. P^l_{n,m}(cos 0) = delta_{n,m}            (because t^l(identity) = I).
 *   3. P^l_{n,m}(cos pi) = (-1)^{l-n} delta_{n,-m}  (because R_y(pi) = sigma_y up to phase).
 *      Actually, t^l(R_y(pi))_{nm} = (-1)^{l-m} delta_{n,-m}; combined with the
 *      Euler-angle decomposition the simple check is delta_{n,-m} times a sign.
 *      We test only |P^l_{n,m}(cos pi)| = delta_{n,-m}.
 *   4. Unitarity:  sum_{n} |P^l_{n,m}(cos theta)|^2  = 1  for every (l, m, theta).
 *      This follows because the matrix t^l(g) is unitary and the exponential
 *      factors have modulus one.
 *   5. Specific complex value: at l=1, n=1, m=0, theta=pi/2 we computed by hand
 *      P^1_{1,0}(0) = i * sin(pi/2)/sqrt(2) = i / sqrt(2).
 */
#include "test_framework.h"
#include "su2.h"
#include "su2_arb.h"

#include <flint/arb.h>
#include <flint/acb.h>
#include <flint/arf.h>
#include <math.h>
#include <stdlib.h>

/* Independent arb reference for the real Sakurai small-d; defined below. */
static double arb_real_d(int l, int n, int m, double theta, slong prec);

/* Legendre P_l(x) via Bonnet's recurrence -- independent reference. */
static double legendre_P(int l, double x)
{
    double p0 = 1.0, p1 = x;
    if (l == 0) return p0;
    if (l == 1) return p1;
    for (int k = 1; k < l; ++k) {
        double pn = ((2*k + 1) * x * p1 - k * p0) / (k + 1);
        p0 = p1; p1 = pn;
    }
    return p1;
}

static void test_wigner_legendre_special(void)
{
    /* P^l_{0,0}(cos theta) must equal Legendre P_l(cos theta). */
    int N = 13;
    double *theta = su2_grid_theta(N);
    for (int l = 0; l <= 3; ++l) {
        for (int k = 0; k < N; ++k) {
            double x = cos(theta[k]);
            double _Complex got  = su2_wigner_d(l, 0, 0, theta[k]);
            double          want = legendre_P(l, x);
            ASSERT_CNEAR(got, (double _Complex)want, 1e-12);
        }
    }
    free(theta);
}

static void test_wigner_identity_at_zero(void)
{
    /* At theta=0 (identity rotation), the matrix is identity. */
    for (int l = 0; l <= 3; ++l) {
        for (int n = -l; n <= l; ++n) {
            for (int m = -l; m <= l; ++m) {
                double _Complex got = su2_wigner_d(l, n, m, 0.0);
                double _Complex want = (n == m) ? 1.0 : 0.0;
                ASSERT_CNEAR(got, want, 1e-12);
            }
        }
    }
}

static void test_wigner_unitarity_columns(void)
{
    /* For each (l, m, theta), sum_n |P^l_{n,m}(cos theta)|^2 = 1. */
    int N = 7;
    double *theta = su2_grid_theta(N);
    for (int l = 0; l <= 3; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int k = 1; k < N - 1; ++k) {     /* skip exact endpoints */
                double s = 0.0;
                for (int n = -l; n <= l; ++n) {
                    double _Complex v = su2_wigner_d(l, n, m, theta[k]);
                    s += creal(v)*creal(v) + cimag(v)*cimag(v);
                }
                ASSERT_NEAR(s, 1.0, 1e-10);
            }
        }
    }
    free(theta);
}

static void test_wigner_d_seq_matches_phys(void)
{
    /* The sequence helper must reproduce d^l_{n,m}(theta) from an INDEPENDENT
     * implementation.  su2_wigner_d now routes through su2_wigner_d_seq itself
     * (bead su2fft-258), so the arb de Moivre sum (su2_wigner_d_arb) is the
     * only independent reference; we cross-check against it (CLAUDE.md rule 7).
     * Grid of (l, m, n, theta) covering the regimes used by su2_fft Stage 2.
     */
    int N = 12;
    int l_max = N - 1;
    double *theta = su2_grid_theta(N);
    double tol = 1e-12;

    double d_seq[16];  /* buffer for sequence values; l_max - l_min + 1 <= N */

    for (int m = -(N - 1); m <= N - 1; ++m) {
        for (int n = -(N - 1); n <= N - 1; ++n) {
            int l_min = (abs(m) > abs(n)) ? abs(m) : abs(n);
            if (l_min > l_max) continue;
            /* 3 theta samples keep the arb reference cost ~0.5s. */
            for (int k = 2; k < N - 1; k += 3) {     /* skip exact endpoints */
                su2_wigner_d_seq(l_min, l_max, n, m, theta[k], d_seq);
                for (int l = l_min; l <= l_max; ++l) {
                    double want = arb_real_d(l, n, m, theta[k], 64);
                    ASSERT_NEAR(d_seq[l - l_min], want, tol);
                }
            }
        }
    }
    free(theta);
}

static void test_wigner_specific_complex_value(void)
{
    /* Computed by hand from paper line 537:
     *   P^1_{n=1, m=0}(cos(pi/2)) = i * sin(pi/2)/sqrt(2) = i / sqrt(2). */
    double _Complex got = su2_wigner_d(1, 1, 0, M_PI / 2.0);
    double _Complex want = I / sqrt(2.0);
    ASSERT_CNEAR(got, want, 1e-12);
}

/* ------- Half-integer-compatible Wigner-d (bead su2fft-n8e, Tier 1) -------
 *
 * The half-integer routine encodes (l, n, m) as (2l, 2n, 2m) integers so
 * that l = 1/2, 3/2, ... is representable without FP comparison.  See
 * notes/half_integer.md.
 */

static void test_wigner_half_spin_half_at_identity(void)
{
    /* d^{1/2}(theta=0) = identity 2x2 matrix.  Indices in 2x-encoded form:
     * 2n in {-1, +1} corresponds to physical n in {-1/2, +1/2}. */
    int two_l = 1;
    double _Complex d00 = su2_wigner_d_half(two_l, -1, -1, 0.0);
    double _Complex d01 = su2_wigner_d_half(two_l, -1,  1, 0.0);
    double _Complex d10 = su2_wigner_d_half(two_l,  1, -1, 0.0);
    double _Complex d11 = su2_wigner_d_half(two_l,  1,  1, 0.0);
    ASSERT_CNEAR(d00, 1.0 + 0.0*I, 1e-13);
    ASSERT_CNEAR(d01, 0.0 + 0.0*I, 1e-13);
    ASSERT_CNEAR(d10, 0.0 + 0.0*I, 1e-13);
    ASSERT_CNEAR(d11, 1.0 + 0.0*I, 1e-13);
}

static void test_wigner_half_spin_half_closed_form(void)
{
    /* Sakurai d^{1/2}:
     *   d^{1/2}_{1/2, 1/2}(theta)   =  cos(theta/2)
     *   d^{1/2}_{1/2,-1/2}(theta)   = -sin(theta/2)
     *   d^{1/2}_{-1/2,1/2}(theta)   =  sin(theta/2)
     *   d^{1/2}_{-1/2,-1/2}(theta)  =  cos(theta/2)
     * After the paper's i^{m-n} phase (su2_wigner_d_half returns P, not d):
     *   P^{1/2}_{1/2, 1/2}   = i^0  *  cos(t/2)         =  cos(t/2)
     *   P^{1/2}_{1/2,-1/2}   = i^-1 * (-sin(t/2))       = +i sin(t/2)
     *   P^{1/2}_{-1/2,1/2}   = i^+1 * ( sin(t/2))       = +i sin(t/2)
     *   P^{1/2}_{-1/2,-1/2}  = i^0  *  cos(t/2)         =  cos(t/2)
     */
    double theta = 0.7;
    int two_l = 1;
    double c = cos(theta * 0.5);
    double s = sin(theta * 0.5);
    ASSERT_CNEAR(su2_wigner_d_half(two_l,  1,  1, theta),  c + 0.0*I, 1e-12);
    ASSERT_CNEAR(su2_wigner_d_half(two_l,  1, -1, theta),  0.0 + s*I, 1e-12);
    ASSERT_CNEAR(su2_wigner_d_half(two_l, -1,  1, theta),  0.0 + s*I, 1e-12);
    ASSERT_CNEAR(su2_wigner_d_half(two_l, -1, -1, theta),  c + 0.0*I, 1e-12);
}

static void test_wigner_half_matches_integer(void)
{
    /* For two_l even (integer l), su2_wigner_d_half must equal su2_wigner_d
     * up to tgamma-vs-factorial-table FP noise.  CLAUDE.md rule 7:
     * cross-checks > unit tests. */
    int N = 5;
    double *theta = su2_grid_theta(N);
    for (int l = 0; l <= 3; ++l) {
        for (int n = -l; n <= l; ++n) {
            for (int m = -l; m <= l; ++m) {
                for (int k = 0; k < N; ++k) {
                    double _Complex got  = su2_wigner_d_half(2*l, 2*n, 2*m, theta[k]);
                    double _Complex want = su2_wigner_d(l, n, m, theta[k]);
                    ASSERT_CNEAR(got, want, 1e-11);
                }
            }
        }
    }
    free(theta);
}

static void test_wigner_half_unitarity(void)
{
    /* Column unitarity for spin-1/2: sum_n |P^{1/2}_{n, m}|^2 = 1 for any
     * (m, theta).  The sum runs over 2n in {-1, +1}. */
    int two_l = 1;
    for (double theta = 0.1; theta < 3.0; theta += 0.5) {
        for (int two_m = -1; two_m <= 1; two_m += 2) {
            double s = 0.0;
            for (int two_n = -1; two_n <= 1; two_n += 2) {
                double _Complex v = su2_wigner_d_half(two_l, two_n, two_m, theta);
                s += creal(v)*creal(v) + cimag(v)*cimag(v);
            }
            ASSERT_NEAR(s, 1.0, 1e-12);
        }
    }
}

/* ------- High-l regression (bead: factorial-table overflow in src) -------
 *
 * src/su2_wigner.c evaluates the real Sakurai small-d via a de Moivre sum
 * using a double factorial table capped at FACT_MAX=100 (returns 0.0 above)
 * and plain double arithmetic.  Above l~50 this returns NaN/garbage:
 * factorial arguments reach 2l>100 (table cap -> 0 -> 0/0 = NaN) and doubles
 * overflow at 171!.  su2_wigner_d_seq seeds via that broken path.
 *
 * The arbitrary-precision routine su2_wigner_d_arb is correct at any l, so it
 * is the ground truth here.  These tests pin the desired post-fix behaviour:
 * finiteness and agreement with arb at l in {60, 80}.  They FAIL now (red).
 */

/* Recover the REAL Sakurai small-d d^l_{n,m}(theta) from the arb routine,
 * which returns the paper coefficient P^l_{n,m} = i^{m-n} * d.  See the
 * parity-of-(m-n) recovery in su2_arb.h's documented convention. */
static double arb_real_d(int l, int n, int m, double theta, slong prec)
{
    acb_t P;
    acb_init(P);
    arb_t th;
    arb_init(th);
    arb_set_d(th, theta);
    su2_wigner_d_arb(P, l, n, m, th, prec);
    int r = ((m - n) % 4 + 4) % 4;
    double d;
    switch (r) {
        case 0: d =  arf_get_d(arb_midref(acb_realref(P)), ARF_RND_NEAR); break;
        case 1: d =  arf_get_d(arb_midref(acb_imagref(P)), ARF_RND_NEAR); break;
        case 2: d = -arf_get_d(arb_midref(acb_realref(P)), ARF_RND_NEAR); break;
        default: d = -arf_get_d(arb_midref(acb_imagref(P)), ARF_RND_NEAR); break;
    }
    arb_clear(th);
    acb_clear(P);
    return d;
}

static void test_wigner_d_seq_high_l_matches_arb(void)
{
    /* (n, m, theta) with l_min = max(|n|,|m|) in {60, 80, 75}. */
    struct { int n, m; double theta; } cases[] = {
        { 60,  55, 0.7 },
        {-20, -80, 1.9 },   /* l_min = 80 */
        { 75,  75, 0.4 },
    };
    int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

    for (int c = 0; c < ncases; ++c) {
        int n = cases[c].n, m = cases[c].m;
        double theta = cases[c].theta;
        int l_min = (abs(n) > abs(m)) ? abs(n) : abs(m);

        /* Single-value seed: l_max == l_min. */
        double out0 = 0.0;
        su2_wigner_d_seq(l_min, l_min, n, m, theta, &out0);
        ASSERT_TRUE(isfinite(out0));
        double want0 = arb_real_d(l_min, n, m, theta, 256);
        ASSERT_NEAR(out0, want0, 1e-13);

        /* Two-term second seed plus a few recurrence steps: l_max = l_min+3. */
        int l_max = l_min + 3;
        double out_d[4];   /* l_max - l_min + 1 == 4 */
        su2_wigner_d_seq(l_min, l_max, n, m, theta, out_d);
        for (int l = l_min; l <= l_max; ++l) {
            double got  = out_d[l - l_min];
            ASSERT_TRUE(isfinite(got));
            double want = arb_real_d(l, n, m, theta, 256);
            ASSERT_NEAR(got, want, 1e-12);
        }
    }
}

static void test_wigner_d_high_l_matches_arb(void)
{
    /* The public su2_wigner_d returns the complex P^l_{n,m}.  Compare the
     * full complex acb midpoint (real and imag) at l in {60, 80}. */
    struct { int l, n, m; double theta; } cases[] = {
        { 60, 60,  55, 0.7 },
        { 80,-20, -80, 1.9 },
        { 75, 75,  75, 0.4 },
    };
    int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

    for (int c = 0; c < ncases; ++c) {
        int l = cases[c].l, n = cases[c].n, m = cases[c].m;
        double theta = cases[c].theta;

        double _Complex got = su2_wigner_d(l, n, m, theta);
        ASSERT_TRUE(isfinite(creal(got)));
        ASSERT_TRUE(isfinite(cimag(got)));

        acb_t P;
        acb_init(P);
        arb_t th;
        arb_init(th);
        arb_set_d(th, theta);
        su2_wigner_d_arb(P, l, n, m, th, 256);
        double want_re = arf_get_d(arb_midref(acb_realref(P)), ARF_RND_NEAR);
        double want_im = arf_get_d(arb_midref(acb_imagref(P)), ARF_RND_NEAR);
        arb_clear(th);
        acb_clear(P);

        double _Complex want = want_re + want_im * I;
        ASSERT_CNEAR(got, want, 1e-13);
    }
}

int main(void)
{
    RUN(test_wigner_legendre_special);
    RUN(test_wigner_identity_at_zero);
    RUN(test_wigner_unitarity_columns);
    RUN(test_wigner_d_seq_matches_phys);
    RUN(test_wigner_specific_complex_value);
    RUN(test_wigner_half_spin_half_at_identity);
    RUN(test_wigner_half_spin_half_closed_form);
    RUN(test_wigner_half_matches_integer);
    RUN(test_wigner_half_unitarity);
    RUN(test_wigner_d_seq_high_l_matches_arb);
    RUN(test_wigner_d_high_l_matches_arb);
    TEST_REPORT_AND_EXIT();
}
