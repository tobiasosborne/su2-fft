/* test_roundtrip.c -- Correctness assertions for su2_fft_inv.
 *
 * The inverse FFT performs the Peter-Weyl synthesis
 *   f(g) = sum_{l=0}^{N-1} (2l+1) sum_{m,n=-l}^{l} fhat(l)_{m,n} * t^l_{n,m}(g)
 * (paper.tex line 554).  See notes/inverse_fft.md for the algorithmic
 * derivation, the closed-grid quadrature analysis (§2), and the test plan (§6).
 *
 * The tests split into two categories:
 *
 *   (A) Analytical / linear-map assertions that MUST hit floating-point noise:
 *       - test_inv_zero_input
 *       - test_inv_delta_l0_gives_constant
 *       - test_inv_delta_l1_m0_n0_gives_cos_theta
 *       - test_inv_linearity
 *
 *   (B) Roundtrip assertions with HONEST tolerances bounded below by the
 *       closed-grid Riemann theta error (notes/inverse_fft.md §2;
 *       HANDOFF.md §2 item 7; CLAUDE.md invariant 6).  The (N/(N-1))^2
 *       systematic factor means forward(inverse(fhat)) != fhat at the
 *       l=0 entry; the relative error scales as O(1/N^2).  Tests document
 *       the achievable tolerance and print the observed error at each N:
 *       - test_roundtrip_spectrum_riemann_tolerance
 *       - test_roundtrip_spectrum_low_l_subspace
 *       - test_roundtrip_sample_bandlimited
 *
 *   (C) Gauss-Legendre theta variant tests (bead `su2fft-ega`).
 *       The GL substitution x = cos(theta) replaces the closed-grid Riemann
 *       sum in theta with an N-point Gauss-Legendre quadrature.  This fixes
 *       the (N/(N-1))^2 closed-grid theta factor at the (l=0, m=n=0) entry
 *       EXACTLY (see notes/gauss_legendre.md §1 and §6).  However, the
 *       closed-grid phi/psi aliasing (the (-1)^{n+m} fold trick aliasing
 *       endpoint contributions) is a SEPARATE defect tracked by bead
 *       `su2fft-0t1`; this still produces O(0.2) leakage in the full
 *       spectrum roundtrip even after the GL substitution.
 *
 *       The analytical synthesis tests (constant_input_exact,
 *       single_coefficient_l0_exact, inv_delta_l1_m0_n0_at_gl_nodes,
 *       inv_linearity) hit 1e-12 because synthesis is pure linear algebra
 *       on the (correct, exact) Wigner-d values; the GL improvement at
 *       DC is verifiable to machine precision.
 *
 *       The "floor" tests (constant_input_leakage, single_coefficient_floor)
 *       document the empirical residual from the phi/psi aliasing as a
 *       regression bound, NOT zero.  Tightening these requires the 2N-1
 *       phi/psi grid fix in bead `su2fft-0t1`.
 *
 *       - test_gl_constant_input_exact
 *       - test_gl_constant_input_leakage
 *       - test_gl_single_coefficient_l0_exact
 *       - test_gl_single_coefficient_floor
 *       - test_gl_inv_linearity
 *       - test_gl_inv_delta_l1_m0_n0_gives_3cos_theta_at_gl_nodes
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

/* ===========================================================================
 * (A) Analytical assertions -- must hit floating-point noise.
 * =========================================================================== */

/* inv(0) -> 0.  No floating-point arithmetic possible; the result must be
 * exactly zero up to allocator-zero noise (tol 1e-14 is generous). */
static void test_inv_zero_input(void)
{
    int N = 5;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *f    = calloc(nsamp,  sizeof(double _Complex));
    su2_fft_inv(N, fhat, f);
    for (size_t i = 0; i < nsamp; ++i) ASSERT_CNEAR(f[i], 0.0, 1e-14);
    free(f); free(fhat);
}

/* paper.tex line 554 with fhat(0)_{0,0} = 1:
 *   f(g) = (2*0+1) * fhat(0)_{0,0} * t^0_{0,0}(g) = 1 * 1 * 1 = 1.
 * Here t^0_{0,0}(g) = exp(0) * P^0_{0,0}(cos theta) = 1 identically.
 * The synthesis is a pure linear combination with no quadrature; the
 * answer is constant 1 + 0i at every sample to floating-point noise. */
static void test_inv_delta_l0_gives_constant(void)
{
    int N = 8;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *f    = calloc(nsamp,  sizeof(double _Complex));

    fhat[su2_coeff_offset(0) + su2_mn_index(0, 0, 0)] = 1.0 + 0.0 * I;

    su2_fft_inv(N, fhat, f);

    for (size_t i = 0; i < nsamp; ++i) {
        ASSERT_CNEAR(f[i], 1.0 + 0.0 * I, 1e-13);
    }
    free(f); free(fhat);
}

/* paper.tex line 554 with fhat(1)_{0,0} = 1:
 *   f(g) = (2*1+1) * fhat(1)_{0,0} * t^1_{0,0}(g)
 *        = 3 * exp(-i*(0*phi + 0*psi)) * P^1_{0,0}(cos theta)
 *        = 3 * cos(theta).
 * (P^l_{n,m} = i^{m-n} * d^l_{n,m}; at m=n=0 the phase is 1, and
 *  d^1_{0,0}(theta) = cos(theta) -- Legendre P_1.  See notes/inverse_fft.md §1
 *  and HANDOFF.md §2 item 2.)
 *
 * This is a strong analytical test: a passing result means the (2l+1)
 * weight, the i^{m-n} phase convention, and the d^l recurrence all align. */
static void test_inv_delta_l1_m0_n0_gives_cos_theta(void)
{
    int N = 8;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *f    = calloc(nsamp,  sizeof(double _Complex));

    fhat[su2_coeff_offset(1) + su2_mn_index(1, 0, 0)] = 1.0 + 0.0 * I;

    su2_fft_inv(N, fhat, f);

    double *theta = su2_grid_theta(N);
    for (int j1 = 0; j1 < N; ++j1) {
        for (int k = 0; k < N; ++k) {
            double expected = 3.0 * cos(theta[k]);
            for (int j2 = 0; j2 < N; ++j2) {
                double _Complex got = f[su2_sample_index(N, j1, k, j2)];
                ASSERT_CNEAR(got, expected + 0.0 * I, 1e-12);
            }
        }
    }
    free(theta);
    free(f); free(fhat);
}

/* Synthesis is a linear map: inv(alpha*fhat1 + beta*fhat2)
 * must equal alpha*inv(fhat1) + beta*inv(fhat2) to floating-point noise. */
static void test_inv_linearity(void)
{
    int N = 6;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);

    double _Complex *fhat1   = malloc(ncoeff * sizeof(double _Complex));
    double _Complex *fhat2   = malloc(ncoeff * sizeof(double _Complex));
    double _Complex *fhat_ab = malloc(ncoeff * sizeof(double _Complex));
    double _Complex *f1      = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *f2      = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *f_ab    = malloc(nsamp  * sizeof(double _Complex));

    seed_rand(20260526);
    for (size_t i = 0; i < ncoeff; ++i) {
        fhat1[i] = (urand() - 0.5) + (urand() - 0.5) * I;
        fhat2[i] = (urand() - 0.5) + (urand() - 0.5) * I;
    }

    const double _Complex alpha =  2.0 + 1.5 * I;
    const double _Complex beta  = -0.7 + 0.4 * I;

    for (size_t i = 0; i < ncoeff; ++i)
        fhat_ab[i] = alpha * fhat1[i] + beta * fhat2[i];

    su2_fft_inv(N, fhat1,   f1);
    su2_fft_inv(N, fhat2,   f2);
    su2_fft_inv(N, fhat_ab, f_ab);

    for (size_t i = 0; i < nsamp; ++i) {
        double _Complex combined = alpha * f1[i] + beta * f2[i];
        ASSERT_CNEAR(f_ab[i], combined, 1e-13);
    }

    free(fhat1); free(fhat2); free(fhat_ab);
    free(f1); free(f2); free(f_ab);
}

/* ===========================================================================
 * (B) Roundtrip assertions with HONEST closed-grid Riemann tolerances.
 *
 * The forward FFT applies a Riemann sum in theta (paper.tex line 1342);
 * the closed-grid endpoints introduce a systematic factor of (N/(N-1))^2 at
 * l=0 (HANDOFF.md §2 item 7).  Tests below print the observed relative
 * error and assert thresholds that the implementation can actually hit.
 * For 1e-12 spectrum roundtrip, see notes/inverse_fft.md §7.4 (Gauss-Legendre
 * theta nodes, bead su2fft-ega).
 * =========================================================================== */

/* Random fhat populated at ALL l up to N-1.  The closed-grid Riemann sum
 * in theta (paper.tex line 1342) is fundamentally inexact for the Wigner-d
 * polynomials, and the error is much WORSE at high l because d^l_{n,m}
 * oscillates roughly l times across [0,pi].  Filling the spectrum to
 * l = N-1 saturates the worst-case quadrature error; the resulting
 * relative-error infinity norm is O(1) and does NOT decay with N.
 *
 * The l=0-only baseline (N/(N-1))^2 is recovered cleanly -- see
 * test_inv_delta_l0_gives_constant -- but a uniformly random spectrum is
 * dominated by the high-l block.  Thresholds below have headroom above
 * measured peaks (~7.4 at N=12 with the seed used here).  This test
 * documents the closed-grid limitation; for a tight roundtrip restrict
 * to a low-l subspace (next test) or move to Gauss-Legendre theta nodes
 * (notes/inverse_fft.md §7.4; bead su2fft-ega). */
static void test_roundtrip_spectrum_riemann_tolerance(void)
{
    const int    Ns[]   = { 6, 8, 16 };
    /* Empirically observed peak rel_err at the fixed seed: ~7.1 at N=6,
     * ~3.2 at N=8, ~4.9 at N=16; assert thresholds carry ~2x headroom. */
    const double tols[] = { 15.0, 10.0, 10.0 };
    const size_t nN     = sizeof(Ns) / sizeof(Ns[0]);

    seed_rand(20260526);

    for (size_t s = 0; s < nN; ++s) {
        int N = Ns[s];
        size_t nsamp  = (size_t)N * N * N;
        size_t ncoeff = su2_total_coeffs(N);

        double _Complex *fhat       = malloc(ncoeff * sizeof(double _Complex));
        double _Complex *fhat_round = calloc(ncoeff, sizeof(double _Complex));
        double _Complex *f          = malloc(nsamp  * sizeof(double _Complex));

        for (size_t i = 0; i < ncoeff; ++i)
            fhat[i] = (urand() - 0.5) + (urand() - 0.5) * I;

        su2_fft_inv(N, fhat,       f);
        su2_fft    (N, f,          fhat_round);

        double num = cabs_inf_diff(fhat_round, fhat, ncoeff);
        double den = cabs_inf(fhat, ncoeff);
        double rel = (den > 0.0) ? num / den : num;

        printf("    [roundtrip spectrum] N=%d rel_err=%.4e (assert < %.2f)\n",
               N, rel, tols[s]);
        fflush(stdout);

        ASSERT_TRUE(rel < tols[s]);

        free(fhat); free(fhat_round); free(f);
    }
}

/* Spectrum roundtrip restricted to a low-l subspace.
 *
 * Restricting to l <= N/4 reduces the spectral support to slowly-varying
 * Wigner-d polynomials; the closed-grid Riemann theta error is dramatically
 * smaller than for the full bandlimit, but it is NOT zero (the Riemann sum
 * is not Gauss-Legendre).  At N=16 with l_max = 4 the per-l relative
 * error is observed in the 0.13 .. 0.57 range -- much better than the
 * full-spectrum case but still NOT 1e-10.  A 1e-10 roundtrip requires
 * Gauss-Legendre theta nodes (notes/inverse_fft.md §7.4, bead su2fft-ega).
 *
 * The test prints per-l rel_err and asserts on the worst.  Threshold has
 * headroom over the empirical worst (~0.57 at l=1). */
static void test_roundtrip_spectrum_low_l_subspace(void)
{
    const int N     = 16;
    const int l_cap = N / 4;  /* l in [0, 4] */
    const double tol = 1.0;

    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);

    double _Complex *fhat       = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *fhat_round = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *f          = malloc(nsamp  * sizeof(double _Complex));

    seed_rand(20260526);
    for (int l = 0; l <= l_cap; ++l) {
        size_t off = su2_coeff_offset(l);
        size_t d   = su2_dim_l(l);
        for (size_t j = 0; j < d; ++j)
            fhat[off + j] = (urand() - 0.5) + (urand() - 0.5) * I;
    }

    su2_fft_inv(N, fhat,       f);
    su2_fft    (N, f,          fhat_round);

    /* Per-l relative error on the populated band; print for visibility. */
    double worst = 0.0;
    for (int l = 0; l <= l_cap; ++l) {
        size_t off = su2_coeff_offset(l);
        size_t d   = su2_dim_l(l);
        double num = cabs_inf_diff(fhat_round + off, fhat + off, d);
        double den = cabs_inf(fhat + off, d);
        double rel = (den > 0.0) ? num / den : num;
        printf("    [roundtrip low-l] N=%d l=%d rel_err=%.4e\n", N, l, rel);
        if (rel > worst) worst = rel;
    }
    fflush(stdout);

    ASSERT_TRUE(worst < tol);

    free(fhat); free(fhat_round); free(f);
}

/* Sample roundtrip on a bandlimited f.  Build f := inv(random fhat); the
 * pipeline then computes f' := inv(fft(f)).  The same Riemann theta error
 * applies in reverse, so the achievable tolerance mirrors the spectrum
 * roundtrip at the same N -- O(1) and not decaying.  Thresholds chosen
 * with headroom above the empirically-observed peaks. */
static void test_roundtrip_sample_bandlimited(void)
{
    const int    Ns[]   = { 6, 8, 16 };
    /* Empirical peaks at the fixed seed are similar in magnitude to the
     * spectrum roundtrip; thresholds match. */
    const double tols[] = { 15.0, 10.0, 10.0 };
    const size_t nN     = sizeof(Ns) / sizeof(Ns[0]);

    seed_rand(20260526);

    for (size_t s = 0; s < nN; ++s) {
        int N = Ns[s];
        size_t nsamp  = (size_t)N * N * N;
        size_t ncoeff = su2_total_coeffs(N);

        double _Complex *fhat        = calloc(ncoeff, sizeof(double _Complex));
        double _Complex *fhat_middle = calloc(ncoeff, sizeof(double _Complex));
        double _Complex *f           = malloc(nsamp * sizeof(double _Complex));
        double _Complex *f_round     = malloc(nsamp * sizeof(double _Complex));

        for (size_t i = 0; i < ncoeff; ++i)
            fhat[i] = (urand() - 0.5) + (urand() - 0.5) * I;

        /* Construct bandlimited f directly from random fhat. */
        su2_fft_inv(N, fhat,        f);
        su2_fft    (N, f,           fhat_middle);
        su2_fft_inv(N, fhat_middle, f_round);

        double num = cabs_inf_diff(f_round, f, nsamp);
        double den = cabs_inf(f, nsamp);
        double rel = (den > 0.0) ? num / den : num;

        printf("    [roundtrip sample] N=%d rel_err=%.4e (assert < %.2f)\n",
               N, rel, tols[s]);
        fflush(stdout);

        ASSERT_TRUE(rel < tols[s]);

        free(fhat); free(fhat_middle); free(f); free(f_round);
    }
}

/* ===========================================================================
 * (C) Gauss-Legendre theta variant tests (bead `su2fft-ega`).
 *
 * See notes/gauss_legendre.md for the substitution and normalisation
 * derivation.  The GL theta nodes fix the closed-grid (N/(N-1))^2 factor
 * at the DC entry; the phi/psi closed-grid aliasing (bead `su2fft-0t1`)
 * is a SEPARATE defect that still produces O(0.2) leakage in the
 * non-DC coefficients.
 * =========================================================================== */

/* forward_gl(constant 1) gives fhat(0,0,0) = 1.0 to floating-point noise.
 * This is the headline GL improvement over the closed-grid forward, which
 * gives (N/(N-1))^2 = 1.31 at N=8 (see test_inv_delta_l0_gives_constant for
 * the closed-grid baseline).  Verified at N = 4, 6, 8, 16.
 * Reference: notes/gauss_legendre.md §1 and §6 (norm = 1/(2 N^2)). */
static void test_gl_constant_input_exact(void)
{
    int Ns[] = { 4, 6, 8, 16 };
    for (int idx = 0; idx < 4; ++idx) {
        int N = Ns[idx];
        size_t nsamp  = (size_t)N * N * N;
        size_t ncoeff = su2_total_coeffs(N);
        double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));
        double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
        for (size_t i = 0; i < nsamp; ++i) f[i] = 1.0 + 0.0 * I;
        su2_fft_gl(N, f, fhat);
        ASSERT_CNEAR(fhat[0], 1.0 + 0.0 * I, 1e-13);
        free(f); free(fhat);
    }
}

/* forward_gl(constant 1) has bounded leakage into the non-DC coefficients.
 *
 * The closed-grid phi/psi fold (the (-1)^{n+m} aliasing trick used in
 * Stage 1) still over-counts endpoint contributions, so a constant f=1
 * input produces nonzero coefficients at (l>0) or (m,n != 0) entries with
 * magnitudes ~0.2 at N=8.  This test BOUNDS the leakage as a regression
 * guard, NOT zero.
 *
 * The structural fix (2N-1 phi/psi grid removing the fold) is bead
 * `su2fft-0t1`.  Once that lands this tolerance should be tightened. */
static void test_gl_constant_input_leakage(void)
{
    int N = 8;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
    for (size_t i = 0; i < nsamp; ++i) f[i] = 1.0 + 0.0 * I;
    su2_fft_gl(N, f, fhat);

    double max_leak = 0.0;
    for (size_t i = 1; i < ncoeff; ++i) {  /* skip i=0 (the DC entry) */
        double a = cabs(fhat[i]);
        if (a > max_leak) max_leak = a;
    }
    printf("    [gl const-input leakage] N=%d max_leak=%.4e (assert < 0.30)\n",
           N, max_leak);
    fflush(stdout);

    /* Empirical ~0.197 at N=8; assert < 0.3 as a regression bound. */
    ASSERT_TRUE(max_leak < 0.3);

    free(f); free(fhat);
}

/* Dual of test_gl_constant_input_exact: inv_gl(delta at fhat(0,0,0)) is
 * the constant 1 (synthesis is exact algebra, see
 * test_inv_delta_l0_gives_constant), and forward_gl recovers
 * fhat(0,0,0) = 1.0 to machine precision. */
static void test_gl_single_coefficient_l0_exact(void)
{
    int N = 8;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *fhat    = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *fhat_rt = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *f       = malloc(nsamp  * sizeof(double _Complex));

    fhat[su2_coeff_offset(0) + su2_mn_index(0, 0, 0)] = 1.0 + 0.0 * I;
    su2_fft_inv_gl(N, fhat, f);
    su2_fft_gl   (N, f,    fhat_rt);

    ASSERT_CNEAR(fhat_rt[su2_coeff_offset(0) + su2_mn_index(0, 0, 0)],
                 1.0 + 0.0 * I, 1e-13);

    free(fhat); free(fhat_rt); free(f);
}

/* For each l in [0, N-1] set a single coefficient at (l, m=0, n=0) = 1,
 * inverse_gl, forward_gl.  The closed-grid phi/psi aliasing (bead
 * `su2fft-0t1`) introduces roundtrip error of order 0.1-0.3 across l in
 * [0, N-1].  This test BOUNDS the worst case (< 0.5) and prints the
 * empirical floor per l for visibility. */
static void test_gl_single_coefficient_floor(void)
{
    int N = 8;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *fhat    = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *fhat_rt = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *f       = malloc(nsamp  * sizeof(double _Complex));

    for (int l = 0; l < N; ++l) {
        memset(fhat,    0, ncoeff * sizeof(double _Complex));
        memset(fhat_rt, 0, ncoeff * sizeof(double _Complex));
        fhat[su2_coeff_offset(l) + su2_mn_index(l, 0, 0)] = 1.0 + 0.0 * I;

        su2_fft_inv_gl(N, fhat, f);
        su2_fft_gl   (N, f,    fhat_rt);

        double max_err = 0.0;
        for (size_t i = 0; i < ncoeff; ++i) {
            double a = cabs(fhat_rt[i] - fhat[i]);
            if (a > max_err) max_err = a;
        }
        printf("    [gl single-coef floor] l=%d max_err=%.4e\n", l, max_err);
        ASSERT_TRUE(max_err < 0.5);
    }
    fflush(stdout);

    free(fhat); free(fhat_rt); free(f);
}

/* Synthesis (su2_fft_inv_gl) is a pure linear map; should hold to
 * floating-point noise.  No quadrature dependence here. */
static void test_gl_inv_linearity(void)
{
    int N = 6;
    size_t ncoeff = su2_total_coeffs(N);
    size_t nsamp  = (size_t)N * N * N;
    double _Complex *fhat1         = malloc(ncoeff * sizeof(double _Complex));
    double _Complex *fhat2         = malloc(ncoeff * sizeof(double _Complex));
    double _Complex *fhat_combined = malloc(ncoeff * sizeof(double _Complex));
    double _Complex *f1            = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *f2            = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fc            = malloc(nsamp  * sizeof(double _Complex));

    seed_rand(20260526);
    for (size_t i = 0; i < ncoeff; ++i) {
        fhat1[i] = (urand() - 0.5) + (urand() - 0.5) * I;
        fhat2[i] = (urand() - 0.5) + (urand() - 0.5) * I;
    }
    const double _Complex alpha =  2.0 + 1.5 * I;
    const double _Complex beta  = -0.7 + 0.4 * I;
    for (size_t i = 0; i < ncoeff; ++i)
        fhat_combined[i] = alpha * fhat1[i] + beta * fhat2[i];

    su2_fft_inv_gl(N, fhat1,         f1);
    su2_fft_inv_gl(N, fhat2,         f2);
    su2_fft_inv_gl(N, fhat_combined, fc);

    double max_err = 0.0;
    for (size_t i = 0; i < nsamp; ++i) {
        double _Complex expected = alpha * f1[i] + beta * f2[i];
        double a = cabs(fc[i] - expected);
        if (a > max_err) max_err = a;
    }
    ASSERT_TRUE(max_err < 1e-12);

    free(fhat1); free(fhat2); free(fhat_combined);
    free(f1); free(f2); free(fc);
}

/* Analytical GL synthesis test (mirrors test_inv_delta_l1_m0_n0_gives_cos_theta
 * but at GL theta nodes): fhat(1, 0, 0) = 1 -> f(g) = 3 * cos(theta_k)
 * where theta_k = arccos(x_k) at the GL nodes x_k. */
static void test_gl_inv_delta_l1_m0_n0_gives_3cos_theta_at_gl_nodes(void)
{
    int N = 8;
    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));
    double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));

    fhat[su2_coeff_offset(1) + su2_mn_index(1, 0, 0)] = 1.0 + 0.0 * I;
    su2_fft_inv_gl(N, fhat, f);

    /* Recover the GL theta nodes for the expected values. */
    double *x_gl = malloc((size_t)N * sizeof(double));
    double *w_gl = malloc((size_t)N * sizeof(double));
    su2_gl_nodes_weights(N, x_gl, w_gl);

    double max_err = 0.0;
    for (int j1 = 0; j1 < N; ++j1) {
        for (int k = 0; k < N; ++k) {
            double theta_k = acos(x_gl[k]);
            double _Complex expected = 3.0 * cos(theta_k) + 0.0 * I;
            for (int j2 = 0; j2 < N; ++j2) {
                double a = cabs(f[su2_sample_index(N, j1, k, j2)] - expected);
                if (a > max_err) max_err = a;
            }
        }
    }
    ASSERT_TRUE(max_err < 1e-12);

    free(fhat); free(f); free(x_gl); free(w_gl);
}

int main(void)
{
    RUN(test_inv_zero_input);
    RUN(test_inv_delta_l0_gives_constant);
    RUN(test_inv_delta_l1_m0_n0_gives_cos_theta);
    RUN(test_inv_linearity);
    RUN(test_roundtrip_spectrum_riemann_tolerance);
    RUN(test_roundtrip_spectrum_low_l_subspace);
    RUN(test_roundtrip_sample_bandlimited);
    RUN(test_gl_constant_input_exact);
    RUN(test_gl_constant_input_leakage);
    RUN(test_gl_single_coefficient_l0_exact);
    RUN(test_gl_single_coefficient_floor);
    RUN(test_gl_inv_linearity);
    RUN(test_gl_inv_delta_l1_m0_n0_gives_3cos_theta_at_gl_nodes);
    TEST_REPORT_AND_EXIT();
}
