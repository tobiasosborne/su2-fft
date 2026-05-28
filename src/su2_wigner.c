/* su2_wigner.c -- Stable closed-form P^l_{n,m}(cos theta) via the
 * Wigner sum (de Moivre form) plus the paper's phase factor.
 *
 * Derivation:
 *   - The Rodrigues form in paper line 537 is mathematically correct but
 *     unstable: the binomial expansion of (1-x)^{l-m}(1+x)^{l+m} produces
 *     coefficients of order C(l,j)*p!/(p-d)! ~ 10^{O(l)}, which catastrophically
 *     cancel under Horner evaluation by l ~ 10.
 *   - Plugging the standard Jacobi Rodrigues identity
 *       D^k[(1-x)^{k+a}(1+x)^{k+b}] = (-2)^k k! (1-x)^a (1+x)^b P_k^{(a,b)}(x)
 *     into the paper formula and converting to half-angle form yields
 *       paper P^l_{n,m}(cos beta) = i^{m-n} * d^l_{n,m}_phys(beta)
 *     where d^l_{n,m}_phys is the Sakurai-convention real Wigner small-d.
 *     The trailing scalar factors of (-1) and 2^k all collapse on careful
 *     algebra.  Verified against the Rodrigues form for l = 1, 2 in
 *     test_wigner.c (which continues to test all the original analytical
 *     identities).
 *   - The Wigner sum
 *       d^l_{n,m}(beta) = sum_t (-1)^{n-m+t} *
 *                        sqrt((l+n)!(l-n)!(l+m)!(l-m)!) /
 *                        [(l+m-t)! t! (n-m+t)! (l-n-t)!]  *
 *                        cos(beta/2)^{2l+m-n-2t} *
 *                        sin(beta/2)^{n-m+2t}
 *     has only O(l) terms.  The per-term factorial ratio is O(1) overall, but
 *     forming numerator and denominator separately overflows the double range
 *     at 171!; the alternating sum also cancels catastrophically across O(l)
 *     terms above l ~ 50 (bead su2fft-258).  Both are cured here:
 *       (a) wigner_d_phys forms each term coefficient by a BALANCED INCREMENTAL
 *           PRODUCT that never overflows -- correct for a 1-2-term seed at any l;
 *       (b) the public su2_wigner_d routes through su2_wigner_d_seq, the stable
 *           ascending-l recurrence, so the many-term cancelling sum is never
 *           evaluated above l_min+1.  Stable in double precision for all l up to
 *           the recurrence range (tested at l = 60, 75, 80 against the arb path).
 *
 * Cost: O(l) per call.  Used by both su2_ft_direct (O(N^6) reference) and the
 * Stage-2 dot products of su2_fft (O(N^4) algorithm).
 */
#include "su2.h"

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>

static double _Complex pow_i(int k)
{
    int r = ((k % 4) + 4) % 4;
    switch (r) {
        case 0:  return  1.0 + 0.0*I;
        case 1:  return  0.0 + 1.0*I;
        case 2:  return -1.0 + 0.0*I;
        default: return  0.0 - 1.0*I;
    }
}

/* Repeated-squaring integer power: O(log2 k) mults, no libm calls.
 * pow(0, 0) = 1.0 convention preserved by initialising result to 1.0
 * (consistent with the theta=0 / theta=pi endpoint behaviour that
 * existing tests rely on; see HANDOFF.md §2 item 4). */
static inline double ipow(double x, int k)
{
    double r = 1.0;
    while (k > 0) {
        if (k & 1) r *= x;
        x *= x;
        k >>= 1;
    }
    return r;
}

/* Overflow-free, full-precision coefficient of one de Moivre term.
 *
 * paper.tex:537 -- the de Moivre sum has per-term factorial coefficient
 *   coeff(t) = sqrt[(l+n)!(l-n)!(l+m)!(l-m)!]
 *            / [(l+m-t)! t! (n-m+t)! (l-n-t)!].
 * Numerator and denominator have equal total degree (numerator product is
 * (l+n)+(l-n)+(l+m)+(l-m)=4l integers; the squared denominator is
 * 2*[(l+m-t)+t+(n-m+t)+(l-n-t)]=4l integers), so coeff is O(1) but overflows
 * the double range (171!) if either side is formed alone.  Compute
 *   R = coeff^2 = num! / (den!)^2
 * as a SINGLE running product, interleaving multiplies by numerator factors
 * and divides by denominator factors so the running value r stays near 1.0 and
 * never overflows.  coeff = sqrt(R).  This is the single-term-at-l_min seed;
 * the balanced product avoids the factorial overflow AND the alternating-sum
 * cancellation that broke double precision above l~50 (bead su2fft-258).
 *
 * num_max / den_max are the largest factors of 2..num_max (numerator multiset)
 * and 2..den_max (denominator multiset, each contributing TWO copies to R). */
static double demoivre_coeff(int l, int n, int m, int t)
{
    /* Numerator multiset: integers 2..(l+n), 2..(l-n), 2..(l+m), 2..(l-m). */
    int num_max[4] = { l + n, l - n, l + m, l - m };
    /* Denominator multiset (each factor squared in R): 2..(l+m-t), 2..t,
     * 2..(n-m+t), 2..(l-n-t). */
    int den_max[4] = { l + m - t, t, n - m + t, l - n - t };

    /* Per-group cursors; cursor[g] is the next factor to consume from group g. */
    int num_cur[4] = { 2, 2, 2, 2 };
    int den_cur[4] = { 2, 2, 2, 2 };

    double r = 1.0;
    int progress = 1;
    while (progress) {
        progress = 0;
        if (r <= 1.0) {
            /* Prefer multiplying by a numerator factor; else divide. */
            int did = 0;
            for (int g = 0; g < 4 && !did; ++g) {
                if (num_cur[g] <= num_max[g]) { r *= (double)num_cur[g]++; did = 1; }
            }
            if (!did) {
                for (int g = 0; g < 4 && !did; ++g) {
                    /* Two copies per denominator factor (R has den squared). */
                    if (den_cur[g] <= den_max[g]) {
                        r /= (double)den_cur[g];
                        r /= (double)den_cur[g];
                        ++den_cur[g];
                        did = 1;
                    }
                }
            }
            progress = did;
        } else {
            int did = 0;
            for (int g = 0; g < 4 && !did; ++g) {
                if (den_cur[g] <= den_max[g]) {
                    r /= (double)den_cur[g];
                    r /= (double)den_cur[g];
                    ++den_cur[g];
                    did = 1;
                }
            }
            if (!did) {
                for (int g = 0; g < 4 && !did; ++g) {
                    if (num_cur[g] <= num_max[g]) { r *= (double)num_cur[g]++; did = 1; }
                }
            }
            progress = did;
        }
    }
    return sqrt(r);
}

/* Physics-convention real Wigner small-d d^l_{n,m}(beta) (Sakurai). */
static double wigner_d_phys(int l, int n, int m, double beta)
{
    int tmin = (m - n > 0) ? (m - n) : 0;
    int tmax = (l + m < l - n) ? (l + m) : (l - n);
    if (tmin > tmax) return 0.0;

    double c2 = cos(beta * 0.5);
    double s2 = sin(beta * 0.5);

    /* Integer powers via repeated squaring (bead su2fft-dyi).  Replaces
     * two libm pow() calls per term with O(log2(2l)) mults each.  The
     * post-m21 calling pattern is seeds only (l = l_min, l_min+1), where
     * the t-range collapses to 1-2 terms — Horner tables over-build by
     * ~20x, so inline ipow is the right primitive. */
    double sum = 0.0;
    for (int t = tmin; t <= tmax; ++t) {
        double sign  = ((n - m + t) & 1) ? -1.0 : 1.0;
        double coeff = demoivre_coeff(l, n, m, t);
        int    pc    = 2*l + m - n - 2*t;
        int    ps    = n - m + 2*t;
        sum += sign * coeff * ipow(c2, pc) * ipow(s2, ps);
    }
    return sum;
}

/**
 * @brief Evaluate the paper's matrix coefficient P^l_{n,m}(cos theta).
 *
 * Computes P^l_{n,m}(cos theta) = i^{m-n} * d^l_{n,m}(theta), where
 * d^l_{n,m} is the Sakurai-convention real Wigner small-d function.  The
 * i^{m-n} phase factor converts from Sakurai's convention to the paper's
 * normalisation (paper.tex line 537, with the algebraic simplification
 * documented in the file header comment).
 *
 * d^l_{n,m} is obtained from su2_wigner_d_seq, the stable ascending-l
 * recurrence: it is seeded by 1-2-term de Moivre evaluations at l_min and
 * l_min+1 (overflow-free via demoivre_coeff) and advanced cancellation-free to
 * the target l.  This is uniformly stable -- the many-term alternating de
 * Moivre sum, which cancels catastrophically above l~50 (bead su2fft-258), is
 * never evaluated at high l.  Reproduces low-l values to machine precision.
 *
 * @param[in] l  Degree, l >= 0.
 * @param[in] n  Row index, -l <= n <= l.
 * @param[in] m  Column index, -l <= m <= l.
 * @param[in] theta  Polar Euler angle in radians, theta in [0, pi].
 * @return Complex value P^l_{n,m}(cos theta); returns 0 if |n| > l or |m| > l.
 * @par Complexity O(l) per call.
 * @par Reference paper.tex lines 537, 542; ALGORITHM.md Section 4.
 */
double _Complex su2_wigner_d(int l, int n, int m, double theta)
{
    if (abs(n) > l || abs(m) > l) return 0.0 + 0.0*I;
    if (theta == 0.0) return (n == m) ? (1.0 + 0.0*I) : (0.0 + 0.0*I);

    /* Route through the stable ascending-l recurrence (cancellation-free for
     * all l; the de Moivre sum is used only for the 1-2-term seeds). */
    int l_min = (abs(n) > abs(m)) ? abs(n) : abs(m);
    double d_seq[l - l_min + 1];
    su2_wigner_d_seq(l_min, l, n, m, theta, d_seq);
    double d = d_seq[l - l_min];
    return pow_i(m - n) * d;
}

/* Half-integer-compatible Sakurai d^l_{n,m}(beta) via tgamma() instead of the
 * fact() table.  Arguments encoded as 2l, 2n, 2m (integers); physical
 * (l, n, m) = (two_l/2, two_n/2, two_m/2).  For integer l (two_l even) this
 * reproduces wigner_d_phys up to tgamma-vs-factorial-table FP noise; for
 * half-integer l (two_l odd) this enables spin-1/2, spin-3/2, ... matrix
 * elements.
 *
 * The de Moivre sum (paper.tex line 537) generalises directly: factorials
 * become Gammas via Gamma(k+1) = k!.  Because 2l, 2n, 2m share parity, the
 * differences (m-n), (l+m), (l-n) are integers; therefore the summation
 * index t still steps by 1 and the t-range is unchanged.  The cos/sin
 * exponents pc = 2l + m - n - 2t = two_l + (m-n) - 2t and ps = n - m + 2t =
 * -(m-n) + 2t are nonnegative integers across the valid t range.
 *
 * See notes/half_integer.md §1, paper.tex line 537. */
static double wigner_d_phys_half(int two_l, int two_n, int two_m, double beta)
{
    /* Physical l, n, m -- may be half-integer. */
    double l  = 0.5 * (double)two_l;
    double nv = 0.5 * (double)two_n;
    double mv = 0.5 * (double)two_m;

    /* Integer offsets (m-n), (l+m), (l-n) via the 2x-encoding. */
    int dnm = (two_m - two_n) / 2;             /* m - n */
    int lpm = (two_l + two_m) / 2;             /* l + m */
    int lmn = (two_l - two_n) / 2;             /* l - n */
    int tmin = (dnm > 0) ? dnm : 0;
    int tmax = (lpm < lmn) ? lpm : lmn;
    if (tmin > tmax) return 0.0;

    double c2 = cos(beta * 0.5);
    double s2 = sin(beta * 0.5);

    /* sqrt(Gamma(l+n+1) Gamma(l-n+1) Gamma(l+m+1) Gamma(l-m+1)) */
    double norm = sqrt(tgamma(l + nv + 1.0) * tgamma(l - nv + 1.0)
                     * tgamma(l + mv + 1.0) * tgamma(l - mv + 1.0));

    double sum = 0.0;
    for (int t = tmin; t <= tmax; ++t) {
        /* sign = (-1)^{n - m + t} = (-1)^{-dnm + t}; -dnm + t mod 2 is the
         * same as (t - dnm) mod 2, which we compute via the integer trick
         * (((-dnm + t) & 1) ? -1 : 1).  Bit-and on a possibly-negative int
         * is implementation-defined for two's complement -- be explicit. */
        int  s_int = ((t - dnm) % 2 + 2) % 2;
        double sign = s_int ? -1.0 : 1.0;

        /* denom = Gamma(l+m-t+1) * Gamma(t+1) * Gamma(n-m+t+1) * Gamma(l-n-t+1)
         * In integer-shift form: Gamma((lpm - t) + 1), Gamma(t + 1),
         * Gamma((-dnm + t) + 1), Gamma((lmn - t) + 1). */
        double denom = tgamma((double)(lpm - t) + 1.0)
                     * tgamma((double)t + 1.0)
                     * tgamma((double)(t - dnm) + 1.0)
                     * tgamma((double)(lmn - t) + 1.0);

        /* Exponents are nonnegative integers across the valid t range:
         *   pc = 2l + m - n - 2t = two_l + dnm - 2t
         *   ps = n - m + 2t      = -dnm + 2t
         * Verified by mental check at l=1/2, m=n=1/2: dnm=0, t=0,
         * pc = 1, ps = 0 -- d^{1/2}_{1/2,1/2} = cos(theta/2). */
        int pc = two_l + dnm - 2 * t;
        int ps = -dnm + 2 * t;

        sum += sign * norm / denom * ipow(c2, pc) * ipow(s2, ps);
    }
    return sum;
}

/**
 * @brief Half-integer-compatible matrix coefficient P^l_{n,m}(cos theta).
 *
 * Computes the paper's matrix element with l, n, m encoded as integers
 * 2l, 2n, 2m so that half-integer spins (two_l odd) are representable
 * without floating-point comparison.  Physical (l, n, m) = (two_l, two_n,
 * two_m) / 2; all three must share parity (all even = integer l, all odd
 * = half-integer l).  For integer l (two_l even) the result equals
 * su2_wigner_d(l, n, m, theta) up to tgamma-vs-factorial-table FP noise.
 *
 * This is Tier 1 of bead `su2fft-n8e`: Wigner-d evaluation only.  The
 * half-integer FFT itself (forward/inverse on a 4pi-period phi/psi grid)
 * is deferred to a follow-up bead (see notes/half_integer.md §2-4).
 *
 * @param[in] two_l  Twice l; must be >= 0.
 * @param[in] two_n  Twice n; must satisfy |two_n| <= two_l, same parity as two_l.
 * @param[in] two_m  Twice m; must satisfy |two_m| <= two_l, same parity as two_l.
 * @param[in] theta  Polar angle in radians.
 * @return Complex value P^l_{n,m}(cos theta); 0 if |n| > l or |m| > l.
 * @par Reference paper.tex line 537; notes/half_integer.md.
 */
double _Complex su2_wigner_d_half(int two_l, int two_n, int two_m, double theta)
{
    assert(two_l >= 0);
    /* Same-parity invariant (CLAUDE.md rule 5 -- fail fast). */
    assert((two_l & 1) == (two_n & 1));
    assert((two_l & 1) == (two_m & 1));

    if (abs(two_n) > two_l || abs(two_m) > two_l) return 0.0 + 0.0*I;

    /* d^l_{n,m}(0) = delta_{n,m} at any spin (rotation by 0 is identity). */
    if (theta == 0.0) return (two_n == two_m) ? (1.0 + 0.0*I) : (0.0 + 0.0*I);

    double d = wigner_d_phys_half(two_l, two_n, two_m, theta);
    /* m - n is an integer (both shift by 1/2 together), so pow_i is well-defined. */
    int dmn = (two_m - two_n) / 2;
    return pow_i(dmn) * d;
}

/**
 * @brief Fill out_d with d^l_{n,m}(theta) for l = l_min .. l_max via the
 *        Jacobi-lifted three-term ascending-l recurrence.
 *
 * Spec: notes/wigner_recurrence.md §2b (recurrence) and §7 (implementation
 * sketch).  The recurrence is the standard Jacobi P_k^{(a,b)} three-term
 * recurrence (DLMF 18.9.1/18.9.2) lifted to d^l_{n,m} via the l-dependent
 * normalisation R(l) = sqrt[(l+M)!(l-M)! / ((l+N)!(l-N)!)] from the
 * Jacobi connection (Wikipedia "Wigner D-matrix", §"Relation to Jacobi
 * polynomials"; verified against paper.tex line 739 `eq:P_l` in the
 * m=n=0 special case, see notes/wigner_recurrence.md §2b).
 *
 *   d^{l+1}_{n,m}(theta) = (A_k cos(theta) + B_k) * F1 * d^l_{n,m}(theta)
 *                        - C_k * F2 * d^{l-1}_{n,m}(theta)
 *
 * with k = l - M, s = 2k + a + b, M = max(|n|,|m|), Nmin = min(|n|,|m|),
 * a = |m - n|, b = |m + n|, and
 *
 *   A_k = (s+1)(s+2) / [2(k+1)(k+a+b+1)]
 *   B_k = (a^2 - b^2)(s+1) / [2(k+1)(k+a+b+1) s]
 *   C_k = (k+a)(k+b)(s+2) / [(k+1)(k+a+b+1) s]
 *   F1  = sqrt[(l+1+M)(l+1-M) / ((l+1+Nmin)(l+1-Nmin))]
 *   F2  = F1 * sqrt[(l+M)(l-M) / ((l+Nmin)(l-Nmin))]
 *
 * Seeded at l = l_min and (when l_max > l_min) l = l_min + 1 by direct
 * evaluation via wigner_d_phys.  By construction the first recurrence
 * step uses k = (l_min + 1) - M = 1 (when l_min = M) or larger, so
 * s = 2k + a + b >= 2 and the B_k, C_k denominators are nonzero; the
 * code asserts this invariant.
 *
 * @par Complexity O(l_max - l_min) flops after two O(l_min) seeds.
 */
void su2_wigner_d_seq(int l_min, int l_max, int n, int m, double theta,
                      double *out_d)
{
    /* Fail fast on out-of-range inputs (CLAUDE.md rule 5). */
    assert(out_d != NULL);
    assert(l_min >= 0);
    assert(l_max >= l_min);
    int an = abs(n), am = abs(m);
    int M    = (an > am) ? an : am;
    int Nmin = (an < am) ? an : am;
    assert(l_min >= M);

    /* Seed l = l_min. */
    out_d[0] = wigner_d_phys(l_min, n, m, theta);
    if (l_max == l_min) return;

    /* Seed l = l_min + 1. */
    out_d[1] = wigner_d_phys(l_min + 1, n, m, theta);

    int a = abs(m - n);
    int b = abs(m + n);
    double x = cos(theta);

    /* Step l -> l+1 for l = l_min+1 .. l_max-1.  notes/wigner_recurrence.md §2b. */
    for (int l = l_min + 1; l <= l_max - 1; ++l) {
        int k = l - M;
        double s = (double)(2*k + a + b);
        /* s = 0 iff k = 0 AND a = b = 0; the loop starts at k = l_min+1-M >= 1
         * so s >= 2.  Verify the invariant. */
        assert(s != 0.0);

        double kp1  = (double)(k + 1);
        double kab1 = (double)(k + a + b + 1);
        double denom_k = 2.0 * kp1 * kab1;

        double Ak = (s + 1.0) * (s + 2.0) / denom_k;
        double Bk = (double)(a*a - b*b) * (s + 1.0) / (denom_k * s);
        double Ck = (double)(k + a) * (double)(k + b) * (s + 2.0)
                  / (kp1 * kab1 * s);

        /* F1 = R(l+1)/R(l);  F2 = R(l+1)/R(l-1).  See notes §2b. */
        double F1num = (double)((l + 1 + M) * (l + 1 - M));
        double F1den = (double)((l + 1 + Nmin) * (l + 1 - Nmin));
        double F1    = sqrt(F1num / F1den);
        double F2num = (double)((l + M) * (l - M));
        double F2den = (double)((l + Nmin) * (l - Nmin));
        double F2    = F1 * sqrt(F2num / F2den);

        int i = l - l_min;        /* d^l lives at out_d[i], d^{l-1} at out_d[i-1] */
        double d_l    = out_d[i];
        double d_lm1  = out_d[i - 1];
        double d_lp1  = (Ak * x + Bk) * F1 * d_l - Ck * F2 * d_lm1;
        out_d[i + 1]  = d_lp1;
    }
}
