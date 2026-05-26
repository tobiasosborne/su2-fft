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
 *     has only O(l) terms, each with bounded factorial ratio.  Numerically
 *     stable up to l ~ 25 in double precision.
 *
 * Cost: O(l) per call.  Used by both su2_ft_direct (O(N^6) reference) and the
 * Stage-2 dot products of su2_fft (O(N^4) algorithm).
 */
#include "su2.h"

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>

#define FACT_MAX 100
static double fact(int n)
{
    static double tab[FACT_MAX + 1];
    static int    initd = 0;
    if (!initd) {
        tab[0] = 1.0;
        for (int i = 1; i <= FACT_MAX; ++i) tab[i] = tab[i - 1] * (double)i;
        initd = 1;
    }
    if (n < 0 || n > FACT_MAX) return 0.0;
    return tab[n];
}

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

/* Physics-convention real Wigner small-d d^l_{n,m}(beta) (Sakurai). */
static double wigner_d_phys(int l, int n, int m, double beta)
{
    assert(l <= FACT_MAX);

    int tmin = (m - n > 0) ? (m - n) : 0;
    int tmax = (l + m < l - n) ? (l + m) : (l - n);
    if (tmin > tmax) return 0.0;

    double c2 = cos(beta * 0.5);
    double s2 = sin(beta * 0.5);
    double norm = sqrt(fact(l + n) * fact(l - n) * fact(l + m) * fact(l - m));

    /* Integer powers via repeated squaring (bead su2fft-dyi).  Replaces
     * two libm pow() calls per term with O(log2(2l)) mults each.  The
     * post-m21 calling pattern is seeds only (l = l_min, l_min+1), where
     * the t-range collapses to 1-2 terms — Horner tables over-build by
     * ~20x, so inline ipow is the right primitive. */
    double sum = 0.0;
    for (int t = tmin; t <= tmax; ++t) {
        double sign  = ((n - m + t) & 1) ? -1.0 : 1.0;
        double denom = fact(l + m - t) * fact(t)
                     * fact(n - m + t) * fact(l - n - t);
        int    pc    = 2*l + m - n - 2*t;
        int    ps    = n - m + 2*t;
        sum += sign * norm / denom * ipow(c2, pc) * ipow(s2, ps);
    }
    return sum;
}

/**
 * @brief Evaluate the paper's matrix coefficient P^l_{n,m}(cos theta).
 *
 * Computes P^l_{n,m}(cos theta) = i^{m-n} * d^l_{n,m}(theta), where
 * d^l_{n,m} is the Sakurai-convention real Wigner small-d function evaluated
 * via the de Moivre sum (O(l) terms, bounded factorial ratios).  The i^{m-n}
 * phase factor converts from Sakurai's convention to the paper's normalisation
 * (paper.tex line 537, with the algebraic simplification documented in the
 * file header comment).  Stable in double precision for l up to roughly 25.
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

    double d = wigner_d_phys(l, n, m, theta);
    return pow_i(m - n) * d;
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
