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

/* Physics-convention real Wigner small-d d^l_{n,m}(beta) (Sakurai). */
static double wigner_d_phys(int l, int n, int m, double beta)
{
    int tmin = (m - n > 0) ? (m - n) : 0;
    int tmax = (l + m < l - n) ? (l + m) : (l - n);
    if (tmin > tmax) return 0.0;

    double c2 = cos(beta * 0.5);
    double s2 = sin(beta * 0.5);
    double norm = sqrt(fact(l + n) * fact(l - n) * fact(l + m) * fact(l - m));

    double sum = 0.0;
    for (int t = tmin; t <= tmax; ++t) {
        double sign  = ((n - m + t) & 1) ? -1.0 : 1.0;
        double denom = fact(l + m - t) * fact(t)
                     * fact(n - m + t) * fact(l - n - t);
        int    pc    = 2*l + m - n - 2*t;
        int    ps    = n - m + 2*t;
        sum += sign * norm / denom * pow(c2, (double)pc) * pow(s2, (double)ps);
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
