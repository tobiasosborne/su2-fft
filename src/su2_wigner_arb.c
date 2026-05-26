/* su2_wigner_arb.c -- Stable arbitrary-precision P^l_{n,m}(cos theta).
 *
 * Direct port of the de Moivre sum in src/su2_wigner.c into arb arithmetic.
 *   P^l_{n,m}(cos beta) = i^{m-n} * d^l_{n,m}_phys(beta)
 *   d^l_{n,m}_phys(beta) = sum_t (-1)^{n-m+t} *
 *                          sqrt((l+n)!(l-n)!(l+m)!(l-m)!) /
 *                          [(l+m-t)! t! (n-m+t)! (l-n-t)!] *
 *                          cos(beta/2)^{2l+m-n-2t} *
 *                          sin(beta/2)^{n-m+2t}
 *
 * Every arithmetic op carries a precision argument.  Inputs are validated
 * trivially (range check on n, m).
 */
#include "su2_arb.h"

#include <flint/arb.h>
#include <flint/acb.h>
#include <stdlib.h>

/* out = i^k for k mod 4 in {0,1,2,3}.  Real cases write the imaginary half
 * to zero; imaginary cases write the real half to zero. */
static void acb_set_pow_i(acb_t out, int k, const arb_t magnitude)
{
    int r = ((k % 4) + 4) % 4;
    arb_zero(acb_realref(out));
    arb_zero(acb_imagref(out));
    switch (r) {
        case 0: arb_set(acb_realref(out), magnitude); break;
        case 1: arb_set(acb_imagref(out), magnitude); break;
        case 2: arb_neg(acb_realref(out), magnitude); break;
        case 3: arb_neg(acb_imagref(out), magnitude); break;
    }
}

/**
 * @brief Arbitrary-precision evaluation of P^l_{n,m}(cos theta) via FLINT arb.
 *
 * Direct port of su2_wigner_d() into arb arithmetic: computes the paper's
 * matrix coefficient P^l_{n,m}(cos theta) = i^{m-n} * d^l_{n,m}(theta) using
 * the de Moivre sum with every operation carried at `prec` bits.  Each
 * coefficient is an interval (acb_t ball), providing certified error bounds.
 * Stable for all l representable at the given precision.
 *
 * @param[out] out    Result P^l_{n,m}(cos theta) as an acb_t ball.  Set to
 *                    exact zero if |n| > l or |m| > l.
 * @param[in]  l      Degree, l >= 0.
 * @param[in]  n      Row index, -l <= n <= l.
 * @param[in]  m      Column index, -l <= m <= l.
 * @param[in]  theta  Polar angle as arb_t, theta in [0, pi] (radians).
 * @param[in]  prec   Working precision in bits (e.g. 53 = IEEE double).
 * @par Complexity O(l) arb operations, each O(prec / 64) limb ops.
 * @par Reference paper.tex lines 537, 542; ALGORITHM.md Section 4.
 */
void su2_wigner_d_arb(acb_t out, int l, int n, int m,
                      const arb_t theta, slong prec)
{
    if (abs(n) > l || abs(m) > l) { acb_zero(out); return; }

    int tmin = (m - n > 0) ? (m - n) : 0;
    int tmax = (l + m < l - n) ? (l + m) : (l - n);
    if (tmin > tmax) { acb_zero(out); return; }

    arb_t half_theta, c2, s2;
    arb_init(half_theta); arb_init(c2); arb_init(s2);
    arb_mul_2exp_si(half_theta, theta, -1);    /* theta / 2 */
    arb_cos(c2, half_theta, prec);
    arb_sin(s2, half_theta, prec);

    /* norm = sqrt((l+n)!(l-n)!(l+m)!(l-m)!) */
    arb_t norm, tmp;
    arb_init(norm); arb_init(tmp);
    arb_fac_ui(norm, (ulong)(l + n), prec);
    arb_fac_ui(tmp,  (ulong)(l - n), prec); arb_mul(norm, norm, tmp, prec);
    arb_fac_ui(tmp,  (ulong)(l + m), prec); arb_mul(norm, norm, tmp, prec);
    arb_fac_ui(tmp,  (ulong)(l - m), prec); arb_mul(norm, norm, tmp, prec);
    arb_sqrt(norm, norm, prec);

    arb_t sum, term, denom, cpow, spow;
    arb_init(sum); arb_init(term); arb_init(denom);
    arb_init(cpow); arb_init(spow);
    arb_zero(sum);

    for (int t = tmin; t <= tmax; ++t) {
        arb_fac_ui(denom, (ulong)(l + m - t), prec);
        arb_fac_ui(tmp,   (ulong)t,            prec); arb_mul(denom, denom, tmp, prec);
        arb_fac_ui(tmp,   (ulong)(n - m + t),  prec); arb_mul(denom, denom, tmp, prec);
        arb_fac_ui(tmp,   (ulong)(l - n - t),  prec); arb_mul(denom, denom, tmp, prec);

        int pc = 2*l + m - n - 2*t;
        int ps = n - m + 2*t;
        arb_pow_ui(cpow, c2, (ulong)pc, prec);
        arb_pow_ui(spow, s2, (ulong)ps, prec);

        arb_div(term, norm,  denom, prec);
        arb_mul(term, term,  cpow,  prec);
        arb_mul(term, term,  spow,  prec);

        if ((n - m + t) & 1) arb_neg(term, term);
        arb_add(sum, sum, term, prec);
    }

    /* out = i^{m-n} * sum */
    acb_set_pow_i(out, m - n, sum);

    arb_clear(half_theta); arb_clear(c2); arb_clear(s2);
    arb_clear(norm); arb_clear(tmp);
    arb_clear(sum); arb_clear(term); arb_clear(denom);
    arb_clear(cpow); arb_clear(spow);
}
