/* su2_arb.h -- Arbitrary-precision SU(2) Fourier transform via FLINT.
 *
 * Mirrors include/su2.h but uses FLINT's acb_t (complex ball) and arb_t
 * (real ball) types.  Every routine takes a precision argument `prec` in bits
 * (e.g. prec = 53 matches IEEE double; prec = 256 is "high precision").
 *
 * Layout: same as su2.h.  fhat[ su2_coeff_offset(l) + su2_mn_index(l,m,n) ]
 * is now an acb_t.
 */
#ifndef SU2_ARB_H
#define SU2_ARB_H

#include "su2.h"
#include <flint/acb.h>
#include <flint/arb.h>

/* Allocate / free a vector of acb_t initialised to zero (FLINT _acb_vec_*
 * idioms are exposed for convenience; users may call them directly). */

/* Paper's P^l_{n,m}(cos theta) at precision `prec`. */
void su2_wigner_d_arb(acb_t out, int l, int n, int m,
                      const arb_t theta, slong prec);

/* Direct O(N^6) Fourier transform.  f and fhat are FLINT acb_ptr arrays of
 * length N*N*N and su2_total_coeffs(N), respectively. */
void su2_ft_direct_arb(int N, acb_srcptr f, acb_ptr fhat, slong prec);

/* Fast O(N^4) Fourier transform via acb_dft_prod for Stage 1. */
void su2_fft_arb(int N, acb_srcptr f, acb_ptr fhat, slong prec);

#endif /* SU2_ARB_H */
