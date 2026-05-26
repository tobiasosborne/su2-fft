/* su2_convolve.c -- Convolution on SU(2) via the Peter-Weyl spectrum.
 *
 * Convolution theorem on a compact group (Folland, "A Course in Abstract
 * Harmonic Analysis", §3.3; Sugiura, "Unitary Representations and Harmonic
 * Analysis" §IV.3): in the Peter-Weyl basis, convolution becomes a per-l
 * matrix product.
 *
 * Spec (bead su2fft-d7v):
 *   (f * g)hat(l)_{mn} = sum_{p} fhat(l)_{mp} * ghat(l)_{pn}
 *
 * Each degree-l block is a (2l+1) x (2l+1) complex matrix; convolve does
 * a matrix multiply per l.  Total cost O(sum_l (2l+1)^3) = O(N^4) -- same
 * order as the FFT itself.
 *
 * Useful for: smoothing on SO(3) (kernel = bandlimited Gaussian), template
 * matching on rotated 3D images, group-averaged neural-net layers.
 */
#include "su2.h"

#include <complex.h>
#include <stddef.h>
#include <stdlib.h>

/**
 * @brief Per-l matrix-product convolution in the Peter-Weyl spectrum.
 *
 * fghat(l)_{mn} = sum_p fhat(l)_{mp} * ghat(l)_{pn}.
 *
 * Storage convention: fhat[offset(l) + (m+l)*(2l+1) + (n+l)] -- row m,
 * column n, row-major within the (2l+1) x (2l+1) block.
 *
 * @param[in]  N      Bandlimit (l in [0, N-1]).
 * @param[in]  fhat   Length su2_total_coeffs(N) input coefficients.
 * @param[in]  ghat   Length su2_total_coeffs(N) input coefficients.
 * @param[out] fghat  Length su2_total_coeffs(N) output coefficients.
 *                    May alias fhat or ghat (writes block-by-block;
 *                    each l-block written after both inputs at that l
 *                    are read -- but a temporary is used to be safe).
 *
 * @par Reference notes/inverse_fft.md (for the synthesis), bead su2fft-d7v.
 * @par Complexity O(N^4).
 */
void su2_convolve(int N,
                  const double _Complex *fhat,
                  const double _Complex *ghat,
                  double _Complex *fghat)
{
    if (N < 1 || !fhat || !ghat || !fghat) return;

    /* Allocate a temporary block to support aliasing (fghat == fhat etc.). */
    /* Max block size is (2*(N-1)+1)^2 = (2N-1)^2. */
    int max_d = 2 * (N - 1) + 1;
    double _Complex *tmp = malloc((size_t)max_d * (size_t)max_d * sizeof(double _Complex));
    if (!tmp) return;

    for (int l = 0; l < N; ++l) {
        int d = 2 * l + 1;
        size_t off = su2_coeff_offset(l);

        /* Compute tmp[m, n] = sum_p fhat[m, p] * ghat[p, n]. */
        for (int m = -l; m <= l; ++m) {
            for (int n = -l; n <= l; ++n) {
                double _Complex acc = 0.0 + 0.0*I;
                for (int p = -l; p <= l; ++p) {
                    double _Complex f_mp = fhat[off + (size_t)(m + l) * (size_t)d + (size_t)(p + l)];
                    double _Complex g_pn = ghat[off + (size_t)(p + l) * (size_t)d + (size_t)(n + l)];
                    acc += f_mp * g_pn;
                }
                tmp[(size_t)(m + l) * (size_t)d + (size_t)(n + l)] = acc;
            }
        }

        /* Copy tmp into fghat. */
        for (size_t i = 0; i < (size_t)d * (size_t)d; ++i) {
            fghat[off + i] = tmp[i];
        }
    }

    free(tmp);
}
