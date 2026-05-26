/* su2_sphere.c -- Spherical-harmonic FFT on S^2 = SU(2)/U(1).
 *
 * Thin wrapper around su2_fft / su2_fft_inv: a function f(theta, phi)
 * on S^2 extends to a psi-independent function on SU(2). Its SU(2)
 * Fourier spectrum lives entirely on the m=0 row of each fhat(l) block.
 *
 * The factorisation t^l_{n,m}(g) = exp(-i n phi) exp(-i m psi) P^l_{n,m}
 * implies that for psi-invariant f, only the m=0 components are non-zero
 * (Folland §3.3; Vilenkin "Special Functions" §III.6).
 *
 * Storage:
 *   f_sph[j1*N + k]   (row-major in (j1, k); k=theta-index, j1=phi-index)
 *   fhat_sph[ sum_{l'<l}(2l'+1) + (n+l) ]  for l in [0, N-1], n in [-l, l]
 *   Total: sum_{l=0..N-1}(2l+1) = N^2 entries.
 *
 * Cost: O(N^4) (dominated by the underlying SU(2) FFT). The psi-replication
 * costs O(N^3) extra memory and O(N^3) data motion; could be optimised
 * later to a direct S^2 FFT, but the "thin wrapper" approach lets us reuse
 * all existing correctness machinery.
 *
 * Bead: su2fft-5fb.
 */
#include "su2.h"

#include <complex.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Total number of spherical-harmonic coefficients at bandlimit N. */
size_t su2_sphere_total_coeffs(int N)
{
    if (N < 1) return 0;
    return (size_t)N * (size_t)N;
}

/**
 * @brief Forward S^2 FFT (thin wrapper over su2_fft).
 *
 * @param[in]  N         Bandlimit (l in [0, N-1]).
 * @param[in]  f_sph     Length N*N complex sample array (row-major (j1, k)).
 * @param[out] fhat_sph  Length su2_sphere_total_coeffs(N) = N^2 spectrum.
 *                       Indexed by (l, n) with l in [0, N-1], n in [-l, l],
 *                       flat layout sum_{l'<l}(2l'+1) + (n+l).
 *
 * @par Reference notes/inverse_fft.md, ALGORITHM.md, bead su2fft-5fb.
 * @par Complexity O(N^4).
 */
void su2_fft_sphere(int N,
                    const double _Complex *f_sph,
                    double _Complex *fhat_sph)
{
    if (N < 2 || !f_sph || !fhat_sph) return;

    size_t nsamp_su2 = (size_t)N * (size_t)N * (size_t)N;
    double _Complex *f_su2 = malloc(nsamp_su2 * sizeof(double _Complex));
    if (!f_su2) return;

    /* Replicate psi-independent input: f_su2[j1, k, j2] = f_sph[j1, k] for all j2. */
    for (int j1 = 0; j1 < N; ++j1) {
        for (int k = 0; k < N; ++k) {
            double _Complex v = f_sph[(size_t)j1 * (size_t)N + (size_t)k];
            for (int j2 = 0; j2 < N; ++j2) {
                f_su2[su2_sample_index(N, j1, k, j2)] = v;
            }
        }
    }

    size_t ncoeff_su2 = su2_total_coeffs(N);
    double _Complex *fhat_su2 = malloc(ncoeff_su2 * sizeof(double _Complex));
    if (!fhat_su2) { free(f_su2); return; }

    su2_fft(N, f_su2, fhat_su2);

    /* Extract the m=0 row from each fhat(l) block.
     * fhat_su2[l][m, n] = fhat_su2[ su2_coeff_offset(l) + (m+l)*(2l+1) + (n+l) ]. */
    size_t idx_sph = 0;
    for (int l = 0; l < N; ++l) {
        size_t off = su2_coeff_offset(l);
        int d = 2 * l + 1;
        for (int n = -l; n <= l; ++n) {
            size_t flat_su2 = off + (size_t)(0 + l) * (size_t)d + (size_t)(n + l);
            fhat_sph[idx_sph++] = fhat_su2[flat_su2];
        }
    }

    free(f_su2);
    free(fhat_su2);
}

/**
 * @brief Inverse S^2 FFT (Peter-Weyl synthesis restricted to m=0).
 *
 * @param[in]  N         Bandlimit.
 * @param[in]  fhat_sph  Length N^2 spectrum (same layout as forward output).
 * @param[out] f_sph     Length N*N complex sample array (row-major (j1, k)).
 *
 * @par Reference paper.tex line 554; bead su2fft-5fb.
 */
void su2_fft_sphere_inv(int N,
                        const double _Complex *fhat_sph,
                        double _Complex *f_sph)
{
    if (N < 2 || !fhat_sph || !f_sph) return;

    size_t ncoeff_su2 = su2_total_coeffs(N);
    double _Complex *fhat_su2 = calloc(ncoeff_su2, sizeof(double _Complex));
    if (!fhat_su2) return;

    /* Embed fhat_sph[l, n] into fhat_su2[l][m=0, n]; rest = 0. */
    size_t idx_sph = 0;
    for (int l = 0; l < N; ++l) {
        size_t off = su2_coeff_offset(l);
        int d = 2 * l + 1;
        for (int n = -l; n <= l; ++n) {
            size_t flat_su2 = off + (size_t)(0 + l) * (size_t)d + (size_t)(n + l);
            fhat_su2[flat_su2] = fhat_sph[idx_sph++];
        }
    }

    size_t nsamp_su2 = (size_t)N * (size_t)N * (size_t)N;
    double _Complex *f_su2 = malloc(nsamp_su2 * sizeof(double _Complex));
    if (!f_su2) { free(fhat_su2); return; }

    su2_fft_inv(N, fhat_su2, f_su2);

    /* By psi-invariance, all j2 slices are equal. Take j2=0. */
    for (int j1 = 0; j1 < N; ++j1) {
        for (int k = 0; k < N; ++k) {
            f_sph[(size_t)j1 * (size_t)N + (size_t)k] =
                f_su2[su2_sample_index(N, j1, k, 0)];
        }
    }

    free(f_su2);
    free(fhat_su2);
}

/* =====================================================================
 * Resolved-grid sphere FFT (bead su2fft-9qk).
 *
 * Same thin-wrapper idea as the closed variant above, but layered over
 * `su2_fft_resolved` / `su2_fft_resolved_inv` (bead su2fft-0t1) instead
 * of `su2_fft` / `su2_fft_inv`.  The underlying SU(2) sample array uses
 * the open P-point phi/psi grid (P = 2N-1) and N-point Gauss-Legendre
 * theta nodes, with layout f_su2[j1*N*P + k*P + j2] via
 * `su2_resolved_sample_index`.  Spectrum layout is unchanged (N^2
 * entries, m=0 row per fhat(l) block).
 *
 * Because `su2_fft_resolved` gives exact spectrum roundtrip at working
 * precision (notes/0t1_resolved_grid_design.md §5, §6), the resolved
 * sphere variant inherits the same property -- no closed-grid Riemann
 * floor.
 * ===================================================================== */

/* Total number of sphere samples on the resolved grid. */
size_t su2_sphere_resolved_total_samples(int N)
{
    if (N < 1) return 0;
    size_t P = (size_t)(2 * N - 1);
    return P * (size_t)N;
}

/**
 * @brief Forward S^2 FFT on the resolved (open P-point phi + GL theta) grid.
 *
 * @param[in]  N         Bandlimit (l in [0, N-1]).
 * @param[in]  f_sph     Length P*N complex sample array, P = 2N-1,
 *                       row-major (j1, k) with j1 = phi index, k = theta index.
 * @param[out] fhat_sph  Length su2_sphere_total_coeffs(N) = N^2 spectrum.
 *                       Same (l, n) layout as `su2_fft_sphere`.
 *
 * @par Reference paper.tex line 554 (Peter-Weyl synthesis);
 *      notes/0t1_resolved_grid_design.md §3-§5; bead su2fft-9qk.
 * @par Complexity O(N^4) (dominated by `su2_fft_resolved`).
 */
void su2_fft_sphere_resolved(int N,
                             const double _Complex *f_sph,
                             double _Complex *fhat_sph)
{
    if (N < 2 || !f_sph || !fhat_sph) return;

    const int P = 2 * N - 1;

    size_t nsamp_su2 = su2_resolved_total_samples(N);
    double _Complex *f_su2 = malloc(nsamp_su2 * sizeof(double _Complex));
    if (!f_su2) return;

    /* Replicate psi-independent input across all j2 in [0, P-1]:
     *   f_su2[j1, k, j2] = f_sph[j1, k]  for all j2.
     * j1, j2 both range over [0, P-1]; k over [0, N-1]. */
    for (int j1 = 0; j1 < P; ++j1) {
        for (int k = 0; k < N; ++k) {
            double _Complex v = f_sph[(size_t)j1 * (size_t)N + (size_t)k];
            for (int j2 = 0; j2 < P; ++j2) {
                f_su2[su2_resolved_sample_index(N, j1, k, j2)] = v;
            }
        }
    }

    size_t ncoeff_su2 = su2_total_coeffs(N);
    double _Complex *fhat_su2 = malloc(ncoeff_su2 * sizeof(double _Complex));
    if (!fhat_su2) { free(f_su2); return; }

    su2_fft_resolved(N, f_su2, fhat_su2);

    /* Extract the m=0 row from each fhat(l) block.  Identical to the
     * closed wrapper -- the spectrum layout is invariant across grids. */
    size_t idx_sph = 0;
    for (int l = 0; l < N; ++l) {
        size_t off = su2_coeff_offset(l);
        int d = 2 * l + 1;
        for (int n = -l; n <= l; ++n) {
            size_t flat_su2 = off + (size_t)(0 + l) * (size_t)d + (size_t)(n + l);
            fhat_sph[idx_sph++] = fhat_su2[flat_su2];
        }
    }

    free(f_su2);
    free(fhat_su2);
}

/**
 * @brief Inverse S^2 FFT on the resolved grid (Peter-Weyl synthesis, m=0).
 *
 * @param[in]  N         Bandlimit.
 * @param[in]  fhat_sph  Length N^2 spectrum (same layout as forward output).
 * @param[out] f_sph     Length P*N complex sample array, P = 2N-1,
 *                       row-major (j1, k).
 *
 * @par Reference paper.tex line 554; notes/0t1_resolved_grid_design.md §3;
 *      bead su2fft-9qk.  Inherits exact spectrum roundtrip from bead 0t1.
 */
void su2_fft_sphere_inv_resolved(int N,
                                 const double _Complex *fhat_sph,
                                 double _Complex *f_sph)
{
    if (N < 2 || !fhat_sph || !f_sph) return;

    const int P = 2 * N - 1;

    size_t ncoeff_su2 = su2_total_coeffs(N);
    double _Complex *fhat_su2 = calloc(ncoeff_su2, sizeof(double _Complex));
    if (!fhat_su2) return;

    /* Embed fhat_sph[l, n] into fhat_su2[l][m=0, n]; rest = 0. */
    size_t idx_sph = 0;
    for (int l = 0; l < N; ++l) {
        size_t off = su2_coeff_offset(l);
        int d = 2 * l + 1;
        for (int n = -l; n <= l; ++n) {
            size_t flat_su2 = off + (size_t)(0 + l) * (size_t)d + (size_t)(n + l);
            fhat_su2[flat_su2] = fhat_sph[idx_sph++];
        }
    }

    size_t nsamp_su2 = su2_resolved_total_samples(N);
    double _Complex *f_su2 = malloc(nsamp_su2 * sizeof(double _Complex));
    if (!f_su2) { free(fhat_su2); return; }

    su2_fft_resolved_inv(N, fhat_su2, f_su2);

    /* By psi-invariance every j2 slice is equal; take j2=0 and copy out. */
    for (int j1 = 0; j1 < P; ++j1) {
        for (int k = 0; k < N; ++k) {
            f_sph[(size_t)j1 * (size_t)N + (size_t)k] =
                f_su2[su2_resolved_sample_index(N, j1, k, 0)];
        }
    }

    free(f_su2);
    free(fhat_su2);
}
