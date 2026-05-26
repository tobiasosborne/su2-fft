/* su2_fft_resolved.c -- O(N^4) SU(2) FFT on the resolved (open P-point) grid.
 *
 * Bead: su2fft-0t1.  Binding spec: notes/0t1_resolved_grid_design.md.
 *
 * What's different from su2_fft_gl
 * --------------------------------
 * `su2_fft_gl` runs an (N-1)x(N-1) FFTW plan after FOLDING the closed-grid
 * endpoints `phi[0] == phi[N-1]` and `psi[0] == psi[N-1]`.  That fold mixes
 * 2N-1 distinct (n, m) bandlimit modes into N-1 FFTW bins, aliasing all
 * |n|, |m| > (N-1)/2.  This bead replaces the closed grid with the OPEN
 * P-point grid P = 2N-1, sampling f at
 *     phi[j1] = -pi + j1 * 2pi/P,   j1 in [0, P-1]
 *     psi[j2] = -pi + j2 * 2pi/P,   j2 in [0, P-1]
 *     theta[k] = arccos(x_k),       x_k the k-th N-point GL node on [-1,1].
 * P = 2N-1 = number of bandlimit modes, so the (-1)^n * (-1)^m phase from
 * the -pi origin shift maps n in [-(N-1), N-1] to P distinct FFTW bins
 * with NO collisions and NO fold.  Stage 2 is identical to su2_fft_gl
 * except the global normalisation drops from 1/(2 N^2) to 1/(2 P^2)
 * (notes/0t1_resolved_grid_design.md §5).
 *
 * References:
 *   paper.tex line 1316 -- discrete forward analysis formula.
 *   paper.tex line 554  -- Peter-Weyl synthesis (used by the inverse).
 *   notes/0t1_resolved_grid_design.md §4 (Stage 1, no fold) and §5 (norm).
 *   HANDOFF.md §2 item 1 (closed-grid fold -- what we deliberately do NOT do).
 *   HANDOFF.md §2 item 7 (the phi/psi aliasing this bead eliminates).
 */
#include "su2.h"

#include <assert.h>
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Forward SU(2) FFT on the resolved (open P-point) phi/psi grid + GL theta.
 *
 * Mirrors `su2_fft_gl`.  The structural differences:
 *   - Stage 1 uses a P x P FFTW BACKWARD plan with P = 2N-1; NO endpoint
 *     fold, NO accumulation -- each FFTW bin is touched once per (n, m).
 *   - The Stage 2 global normalisation is 1/(2 P^2) instead of 1/(2 N^2)
 *     (notes/0t1_resolved_grid_design.md §5).
 *   - Sample storage is f[j1*N*P + k*P + j2] (via su2_resolved_sample_index),
 *     length P*P*N = (2N-1)^2 * N.
 *
 * Stage 2 (the GL theta inner product, the i^{n-m} phase, and the Wigner
 * recurrence `su2_wigner_d_seq`) is byte-for-byte identical to su2_fft_gl.
 *
 * @param[in]  N     Bandlimit; sample grid is P x N x P, P = 2N-1.
 * @param[in]  f     Length-(P*P*N) complex sample array, row-major (j1, k, j2).
 * @param[out] fhat  Length-su2_total_coeffs(N) complex coefficient array.
 * @par Complexity O(N^4) flops; O(N^3) auxiliary memory for F2.
 * @par Reference paper.tex line 1316; notes/0t1_resolved_grid_design.md §4-§5.
 */
void su2_fft_resolved(int N,
                      const double _Complex *f,
                      double _Complex *fhat)
{
    if (N < 2 || !f || !fhat) return;

    const int P = 2 * N - 1;  /* open-grid size; nrange = 2N-1 = P. */

    /* N-point Gauss-Legendre nodes and weights on [-1, 1]; theta_k = arccos(x_k).
     * notes/gauss_legendre.md §3. */
    double *x_gl = malloc((size_t)N * sizeof(double));
    double *w_gl = malloc((size_t)N * sizeof(double));
    double *theta = malloc((size_t)N * sizeof(double));
    assert(x_gl && w_gl && theta && "su2_fft_resolved: alloc");
    su2_gl_nodes_weights(N, x_gl, w_gl);
    for (int k = 0; k < N; ++k) theta[k] = acos(x_gl[k]);

    /* -------- Stage 1: P x P FFTW BACKWARD per theta slice, NO fold.
     *
     * For j1, j2 in [0, P-1] (no endpoint duplication on the open grid):
     *   g[j1, j2] = f[j1, k, j2]
     * FFTW BACKWARD (sign +1, our convention) gives
     *   G[a, b] = sum_{j1,j2} g[j1,j2] * exp(+i 2pi (a j1 + b j2) / P)
     * The (-1)^n shift from phi[j] = -pi + j*2pi/P yields
     *   sum_j f * exp(+i n phi[j]) = (-1)^n * G[n mod P, .].
     * Each n in [-(N-1), N-1] maps to a unique bin (range size 2N-1 = P),
     * so the F2 write below is a single assignment per (k, n, m).
     * notes/0t1_resolved_grid_design.md §4 (Stage 1).
     */
    fftw_complex *g = fftw_malloc(sizeof(fftw_complex) * (size_t)P * (size_t)P);
    fftw_complex *G = fftw_malloc(sizeof(fftw_complex) * (size_t)P * (size_t)P);
    assert(g && G && "su2_fft_resolved: fftw_malloc");
    fftw_plan plan = fftw_plan_dft_2d(P, P, g, G, FFTW_BACKWARD, FFTW_ESTIMATE);

    /* F2 layout matches su2_fft_gl: stride_n = 2N-1, stride_k = (2N-1)^2.
     * For the resolved path nrange = 2N-1 = P -- they coincide. */
    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;
    double _Complex *F2 = malloc((size_t)N * stride_k * sizeof(double _Complex));
    assert(F2 && "su2_fft_resolved: alloc F2");

    for (int k = 0; k < N; ++k) {
        /* Direct copy -- no fold, no accumulation. */
        for (int j1 = 0; j1 < P; ++j1) {
            for (int j2 = 0; j2 < P; ++j2) {
                g[(size_t)j1 * (size_t)P + (size_t)j2] =
                    f[su2_resolved_sample_index(N, j1, k, j2)];
            }
        }
        fftw_execute(plan);

        for (int n = -(N - 1); n <= N - 1; ++n) {
            int    n_mod = ((n % P) + P) % P;
            double sn    = (n & 1) ? -1.0 : 1.0;   /* (-1)^n from -pi shift. */
            for (int m = -(N - 1); m <= N - 1; ++m) {
                int    m_mod = ((m % P) + P) % P;
                double sm    = (m & 1) ? -1.0 : 1.0;
                F2[(size_t)k * stride_k
                   + (size_t)(n + N - 1) * stride_n
                   + (size_t)(m + N - 1)]
                    = sn * sm * G[(size_t)n_mod * (size_t)P + (size_t)m_mod];
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(g);
    fftw_free(G);

    /* -------- Stage 2: GL theta inner product (identical to su2_fft_gl
     * except for the global norm).
     *
     * For constant input f == 1: G[0,0] = P*P (no over-count), (-1)^0=1, so
     * F2[k, 0, 0] = P^2.  GL weights sum to 2.  Requiring fhat(0)_{0,0} = 1:
     *   norm * P^2 * 2 = 1   =>   norm = 1 / (2 P^2).
     * notes/0t1_resolved_grid_design.md §5.
     *
     * Identity (paper.tex line 1316, HANDOFF.md §2 item 2):
     *   conj(P^l_{n,m}) = i^{n-m} * d^l_{n,m}     (d real, Sakurai convention)
     * giving
     *   fhat(l)_{m,n} = norm * i^{n-m} * sum_k w_gl[k] * d^l_{n,m}(theta_k)
     *                                       * F2[k, n, m].
     */
    const double norm = 1.0 / (2.0 * (double)P * (double)P);

    double          *d_seq = malloc((size_t)N * sizeof(double));
    double _Complex *acc   = malloc((size_t)N * sizeof(double _Complex));
    assert(d_seq && acc && "su2_fft_resolved: alloc");

    for (int m = -(N - 1); m <= N - 1; ++m) {
        for (int n = -(N - 1); n <= N - 1; ++n) {
            int l_min = (abs(m) > abs(n)) ? abs(m) : abs(n);
            if (l_min > N - 1) continue;

            for (int l = l_min; l < N; ++l) acc[l] = 0.0 + 0.0*I;

            for (int k = 0; k < N; ++k) {
                su2_wigner_d_seq(l_min, N - 1, n, m, theta[k], d_seq);
                double _Complex w = F2[(size_t)k * stride_k
                                       + (size_t)(n + N - 1) * stride_n
                                       + (size_t)(m + N - 1)] * w_gl[k];
                for (int l = l_min; l < N; ++l) {
                    acc[l] += d_seq[l - l_min] * w;
                }
            }

            /* Apply i^{n-m} phase + global norm; write fhat. */
            int r = ((n - m) % 4 + 4) % 4;
            double _Complex phase;
            switch (r) {
                case 0:  phase =  1.0 + 0.0*I; break;
                case 1:  phase =  0.0 + 1.0*I; break;
                case 2:  phase = -1.0 + 0.0*I; break;
                default: phase =  0.0 - 1.0*I; break;
            }
            double _Complex c = norm * phase;
            for (int l = l_min; l < N; ++l) {
                fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)] = c * acc[l];
            }
        }
    }

    free(acc);
    free(d_seq);
    free(F2);
    free(theta);
    free(w_gl);
    free(x_gl);
}

/**
 * @brief Inverse SU(2) FFT on the resolved (open P-point) phi/psi grid + GL theta.
 *
 * Mirrors `su2_fft_inv_gl`.  Stage 2-inv (synthesis of G[k, n, m] from fhat
 * via the (2l+1)-weighted Wigner sum, plus the i^{m-n} phase) is byte-for-byte
 * identical -- the synthesis is exact and carries NO extra normalisation.
 * Only Stage 1-inv changes:
 *   - P x P FFTW FORWARD plan (P = 2N-1).
 *   - The (n, m) -> bin scatter is an assignment (NOT +=), since the open
 *     grid maps each n in [-(N-1), N-1] to a unique residue mod P = 2N-1
 *     -- no aliasing, no accumulation.
 *   - NO closed-grid edge fill (f[j=N-1] = f[j=0]).  The open grid has no
 *     duplicated endpoint.
 *
 * @param[in]  N     Bandlimit; output sample grid is P x N x P, P = 2N-1.
 * @param[in]  fhat  Length-su2_total_coeffs(N) complex coefficient array.
 * @param[out] f     Length-(P*P*N) complex sample array, row-major (j1, k, j2).
 * @par Reference paper.tex line 554; notes/0t1_resolved_grid_design.md §4.
 */
void su2_fft_resolved_inv(int N,
                          const double _Complex *fhat,
                          double _Complex *f)
{
    if (N < 2 || !fhat || !f) return;

    const int P = 2 * N - 1;

    /* GL nodes; weights are unused by the synthesis. */
    double *x_gl  = malloc((size_t)N * sizeof(double));
    double *w_gl  = malloc((size_t)N * sizeof(double));
    double *theta = malloc((size_t)N * sizeof(double));
    assert(x_gl && w_gl && theta && "su2_fft_resolved_inv: alloc");
    su2_gl_nodes_weights(N, x_gl, w_gl);
    for (int k = 0; k < N; ++k) theta[k] = acos(x_gl[k]);

    /* ---- Stage 2-inv (identical to su2_fft_inv_gl) ----
     * paper.tex line 554:
     *   f(g) = sum_l (2l+1) sum_{m,n} fhat(l)_{mn} t^l_{nm}(g)
     * with t^l_{nm}(g) = exp(-i(n phi + m psi)) * P^l_{n,m}(cos theta)
     *                  = exp(-i(n phi + m psi)) * i^{m-n} * d^l_{n,m}(cos theta).
     * Pulling the i^{m-n} phase outside the l-sum:
     *   G[k, n, m] = i^{m-n} * sum_l (2l+1) * fhat(l)_{m,n} * d^l_{n,m}(theta_k).
     */
    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;
    double _Complex *G = malloc((size_t)N * stride_k * sizeof(double _Complex));
    assert(G && "su2_fft_resolved_inv: alloc G");

    double *d_seq = malloc((size_t)N * sizeof(double));
    assert(d_seq && "su2_fft_resolved_inv: alloc d_seq");

    for (int m = -(N - 1); m <= N - 1; ++m) {
        for (int n = -(N - 1); n <= N - 1; ++n) {
            int l_min = (abs(m) > abs(n)) ? abs(m) : abs(n);
            if (l_min > N - 1) continue;

            int r = ((m - n) % 4 + 4) % 4;
            double _Complex phase;
            switch (r) {
                case 0:  phase =  1.0 + 0.0*I; break;
                case 1:  phase =  0.0 + 1.0*I; break;
                case 2:  phase = -1.0 + 0.0*I; break;
                default: phase =  0.0 - 1.0*I; break;
            }

            for (int k = 0; k < N; ++k) {
                su2_wigner_d_seq(l_min, N - 1, n, m, theta[k], d_seq);
                double _Complex sum = 0.0 + 0.0*I;
                for (int l = l_min; l < N; ++l) {
                    double _Complex coef = fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)];
                    sum += (double)(2*l + 1) * coef * d_seq[l - l_min];
                }
                G[(size_t)k * stride_k
                  + (size_t)(n + N - 1) * stride_n
                  + (size_t)(m + N - 1)] = phase * sum;
            }
        }
    }

    /* ---- Stage 1-inv: P x P FFTW FORWARD per theta slice, NO fold.
     *
     * For each k:
     *   G_slice[n_mod, m_mod] = (-1)^n * (-1)^m * G[k, n, m]   (assignment,
     *     NOT +=:  n in [-(N-1), N-1] hits each residue mod P = 2N-1 once).
     * FFTW FORWARD then yields f[j1, k, j2] for j1, j2 in [0, P-1].
     * NO closed-grid edge fill -- there is no duplicated j = P-1 sample
     * on the open grid.
     * notes/0t1_resolved_grid_design.md §4 (Stage 1-inv).
     */
    fftw_complex *G_slice = fftw_malloc(sizeof(fftw_complex) * (size_t)P * (size_t)P);
    fftw_complex *f_slice = fftw_malloc(sizeof(fftw_complex) * (size_t)P * (size_t)P);
    assert(G_slice && f_slice && "su2_fft_resolved_inv: fftw_malloc");
    fftw_plan plan = fftw_plan_dft_2d(P, P, G_slice, f_slice, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int k = 0; k < N; ++k) {
        memset(G_slice, 0, sizeof(fftw_complex) * (size_t)P * (size_t)P);

        for (int n = -(N - 1); n <= N - 1; ++n) {
            int    n_mod = ((n % P) + P) % P;
            double sn    = (n & 1) ? -1.0 : 1.0;
            for (int m = -(N - 1); m <= N - 1; ++m) {
                int    m_mod = ((m % P) + P) % P;
                double sm    = (m & 1) ? -1.0 : 1.0;
                G_slice[(size_t)n_mod * (size_t)P + (size_t)m_mod]
                    = sn * sm * G[(size_t)k * stride_k
                                  + (size_t)(n + N - 1) * stride_n
                                  + (size_t)(m + N - 1)];
            }
        }
        fftw_execute(plan);

        for (int j1 = 0; j1 < P; ++j1) {
            for (int j2 = 0; j2 < P; ++j2) {
                f[su2_resolved_sample_index(N, j1, k, j2)] =
                    f_slice[(size_t)j1 * (size_t)P + (size_t)j2];
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(G_slice);
    fftw_free(f_slice);

    free(d_seq);
    free(G);
    free(theta);
    free(w_gl);
    free(x_gl);
}
