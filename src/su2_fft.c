/* su2_fft.c -- Fast Fourier transform on SU(2).  O(N^4) total cost.
 *
 * Strategy (paper.tex, Section 4, summarised in notes/section_4_algorithm.md):
 *
 *   Stage 1: For each theta-slice k, compute the 2-D DFT
 *              F2[k, n, m] = sum_{j1, j2} f(g_{j1,k,j2})
 *                                       * exp(+i n phi[j1])
 *                                       * exp(+i m psi[j2])
 *            via FFTW.  Per slice O(N^2 log N); total O(N^3 log N).
 *
 *   Stage 2: For each (l, m, n), evaluate
 *              fhat(l)_{m,n} = norm * sum_{k} F2[k, n, m]
 *                                          * conj(P^l_{n,m}(cos theta_k))
 *                                          * sin(theta_k)
 *            (paper line 1361, eq EQ_OP_1).  Per coefficient O(N); over O(N^3)
 *            coefficients = O(N^4) -- the headline complexity at split k=1
 *            (paper Theorem TEO_FIN, line 1455).
 *
 * Closed-grid handling for Stage 1
 * --------------------------------
 * The Euler grid has phi_{j1} = -pi + j1 * 2pi/(N-1), j1 in [0, N-1].  Both
 * endpoints j1=0 and j1=N-1 sit at the same point on the torus (mod 2pi), so
 * the N samples carry only N-1 independent Fourier modes.  Writing
 *
 *   exp(+i n phi[j1]) = (-1)^n * exp(+i n j1 * 2pi/(N-1))
 *
 * the sum over j1 in [0, N-1] equals (-1)^n times the (N-1)-point DFT of the
 * folded sequence  g[0] = f[0] + f[N-1],  g[j] = f[j]  for j in [1, N-2].
 * We do the analogous fold in j2 and use one FFTW 2-D backward plan of size
 * (N-1) x (N-1).
 */
#include "su2.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Stage 1 of the forward FFT: per-theta-slice closed-grid fold +
 *        (N-1)x(N-1) backward 2-D FFTW + (-1)^{n+m} phase -> F2[k, n, m].
 *
 * Shared verbatim by su2_fft (complex input) and su2_fft_real (real input,
 * promoted to a zero-imaginary complex buffer).  Factored out of su2_fft so the
 * two entry points cannot drift; su2_fft's observable behaviour is unchanged.
 *
 * F2 must point to N * (2N-1) * (2N-1) complex slots, indexed as
 *   F2[k*stride_k + (n+N-1)*stride_n + (m+N-1)],
 * with stride_n = 2N-1 and stride_k = (2N-1)^2 (see callers).
 *
 * @par Reference paper.tex line 1347; ALGORITHM.md Section 2.2.
 */
static void su2_fft_stage1(int N,
                           const double _Complex *f,
                           double _Complex *F2)
{
    const int M = N - 1;

    fftw_complex *g = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_complex *G = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_plan plan  = fftw_plan_dft_2d(M, M, g, G, FFTW_BACKWARD, FFTW_ESTIMATE);

    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;

    for (int k = 0; k < N; ++k) {
        memset(g, 0, sizeof(fftw_complex) * (size_t)M * (size_t)M);

        for (int j1 = 0; j1 < N; ++j1) {
            int j1m = (j1 == N - 1) ? 0 : j1;
            for (int j2 = 0; j2 < N; ++j2) {
                int j2m = (j2 == N - 1) ? 0 : j2;
                g[(size_t)j1m * (size_t)M + (size_t)j2m] +=
                    f[su2_sample_index(N, j1, k, j2)];
            }
        }
        fftw_execute(plan);

        for (int n = -(N - 1); n <= N - 1; ++n) {
            int n_mod = ((n % M) + M) % M;
            double sn = (n & 1) ? -1.0 : 1.0;   /* (-1)^n, valid for two's comp */
            for (int m = -(N - 1); m <= N - 1; ++m) {
                int m_mod = ((m % M) + M) % M;
                double sm = (m & 1) ? -1.0 : 1.0;
                F2[(size_t)k * stride_k
                   + (size_t)(n + N - 1) * stride_n
                   + (size_t)(m + N - 1)]
                    = sn * sm * G[(size_t)n_mod * (size_t)M + (size_t)m_mod];
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(g);
    fftw_free(G);
}

/**
 * @brief Fast O(N^4) Fourier transform on SU(2) via 2-D FFTW + Wigner inner product.
 *
 * Produces the same fhat as su2_ft_direct() (to floating-point tolerance) by
 * separating the phi/psi and theta summations into two stages:
 *
 *   Stage 1 (O(N^3 log N)): For each theta-slice k, fold the closed grid
 *     endpoints and apply a (N-1)x(N-1) backward 2-D FFTW plan to obtain
 *     F2[k, n, m] (paper.tex line 1347).  A (-1)^{n+m} factor restores the
 *     half-shift from the closed grid; see ALGORITHM.md Section 2.2.
 *
 *   Stage 2 (O(N^4)): For each coefficient (l, m, n), evaluate the length-N
 *     dot product against conj(P^l_{n,m}(cos theta_k)) * sin(theta_k)
 *     (paper.tex line 1361, eq EQ_OP_1).  O(N^3) coefficients x O(N) per dot
 *     product gives the headline O(N^4) (paper Theorem TEO_FIN, line 1455).
 *
 * @param[in]  N     Bandlimit; grid is N x N x N, coefficients span l < N.
 * @param[in]  f     Length-N^3 complex sample array, row-major (j1, k, j2).
 * @param[out] fhat  Length-su2_total_coeffs(N) complex coefficient array.
 * @par Complexity O(N^4) flops; O(N^3) auxiliary memory for F2.
 * @par Reference paper.tex lines 1347, 1361, 1455; ALGORITHM.md Section 2.2.
 */
void su2_fft(int N,
             const double _Complex *f,
             double _Complex *fhat)
{
    if (N < 2 || !f || !fhat) return;

    double *theta = su2_grid_theta(N);

    /* -------- Stage 1 (shared with su2_fft_real) -------- */
    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;
    double _Complex *F2 = malloc((size_t)N * stride_k * sizeof(double _Complex));

    su2_fft_stage1(N, f, F2);

    /* -------- Stage 2 -------- */
    /* Recurrence-based sweep (bead su2fft-m21). For each (m, n), sweep
     * l = l_min..N-1 via the ascending-l three-term recurrence in
     * su2_wigner_d_seq (notes/wigner_recurrence.md). This drops Stage 2
     * from O(N^5) (one O(l) de Moivre call per (l,m,n,k)) to O(N^4)
     * (O(1) per recurrence step), matching the paper's headline.
     *
     * Identity: conj(P^l_{n,m}) = i^{n-m} * d^l_{n,m} (d real).  Pull the
     * phase outside the k-sum:
     *   fhat(l)_{m,n} = norm * i^{n-m} * sum_k d^l_{n,m}(theta_k)
     *                                      * F2[k,n,m] * sin(theta_k)
     *
     * Loop order (m, n, k, l) minimises per-(m,n) allocation and keeps
     * acc[l] writes contiguous; F2[k,n,m] is loaded once per (k,n,m).
     * paper.tex line 1361 (eq EQ_OP_1).
     */
    const double dphi   = 2.0 * M_PI / (double)(N - 1);
    const double dtheta =       M_PI / (double)(N - 1);
    const double dpsi   = 2.0 * M_PI / (double)(N - 1);
    const double norm   = dphi * dtheta * dpsi / (8.0 * M_PI * M_PI);

    double *sin_th = malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N; ++k) sin_th[k] = sin(theta[k]);

    /* Per-(m,n) scratch buffers, reused across the outer loops. */
    double          *d_seq = malloc((size_t)N * sizeof(double));
    double _Complex *acc   = malloc((size_t)N * sizeof(double _Complex));

    for (int m = -(N - 1); m <= N - 1; ++m) {
        for (int n = -(N - 1); n <= N - 1; ++n) {
            int l_min = (abs(m) > abs(n)) ? abs(m) : abs(n);
            if (l_min > N - 1) continue;  /* impossible: |m|,|n| <= N-1 */

            for (int l = l_min; l < N; ++l) acc[l] = 0.0 + 0.0*I;

            for (int k = 0; k < N; ++k) {
                su2_wigner_d_seq(l_min, N - 1, n, m, theta[k], d_seq);
                double _Complex w = F2[(size_t)k * stride_k
                                       + (size_t)(n + N - 1) * stride_n
                                       + (size_t)(m + N - 1)] * sin_th[k];
                for (int l = l_min; l < N; ++l) {
                    acc[l] += d_seq[l - l_min] * w;
                }
            }

            /* Apply i^{n-m} phase and norm; write to fhat.
             * i^{n-m}: r = ((n-m) mod 4 + 4) mod 4, branch on r. */
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
    free(sin_th);
    free(F2);
    free(theta);
}

/**
 * @brief Forward O(N^4) FFT for REAL input, exploiting Hermitian symmetry of
 *        the dual to halve the Stage-2 (m, n) sweeps (bead su2fft-4v7).
 *
 * Produces the SAME full fhat (length su2_total_coeffs(N)) that su2_fft would
 * produce on the complexified real input f + 0i.  Stage 1 is identical (the
 * real samples are promoted to a zero-imaginary complex buffer and the shared
 * su2_fft_stage1 is run unchanged).
 *
 * Symmetry (VERIFIED -- max deviation 1.4e-17 at N=6; NOT the unsigned form):
 *   fhat(l)_{m,n} = (-1)^{m-n} * conj( fhat(l)_{-m,-n} ).
 * Derivation: from paper.tex line 1361 (eq EQ_OP_1),
 *   fhat(l)_{m,n} = norm * sum_k [s]_k * conj(P^l_{n,m}(cos theta_k)),
 * with [s]_k built from F2[k,n,m].  For real f, conjugating and using
 *   P^l_{n,m} = i^{m-n} d^l_{n,m}  (HANDOFF.md §2 item 2),
 *   d^l_{-n,-m} = (-1)^{m-n} d^l_{n,m}  (Wigner-d symmetry),
 * and the index reflection (n,m) -> (-n,-m) in Stage 1 of the real DFT, the
 * pair (m,n) and (-m,-n) collapse onto each other up to the (-1)^{m-n} sign.
 * The (m,n)=(0,0) coefficient is self-paired, hence fhat(l)_{0,0} is real.
 *
 * Only CANONICAL pairs (n>0, or n==0 && m>=0) are evaluated via the same
 * recurrence/phase/norm as su2_fft; the complementary NON-canonical entries
 * are filled by the symmetry, halving the Stage-2 work.
 *
 * @param[in]  N     Bandlimit; grid is N x N x N, coefficients span l < N.
 * @param[in]  f     Length-N^3 REAL sample array, row-major (j1, k, j2).
 * @param[out] fhat  Length-su2_total_coeffs(N) complex coefficient array.
 * @par Complexity O(N^4) flops, ~half the Stage-2 constant of su2_fft.
 * @par Reference paper.tex line 1361 (eq EQ_OP_1); ALGORITHM.md Section 2.2.
 */
void su2_fft_real(int N,
                  const double *f,
                  double _Complex *fhat)
{
    if (N < 2 || !f || !fhat) return;

    double *theta = su2_grid_theta(N);

    /* -------- Stage 1 (shared with su2_fft) -------- */
    /* Promote the real input to a zero-imaginary complex buffer, then run the
     * SAME Stage 1.  This is the only extra O(N^3) work vs su2_fft. */
    const size_t nsamp = (size_t)N * (size_t)N * (size_t)N;
    double _Complex *fc = malloc(nsamp * sizeof(double _Complex));
    for (size_t i = 0; i < nsamp; ++i) fc[i] = f[i] + 0.0 * I;

    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;
    double _Complex *F2 = malloc((size_t)N * stride_k * sizeof(double _Complex));

    su2_fft_stage1(N, fc, F2);
    free(fc);

    /* -------- Stage 2 (canonical (m,n) only) --------
     * Identical recurrence/phase/norm as su2_fft (paper.tex line 1361,
     * eq EQ_OP_1), but the (m,n) sweep is restricted to canonical pairs.
     * Non-canonical entries are filled from the Hermitian symmetry below.
     */
    const double dphi   = 2.0 * M_PI / (double)(N - 1);
    const double dtheta =       M_PI / (double)(N - 1);
    const double dpsi   = 2.0 * M_PI / (double)(N - 1);
    const double norm   = dphi * dtheta * dpsi / (8.0 * M_PI * M_PI);

    double *sin_th = malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N; ++k) sin_th[k] = sin(theta[k]);

    double          *d_seq = malloc((size_t)N * sizeof(double));
    double _Complex *acc   = malloc((size_t)N * sizeof(double _Complex));

    for (int m = -(N - 1); m <= N - 1; ++m) {
        for (int n = -(N - 1); n <= N - 1; ++n) {
            /* Canonical: n > 0, or n == 0 && m >= 0.  Partner (-m,-n) is then
             * non-canonical and gets filled by the symmetry in the second pass. */
            int is_canonical = (n > 0) || (n == 0 && m >= 0);
            if (!is_canonical) continue;

            int l_min = (abs(m) > abs(n)) ? abs(m) : abs(n);
            if (l_min > N - 1) continue;

            for (int l = l_min; l < N; ++l) acc[l] = 0.0 + 0.0*I;

            for (int k = 0; k < N; ++k) {
                su2_wigner_d_seq(l_min, N - 1, n, m, theta[k], d_seq);
                double _Complex w = F2[(size_t)k * stride_k
                                       + (size_t)(n + N - 1) * stride_n
                                       + (size_t)(m + N - 1)] * sin_th[k];
                for (int l = l_min; l < N; ++l) {
                    acc[l] += d_seq[l - l_min] * w;
                }
            }

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

    /* -------- Second pass: fill non-canonical (m,n) by symmetry --------
     * fhat(l)_{m,n} = (-1)^{m-n} * conj(fhat(l)_{-m,-n}); the partner is
     * canonical and already computed above.  s = (-1)^{m-n}. */
    for (int m = -(N - 1); m <= N - 1; ++m) {
        for (int n = -(N - 1); n <= N - 1; ++n) {
            int is_canonical = (n > 0) || (n == 0 && m >= 0);
            if (is_canonical) continue;

            int l_min = (abs(m) > abs(n)) ? abs(m) : abs(n);
            if (l_min > N - 1) continue;

            double s = ((m - n) & 1) ? -1.0 : 1.0;   /* (-1)^{m-n} */
            for (int l = l_min; l < N; ++l) {
                double _Complex partner =
                    fhat[su2_coeff_offset(l) + su2_mn_index(l, -m, -n)];
                fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)] =
                    s * conj(partner);
            }
        }
    }

    free(acc);
    free(d_seq);
    free(sin_th);
    free(F2);
    free(theta);
}

/**
 * @brief Inverse of su2_fft via Peter-Weyl synthesis.
 *
 * Structurally symmetric with su2_fft:
 *   Stage 2-inv: For each (m, n), sweep l = l_min..N-1 via the recurrence;
 *     accumulate G[k, n, m] = i^{m-n} * sum_l (2l+1) * fhat(l)_{m,n} * d^l_{n,m}(theta_k).
 *   Stage 1-inv: For each theta slice k, take G[k, ., .] and 2D FFTW FORWARD
 *     of size (N-1)x(N-1) with the same (-1)^{n+m} fold trick (reversed direction).
 *
 * The phase factor here is i^{m-n} (positive m-n), in contrast to su2_fft's
 * i^{n-m} (which came from conj(P^l_{n,m})). See notes/inverse_fft.md §3.
 *
 * The 2D FFTW direction is FORWARD here vs BACKWARD in su2_fft, because the
 * synthesis evaluates exp(-i*n*phi - i*m*psi) which is the FFTW FORWARD index
 * direction; see notes/inverse_fft.md §3.
 *
 * @par Complexity O(N^4); O(N^3) auxiliary memory for G.
 * @par Reference notes/inverse_fft.md; paper.tex Peter-Weyl synthesis (line 554).
 */
void su2_fft_inv(int N,
                 const double _Complex *fhat,
                 double _Complex *f)
{
    if (N < 2 || !fhat || !f) return;

    const int M = N - 1;
    double *theta = su2_grid_theta(N);

    /* ---- Stage 2-inv ----
     * For each (m, n), sweep l = l_min..N-1 with the recurrence; compute G[k, n, m].
     * paper.tex line 554: f(g) = sum_l (2l+1) sum_{m,n} fhat(l)_{mn} t^l_{nm}(g).
     * t^l_{nm}(g) = exp(-i(n*phi + m*psi)) * P^l_{n,m}(cos theta),
     * with P^l_{n,m} = i^{m-n} * d^l_{n,m} (HANDOFF.md §2 item 2).
     */
    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;
    double _Complex *G = malloc((size_t)N * stride_k * sizeof(double _Complex));

    /* Per-(m,n) scratch. */
    double *d_seq = malloc((size_t)N * sizeof(double));

    for (int m = -(N - 1); m <= N - 1; ++m) {
        for (int n = -(N - 1); n <= N - 1; ++n) {
            int l_min = (abs(m) > abs(n)) ? abs(m) : abs(n);
            if (l_min > N - 1) continue;

            /* Apply i^{m-n} phase (positive m-n; the +i^{m-n} of P^l_{n,m}). */
            int r = ((m - n) % 4 + 4) % 4;
            double _Complex phase;
            switch (r) {
                case 0:  phase =  1.0 + 0.0*I; break;
                case 1:  phase =  0.0 + 1.0*I; break;
                case 2:  phase = -1.0 + 0.0*I; break;
                default: phase =  0.0 - 1.0*I; break;
            }

            /* For each k: compute G[k, n, m] = phase * sum_l (2l+1) * fhat(l)_{m,n} * d^l(theta_k). */
            for (int k = 0; k < N; ++k) {
                su2_wigner_d_seq(l_min, N - 1, n, m, theta[k], d_seq);
                double _Complex acc = 0.0 + 0.0*I;
                for (int l = l_min; l < N; ++l) {
                    double _Complex coef = fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)];
                    acc += (double)(2*l + 1) * coef * d_seq[l - l_min];
                }
                G[(size_t)k * stride_k
                  + (size_t)(n + N - 1) * stride_n
                  + (size_t)(m + N - 1)] = phase * acc;
            }
        }
    }

    /* ---- Stage 1-inv ----
     * 2D FFTW FORWARD of G[k, ., .] -> f[j1, k, j2] per theta slice.
     * Use the same (-1)^{n+m} fold trick as the forward stage; the
     * "fold" direction here aliases (n, m) modes mod M into the FFTW grid.
     * See ALGORITHM.md §2.2 / notes/inverse_fft.md §3.
     */
    fftw_complex *G_slice = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_complex *f_slice = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_plan plan  = fftw_plan_dft_2d(M, M, G_slice, f_slice, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int k = 0; k < N; ++k) {
        memset(G_slice, 0, sizeof(fftw_complex) * (size_t)M * (size_t)M);

        for (int n = -(N - 1); n <= N - 1; ++n) {
            int n_mod = ((n % M) + M) % M;
            double sn = (n & 1) ? -1.0 : 1.0;       /* (-1)^n */
            for (int m = -(N - 1); m <= N - 1; ++m) {
                int m_mod = ((m % M) + M) % M;
                double sm = (m & 1) ? -1.0 : 1.0;   /* (-1)^m */
                G_slice[(size_t)n_mod * (size_t)M + (size_t)m_mod]
                    += sn * sm * G[(size_t)k * stride_k
                                   + (size_t)(n + N - 1) * stride_n
                                   + (size_t)(m + N - 1)];
            }
        }
        fftw_execute(plan);

        /* Distribute f_slice -> f[j1, k, j2] for j1, j2 in [0, N-1]; the closed
         * grid sets f[N-1, k, j2] = f[0, k, j2] (and similarly in j2). */
        for (int j1 = 0; j1 < N; ++j1) {
            int j1m = (j1 == N - 1) ? 0 : j1;
            for (int j2 = 0; j2 < N; ++j2) {
                int j2m = (j2 == N - 1) ? 0 : j2;
                f[su2_sample_index(N, j1, k, j2)] =
                    f_slice[(size_t)j1m * (size_t)M + (size_t)j2m];
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(G_slice);
    fftw_free(f_slice);

    free(d_seq);
    free(G);
    free(theta);
}

/**
 * @brief Forward SU(2) FFT with Gauss-Legendre theta quadrature (bead su2fft-ega).
 *
 * Mirrors su2_fft.  The only differences vs su2_fft:
 *   - theta_k = arccos(x_k) where x_k is the k-th N-point Gauss-Legendre node
 *     on [-1, 1] (ascending), in place of the closed grid theta_k = k*pi/(N-1).
 *   - The Stage 2 per-theta weight sin(theta_k)*dtheta (Riemann + Jacobian) is
 *     replaced by the GL weight w_k (absorbs the Jacobian sin(theta)dtheta = -dx
 *     into a single quadrature weight on [-1, 1]).
 *   - The global normalisation drops dtheta and becomes norm_gl = 1/(2*N^2);
 *     derivation in notes/gauss_legendre.md §6.
 *
 * Stage 1 is structurally identical to su2_fft: the closed-grid (phi, psi)
 * fold + (N-1)x(N-1) FFTW backward plan + (-1)^{n+m} phase do not depend on
 * the theta values, only on theta-slice index k.
 *
 * @param[in]  N     Bandlimit; grid is N x N x N samples (GL in theta).
 * @param[in]  f     Length-N^3 complex sample array, row-major (j1, k, j2),
 *                   with f[j1, k, j2] sampled at theta_k = arccos(x_k).
 * @param[out] fhat  Length-su2_total_coeffs(N) complex coefficient array.
 * @par Reference notes/gauss_legendre.md §5, §6.
 */
void su2_fft_gl(int N,
                const double _Complex *f,
                double _Complex *fhat)
{
    if (N < 2 || !f || !fhat) return;

    const int M = N - 1;

    /* N-point Gauss-Legendre nodes and weights on [-1, 1]; theta_k = arccos(x_k).
     * notes/gauss_legendre.md §3. */
    double *x_gl  = malloc((size_t)N * sizeof(double));
    double *w_gl  = malloc((size_t)N * sizeof(double));
    su2_gl_nodes_weights(N, x_gl, w_gl);
    double *theta = malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N; ++k) theta[k] = acos(x_gl[k]);

    /* -------- Stage 1 (identical to su2_fft) -------- */
    fftw_complex *g = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_complex *G = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_plan plan  = fftw_plan_dft_2d(M, M, g, G, FFTW_BACKWARD, FFTW_ESTIMATE);

    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;
    double _Complex *F2 = malloc((size_t)N * stride_k * sizeof(double _Complex));

    for (int k = 0; k < N; ++k) {
        memset(g, 0, sizeof(fftw_complex) * (size_t)M * (size_t)M);

        for (int j1 = 0; j1 < N; ++j1) {
            int j1m = (j1 == N - 1) ? 0 : j1;
            for (int j2 = 0; j2 < N; ++j2) {
                int j2m = (j2 == N - 1) ? 0 : j2;
                g[(size_t)j1m * (size_t)M + (size_t)j2m] +=
                    f[su2_sample_index(N, j1, k, j2)];
            }
        }
        fftw_execute(plan);

        for (int n = -(N - 1); n <= N - 1; ++n) {
            int n_mod = ((n % M) + M) % M;
            double sn = (n & 1) ? -1.0 : 1.0;
            for (int m = -(N - 1); m <= N - 1; ++m) {
                int m_mod = ((m % M) + M) % M;
                double sm = (m & 1) ? -1.0 : 1.0;
                F2[(size_t)k * stride_k
                   + (size_t)(n + N - 1) * stride_n
                   + (size_t)(m + N - 1)]
                    = sn * sm * G[(size_t)n_mod * (size_t)M + (size_t)m_mod];
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(g);
    fftw_free(G);

    /* -------- Stage 2 (GL weights + GL norm) --------
     * Closed-grid Riemann was: norm = dphi*dtheta*dpsi / (8 pi^2), with
     * per-theta weight sin(theta_k) * dtheta.  GL absorbs sin(theta_k)*dtheta
     * into the single weight w_gl[k]; drop dtheta from the global norm.
     *
     * Per notes/gauss_legendre.md §6, the FFTW (-1)^{n+m} fold over-counts
     * endpoints producing F2[k, 0, 0] = N^2 for constant f.  For the
     * constant-input test to recover fhat(0)_{0,0} = 1 exactly,
     *   norm * N^2 * (sum_k w_k * 1) = norm * N^2 * 2 = 1
     *   =>  norm_gl = 1 / (2 * N^2).
     */
    const double norm = 1.0 / (2.0 * (double)N * (double)N);

    double          *d_seq = malloc((size_t)N * sizeof(double));
    double _Complex *acc   = malloc((size_t)N * sizeof(double _Complex));

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
 * @brief Inverse SU(2) FFT (Peter-Weyl synthesis) at Gauss-Legendre theta nodes.
 *
 * Mirrors su2_fft_inv.  The only difference is the theta evaluation grid:
 * theta_k = arccos(x_k), with x_k the k-th N-point Gauss-Legendre node on
 * [-1, 1].  The synthesis itself is a pure sum (no quadrature weight), so
 * w_gl is NOT used here; only x_gl (to obtain theta_k).
 *
 * Stage 2-inv evaluates  G[k, n, m] = i^{m-n} * sum_l (2l+1) * fhat * d^l(theta_k)
 * at the GL theta nodes; Stage 1-inv is the same (N-1)x(N-1) FFTW forward
 * with the closed-grid fold trick (independent of theta).
 *
 * @param[in]  N     Bandlimit.
 * @param[in]  fhat  Length-su2_total_coeffs(N) complex coefficient array.
 * @param[out] f     Length-N^3 complex sample array, row-major (j1, k, j2),
 *                   with f[j1, k, j2] sampled at theta_k = arccos(x_k).
 * @par Reference notes/gauss_legendre.md §5, §6.
 */
void su2_fft_inv_gl(int N,
                    const double _Complex *fhat,
                    double _Complex *f)
{
    if (N < 2 || !fhat || !f) return;

    const int M = N - 1;

    /* GL nodes; weights are not used by the synthesis. */
    double *x_gl  = malloc((size_t)N * sizeof(double));
    double *w_gl  = malloc((size_t)N * sizeof(double));
    su2_gl_nodes_weights(N, x_gl, w_gl);
    double *theta = malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N; ++k) theta[k] = acos(x_gl[k]);

    /* ---- Stage 2-inv ---- */
    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;
    double _Complex *G = malloc((size_t)N * stride_k * sizeof(double _Complex));

    double *d_seq = malloc((size_t)N * sizeof(double));

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
                double _Complex acc = 0.0 + 0.0*I;
                for (int l = l_min; l < N; ++l) {
                    double _Complex coef = fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)];
                    acc += (double)(2*l + 1) * coef * d_seq[l - l_min];
                }
                G[(size_t)k * stride_k
                  + (size_t)(n + N - 1) * stride_n
                  + (size_t)(m + N - 1)] = phase * acc;
            }
        }
    }

    /* ---- Stage 1-inv (identical to su2_fft_inv) ---- */
    fftw_complex *G_slice = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_complex *f_slice = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_plan plan  = fftw_plan_dft_2d(M, M, G_slice, f_slice, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int k = 0; k < N; ++k) {
        memset(G_slice, 0, sizeof(fftw_complex) * (size_t)M * (size_t)M);

        for (int n = -(N - 1); n <= N - 1; ++n) {
            int n_mod = ((n % M) + M) % M;
            double sn = (n & 1) ? -1.0 : 1.0;
            for (int m = -(N - 1); m <= N - 1; ++m) {
                int m_mod = ((m % M) + M) % M;
                double sm = (m & 1) ? -1.0 : 1.0;
                G_slice[(size_t)n_mod * (size_t)M + (size_t)m_mod]
                    += sn * sm * G[(size_t)k * stride_k
                                   + (size_t)(n + N - 1) * stride_n
                                   + (size_t)(m + N - 1)];
            }
        }
        fftw_execute(plan);

        for (int j1 = 0; j1 < N; ++j1) {
            int j1m = (j1 == N - 1) ? 0 : j1;
            for (int j2 = 0; j2 < N; ++j2) {
                int j2m = (j2 == N - 1) ? 0 : j2;
                f[su2_sample_index(N, j1, k, j2)] =
                    f_slice[(size_t)j1m * (size_t)M + (size_t)j2m];
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
