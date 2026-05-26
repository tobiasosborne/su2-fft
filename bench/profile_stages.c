/* profile_stages.c -- Stage-level instrumentation of the SU(2) fast FFT.
 *
 * Mirrors the structure of src/su2_fft.c locally with clock_gettime
 * fences around each stage, so we can attribute wall time to the
 * candidate hot regions.  Tracks the new O(N^4) Stage 2 introduced in
 * bead su2fft-m21 (notes/wigner_recurrence.md): per (m, n, k) we sweep
 * l = l_min..N-1 via the ascending-l three-term recurrence, instead of
 * paying O(l) per (l, m, n, k).
 *
 * Stage 2 is split into three timed regions, matching the natural
 * decomposition of the new code:
 *
 *   1. wigner seed     -- the two wigner_d_phys evaluations that seed the
 *                         recurrence at l = l_min and l = l_min + 1
 *                         (notes/wigner_recurrence.md §2b, "Recommended").
 *   2. wigner recur    -- the Jacobi-lifted three-term recurrence sweep
 *                         from l = l_min+1 to l = N-2, producing
 *                         d_seq[2..l_max-l_min] (notes §2b).
 *   3. inner product   -- the acc[l] += d_seq[l-l_min] * w accumulator
 *                         plus the per-(m,n,l) final write
 *                         fhat[...] = c * acc[l] (paper.tex line 1361).
 *
 * Splitting inside su2_wigner_d_seq requires inlining its body here so
 * fences can be placed between the seed and the recurrence; this file
 * therefore mirrors src/su2_wigner.c's wigner_d_phys/fact/pow_i as
 * static helpers (CLAUDE.md rule: harness may be a self-contained
 * mirror; no widening of public API for benchmark convenience).
 *
 * Reports nanosecond totals, percent share, and counters of:
 *   seed_calls   = O(N^3) wigner_d_phys calls (two per (m,n,k))
 *   recur_steps  = O(N^4) recurrence steps
 *   inner_ops    = O(N^4) acc multiply-adds
 *
 * Usage:  build/profile_stages [N] [iter]
 */
#include "su2.h"

#include <assert.h>
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static double drand(unsigned *s)
{
    *s = (*s) * 1103515245u + 12345u; return ((*s >> 16) & 0x7FFF) / 32767.0;
}

/* ---- Mirror of src/su2_wigner.c helpers (so timing fences can sit
 *      between the seed evaluations and the recurrence sweep). ---- */

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

/* Physics-convention real Wigner small-d d^l_{n,m}(beta) (Sakurai).
 * Verbatim mirror of the static helper in src/su2_wigner.c. */
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

typedef struct {
    double setup, stage1, wigner_seed, wigner_recur, inner, total;
    long   seed_calls, recur_steps, inner_ops;
} timings_t;

static void run_once(int N, const double _Complex *f, double _Complex *fhat,
                     timings_t *T)
{
    double t0 = now();

    const int M = N - 1;
    double *theta = su2_grid_theta(N);
    double *sin_th = malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N; ++k) sin_th[k] = sin(theta[k]);

    const int    nrange   = 2 * N - 1;
    const size_t stride_n = (size_t)nrange;
    const size_t stride_k = (size_t)nrange * (size_t)nrange;
    double _Complex *F2 = malloc((size_t)N * stride_k * sizeof(double _Complex));

    fftw_complex *g = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_complex *G = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_plan plan  = fftw_plan_dft_2d(M, M, g, G, FFTW_BACKWARD, FFTW_ESTIMATE);

    /* Stage-2 scratch buffers (mirrors src/su2_fft.c). */
    double          *d_seq = malloc((size_t)N * sizeof(double));
    double _Complex *acc   = malloc((size_t)N * sizeof(double _Complex));
    double t1 = now(); T->setup += t1 - t0;

    /* ---- Stage 1: FFTW + closed-grid fold (mirrors src/su2_fft.c). ---- */
    for (int k = 0; k < N; ++k) {
        memset(g, 0, sizeof(fftw_complex) * (size_t)M * (size_t)M);
        for (int j1 = 0; j1 < N; ++j1) {
            int j1m = (j1 == N - 1) ? 0 : j1;
            for (int j2 = 0; j2 < N; ++j2) {
                int j2m = (j2 == N - 1) ? 0 : j2;
                g[(size_t)j1m * (size_t)M + (size_t)j2m]
                    += f[su2_sample_index(N, j1, k, j2)];
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
    fftw_free(g); fftw_free(G);
    double t2 = now(); T->stage1 += t2 - t1;

    /* ---- Stage 2: recurrence sweep over l, three timed regions. ----
     *
     * Loop order mirrors src/su2_fft.c (bead su2fft-m21):
     *   for m
     *     for n
     *       zero acc[l_min..N-1]
     *       for k
     *         seed d_seq[0], d_seq[1] via wigner_d_phys      [wigner_seed]
     *         recurrence sweep d_seq[2..]                    [wigner_recur]
     *         acc[l] += d_seq[l-l_min] * w                   [inner]
     *       apply i^{n-m} phase + norm, write fhat[...]      [inner]
     *
     * The recurrence body is inlined verbatim from src/su2_wigner.c
     * (su2_wigner_d_seq, notes/wigner_recurrence.md §2b) so that
     * clock fences can sit between the seed and the sweep.
     */
    const double dphi   = 2.0 * M_PI / (double)(N - 1);
    const double dtheta =       M_PI / (double)(N - 1);
    const double dpsi   = 2.0 * M_PI / (double)(N - 1);
    const double norm   = dphi * dtheta * dpsi / (8.0 * M_PI * M_PI);

    for (int m_idx = -(N - 1); m_idx <= N - 1; ++m_idx) {
        for (int n_idx = -(N - 1); n_idx <= N - 1; ++n_idx) {
            int an    = abs(n_idx), am = abs(m_idx);
            int Mmax  = (an > am) ? an : am;       /* max(|n|,|m|) */
            int Nmin  = (an < am) ? an : am;       /* min(|n|,|m|) */
            int l_min = Mmax;
            int l_max = N - 1;
            if (l_min > l_max) continue;

            for (int l = l_min; l <= l_max; ++l) acc[l] = 0.0 + 0.0*I;

            int a_ab = abs(m_idx - n_idx);
            int b_ab = abs(m_idx + n_idx);

            for (int k = 0; k < N; ++k) {
                /* --- wigner seed (two wigner_d_phys per (m,n,k)) --- */
                double ts0 = now();
                d_seq[0] = wigner_d_phys(l_min, n_idx, m_idx, theta[k]);
                T->seed_calls++;
                if (l_max > l_min) {
                    d_seq[1] = wigner_d_phys(l_min + 1, n_idx, m_idx, theta[k]);
                    T->seed_calls++;
                }
                double ts1 = now(); T->wigner_seed += ts1 - ts0;

                /* --- wigner recurrence (Jacobi-lifted three-term sweep) ---
                 * notes/wigner_recurrence.md §2b; identical body to
                 * src/su2_wigner.c::su2_wigner_d_seq. */
                double x = cos(theta[k]);
                for (int l = l_min + 1; l <= l_max - 1; ++l) {
                    int kj = l - Mmax;
                    double s = (double)(2*kj + a_ab + b_ab);
                    assert(s != 0.0);

                    double kp1  = (double)(kj + 1);
                    double kab1 = (double)(kj + a_ab + b_ab + 1);
                    double denom_k = 2.0 * kp1 * kab1;

                    double Ak = (s + 1.0) * (s + 2.0) / denom_k;
                    double Bk = (double)(a_ab*a_ab - b_ab*b_ab) * (s + 1.0)
                              / (denom_k * s);
                    double Ck = (double)(kj + a_ab) * (double)(kj + b_ab)
                              * (s + 2.0) / (kp1 * kab1 * s);

                    /* F1 = R(l+1)/R(l); F2r = R(l+1)/R(l-1).  Renamed to
                     * avoid -Wshadow against the F2[] Stage-1 buffer. */
                    double F1num = (double)((l + 1 + Mmax) * (l + 1 - Mmax));
                    double F1den = (double)((l + 1 + Nmin) * (l + 1 - Nmin));
                    double F1    = sqrt(F1num / F1den);
                    double F2rnum = (double)((l + Mmax) * (l - Mmax));
                    double F2rden = (double)((l + Nmin) * (l - Nmin));
                    double F2r    = F1 * sqrt(F2rnum / F2rden);

                    int i = l - l_min;
                    double d_l   = d_seq[i];
                    double d_lm1 = d_seq[i - 1];
                    d_seq[i + 1] = (Ak * x + Bk) * F1 * d_l - Ck * F2r * d_lm1;
                    T->recur_steps++;
                }
                double ts2 = now(); T->wigner_recur += ts2 - ts1;

                /* --- inner product accumulator --- */
                double _Complex w = F2[(size_t)k * stride_k
                                       + (size_t)(n_idx + N - 1) * stride_n
                                       + (size_t)(m_idx + N - 1)] * sin_th[k];
                for (int l = l_min; l <= l_max; ++l) {
                    acc[l] += d_seq[l - l_min] * w;
                    T->inner_ops++;
                }
                double ts3 = now(); T->inner += ts3 - ts2;
            }

            /* Apply i^{n-m} phase + norm; write fhat.  Counted under "inner"
             * (it's the final write of the per-(m,n) accumulator). */
            double tw0 = now();
            int r = ((n_idx - m_idx) % 4 + 4) % 4;
            double _Complex phase;
            switch (r) {
                case 0:  phase =  1.0 + 0.0*I; break;
                case 1:  phase =  0.0 + 1.0*I; break;
                case 2:  phase = -1.0 + 0.0*I; break;
                default: phase =  0.0 - 1.0*I; break;
            }
            double _Complex c = norm * phase;
            for (int l = l_min; l <= l_max; ++l) {
                fhat[su2_coeff_offset(l) + su2_mn_index(l, m_idx, n_idx)]
                    = c * acc[l];
            }
            double tw1 = now(); T->inner += tw1 - tw0;
        }
    }
    double t4 = now();

    free(acc); free(d_seq);
    free(F2);
    free(sin_th);
    free(theta);
    T->total += t4 - t0;
}

int main(int argc, char **argv)
{
    int N    = (argc > 1) ? atoi(argv[1]) : 16;
    int iter = (argc > 2) ? atoi(argv[2]) : 10;

    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fhat = malloc(ncoeff * sizeof(double _Complex));
    unsigned s = 1;
    for (size_t i = 0; i < nsamp; ++i)
        f[i] = (drand(&s) - 0.5) + (drand(&s) - 0.5) * I;

    timings_t T = {0};
    /* Warmup */
    run_once(N, f, fhat, &T); T = (timings_t){0};

    /* Self-consistency: the harness mirrors su2_fft.c by construction.
     * Cross-check against the library at this N: max-diff must be at
     * floating-point noise (~1e-12).  Skepticism (CLAUDE.md rule 3). */
    double _Complex *fhat_ref = malloc(ncoeff * sizeof(double _Complex));
    su2_fft(N, f, fhat_ref);
    double maxdiff = 0.0;
    for (size_t i = 0; i < ncoeff; ++i) {
        double d = cabs(fhat[i] - fhat_ref[i]);
        if (d > maxdiff) maxdiff = d;
    }
    free(fhat_ref);
    if (maxdiff > 1e-10) {
        fprintf(stderr, "profile_stages: harness vs su2_fft max-diff %.3g exceeds 1e-10\n",
                maxdiff);
        free(f); free(fhat);
        return 1;
    }

    for (int i = 0; i < iter; ++i) run_once(N, f, fhat, &T);

    double wigner_total = T.wigner_seed + T.wigner_recur;

    printf("N=%d iter=%d  (harness vs su2_fft max-diff = %.2g)\n",
           N, iter, maxdiff);
    printf("%-18s %12s %8s   %s\n", "stage", "total(s)", "%", "rate");
    printf("-------------------+------------+-------+----------------\n");
    printf("%-18s %12.4f %7.2f%%  --\n",
           "setup",  T.setup,  100.0 * T.setup  / T.total);
    printf("%-18s %12.4f %7.2f%%  fftw + fold/unfold\n",
           "stage1", T.stage1, 100.0 * T.stage1 / T.total);
    printf("%-18s %12.4f %7.2f%%  %.3g calls/sec\n",
           "wigner seed",  T.wigner_seed, 100.0 * T.wigner_seed / T.total,
           T.wigner_seed > 0 ? (double)T.seed_calls / T.wigner_seed : 0.0);
    printf("%-18s %12.4f %7.2f%%  %.3g steps/sec\n",
           "wigner recurrence", T.wigner_recur,
           100.0 * T.wigner_recur / T.total,
           T.wigner_recur > 0 ? (double)T.recur_steps / T.wigner_recur : 0.0);
    printf("%-18s %12.4f %7.2f%%  (seed+recur)\n",
           "  wigner subtotal", wigner_total,
           100.0 * wigner_total / T.total);
    printf("%-18s %12.4f %7.2f%%  %.3g muladds/sec\n",
           "inner product", T.inner, 100.0 * T.inner / T.total,
           T.inner > 0 ? (double)T.inner_ops / T.inner : 0.0);
    printf("%-18s %12.4f %7.2f%%\n", "TOTAL", T.total, 100.0);

    free(f); free(fhat);
    return 0;
}
