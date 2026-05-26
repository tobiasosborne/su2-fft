/* profile_stages.c -- Stage-level instrumentation of the SU(2) fast FFT.
 *
 * Reproduces the structure of src/su2_fft.c locally with clock_gettime
 * fences around each stage, so we can attribute wall time to the four
 * candidate hot regions:
 *
 *   1. Stage 1 FFTW: plan creation + N executions on (N-1)x(N-1) folded grid.
 *   2. Stage 2 Wigner build: su2_wigner_d() calls over all (l, m, n, k).
 *   3. Stage 2 inner product: the per-coefficient dot product loop.
 *   4. Memory + grid setup.
 *
 * Reports nanosecond totals + per-call averages + percent share.  No perf,
 * no kernel involvement -- pure libc clock_gettime(CLOCK_MONOTONIC).
 *
 * Usage:  build/profile_stages [N] [iter]
 */
#include "su2.h"

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

typedef struct {
    double setup, stage1, wigner, inner, total;
    long   wigner_calls, inner_ops;
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
    double t1 = now(); T->setup += t1 - t0;

    /* ---- Stage 1 ---- */
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

    /* ---- Stage 2: Wigner build (separated for measurement) ---- */
    /* Build per-(m, n, k) vector of conj(P^l_{n,m}(cos theta_k)) * sin_th[k] up
     * front -- this isolates wigner cost from the inner product cost.  Memory
     * is O(N^4) which is acceptable for N <= ~32. */
    size_t kvec_stride = (size_t)N;                        /* per (l,m,n) row */
    size_t kvec_per_l  = (size_t)(2 * (N - 1) + 1) * (size_t)(2 * (N - 1) + 1);
    size_t kvec_n_total = (size_t)N * kvec_per_l * kvec_stride;
    double _Complex *Pk = malloc(kvec_n_total * sizeof(double _Complex));

    for (int l = 0; l < N; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int n = -l; n <= l; ++n) {
                size_t base = ((size_t)l * kvec_per_l
                               + (size_t)(m + N - 1) * (size_t)(2 * N - 1)
                               + (size_t)(n + N - 1)) * kvec_stride;
                for (int k = 0; k < N; ++k) {
                    Pk[base + (size_t)k] =
                        conj(su2_wigner_d(l, n, m, theta[k])) * sin_th[k];
                    T->wigner_calls++;
                }
            }
        }
    }
    double t3 = now(); T->wigner += t3 - t2;

    /* ---- Stage 2: inner product loop ---- */
    const double dphi   = 2.0 * M_PI / (double)(N - 1);
    const double dtheta =       M_PI / (double)(N - 1);
    const double dpsi   = 2.0 * M_PI / (double)(N - 1);
    const double norm   = dphi * dtheta * dpsi / (8.0 * M_PI * M_PI);

    for (int l = 0; l < N; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int n = -l; n <= l; ++n) {
                size_t base = ((size_t)l * kvec_per_l
                               + (size_t)(m + N - 1) * (size_t)(2 * N - 1)
                               + (size_t)(n + N - 1)) * kvec_stride;
                double _Complex acc = 0.0;
                for (int k = 0; k < N; ++k) {
                    acc += Pk[base + (size_t)k]
                         * F2[(size_t)k * stride_k
                              + (size_t)(n + N - 1) * stride_n
                              + (size_t)(m + N - 1)];
                    T->inner_ops++;
                }
                fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)] = norm * acc;
            }
        }
    }
    double t4 = now(); T->inner += t4 - t3;

    free(Pk);
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

    for (int i = 0; i < iter; ++i) run_once(N, f, fhat, &T);

    printf("N=%d iter=%d\n", N, iter);
    printf("%-18s %12s %8s   %s\n", "stage", "total(s)", "%", "rate");
    printf("-------------------+------------+-------+----------------\n");
    printf("%-18s %12.4f %7.1f%%  --\n", "setup",   T.setup,   100*T.setup/T.total);
    printf("%-18s %12.4f %7.1f%%  fftw + fold/unfold\n",
           "stage1",  T.stage1,  100*T.stage1/T.total);
    printf("%-18s %12.4f %7.1f%%  %.2g calls/sec\n",
           "wigner build", T.wigner, 100*T.wigner/T.total,
           T.wigner_calls / T.wigner);
    printf("%-18s %12.4f %7.1f%%  %.2g muladds/sec\n",
           "inner product", T.inner, 100*T.inner/T.total,
           T.inner_ops / T.inner);
    printf("%-18s %12.4f %7.1f%%\n", "TOTAL", T.total, 100.0);

    free(f); free(fhat);
    return 0;
}
