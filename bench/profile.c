/* profile.c -- Sustained FFT load for perf.
 * Runs su2_fft N_ITER times at fixed bandlimit on the same random input. */
#include "su2.h"
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef PROFILE_N
#define PROFILE_N 20
#endif
#ifndef PROFILE_ITER
#define PROFILE_ITER 30
#endif

static double drand(unsigned *s)
{
    *s = (*s) * 1103515245u + 12345u;
    return ((*s >> 16) & 0x7FFF) / 32767.0;
}
static double now_seconds(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

int main(int argc, char **argv)
{
    int N    = (argc > 1) ? atoi(argv[1]) : PROFILE_N;
    int iter = (argc > 2) ? atoi(argv[2]) : PROFILE_ITER;

    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = malloc(nsamp  * sizeof(double _Complex));
    double _Complex *fhat = malloc(ncoeff * sizeof(double _Complex));

    unsigned s = 1;
    for (size_t i = 0; i < nsamp; ++i)
        f[i] = (drand(&s) - 0.5) + (drand(&s) - 0.5) * I;

    /* Warm up FFTW plans, cache, factorial table. */
    su2_fft(N, f, fhat);

    double t0 = now_seconds();
    for (int it = 0; it < iter; ++it) su2_fft(N, f, fhat);
    double t  = now_seconds() - t0;

    printf("profile: N=%d iter=%d total=%.3fs per_call=%.3fs\n",
           N, iter, t, t / iter);
    free(f); free(fhat);
    return 0;
}
