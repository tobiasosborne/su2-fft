/* compare.c -- Cross-comparison of direct FT vs fast FFT on SU(2).
 *
 * Prints a table of N, direct timing, fast timing, speedup, and the maximum
 * absolute error |fhat_fast - fhat_direct|_inf across all coefficients.
 *
 * The error should sit at machine precision (~1e-12) -- the two algorithms
 * compute the same discrete sum (paper line 1316).  The timing ratio should
 * widen with N because direct is O(N^6) and fast is O(N^4).
 */
#include "su2.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static double max_diff(const double _Complex *a, const double _Complex *b, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = cabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void)
{
    int sizes[] = {4, 6, 8, 10, 12, 14, 16, 20, 24};
    int n_sizes = (int)(sizeof(sizes)/sizeof(sizes[0]));

    printf("%4s | %12s | %12s | %8s | %10s | %s\n",
           "N", "direct(s)", "fast(s)", "speedup", "max|diff|", "status");
    printf("-----+--------------+--------------+----------+------------+---------\n");

    srand(20260526);

    for (int idx = 0; idx < n_sizes; ++idx) {
        int N = sizes[idx];
        size_t nsamp  = (size_t)N * N * N;
        size_t ncoeff = su2_total_coeffs(N);

        double _Complex *f   = malloc(nsamp  * sizeof(double _Complex));
        double _Complex *hd  = calloc(ncoeff, sizeof(double _Complex));
        double _Complex *hf  = calloc(ncoeff, sizeof(double _Complex));
        for (size_t i = 0; i < nsamp; ++i) {
            f[i] = ((double)rand()/RAND_MAX - 0.5)
                 + ((double)rand()/RAND_MAX - 0.5) * I;
        }

        double t0 = now_seconds();
        su2_ft_direct(N, f, hd);
        double td = now_seconds() - t0;

        t0 = now_seconds();
        su2_fft(N, f, hf);
        double tf = now_seconds() - t0;

        double err = max_diff(hd, hf, ncoeff);
        const char *status = (err < 1e-9) ? "OK" : "FAIL";

        printf("%4d | %12.6f | %12.6f | %8.2f | %10.2e | %s\n",
               N, td, tf, td/tf, err, status);
        fflush(stdout);

        free(f); free(hd); free(hf);
    }
    return 0;
}
