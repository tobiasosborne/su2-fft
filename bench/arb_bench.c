/* arb_bench.c -- Time the arbitrary-precision FFT at various precisions.
 * Shows wall time and accumulated error-ball radius. */
#include "su2.h"
#include "su2_arb.h"

#include <flint/arb.h>
#include <flint/acb.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static double drand(unsigned *seed)
{
    *seed = (*seed) * 1103515245u + 12345u;
    return ((*seed >> 16) & 0x7FFF) / 32767.0;
}

static double max_ball_radius(acb_srcptr v, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double r = mag_get_d(arb_radref(acb_realref(v + i)));
        double s = mag_get_d(arb_radref(acb_imagref(v + i)));
        if (r > m) m = r;
        if (s > m) m = s;
    }
    return m;
}

int main(void)
{
    int N = 6;
    slong precs[] = {53, 128, 256, 512};
    int n_precs = (int)(sizeof(precs)/sizeof(precs[0]));

    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);

    acb_ptr f    = _acb_vec_init(nsamp);
    acb_ptr fhat = _acb_vec_init(ncoeff);

    unsigned seed = 7;
    for (size_t i = 0; i < nsamp; ++i) {
        arb_set_d(acb_realref(f + i), drand(&seed) - 0.5);
        arb_set_d(acb_imagref(f + i), drand(&seed) - 0.5);
    }

    printf("Arb-precision FFT at N=%d\n", N);
    printf("%6s | %10s | %12s\n", "prec", "time(s)", "max ball");
    printf("-------+------------+-------------\n");
    for (int i = 0; i < n_precs; ++i) {
        slong p = precs[i];
        double t0 = now_seconds();
        su2_fft_arb(N, f, fhat, p);
        double t = now_seconds() - t0;
        printf("%6ld | %10.3f | %12.2e\n", p, t, max_ball_radius(fhat, ncoeff));
    }

    _acb_vec_clear(f, nsamp);
    _acb_vec_clear(fhat, ncoeff);
    return 0;
}
