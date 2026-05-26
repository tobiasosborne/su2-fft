/* vis_dump.c -- Synthesise a known input on SU(2), run the fast FFT, dump
 * fhat as text to stdout.  Consumed by bench/visualize.py.
 *
 * Usage:  vis_dump N > spectrum.txt
 *
 * The input is the bandlimited function
 *
 *   f(g) = 1.0   * t^1_{ 0,  0}(g)
 *        + 0.7  * t^2_{ 1, -1}(g)
 *        + 0.5i * t^3_{-2,  1}(g)
 *        + 0.3  * t^3_{ 0,  3}(g)
 *
 * Choosing four narrow Fourier modes makes the spectrum's structure obvious
 * in the plot.  Up to Riemann error these coefficients should re-appear as
 *   fhat(l)[m,n] approx coeff(l,m,n) / (2l+1)
 * in the output (the (2l+1) factor from the Peter-Weyl synthesis identity,
 * paper line 554).
 *
 * Output format -- one line per coefficient:
 *   l  m  n  Re(fhat)  Im(fhat)
 */
#include "su2.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int l, m, n; double _Complex c; } fmode_t;

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 16;
    if (N < 4) { fprintf(stderr, "need N >= 4\n"); return 2; }

    fmode_t modes[] = {
        { 1,  0,  0, 1.0 + 0.0*I },
        { 2,  1, -1, 0.7 + 0.0*I },
        { 3, -2,  1, 0.0 + 0.5*I },
        { 3,  0,  3, 0.3 + 0.0*I },
    };
    int n_modes = (int)(sizeof(modes)/sizeof(modes[0]));

    size_t nsamp  = (size_t)N * N * N;
    size_t ncoeff = su2_total_coeffs(N);
    double _Complex *f    = calloc(nsamp,  sizeof(double _Complex));
    double _Complex *fhat = calloc(ncoeff, sizeof(double _Complex));

    double *phi   = su2_grid_phi(N);
    double *theta = su2_grid_theta(N);
    double *psi   = su2_grid_psi(N);

    /* f(g) = sum_modes coeff * t^l_{n,m}(g)
     *      = sum_modes coeff * P^l_{n,m}(cos theta) * exp(-i(n phi + m psi))
     */
    for (int j1 = 0; j1 < N; ++j1) {
        for (int k = 0; k < N; ++k) {
            for (int j2 = 0; j2 < N; ++j2) {
                double _Complex v = 0.0;
                for (int mi = 0; mi < n_modes; ++mi) {
                    int l = modes[mi].l, m = modes[mi].m, n = modes[mi].n;
                    double _Complex P = su2_wigner_d(l, n, m, theta[k]);
                    double _Complex ex = cexp(-I * (n * phi[j1] + m * psi[j2]));
                    v += modes[mi].c * P * ex;
                }
                f[su2_sample_index(N, j1, k, j2)] = v;
            }
        }
    }

    su2_fft(N, f, fhat);

    /* Header line for the Python loader. */
    printf("# N %d  ncoeff %zu\n", N, ncoeff);
    printf("# input_modes:\n");
    for (int mi = 0; mi < n_modes; ++mi) {
        printf("#   l=%d m=%d n=%d  coeff=%g+%gi  expected_fhat=%g+%gi\n",
               modes[mi].l, modes[mi].m, modes[mi].n,
               creal(modes[mi].c), cimag(modes[mi].c),
               creal(modes[mi].c)/(2*modes[mi].l + 1),
               cimag(modes[mi].c)/(2*modes[mi].l + 1));
    }
    printf("# l m n Re Im\n");
    for (int l = 0; l < N; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int n = -l; n <= l; ++n) {
                double _Complex v = fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)];
                printf("%d %d %d %.17g %.17g\n", l, m, n, creal(v), cimag(v));
            }
        }
    }

    free(phi); free(theta); free(psi);
    free(f); free(fhat);
    return 0;
}
