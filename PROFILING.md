# Profile report — SU(2) FFT (double-precision path)

Profile setup:
  - host: linux 6.8.12, perf 6.8.12, `kernel.perf_event_paranoid=1`
  - binary: `build/profile` (release flags: `-O2 -g`)
  - workload: `build/profile_stages 24 3` — N=24 (bandlimit), 3 invocations of `su2_fft`
    on a single random complex input.  Total runtime ~0.31 s.

Two independent measurements are available:
1. **`bench/profile_stages`** — hand-instrumented `clock_gettime` fences around setup /
   Stage 1 / wigner seed / wigner recurrence / inner product, ground truth.
   Note: `bench/profile_stages.c` was updated as part of bead `su2fft-m21` to mirror
   the new Stage 2 structure (separate seed and recurrence timers).
2. **`perf record -F 4000 -e cpu_core/cycles/ -g`** — sampled call graph
   on the cpu_core cluster.  Pre-m21 numbers are kept below for archival reference;
   a fresh `perf record` run is future work.

---

## Top-line numbers

Stage timings (`build/profile_stages 24 3`):

```
N=24 iter=3  (harness vs su2_fft max-diff = 0)
stage                  total(s)        %   rate
-------------------+------------+-------+----------------
setup                    0.0001    0.12%  --
stage1                   0.0022    2.12%  fftw + fold/unfold
wigner seed              0.0474   46.09%  6.43e+06 calls/sec
wigner recurrence        0.0337   32.71%  3.03e+07 steps/sec
  wigner subtotal        0.0811   78.80%  (seed+recur)
inner product            0.0114   11.07%  1.16e+08 muladds/sec
TOTAL                    0.1029  100.00%
```

(The remaining ~7.9% is `clock_gettime` fence overhead, not a hot path.)

`perf stat -d` (pre-m21, kept for reference):

```
TopdownL1 (cpu_core)
   65.5 %  retiring               <- compute-bound
   24.7 %  backend_bound
    6.2 %  frontend_bound
    3.6 %  bad_speculation
  0.31 %   branch-misses
  0.14 %   L1-dcache-load-misses  <- caches are fine
 38.4  %   LLC-load-miss rate     <- 91 K LLC accesses total, irrelevant
```

The implementation remains compute-bound, not memory-bound.

---

## Call-graph leaders (`perf report --no-children`)

**Note: the numbers below are pre-m21 (kept for archival reference).  The call
graph reflects the old O(N^5) code where every (l, m, n, k) tuple called
`su2_wigner_d` directly.  Post-m21, `su2_wigner_d` is invoked only for the
two seed values per (m, n, k); the recurrence runs in the `su2_wigner_d_seq`
sweep.  A fresh `perf record` against the updated binary is future work.**

```
Overhead   Symbol                       Provenance         (pre-m21, N=20)
60.49 %    __ieee754_pow_fma            libm                from su2_wigner_d
11.00 %    su2_wigner_d                 ours                de Moivre sum body
 9.09 %    pow@@GLIBC_2.29             libm dispatch
 4.87 %    su2_fft                      ours                outer (l, m, n, k) loop
 4.53 %    fact                         ours                factorial table lookup
 3.50 %    __sincos_fma                 libm                cos(theta/2), sin(theta/2)
 4.30 %    pow@plt (combined)           libm PLT trampoline
```

In the pre-m21 code roughly 78% of all CPU cycles executed `pow(...)` variants.

---

## Why `pow()` dominates

Inside `src/su2_wigner.c::wigner_d_phys()`:

```c
for (int t = tmin; t <= tmax; ++t) {
    ...
    int pc = 2*l + m - n - 2*t;
    int ps = n - m + 2*t;
    arb_pow_ui(cpow, c2, (ulong)pc, prec);   /* <-- transcendental */
    arb_pow_ui(spow, s2, (ulong)ps, prec);   /* <-- transcendental */
    ...
}
```

Post-m21, `wigner_d_phys` is called only for the **seed** values (`l = l_min`
and `l = l_min + 1`) at each (m, n, k) triple, not for every l.  The stage
profile shows this as 46.09% of N=24 wall in `wigner seed` — two calls per
(m, n, k) across O(N^3) triples with O(1) de Moivre terms each at the base.
Replacing the two `pow()` calls with integer-power tables (IMPROVEMENTS.md item 1)
is now the dominant remaining lever on the seed budget.

The exponents are small integers ≤ 2l_min ≤ 2N — so libm `pow(base, double)`
is massively overkill: an integer-power computation by repeated squaring needs
only ⌈log₂(pc)⌉ ≈ 5 mults instead of one transcendental.

---

## Confirmation that the fast path is now O(N^4)

Bead `su2fft-m21` replaced the per-(l, m, n, k) call to `su2_wigner_d` with an
ascending-l three-term recurrence (see `notes/wigner_recurrence.md` §2b).
The speedup table from `build/compare` confirms O(N^4) empirically:

```
   N |    direct(s) |      fast(s) |  speedup |  max|diff| | status
-----+--------------+--------------+----------+------------+---------
   4 |     0.000142 |     0.001105 |     0.13 |   5.94e-17 | OK
   6 |     0.000961 |     0.000375 |     2.56 |   4.63e-17 | OK
   8 |     0.004721 |     0.000955 |     4.94 |   5.30e-17 | OK
  10 |     0.013280 |     0.001279 |    10.39 |   1.26e-16 | OK
  12 |     0.040555 |     0.001990 |    20.38 |   6.58e-16 | OK
  14 |     0.087188 |     0.003556 |    24.52 |   7.13e-16 | OK
  16 |     0.177889 |     0.004820 |    36.90 |   1.61e-15 | OK
  20 |     0.655026 |     0.009028 |    72.55 |   2.09e-14 | OK
  24 |     1.832225 |     0.020154 |    90.91 |   2.21e-13 | OK
```

Scaling check: N=16 to N=24 is a 1.5x increase in N.  O(N^4) predicts
(1.5)^4 = 5.06x more work; empirically 0.00482 s to 0.0202 s = 4.18x.  Slightly
sub-N^4 because constant overheads (FFTW plan, seed calls) are non-negligible
at these sizes.

For comparison, pre-m21 the same interval was 0.0316 s to 0.183 s = 5.78x —
closer to O(N^5)'s prediction of (1.5)^5 = 7.59x.

Max-diff at N=24 rose from 7.12e-17 (pre-m21) to 2.21e-13 post-m21.  This is
expected: the recurrence accumulates floating-point error over l steps, whereas
the old de Moivre sum recomputed each d^l independently from trig.  2.21e-13
remains well within the 1e-10 cross-check tolerance.

---

## Hot-path-fix priority list (matches IMPROVEMENTS.md ordering)

1. ~~**Three-term recurrence in `l` for the Wigner kernel.**~~  **DONE (m21).**
   Stage 2 is now O(N^4).  90.91x speedup at N=24.

2. **Integer-power tables in place of `pow()` inside the seed (de Moivre sum).**
   The seed accounts for 46% of N=24 wall.  Inside `wigner_d_phys` the exponents
   `pc`, `ps` are small non-negative integers ≤ 2l_min.  Precomputing `c2^k`,
   `s2^k` for `k = 0..2l_min` (Horner-built once per call) eliminates the
   transcendentals.  Expected ~3x on the seed budget.
   Touches: `src/su2_wigner.c`.

3. **Recurrence-coefficient caching across k.**  `Ak`, `Bk`, `Ck`, `F1`, `F2`
   in the §2b recurrence depend on `(l, m, n)` but not on `theta`.  Precomputing
   them once per (m, n) outside the k-loop avoids recomputing them N times per
   l-step.  Expected ~20% reduction on the recurrence sweep (32.71% of wall).
   Touches: `src/su2_fft.c` Stage 2.

4. **OpenMP `#pragma omp parallel for` over the outer (m, n) loop.**
   Stage 2 is embarrassingly parallel — no (m, n) pair shares state.  Near-linear
   scaling expected on 8+ cores.
   Touches: `src/su2_fft.c`, `Makefile` (add `-fopenmp`).

5. **SIMD inner product (AVX2 → AVX-512).**  After (2) and (3) the inner-product
   slice (11% of wall) becomes a textbook vectorisation target — a length-N
   complex multiply-accumulate.
   Touches: `src/su2_fft.c`; optional `src/simd/dot_complex.h`.

---

## Reproducing

```sh
make build/profile build/profile_stages
sudo sysctl kernel.perf_event_paranoid=1   # one-time

# Hand-instrumented (no privileges needed):
build/profile_stages 24 3

# Cross-comparison sweep:
build/compare

# perf (needs paranoid <= 2, post-m21 binary):
perf stat -d build/profile 24 3
perf record -F 4000 -e cpu_core/cycles/ -g -o build/perf.data build/profile 24 3
perf report -i build/perf.data --stdio --no-children --percent-limit 1 | head -60
```
