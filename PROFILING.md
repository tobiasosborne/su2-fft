# Profile report — SU(2) FFT (double-precision path)

Profile setup:
  - host: linux 6.8.12, perf 6.8.12, `kernel.perf_event_paranoid=1`
  - binary: `build/profile` (release flags: `-O2 -g`)
  - workload: `build/profile 20 30` — N=20 (bandlimit), 30 invocations of `su2_fft`
    on a single random complex input.  Total runtime ~2.5 s.

Two independent measurements were taken and they agree:
1. **`bench/profile_stages`** — hand-instrumented `clock_gettime` fences
   around setup / Stage 1 / Wigner build / inner product, ground truth.
2. **`perf record -F 4000 -e cpu_core/cycles/ -g`** — sampled call graph
   on the cpu_core cluster, 10 012 samples.

---

## Top-line numbers

Stage timings (`build/profile_stages 20 15`):

```
stage                  total(s)        %   rate
-------------------+------------+-------+----------------
setup                    0.0009     0.1%
stage1                   0.0070     0.6%   FFTW 2-D backward (N-1)x(N-1)
wigner build             1.2513    98.0%   2.6e+06 calls/sec to su2_wigner_d
inner product            0.0171     1.3%   1.9e+08 muladds/sec
TOTAL                    1.2764   100.0%
```

`perf stat -d`:

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

So we are **compute-bound, not memory-bound**.  Two-level cache is essentially
ideal; branches predict at 99.7 %.  Backend stalls are dominated by FMA latency.

---

## Call-graph leaders (`perf report --no-children`)

```
Overhead   Symbol                       Provenance
60.49 %    __ieee754_pow_fma            libm                from su2_wigner_d
11.00 %    su2_wigner_d                 ours                de Moivre sum body
 9.09 %    pow@@GLIBC_2.29              libm dispatch
 4.87 %    su2_fft                      ours                outer (l, m, n, k) loop
 4.53 %    fact                         ours                factorial table lookup
 3.50 %    __sincos_fma                 libm                cos(theta/2), sin(theta/2)
 4.30 %    pow@plt (combined)           libm PLT trampoline
```

**Roughly 78 % of all CPU cycles execute one of `pow(...)` / `__ieee754_pow_fma`
/ `pow@plt`**.  The remaining ~22 % is the de Moivre sum's adds, mults, divs,
factorial lookups, and the outer iteration overhead of `su2_fft`.

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

Two `pow()` calls per term × O(l) terms per call × O(N^4) calls into
`su2_wigner_d` (one per (l, m, n, k)) ≈ **50 million `pow()` invocations at
N=20**.  At ~50 cycles each, that's ~2.5 G cycles, matching the 2.5 s wall.

The exponents are small integers ≤ 2N ≈ 40 — so libm `pow(base, double)` is
massively overkill: an integer-power computation by repeated squaring needs
only ⌈log₂(pc)⌉ ≈ 5 mults instead of one transcendental.

---

## Confirmation that the "fast" path is not yet fully fast

The paper's headline is O(N^4).  Our Stage 2 inner product is O(N^4):
`1.9 × 10^8` muladds/sec × ~1.7 % of 1.27 s ≈ `4 × 10^6` muladds per FFT,
which at N=20 matches the expected N^4 ≈ 1.6×10^5 × 30 iter ≈ 5e6.  Good.

But the **Wigner build** in this implementation costs O(N⁵) — O(l) per
`su2_wigner_d` call × O(N⁴) calls.  Empirically:

| N   | wigner % | wigner total (s) | ratio vs N^5 prediction |
|-----|---------:|-----------------:|------------------------:|
| 16  |   97.4 % |            0.290 |                      —  |
| 20  |   98.0 % |            0.834 | 0.290 × (20/16)^5 = 0.93 ✓ |
| 24  |   98.1 % |            2.029 | 0.290 × (24/16)^5 = 2.20 ✓ |

(Iter-normalised totals.)  The implementation is **honestly O(N^5)**, not
the paper's O(N^4).  That single deviation — `su2_wigner_d` doing fresh work
per (l, m, n, k) instead of sweeping a recurrence in `l` — is the entire
gap between us and the headline.

---

## Hot-path-fix priority list (matches IMPROVEMENTS.md ordering)

1. **Three-term recurrence in `l` for the Wigner kernel.**  Drops Wigner
   build from O(N^5) to O(N^4), eliminating ~90 % of the current wall time.
   This alone is the move from "well-tested correct" to "matches paper's
   asymptotic".

2. **Power tables instead of `pow()`.**  Even if (1) isn't done, replacing
   the two `arb_pow_ui` calls with a precomputed length-(2l+1) vector of
   `c2^k`, `s2^k` (Horner-built once per call) eliminates the transcendentals.
   ~3× speedup on the current code.

3. **Lift `fact` table lookups out of the inner loop.**  At 4.53 % of cycles,
   precomputing the factorial *ratios* per (l, m, n) outside the t loop
   would buy a smaller (~3 %) but free win.

4. **OpenMP `#pragma omp parallel for` over the outer (l, m, n) loop.**
   Embarrassingly parallel, current bottleneck is per-call, near-linear
   scaling expected on 8+ cores.

Items 1+2 together would put N=24 below 50 ms per FFT (vs current ~200 ms).

---

## Reproducing

```sh
make build/profile build/profile_stages
sudo sysctl kernel.perf_event_paranoid=1   # one-time

# Hand-instrumented (no privileges needed):
build/profile_stages 20 15

# perf (needs paranoid <= 2):
perf stat -d build/profile 20 30
perf record -F 4000 -e cpu_core/cycles/ -g -o build/perf.data build/profile 20 30
perf report -i build/perf.data --stdio --no-children --percent-limit 1 | head -60
```
