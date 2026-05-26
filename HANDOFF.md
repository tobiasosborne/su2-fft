# Handoff — for the next agent picking up this project

You are inheriting a working, tested, documented C implementation of the FFT
on SU(2) (arxiv 2605.23923).  Read this file first.  It is the shortest
path to productive work.

---

## 0. Five-minute orientation

```
README.md         <-- what the project is and why anyone cares (read first)
ALGORITHM.md      <-- canonical math + code narrative (read second)
PROFILING.md      <-- where the cycles go (read before optimising)
IMPROVEMENTS.md   <-- the prioritised roadmap (read before picking work)
HANDOFF.md        <-- this file
```

Repo: https://github.com/tobiasosborne/su2-fft (public, AGPL-3.0).
Paper (third-party, NOT in repo; fetch with `curl -L -o source.tar.gz https://arxiv.org/e-print/2605.23923 && tar -xzf source.tar.gz` if you need line-by-line refs):
https://arxiv.org/abs/2605.23923 — Delgado et al., May 2026.

To verify nothing rotted before you start:

```sh
make test                  # 17 tests, 5 binaries; takes ~5 s
make bench && build/compare  # benchmark sweep, ~3 s
```

If those don't both end in `ALL TESTS PASSED` + a clean table, **fix that
first**.  Do not start new work on a broken baseline.

---

## 1. Architecture in one paragraph

Four implementations of the same algorithm:

| path           | precision        | hot stage 1            | hot stage 2          |
|----------------|------------------|------------------------|----------------------|
| `su2_ft_direct`| double           | --                     | brute O(N^6)         |
| `su2_fft`      | double           | FFTW3 2-D backward     | O(N^4) recurrence    |
| `su2_ft_direct_arb` | arb (acb_t) | --                     | brute O(N^6)         |
| `su2_fft_arb`  | arb (acb_t)      | `acb_dft_prod` + conj  | O(N^5) wigner-bound  |

All four compute the *same discrete sum* (paper line 1316).  Cross-checks in
`tests/test_fft.c` (double-direct vs double-fft) and `tests/test_arb.c`
(arb-direct vs arb-fft, plus arb vs double) lock that down to ~1e-10.

Bead `su2fft-m21` (ascending-l three-term recurrence) shipped: the double-precision
fast path is now **honestly O(N^4)**.  90.91x speedup at N=24 vs the O(N^6) direct
path.  The arb path remains O(N^5).

---

## 2. The conventions you will trip over

These are non-obvious and the lab notebook is them.  Each is also called out
in source comments at point of use.

1. **Closed grid.**  `phi[j] = -pi + j * 2pi/(N-1)`.  Endpoints `j=0` and
   `j=N-1` are the *same point* on the torus.  `su2_fft.c` folds them and
   applies a `(-1)^{n+m}` phase to recover the closed-grid DFT from an
   `(N-1) x (N-1)` FFTW plan.  If you change the grid, change both fold and
   phase together.

2. **Paper's P^l_{n,m} vs Sakurai's d^l_{n,m}.**  They differ by a phase:
   `P^l_{n,m}(cos beta) = i^{m-n} * d^l_{n,m}(beta)`.  The implementation
   carries Sakurai's real `d` through the de Moivre sum and applies
   `pow_i(m-n)` at the end.  Get this wrong and the direct/fast cross-check
   fails by a unit-magnitude factor (it will look "right but rotated").

3. **Factorial table is double, capped at 50 (now 100 in wigner_arb).**
   `src/su2_wigner.c::fact()` is a precomputed array.  For l > 50 you need
   either `tgamma` or to enlarge the table.  Tests cover up to l=24
   indirectly via N=24 in `compare.c`.

4. **`pow(0.0, 0)` returns 1.0** in glibc IEEE 754 mode — this is exploited
   at the theta=pi endpoint in the unitarity test (see
   `test_wigner.c::test_wigner_unitarity_columns` which skips endpoints to
   avoid relying on this).

5. **`acb_dft_prod` is forward DFT.**  Backward is `conj(forward(conj(.)))`.
   `src/su2_fft_arb.c` does exactly this; do not "simplify" it.

6. **Integer-only l.**  Half-integer support is a substantial change
   (bead `su2fft-n8e`) — factorials become Gammas, the phi/psi grids
   become non-periodic on 2pi.  Three downstream beads block on it.

7. **The closed-grid (N/(N-1))^2 factor** is the systematic Riemann error
   that makes `test_ft_direct_constant` use the modified target `(N/(N-1))^2`
   rather than 1.0.  This is intrinsic to the paper's grid, not a bug.

---

## 3. How to actually make progress

Beads (`bd`) is the issue tracker.  21 issues, 16 ready, 5 blocked by
prerequisites.  Mandatory commands:

```sh
bd ready -p 1            # P1 actionable work (start here)
bd show <id>             # read the full bead body
bd update <id> --status in_progress   # claim a bead BEFORE coding
bd close <id> --reason "shipped in <commit>"  # when done
bd dep tree <id>         # see prerequisites if blocked
```

**`su2fft-m21` shipped.**  The ascending-l three-term Wigner recurrence
(`notes/wigner_recurrence.md` §2b, Jacobi-lifted) replaced the per-(l,m,n,k)
call to `wigner_d_phys` in Stage 2.  Result: 90.91x speedup at N=24
(direct 1.83 s vs fast 0.020 s), and the implementation matches the paper's
O(N^4) asymptotic for the first time.  Stage breakdown at N=24: wigner seed
46%, recurrence 33%, inner product 11%, Stage 1 2%.

**`su2fft-dyi` shipped.**  The libm `pow(c2, pc) * pow(s2, ps)` calls in
`wigner_d_phys` were replaced with an inline `ipow(double x, int k)` (repeated
squaring, ⌈log₂(k)⌉ ≈ 5–6 mults for k ≤ 50).  The pre-m21 premise of "~2–3x
seed speedup" did not hold post-m21: with `wigner_d_phys` called only O(N^3)
times as seeds (not O(N^5) times across all l), `pow()` was already a small
fraction of seed cost.  Empirical result: ~4% wall improvement at N=24 (warm-run
`bench/compare` ~91x → ~100–108x; `build/profile_stages 24 10` per-FFT wall
34.3 ms → ~33.0 ms).  Seed % unchanged at 46% — the seed is now
trig-dominated (`cos(beta/2)` + `sin(beta/2)`), not `pow()`-dominated.

**Recommended first move: `su2fft-xxb` (trig-sharing across the seed pair).**
Both seed calls at a (m, n, k) triple evaluate `cos(theta_k/2)` and
`sin(theta_k/2)` independently, despite sharing the same `theta_k`.  Hoisting
the trig pair out of the (m, n) inner loops eliminates O(N^2) redundant trig
calls per k-slice.  Expected ~10% reduction in seed cost (~3 ms at N=24).
Touches `src/su2_wigner.c` and `src/su2_fft.c` Stage 2 only.

After that, bead `su2fft-cvh` (SIMD) is the next move on the recurrence and
inner-product slices, and `su2fft-lg8` (OpenMP over the (m, n) loop) is a
near-linear scaling win at the cost of added complexity.

---

## 4. The TDD/literate-programming discipline that produced this codebase

This is what the user expects and what kept the cross-comparison clean
through five rewrites of the Wigner routine.  Don't drop it.

1. **Red first.**  Add a failing test for the property the new code must
   uphold, *before* writing the new code.  See `tests/test_arb.c` for the
   pattern: `test_arb_direct_vs_fast` is the gold-standard property
   ("two implementations agree to floating-point noise on random input").

2. **Read ground truth.**  Before changing a math routine, read the paper
   line being implemented.  Source comments cite `paper.tex` line numbers;
   open the paper at that line.  When the paper formula and code don't
   match, the code is wrong — don't "fix" it by tweaking until tests pass.

3. **Cross-checks > unit tests.**  Each algorithm has at least one
   independent implementation it agrees with.  Direct vs FFT is the main
   one; the Wigner closed-form has Legendre P_l as the m=n=0 special case;
   the arb path has the double path as its prec=53 anchor.  When you add
   a new path, add its cross-check immediately.

4. **One bead, one PR, one commit.**  Bead `su2fft-m21` should be its own
   commit message line.  No drive-by cleanup; the diff should be readable
   in one sitting.  Drive-by improvements go in their own beads first.

5. **Literate updates.**  When you touch the algorithm, update `ALGORITHM.md`.
   When you touch performance, update `PROFILING.md`.  When you finish a bead
   that opens up new work, add the new beads with `bd create`.  The
   document is the contract; if the document is stale, future-you cannot
   recover state from `git log`.

---

## 5. Style preferences observed from the user

- Concrete numbers always.  "1e-17 max-diff at N=24" not "very accurate".
- No emojis.  No marketing words.  "Seriously impressive" was the user's
  own phrasing for the roadmap, but the code itself should read like
  SQLite / TigerBeetle docs.
- Cite paper line numbers in source comments.  They survive across the
  paper being moved out of the public repo.
- Prefer C11, `-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes -O2 -g`.
  These catch real bugs (the `mode_t` collision with sys/types.h that I hit
  was caught by `-Wshadow`).
- Don't add features that aren't paid for by a passing test.  Every feature
  in the codebase right now has a test or a benchmark that exercises it.
- The user values FLINT explicitly.  Don't suggest replacing the arb path
  unless you've shipped a measurable improvement first.

---

## 6. Surprises and lessons from this iteration

These cost time when you don't know them.

1. **`pow()` is shockingly expensive — but context matters.**  Pre-m21,
   60.5% of cycles were in `__ieee754_pow_fma` at N=20 (O(N^5) calls to
   `wigner_d_phys`).  Post-m21, `wigner_d_phys` is called only O(N^3) times
   as seeds, so `pow()` dropped from dominant cost to a minor contributor.
   Bead `su2fft-dyi` replaced it with inline `ipow` (repeated squaring)
   anyway — cleaner code and ~4% wall improvement — but not the ~2–3x
   originally predicted.  The lesson: always profile the post-refactor call
   pattern before sizing an optimisation.  Always reach for repeated-squaring
   on integer exponents; never call `pow(x, k)` for `k` integer in a hot loop.

2. **The Rodrigues form of P^l_{nm} catastrophically cancels by l~10.**
   It was the first implementation; it passes tests up to l=8 and silently
   produces NaNs by l=15.  The de Moivre sum is mathematically equivalent
   and numerically stable.

3. **Closed-grid samples are not independent.**  The N-point closed grid
   on a 2pi-periodic angle has rank N-1.  The fold trick recovers a clean
   FFTW plan; the alternative is a slow O(N^2) DFT.  Half-integer l support
   will need a different grid entirely (bead `su2fft-n8e`).

4. **`fftw_complex` is layout-compatible with `double _Complex`.**  This
   isn't documented loudly but FFTW relies on it.  The arb path uses
   `acb_dft_prod` which is *not* layout-compatible — different storage
   entirely.

5. **`perf` needs `kernel.perf_event_paranoid <= 2`.**  If you get
   "Access to performance monitoring and observability operations is
   limited" run:
   ```
   sudo sysctl kernel.perf_event_paranoid=1
   ```
   For senior-level profiling you also want call-graph sampling at high
   rate: `perf record -F 4000 -e cpu_core/cycles/ -g`.  On Intel hybrid
   CPUs you must specify `cpu_core` or you'll get a useless 12-sample
   trace from the efficiency cluster.

6. **`bd create` accepts `-f markdown_file` for batch input.**  If you have
   many issues to register at once, write them as a markdown file with one
   heading per issue and pipe it in.  This is much faster than 16 shell
   commands.

7. **The af proof workspace under `proof/` is reference-only.**  The user
   explicitly said "no need to run verifiers, the af tree is for reference
   purposes".  Don't spend time validating it.

---

## 7. What "done" looks like for the top priorities

If you ship one bead per day for two weeks, you can credibly claim the
implementation matches the paper.  Concrete acceptance for the first few:

**`su2fft-m21` (Wigner recurrence) — DONE.**  All acceptance criteria met:
- `tests/test_fft.c::test_fft_matches_direct_random` passes at 1e-10 tolerance.
- `bench/profile_stages 24 3` reports wigner seed 46% + recurrence 33% = 79% total;
  no single `wigner build` phase, but the combined wigner subtotal is well under
  the old 98%.
- `bench/compare` at N=24 shows 90.91x speedup (criterion was > 50x).
- Stage timings updated in PROFILING.md.

**`su2fft-dyi` (inline `ipow`) — DONE.**
Shipped: `src/su2_wigner.c` no longer calls `pow()` inside the de Moivre
t-loop.  `bench/compare` max-diff at N=24 remains below 1e-10.  Wall
improvement ~4% (34.3 ms → ~33.0 ms per FFT at N=24); seed % unchanged at
46% because the seed is now trig-dominated, not `pow()`-dominated.  The
original "seed < 20%" criterion was set against the pre-m21 calling pattern
(O(N^5) calls) and was not achievable post-m21 without trig-sharing
(`su2fft-xxb`).

**`su2fft-3lx` (Inverse FFT) — DONE when:**
- `tests/test_roundtrip.c` exists and shows
  `||su2_fft_inv(su2_fft(f)) - f|| < 1e-12` at N=8, 16, 24.
- Three downstream beads (`su2fft-d7v`, `su2fft-31x`, `su2fft-5fb`)
  unblock automatically (`bd dep tree <id>` shows them ready).

**`su2fft-n8e` (Half-integer l) — DONE when:**
- A new test verifies the spin-1/2 representation explicitly:
  for `f(g) = u_{00}(g) = e^{i(phi+psi)/2} cos(theta/2)` (the (0,0)
  entry of the defining rep, paper line 503), `fhat(1/2)[0,0]` is the
  expected scalar to ~1e-10.
- Two downstream beads (`su2fft-erv`, `su2fft-5fb`) unblock.

---

## 8. Things explicitly out of scope unless the user asks

- Replacing FFTW3 with anything else (user wants FFTW3 + FLINT).
- Replacing FLINT with anything else (user explicitly stated preference).
- Adding a custom DSL or build system (Makefile is enough).
- Rewriting in Rust / Zig / C++ (user wrote this in C deliberately).
- The af proof workspace (reference only; do not modify).

If a bead requires touching one of these, surface the question first and
wait for an answer.

---

## 9. Where the user is reachable for ambiguities

Pull requests / issues at https://github.com/tobiasosborne/su2-fft .
For paper-math disambiguation, the paper itself plus the four notes/
summaries under `notes/` give independent renderings of every key formula.
When the paper and a note disagree, the paper is canonical.

Welcome aboard.
