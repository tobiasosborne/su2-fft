# Handoff — for the next agent picking up this project

You are inheriting a working, tested, documented C implementation of the FFT
on SU(2) (arxiv 2605.23923), plus Julia bindings, plus convolution and
spherical-harmonic specialisations.  Read this file first.

---

## 0. Five-minute orientation

```
README.md             <-- what the project is and why anyone cares
ALGORITHM.md          <-- canonical math + code narrative; every formula cited
PROFILING.md          <-- where the cycles go
IMPROVEMENTS.md       <-- the prioritised roadmap
notes/                <-- per-bead design briefs (wigner_recurrence.md,
                         inverse_fft.md, gauss_legendre.md, half_integer.md)
HANDOFF.md            <-- this file
```

Repo: https://github.com/tobiasosborne/su2-fft (public, AGPL-3.0).
Paper (NOT in repo; AGPL conflicts with arxiv): https://arxiv.org/abs/2605.23923 — Delgado et al., May 2026.
Fetch if you need line refs:
```sh
curl -L -o source.tar.gz https://arxiv.org/e-print/2605.23923
tar -xzf source.tar.gz
```

Verify the baseline before starting any work:

```sh
make test          # 49 tests, 9 binaries, ~5 s
make bench         # cross-comparison + arb timing, ~3 s
cd julia && julia --project=. -e 'using Pkg; Pkg.test()'   # 744 tests, ~8 s
```

If any of those don't pass, **fix that first**.

---

## 1. Architecture (current state)

Four implementations of the SU(2) FFT, plus three derived transforms:

| path                  | what                              | cost          | status     |
|-----------------------|-----------------------------------|---------------|------------|
| `su2_ft_direct`       | direct O(N^6) reference           | brute         | shipped    |
| `su2_fft`             | fast forward (closed-grid theta)  | O(N^4)        | shipped    |
| `su2_fft_inv`         | Peter-Weyl synthesis (closed)     | O(N^4)        | shipped (bead `3lx`) |
| `su2_fft_gl`          | forward with Gauss-Legendre theta | O(N^4)        | shipped (bead `ega`) |
| `su2_fft_inv_gl`      | inverse at GL theta nodes         | O(N^4)        | shipped (bead `ega`) |
| `su2_ft_direct_arb`   | arb (acb_t) direct reference      | brute O(N^6)  | shipped    |
| `su2_fft_arb`         | arb fast                          | O(N^5) wigner | shipped (recurrence not yet ported — `2r2`) |
| `su2_convolve`        | per-l matrix product in spectrum  | O(N^4)        | shipped (bead `d7v`) |
| `su2_fft_sphere(_inv)`| S^2 = SU(2)/U(1) thin wrapper     | O(N^4)        | shipped (bead `5fb`) |
| `su2_wigner_d_half`   | half-integer Wigner-d evaluation  | O(l)          | shipped (bead `n8e` Tier 1) |
| `su2_gl_nodes_weights`| GL nodes via Newton iteration     | O(N^2)        | shipped (bead `ega`) |

Julia bindings (`julia/SU2FFT.jl`) wrap all the above via `ccall` through
`build/libsu2.so`.  Sample layout convention: Julia
`f[j2+1, k+1, j1+1] = sample at Euler grid (j1, k, j2)`; this matches
C row-major `f[j1*N*N + k*N + j2]` without any permutation.

---

## 2. THE STRUCTURAL INSIGHT FROM THE LAST SESSION (READ THIS)

**The closed-grid phi/psi aliasing is the dominant precision floor — not
theta quadrature, not `pow()`.**  This was non-obvious; the previous-PROFILING
data pointed at `pow()` (60-78 % of cycles), which made `dyi` look like an
easy ~2x speedup.  Three iterations later, the actual story is:

- `m21` (Wigner recurrence) shifted the calling pattern so `pow()` was no
  longer the bottleneck.  `dyi` (ipow) ended up a ~4 % wash because the
  seed t-range collapsed to 1-2 terms post-m21.
- `ega` (Gauss-Legendre theta) fixed the DC entry exactly
  (forward(constant) = 1.0 instead of 1.31 at N=8).  But the leakage to
  non-DC modes stayed at ~0.197.
- Reason: the closed N-point grid in phi/psi resolves only ~N independent
  modes via the FFTW + fold + `(-1)^{n+m}` phase trick, but the SU(2)
  bandlimit demands |m|, |n| up to N-1 (= 2N-1 modes).  Modes with
  |m| > N/2 alias.  Empirical at N=8:
  - Constant-input leakage: ~0.197.
  - Single-coefficient roundtrip at (l, 0, 0): max ~0.344 across l.
  - Worst case at (l=N-1, m=±(N-1), 0): ~3.

**This is bead `su2fft-0t1` (P1).**  Two structural fixes proposed:
(a) widen the phi/psi grid to 2N-1 points (open or closed); or
(b) restrict bandlimit to |m|, |n| < N/2.  Pick one, ship it, then the
forward+inverse roundtrip will hit machine precision and the application
beads (`d7v`, `5fb`) gain usable end-to-end accuracy.

**Synthesis is mathematically exact.**  All single-coefficient analytical
tests hit 1e-12 to 1e-13.  The structural floor is in the FORWARD ANALYSIS
under the current grid, not in the math.

---

## 3. The conventions you will trip over

These are non-obvious and the lab notebook is them.  Each is also called out
in source comments at point of use.

1. **Closed grid.**  `phi[j] = -pi + j*2pi/(N-1)`.  Endpoints `j=0` and
   `j=N-1` are the *same point* on the torus.  `su2_fft.c` folds them and
   applies a `(-1)^{n+m}` phase to recover a DFT from an `(N-1)x(N-1)` FFTW
   plan.  This is the source of the aliasing in §2.

2. **Paper's P^l_{n,m} vs Sakurai's d^l_{n,m}.**  They differ by a phase:
   `P^l_{n,m}(cos beta) = i^{m-n} * d^l_{n,m}(beta)`.  The implementation
   carries Sakurai's real `d` through the de Moivre sum / Wigner recurrence
   and applies `pow_i(m-n)` at the end.

3. **Factorial table capped at 100** (`FACT_MAX` in `src/su2_wigner.c`).
   `wigner_d_phys` asserts `l <= FACT_MAX`.  Above that, use
   `su2_wigner_d_half` (which uses `tgamma`) or extend the table.

4. **`pow(0, 0) = 1` in glibc IEEE 754 mode** — exploited at theta=0 and
   theta=pi endpoints in the de Moivre sum.  `ipow(0, 0) = 1` via the
   `r = 1.0` initial value and the `while (k > 0)` early-exit; matches.

5. **`acb_dft_prod` is forward DFT.**  Backward is
   `conj(forward(conj(.)))`.  `src/su2_fft_arb.c` does exactly this; do not
   "simplify" it.

6. **Integer-l for the full FFT; half-integer Wigner-d via `su2_wigner_d_half`.**
   Bead `n8e` Tier 1 (the evaluator) shipped.  Tier 2 (full half-integer
   forward+inverse with 4pi-periodic phi/psi grid and factorial→Gamma in the
   recurrence) is bead `u9q`.  `erv` (SO(3) FFT via Z_2 quotient) is blocked
   on `u9q`.

7. **`(N/(N-1))^2` Riemann factor**, see §2.  GL theta (`ega`) eliminated
   the theta part of this; the phi/psi part remains (bead `0t1`).

8. **`pow()` is no longer the bottleneck.**  Post-`m21` + `dyi`, the seed
   calls account for ~46 % of N=24 wall but are dominated by trig
   (`cos(beta/2)`, `sin(beta/2)`) and sqrt, NOT pow.  Bead `xxb` (share
   trig across seed pair) is the next per-call lever.

9. **Julia / libgmp ABI workaround.**  Julia 1.12 ships `libgmp.so.10`
   older than system FLINT.  `SU2FFT.jl::__init__` pre-loads
   `/lib/x86_64-linux-gnu/libgmp.so.10` with `RTLD_GLOBAL` then `dlopen`s
   libsu2 with `RTLD_DEEPBIND`.  Works on Debian/Ubuntu; portable fix is
   bead `e5z` (BinaryBuilder, libflint_jll, or static link).

---

## 4. How to actually make progress

```sh
bd ready -p 1                     # P1 actionable work
bd show <id>                      # read the full bead body + design notes
bd update <id> --status in_progress
bd close <id> --reason "shipped in <hash>"
bd dep tree <id>                  # see what's blocking or blocked
```

**Recommended next move: `su2fft-0t1` (Fully resolved phi/psi grid).**
This is the single biggest precision lift available.  Two design choices
to pick from; recommend prototyping (a) — widen the grid to 2N-1 in phi/psi
— since it preserves the existing bandlimit.  Touches `src/su2_fft.c`
Stage 1 (FFTW plan dimensions, fold trick), `src/su2_fft_inv.c`,
the `su2_total_coeffs` calculation may or may not change depending on
which API choice you take.  Tests: replace the current "documented floor"
testsets with strict 1e-12 roundtrip assertions.  Once it lands, run all
existing tests; the analytical synthesis tests should still pass at
machine precision because the synthesis is exact.

After `0t1`, the natural sequence is:
1. `cvh` (SIMD inner product) — perf, the inner-product is ~11 % of wall;
   AVX2 should ~3x it.
2. `lg8` (OpenMP over outer (l, m, n) loop) — perf, near-linear on 8 cores.
3. `31x` (QSP primitives) — needs Bastidas-Joven 2024 research; the bead
   description cites it.  Probably 300-500 LOC.
4. `u9q` (n8e Tier 2 full half-integer FFT) — substantial; ~500-1000 LOC.
   Unblocks `erv` (SO(3) via Z_2 quotient).
5. `cce` (weather-data sphere FFT viz) — now formally ready since `5fb`
   shipped.  Probably 200-400 LOC of Python + a notebook.  Excellent demo
   for the README.

---

## 5. What shipped in the previous session (and what to read about each)

| Bead    | What                                          | Where to read                           |
|---------|-----------------------------------------------|-----------------------------------------|
| `m21`   | Wigner three-term recurrence (O(N^5) → O(N^4))| `notes/wigner_recurrence.md`            |
| `dyi`   | Inline ipow vs libm pow                       | commit `1a278d3` body                   |
| `t6z`   | Julia bindings (SU2FFT.jl)                    | `julia/DESIGN.md`, `julia/src/SU2FFT.jl`|
| `3lx`   | Inverse FFT (Peter-Weyl synthesis)            | `notes/inverse_fft.md`                  |
| `ega`   | Gauss-Legendre theta nodes                    | `notes/gauss_legendre.md`               |
| `n8e`-1 | Half-integer Wigner-d evaluation              | `notes/half_integer.md`                 |
| `d7v`   | SU(2) convolution via spectrum                | `src/su2_convolve.c` header             |
| `5fb`   | Spherical FFT on S^2 (thin wrapper)           | `src/su2_sphere.c` header               |

The `notes/` design briefs are gold — they contain the math derivations,
the failure modes that were caught during research, and the implementation
strategy at the level a code-review can verify against.  Match this
discipline when adding new functions.

---

## 6. The TDD/literate-programming discipline that produced this codebase

Don't drop it.

1. **Red first.**  Add a failing test for the property the new code must
   uphold, *before* writing the new code.

2. **Read ground truth.**  Before changing a math routine, read the paper
   line being implemented.  Source comments cite `paper.tex` line numbers;
   open the paper at that line.  When the paper formula and code don't
   match, the code is wrong — don't "fix" it by tweaking until tests pass.

3. **Cross-checks > unit tests.**  Direct vs FFT is the main one.  The
   Wigner closed-form has Legendre P_l as the m=n=0 special case.  When
   you add a new path, add its cross-check immediately.

4. **One bead, one PR, one commit.**  No drive-by cleanup.

5. **Literate updates.**  When you touch the algorithm, update
   `ALGORITHM.md`.  When you touch performance, update `PROFILING.md`.  Each
   bead in `notes/` is part of the contract; future-you can't reconstruct
   the reasoning from `git log` alone.

6. **Be honest about precision.**  If you assert 1e-12 and only achieve
   0.2 (closed-grid floor), say so in the test and in the docs.  The
   previous session learned this the hard way; the docs now consistently
   distinguish "analytical synthesis at machine precision" from "spectrum
   roundtrip floor under closed-grid".

---

## 7. Style preferences observed from the user

- Concrete numbers always.  "1e-17 max-diff at N=24" not "very accurate".
- No emojis.  No marketing words.  The code reads like SQLite / TigerBeetle
  docs.
- Cite paper line numbers in source comments.  They survive moves.
- Prefer C11, `-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes -O2 -g`.
  `-Wshadow` in particular has caught real bugs.
- Don't add features that aren't paid for by a passing test.
- The user values FLINT and FFTW3 explicitly.  Don't suggest replacing them.
- For non-trivial changes the orchestration pattern from the previous session
  (research → implement → test → docs → close + commit, with the right
  subagent at each step) worked well.

---

## 8. Things explicitly out of scope unless the user asks

- Replacing FFTW3 with anything else.
- Replacing FLINT with anything else.
- Adding a custom DSL or build system (Makefile is enough).
- Rewriting in Rust / Zig / C++ (user wrote this in C deliberately).
- The af proof workspace under `proof/` is reference-only.
- GitHub Actions / external CI (user has not asked).

---

## 9. Surprises and lessons from the previous session

These cost time when you don't know them.

1. **PROFILING data goes stale fast.**  Pre-m21 perf showed pow() at 73 %.
   Post-m21 it's a small slice and the bottleneck shifted.  Re-profile
   before targeting any optimisation.  Bead `58h` is filed for a fresh
   perf record but isn't done.

2. **Honest tolerance is non-negotiable.**  An early `dyi` attempt with
   Horner tables over-built ~20x (kmax=48 entries for 2-3 actual lookups
   per seed call) and was a wash; the inline ipow was the right primitive.
   The lesson: measure the actual work the code does, don't trust a high-
   level model.

3. **The bead description can be wrong.**  `5fb`'s description said it
   needed `n8e`; in fact spherical harmonics Y_l^m are integer-l, so 5fb
   shipped as soon as 3lx did.  Read the math, not just the metadata.

4. **Closed-grid is more than a Riemann error.**  HANDOFF originally
   framed `(N/(N-1))^2` as a single closed-grid factor and suggested GL
   theta would fix it.  Reality: theta and phi/psi each contribute, and
   the phi/psi part is a structural aliasing limit that GL doesn't touch.

5. **Julia ABI workarounds are common.**  Julia ships its own libgmp,
   libfftw, libblas etc., and when your C library links system versions
   you can get silent symbol resolution against the wrong one.
   `RTLD_DEEPBIND` and `RTLD_GLOBAL` libgmp preload are the standard
   escape hatch.

6. **`bd dolt push` always before `git push`.**  Beads state needs to
   sync to the Dolt remote before the source-code push, otherwise next
   session sees an inconsistent state.

---

## 10. Where the user is reachable for ambiguities

GitHub: https://github.com/tobiasosborne/su2-fft (issues/PRs).
For paper-math disambiguation, the paper itself plus the four notes/
summaries give independent renderings of every key formula.  When the
paper and a note disagree, the paper is canonical.

Welcome aboard.  The biggest single lift available right now is `0t1` —
ship that, the application beads gain usable precision, and the rest of
the roadmap becomes more useful in practice.
