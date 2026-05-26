# CLAUDE.md — su2-fft

## What this is

A C implementation of the O(N^4) Fast Fourier Transform on SU(2), following
Delgado et al., "On The Fast Fourier Transform on SU(2)", arXiv 2605.23923
(May 2026). Four parallel paths (direct/fast x double/arb) cross-validated
to floating-point noise (~1e-17). FLINT arb path certifies 154 decimal digits
at prec=512.

## Canonical documents — read these BEFORE writing code

```
HANDOFF.md        <-- start here. Five-minute orientation for a fresh agent.
README.md         <-- what the project is and why anyone cares
ALGORITHM.md      <-- canonical math + code narrative. Every formula cited.
PROFILING.md      <-- where the cycles go. Read before optimising.
IMPROVEMENTS.md   <-- the prioritised roadmap.
paper.tex         <-- verbatim arxiv source. Paper is canonical when notes disagree.
notes/            <-- four independent renderings of every key formula.
```

If a claim in code is not anchored in `ALGORITHM.md` or `paper.tex` with a
line number, it is undocumented and you should fix that before changing it.

## Tobias's rules — follow to the letter

These are NON-NEGOTIABLE. Every agent, every session, every commit.

1. **GROUND TRUTH = PAPER.** Not your memory of the paper. Not what "should"
   work. The paper.tex line being implemented. Every nontrivial math routine
   in `src/` cites its paper line in a source comment; keep that discipline.
   When the paper and a note disagree, the paper wins.

2. **CITE EVERYTHING.** Every physics/math formula in source must cite:
   (a) `paper.tex` line number, (b) a verbatim copy of the equation it
   implements. This is how the implementation survived five rewrites of
   the Wigner routine. Example:
   ```c
   /* paper.tex:1316 — fhat(l)_{m,n} = (dphi*dtheta*dpsi/8pi^2)
    *   * sum_{j1,k,j2} f(g) * conj(t^l_{n,m}(g)) * sin(theta_k)
    */
   ```

3. **SKEPTICISM.** Be skeptical of: your own assumptions, subagent output,
   previous agent work, the existing tests, and the paper itself (LaTeX
   typos happen). Verify. Reproduce. When a test passes, ask whether it
   was actually exercising the property you think it was.

4. **ALL BUGS ARE DEEP.** No bandaids. The Rodrigues form passed tests up
   to l=8 and silently produced NaNs by l=15. A fix that makes one test
   pass but doesn't address the root cause is not a fix; it is a future
   incident with a longer fuse. Investigate root causes.

5. **FAIL FAST, FAIL LOUD.** `assert()` invariants, don't silently return.
   If a factorial overflows, if a precision is insufficient, if a grid
   parameter is out of range — abort with a clear message at the call site.
   Corrupted output is worse than a crash.

6. **RED-GREEN TDD.** Write the failing test first; watch it fail; write
   the minimum code to pass; then refactor. The cross-check pattern is:
   "two independent implementations agree on random input to ~1e-10". See
   `tests/test_fft.c::test_fft_matches_direct_random` and
   `tests/test_arb.c::test_arb_direct_vs_fast` for the gold standard.
   Every new path needs its cross-check added immediately.

7. **CROSS-CHECKS > UNIT TESTS.** Direct vs FFT is the main one. The
   Wigner closed-form has Legendre P_l as the m=n=0 special case. The arb
   path has the double path as its prec=53 anchor. Unit tests catch typos;
   cross-checks catch algorithmic errors. Prefer the latter.

8. **GET FEEDBACK FAST.** `make test` runs 34 tests across 7 binaries in
   ~5s. Run it after every non-trivial change — don't code blind for 500
   lines then check. For a single binary: `make test/test_fft && build/test_fft`.

9. **WORKFLOW (TIERED).** Scale effort to change size:
   - **Trivial** (<5 LOC, typo, comment fix): direct fix, no subagents.
   - **Small** (<20 LOC, single function): write the test, write the code.
   - **Core** (new algorithm, new path, >20 LOC, math routine): research
     paper + notes first, write the cross-check before the code, propose
     the change in a bead before implementing.

10. **ONE BEAD, ONE PR, ONE COMMIT.** Bead `su2fft-m21` is its own commit
    line. No drive-by cleanup. Drive-by improvements go in their own beads
    first. The diff should be readable in one sitting.

11. **LITERATE UPDATES.** When you touch the algorithm, update `ALGORITHM.md`.
    When you touch performance, update `PROFILING.md`. When you finish work
    that opens new opportunities, file beads. The documents are the contract;
    if they are stale, future-you cannot recover state from `git log`.

12. **NO EMOJIS, NO MARKETING.** The code reads like SQLite / TigerBeetle
    docs. Concrete numbers always: "1e-17 max-diff at N=24", not "very
    accurate". No "blazing fast", "robust", "production-grade". State the
    measurement.

13. **DON'T REPLACE FFTW3 OR FLINT.** The user picked them deliberately.
    Don't suggest alternatives unless you have shipped a measurable
    improvement first. Do not "modernise" the arb path. The
    `conj(forward(conj(.)))` pattern in `src/su2_fft_arb.c` is correct;
    `acb_dft_prod` is forward-only.

14. **proof/ IS REFERENCE-ONLY.** The af proof workspace under `proof/` is
    not part of the build. The user explicitly said "no need to run
    verifiers, the af tree is for reference purposes". Don't spend time
    validating it; don't modify it.

## Project-specific invariants you will trip over

These are non-obvious. **Read `HANDOFF.md` §2 for the full set with examples.**
Quick reference:

1. **Closed grid.** `phi[j] = -pi + j*2pi/(N-1)`. Endpoints `j=0` and
   `j=N-1` are the same point. Fold + `(-1)^{n+m}` phase recovers the
   DFT from an `(N-1)x(N-1)` FFTW plan. Change fold and phase together.

2. **P^l_{n,m} vs d^l_{n,m} differ by `i^{m-n}`.** The implementation
   carries Sakurai's real `d` and applies `pow_i(m-n)` at the end.

3. **Factorial table is double, capped at 50** (100 in `wigner_arb`).
   `src/su2_wigner.c::fact()` is precomputed.

4. **`acb_dft_prod` is forward-only.** Backward = `conj(forward(conj(.)))`.
   Do not "simplify".

5. **Integer-l only.** Half-integer support is bead `su2fft-n8e`; it
   requires factorials → Gammas and a non-periodic grid.

6. **`(N/(N-1))^2`** is the systematic Riemann error from the closed theta
   grid, not a bug. `test_ft_direct_constant` uses the modified target.  The GL
   path (`su2_fft_gl`) eliminates this factor for theta -- `forward(constant) = 1.0`
   to 1e-12 -- but the phi/psi closed-grid aliasing remains (bead `su2fft-0t1`).

7. **`pow()` is shockingly expensive.** 60–78% of cycles. Never call
   `pow(x, k)` for integer `k` in a hot loop — use repeated squaring or
   precomputed tables.

## File structure

```
su2-fft/
  README.md                         # public-facing intro
  CLAUDE.md                         # this file — non-negotiable project rules
  AGENTS.md                         # non-interactive shell rules, bd quick ref
  HANDOFF.md                        # five-minute orientation (read first)
  ALGORITHM.md                      # canonical math narrative
  PROFILING.md                      # perf report; hot path = pow() at 78%
  IMPROVEMENTS.md                   # ranked roadmap
  paper.tex                         # verbatim arxiv source (canonical)
  notes/                            # four section-by-section paper summaries
  proof/                            # af proof workspace (REFERENCE ONLY)
  Makefile                          # `make test`, `make bench`

  include/
    su2.h                           # public API + storage-layout contract

  src/
    su2_grid.c                      # Euler-angle grid + coefficient layout
    su2_wigner.c                    # stable Wigner small-d via de Moivre sum
    su2_ft.c                        # direct O(N^6) reference FT
    su2_fft.c                       # O(N^4) fast FFT (double): su2_fft (forward) + su2_fft_inv (inverse, bead 3lx) + su2_fft_gl / su2_fft_inv_gl (GL theta nodes, bead ega)
    su2_gauss_legendre.c            # GL node/weight computation via Newton iteration on P_N (bead ega)
    su2_wigner_arb.c                # arb (acb_t) Wigner
    su2_ft_arb.c                    # arb direct FT
    su2_fft_arb.c                   # arb fast FFT (acb_dft_prod + conj trick)

  tests/                            # 34 tests, 7 binaries (post ega)
    test_grid.c                     # Euler-angle grid invariants
    test_wigner.c                   # small-d unitarity, P_l limit
    test_ft.c                       # direct FT against analytic ground truths
    test_fft.c                      # FFT cross-check vs direct (gold standard)
    test_arb.c                      # arb direct vs arb fast; arb vs double
    test_roundtrip.c                # su2_fft_inv + GL variants: analytical synthesis 1e-12; roundtrip floors documented
    test_gauss_legendre.c           # GL nodes: basic properties, degree exactness to 2N-1, N=4 analytical

  bench/
    compare.c                       # direct vs fast timing + max-diff sweep
    vis_dump.c                      # spectrum dump in Python-readable format
    visualize.py                    # matplotlib spectrum plot
    arb_bench.c                     # arb-precision timing vs ball radius
    profile_stages.c                # per-stage instrumentation

  julia/                            # SU2FFT.jl — Julia bindings (bead t6z)
    Project.toml                    # package metadata; deps: Libdl
    DESIGN.md                       # design brief: layout, linkage, API
    src/SU2FFT.jl                   # ccall wrappers + accessors
    deps/build.jl, deps.jl          # local libsu2.so build (run via Pkg.build)
    test/runtests.jl                # gold-standard cross-check via Pkg.test

  build/                            # binaries land here (gitignored)
  .beads/                           # bd issue tracker storage
```

## Build & test

```bash
# Full test suite — 34 tests, 7 binaries, ~5s
make test

# Benchmark sweep — direct vs fast, ~3s
make bench && build/compare

# Per-stage profiling
build/profile_stages 24

# Single test binary
make build/test_fft && build/test_fft

# Spectrum visualisation
build/vis_dump 16 > /tmp/spec.dat && python3 bench/visualize.py /tmp/spec.dat out.png
```

Dependencies: `libfftw3-dev`, `libflint-dev` (FLINT >= 3.0), C11 compiler.

Required compiler flags: `-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes
-O2 -g -std=c11`. `-Wshadow` in particular has caught real bugs (the `mode_t`
collision with `sys/types.h`); do not remove it.

For `perf` profiling on Intel hybrid CPUs:
```bash
sudo sysctl kernel.perf_event_paranoid=1
perf record -F 4000 -e cpu_core/cycles/ -g build/compare
# `cpu_core` is mandatory on hybrid CPUs — without it you get a 12-sample
# trace from the efficiency cluster.
```

## "Done" checklist for a math-routine change

Before claiming a Wigner / FT / FFT change is done:

- [ ] `make test` passes — all 34 tests green.
- [ ] `tests/test_fft.c::test_fft_matches_direct_random` still passes at
      the same 1e-10 tolerance (or tighter).
- [ ] `tests/test_arb.c::test_arb_direct_vs_fast` still passes.
- [ ] `make bench` shows the expected speedup (or no regression).
- [ ] Source comment cites the paper line number for the formula.
- [ ] `ALGORITHM.md` updated if the algorithm changed; `PROFILING.md`
      updated if the hot path changed.
- [ ] Bead closed with reference to the commit hash.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
