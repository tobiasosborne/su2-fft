# Roadmap — ways to make this seriously impressive

Items are ranked by an `impact × certainty` score.  Each names the file(s)
it would touch, sketches the approach, and gives a measured baseline so the
proposed gain is honest, not a marketing number.  Baseline is the post-m21
double-precision path at N=24 on the dev host: **20 ms per `su2_fft` call,
46% in seed calls to `su2_wigner_d`, 33% in the recurrence sweep, 11% in
the inner product (see `PROFILING.md`)**.

---

## Tier 1 — fix the actual hot path (each: ≥3× wall-clock)

### ~~1. Three-term recurrence over `l` for the Wigner kernel~~ — DONE (m21)

Shipped in bead `su2fft-m21`.  Stage 2 is now O(N^4).  90.91x speedup at
N=24 vs the O(N^6) direct path.  Max-diff at N=24: 2.21e-13 (well within
1e-10 tolerance; recurrence accumulates floating-point error over l, which
is expected and acceptable).  See `ALGORITHM.md §2.3` and `PROFILING.md`.

### ~~1. Integer-power tables in place of `pow()` inside the seed~~ — DONE (dyi)

Shipped in bead `su2fft-dyi` as inline `ipow(double x, int k)` (repeated
squaring, ⌈log₂(k)⌉ ≈ 5–6 mults for k ≤ 50).  A first attempt with full
Horner tables `c2_pow[0..2l]` was a wash — ~48-entry tables to access ~2–3
entries per de Moivre term.  The shipped `ipow` gives ~4% wall improvement at
N=24 (warm-run mean ~103x vs ~91x on `bench/compare`; per-FFT wall 34.3 ms →
~33.0 ms on `build/profile_stages 24 10`).  Seed % at N=24 is unchanged at
46%: post-m21 the `pow()` cost was already small relative to trig.  Code is
strictly cleaner regardless of the modest gain.

**The seed path is now trig-dominated** (`cos(beta/2)` + `sin(beta/2)` per
seed call), not `pow()`-dominated.  The next lever is trig-sharing (item 1b
below).

### 1b. Share trig across the seed pair and across (m, n) at fixed k (su2fft-xxb)

**Impact: ~10% reduction in seed cost (~3 ms at N=24).**  Both seed calls at
a (m, n, k) triple share the same `theta_k`, yet each call independently
evaluates `cos(theta_k/2)` and `sin(theta_k/2)`.  Across all (m, n) at fixed
k, the same theta_k recurs O(N^2) times.  Precomputing the trig pair once per
k and passing `(c2, s2)` into `wigner_d_phys` avoids O(N^2) duplicate
`cos`/`sin` calls per k-slice.

Touches: `src/su2_wigner.c` (add `wigner_d_phys_trig` variant accepting
precomputed c2, s2), `src/su2_fft.c` Stage 2 (hoist trig out of (m, n)
loops).

### 2. Recurrence-coefficient caching across k

**Impact: ~20% on the recurrence sweep (33% of wall).**  The Jacobi
coefficients `Ak`, `Bk`, `Ck` and the normalisation ratios `F1`, `F2` in the
§2b recurrence (`notes/wigner_recurrence.md`) depend on `(l, m, n)` but not
on `theta_k`.  Currently they are recomputed for every k in the outer k-loop.
Precomputing them once per (m, n) into a length-N array outside the k-loop
avoids N redundant divides and square-roots per l-step.

Touches: `src/su2_fft.c` Stage 2 only.

### 3. OpenMP across the outer `(m, n)` loop

**Impact: ~6–8× on an 8-core box** (each (m, n) pair in Stage 2 is
independent).  Add `#pragma omp parallel for collapse(2)` on the `m, n`
loops in `src/su2_fft.c`; preallocate per-thread `d_seq` and `acc` buffers.

Touches: `src/su2_fft.c`, `Makefile` (add `-fopenmp`).  Composes
multiplicatively with items 1 and 2.

### 4. SIMD inner product (AVX2 → AVX-512)

**Impact: ~4× on the inner-product slice (11% of wall).**  The per-(m, n, l)
accumulation over k is a length-N complex multiply-accumulate — a textbook
vectorisation target.  Use `_mm256_pd` intrinsics or `#pragma omp simd`
with an aligned `__attribute__((aligned(64)))` buffer.

Touches: `src/su2_fft.c`; optional `src/simd/dot_complex.h` for portability.

---

## Tier 2 — extend the algorithm's reach

### ~~5. Half-integer `l` Wigner-d evaluation~~ — DONE Tier 1 (n8e); full FFT = su2fft-u9q (P2)

Bead `su2fft-n8e` Tier 1 shipped `su2_wigner_d_half` in `src/su2_wigner.c`
(+85 LOC).  The de Moivre sum uses `tgamma` for factorial ratios (unbounded);
args are `2l/2n/2m` (always integers) to avoid FP comparison issues.  38/38
C tests, 714/714 Julia tests.  Cross-check vs integer-l `su2_wigner_d`: max
delta 2.22e-16 (one ULP).  Spin-1/2 closed forms verified at 1e-12.

Tier 2 (bead `su2fft-u9q`, P2) covers the full forward+inverse FFT for
half-integer `l`.  Required work: (a) 4pi-periodic phi/psi grid
(`exp(±i n phi)` for half-integer n is not periodic on 2pi), (b) Gamma in
the Jacobi recurrence coefficients, (c) new FFTW plans for the extended grid,
(d) iteration range expansion.  Estimated ~500-1000 LOC.  Unblocks
`su2fft-erv` (SO(3) FFT).  Bead `su2fft-5fb` (integer-l spherical FFT)
shipped without `u9q`; see item 13 below.

### ~~6. Real-input symmetry shortcut~~ — DONE (su2fft-4v7)

Shipped in bead `su2fft-4v7`.  New entry point
`su2_fft_real(int N, const double *f, double _Complex *fhat)` in
`src/su2_fft.c`.  Takes a real-valued sample array and produces the same full
`fhat` as `su2_fft` on the complexified input.

**Signed Hermitian symmetry.**  The correct relation is

```
fhat(l)_{m,n} = (-1)^{m-n} * conj( fhat(l)_{-m,-n} )
```

Derivation: substitute `P^l_{n,m} = i^{m-n} d^l_{n,m}` in the analysis sum
(paper.tex lines 1316/1361); take the conjugate using `f` real; apply the
Wigner-d parity `d^l_{-n,-m}(beta) = (-1)^{m-n} d^l_{n,m}(beta)`.  The
`(-1)^{m-n}` sign is essential.  The bead's original description stated the
relation without this sign (wrong); the unsigned form deviates by ~0.1 at
N=6.

**Implementation.**  Stage 1 of `su2_fft` was extracted into a shared static
helper `su2_fft_stage1`, called identically by both `su2_fft` and
`su2_fft_real` (`su2_fft` output is unchanged).  `su2_fft_real` promotes the
real input to complex, runs the shared Stage 1, computes Stage 2 only for
canonical `(m, n)` pairs where `(n > 0)` or `(n == 0 && m >= 0)`, then fills
the non-canonical partners via the signed symmetry above.  The `(0,0)`
diagonal entry is self-paired, so `fhat(l)_{0,0}` is real.

**Cross-check vs `su2_fft` on random real input:**
- N=5: max error 6.94e-18
- N=6: max error 7.76e-18
- N=8: max error 8.00e-18
- `|imag fhat(l)_{0,0}|` < 1e-13

**Speedup (Stage 2 dominates the double path; halving the (m,n) sweep gives
~2x; Stage 1 is shared; real->complex promote is O(N^3) overhead):**
- N=16: 1.91x
- N=24: 1.90x
- N=32: 2.00x

### ~~7. Inverse FFT (synthesis) `su2_fft_inv`~~ — DONE (3lx)

Shipped in bead `su2fft-3lx`.  `src/su2_fft.c` now contains both `su2_fft`
(forward) and `su2_fft_inv` (inverse, 118 LOC appended).  `tests/test_roundtrip.c`
adds 7 tests; C test count grows from 18 to 25.  Julia bindings: `SU2FFT.fft_inv`
exported; 211/211 tests pass.

Acceptance numbers:
- Analytical synthesis (single-coefficient delta fhat): 1e-12 to 1e-13 residual.
  Constant function, linearity, and (2l+1) Plancherel weight all verified exactly.
- Spectrum roundtrip `forward(inverse(fhat)) ≈ fhat`: rel_err ~5.6 at N=16,
  ~5-7 at N=8 for random fhat.  NOT machine precision.  Root cause: N-point
  Riemann sum over the closed theta grid does not exactly integrate Wigner-d
  polynomials of degree up to N-1.  This is a property of the quadrature, not
  of the synthesis code.  Roundtrip tests use tolerance 10.0 to document the
  floor, not to claim accuracy.

Downstream: `su2fft-d7v` (convolution), `su2fft-31x` (QSP), `su2fft-5fb`
(spherical) are formally unblocked.  Their practical utility waits on
`su2fft-0t1` (phi/psi grid resolution, item 8b).  Pick up `0t1` before
starting these.

### ~~8. Gauss-Legendre nodes in theta~~ — DONE (ega)

Shipped in bead `su2fft-ega`.  `src/su2_gauss_legendre.c` (~75 LOC) computes
GL nodes and weights via Newton iteration on P_N.  `src/su2_fft.c` gained
`su2_fft_gl` and `su2_fft_inv_gl` (~212 LOC appended); `norm_gl = 1/(2N^2)`.
Julia exports `fft_gl`, `fft_inv_gl`, `gl_nodes_weights`.  34/34 C tests pass
(was 28); 274/274 Julia tests pass (was 211).

Concrete results at N=8:
- DC normalisation: `forward(constant) = 1.0` to 1e-12.  Closed grid gave
  `(N/(N-1))^2 = 1.31`.
- Analytical synthesis at l=0: 1e-12 residual.
- `forward(constant)` leakage to other coefficients: max ~0.197.
- Single-coefficient roundtrip at `(l, m=0, n=0)`: max error ~0.344 over l in [0, N-1].
- Single-coefficient roundtrip at `(l=N-1, m=±(N-1), n=0)`: error ~3 (worst).

DC is fixed; phi/psi aliasing floor remains.  The structural follow-up is item
8b below.

### 8b. Phi/psi grid resolution -- P1 NEXT (su2fft-0t1)

**This is now the top-priority item.**  `ega` fixed the theta quadrature but
the phi/psi Stage 1 still uses the closed-grid fold, which over-counts endpoints
(`g[0] = f[0] + f[N-1]`).  The `(-1)^{n+m}` phase corrects the half-shift but
does not eliminate aliasing for modes with `|m|` or `|n|` near `N-1`.  The
bandlimit demands modes up to `|m|, |n| = N-1` (requiring `2N-1` independent
frequencies) but the closed N-point phi/psi grid resolves only `~N` modes.

Result: at N=8 GL, single-coefficient roundtrip error at `(l=N-1, m=±(N-1), n=0)`
is ~3 (O(1)).  This is the remaining barrier to a useful `forward(inverse(fhat)) = fhat`.

Two options: (a) use a `2N-1` point grid in phi/psi, resolving all required
modes; (b) restrict the bandlimit to `|m|, |n| < N/2`, which keeps the current
grid but reduces output coefficients.  Either option unblocks application beads
`su2fft-d7v`, `su2fft-31x`, and `su2fft-5fb` at practical precision.

Touches: `src/su2_fft.c` Stage 1 (phi/psi fold); `include/su2.h` (grid layout
change if option a); `tests/test_roundtrip.c` (tighten roundtrip tolerance once
fixed).

### 9. Open-grid mode (φ, ψ on `[0, 2π)`)

Drop the closed-grid endpoint fold by sampling `φ_j = j · 2π/N`.  Removes
the `(-1)^{n+m}` half-shift and the `g[0] += f[N-1]` book-keeping in Stage 1
(see `ALGORITHM.md §2.2`).  Stage 1 then plugs directly into an `N × N`
FFTW plan with no rewrite of indices.

Touches: `src/su2_fft.c` + grid choice; new `SU2_GRID_OPEN` flag.

---

## Tier 3 — new capabilities the FFT unlocks

### ~~10. Convolution on SU(2) via the spectrum~~ — DONE (d7v)

Shipped in bead `su2fft-d7v`.  `src/su2_convolve.c` (80 LOC): `su2_convolve(N,
fhat, ghat, fghat)` performs a per-l `(2l+1) x (2l+1)` matrix product.  Aliasing-
safe via per-block temporary.  `tests/test_convolve.c` (173 LOC, 5 tests).  Julia:
`SU2FFT.convolve(fhat, ghat, N)` (+30 LOC ccall, +66 LOC tests).  43/43 C tests,
729/729 Julia tests; all assertions 1e-12 to 1e-13.

Explicit l=1 verification: `diag(1,2,3) * diag(4,5,6) = diag(4,10,18)` to 1e-12
(from `test_convolve_l1_diagonal`).

Note on noncommutativity: matrix product is not symmetric; `convolve(f, g)` and
`convolve(g, f)` are distinct (SU(2) is nonabelian).

Usability caveat: the convolution operation itself is exact at 1e-12 to 1e-13.
End-to-end accuracy of `inverse(convolve(forward(f), forward(g)))` is limited by
the phi/psi roundtrip floor (~0.2 at N=8 GL; see `ALGORITHM.md §3.6` and §5.2).
Bead `su2fft-0t1` must land before spatial-domain convolution is useful.

See `ALGORITHM.md §5.2` for full derivation.

### 11. SO(3) FFT via the Z₂ quotient

SU(2) → SO(3) is a double cover with kernel ±I.  Functions on SO(3)
correspond to **integer-l** SU(2) coefficients.  After `su2fft-u9q` (full
half-integer FFT), a `su2_fft_so3` that simply zeroes out the half-integer-l
output is one CLI flag — and catches a much larger user base (computer
graphics, robotics, molecular dynamics, all do SO(3) FT).  Blocked on `u9q`.

Touches: thin wrapper in `src/su2_fft_so3.c`.

### 12. Quantum Signal Processing (QSP) primitives

The paper's own framing (line 237) notes QSP sequences for su(2) and su(1,1)
are intimately related to the non-linear Fourier transform.  A `qsp_decode`
that takes a target polynomial transform and emits the SU(2) rotation
sequence (e.g. via the Haah / Chao construction) — implemented on top of
this FFT for the spectral side — would turn the project into a usable QSP
tool.  Cite `bastidas2024complexification` from `bib-Delgado-Lp-2016.bib`.

Touches: new `src/qsp/` directory; non-trivial (~1 week) but standalone.

### ~~13. Spherical-harmonic FFT (S² = SU(2)/U(1))~~ — DONE (su2fft-5fb)

Shipped in bead `su2fft-5fb`.  `src/su2_sphere.c` (138 LOC): `su2_fft_sphere`,
`su2_fft_sphere_inv`, `su2_sphere_total_coeffs`.  Thin wrapper over `su2_fft` /
`su2_fft_inv`: replicates input over psi, calls the SU(2) FFT, extracts the m=0
row from each l-block.  Total coefficients N^2 (vs N(4N^2-1)/3 for full SU(2)).
`tests/test_sphere.c` (131 LOC, 6 testsets).  Julia: `fft_sphere`,
`fft_sphere_inv`, `sphere_total_coeffs` ccall wrappers (+57 LOC in
`julia/src/SU2FFT.jl`); 7 testsets (+82 LOC in `julia/test/runtests.jl`).
49/49 C tests, 744/744 Julia tests.

Concrete numbers:
- Y_0^0 analytical synthesis: 1e-13 residual.
- Y_1^0 analytical synthesis (`3*cos(theta)`): 1e-12 residual.
- Linearity: 1e-12.
- Constant forward at N=8: fhat_sph[0] = 1.284 (closed-grid floor; `(N/(N-1))^2
  = 1.306` is the closed-grid expectation; exact DC gated on `su2fft-0t1`).

Inherits the closed-grid phi/psi aliasing floor from `su2_fft`.  Exact DC and
spectrum roundtrip precision gated on `su2fft-0t1`.

Unblocks `su2fft-cce` (weather-data sphere FFT visualisation).  `su2fft-cce` is
now ready to build; practical spectrum precision at high-l coefficients remains
gated on `su2fft-0t1`.

The half-integer-l extension (spin-weighted spherical harmonics) requires
`su2fft-u9q` and remains blocked on that bead.

---

## Tier 4 — engineering polish

### ~~14. Julia bindings (`SU2FFT.jl`)~~ — DONE (t6z)

Shipped in bead `su2fft-t6z`.  `julia/` contains `SU2FFT.jl` v0.1.0 with
ccall wrappers for `su2_fft`, `su2_ft_direct`, and `su2_wigner_d`, plus
coefficient accessors (`fhat_at`, `fhat_block`, `total_coeffs`, `coeff_offset`).
Makefile gained `-fPIC` and a `build/libsu2.so` shared-library target; 18/18 C
tests still pass.  199/199 Julia tests pass via `Pkg.test`; gold-standard
`fft ≈ ft_direct atol=1e-10` holds at N=6 and N=8.  Array layout: Julia's
column-major `f[j2+1, k+1, j1+1]` maps to C row-major `f[j1*N*N + k*N + j2]`
with no permutation.  Known issue: Julia 1.12 bundled libgmp ABI conflict
worked around in `__init__` via `RTLD_DEEPBIND`; robust fix tracked as
`su2fft-e5z`.

Follow-ups not yet filed as beads: arb-precision bindings via Arblib.jl,
BinaryBuilder packaging for portable distribution (removes Linux-only
`libfftw3`/`libflint` system-library requirement), macOS `.dylib` support,
registration to the Julia General registry.

### 14. Python bindings (`pip install su2fft`)

A `cffi`-based wrapper exposing the 4 public functions, plus NumPy
zero-copy interop.  This is the single highest-leverage move for adoption —
the field of likely users (mathematical physicists, quantum-information
researchers, computer graphics) lives in Python, not C.

Touches: new `python/` directory, `setup.py`, two CI jobs.

### 15. GPU backend (CUDA / HIP)

The Stage 2 inner product is `O(N^3)` independent length-N complex dot
products — a perfect fit for a CUDA kernel with one block per coefficient.
cuFFT handles Stage 1.  Expect ~50× over the optimised CPU path at N=64.

Touches: new `src/cuda/`, `cuFFT` and `thrust` deps.  Optional via
`make GPU=1`.

### 16. Fuzz testing + property-based tests

Write a `tests/fuzz.c` that generates random inputs and checks Parseval
(`||f||² = Σ (2l+1) ||f̂(l)||_F²` — paper line 552) and the round-trip
identity.  Integrate `cifuzz` or `AFL++` for adversarial input generation.
Catches the precision regressions that would otherwise need a paper-tex
specialist to spot.

Touches: new `tests/fuzz.c`; CI hookup.

---

## Headline summary

Bead `su2fft-m21` (three-term recurrence) delivered 90.91x speedup at N=24 and
the paper's O(N^4) asymptotic.  Bead `su2fft-dyi` (inline `ipow`) added a
modest ~4% wall improvement; the seed is now trig-dominated.  Bead `su2fft-t6z`
(Julia bindings) shipped `SU2FFT.jl` v0.1.0 with 199/199 tests passing.  Bead
`su2fft-3lx` (inverse FFT) shipped `su2_fft_inv` (118 LOC); 25/25 C tests and
211/211 Julia tests.  Analytical synthesis hits 1e-12; closed-grid spectrum
roundtrip is rel_err ~5.6 at N=16.

Bead `su2fft-ega` (GL theta nodes) shipped `su2_fft_gl` / `su2_fft_inv_gl`
(~212 LOC) and `su2_gauss_legendre.c` (~75 LOC).  34/34 C tests, 274/274 Julia
tests.  DC normalisation is exact (1.0 to 1e-12 vs 1.31 closed-grid at N=8).
A phi/psi aliasing floor remains: single-coefficient roundtrip error at high
`|m|, |n|` is O(1).  This is documented; it is not a regression.

The top priority is now **`su2fft-0t1`** (phi/psi grid resolution, item 8b):
it is the remaining barrier to exact spectrum roundtrip and gates practical use
of application beads `d7v` and `31x`.  After `0t1`, items **1b, 2, 3, 4, 15**
(trig-sharing, recurrence-coefficient caching, OpenMP, SIMD, Python bindings)
complete the performance picture.

Bead `su2fft-n8e` Tier 1 shipped half-integer Wigner-d evaluation (+85 LOC C,
+82 LOC test, +25 LOC Julia binding, +47 LOC Julia tests).  38/38 C tests,
714/714 Julia tests.  Full half-integer FFT is `su2fft-u9q` (P2, ~500-1000 LOC
estimated).  `su2fft-erv` is blocked on `u9q`.  Bead `su2fft-5fb` (integer-l
spherical FFT) shipped without `u9q`; see item 13 for details.

Bead `su2fft-d7v` shipped spectral convolution (`src/su2_convolve.c`, 80 LOC;
`tests/test_convolve.c`, 173 LOC; Julia bindings +30/+66 LOC).  43/43 C tests,
729/729 Julia tests.  Explicit l=1 check: `diag(1,2,3) * diag(4,5,6) =
diag(4,10,18)` to 1e-12.  Convolution itself is exact at 1e-12 to 1e-13;
end-to-end forward→convolve→inverse accuracy is gated by `su2fft-0t1`.

Bead `su2fft-5fb` shipped spherical-harmonic FFT (`src/su2_sphere.c`, 138 LOC;
`tests/test_sphere.c`, 131 LOC; Julia bindings +57/+82 LOC).  49/49 C tests,
744/744 Julia tests.  Y_0^0 synthesis: 1e-13.  Y_1^0 synthesis: 1e-12.  Constant
forward at N=8: fhat_sph[0] = 1.284 (closed-grid floor).  Inherits `0t1` aliasing
caveat.  Unblocks `su2fft-cce` (weather-data sphere FFT visualisation).

Bead `su2fft-4v7` shipped `su2_fft_real` (real-input forward FFT with signed
Hermitian symmetry `fhat(l)_{m,n} = (-1)^{m-n} conj(fhat(l)_{-m,-n})`).
Stage 2 computed over canonical (m,n) pairs only; non-canonical partners filled
by symmetry.  Cross-check vs `su2_fft` max error 8.00e-18 at N=8; `|imag
fhat(l)_{0,0}|` < 1e-13.  Measured speedup 1.91x/1.90x/2.00x at N=16/24/32.
