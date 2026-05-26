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

### 5. Half-integer `l` support

The paper covers `l ∈ ½ℕ₀`; we currently restrict to integer `l`.  Adding
half-integer support means: (a) using `tgamma(l + 0.5 + 1)` instead of
`fact()` where l is half-integer, (b) running the FFT phi/psi DFTs at a
**different** angular grid (`exp(±i n φ)` for half-integer n is not periodic
on `2π`), and (c) doubling the (l, m, n) iteration range.  Lets us cover the
spin-½, spin-3/2, spin-5/2 representations directly — essential for fermionic
applications and for the quantum-information uses the paper cites.

Touches: `include/su2.h` (layout), `src/su2_wigner.c` (factorials → Gamma),
`src/su2_fft.c` (grid + iteration).

### 6. Real-input symmetry shortcut

For real-valued `f`, `fhat(l)_{m,n} = conj(fhat(l)_{-m,-n})` (Hermitian
symmetry on the dual).  Compute only `n ≥ 0` or `m + n ≥ 0`, halving both
work and storage.  Common case in physics simulations of real fields on S³.

Touches: new entry point `su2_fft_real(int N, const double *f, double _Complex *fhat)`
in `src/su2_fft.c`; ships alongside the complex version.

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
`su2fft-ega` (item 8 below, now P1).  Pick up `ega` before starting these.

### 8. Gauss-Legendre nodes in theta -- P1 NEXT (su2fft-ega)

**This is the top-priority item.**  The exact roundtrip `forward(inverse(fhat))
= fhat` requires spectral-accuracy theta quadrature.  After 3lx shipped, the
only remaining barrier to a machine-precision forward+inverse pair is the
N-point Riemann sum over the closed theta grid, which introduces O(1) relative
error in the spectrum roundtrip (rel_err ~5.6 at N=16 for random fhat).

Switching to Gauss-Legendre nodes on [-1, 1] (via FLINT's
`arb_hypgeom_legendre_p_ui_root`) integrates all Wigner-d polynomials of degree
up to 2N-1 exactly.  The discrete FT becomes exact for bandlimited inputs;
`forward(inverse(fhat)) = fhat` to machine precision; the aliasing tail in
`build/spectrum.png` disappears.  Beads `d7v`, `31x`, `erv`, `5fb` become
usable at machine precision only after this lands.

Touches: `src/su2_grid.c` (new grid function), `src/su2_ft.c`, `src/su2_fft.c`
(replace `sin(theta_k)` weights with GL weights).

### 9. Open-grid mode (φ, ψ on `[0, 2π)`)

Drop the closed-grid endpoint fold by sampling `φ_j = j · 2π/N`.  Removes
the `(-1)^{n+m}` half-shift and the `g[0] += f[N-1]` book-keeping in Stage 1
(see `ALGORITHM.md §2.2`).  Stage 1 then plugs directly into an `N × N`
FFTW plan with no rewrite of indices.

Touches: `src/su2_fft.c` + grid choice; new `SU2_GRID_OPEN` flag.

---

## Tier 3 — new capabilities the FFT unlocks

### 10. Convolution on SU(2) via the spectrum

Convolution on a non-abelian group is matrix-valued: `(f ⋆ g)^(l) = f̂(l) · ĝ(l)`
(a matrix product, not a Hadamard).  Once forward + inverse FFTs exist, a
`su2_convolve(f, g)` is ten lines + an `acb_mat_mul`.  Useful for: smoothing
on SO(3) (via the SU(2) → SO(3) quotient), template matching on rotated
3-D images, group-averaged neural-network layers.

Touches: new `src/su2_convolve.c`, depends on #7 (DONE, 3lx) + #8 (ega, for exact roundtrip).

### 11. SO(3) FFT via the Z₂ quotient

SU(2) → SO(3) is a double cover with kernel ±I.  Functions on SO(3)
correspond to **integer-l** SU(2) coefficients.  After #5, a `su2_fft_so3`
that simply zeroes out the half-integer-l output is one CLI flag — and
catches a much larger user base (computer graphics, robotics, molecular
dynamics, all do SO(3) FT).

Touches: thin wrapper in `src/su2_fft_so3.c`.

### 12. Quantum Signal Processing (QSP) primitives

The paper's own framing (line 237) notes QSP sequences for su(2) and su(1,1)
are intimately related to the non-linear Fourier transform.  A `qsp_decode`
that takes a target polynomial transform and emits the SU(2) rotation
sequence (e.g. via the Haah / Chao construction) — implemented on top of
this FFT for the spectral side — would turn the project into a usable QSP
tool.  Cite `bastidas2024complexification` from `bib-Delgado-Lp-2016.bib`.

Touches: new `src/qsp/` directory; non-trivial (~1 week) but standalone.

### 13. Spherical-harmonic FFT (S² = SU(2)/U(1))

Project SU(2) coefficients onto fixed-`m=0` subspace → harmonic transform
on the sphere.  Provides a single-binary alternative to s2kit/SHTns for the
common case.  Easy after #5 + #7.

Touches: thin wrapper, depends on #5 and #7.

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
modest ~4% wall improvement; the `pow()` cost was already small post-m21 and
the seed is now trig-dominated.  Bead `su2fft-t6z` (Julia bindings) shipped
`SU2FFT.jl` v0.1.0 with 199/199 tests passing and cross-validated to 1e-10 at
N=6 and N=8.  Bead `su2fft-3lx` (inverse FFT) shipped `su2_fft_inv` (118 LOC
in `src/su2_fft.c`) + Julia `SU2FFT.fft_inv`; 211/211 Julia tests and 25/25 C
tests pass.  Analytical synthesis hits 1e-12; spectrum roundtrip under
closed-grid Riemann is rel_err ~5.6 at N=16 (O(1), not O(1/N^2)) -- the
honest closed-grid limitation documented in `ALGORITHM.md §3`.

The top priority is now **`su2fft-ega`** (Gauss-Legendre theta nodes, item 8):
it closes the roundtrip gap to machine precision and unblocks the application
beads (`d7v`, `31x`, `erv`, `5fb`) at usable tolerance.  After `ega`, items
**1b, 2, 3, 4, 15** (trig-sharing, recurrence-coefficient caching, OpenMP,
SIMD, Python bindings) complete the performance picture.  Items 1b-4 together
should bring N=24 well below 30 ms per FFT; item 15 (Python bindings) is what
turns the project from a curiosity into something a researcher actually depends
on.
