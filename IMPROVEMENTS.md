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

### 1. Integer-power tables in place of `pow()` inside the seed (de Moivre sum)

**Impact: ~3× on the seed budget (46% of N=24 wall).**  Inside `wigner_d_phys`
the exponents `pc`, `ps` are small non-negative integers ≤ 2l_min ≤ 2N.
Precompute `c2^k`, `s2^k` for `k = 0..2l_min` once per call (length 2l_min+1,
Horner-built in 2l_min mults), then the t-loop is two table lookups per term.
This eliminates all `pow()` calls from the seed path.

Touches: `src/su2_wigner.c`.

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

### 7. Inverse FFT (synthesis) `su2_fft_inv`

Adjoint of the forward FFT: takes coefficients `fhat`, reconstructs `f` on
the grid.  Trivial to derive from Stage 1 + Stage 2 reversed.  Together with
the existing forward FFT this gives a round-trip test of the form
`|| su2_fft_inv(su2_fft(f)) - f ||` — the strongest correctness assertion
possible without a closed-form reference.

Touches: new `src/su2_fft_inv.c`, `tests/test_roundtrip.c`.

### 8. Gauss–Legendre nodes in θ

The current closed grid (`θ_k = k π/(N-1)`) carries O(1/N²) Riemann error
on the θ-integral, visible as the high-l aliasing tail in `build/spectrum.png`.
Switching to Gauss–Legendre nodes on `[-1, 1]` (via FLINT's
`arb_hypgeom_legendre_p_ui_root`) gives spectral accuracy — the discrete FT
becomes exact for bandlimited inputs, no more aliasing.

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

Touches: new `src/su2_convolve.c`, depends on #7.

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

Item 1 of the original list (three-term recurrence) shipped as bead `su2fft-m21`,
delivering 90.91x speedup at N=24 and bringing the implementation to the paper's
O(N^4) asymptotic.  If we now ship items **1, 2, 3, 4, 7, 14** (renumbered above)
the codebase goes from "honest O(N^4) implementation that matches the paper" to
"the obvious tool you reach for if you need an SU(2) FFT".  Items 1–4 alone should
bring N=24 from ~20 ms to an estimated < 5 ms per FFT; item 14 (Python bindings)
is what turns the project from a curiosity into something a researcher actually
depends on.
