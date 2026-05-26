# SU(2) Fast Fourier Transform

A C implementation of the O(N^4) Fast Fourier Transform on SU(2), following
the algorithm of Delgado et al., "On The Fast Fourier Transform on SU(2)",
arXiv 2605.23923 (May 2026).

---

## Why this exists

### The classical miracle

In 1965 James Cooley and John Tukey published an algorithm that changed
computation permanently. The observation is deceptively simple: a Discrete
Fourier Transform on N points shares sub-expressions across frequencies in a
way that lets you recurse, halving the problem at each level. The consequence
is not a mild improvement. To compute the DFT of N = 2^30 samples directly
costs O(N^2) operations; at one nanosecond per operation that is roughly
13,343 days. The Cooley-Tukey FFT reduces the count to O(N log N), bringing
the same computation down to 64 seconds -- a speedup of nearly seven orders
of magnitude [Delgado et al., line 262].

That is what it means to find the right structure in a computation. The FFT
does not approximate anything; it computes the exact same sum, just in a
different order. Convolution becomes pointwise multiplication in the frequency
domain. Differential equations on periodic domains become algebraic systems.
Signal filtering, audio compression, MRI reconstruction, gravitational wave
detection -- all of it rests on one recursive decomposition discovered in the
Cold War to identify covert nuclear tests.

### When the domain curves

The classical FFT lives on flat space: functions on R^n, or on the torus T^n.
When the domain is a curved manifold -- a sphere, a rotation group, the space
of quantum states -- the notion of "frequency" does not disappear, but it
changes character. The Peter-Weyl theorem tells us that L^2 of a compact Lie
group decomposes into irreducible unitary representations indexed by a
discrete label l. The "basis functions" are no longer scalars e^{ikx} but
(2l+1) x (2l+1) matrix-valued objects: the matrix coefficients t^l_{mn}.
Harmonic analysis on these spaces is both richer and harder than the flat
case.

SU(2) -- the group of 2x2 unitary matrices with determinant 1 -- is the
simplest non-abelian compact Lie group. It is the double cover of SO(3), the
rotation group of physical 3-space. It is diffeomorphic to the 3-sphere. It
is the spin group underlying half-integer angular momenta in quantum
mechanics. And it is the natural domain for a class of algorithms in quantum
computing that has attracted intense recent interest.

### The quantum signal processing connection

Quantum Signal Processing (QSP) encodes a polynomial transformation of a
unitary operator as a product of SU(2) rotations parameterised by a sequence
of phase angles. Computing or inverting a QSP sequence requires understanding
the spectrum of that product -- which is precisely a question about the
Fourier analysis of a function on SU(2). As Delgado et al. note (paper line
237, citing Bastidas and Joven 2024), "QSP sequences for su(2) and su(1,1)
are intimately related to the nonlinear Fourier transform." A fast SU(2) FT
is therefore a building block for fast QSP simulation and inversion, two
operations central to the compilation of quantum algorithms.

The O(N^6) direct transform is already computed in the literature. What has
been missing -- until this paper -- is the FFT analogue: an algorithm that
exploits the structure of the matrix coefficients to reduce the operation
count, the same way Cooley-Tukey exploits the structure of e^{ikx}.

### What this repository does

This repo contains a working C implementation of the Delgado et al. O(N^4)
algorithm, built test-first and cross-validated against the O(N^6) direct
transform at bandlimits up to N=24. The two implementations agree to within
2.21e-13 at N=24 (well within the 1e-10 cross-check tolerance), and the fast
path runs 90–108x faster at N=24 depending on warmup and OS jitter
(cold-cache baseline 90.91x; warm-run mean near 103x after bead `su2fft-dyi`
replaced libm `pow` with inline integer-power) -- the implementation now
matches the paper's O(N^4) asymptotic via an ascending-l three-term Wigner
recurrence (bead `su2fft-m21`). An FLINT arb-precision parallel path provides
certified interval
arithmetic: at prec=512 the worst-case ball radius on all output coefficients
is 9.61e-154, corresponding to 154 verified decimal digits.

---

## Demo: weather on the sphere (bead `su2fft-cce`)

January 2024 surface air temperature from the NOAA NCEP/NCAR Reanalysis
(2.5° lat-lon, 30 MB NetCDF) interpolated onto the resolved sphere FFT grid
(P=2N-1 open phi, N=64 Gauss-Legendre theta nodes), rendered on the unit
sphere, and Fourier-transformed to expose its spherical-harmonic spectrum.

![Surface temperature, 2024-01](examples/figures/weather_pyvista.png)

The fit slope on the per-l power spectrum is `power(l) ~ l^-2.76` over
`l in [3, 29]` -- within the typical geophysical 1/l^p decay band -- and the
operator-identity residual `|fhat - forward(inverse(fhat))|` lands at
1.71e-13, i.e. machine precision on the resolved-grid sphere wrapper
(`su2_fft_sphere_resolved`, bead `9qk`).

![Spherical-harmonic spectrum](examples/figures/weather_spectrum.png)

Interactive 3D renders (drag/rotate in a browser):
- [examples/figures/weather_pyvista.html](examples/figures/weather_pyvista.html)
- [examples/figures/weather_plotly.html](examples/figures/weather_plotly.html)

Reproduce:
```sh
make lib                                              # build libsu2.so
python3 -m venv .venv
.venv/bin/pip install -r examples/requirements.txt
.venv/bin/python examples/weather_demo.py             # ~7 s on a laptop
```

Pipeline files:
```
python/su2fft.py                              ctypes bindings to libsu2.so
examples/weather/fetch_interpolate.py         NOAA download + lat-lon -> sphere grid
examples/weather/render_3d.py                 matplotlib + pyvista + plotly
examples/weather/spectrum.py                  per-l power + |fhat(l, n)| heatmap
examples/weather_demo.py                      end-to-end orchestrator
```

---

## Quick start

```sh
# Build everything and run the test suite
make test

# Cross-comparison benchmark (FT vs FFT, timing and max-diff at each N)
make bench

# Visualise the spectrum (paper Figure 1 style)
build/vis_dump 16 > /tmp/spec.dat && python3 bench/visualize.py /tmp/spec.dat out.png
```

`make test` runs five test binaries covering the Euler-angle grid, Wigner
small-d, the direct FT, the fast FFT, and the arb-precision path. All must
pass green before the bench targets are meaningful.

Dependencies: `libfftw3-dev`, `libflint-dev` (FLINT >= 3.0), a C99 compiler.

---

## What is inside

```
include/su2.h              Public API and storage-layout contract
src/su2_grid.c             Euler-angle grid, coefficient layout helpers
src/su2_wigner.c           Stable Wigner small-d via the de Moivre sum
src/su2_ft.c               Direct O(N^6) reference Fourier transform
src/su2_fft.c              O(N^4) fast Fourier transform (double precision): su2_fft (forward), su2_fft_inv (inverse), su2_fft_gl and su2_fft_inv_gl (Gauss-Legendre theta nodes, bead ega)
src/su2_gauss_legendre.c   GL node/weight computation via Newton iteration on P_N (bead ega)
src/su2_convolve.c         Spectral convolution: su2_convolve(N, fhat, ghat, fghat) -- per-l matrix product, O(N^4) (bead d7v)
src/su2_sphere.c           Spherical-harmonic FFT on S^2: su2_fft_sphere, su2_fft_sphere_inv, su2_sphere_total_coeffs -- thin wrapper over su2_fft/su2_fft_inv, m=0 projection (bead 5fb)
src/su2_{wigner,ft,fft}_arb.c   FLINT arb/acb arbitrary-precision parallel path
tests/test_*.c             TDD red/green checks for every piece (includes test_roundtrip.c: 13 tests for su2_fft_inv + GL variants; test_gauss_legendre.c: 3 testsets for GL nodes)
tests/test_sphere.c        Spherical-harmonic FFT: 6 testsets (Y_0^0 synthesis 1e-13, Y_1^0 synthesis 1e-12, linearity 1e-12, roundtrip, total-coeffs, constant forward floor) (bead 5fb)
bench/compare.c            Cross-comparison and timing at each N
bench/vis_dump.c           Spectrum dump in Python-readable format
bench/visualize.py         Matplotlib spectrum plot
bench/arb_bench.c          Arb-precision timing vs ball radius
notes/                     Four section-by-section summaries of the paper
proof/                     11-node structural proof of the O(N^4) claim (af)
paper.tex                  Verbatim arxiv source
ALGORITHM.md               Full narrative companion: every design decision, derivation
PROFILING.md               perf + stage-instrumentation report (seed 46%, recurrence 33%, inner 11% at N=24)
IMPROVEMENTS.md            Ranked roadmap (recurrence, SIMD, OpenMP, QSP, GPU, ...)
```

The canonical narrative is `ALGORITHM.md`. Every claim in the code is anchored
there, and every departure from the paper is called out explicitly.

---

## Julia bindings (SU2FFT.jl)

A thin Julia wrapper lives at `julia/`. The package exposes `su2_fft`,
`su2_ft_direct`, and `su2_wigner_d` as `ccall`s through a locally-built
`build/libsu2.so`.

### Quick start

```julia
using Pkg
Pkg.develop(path="/path/to/su2-fft/julia")
Pkg.build("SU2FFT")     # builds libsu2.so via make
Pkg.test("SU2FFT")      # 744 tests; gold-standard fft ≈ ft_direct at 1e-10

using SU2FFT
N = 8
f = randn(ComplexF64, N, N, N)   # f[j2+1, k+1, j1+1] = sample at (j1, k, j2)
fhat_fast   = SU2FFT.fft(f)        # O(N^4) forward
fhat_direct = SU2FFT.ft_direct(f)  # O(N^6), reference
@assert maximum(abs, fhat_fast .- fhat_direct) < 1e-10

# Inverse FFT (Peter-Weyl synthesis): reconstruct f from fhat
f_reconstructed = SU2FFT.fft_inv(fhat_fast, N)   # O(N^4)

# Layout: f_reconstructed[j2+1, k+1, j1+1] = sample at Euler grid (j1, k, j2)
# Same indexing convention as the input array above.

# Coefficient access
fhat_at_l1_m0_n0 = SU2FFT.fhat_at(fhat_fast, 1, 0, 0)
block_l2         = SU2FFT.fhat_block(fhat_fast, 2)   # (5, 5) view
```

**Gauss-Legendre variant (bead `ega`).** `SU2FFT.fft_gl` and `SU2FFT.fft_inv_gl`
use Gauss-Legendre theta nodes instead of the closed grid.  This fixes the DC
normalisation: `fft_gl(constant_function)` returns `fhat(0,0,0) = 1.0` exactly
(to 1e-12), where the closed-grid path returned `(N/(N-1))^2 = 1.31` at N=8.
Single-coefficient synthesis at l=0 also hits 1e-12.  However, a separate
phi/psi aliasing error remains in the `_gl` path: at N=8 the spectrum roundtrip
`fft_gl(fft_inv_gl(fhat))` shows leakage of ~0.197 to non-DC coefficients for a
constant input, and single-coefficient roundtrip error of ~0.344 (max over l in
[0, N-1]).  The root cause is the closed-grid phi/psi fold over-counting
endpoints; the `(-1)^{n+m}` phase does not undo this for modes near the
bandlimit.  Bead `su2fft-0t1` eliminated this floor in the `su2_fft_resolved`
path (spectrum roundtrip 4.45e-15 at N=8); the `_gl` path retains the documented
floor as a historical record.

**Closed-grid roundtrip caveat.**  `SU2FFT.fft(SU2FFT.fft_inv(fhat, N))` is NOT
machine precision under the closed-grid Riemann theta quadrature.  At N=16 with
random `fhat`, the relative error is approximately 5.6.  Single-coefficient
analytical synthesis IS machine precision (1e-12 to 1e-13): if `fhat` has exactly
one non-zero entry `(l, m, n)` then `fft_inv(fhat, N)` recovers the closed-form
sample `t^l_{n,m}(g)` to 1e-12.  Applications that require tight spectrum
roundtrip tolerance should use the resolved-grid path (`su2_fft_resolved` /
`su2_fft_resolved_inv`, bead `su2fft-0t1`): max relative error 4.45e-15 at N=8.

### Layout convention

C storage `f[j1*N*N + k*N + j2]` (row-major) matches Julia's column-major
`Array{ComplexF64,3}` of shape `(N, N, N)` indexed `f[j2+1, k+1, j1+1]`.
No permutation needed for ccall. Coefficients are returned as a flat
`Vector{ComplexF64}` of length `total_coeffs(N) = sum_{l=0..N-1}(2l+1)^2`;
accessors handle the (l, m, n) offset arithmetic.

### Spherical-harmonic FFT on S^2 (bead `su2fft-5fb`)

`SU2FFT.fft_sphere`, `SU2FFT.fft_sphere_inv`, and `SU2FFT.sphere_total_coeffs`
wrap `su2_fft_sphere` / `su2_fft_sphere_inv` / `su2_sphere_total_coeffs` from
`src/su2_sphere.c`.  Input is a 2-D array indexed `f_sph[k+1, j1+1]` (theta
row k, phi column j1); output is a flat vector of length N^2.  Storage:
`fhat_sph[l^2 + (n+l)]` indexed by (l, n).

```julia
using SU2FFT
N = 8
f = rand(ComplexF64, N, N)          # f[k+1, j1+1] = f(theta_k, phi_{j1})
fhat_sph = SU2FFT.fft_sphere(f)    # length N^2 = 64
f_back   = SU2FFT.fft_sphere_inv(fhat_sph, N)
```

Thin wrapper over `fft` / `fft_inv`: replicates input over psi, calls the
full SU(2) FFT, and extracts the m=0 row from each l-block.  Inherits the
closed-grid floor from the SU(2) path; a resolved-grid sphere variant is
tracked as a follow-on to bead `su2fft-0t1`.
Y_0^0 and Y_1^0 analytical synthesis pass at 1e-13 and 1e-12 respectively.

### Spectral convolution (bead `su2fft-d7v`)

`su2_convolve(int N, const double _Complex *fhat, const double _Complex *ghat,
double _Complex *fghat)` computes the convolution of two functions whose spectra
are already known.  For each `l in [0, N-1]` it multiplies the `(2l+1) x (2l+1)`
blocks as matrices: `fghat_block(l) = fhat_block(l) * ghat_block(l)`.  This is
the Peter-Weyl convolution theorem for nonabelian groups; on SU(2) the product is
not commutative.  Cost: O(N^4).  Aliasing-safe for in-place use (`fghat == fhat`
or `fghat == ghat`).  The operation itself achieves 1e-12 to 1e-13 residual on the
spectral blocks.  End-to-end accuracy of `inverse(convolve(forward(f), forward(g)))`
is limited by the roundtrip floor of the chosen forward/inverse path.  Using
`su2_fft_resolved` / `su2_fft_resolved_inv` (bead `0t1`) gives roundtrip
4.45e-15 at N=8; the closed-grid paths retain their documented ~0.197 floor.

Julia: `SU2FFT.convolve(fhat, ghat, N)` wraps the C function via ccall.

### Half-integer Wigner-d (bead `su2fft-n8e` Tier 1)

`su2_wigner_d_half(int two_l, int two_n, int two_m, double theta)` evaluates
the paper's `P^l_{n,m}(cos theta)` for half-integer `l`, `n`, `m`.  Arguments
are `2l`, `2n`, `2m` (always integers) to avoid floating-point comparison
issues.  The implementation uses the same de Moivre sum as `su2_wigner_d` with
`tgamma` replacing the `fact()` table.  Spin-1/2 closed forms verified at
1e-12; cross-check against integer-l `su2_wigner_d` gives max delta 2.22e-16
(one ULP) at `l in [0, 6]`.  A Julia binding `SU2FFT.wigner_d_half` is included.

This is Tier 1 (evaluation only).  Full half-integer FFT (phi/psi grid on 4pi,
Gamma in the Jacobi recurrence, new FFTW plans) is bead `su2fft-u9q` (Tier 2).
Bead `su2fft-erv` is blocked on `u9q`.  Bead `su2fft-5fb` (integer-l
spherical FFT) shipped without `u9q` — see the spherical-harmonic FFT section
above.

### Status

First cut: double-precision only, Linux-only (system `libfftw3` + `libflint`).
Known linkage workaround for Julia 1.12 bundled libgmp / system libflint ABI
mismatch is in `__init__`. See bead `su2fft-e5z` for the robust-fix plan.

Follow-ups: arb-precision bindings (Arblib.jl), BinaryBuilder for portable
distribution, macOS `.dylib`, registration to the Julia General registry.

---

## What is working

- `su2_fft` / `su2_fft_inv`: double-precision O(N^4) forward+inverse, closed grid. Cross-validated to 2.21e-13 vs direct at N=24.
- `su2_fft_gl` / `su2_fft_inv_gl`: Gauss-Legendre theta nodes; DC normalisation exact. Phi/psi aliasing documented (bead `ega`).
- `su2_fft_resolved` / `su2_fft_resolved_inv` (bead `0t1`, May 2026): open P=2N-1 phi/psi grid + GL theta; exact spectrum roundtrip. Max relative error 4.45e-15 at N=8 (was ~0.197 on the closed-grid GL path).
- `su2_convolve`: per-l matrix product in spectrum domain; 1e-12 residual (bead `d7v`).
- `su2_fft_sphere` / `su2_fft_sphere_inv`: spherical-harmonic FFT on S^2; Y_0^0 synthesis 1e-13 (bead `5fb`).
- `su2_wigner_d_half`: half-integer Wigner-d evaluation via de Moivre + tgamma; spin-1/2 closed forms at 1e-12 (bead `n8e`).
- Arb path (`src/su2_fft_arb.c`): certified interval arithmetic at prec=512; worst-case ball radius 9.61e-154 at N=6.
- Julia bindings (`julia/SU2FFT.jl`): ccall wrappers for all the above (bead `t6z`).

---

## Observed results

### Cross-comparison (direct vs fast, `make bench`)

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

The maximum coefficient-wise difference stays within floating-point noise
across the full range: the two algorithms compute the same discrete sum.
At N=24 the max-diff is 2.21e-13, higher than the pre-recurrence figure of
7.12e-17 because the ascending-l recurrence accumulates floating-point error
over l steps; it remains well within the 1e-10 cross-check tolerance.
The table above is the cold-cache baseline; at N=24 the speedup is 90.91x
cold.  Warm-run variance is ~15% and the mean of warm runs sits near 103x
after bead `su2fft-dyi` (inline `ipow` replacing libm `pow`).  At small N
the FFTW plan overhead dominates and the direct path wins -- the standard FFT
crossover.

### Arb-precision path (`bench/arb_bench`)

```
Arb-precision FFT at N=6
  prec |    time(s) |     max ball
-------+------------+-------------
    53 |      0.008 |     1.44e-15
   128 |      0.009 |     3.78e-38
   256 |      0.015 |     1.64e-76
   512 |      0.023 |    9.61e-154
```

At prec=512 every output coefficient is certified to 154 decimal digits. The
overhead from prec=53 to prec=512 is only ~3x: the loop is dominated by
O(N^4) acb multiplications whose cost grows nearly linearly with limb count
at these sizes.

### Spectrum visualisation

![spectrum](build/spectrum.png)

---

## Math primer

<details>
<summary>Core formulas (click to expand)</summary>

The **SU(2) Fourier transform** at bandlimit N turns a function
f : SU(2) -> C sampled on an N x N x N Euler-angle grid into coefficients
indexed by integer l in [0, N-1] and m, n in [-l, l]:

```
fhat(l)_{m,n} = (dphi * dtheta * dpsi / 8pi^2)
                * sum_{j1, k, j2}  f(g_{j1,k,j2})
                                   * conj(t^l_{n,m}(g_{j1,k,j2}))
                                   * sin(theta_k)
```

The **matrix coefficient** (Peter-Weyl basis element) factorises over the
three Euler angles phi, theta, psi:

```
t^l_{n,m}(phi, theta, psi) = P^l_{n,m}(cos theta) * exp(-i(n*phi + m*psi))
```

The **Wigner function** P^l_{n,m} is related to the Sakurai small-d by a
phase:

```
P^l_{n,m}(cos beta) = i^{m-n} * d^l_{n,m}(beta)
```

where the small-d is evaluated stably via the de Moivre sum (O(l) terms,
bounded factorial ratios), not the Rodrigues derivative formula which loses
all double-precision significance around l ~ 10.

The **O(N^4) speedup** comes from one observation: the (phi, psi) part of
every matrix coefficient is exp(+i n*phi) * exp(+i m*psi), independent of l.
A single 2-D FFT per theta-slice (Stage 1, cost O(N^3 log N)) produces
F2[k, n, m] that is reused across all O(N) values of l sharing each (m, n)
pair. Stage 2 then evaluates O(N^3) inner products of length N, giving the
O(N^4) total. The paper's own optimisation analysis (Theorem TEO_FIN, line
1484) identifies k=1 as the cost-minimal choice -- which is exactly what this
implementation uses.

</details>

---

## Credit

This implementation follows the algorithm of:

> Delgado et al., "On The Fast Fourier Transform on SU(2)", arXiv 2605.23923,
> May 2026.  https://arxiv.org/abs/2605.23923

This is an **independent realisation** of their algorithm. The paper authors
have no involvement with this codebase. Any errors in the implementation --
numerical, algorithmic, or otherwise -- are ours alone.

The quantum signal processing motivation draws on:

> Bastidas, V. M. and Joven, K. J., "Complexification of quantum signal
> processing and its applications" (2024), cited as [bastidas2024complexification]
> in the paper bibliography.

### Prior art

The Delgado et al. algorithm is itself an SU(2) extension of the
divide-and-conquer FFT lineage on S^2 and SO(3). The relevant prior work,
in roughly chronological order:

> Driscoll, J. R. and Healy, D. M., "Computing Fourier transforms and
> convolutions on the 2-sphere", Adv. Appl. Math. 15 (1994) 202-250.

> Healy, D. M., Rockmore, D. N., Kostelec, P. J. and Moore, S. S. B.,
> "FFTs for the 2-Sphere -- Improvements and Variations", J. Fourier Anal.
> Appl. 9 (2003) 341-385. Cited as `\cite{2-sphere}` in `paper.tex` lines 232
> and 598 as the **primary inspiration** for Delgado et al.

> Kostelec, P. J. and Rockmore, D. N., "FFTs on the Rotation Group",
> J. Fourier Anal. Appl. 14 (2008) 145-179. The C library **SOFT 2.0** that
> ships with this paper is the closest existing implementation of an SO(3)
> FFT and uses an open uniform alpha/gamma grid + Gauss-Legendre beta, the
> same convention we adopt in `su2_fft_resolved` (bead `su2fft-0t1`).

> Price, M. A. and McEwen, J. D., "Differentiable and accelerated
> spherical harmonic and Wigner transforms", JCP (2024), arXiv:2311.14670.
> The JAX-based **s2fft** library implements the integer-l SO(3) Wigner
> transform; its Gauss-Legendre sampling mode uses P = 2L+1 open uniform
> in alpha/gamma, identical to the resolved-grid convention here.

What `su2-fft` adds beyond these references:
1. **Arbitrary-precision certification** via FLINT `acb_t` ball arithmetic.
   SOFT 2.0 is double-precision only; s2fft is float32/64 JAX. The arb path
   delivers spectrum roundtrip at user-chosen precision -- 4.40e-72 relative
   error at N=8, prec=256 in `test_resolved_arb` -- which neither surveyed
   library provides.
2. **Half-integer Wigner-d evaluation** (bead `su2fft-n8e` Tier 1). SOFT and
   s2fft target integer-l SO(3); half-integer (true SU(2)) is out of scope
   for them. Full half-integer FFT is tracked as bead `su2fft-u9q`.
3. **Cross-validated four-path implementation** (direct/fast x double/arb).
   The gold-standard `test_resolved_arb_fast_vs_direct` cross-check at
   prec=128 catches algorithmic errors that single-precision unit tests
   cannot.

The structural divide-and-conquer FFT is not novel to this codebase; the
contribution is in the C+FLINT engineering, the arbitrary-precision
certification, and the half-integer support. The grid convention
(P = 2L+1 open phi/psi + Gauss-Legendre theta) is the community standard
shared with SOFT 2.0 and s2fft.

### Libraries

- **FFTW3** (Matteo Frigo, Steven G. Johnson). Powers Stage-1 2-D backward
  DFT in the double-precision fast algorithm. License: GPL-2.0-or-later.
  https://fftw.org/

- **FLINT** (The FLINT development team). Powers the arb/acb arbitrary-precision
  parallel path: `acb_dft_prod`, `arb_fac_ui`, `arb_cos`, `arb_sin`, and all
  ball arithmetic. License: LGPL-2.1-or-later.
  https://flintlib.org/

---

## License

AGPL-3.0. See `LICENSE`.
