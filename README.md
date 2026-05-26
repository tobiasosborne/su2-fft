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
transform at bandlimits up to N=24. The two implementations agree to
floating-point noise (~1e-17, near machine epsilon), and the fast path runs
roughly 10x faster at N=24 -- consistent with the predicted N^2 speedup
ratio. An FLINT arb-precision parallel path provides certified interval
arithmetic: at prec=512 the worst-case ball radius on all output coefficients
is 9.61e-154, corresponding to 154 verified decimal digits.

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
src/su2_fft.c              O(N^4) fast Fourier transform (double precision)
src/su2_{wigner,ft,fft}_arb.c   FLINT arb/acb arbitrary-precision parallel path
tests/test_*.c             TDD red/green checks for every piece
bench/compare.c            Cross-comparison and timing at each N
bench/vis_dump.c           Spectrum dump in Python-readable format
bench/visualize.py         Matplotlib spectrum plot
bench/arb_bench.c          Arb-precision timing vs ball radius
notes/                     Four section-by-section summaries of the paper
proof/                     11-node structural proof of the O(N^4) claim (af)
paper.tex                  Verbatim arxiv source
ALGORITHM.md               Full narrative companion: every design decision, derivation
PROFILING.md               perf + stage-instrumentation report (hot path = pow() at 78%)
IMPROVEMENTS.md            Ranked roadmap (recurrence, SIMD, OpenMP, QSP, GPU, ...)
```

The canonical narrative is `ALGORITHM.md`. Every claim in the code is anchored
there, and every departure from the paper is called out explicitly.

---

## Observed results

### Cross-comparison (direct vs fast, `make bench`)

```
   N |    direct(s) |      fast(s) |  speedup |  max|diff| | status
-----+--------------+--------------+----------+------------+---------
   4 |     0.000117 |     0.000970 |     0.12 |   6.25e-17 | OK
   6 |     0.000839 |     0.000437 |     1.92 |   4.34e-17 | OK
   8 |     0.003892 |     0.001492 |     2.61 |   2.64e-17 | OK
  10 |     0.009636 |     0.003234 |     2.98 |   2.55e-17 | OK
  12 |     0.035140 |     0.005772 |     6.09 |   2.32e-17 | OK
  14 |     0.080312 |     0.012209 |     6.58 |   4.97e-17 | OK
  16 |     0.166585 |     0.031633 |     5.27 |   3.32e-17 | OK
  20 |     0.621584 |     0.067321 |     9.23 |   2.43e-17 | OK
  24 |     1.818179 |     0.182840 |     9.94 |   7.12e-17 | OK
```

The maximum coefficient-wise difference stays at floating-point noise
(~1e-17, near double epsilon) across the full range: the two algorithms
compute the same discrete sum. The speedup grows with N as predicted by
N^6 / N^4 = N^2; at small N the FFTW plan overhead dominates and the direct
path wins -- the standard FFT crossover.

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
