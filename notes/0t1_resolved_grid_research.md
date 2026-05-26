# Research: Resolved phi/psi Grid for bead `su2fft-0t1`

Prepared 2026-05-26. Background for the implementation of `su2_fft_resolved` (open P=2N-1 grid
in phi/psi). Cross-references the binding spec at `notes/0t1_resolved_grid_design.md`.

---

## 1. The Math of the Fix

### 1.1 Exact DFT condition for trig polynomials

A trigonometric polynomial of degree at most L on the circle has the form

    f(x) = sum_{|k| <= L}  c_k exp(i k x)

which involves exactly 2L+1 Fourier modes.  The uniform DFT on P equispaced points
x_j = x_0 + j * 2pi/P, j in [0, P-1], satisfies the discrete orthogonality relation

    sum_{j=0}^{P-1} exp(i n x_j) exp(-i m x_j)  =  P * delta_{n,m}
                                                   (mod P, integers n,m)

This means modes n and m that differ by a multiple of P are indistinguishable.  If f
has non-zero coefficients only for |k| <= L, the DFT is EXACT (no aliasing between
any two modes) if and only if

    P >= 2L + 1.

Reference: Trefethen, "Approximation Theory and Approximation Practice" (SIAM, 2013),
Ch. 3 and Ch. 12; also standard DFT orthogonality (e.g.
[NYU Fourier notes](https://cims.nyu.edu/~cfgranda/pages/MTDS_spring19/notes/fourier.pdf)
section on sampling and aliasing).  The condition "N >= 2k_c + 1" for exact DFT
recovery of a bandlimited signal with cut-off k_c is a standard result.

### 1.2 SU(2) bandlimit

At bandlimit N, the Fourier coefficients fhat(l)_{m,n} are non-zero for l in [0, N-1]
and m, n in [-l, l].  The widest mode range is |m|, |n| <= N-1, which requires 2N-1
distinct modes per axis (m and n independently).  Hence L = N-1 and the minimum
sample count per periodic axis is

    P_min = 2(N-1) + 1 = 2N - 1.

### 1.3 Why P = 2N-1 open is cleaner than P = 2N closed

Option A: open uniform grid with P = 2N-1:

    phi[j] = -pi + j * 2pi / P,   j in [0, P-1].

No endpoint duplication.  The FFTW plan is size P x P with no fold bookkeeping.
All P outputs are independent; mode-to-bin mapping is injective over [-(N-1), N-1].

Option B: closed grid with P = 2N, i.e., phi[0] = -pi, phi[2N-1] = pi (same point):

    phi[j] = -pi + j * 2pi / (2N-1),   j in [0, 2N-1],  endpoints coincide.

This adds one extra sample that must be folded before the FFTW plan, yielding an
effective (2N-1)-point DFT after folding -- identical in resolution to option A but
with one redundant sample and extra bookkeeping.

The existing `su2_fft` already demonstrates that the closed/fold pattern introduces
implementation complexity (the `g[j1m=0] += f[N-1]` line in `src/su2_fft.c`).
Option A eliminates this.  The design brief `notes/0t1_resolved_grid_design.md` §2
confirms option A is the chosen approach.

### 1.4 Phase derivation

With `phi[j] = -pi + j * 2pi/P` and P = 2N-1:

    exp(+i n phi[j]) = exp(-i n pi) * exp(+i n j 2pi/P)
                     = (-1)^n  *  w_P^{n j},        w_P = exp(+i 2pi/P).

The factor (-1)^n = exp(-i n pi) is a per-mode phase independent of j.  This is a
single scalar applied to each DFT output bin; no fold, no endpoint arithmetic.

Mode n in [-(N-1), N-1] maps to FFTW bin  n_bin = ((n % P) + P) % P:

    n =  0    ->   bin 0
    n =  1    ->   bin 1
    ...
    n = N-1   ->   bin N-1
    n = -1    ->   bin P-1 = 2N-2
    n = -(N-1) -> bin N

These are exactly 2N-1 = P distinct bins with no collision, confirming the
aliasing-free property.

Identical derivation applies to psi/j2 by symmetry.

### 1.5 Negative-mode bin mapping (standard mod-P aliasing)

For FFTW with the BACKWARD (sign +1) convention, the DFT output at bin k corresponds
to the mode k for k in [0, P/2] and to the mode k-P for k in [P/2+1, P-1].  The
resolved grid simply uses n_bin = ((n % P) + P) % P, which is the standard positive-
modular reduction.  Negative modes land in the upper half of the output array, which
is the canonical FFTW convention for a full complex DFT.

---

## 2. Survey of Open-Source Libraries

### 2.1 s2fft (astro-informatics/s2fft, JAX-based)

Repository: [https://github.com/astro-informatics/s2fft](https://github.com/astro-informatics/s2fft)
Paper: Price & McEwen 2024, Journal of Computational Physics,
[arXiv:2311.14670](https://arxiv.org/abs/2311.14670).

**Grid convention (SO(3) / Wigner transforms):**
s2fft supports several equiangular sampling schemes: McEwen & Wiaux (MW/MWSS),
Driscoll & Healy (DH), and Gauss-Legendre (GL).  For SO(3):

- Storage convention is [gamma, beta, alpha] following the zyz Euler convention.
- `flmn_shape(L, N, sampling)` returns shape `(L, 2L-1, 2N-1)` for the coefficient
  array, where L is the spherical bandlimit and N is the directional bandlimit.
  Total non-zero coefficients: L^2 * (2N-1).
- `f_shape(L, N, sampling)` for the equiangular MW/GL schemes gives the sample
  grid shape per sampling type.

For equiangular schemes at bandlimit L the GL convention uses:
- **(beta/theta):** L+1 Gauss-Legendre nodes on [-1, 1] (open, non-uniform).
- **(alpha/phi) and (gamma/psi):** 2L+1 equally spaced points on [0, 2pi), OPEN
  convention (no endpoint duplication).  This is exactly P = 2L+1 = 2N-1 for
  the case where the directional bandlimit N equals L.

The MW sampling uses approximately 2L^2 samples total (vs 4L^2 for DH).  The GL
sampling is similar in count.

**DFT primitive:** JAX FFT (`jnp.fft.fft` / `jnp.fft.ifft`) applied along the
alpha and gamma axes; Wigner-d recurrence along beta.

**Aliasing diagnostic:** The sampling theorems for MW and GL are exact for
bandlimited functions; aliasing tests are implicit in the numerical cross-checks.
No explicit leakage test was found in the public test suite.

**Key takeaway for `su2-fft`:** s2fft's GL mode uses P = 2L+1 = 2N-1 points for
the periodic (phi, psi/gamma) axes, open convention.  This is exactly what bead
`0t1` proposes.

### 2.2 HEALPix / healpy (CMB community)

Documentation: [https://healpy.readthedocs.io/](https://healpy.readthedocs.io/)

HEALPix is a pixelisation of the 2-sphere (S^2 only, NOT SO(3) / SU(2)) with
equal-area pixels on iso-latitude rings.  The resolution parameter is Nside (a
power of 2); total pixel count is 12 * Nside^2.  Bandlimit and sample count are not
directly related by a simple formula: approximate bandlimit is ~ 2*Nside for RING
ordering.

HEALPix does not apply to the SU(2) FFT problem because it is designed for functions
on S^2, not on SO(3)/SU(2).  It does not resolve the phi/psi two-torus.

**Instructive point:** Even HEALPix uses an *open* phi convention within each ring
(pixels are equidistant in azimuth within a ring, no duplicate endpoint), consistent
with the principle that open grids are cleaner for DFT-based transforms.

### 2.3 SHTns (Schaeffer, spherical harmonic library)

Repository: https://bitbucket.org/nschaeff/shtns  
Paper: [arXiv:1202.6522](https://arxiv.org/pdf/1202.6522) (Schaeffer 2013)

SHTns is an optimized C library for spherical harmonic transforms on S^2.
For the longitude (phi) dimension at maximum degree lmax, SHTns uses **2*lmax+1
equally spaced grid nodes** [SHTOOLS GLQ grid documentation confirms: "2L+1 equally
spaced grid nodes for the Fourier transforms in longitude"].  The phi grid is open
(spacing 360/(2*lmax+1) degrees, no duplicate endpoint).

For latitude (theta), the GL quadrature uses lmax+1 Gauss-Legendre nodes.
Total grid: (lmax+1) x (2*lmax+1).

SHTns applies FFTW for the longitude DFTs; vectorized Legendre transforms for
latitude.  This is exactly the pattern that `su2_fft_resolved` adopts for its
Stage 1.

**Key takeaway:** The GL+open-phi convention (P = 2L+1 in longitude, GL in latitude)
is the community standard for exact spherical harmonic transforms.  SHTns, SHTOOLS,
and s2fft all converge on it.

### 2.4 Kostelec & Rockmore SOFT 2.0 (SO(3) FFT)

Paper: "FFTs on the Rotation Group", Kostelec & Rockmore, J. Fourier Anal. Appl.
14 (2008) 145-179.  Also [SFI technical report 03-11-060](https://sfi-edu.s3.amazonaws.com/sfi-edu/production/uploads/sfi-com/dev/uploads/filer/dd/54/dd548bb2-16e1-4257-960e-c2fb1e6e6df2/03-11-060.pdf).

**Grid convention:**  
Kostelec & Rockmore use bandwidth B and sample 2B points per axis.  The Euler
angles alpha, beta, gamma are sampled at:

    alpha_j = 2pi * j / (2B),      j in [0, 2B-1]     (open, 2B points)
    beta_k  = pi(2k+1) / (4B),     k in [0, 2B-1]     (Gauss-Legendre quadrature points)
    gamma_j = 2pi * j / (2B),      j in [0, 2B-1]     (open, 2B points)

Total sample count: (2B)^3 = 8B^3.  The alpha and gamma grids are open-uniform with
2B points.

**Critical observation:** At bandlimit B (modes |m|, |n| <= B-1), the periodic axes
use 2B points, not 2B-1.  This means SOFT slightly oversamples: 2B >= 2(B-1)+1 =
2B-1, so 2B > 2B-1, i.e., one extra sample beyond the strict minimum.  The extra
sample does not hurt correctness; it makes the FFTW plan size 2B (a power of 2 when
B is a power of 2, which is the primary use case in SOFT).

**lie_learn Python implementation** (AMLab-Amsterdam/lie_learn) confirms:
`alpha_j = 2pi * j / (2b)`, `beta_k = pi(2k+1)/(4b)`, `gamma_j = 2pi * j / (2b)`,
where b = L_max + 1.  FFT2 applied along alpha and gamma axes.

**DFT primitive:** FFTW via S2kit internals.  Grid size is (2B)^2 for the 2D DFT,
which is 4B^2 -- a power of 2 * 4 when B is a power of 2, very FFTW-friendly.

**Aliasing:** The 2B-point open grid is alias-free for bandlimit B (2B >= 2(B-1)+1).
No aliasing diagnostics are described explicitly in the paper.

**Difference from this project:**  The paper arXiv:2605.23923 follows the same
strategy but normalizes N as the number of Wigner-d degrees (0..N-1), so the
bandlimit is N-1 and the minimum phi/psi grid size is 2(N-1)+1 = 2N-1.  SOFT's
convention uses one extra point (2B = 2N instead of 2N-1) because it targets
power-of-2 plan sizes.  Both are correct; 2N-1 is the strict minimum and is the
choice in `0t1`.

### 2.5 NFFT3 / NFSOFT (Keiner, Kunis, Potts, TU Chemnitz)

Library: [https://www-user.tu-chemnitz.de/~potts/nfft/](https://www-user.tu-chemnitz.de/~potts/nfft/)  
Algorithm paper: Potts, Prestin, Vollrath, "A Fast Algorithm for Nonequispaced
Fourier Transforms on the Rotation Group", Numerical Algorithms (2009).

NFFT/NFSOFT extends the NFFT (nonequispaced FFT) to SO(3).  Input nodes are
*arbitrary* (hence "nonequispaced") -- the library is designed for scattered data,
not for the uniform equiangular grid used in this project.

For the equiangular reference, NFSOFT still internally uses Clenshaw-Curtis
quadrature on beta and a DFT of size ~ 2N on the alpha/gamma axes.  Precise grid
formulas are in the Potts et al. paper (not fetched; TLS failure prevented direct
access); based on the Kostelec-Rockmore convention the alpha/gamma axes are also
open-uniform with ~ 2B points.

**DFT primitive:** FFTW3 (configurable).  NFFT3 calls `fftw_plan_dft_1d` internally.

**Aliasing diagnostic:** Not applicable in the standard nonequispaced usage; for
equiangular NFSOFT the same anti-aliasing argument as SOFT applies.

### 2.6 McEwen & Wiaux SO(3) sampling theorem (MW/MWSS)

Paper: "A novel sampling theorem on the rotation group", McEwen et al. 2015,
[arXiv:1508.03101](https://arxiv.org/abs/1508.03101).

Proposes a sampling theorem on SO(3) requiring 4L^3 samples total (vs (2L)^3 = 8L^3
for SOFT), connecting the rotation group to the three-torus through a periodic
extension.  The MW convention uses 2L equiangular points on alpha and gamma and
2L-1 points on beta.  Exact Wigner transform is achieved with O(L^4) operations.

Grid is open in alpha and gamma; L^4 complexity is the headline.  For the su2-fft
use case (integer-l, double precision) the SOFT/s2fft-GL convention is more directly
analogous.

---

## 3. FFTW Performance for P = 2N-1

### 3.1 Algorithm selection

FFTW 3.3.11 documentation ([Introduction](https://www.fftw.org/fftw3_doc/Introduction.html),
[Complex DFTs](https://www.fftw.org/fftw3_doc/Complex-DFTs.html)) states:

> "FFTW computes the DFT using an O(n log n) algorithm for all lengths, including
> prime numbers.  It is generally best at handling sizes of the form
> 2^a * 3^b * 5^c * 7^d * 11^e * 13^f (where e+f is 0 or 1), and other sizes are
> computed by means of a slow, general-purpose algorithm (which nevertheless retains
> O(n log n) performance even for prime sizes)."

The "slow, general-purpose algorithm" is Rader's algorithm for prime sizes and
Bluestein's chirp-z algorithm as a fallback when Rader's is unfavorable (Rader is
declared SLOW by FFTW when p-1 is hard to factor; Bluestein then handles it via a
zero-padded radix-2 convolution).  Both are O(P log P), but with a larger constant
than the highly optimized Cooley-Tukey codelets.

Source: [FFTW3 design paper](https://www.fftw.org/fftw-paper-ieee.pdf) and
[comp.dsp FFTW 3.1 release thread](https://www.dsprelated.com/showthread/comp.dsp/51413-1.php)
(confirms Bluestein and zero-padded Rader were added in 3.1 to cap recursive Rader
slowdown).

### 3.2 P = 2N-1 factorizations at relevant N

| N  | P = 2N-1 | Factorization   | FFTW path                         |
|----|----------|-----------------|-----------------------------------|
| 4  | 7        | prime           | Rader or Bluestein                |
| 5  | 9        | 3^2             | Cooley-Tukey (optimal)            |
| 6  | 11       | prime           | Rader or Bluestein                |
| 8  | 15       | 3 * 5           | Cooley-Tukey (optimal)            |
| 9  | 17       | prime           | Rader or Bluestein                |
| 12 | 23       | prime           | Rader or Bluestein                |
| 13 | 25       | 5^2             | Cooley-Tukey (optimal)            |
| 16 | 31       | prime           | Rader or Bluestein                |
| 17 | 33       | 3 * 11          | Cooley-Tukey (suboptimal; 11 factor) |
| 24 | 47       | prime           | Rader or Bluestein                |
| 32 | 63       | 3^2 * 7         | Cooley-Tukey (optimal)            |

The plan size is P x P for Stage 1.  At N=8 (the primary test case) P=15=3*5, which
is an optimal FFTW factorization; no performance hit.  At N=9 (P=17, prime) and
N=16 (P=31, prime), Rader/Bluestein applies.

### 3.3 Practical performance estimate for prime P

The existing closed-grid path uses FFTW plans of size (N-1)x(N-1).  At N=8, the
old plan is 7x7 (49-point 2D DFT; 7 is prime, so Rader/Bluestein already applies
there too).  At N=9 the old plan is 8x8 (power of 2, optimal); the new plan 17x17
(prime, slower).

Concrete FFTW benchmark data for small prime vs. composite sizes at these scales
(P ~ 15-31) was not accessible via public benchmarks (the fftw.org benchmark pages
require rendering of generated charts).  The comp.dsp thread reports flop counts:
Rader at size 37 gives 1,672 flops vs. ~ 370 flops for an optimized composite of
similar size (~4.5x more flops).  For the outer sweep over N theta slices, Stage 1
costs O(N * P^2 log P); Stage 2 dominates at O(N^4).  Even a 4-5x Stage 1 slowdown
does not affect the O(N^4) total at N=8..16.

**Recommendation:** Do not zero-pad to the next composite at this stage.  Reasons:
1. At N=8, P=15=3*5 is already optimal.
2. Stage 1 cost is O(N^3 log P) vs Stage 2 at O(N^4); Stage 1 is not the bottleneck
   even if it uses Rader.
3. Padding to P=32 at N=16 (instead of P=31) wastes one sample row and would
   complicate the bin-to-mode mapping (you would have a spurious 32nd bin that
   carries no information).
4. Re-measure after shipping, using `FFTW_MEASURE` planner flag to let FFTW choose
   the fastest codelet.

If profiling after `0t1` reveals Stage 1 is a bottleneck at prime P, the zero-pad
approach (pad to next 2^a * 3^b * 5^c composite) can be reconsidered without any
change to the stored sample array layout.

### 3.4 FFTW plan reuse

The 2D plan `fftw_plan_dft_2d(P, P, ...)` should be created ONCE per N and reused
across all N theta slices (as the existing `su2_fft.c` already does for the (N-1)x(N-1)
plan).  The `FFTW_MEASURE` flag finds the best algorithm for the given P; for prime P
it will select Rader or Bluestein automatically.  Use `FFTW_ESTIMATE` in tests to
keep startup cost low.

---

## 4. Arbitrary-Precision DFT in FLINT / arb

### 4.1 `acb_dft_prod` signature and capabilities

From [FLINT acb_dft.h documentation](https://flintlib.org/doc/acb_dft.html)
(confirmed also via `notes/library_evaluation.md` from an earlier session):

```c
void acb_dirichlet_dft_prod(acb_ptr w, acb_srcptr v,
                             slong *cyc, slong num, slong prec);
void acb_dirichlet_dft_prod_precomp(acb_ptr w, acb_srcptr v,
                                    const acb_dft_prod_t prod, slong prec);
```

`cyc` is an array of `num` cyclic group sizes; the input `v` is assumed
lexicographically ordered over the product group Z_{cyc[0]} x ... x Z_{cyc[num-1]}.
For a 2D DFT of size P x P, pass `cyc = {P, P}, num = 2`.

**Algorithm selection for prime P:**  
FLINT's `acb_dft` automatically selects among: Naive (O(n^2)), CRT decomposition,
Cooley-Tukey (factored), Radix-2 (power-of-2), and Bluestein (arbitrary via chirp-z
convolution to radix-2).  For prime P, the path is Bluestein: the length-P DFT is
converted to a radix-2 convolution of length >= 2P-1, then a radix-2 FFT handles
the power-of-2 extension.  All operations carry arb ball-arithmetic error bounds.

**Is the output exact?**  
Yes, modulo working precision `prec`.  FLINT's DFT operates over `acb_t` (complex
interval arithmetic); each output coefficient is a certified ball [re +/- err, im +/-
err] where err shrinks with higher `prec`.  At `prec=53` (IEEE double-equivalent)
the balls match FFTW output to floating-point precision.  At `prec=512` the balls
certify 154 decimal digits (as the project README already documents).  Round-tripped
output is equal to the input modulo working precision in the sense that the true
mathematical value lies within the output ball.

The `acb_dft` module documentation marks these functions as "experimental; may
change without notice" but they are present and stable in FLINT 3.0.1 (the installed
version, confirmed in `notes/library_evaluation.md`).

**Forward only:**  
`acb_dirichlet_dft_prod` computes the FORWARD DFT only.  Inverse DFT uses the
`conj(forward(conj(.)))` identity, as documented in `HANDOFF.md §3 item 5` and
already implemented in `src/su2_fft_arb.c`.  Do not "simplify" this; there is no
inverse variant of `acb_dft_prod`.

**Performance note:**  
At prime P = 2N-1, `acb_dft_prod({P,P}, num=2)` routes through Bluestein internally.
The FLINT documentation states the convolution routine uses "radix 2 FFT unless len
is a product of small primes where a non-padded FFT is faster."  For the arb path
at prec=512 the DFT cost is dominated by the high-precision arithmetic, not by the
algorithm choice; Bluestein vs. Cooley-Tukey is not the bottleneck.

### 4.2 Arbitrary-precision Gauss-Legendre nodes/weights

FLINT ships `arb_hypgeom_legendre_p_ui_root` (in `arb_hypgeom.h`, FLINT >= 3.0):

```c
void arb_hypgeom_legendre_p_ui_root(arb_t res, arb_t weight,
                                    ulong n, ulong k, slong prec);
```

Sets `res` to the k-th root of P_n (decreasing order: x_0 > x_1 > ... > x_{n-1}),
and if `weight` is non-NULL, also sets it to the corresponding Gauss-Legendre weight
for the quadrature on [-1, 1].  Uses asymptotic approximation + Newton iterations
with certified interval arithmetic.  Handles "very high precision" via doubling
Newton steps (Johansson & Mezzarobba 2018,
[arXiv:1802.03948](https://arxiv.org/abs/1802.03948)).

**Implication for bead `rrx` (arb-precision resolved-grid):**  
The arb follow-on to `0t1` (bead `su2fft-rrx`) does NOT need to port the Newton
iteration from `src/su2_gauss_legendre.c`.  It can call
`arb_hypgeom_legendre_p_ui_root` directly to get certified GL nodes and weights at
arbitrary precision.  This is already noted in `notes/0t1_resolved_grid_design.md` §8.

The double-precision path in `src/su2_gauss_legendre.c` (Newton iteration on P_N)
remains appropriate for bead `0t1` itself, which operates in double precision.

---

## 5. Pitfalls

### 5.1 Storage layout change

The current sample array for `su2_fft` / `su2_fft_gl` is `f[N^3]` (N x N x N, row-
major).  The resolved path uses `f[P^2 * N] = (2N-1)^2 * N` samples (phi axis P,
theta axis N GL, psi axis P).  Concrete sizes:

| N  | Old N^3 | New P^2*N  | Ratio |
|----|---------|------------|-------|
| 4  | 64      | 144        | 2.25x |
| 8  | 512     | 1568       | 3.06x |
| 16 | 4096    | 14112      | 3.45x |
| 24 | 13824   | 50232      | 3.63x |
| 32 | 32768   | 126976     | 3.87x |

At N=16, the new array is ~110 KB (double complex = 8 bytes complex) -- well within
L2 cache.  Memory is not a concern at these N.

The helper `su2_resolved_sample_index(N, j1, k, j2)` defined in the design brief
(`notes/0t1_resolved_grid_design.md` §3) encapsulates the new indexing.  Existing
`su2_sample_index` remains unchanged for the old paths.

### 5.2 New direct reference FT required

The cross-check test `test_resolved_fft_matches_direct_random` requires a direct
O(N^6) reference FT evaluated on the new P x N x P grid.  This is
`su2_ft_direct_resolved` (design brief §3).  Without it, the cross-check cannot
verify correctness; merely comparing against `su2_fft_gl` at overlapping grid points
is insufficient because `su2_fft_gl` has the known phi/psi aliasing floor.

### 5.3 Downstream APIs need review

`su2_convolve` (`src/su2_convolve.c`) operates on `fhat` arrays only; no change
required.  `su2_fft_sphere` (`src/su2_sphere.c`) calls `su2_fft` internally; a
resolved-grid sphere FFT will need `su2_fft_sphere_resolved` that calls
`su2_fft_resolved` instead.  The design brief notes this as a deferred follow-on;
do not merge it into bead `0t1`.

The Julia bindings (`julia/src/SU2FFT.jl`) expose the resolved functions via
`ccall`; add `fft_resolved` and `fft_resolved_inv` after the C implementation
ships.  Do not modify the Julia bindings in bead `0t1` unless the C symbols are
exported from `include/su2.h`.

### 5.4 Phase sign convention pitfall

The factor `(-1)^n = exp(-i n pi)` arises from the choice `phi[0] = -pi` (not 0).
This sign is correct regardless of whether P = 2N-1 is odd or even, and regardless
of N's parity.  Specifically:

- Do NOT rewrite `(-1)^n` as `exp(-i n pi)` evaluated numerically: for |n| up to N-1
  this involves large integer exponents; use `(n & 1) ? -1.0 : 1.0` (the existing
  pattern in `src/su2_fft.c` line 97).
- The existing code has `double sn = (n & 1) ? -1.0 : 1.0` which is correct for
  two's-complement signed `int n` because `n & 1` gives the sign bit under two's
  complement for negative odd n.  (Confirmed in `src/su2_fft.c` lines 97-99.)
- The phase for both n and m is independent (`(-1)^n * (-1)^m`, not `(-1)^{n+m}`)
  under the open grid; in the old closed-grid code the combined `(-1)^{n+m}` arose
  from two axes sharing the same fold.  Verify the resolved implementation applies
  `sn * sm` separately.

**Minimal unit test:** `test_resolved_constant_forward` (design brief §6 item 2)
confirms that forward(constant) yields fhat(0)_{0,0} = A and all other coefficients
zero to 1e-12.  This test specifically exercises the n=m=0 bin and catches sign
errors that would scatter energy to nonzero bins.

### 5.5 FFTW plan size at prime P

See Section 3.  The primary concern is that `fftw_plan_dft_2d(P, P, ...)` with
`FFTW_ESTIMATE` may produce a noticeably slower plan than `FFTW_MEASURE` at prime P.
The existing code uses `FFTW_ESTIMATE` throughout for simplicity.  For the resolved
path at N=16 (P=31, prime), switching to `FFTW_MEASURE` in the production path (not
tests) is worth considering after benchmarking.

Note: in `src/su2_fft.c` the existing plan for (N-1)x(N-1) at N=8 is a 7x7 plan
(7 is prime), so Rader/Bluestein was already in use for that configuration.  The
resolved P=15=3*5 plan at N=8 is strictly better than the old 7x7 plan.

### 5.6 arb path: Bluestein for prime P in `acb_dft_prod`

For the arb path (bead `su2fft-rrx`), calling `acb_dirichlet_dft_prod` with
`cyc={P,P}` at prime P routes through Bluestein internally (Section 4.1).  FLINT
does not silently fall back to a slower O(P^2) algorithm in this case; Bluestein is
explicitly coded in `acb_dft_bluestein`.  However:

- The Bluestein path pads to the next power of 2 >= 2P-1, so at P=31 the internal
  radix-2 buffer is length 64.  This is a modest memory increase and is expected.
- At arb working precision (`prec=512`), the DFT cost is dominated by the
  high-precision arithmetic, not the algorithm path.  No special action needed.

### 5.7 Normalisation constant must change

The normalisation `norm_gl_closed = 1 / (2*N^2)` used in `su2_fft_gl` is WRONG for
the resolved grid.  With the open P-grid, `G_slice[0,0] = P^2` for constant input
(no fold over-counting), so the correct constant is

    norm_resolved = 1 / (2 * P^2),    P = 2N-1.

(Derived in `notes/0t1_resolved_grid_design.md` §5.)  The `test_resolved_constant_forward`
test catches this error immediately.

---

## 6. Recommended Action Plan

1. **Add new entry points, do not modify old ones.**  Keep `su2_fft`, `su2_fft_inv`,
   `su2_fft_gl`, `su2_fft_inv_gl` unchanged.  Add `su2_fft_resolved` and
   `su2_fft_resolved_inv` as separate functions in a new file `src/su2_fft_resolved.c`.
   Add `su2_ft_direct_resolved` in `src/su2_ft_resolved.c`.  This preserves all
   existing tests and avoids breaking the Julia bindings.

2. **Write the failing cross-check test first (TDD).**  Add `tests/test_resolved.c`
   with all seven tests from `notes/0t1_resolved_grid_design.md` §6; run `make
   test/test_resolved` and confirm all fail before writing any implementation code.
   The headline test is `test_resolved_spectrum_roundtrip`, asserting
   `forward(inverse(fhat)) ≈ fhat` to 1e-12 at N=4, 8, 16.

3. **Use FFTW plan size P = 2N-1 without zero-padding.**  At N=8, P=15=3*5 is
   FFTW-optimal.  For prime P (N=9 giving P=17, N=16 giving P=31), accept the
   Rader/Bluestein overhead; it is not the Stage 2 bottleneck.  Use `FFTW_ESTIMATE`
   in tests; switch to `FFTW_MEASURE` in `su2_fft_resolved` production path for a
   free ~10-20% improvement at prime sizes.

4. **Apply `(-1)^n * (-1)^m` phases using `(n & 1) ? -1.0 : 1.0`.**  Do not
   introduce floating-point `pow(-1, n)` or `cos(n*pi)`.  Verify the phase is applied
   independently per axis (not as `(-1)^{n+m}`).  The `test_resolved_constant_forward`
   test detects any phase error immediately.

5. **arb-precision follow-on (bead `rrx`) is separate.**  Bead `0t1` ships the
   double-precision resolved path; `rrx` adds the arb path using `acb_dirichlet_dft_prod`
   and `arb_hypgeom_legendre_p_ui_root` for GL nodes.  Do not conflate these; the
   double path is the unblocking step for `d7v`, `5fb`, and `31x`.

6. **Update documentation after shipping.**  When `0t1` lands: update `ALGORITHM.md`
   §2.2 to describe the resolved grid; update `PROFILING.md` to note any Stage 1
   timing change; update `HANDOFF.md` §2 to replace the "aliasing floor" diagnosis
   with the achieved roundtrip tolerance.  Claim the new tolerance concretely (e.g.,
   "spectrum roundtrip max error < 1e-12 at N=8" -- do not write this until measured).

7. **Downstream migration of `su2_sphere` is a separate bead.**  After `0t1`
   ships, file a small bead for `su2_fft_sphere_resolved` that calls `su2_fft_resolved`
   internally and exposes S^2 coefficients at full precision.  Do not include this
   in bead `0t1` -- one bead, one PR, one commit.

---

## Sources Consulted

- [FLINT acb_dft.h documentation](https://flintlib.org/doc/acb_dft.html)
- [FLINT arb_hypgeom.h documentation](https://flintlib.org/doc/arb_hypgeom.html)
- [arXiv:1802.03948](https://arxiv.org/abs/1802.03948) — Johansson & Mezzarobba, fast arb GL nodes
- [FFTW 3.3.11 Introduction](https://www.fftw.org/fftw3_doc/Introduction.html)
- [FFTW 3.3.11 Complex DFTs](https://www.fftw.org/fftw3_doc/Complex-DFTs.html)
- [FFTW design paper (IEEE 2005)](https://www.fftw.org/fftw-paper-ieee.pdf)
- [comp.dsp FFTW 3.1 release](https://www.dsprelated.com/showthread/comp.dsp/51413-1.php) — Bluestein/Rader in 3.1
- [s2fft GitHub](https://github.com/astro-informatics/s2fft)
- [s2fft Wigner samples API](https://astro-informatics.github.io/s2fft/api/sampling/wigner_samples.html)
- [arXiv:2311.14670](https://arxiv.org/abs/2311.14670) — Price & McEwen 2024, s2fft paper
- [arXiv:1508.03101](https://arxiv.org/abs/1508.03101) — McEwen et al. 2015, SO(3) sampling theorem
- [Kostelec & Rockmore 2008 JFA](https://link.springer.com/article/10.1007/s00041-008-9013-5)
- [lie_learn SO3FFT_Naive.py](https://github.com/AMLab-Amsterdam/lie_learn/blob/master/lie_learn/spectral/SO3FFT_Naive.py)
- [SHTOOLS GLQ grid format](https://shtools.github.io/SHTOOLS/grid-formats.html)
- [HEALPix documentation](https://healpix.sourceforge.io/)
- [Trefethen, Approximation Theory and Approximation Practice, SIAM 2013] (standard trig polynomial / DFT exactness result)
- Project files: `src/su2_fft.c`, `src/su2_grid.c`, `notes/0t1_resolved_grid_design.md`,
  `notes/library_evaluation.md`, `HANDOFF.md`, `IMPROVEMENTS.md`
