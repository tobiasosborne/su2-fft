# SU(2) Fourier Transform: literate companion to the code

This document is the canonical narrative for the project.  Every code file in
`src/`, `tests/`, and `bench/` is a faithful realisation of a step described
here.  Every claim about the algorithm is anchored to a line in `paper.tex`
(arxiv 2605.23923 -- Delgado et al., "On The Fast Fourier Transform on SU(2)",
May 2026) or to a deliberate departure from it, which is called out explicitly.

---

## 0. Map of the repository

| Path                                 | Role                                                                |
|--------------------------------------|---------------------------------------------------------------------|
| `paper.tex`, `bib-...bib`, `FT_SU2.png` | Verbatim arxiv source.                                           |
| `notes/section_2_3_background.md`    | Section-2/3 summary (definitions, irreps, sampling grid).           |
| `notes/section_4_algorithm.md`       | Section-4 summary (the FFT itself, with pseudocode).                |
| `notes/section_5_6_7_complexity.md`  | Section-5/6/7 summary (complexity bounds, comparison table).        |
| `notes/library_evaluation.md`        | FLINT + FFTW3 stack rationale.                                      |
| `proof/`                             | `af` proof workspace: 11-node structural proof of the O(N^4) claim. |
| `include/su2.h`                      | Public API and storage-layout contract.                             |
| `src/su2_grid.c`                     | Euler-angle grid, coefficient layout helpers.                       |
| `src/su2_wigner.c`                   | Stable evaluation of paper's P^l_{n,m}(cos theta).                  |
| `src/su2_ft.c`                       | Direct O(N^6) reference Fourier transform.                          |
| `src/su2_fft.c`                      | O(N^4) fast Fourier transform (double precision).                   |
| `include/su2_arb.h`, `src/su2_*_arb.c` | FLINT arb/acb arbitrary-precision parallel path.                  |
| `tests/test_*.c`                     | Red/green TDD checks for each piece.                                |
| `bench/compare.c`                    | Cross-comparison and timing across N.                               |
| `bench/vis_dump.c` + `visualize.py`  | Spectrum visualisation (paper Figure 1 style).                      |
| `bench/arb_bench.c`                  | Arb-precision timing vs ball radius at prec = 53/128/256/512.       |

To build everything and run the test battery:

```sh
make test
make bench   # runs bench/compare
```

---

## 1. The object to compute

For an integer bandlimit `N`, the SU(2) Fourier transform takes a function
`f : SU(2) -> C` sampled on the Euler-angle grid (paper line 684):

```
phi[j1]   = -pi + j1 * 2pi/(N-1)        j1 in [0, N-1]
theta[k]  =      k  *  pi/(N-1)         k  in [0, N-1]
psi[j2]   = -pi + j2 * 2pi/(N-1)        j2 in [0, N-1]
```

and produces, for each integer `l in [0, N-1]` and each `m, n in [-l, l]`,

```
fhat(l)_{m,n} = (dphi * dtheta * dpsi / (8 pi^2)) *
                sum_{j1, k, j2}
                    f(g_{j1, k, j2}) *
                    conj( t^l_{n, m}(g_{j1, k, j2}) ) *
                    sin(theta_k)                                      (paper 1316)
```

with the matrix coefficient

```
t^l_{n, m}(phi, theta, psi) = P^l_{n, m}(cos theta) * exp(-i (n phi + m psi))   (line 534)
```

and the Rodrigues-form

```
P^l_{n,m}(x) = c^l_{n,m} *
               (1-x)^{(m-n)/2} / (1+x)^{(n+m)/2} *
               D^{l-n} [ (1-x)^{l-m} (1+x)^{l+m} ]                    (line 537)

c^l_{n,m} = 2^{-l} * (-1)^{l-m} * i^{m-n}
            / sqrt((l-m)!(l+m)!)
            * sqrt((l+n)!/(l-n)!)                                     (line 542)
```

Two storage conventions are used throughout the code, fixed once in
`include/su2.h`:

| symbol  | layout                                                                  |
|---------|-------------------------------------------------------------------------|
| `f`     | `f[j1*N*N + k*N + j2]`                                                  |
| `fhat`  | flat array of length `sum_{l<N} (2l+1)^2 = N(2N-1)(2N+1)/3`             |
| `fhat[l][m,n]` | `fhat[ su2_coeff_offset(l) + (m+l)*(2l+1) + (n+l) ]`             |

---

## 2. The two implementations

Both algorithms compute the *same discrete sum*.  The only difference is the
order in which the summations are performed.

### 2.1 Direct FT, `src/su2_ft.c` -- O(N^6)

Brute force.  For every coefficient `(l, m, n)`, walk the entire grid:

```c
for each (l, m, n):
    sum = 0
    for k:                                    /* O(N) */
        precompute P_k = conj(P^l_{n,m}(cos theta[k])) * sin(theta[k])
        for j1:                               /* O(N) */
            for j2:                           /* O(N) */
                sum += f[j1,k,j2] * P_k * exp(+i n phi[j1]) * exp(+i m psi[j2])
    fhat[l, m, n] = norm * sum
```

The triple inner loop is O(N^3) per coefficient and there are O(N^3) coefficients,
giving the O(N^6) headline (paper line 1322).  Small optimisations (precomputing
exponential and sine tables, factoring `sin theta_k` out of the inner pair) are
included; none of them changes the asymptotic.

### 2.2 Fast FFT, `src/su2_fft.c` -- O(N^4)

The novel part of the paper is to share work across coefficients.

**Stage 1 -- 2-D FFT over (phi, psi).** Define

```
F2[k, n, m] = sum_{j1, j2} f(g_{j1, k, j2}) *
              exp(+i n phi[j1]) * exp(+i m psi[j2])                   (line 1347)
```

The exponentials depend only on `(j1, n)` and `(j2, m)`, not on `l`, so the
sum can be computed once per `k` and reused across all `l`.  We use FFTW's
backward 2-D DFT for this.  The grid hits both `+pi` and `-pi`, so the N
samples carry only `N-1` independent Fourier modes: we fold the duplicate
endpoints (`g[0] = f[0] + f[N-1]`, etc.), apply a `(N-1) x (N-1)` FFTW plan,
and undo the `(-1)^{n+m}` half-shift on the way out.  See the file header
comment in `src/su2_fft.c` for the bit-by-bit derivation.

Stage-1 cost: one `(N-1)x(N-1)` FFT per `k`, so O(N^2 log N) per slice and
O(N^3 log N) total (paper line 1356).

**Stage 2 -- beta-stage inner product.** After Stage 1 the remaining work is

```
fhat(l)_{m,n} = norm * sum_k F2[k, n, m] *
                conj(P^l_{n,m}(cos theta_k)) * sin(theta_k)          (line 1361)
```

We evaluate this directly for each `(l, m, n)`: an O(N) inner product per
coefficient, O(N^4) total across O(N^3) coefficients.

This corresponds to the paper's `k = 1` choice of the divide-and-conquer
splitting parameter (paper Theorem TEO_FIN, line 1455), which the paper
proves to be the *optimum* -- finer splits introduce `2^k * k` log factors
that strictly worsen the count.

A more elaborate Stage 2 (recursive shifted-coefficient matrices
A_r^L, B_r^L from paper line 773; composition rule line 981) could be
substituted with no change in the answer or its asymptotic; it just
shuffles the same O(N^4) operations differently.

### 2.3 Wigner sweep via ascending-l three-term recurrence (m21)

Bead `su2fft-m21` replaced the per-(l, m, n, k) call to `wigner_d_phys`
with a recurrence-based sweep that reduces Stage 2 from O(N^5) to O(N^4),
matching the paper's headline complexity for the first time.

**Phase identity.**  The paper's P^l_{n,m} and Sakurai's real Wigner small-d
are related by `P^l_{n,m}(cos beta) = i^{m-n} * d^l_{n,m}(beta)` (see
Section 4 of this file).  Taking the conjugate:

```
conj(P^l_{n,m}(cos beta)) = i^{n-m} * d^l_{n,m}(beta)
```

Because d^l_{n,m} is real, the phase `i^{n-m}` is constant across all k
for fixed (m, n, l).  It can therefore be pulled outside the k-sum in Stage 2:

```
fhat(l)_{m,n} = norm * i^{n-m} * sum_k  d^l_{n,m}(theta_k)
                                         * F2[k, n, m] * sin(theta_k)
```

**Loop order.**  The outer loops run over (m, n); for each pair, a k-loop
accumulates the dot products `acc[l] += d^l_{n,m}(theta_k) * F2[k,n,m] * sin(theta_k)`
for all l simultaneously; then the phase and norm are applied once per (m, n, l).
Innermost is the l-sweep over the recurrence values.

**Recurrence.**  For fixed (m, n, theta_k), the values `d^l_{n,m}(theta_k)` for
ascending l satisfy the Jacobi-lifted three-term recurrence derived in
`notes/wigner_recurrence.md` §2b:

```
d^{l+1}_{n,m}(beta) = (A_k cos(beta) + B_k) * F1 * d^l_{n,m}(beta)
                     -  C_k * F2 * d^{l-1}_{n,m}(beta)
```

where `A_k`, `B_k`, `C_k` are the standard Jacobi polynomial recurrence
coefficients (DLMF 18.9.1) at index k = l - max(|m|, |n|), and F1, F2 are
normalisation ratios that carry the factorial weight between consecutive l.
See `notes/wigner_recurrence.md` §2b for the exact coefficient formulas and
the singularity analysis showing no division-by-zero occurs for l >= l_min + 1.
The ascending direction is the numerically stable direction for Jacobi
polynomials (DLMF §18.2(iii); Risbo 1996 §3).

**Seed.**  The recurrence is seeded with two `wigner_d_phys` calls at
`l = l_min` and `l = l_min + 1`, where `l_min = max(|m|, |n|)`.  At `l = l_min`
the de Moivre sum has exactly one term (tmin = tmax); at `l = l_min + 1` it has
at most two terms.  Both are O(1)-cost calls, not O(l).

The relevant loop from `src/su2_fft.c` Stage 2:

```c
for (int k = 0; k < N; ++k) {
    su2_wigner_d_seq(l_min, N - 1, n, m, theta[k], d_seq);
    double _Complex w = F2[k * stride_k + (n+N-1)*stride_n + (m+N-1)] * sin_th[k];
    for (int l = l_min; l < N; ++l)
        acc[l] += d_seq[l - l_min] * w;
}
```

`su2_wigner_d_seq` seeds with two `wigner_d_phys` calls and then runs the §2b
recurrence to fill the output array for l = l_min..N-1 in O(N - l_min) steps.

**Empirical scaling confirmation.**  The speedup table from `build/compare` post-m21:

```
   N |    direct(s) |      fast(s) |  speedup
-----+--------------+--------------+----------
  16 |     0.177889 |     0.004820 |    36.90
  20 |     0.655026 |     0.009028 |    72.55
  24 |     1.832225 |     0.020154 |    90.91
```

N=16 to N=24 is a 1.5x increase in N.  O(N^4) predicts (1.5)^4 = 5.06x; the
observed ratio is 0.00482 s to 0.0202 s = 4.18x.  Slightly sub-N^4 because
seed cost and FFTW overhead are non-negligible at these sizes.  Pre-m21 the
same interval was 5.78x, closer to O(N^5)'s prediction of 7.59x.

Cross-reference: `notes/wigner_recurrence.md` (full derivation and singularity
analysis); `PROFILING.md` (stage timings and hot-path breakdown).

---

## 3. Why this works

The reduction from O(N^6) to O(N^4) is one observation:

> For every `k`, the `(phi, psi)` part of every matrix coefficient
> `conj(t^l_{n,m}(g_{j1,k,j2}))` is exactly the (j1, j2) entry of the kernel
> `exp(+i n phi[j1]) * exp(+i m psi[j2])`, independent of `l`.

So the entire `(j1, j2)` double sum can be computed once per slice and recycled
across all `O(N)` values of `l` that share that `(m, n)` pair.  Without this
reuse, the same `(j1, j2)` sum is recomputed `O(N)` times -- the factor of `N`
that distinguishes O(N^6) from O(N^4).

The `theta` dimension still requires `O(N^3)` distinct coefficients, each with
an `O(N)` inner product against the precomputed `F2[*, n, m]`, giving the
remaining O(N^4) cost.

The paper's full machinery (shifted Legendre/Jacobi coefficients, composition
rule, divide-and-conquer over the `theta` integration) is needed to push the
count further, but only attains O(N^4 log N log log N) at best -- strictly
worse than the `k = 1` result already used here (paper line 1612).

---

## 4. Wigner small-d / paper's P^l_{n,m}

`src/su2_wigner.c` evaluates the paper's `P^l_{n,m}(cos theta)`.  A literal
implementation of the Rodrigues form (binomial expansion + Horner) was the
first attempt and is preserved in git history; it loses all double-precision
significance around `l ~ 10`.

The stable replacement combines two facts:

1. The standard Sakurai Wigner small-d satisfies the **de Moivre sum**

   ```
   d^l_{n,m}(beta) = sum_t (-1)^{n-m+t} *
                     sqrt((l+n)!(l-n)!(l+m)!(l-m)!) /
                     [(l+m-t)! * t! * (n-m+t)! * (l-n-t)!] *
                     cos(beta/2)^{2l+m-n-2t} *
                     sin(beta/2)^{n-m+2t}
   ```

   This has `O(l)` terms with bounded factorial ratios; stable to `l ~ 25`.

2. Substituting the Jacobi-Rodrigues identity into the paper formula and
   simplifying yields

   ```
   P^l_{n,m}(cos beta) = i^{m-n} * d^l_{n,m}(beta)
   ```

   (The `(-1)^{l-m}`, `2^{-l}`, `(l-n)!`, and `(-2)^{l-n}` factors all
   cancel under careful algebra; the result is just the `i^{m-n}` phase
   distinguishing paper's convention from Sakurai's.)

So `su2_wigner_d` evaluates the stable sum and multiplies by `pow_i(m - n)`.

Tests in `tests/test_wigner.c` lock down four invariants:

  1. `P^l_{0,0}(cos theta) = Legendre P_l(cos theta)` for `l = 0..3`.
  2. `P^l_{n,m}(1) = delta_{n,m}` (identity at theta = 0).
  3. column unitarity `sum_n |P^l_{n,m}(cos theta)|^2 = 1`.
  4. one spot value at `(l, n, m) = (1, 1, 0)`, theta = pi/2 (computed by
     hand from the paper formula).

---

## 5. TDD record

The test files were written before each corresponding implementation file.
The order in which red-then-green cycles were closed:

  1. `tests/test_grid.c` -> `src/su2_grid.c`
  2. `tests/test_wigner.c` -> `src/su2_wigner.c` (Rodrigues version, then
     rewritten to the de Moivre sum after the FFT cross-compare exposed
     catastrophic cancellation at N >= 14)
  3. `tests/test_ft_direct.c` -> `src/su2_ft.c`
  4. `tests/test_fft.c` -> `src/su2_fft.c`
  5. `bench/compare.c` -- end-to-end cross-validation across N.

The gold-standard test is `tests/test_fft.c::test_fft_matches_direct_random`:
both algorithms must produce identical `fhat` (to `1e-10`) on a random complex
input.  `bench/compare.c` extends this to larger N and reports timing.

---

## 6. Observed cross-comparison

A sample `make bench` run on this host (post-m21, Stage 2 uses the ascending-l
recurrence; see Section 2.3):

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

The maximum coefficient-wise difference is within the 1e-10 cross-check
tolerance across the full range, confirming the two algorithms realise the same
discrete sum.  At N=24 the max-diff is 2.21e-13, higher than the pre-recurrence
figure of 7.12e-17 because the ascending-l recurrence accumulates floating-point
error over l steps rather than recomputing each d^l independently; 2.21e-13 is
still four orders of magnitude inside the tolerance.  The speedup at N=24 is
90.91x; at small N the FFTW plan overhead dominates and the direct path wins
(the standard FFT crossover).

---

## 7. Library use

Per `notes/library_evaluation.md`:

  - **FFTW3 3.3.10**.  Powers the 2-D backward DFT in Stage 1 of the
    double-precision fast algorithm.  Linked via `-lfftw3`.
  - **FLINT 3.0.1**.  Powers the arb/acb parallel path:
      - `acb_dft_prod` -- Stage-1 2-D DFT in arb arithmetic.
      - `arb_fac_ui`, `arb_pow_ui`, `arb_cos`, `arb_sin`, `arb_sqrt` --
        Wigner d via the de Moivre sum at arbitrary precision.
      - `acb_add`, `acb_mul`, `acb_conj`, `acb_mul_arb` -- everything else.

The double and arb paths share the same algorithmic structure
(`src/su2_fft.c` vs `src/su2_fft_arb.c`).  Both stages and the closed-grid
endpoint fold are mirrored.

GSL / MPFR / SHTns are not used: every routine they would provide is also in
FLINT, and bringing in extra runtime dependencies bought nothing for this
project.

### 7.1 Observed arb-precision behaviour

`make bench` also runs `bench/arb_bench`, which times the arb FFT and reports
the worst-case ball radius across all output coefficients:

```
Arb-precision FFT at N=6
  prec |    time(s) |     max ball
-------+------------+-------------
    53 |      0.008 |     1.44e-15
   128 |      0.009 |     3.78e-38
   256 |      0.015 |     1.64e-76
   512 |      0.023 |    9.61e-154
```

At prec=53 the arb FFT matches the double FFT to ~1e-15 (within accumulated
double-rounding tolerance, verified by `tests/test_arb.c::test_arb_vs_double`).
At higher precisions the ball radius decreases roughly as 2^{-prec}, giving
certified precision suitable for studies of high-bandlimit / ill-conditioned
inputs where the double path would lose significance.

The overhead from prec=53 to prec=512 is only ~3x because the loop is
dominated by O(N^4) acb multiplications whose cost grows nearly linearly
with the limb count for limb counts the underlying GMP routines handle in
one or two FFT levels.

---

## 8. Departures from the paper

1. **Closed-grid FFT.**  The paper uses `phi[j1] = -pi + j1 * 2pi/(N-1)` which
   places samples at both `-pi` and `+pi`.  Standard FFTs assume an "open"
   periodic grid of spacing `2pi/N`.  We honour the paper's grid and fold the
   duplicated endpoints into a length-`(N-1)` periodic array before calling
   FFTW; the post-FFTW `(-1)^{n+m}` factor restores the `e^{-i n pi}` shift.
   This is bookkeeping, not an algorithmic change.

2. **Stage 2 chosen at `k=1` directly.**  The paper's Stage-2 pseudocode walks
   a `log r`-deep divide-and-conquer tree over shifted-coefficient matrices
   `(A_r^L, B_r^L)` (paper line 773 onwards).  We collapse this to direct
   inner products, which is the `k=1` choice the paper's own optimisation
   identifies as the cost-minimal one (line 1484).  Same asymptotic, simpler
   code, identical numerics.

3. **Integer-l only.**  The paper covers `l in (1/2) N_0` (including half-
   integer).  We restrict to integer `l`.  Half-integer support would require
   threading `l - n, l - m` consistency through every factorial; the
   algorithmic structure is unchanged.
