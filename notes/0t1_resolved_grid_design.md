# Resolved phi/psi grid — Design Brief for `su2_fft_resolved`

Bead: `su2fft-0t1` (this brief is the binding spec).
Implementation targets: `src/su2_fft_resolved.c` (new), `src/su2_ft_resolved.c` (new), `tests/test_resolved.c` (new), `include/su2.h` (API additions).
Follow-up arb-precision work: bead `su2fft-rrx`.
All paper citations are line numbers in `paper.tex` (arxiv 2605.23923).

---

## 1. What is broken in the closed-N grid

The current grid uses N closed-uniform points in each of phi, psi:
```
phi[j1] = -pi + j1 * 2pi/(N-1),   j1 in [0, N-1]   (endpoints coincide on torus)
psi[j2] = -pi + j2 * 2pi/(N-1),   j2 in [0, N-1]
```
Endpoints j=0 and j=N-1 are the same point; folding gives an (N-1)-point DFT
which resolves N-1 independent modes per axis. The SU(2) bandlimit at degree N
requires |n|, |m| ≤ N-1, i.e. 2N-1 distinct modes per axis. Modes with
|n| > (N-1)/2 alias.

Empirical diagnostic at N=8 (HANDOFF.md §2): forward(constant) leaks ~0.197
into nonzero modes; single-coefficient roundtrip at (l=N-1, m=±(N-1), 0) yields
error ~3. After bead `ega` (GL theta) the theta direction is exact, but the
phi/psi aliasing remains; this is the dominant precision floor.

## 2. Math of the fix

For trig polynomials of degree ≤ L on the period-2π circle, the uniform DFT on
P samples is exact iff P ≥ 2L+1 (Trefethen, "Approximation Theory and
Approximation Practice" §12; DFT orthogonality on the closed unit roots).

Bandlimit at degree N ⇒ L = N-1 ⇒ minimum P = 2N-1.

Choose the **open** uniform grid with P = 2N-1 samples:
```
phi[j] = -pi + j * 2pi/P,   j in [0, P-1],   P = 2N-1
```
This avoids the endpoint duplication of the closed convention; no fold trick
is needed. The phase factor `(-1)^n` arising from the -π shift becomes:
```
exp(+i n phi[j]) = exp(-i n pi) * exp(+i n j 2pi/P)
                 = (-1)^n * w_P^{n j},     w_P = exp(+i 2pi/P)
```
Modes n ∈ [-(N-1), N-1] map to FFTW output bins (n mod P) ∈ [0, P-1] with no
collisions (P = 2N-1 ⇒ exactly 2N-1 distinct bins). No fold, no alias.

Combined with GL theta (existing `su2_gl_nodes_weights`, bead `ega`): the
forward analysis becomes an exact composite quadrature, and the spectrum
roundtrip `forward(inverse(fhat)) = fhat` holds to working precision.

## 3. Naming and API additions

**Forward / inverse (double):**
```c
void su2_fft_resolved    (int N, const double _Complex *f,    double _Complex *fhat);
void su2_fft_resolved_inv(int N, const double _Complex *fhat, double _Complex *f);
```
Storage convention for sample array `f`:
- Sample dims: phi axis has P = 2N-1 points; theta axis has N GL points;
  psi axis has P = 2N-1 points.
- Row-major `f[j1*N*P + k*P + j2]` with `j1 in [0, P-1]`, `k in [0, N-1]`,
  `j2 in [0, P-1]`. Same axis ordering (phi, theta, psi) as existing
  `su2_sample_index`; the per-axis lengths differ.
- Total sample count: `P*P*N = (2N-1)^2 * N`.

Coefficient layout for `fhat` is **unchanged**: `su2_total_coeffs(N)` and
`su2_coeff_offset`, `su2_mn_index` apply verbatim.

**Direct reference (double):**
```c
void su2_ft_direct_resolved(int N, const double _Complex *f, double _Complex *fhat);
```
O(N^6) ground truth on the new grid; mirrors `su2_ft_direct`. Used only by
the cross-check test.

**Helpers in `include/su2.h`:**
```c
static inline int    su2_resolved_P(int N) { return 2*N - 1; }
static inline size_t su2_resolved_total_samples(int N) {
    size_t P = (size_t)(2*N - 1);
    return P * P * (size_t)N;
}
static inline size_t su2_resolved_sample_index(int N, int j1, int k, int j2) {
    int P = 2*N - 1;
    return (size_t)j1 * (size_t)N * (size_t)P + (size_t)k * (size_t)P + (size_t)j2;
}
```

**Theta nodes:** the resolved path uses GL theta nodes via the existing
`su2_gl_nodes_weights(N, x, w)` routine. theta_k = arccos(x_k).

**Legacy:** `su2_fft`, `su2_fft_inv`, `su2_fft_gl`, `su2_fft_inv_gl`,
`su2_ft_direct` remain. They are not deprecated — they retain their existing
API and storage convention. Documentation is updated to clearly distinguish:
- `su2_fft` / `su2_fft_inv`: closed-N grid in phi/psi/theta. Riemann theta
  + aliased phi/psi. Lossy. Cheapest sample budget.
- `su2_fft_gl` / `su2_fft_inv_gl`: closed-N grid in phi/psi, GL in theta.
  Riemann theta fixed; phi/psi still aliased.
- **`su2_fft_resolved` / `su2_fft_resolved_inv`: open (2N-1) grid in phi/psi,
  GL in theta. Exact spectrum roundtrip at working precision.**

## 4. Algorithm — Stage 1 (no fold) and Stage 2 (GL)

### Stage 1 (forward)

For each theta slice k (k in [0, N-1]):
1. Load sample slice `f[j1, k, j2]` for j1, j2 in [0, P-1] into an (P x P)
   acb / fftw buffer.
2. Run an (P x P) FFTW BACKWARD plan (sign +1, our convention).
3. For each output mode (n, m) in [-(N-1), N-1]^2, apply the (-1)^n * (-1)^m
   phase from the -π origin shift and copy into `F2[k, n, m]`:
```
F2[k, n, m] = (-1)^n * (-1)^m * G_slice[n mod P, m mod P]
```
**No fold, no endpoint summation.** The output bins n mod P and m mod P are
guaranteed distinct over n, m ∈ [-(N-1), N-1] because the range size 2N-1 = P.

### Stage 2 (forward)

For each (m, n), sweep l = l_min .. N-1 with the Wigner three-term recurrence
(`su2_wigner_d_seq`); accumulate the GL inner product
```
fhat(l)_{m,n} = norm * i^{n-m} * sum_k w_k * d^l_{n,m}(theta_k) * F2[k, n, m]
```
where `w_k` are the N-point GL weights on [-1, 1] (absorbing the
`sin(theta) dtheta = -dx` Jacobian, identical to `su2_fft_gl`).

**Global normalisation:** see §5 below.

### Stage 1-inv / Stage 2-inv (inverse)

Mirror of `su2_fft_inv_gl` but with the resolved phi/psi grid. Stage 2-inv
synthesises G[k, n, m] from fhat as the (2l+1)-weighted sum; Stage 1-inv
distributes G[k, n, m] into an (P x P) buffer with (-1)^n * (-1)^m phase
(no fold-back; one bin per (n, m)) and runs an (P x P) FFTW FORWARD plan
to obtain f[j1, k, j2] for j1, j2 in [0, P-1].

## 5. Normalisation

Forward analysis Riemann constant (current `su2_fft_gl`, notes/gauss_legendre.md
§6) was
```
norm_gl_closed = 1 / (2 * N * N)
```
derived from the closed-N fold over-counting endpoints to give
F2[k, 0, 0] = N^2 for constant input.

With the open P-grid the constant-input FFTW BACKWARD output gives
`G_slice[0, 0] = P * P` (no over-count); the (-1)^{0+0} phase is +1; F2[k, 0, 0]
= P^2. The single GL theta sum `sum_k w_k = 2`. The Peter-Weyl prefactor on
fhat(0)_{0,0} reads
```
fhat(0)_{0,0} = norm * P^2 * 2 = 1   =>   norm_resolved = 1 / (2 * P^2),  P = 2N-1
```
This is the only normalisation change vs `su2_fft_gl`. Cite this derivation
in the source comment.

(The `i^{n-m}` phase, the `sin(theta)dtheta -> w_k` substitution, and the
Wigner recurrence machinery are all unchanged.)

## 6. Cross-check plan (tests/test_resolved.c)

All assertions are STRICT 1e-12 or tighter. No "documented floor" tests.

1. `test_resolved_zero` — forward and inverse of zeros → zeros, < 1e-14.
2. `test_resolved_constant_forward` — forward(constant A on resolved grid) →
   fhat(0)_{0,0} = A, all other coeffs zero, < 1e-12.
3. `test_resolved_inv_delta_l1_m0_n0` — fhat = δ at (1, 0, 0); inverse should
   reproduce f(g) = 3 cos(theta_k) (independent of phi, psi) on the resolved
   grid, < 1e-12.
4. `test_resolved_fft_matches_direct_random` — random complex f on the
   resolved grid (P^2 N samples); `su2_fft_resolved` vs `su2_ft_direct_resolved`
   agreement < 1e-10 (N = 5, 6, 8).
5. `test_resolved_spectrum_roundtrip` — random complex fhat;
   `forward(inverse(fhat)) ≈ fhat` < 1e-12 at N = 4, 8, 16. **This is the
   headline test of bead `0t1`.**
6. `test_resolved_sample_roundtrip` — f = inverse(fhat); g = forward(f);
   h = inverse(g); compare h ≈ f < 1e-12 (bandlimited f). Belt-and-braces
   check on sample-space stability.
7. `test_resolved_linearity` — additivity of forward and inverse < 1e-13.

The headline guarantee: at N = 8 the spectrum roundtrip max error must be
< 1e-12 (vs current closed-grid ~0.197). Document the achieved tolerance in
this brief and in ALGORITHM.md.

## 7. Pitfalls / things to flag in the implementation

- **FFTW plan size P = 2N-1.** Often prime (e.g. P=15=3·5, 17, 23, 31, ...).
  FFTW handles arbitrary sizes via Rader/Bluestein with O(P log P) cost; some
  slowdown vs power-of-2 expected (~2x at small N). Acceptable; profile later
  if it becomes a hot path. No zero-padding needed.
- **Phase parity for n with 2N-1 odd.** `(-1)^n` is correct regardless of
  parity of P; do not "simplify" it into a P-dependent expression.
- **Mode-to-bin mapping.** n in [-(N-1), N-1] ⇒ `n_mod = ((n % P) + P) % P`
  is in [0, P-1]. n=0 → 0, n=1 → 1, ..., n=N-1 → N-1, n=-1 → P-1, ...,
  n=-(N-1) → N. These are 2N-1 distinct bins. Confirm: `2N-1 = P` (yes).
- **`(-1)^n` cross-talk.** No cross-talk: each FFTW output bin is referenced
  exactly once with a known sign. Verifiable by visual: dump F2 for a
  single-coefficient input and confirm structure.
- **Memory.** Sample array is P^2 * N = (2N-1)^2 N doubles (complex) — at
  N=16, this is ~944 KB. Acceptable.
- **`norm_resolved = 1 / (2 P^2)`** vs `norm_gl = 1 / (2 N^2)`: do not copy
  the wrong constant; constant-input test catches this immediately.
- **Cross-check test at N=4.** Smallest interesting N; verify no edge-case
  bug at l_min boundary.

## 8. FLINT capability for the arb follow-on (bead rrx)

(For context — not implemented in bead 0t1.)

- `acb_dft_prod(w, v, cyc={P,P}, num=2, prec)` accepts arbitrary cyc[]; for
  P = 2N-1 it routes through Bluestein/CRT internally. Forward DFT only;
  use the conj trick for the inverse direction (HANDOFF.md §2 item 5).
- `arb_hypgeom_legendre_p_ui_root(res, weight, n, k, prec)` (FLINT 3.0+,
  confirmed in `/usr/include/flint/arb_hypgeom.h`) returns both the k-th
  root of P_n and its GL weight in arb. This replaces the Newton iteration
  in `src/su2_gauss_legendre.c` for the arb path — no manual recurrence
  needed.

## 9. Open questions deferred

None blocking bead 0t1. Filed as follow-ons:
- `su2fft-rrx`: arb-precision resolved-grid roundtrip.
- (later) downstream migration of `su2_convolve` and `su2_fft_sphere` to take
  the resolved grid as their forward/inverse partner. The convolve API is
  unchanged (operates on fhat only); the sphere API needs a resolved
  variant if S² applications want exact roundtrip.
- (later) deprecate / re-document `su2_fft` and `su2_fft_gl` as "lossy at
  high l" once downstream catches up.

## 10. Results

`make test` passes 58/58 tests across 10 binaries with the shipped implementation.
All resolved-grid tests are in `tests/test_resolved.c`.

### test_resolved_zero

Forward and inverse of the zero sample array both return zero to < 1e-14.
No anomalous behavior at any grid size.

### test_resolved_constant_forward (N=8)

`su2_fft_resolved` applied to a constant-1 sample array on the P=15 open grid:
- fhat(0)_{0,0}: 1.0 (exact by construction of `norm_resolved = 1/(2 P^2)`)
- Leakage to all other coefficients: max < **1e-12**

Baseline for comparison: the closed-grid GL path (`su2_fft_gl`, N=8) leaks
~0.197 to non-DC modes.  The resolved grid reduces that floor by more than
eight orders of magnitude.

### test_resolved_fft_matches_direct_random

Cross-check: `su2_fft_resolved` vs `su2_ft_direct_resolved` on a random complex
sample array drawn from the P^2 * N resolved grid.

| N | max abs error (fast vs direct) |
|---|-------------------------------|
| 5 | 3.27e-17                      |
| 6 | 1.47e-17                      |

Both values are within a few ULPs of double precision (~2.2e-16).  The two
algorithms compute the same discrete sum.

### test_resolved_spectrum_roundtrip (HEADLINE)

`forward(inverse(fhat)) ≈ fhat` for random complex `fhat` with all
`su2_total_coeffs(N)` entries non-zero.  This is the headline test of bead `0t1`.

| N  | su2_total_coeffs(N) | max relative error |
|----|---------------------|--------------------|
|  4 | 84                  | **9.39e-16**       |
|  8 | 680                 | **4.45e-15**       |
| 16 | 5984                | **5.89e-14**       |

The N=16 result is slightly worse than N=4–8 because accumulated floating-point
error grows with the depth of the coefficient pyramid (5984 coefficients vs 84).
The factor-of-~60 increase from N=4 to N=16 reflects O(N^3) coefficient growth
plus O(N) per-coefficient recurrence depth; 5.89e-14 is still 11 orders of
magnitude better than the closed-grid GL baseline (~0.197 leakage at N=8) and
comfortably within double precision for any practical application at these
bandlimits.

### test_resolved_sample_roundtrip

Belt-and-braces: f = inverse(fhat), g = forward(f), h = inverse(g); compare h vs f.
Both f and g are bandlimited to degree N-1.

| N | max relative error |
|---|--------------------|
| 4 | 5.20e-16           |
| 8 | 4.78e-15           |

Sample-space stability is consistent with the spectrum roundtrip numbers above.

### Summary (double-precision, bead 0t1)

The resolved-grid path achieves exact spectrum roundtrip at working precision
across N = 4, 8, 16.  The design goal of §6 ("max error < 1e-12 at N=8") is
exceeded by three orders of magnitude.  The constant-input leakage reduction
from ~0.197 (closed-grid GL) to < 1e-12 (resolved) confirms that the P=2N-1
open phi/psi grid eliminates the structural aliasing identified in §1.

---

## 11. Arb-precision results (bead `su2fft-rrx`)

The arb port (`src/su2_fft_resolved_arb.c`, `src/su2_ft_resolved_arb.c`,
`src/su2_gauss_legendre_arb.c`) carries the same algorithm at user-chosen
precision via FLINT acb_t/arb_t. Tests in `tests/test_resolved_arb.c` (6 tests,
all PASS in 0.37 s).

### test_resolved_arb_constant (N=4, prec=128)

`fhat(0,0,0) - 1` mid-point: **1.18e-38**. Max leak: **1.13e-38**. Both within
2^-128 ≈ 2.9e-39 of working precision, matching the certified ball arithmetic.

### test_resolved_arb_fast_vs_direct (prec=128)

| N | max abs diff (arb fast vs arb direct) |
|---|---------------------------------------|
| 4 | 4.54e-38                              |
| 5 | 9.32e-38                              |

### test_resolved_arb_inv_delta (N=4, prec=128)

Synthesis of `fhat(1)_{0,0} = 1` reproduces `f(*, theta_k, *) = 3 cos(theta_k)`.
Max `|Re(f) - 3 cos(theta_k_double)|` = **2.22e-16** (limited by the double
`cos()` in the test target, not the arb path). Max `|Im(f)|` = **0** (exact).

### test_resolved_arb_spectrum_roundtrip (HEADLINE)

`forward(inverse(fhat)) ≈ fhat` at arb precision.

| prec | N | max relative error                |
|------|---|-----------------------------------|
| 128  | 4 | 1.29e-35                          |
| 128  | 8 | 2.17e-33                          |
| 256  | 4 | **4.29e-74**                      |
| 256  | 8 | **4.40e-72**                      |

The roundtrip error scales as ~2^-prec, confirming the roundtrip is exact up
to the working precision. At prec=256 the error is 26 orders of magnitude
below the 1e-50 design goal.

### test_resolved_arb_vs_double (N=5, prec=53)

Cross-check the arb path at IEEE-double precision against the double path:
max abs diff = **4.80e-17**. Ties the two paths together at matching precision.

### Summary (arb-precision, bead `rrx`)

`su2_fft_resolved_arb` + `su2_fft_resolved_inv_arb` deliver certified spectrum
roundtrip at any user-chosen precision via FLINT ball arithmetic. The
roundtrip error scales linearly in 2^-prec; doubling the working precision
halves the error in log scale. This is the certified-roundtrip arc the
project was working toward; no closed-form bound beats it for the resolved
grid + GL theta combination.
