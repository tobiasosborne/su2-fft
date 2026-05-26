# Inverse FFT on SU(2) — Design Brief for `su2_fft_inv`

Bead: `su2fft-3lx`.  Implementation target: `src/su2_fft_inv.c`.
All paper citations are line numbers in `paper.tex` (arxiv 2605.23923).

---

## 1. Mathematical statement

Peter-Weyl synthesis (paper.tex line 554):

```
f(g) = sum_{l=0}^{N-1} (2l+1) * sum_{m,n=-l}^{l} fhat(l)_{m,n} * t^l_{n,m}(g)
```

- `(2l+1)` is the Plancherel dimension weight; it appears because
  `<t^l_{nm}, t^l_{nm}> = 1/(2l+1)` (paper line 564).
- Matrix coefficient (paper line 534):
  `t^l_{n,m}(phi, theta, psi) = exp(-i(n*phi + m*psi)) * P^l_{n,m}(cos theta)`.
- `P^l_{n,m} = i^{m-n} * d^l_{n,m}` (ALGORITHM.md §4; HANDOFF.md §2 item 2).
- Truncation to `l = 0..N-1` is exact for bandlimited f.

External references if paper not available: Ruzhansky & Turunen,
"Pseudo-Differential Operators and Symmetries", p. 629 (Peter-Weyl reconstruction);
Vilenkin, "Special Functions and Theory of Group Representations" §III.6.

---

## 2. Roundtrip identity — which direction is exact?

### Forward discrete FT (paper lines 693, 1316, 1342)

The forward analysis is a Riemann sum — NOT Gauss-Legendre quadrature.
Grid: `theta_k = k*pi/(N-1)`, uniform spacing `dtheta = pi/(N-1)`.
The phi and psi integrals are trapezoidal on a closed periodic grid of N points
with N-1 independent modes; the trapezoidal rule is EXACT for trigonometric
polynomials of degree <= N-1 (Trefethen, "Approximation Theory and Approximation
Practice" §12; DFT orthogonality is algebraic on the closed grid).

The theta integral is `sum_k sin(theta_k) * dtheta`, a Riemann sum over [0,pi].
This is NOT exact for Jacobi / Wigner-d polynomials; it has systematic error.

### The (N/(N-1))^2 Riemann error (HANDOFF.md §2 item 7; CLAUDE.md invariant 6)

The l=0 case is the clearest diagnostic.  For fhat(0)_{0,0} = 1, synthesis gives
the constant function f = 1.  Forward analysis returns:

```
fhat'(0)_{0,0} = norm * sum_k sin(theta_k)  (where norm = dphi*dtheta*dpsi / 8pi^2)
```

The trapezoidal sum `sum_{k=0}^{N-1} sin(k*pi/(N-1)) * pi/(N-1)` over a closed
uniform grid on [0,pi] equals `pi/(N-1) * (sin(0)/2 + sum_{k=1}^{N-2} sin(k*pi/(N-1)) + sin(pi)/2)`.
The interior sum evaluates to `(N-1) * (1 - cos(pi*(N-1)/(N-1))) / (2*sin(pi/(2*(N-1)))) * ...`.
In practice, the closed-grid result differs from the true integral (= 2) by a factor
`(N/(N-1))^2`; this is documented and tested in `test_ft_direct.c::test_ft_direct_constant`.

**Consequence for spectrum roundtrip:**
`forward(inverse(fhat)) == fhat * (N/(N-1))^2`  at the (l=0, m=0, n=0) entry —
i.e., the spectrum roundtrip is NOT 1.0 at machine precision.  The error is
`O((N-1)^{-2})` relative, dominated by the closed-grid endpoint quadrature error.

At N=8:  `(8/7)^2 - 1 ≈ 0.31` — a 31% error, not a floating-point rounding artifact.
At N=24: `(24/23)^2 - 1 ≈ 0.089` — still ~9%.

The phi and psi roundtrip IS exact for trig polys of degree < N-1 because the
closed-grid DFT folds endpoints exactly and FFTW backward/forward are exact inverses.

### Achievable tolerances

- **Spectrum roundtrip** `forward(inverse(fhat)) ≈ fhat`:
  NOT 1e-12 without correction.  Achievable relative error scales as O(1/N^2).
  Three paths to 1e-12 (see §7).

- **Sample roundtrip** `inverse(forward(f)) ≈ f` for bandlimited f:
  Also limited by the same Riemann theta error; it is NOT 1e-12 for f synthesised
  from arbitrary fhat, for the same reason in reverse.

**Recommendation:** document the achievable tolerance honestly in tests.
Use `(N/(N-1))^2` compensation factor in test assertions, OR adopt Gauss-Legendre
theta nodes (bead `su2fft-ega`).  Do NOT silently claim 1e-12 in the test plan.

---

## 3. Algorithm — structural symmetry with forward FFT

Forward FFT (src/su2_fft.c):
- Stage 1 (FFTW BACKWARD): fold closed-grid endpoints; 2D backward DFT over
  `(j1,j2)` -> `(n,m)` per theta-slice; apply `(-1)^{n+m}` half-shift correction.
- Stage 2 (Wigner sweep): accumulate `fhat(l)_{m,n} = norm * i^{n-m} * sum_k d^l_{n,m}(theta_k) * F2[k,n,m] * sin(theta_k)`.

**Inverse is the structural reverse.**

### Stage 2-inv: Wigner synthesis sweep

For each `(m, n)`, compute:

```
G[k, n, m] = i^{m-n} * sum_{l=l_min}^{N-1} (2l+1) * fhat(l)_{m,n} * d^l_{n,m}(theta_k)
```

Derivation: synthesis term is `(2l+1) * fhat * t^l_{n,m} = (2l+1) * fhat * P^l_{n,m} * exp(...)`.
With `P^l_{n,m} = i^{m-n} * d^l_{n,m}`, the phase `i^{m-n}` is constant in l and k,
so it factors out of the l-sum.  `d^l_{n,m}` is real; the l-sum is a dot product
with real Wigner-d values, accumulating a complex scalar (because fhat is complex).

Note: forward uses `conj(P^l_{n,m}) = i^{n-m} * d` (negative phase); inverse uses
`P^l_{n,m} = i^{m-n} * d` (positive phase).  Confirm: `r = ((m-n) % 4 + 4) % 4`
in the inverse vs `r = ((n-m) % 4 + 4) % 4` in the forward.

### Stage 1-inv: 2D FFTW FORWARD per theta slice

From the synthesis formula, for fixed k:

```
f(phi_{j1}, theta_k, psi_{j2})
  = sum_{n,m} G[k,n,m] * exp(-i*n*phi_{j1}) * exp(-i*m*psi_{j2})
```

This is a 2D sum with negative-exponent exponentials — a FFTW FORWARD (sign = -1) DFT
over `(n,m)` -> `(j1,j2)`.

Closed-grid derivation (ALGORITHM.md §2.2):
```
exp(-i*n*phi_{j1}) = exp(-i*n*(-pi + j1*2pi/(N-1)))
                   = exp(i*n*pi) * exp(-i*n*j1*2pi/(N-1))
                   = (-1)^n * exp(-i*n*j1*2pi/(N-1))
```
The `(-1)^n` factor is `sn = (n & 1) ? -1.0 : 1.0`; same as in the forward, but
now applied to accumulate into the G-slice (not read from the FFT output).

Fold: G[k, n, m] for n, m in [-(N-1), N-1] must be folded onto an (M x M)
array before the FFTW call, where M = N-1.  The fold rule is the inverse of
the unfold in Stage 1 of the forward:

```
G_slice[n mod M, m mod M] += (-1)^n * (-1)^m * G[k, n, m]
```

Duplicates (n differing by M, or m differing by M) alias onto the same bin and
are summed.  After FFTW FORWARD, `f_slice[j1, j2]` is the result; copy to
`f[j1, k, j2]` for j1, j2 in [0, M-1] and set `f[N-1, k, j2] = f[0, k, j2]`,
`f[j1, k, N-1] = f[j1, k, 0]` (closed-grid symmetry of a synthesised function).

**FFTW direction: FFTW_FORWARD** (negative exponent convention).
Cite: ALGORITHM.md §2.2 for the closed-grid handling.

---

## 4. C function signature

```c
/* Inverse of su2_fft: Peter-Weyl synthesis from fhat to samples f.
 *
 * Cost: O(N^4) -- mirrors the forward fast FFT structure.
 * Stage 2-inv: Wigner sweep producing G[k,n,m] from fhat.
 * Stage 1-inv: 2D FFTW_FORWARD (N-1)x(N-1) per theta slice with fold.
 *
 * paper.tex line 554: f(g) = sum_l (2l+1) sum_{m,n} fhat(l)_{mn} t^l_{nm}(g).
 * The (2l+1) Plancherel weight is in the synthesis (not in the forward analysis).
 * No 1/(8pi^2) factor in the inverse; that factor lives entirely in su2_fft.
 *
 * Normalisation note: forward(inverse(fhat)) == fhat only up to O((1/(N-1))^2)
 * Riemann quadrature error in the theta dimension (HANDOFF.md §2 item 7).
 * For exact roundtrip use Gauss-Legendre theta nodes (bead su2fft-ega).
 *
 * @param[in]  N     Bandlimit; grid N x N x N; fhat has su2_total_coeffs(N) entries.
 * @param[in]  fhat  Length-su2_total_coeffs(N) complex coefficient array.
 * @param[out] f     Length-N^3 complex sample array, row-major (j1, k, j2).
 */
void su2_fft_inv(int N,
                 const double _Complex *fhat,
                 double _Complex *f);
```

---

## 5. Implementation sketch

```c
void su2_fft_inv(int N, const double _Complex *fhat, double _Complex *f) {
    if (N < 2 || !fhat || !f) return;

    double *theta = su2_grid_theta(N);
    const int M = N - 1;

    /* ---- Allocate G[k, n_index, m_index]: n,m in [-(N-1), N-1] ---- */
    const int nrange = 2*N - 1;
    double _Complex *G = calloc((size_t)N * (size_t)nrange * (size_t)nrange,
                                 sizeof(double _Complex));
    double *d_seq = malloc((size_t)N * sizeof(double));

    /* ---- Stage 2-inv: fhat -> G ---- */
    /* paper.tex line 554: synthesis weight (2l+1), phase i^{m-n}. */
    for (int m = -(N-1); m <= N-1; ++m) {
        for (int n = -(N-1); n <= N-1; ++n) {
            int l_min = (abs(m) > abs(n)) ? abs(m) : abs(n);
            if (l_min > N-1) continue;
            /* phase = i^{m-n}, positive; contrast forward which uses i^{n-m} */
            int r = ((m - n) % 4 + 4) % 4;
            double _Complex phase;
            switch (r) {
                case 0:  phase =  1.0 + 0.0*I; break;
                case 1:  phase =  0.0 + 1.0*I; break;
                case 2:  phase = -1.0 + 0.0*I; break;
                default: phase =  0.0 - 1.0*I; break;
            }
            for (int k = 0; k < N; ++k) {
                su2_wigner_d_seq(l_min, N-1, n, m, theta[k], d_seq);
                double _Complex acc = 0.0;
                for (int l = l_min; l < N; ++l) {
                    double wt = (double)(2*l + 1);
                    acc += wt * fhat[su2_coeff_offset(l) + su2_mn_index(l, m, n)]
                              * d_seq[l - l_min];
                }
                G[(size_t)k*(size_t)nrange*(size_t)nrange
                  + (size_t)(n + N-1)*(size_t)nrange
                  + (size_t)(m + N-1)] = phase * acc;
            }
        }
    }

    /* ---- Stage 1-inv: G -> f via FFTW_FORWARD per theta slice ---- */
    fftw_complex *g_slice = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_complex *f_slice = fftw_malloc(sizeof(fftw_complex) * (size_t)M * (size_t)M);
    fftw_plan plan = fftw_plan_dft_2d(M, M, g_slice, f_slice,
                                       FFTW_FORWARD, FFTW_ESTIMATE);

    memset(f, 0, sizeof(double _Complex) * (size_t)N * (size_t)N * (size_t)N);

    for (int k = 0; k < N; ++k) {
        memset(g_slice, 0, sizeof(fftw_complex) * (size_t)M * (size_t)M);
        for (int n = -(N-1); n <= N-1; ++n) {
            int n_mod = ((n % M) + M) % M;
            double sn = (n & 1) ? -1.0 : 1.0;  /* (-1)^n */
            for (int m = -(N-1); m <= N-1; ++m) {
                int m_mod = ((m % M) + M) % M;
                double sm = (m & 1) ? -1.0 : 1.0;
                double _Complex val = G[(size_t)k*(size_t)nrange*(size_t)nrange
                                        + (size_t)(n+N-1)*(size_t)nrange
                                        + (size_t)(m+N-1)];
                g_slice[(size_t)n_mod*(size_t)M + (size_t)m_mod] +=
                    (fftw_complex){ creal(sn * sm * val), cimag(sn * sm * val) };
            }
        }
        fftw_execute(plan);
        /* Copy M x M output; then fill closed-grid edge j=N-1 <- j=0. */
        for (int j1 = 0; j1 < M; ++j1) {
            for (int j2 = 0; j2 < M; ++j2) {
                fftw_complex v = f_slice[(size_t)j1*(size_t)M + (size_t)j2];
                f[su2_sample_index(N, j1, k, j2)] = v[0] + I*v[1];
            }
            f[su2_sample_index(N, j1, k, N-1)] = f[su2_sample_index(N, j1, k, 0)];
        }
        for (int j2 = 0; j2 < N; ++j2)
            f[su2_sample_index(N, N-1, k, j2)] = f[su2_sample_index(N, 0, k, j2)];
    }

    fftw_destroy_plan(plan);
    fftw_free(g_slice);
    fftw_free(f_slice);
    free(d_seq);
    free(G);
    free(theta);
}
```

---

## 6. Test plan — `tests/test_roundtrip.c`

All tests should mirror the structure of `tests/test_fft.c`.

### 6.1 Synthesis of a single coefficient

- fhat = zeros; set `fhat(1)_{0,0} = 1.0`.
- Synthesise: `su2_fft_inv(N, fhat, f)`.
- Expected: `f(phi, theta, psi) = (2*1+1) * P^1_{0,0}(cos theta) = 3 * cos(theta)`.
  (`P^1_{0,0} = i^{0-0} * d^1_{0,0} = Legendre P_1(cos theta) = cos(theta)`.)
- Compare pointwise on grid; tolerance 1e-12.
- Run at N = 4, 8.

### 6.2 Linearity

- `su2_fft_inv(alpha*fhat1 + beta*fhat2) == alpha*su2_fft_inv(fhat1) + beta*su2_fft_inv(fhat2)`.
- Tolerance 1e-13 (only floating-point rounding).

### 6.3 Zero input -> zero output

- `su2_fft_inv(zeros) == zeros`.

### 6.4 Spectrum roundtrip — HONEST tolerance

`fhat' = su2_fft(su2_fft_inv(fhat))`.  Expected error is NOT 1e-12.

The closed-grid Riemann sum in the forward analysis introduces systematic error
proportional to `(N/(N-1))^2 - 1 = O(1/N^2)`.  The test MUST use the corrected
target `fhat * (N/(N-1))^2` or a per-entry relative tolerance of `O(1/N^2)`.

Alternatively: assert `||fhat' - fhat * (N/(N-1))^2|| / ||fhat|| < 1e-10`
(only floating-point noise on top of the known Riemann correction).

At N = 4, 8, 16.  Document the achieved value, e.g., "spectrum roundtrip error
at N=8: 2.3e-14 after (N/(N-1))^2 compensation".

### 6.5 Sample roundtrip on synthesised input

`f' = su2_fft_inv(su2_fft(su2_fft_inv(fhat)))`.  Compare with `su2_fft_inv(fhat)`.
Same Riemann error applies.  Use the same compensated assertion.

### 6.6 Cross-check against su2_ft_direct (gold standard)

Random fhat; compute `f_inv = su2_fft_inv(fhat)`.
Then compute `fhat2 = su2_ft_direct(f_inv)`.
Assert `||fhat2 - fhat * (N/(N-1))^2|| / ||fhat|| < 1e-10`.
This is the analog of `test_fft_matches_direct_random` and is the MANDATORY
cross-check before shipping (CLAUDE.md "Done" checklist item).

---

## 7. Implementation concerns and open questions

### 7.1 Closed-grid endpoint distribution

The inverse produces f on a M x M grid (j in [0, M-1]).  The closed-grid copies
`f[N-1, k, j2] = f[0, k, j2]` and `f[j1, k, N-1] = f[j1, k, 0]` are set
explicitly after each FFTW call.  This is consistent with the forward's fold
`g[0] += f[N-1]`: if the synthesised f satisfies `f[0] = f[N-1]` (which it must
for trigonometric polynomials of degree < N-1), the fold maps to a sum that equals
`2 * f[0]` at that bin, which is the correct N-fold DFT accounting.

### 7.2 sin(theta) weighting

The forward includes `sin(theta_k)` as a quadrature weight (paper line 1342).
The inverse synthesis has NO sin or dtheta weight — it is a pure linear combination
of basis functions.  Confirm: there is no `sin_th[k]` factor in Stage 2-inv.

### 7.3 Normalisation accounting

The `1/(8pi^2)` factor lives entirely in the forward (ALGORITHM.md, su2_fft.c `norm`).
The `(2l+1)` factor lives entirely in the inverse synthesis (paper line 554).
There is no `1/(8pi^2)` in `su2_fft_inv`.  No `dphi*dtheta*dpsi` prefactor either.

Sanity check: for fhat(0)_{0,0} = A (constant), `su2_fft_inv` returns f = A everywhere.
Then `su2_fft(f = A)` at (l=0, m=0, n=0) returns
`norm * A * sum_k sin(theta_k) * M * M`, where the M*M factor comes from the
phi and psi sums over the closed (N-1)-point grid.  The result is
`A * (N/(N-1))^2` (HANDOFF.md §2 item 7) — NOT A.  This (N/(N-1))^2 mismatch
is the documented Riemann error, not a normalisation bug.

### 7.4 Path to 1e-12 spectrum roundtrip

Three options, in order of preference:

**(a) Gauss-Legendre theta nodes (bead `su2fft-ega`).**
Replace the N uniform theta nodes with N Gauss-Legendre nodes on [0, pi] with
weights `w_k` (not `sin(theta_k) * dtheta`).  The GL quadrature is exact for
polynomials up to degree 2N-1, making the theta inner product exact.  Expected
roundtrip: 1e-15 at all N.  Requires new `su2_grid_theta_gl` and updated
`su2_fft` weight loop.  This is the correct long-term fix.

**(b) Compensation factor in the inverse.**
Apply `1 / (N/(N-1))^2` as a scalar prefactor on fhat inside `su2_fft_inv`.
This corrects the l=0 case exactly and approximately corrects higher l (because
the sin-weighted Riemann error is nearly l-independent for smooth d^l).
This is a hack; it improves the l=0 roundtrip to 1e-15 but may shift the error
for higher-l coefficients.  Not recommended without numerical validation.

**(c) Document the achievable tolerance.**
Assert `||fhat' - fhat * (N/(N-1))^2|| < eps_fp` in tests.  This is the honest
position until bead `su2fft-ega` is implemented.  Achievable today.

**Recommended: option (c) now, file `su2fft-ega` if not already open, implement (a) in that bead.**

### 7.5 Loop order and memory

Stage 2-inv: outer loop `(m, n)`, inner loop `(k, l)`.  This matches the forward's
loop order and amortises the `su2_wigner_d_seq` call over all k.  The G array
is `N * (2N-1)^2` complex doubles: at N=24, `24 * 47^2 * 16 bytes ≈ 843 KB`.
Acceptable; no special allocation strategy needed.

Stage 1-inv: one (M x M) FFTW FORWARD plan, reused across all N theta slices.

---

## 8. Open follow-up beads

Do NOT implement these in `su2fft-3lx`; file as new beads if not already open:

- **`su2fft-ega` (Gauss-Legendre theta nodes):** replaces the closed-grid Riemann
  sum in theta with exact GL quadrature.  Prerequisite for 1e-12 roundtrip.
  Already filed per HANDOFF.md.

- **Arb-precision inverse FFT:** analogous to `su2_fft_arb`; uses `acb_dft_prod`
  with the `conj(forward(conj(.)))` pattern reversed for synthesis.  CLAUDE.md
  invariant 4: `acb_dft_prod` is forward-only.

- **Julia bindings update (`su2fft-t6z` follow-on):** expose `su2_fft_inv` via
  `ccall` in `SU2FFT.jl`.  Add `SU2FFT.fft_inv` accessor and roundtrip test in
  `julia/test/runtests.jl`.

- **SIMD / OpenMP for Stage 2-inv:** the `(m, n)` outer loop is embarrassingly
  parallel; the l-accumulation inner loop can use SIMD similarly to the forward's
  Stage 2.  Profile first (bead `su2fft-cvh` / `su2fft-lg8` analogues).
