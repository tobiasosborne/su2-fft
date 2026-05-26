# Gauss-Legendre Quadrature for the SU(2) Fourier Transform

Bead: `su2fft-ega`.  Implementation target: `src/su2_gauss_legendre.c` (new),
`src/su2_fft.c` (two new functions appended).
All paper citations are line numbers in `paper.tex` (arxiv 2605.23923).

---

## 1. The Quadrature Substitution

The forward analysis integral (paper.tex line 1316):

```
fhat(l)_{m,n} = (1/8pi^2) * int_0^{2pi} dphi int_0^{2pi} dpsi
                            int_0^pi sin(theta) dtheta
                            * f(g) * conj(t^l_{n,m}(g))
```

Substitute `x = cos(theta)`, `dx = -sin(theta) dtheta`, domain maps as
[0,pi] -> [-1,1] (with sign flip absorbed):

```
int_0^pi sin(theta) dtheta * F(theta) = int_{-1}^{1} F(arccos(x)) dx
```

Approximate the x-integral with an N-point Gauss-Legendre quadrature on [-1,1]:

```
int_{-1}^{1} G(x) dx ≈ sum_{k=0}^{N-1} w_k * G(x_k)
```

where `x_k` are the N roots of Legendre polynomial `P_N(x)` (ascending order) and
`w_k` are the corresponding GL weights.  Defining `theta_k = arccos(x_k)`:

```
fhat(l)_{m,n} ≈ c * sum_{j1,k,j2} w_k * f(g_{j1,k,j2}) * conj(t^l_{n,m}(g))
```

KEY: the `sin(theta_k) * dtheta` weight in the existing closed-grid Riemann sum
is replaced by the single GL weight `w_k`.  The Jacobian `sin(theta) dtheta = -dx`
is absorbed into `w_k` by the substitution.  Both are dimensionless: `w_k` carries
units of dx (the dx quadrature weight), and `sin(theta) dtheta` is the angle-element
Jacobian.  They represent the same measure on [-1,1].

The GL nodes are NOT uniformly spaced: they cluster near the endpoints `x = +/-1`
(theta near 0 and pi), which is where Wigner-d functions vary most rapidly.

---

## 2. Exactness Range of N-Point Gauss-Legendre

N-point GL is exact for polynomials in x of degree <= 2N-1 (Press et al.,
"Numerical Recipes in C", 3rd ed., §4.6; DLMF §3.5(v)).

**Is the integrand polynomial in x = cos(theta)?**

From `notes/wigner_recurrence.md` §1, the Wigner-d function factors as:

```
d^l_{n,m}(theta) = sigma * R(l) * sin(theta/2)^a * cos(theta/2)^b
                 * P_{l-M}^{(a,b)}(cos theta)
```

where `a = |m-n|`, `b = |m+n|`, `M = max(|m|,|n|)`, `sigma` is a sign constant,
`R(l)` is a square-root factorial weight, and `P_k^{(a,b)}` is the Jacobi polynomial
of degree `k = l-M`.

The half-angle factors `sin(theta/2)^a = ((1-x)/2)^{a/2}` and
`cos(theta/2)^b = ((1+x)/2)^{b/2}` are polynomials in x of degree `a/2` and `b/2`
when `a` and `b` are even (i.e., when `|m-n|` and `|m+n|` are even).  When they
are odd (half-integer powers of `1-x` or `1+x`), the product is NOT a polynomial
in x; rather it is a polynomial in `sin(theta/2)` and `cos(theta/2)` separately.

**However**, the relevant integral for the forward analysis is:

```
int_{-1}^{1} d^l_{n,m}(arccos x) * conj(d^{l'}_{n,m}(arccos x)) dx
```

By the Schur orthogonality for the group SU(2) (Vilenkin, "Special Functions and
the Theory of Group Representations", §III.6, eq. (8)), the Wigner-d functions
satisfy:

```
int_0^pi d^l_{n,m}(theta) * d^{l'}_{n,m}(theta) * sin(theta) dtheta
  = 2/(2l+1) * delta_{l,l'}
```

Changing variables to x = cos(theta) (so sin(theta) dtheta = -dx):

```
int_{-1}^1 d^l_{n,m}(arccos x) * d^{l'}_{n,m}(arccos x) dx = 2/(2l+1) * delta_{l,l'}
```

From the Jacobi polynomial connection, the product `d^l * d^{l'}` under dx on
[-1,1] behaves as a degree `(l+l')` polynomial in x multiplied by powers of
`(1-x)` and `(1+x)` (from the endpoint factors).  The Gauss-Jacobi quadrature
with parameters `(a,b)` handles this exactly.  For the specific case of
Gauss-Legendre (parameters a=b=0), exactness holds when the product is a
polynomial in x of degree <= 2N-1.

For the roundtrip integral at bandlimit N, the maximum product degree is
`l + l' <= 2(N-1)`.  Since `2(N-1) <= 2N-1`, **N-point GL is exact for all
pairwise products of Wigner-d functions at bandlimit N-1**.  This is the
polynomial-degree bound that guarantees exact roundtrip.

**Caveat:** this exactness claim relies on the Wigner-d product being a polynomial
in x (not merely a polynomial in the half-angle factors).  For the specific
diagonal and off-diagonal integrals needed for the Fourier analysis, the relevant
quantity is `sum_k w_k * d^l * F2[k]` where `F2[k]` is the Stage 1 output at GL
node `x_k`.  The GL integration is exact if this is equivalent to integrating a
degree-`2(N-1)` function, which it is by the orthogonality argument above.

---

## 3. Newton Iteration for Gauss-Legendre Nodes and Weights

Standard algorithm (Press et al. §4.6):

**Initial guess** for the k-th root (k = 1..N, 1-indexed):
```
x_k^{(0)} = cos(pi * (k - 0.25) / (N + 0.5))
```

**Legendre polynomial via three-term recurrence** (DLMF 18.9.1):
```
P_0(x) = 1
P_1(x) = x
(j+1) P_{j+1}(x) = (2j+1) x P_j(x) - j P_{j-1}(x)     for j >= 1
```

**Derivative** from the standard formula:
```
P_N'(x) = N * (x * P_N(x) - P_{N-1}(x)) / (x^2 - 1)
```

**Newton iteration** (converges quadratically; typically 4-6 iterations to 1e-15):
```
x_k^{(i+1)} = x_k^{(i)} - P_N(x_k^{(i)}) / P_N'(x_k^{(i)})
```
Iterate until `|x_k^{(i+1)} - x_k^{(i)}| < 1e-15`.

**Weight:**
```
w_k = 2 / ((1 - x_k^2) * (P_N'(x_k))^2)
```

**Pseudocode (C-style, ~30 LOC):**

```c
void su2_gl_nodes_weights(int N, double *x, double *w)
{
    /* Roots come in +/- pairs; compute only the lower half. */
    int m = (N + 1) / 2;
    for (int i = 1; i <= m; i++) {
        /* Initial guess (Press §4.6). */
        double xi = cos(M_PI * (i - 0.25) / (N + 0.5));
        double pp, p1, p2, p3;
        do {
            /* Build P_N(xi) and P_{N-1}(xi) via recurrence. */
            p1 = 1.0; p2 = 0.0;
            for (int j = 1; j <= N; j++) {
                p3 = p2; p2 = p1;
                p1 = ((2*j - 1) * xi * p2 - (j-1) * p3) / (double)j;
            }
            /* p1 = P_N(xi), p2 = P_{N-1}(xi) */
            pp = N * (xi * p1 - p2) / (xi * xi - 1.0);
            double xi_old = xi;
            xi -= p1 / pp;
        } while (fabs(xi - xi_old) > 1e-15);

        /* Symmetric pair. */
        x[i - 1]     = -xi;        /* lower half: ascending order */
        x[N - i]     =  xi;        /* upper half */
        w[i - 1] = w[N - i] = 2.0 / ((1.0 - xi*xi) * pp*pp);
    }
}
```

Note: the roots come in symmetric pairs `(x_k, -x_{N-1-k})`; the loop exploits
symmetry by computing only the lower half.  Output is in ascending order (x[0]
is most negative, x[N-1] is most positive).

**Achievable tolerance:** Newton iteration to `1e-15` converges in 4-6 steps for
N up to several hundred.  The resulting nodes and weights have 14-15 significant
digits, appropriate for double precision.  Stable for N up to a few thousand;
no special scaling or extended precision is required within the Newton loop.

---

## 4. C Interface Design

New helper (declared in `include/su2.h`, implemented in new file
`src/su2_gauss_legendre.c`):

```c
/* Compute N-point Gauss-Legendre nodes and weights on [-1, 1].
 *
 * @param[in]  N     Number of nodes (>= 1).
 * @param[out] x     Length-N array; x[k] = k-th GL root in [-1, 1], ascending.
 * @param[out] w     Length-N array; w[k] = corresponding GL weight.
 *
 * Cost: O(N^2) Newton iterations; each iteration evaluates P_N in O(N).
 * Stable for N <= a few thousand (sufficient for SU(2) FFT bandlimits).
 */
void su2_gl_nodes_weights(int N, double *x, double *w);
```

This function is called once at the start of `su2_fft_gl` and `su2_fft_inv_gl`
and its O(N^2) cost is dominated by the O(N^4) Stage 2 at any useful N.
No persistent state; callers free `x` and `w`.

---

## 5. `su2_fft_gl` and `su2_fft_inv_gl` Interface

```c
/* Forward FFT with Gauss-Legendre theta nodes.
 *
 * Same signature as su2_fft but uses N-point Gauss-Legendre quadrature for
 * the theta integral instead of the closed-grid Riemann sum.  The cos(theta_k)
 * values are the N roots of Legendre P_N(x) on [-1, 1] (ascending), and the
 * theta integral weight is the GL weight w_k (no separate sin(theta_k)*dtheta).
 *
 * Sample storage: f[j1*N*N + k*N + j2] where theta_k = arccos(x_k), x_k the
 * k-th GL node (ascending: x[0] < x[1] < ... < x[N-1]).  The phi and psi grids
 * are unchanged from su2_fft (closed grid, N-1 independent modes).
 *
 * Normalisation: norm = 1 / (2 * N^2).  Derivation in notes/gauss_legendre.md §6.
 *
 * Roundtrip: forward_gl(inverse_gl(fhat)) == fhat to machine precision because
 * N-point GL is exact for polynomials of degree <= 2N-1, covering the Wigner-d
 * orthogonality integrals at bandlimit N-1 (see §2).
 */
void su2_fft_gl(int N, const double _Complex *f, double _Complex *fhat);

/* Inverse FFT (Peter-Weyl synthesis) sampling at Gauss-Legendre theta nodes.
 *
 * Same signature as su2_fft_inv but the output `f` is sampled at the GL
 * theta nodes (same convention as su2_fft_gl input).  The phi and psi grids
 * are the same closed grid as in su2_fft_inv.
 *
 * The inverse has no su2_gl_nodes_weights dependency at runtime if the caller
 * only needs samples for immediate re-analysis with su2_fft_gl.  However, if
 * the sample positions are needed separately, call su2_gl_nodes_weights
 * directly to obtain theta_k = arccos(x_k).
 */
void su2_fft_inv_gl(int N, const double _Complex *fhat, double _Complex *f);
```

Both functions live at the end of `src/su2_fft.c`, appended after `su2_fft_inv`.
Declarations go in `include/su2.h`.

---

## 6. Implementation Strategy

### Stage 1 (FFTW + closed-grid fold + (-1)^{n+m} phase): UNCHANGED

The phi and psi grids remain closed (`phi[j] = -pi + j*2pi/(N-1)`).  FFTW
backward 2D plan of size (N-1)x(N-1) with the fold trick is identical to
`su2_fft`.  Stage 1 output `F2[k,n,m]` is computed the same way; only the theta
values at which the samples are taken differ (GL nodes vs uniform grid).

### Stage 2 (Wigner sweep): CHANGED weights only

1. Call `su2_gl_nodes_weights(N, x_arr, w_arr)` to obtain GL nodes and weights.
2. For each k: `theta_k = acos(x_arr[k])` (use `acos`, not `arccos`; C math.h).
3. In the Stage 2 accumulator, replace:
   ```c
   double _Complex w = F2[...] * sin_th[k];     /* old Riemann weight */
   ```
   with:
   ```c
   double _Complex w = F2[...] * w_arr[k];      /* GL weight */
   ```
4. Remove `sin_th[k]` array; allocate `w_arr` instead.
5. Drop `dtheta = pi/(N-1)` from `norm`:

### Normalisation derivation

The forward analysis is the discrete quadrature:
```
fhat(l)_{m,n} = norm * sum_{j1,k,j2} f * exp(+i*n*phi_{j1}+i*m*psi_{j2}) * w_k * d^l_{n,m}(theta_k) * i^{n-m}
```

The phi/psi closed-grid DFT (via Stage 1 fold + FFTW BACKWARD, unnormalized) gives:
```
F2[k,n,m] = sum_{j1=0}^{N-1} sum_{j2=0}^{N-1} f[j1,k,j2] * exp(+i*n*phi_{j1}) * exp(+i*m*psi_{j2})
```

For a constant f=1: F2[k,0,0] = N^2 (fold doubles the endpoints in both
dimensions; sum over N+N-(N-1) = ... accumulated as N^2 total contributions;
see code trace in su2_fft.c Stage 1 comments).

The inverse synthesis evaluates `f` at the GL theta nodes.  Substituting
back into the forward:

```
fhat'(l)_{m,n} = norm * N^2 * sum_k w_k * d^l(theta_k)
                 * [sum_{l'} (2l'+1) * fhat(l') * d^{l'}(theta_k)]
```

The GL sum (exact by the polynomial-degree bound of §2):
```
sum_k w_k * d^l(theta_k) * d^{l'}(theta_k) = int_{-1}^1 d^l(x) d^{l'}(x) dx
                                             = 2/(2l+1) * delta_{l,l'}
```

Therefore:
```
fhat'(l)_{m,n} = norm * N^2 * (2l+1) * fhat(l) * 2/(2l+1)
               = norm * N^2 * 2 * fhat(l)
```

For `fhat'(l)_{m,n} = fhat(l)_{m,n}`:
```
norm * N^2 * 2 = 1   =>   norm = 1 / (2 * N^2)
```

**The GL forward normalization is `norm_gl = 1 / (2 * N^2)`.**

This differs from the Riemann norm `dphi*dtheta*dpsi/(8pi^2) = pi/(2(N-1)^3)` by the
factor `N^2/pi` times `(N-1)^3/N^2 = (N-1)^3/pi`, which accounts for:
- Absorbing the theta Jacobian into GL weights (removing `dtheta` from norm).
- The N^2 factor in the unnormalized phi/psi fold sum being exactly compensated by
  `1/N^2` in the new norm.

Note: this is NOT the same as simply removing `dtheta` from the Riemann norm.
The factor `N^2` (from the fold sum) vs `(N-1)^2 = M^2` (from the trapezoidal phi/psi
integral) is the source of the (N/(N-1))^2 factor in the Riemann roundtrip.
With GL theta and `norm_gl = 1/(2N^2)`, this factor is exactly cancelled.

**C code change:**
```c
/* Riemann (existing): */
const double norm = dphi * dtheta * dpsi / (8.0 * M_PI * M_PI);

/* GL (new): */
const double norm = 1.0 / (2.0 * (double)N * (double)N);
```

### Sanity check at l=0, m=n=0

For constant f=1, `norm_gl * N^2 * 2 = 1/(2N^2) * N^2 * 2 = 1`.
For l=1, m=n=0: `d^1_{0,0}(theta) = cos(theta) = x`.
`sum_k w_k * x_k^2 = int_{-1}^1 x^2 dx = 2/3`.
`fhat(1,0,0) = norm_gl * N^2 * 3 * fhat(1,0,0) * (2/3) = 1/(2N^2) * N^2 * 2 * fhat = fhat`. Correct.

---

## 7. Test Plan

All GL tests live in `tests/test_roundtrip.c` alongside the existing 7 roundtrip tests,
or in a new `tests/test_gl_roundtrip.c` if preferred.

### Test 1: GL roundtrip exact to 1e-12

Random fhat (N coefficients, bandlimit N). Compute:
```
f       = su2_fft_inv_gl(fhat, N)
fhat2   = su2_fft_gl(f, N)
err     = max |fhat2[i] - fhat[i]| / max |fhat[i]|
```
Assert `err < 1e-12` at N = 4, 8, 16, 24.
The tolerance 1e-12 is achievable because GL quadrature is exact for polynomials of
degree <= 2N-1 (§2), and the only remaining error is floating-point rounding in the
Wigner recurrence and FFTW.

Compare with the closed-grid roundtrip which gives relative error ~5.6 at N=16
(`test_roundtrip.c` documents this).

### Test 2: Constant-function test

`f = 1` at all GL theta nodes, all j1, j2.
`fhat = su2_fft_gl(f, N)`.
Assert `|fhat(0)_{0,0} - 1.0| < 1e-14`.
Assert all other `fhat(l)_{m,n} = 0` for `l >= 1` to < 1e-12.

This verifies the normalization `norm_gl = 1/(2N^2)` is correct.

### Test 3: Single-coefficient analytical synthesis

Set fhat = zeros; fhat(l0)_{m0,n0} = 1 for small l0, m0, n0.
Compute `f = su2_fft_inv_gl(fhat, N)`.
The expected sample is `f(theta_k) = (2l0+1) * d^{l0}_{n0,m0}(theta_k)` (at phi=psi=0).
Compare pointwise on GL grid to machine precision (1e-12).
Run at (l0=1, m0=n0=0), (l0=1, m0=1, n0=0), (l0=2, m0=0, n0=0) at N=4, 8.

### Test 4: Cross-tolerance on low-l functions

Synthesise a bandlimited f with support only on l = 0..L for L << N.
Compare `su2_fft_gl(f)` and `su2_fft(f)` (Riemann forward): the GL coefficients
should be more accurate for high-l, but both should agree for l <= L where Riemann
is also adequate.

### Test 5: su2_ft_direct cross-check (mandatory)

For N = 4, 6:
```
f     = su2_fft_inv_gl(fhat, N)    (random fhat)
fhat2 = su2_ft_direct(f, N)        (O(N^6), on GL-node grid)
```
Assert `||fhat2 - fhat|| / ||fhat|| < 1e-10`.

Note: `su2_ft_direct` must be called with f sampled at GL theta nodes.  Either
pass GL-node samples directly (since `su2_ft_direct` uses the same theta_k values
it computes internally via `su2_grid_theta`), or implement a GL-specific direct FT.
Simplest: implement `su2_ft_direct_gl` that calls `su2_gl_nodes_weights` for its
theta grid; this is a small addition to `src/su2_ft.c`.  Alternatively, verify
by the `su2_fft_gl vs su2_fft_inv_gl` roundtrip alone (Test 1).

---

## 8. Out of Scope (Future Work)

The following are plausible extensions.  Do NOT file new beads for these now:

- **Arbitrary-precision GL nodes** via FLINT arb arithmetic.  Achievable with
  the same Newton iteration using `arb_t`; would give certified 100+ digit GL
  nodes for the arb FFT path.

- **Alternative quadratures** (Clenshaw-Curtis, Gauss-Lobatto, Gauss-Chebyshev).
  Clenshaw-Curtis on N+1 Chebyshev points has similar exactness properties and
  uses FFT internally (O(N log N) instead of O(N^2) Newton).  Gauss-Lobatto
  includes endpoints and integrates degree <= 2N-3 exactly; slightly less efficient.

- **Direct FT path at GL nodes** (`su2_ft_direct_gl`).  Needed for the cross-check
  in Test 5; straightforward to add but not blocking any application bead.

- **GL grid helper functions** analogous to `su2_grid_theta` / `su2_grid_phi` in
  `src/su2_grid.c`: `su2_grid_theta_gl(N)` returning `acos(x_k)` for each k.

---

## Summary of Key Results

| Quantity | Value |
|---|---|
| GL normalization constant | `norm_gl = 1 / (2 * N^2)` |
| Riemann normalization | `norm = pi / (2 * (N-1)^3)` |
| GL theta exactness | exact for poly degree <= 2N-1 |
| Max Wigner-d product degree | 2(N-1) <= 2N-1, so GL is exact |
| Expected roundtrip error (GL) | ~1e-14 (floating-point noise only) |
| Existing roundtrip error (Riemann) | ~5.6 relative at N=16 |
| New files | `src/su2_gauss_legendre.c` |
| Modified files | `src/su2_fft.c` (append), `include/su2.h` (3 declarations) |
| GL node computation cost | O(N^2), dominated by O(N^4) Stage 2 |
