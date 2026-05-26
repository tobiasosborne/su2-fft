# Library Evaluation for SU(2) FFT

**Date:** 2026-05-26  
**Context:** Implementing a direct O(N^6) reference DFT and the O(N^4 log N log log N) divide-and-conquer FFT on SU(2) from arXiv 2605.23923.

---

## Section 1: What FLINT 3.0.1 Actually Provides

**Installed:** `libflint.so.18`, `libflint-dev 3.0.1-3.1build1` (confirmed via `ldconfig -p` and `dpkg -l`).

### `acb_dft.h` — 1D complex DFT/FFT (arbitrary precision)

Source: `/usr/include/flint/acb_dft.h`, confirmed at https://flintlib.org/doc/acb_dft.html

Provides 1D DFT over `acb_t` (ball-arithmetic complex numbers at arbitrary precision):
- `acb_dft(w, v, len, prec)` — auto-selects algorithm
- `acb_dft_inverse(w, v, len, prec)`
- `acb_dft_rad2(w, v, e, prec)` — radix-2 FFT for length 2^e
- `acb_dft_bluestein(w, v, len, prec)` — arbitrary length via chirp-z
- `acb_dft_prod(w, v, cyc, num, prec)` — DFT on a product group Z_{n1} x ... x Z_{nk}
- Precomputed variants: `acb_dft_precomp`, `acb_dft_inverse_precomp`

**2D FFT:** No dedicated 2D routine. `acb_dft_prod` can compute a 2D DFT over Z_M x Z_N by treating it as a product group, but this requires manual index mapping (row-major linearisation). No `fftw_plan_dft_2d` equivalent.

### `acb_hypgeom.h` / `arb_hypgeom.h` — Special functions (arbitrary and double precision)

Source: `/usr/include/flint/acb_hypgeom.h`, `/usr/include/flint/arb_hypgeom.h`, confirmed at https://flintlib.org/doc/acb_hypgeom.html and https://flintlib.org/doc/arb_hypgeom.html

**Jacobi polynomials P_n^{(a,b)}(x):**
- `arb_hypgeom_jacobi_p(res, n, a, b, z, prec)` — real, arbitrary precision
- `acb_hypgeom_jacobi_p(res, n, a, b, z, prec)` — complex, arbitrary precision
- `arb_fpwrap_double_jacobi_p(res, n, a, b, x, flags)` — IEEE double wrapper
- `arb_fpwrap_cdouble_jacobi_p(res, n, a, b, x, flags)` — complex double wrapper

**Associated Legendre functions P_n^m(z):**
- `acb_hypgeom_legendre_p(res, n, m, z, type, prec)` — arbitrary precision, m can be nonzero
- `acb_hypgeom_legendre_p_uiui_rec(res, n, m, z, prec)` — fast recurrence for unsigned integer n,m
- `arb_hypgeom_legendre_p(res, n, m, z, type, prec)` — real version
- `arb_fpwrap_double_legendre_p(res, n, m, x, type, flags)` — double wrapper

**Gauss-Legendre quadrature nodes and weights:**
- `arb_hypgeom_legendre_p_ui_root(res, weight, n, k, prec)` — k-th root of P_n and corresponding GL weight; confirmed "set to the weight corresponding to the node for Gaussian quadrature on [-1,1]"
- `arb_fpwrap_double_legendre_root(res1, res2, n, k, flags)` — double wrapper

### `acb_mat.h` — Complex matrix arithmetic

Source: `/usr/include/flint/acb_mat.h`

Full complex matrix library over `acb_t`: `acb_mat_init`, `acb_mat_mul`, `acb_mat_add`, `acb_mat_scalar_mul_acb`, etc. Used for accumulating (2l+1)x(2l+1) matrix coefficients in the reference DFT.

---

## Section 2: Gaps in FLINT for This Project

### Gap 1: No 2D (or strided/batched) FFT

`acb_dft_prod` can perform a 2D DFT over Z_M x Z_N, but only via product-group decomposition with manual linearisation. There is no FFTW-style `fftw_plan_dft_2d`. For the (α, γ) stage of the arXiv algorithm, where M = N = 2*bandlimit and the array is a full 2D grid, FFTW3 is a far more natural fit: its `fftw_plan_dft_2d` handles arbitrary sizes, supports in-place or out-of-place, and is highly optimised with SIMD/multithreading. FLINT's `acb_dft` is slower for pure IEEE double work because it carries interval arithmetic overhead.

**FLINT `acb_dft` is also only for 1D arrays.** Confirmed by reading the full 331-line header: the `_prod` variant handles product groups but its signature is `acb_dft_prod(w, v, slong* cyc, slong num, prec)` — a 1D output array with a cycle decomposition, not a 2D array type.

### Gap 2: No Wigner small-d functions

Neither `acb_hypgeom` nor any other FLINT header provides `d^l_{mn}(β)`. They must be computed from scratch using the Jacobi polynomial relation:

    d^l_{mn}(β) = c_{lmn} * sin^{|m-n|}(β/2) * cos^{|m+n|}(β/2) * P_k^{(|m-n|, |m+n|)}(cos β)

FLINT provides all the ingredients (`arb_hypgeom_jacobi_p`, trig functions via `arb`), so this is buildable but not pre-packaged.

### Gap 3: No batch/recurrence interface for Jacobi evaluation

`arb_hypgeom_jacobi_p` evaluates a single P_n^{(a,b)}(x). The β-stage of the divide-and-conquer algorithm needs P_0, P_1, ..., P_{2l}^{(a,b)}(x) for fixed x — a length-(2l+1) sequence best computed by the three-term recurrence. FLINT offers no vectorised batch evaluator for this; the recurrence must be coded manually (which is straightforward).

### Gap 4: FFTW3 dev headers not installed

`libfftw3-double3` (runtime) is present but `libfftw3-dev` is **not** installed (`dpkg -l` confirms `un libfftw3-dev`). Headers (`fftw3.h`) are absent from `/usr/include`. Must install before use.

### Gap 5: GSL dev headers not installed

`libgsl.so.27` is present (runtime) but no GSL headers were found. If GSL is needed for Jacobi/Legendre at double precision only, `arb_fpwrap_double_*` from FLINT can substitute.

---

## Section 3: Recommended Toolkit

| Library | Version installed | Role | Link flag |
|---------|------------------|------|-----------|
| FLINT (libflint) | 3.0.1 | Complex numbers (`acb_t`), arbitrary-precision arithmetic, `acb_mat` matrix ops, Jacobi polynomials, associated Legendre functions, Gauss-Legendre quadrature nodes/weights, 1D DFT (reference transform) | `-lflint` |
| FFTW3 | 3.3.10 (runtime); dev needed | 2D complex DFT over (α, γ) grid in the fast algorithm; also 1D FFT for the α and γ marginals | `-lfftw3` |
| GMP / MPFR | system (pulled by FLINT) | Underlying arbitrary precision, already a transitive dependency | (automatic) |

**Not needed:**
- GSL: FLINT's `arb_fpwrap_double_jacobi_p`, `arb_fpwrap_double_legendre_p`, and `arb_fpwrap_double_legendre_root` fully replace GSL's `gsl_sf_jacobi`, `gsl_sf_legendre_Plm`, and `gsl_integration_glfixed_table` at IEEE double precision. No separate GSL dependency required.
- SHTns / s2kit / S2HAT: These do spherical harmonic transforms on S^2 (the 2-sphere), not on SU(2) = S^3. They do not cover the Wigner D-matrix / (α,β,γ) structure needed here.

**Installation action required:**
```sh
sudo apt install libfftw3-dev   # adds fftw3.h and -lfftw3
```

---

## Section 4: Ranked Recommendation

**Primary recommendation: FLINT 3.0.1 + FFTW3**

FLINT covers almost everything:
1. `acb_t` complex arithmetic with certified error bounds — useful for validating the reference O(N^6) transform.
2. `acb_mat` for the (2l+1)×(2l+1) matrix coefficient accumulation.
3. `arb_hypgeom_jacobi_p` / `acb_hypgeom_jacobi_p` for Wigner d-function construction via Jacobi recurrence.
4. `acb_hypgeom_legendre_p_uiui_rec` for associated Legendre at integer n,m (fast path).
5. `arb_hypgeom_legendre_p_ui_root` for Gauss-Legendre nodes and weights at arbitrary precision.
6. `arb_fpwrap_double_*` wrappers expose all of the above at IEEE double with a standard `double` interface, avoiding ball-arithmetic overhead in production runs.
7. `acb_dft` for the reference transform's discrete α/γ sums and for correctness checks.

The single gap FFTW3 fills is the **2D complex DFT over the (α, γ) grid** in the fast algorithm. FLINT's `acb_dft_prod` could substitute (it supports product-group DFTs), but FFTW3 is simpler to use for a plain 2D array, significantly faster at IEEE double, and is already present as a runtime library.

**Wigner small-d functions** must be hand-coded (≈ 20 lines using the Jacobi relation above), calling `arb_fpwrap_double_jacobi_p` for each (l, m, n, β).

**Alternative stack (no FLINT):** GSL + FFTW3 + hand-coded recurrences. This is double-precision only, has no error bounds, and provides nothing beyond what FLINT already does for this use case. Not recommended given the user preference for FLINT and the strong coverage confirmed above.

**Docs consulted:**
- https://flintlib.org/doc/acb_dft.html
- https://flintlib.org/doc/arb_hypgeom.html
- https://flintlib.org/doc/acb_hypgeom.html
- Headers read: `/usr/include/flint/acb_dft.h`, `acb_hypgeom.h`, `arb_hypgeom.h`, `acb_mat.h`, `arb_fpwrap.h`
