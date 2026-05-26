# Section 4 Algorithm Summary: Fast Fourier Transform on SU(2)

**Source:** `paper.tex`, lines 594–1298.  
**Paper:** "On The Fast Fourier Transform on SU(2)", arXiv 2605.23923.

---

## 0. Overview

Section 4 organises the SU(2) FFT into four steps (lines 645–653):

1. **Formal definition** — the continuous Fourier transform via irreducible representations.
2. **Discretisation** — replace integrals with Riemann sums on a uniform grid.
3. **2D FFT over (φ, ψ)** — the angular exponentials are handled by a standard 2-D DFT.
4. **Divide-and-conquer over β = θ** — the Jacobi/Legendre inner products are split
   recursively using the composition rule for shifted polynomials.

Only steps 3 and 4 are algorithmic; step 4 is the novel contribution.

---

## 1. Setup and Notation

### 1.1 Fourier transform definition (lines 658–676)

For a band-limited function f with bandwidth N (i.e. f̂(l) = 0 for all l ≥ N),
each Fourier coefficient is the matrix

```
f̂(l)_{mn} = (1 / 8π²) ∫_{-π}^{π} ∫_0^{π} ∫_{-π}^{π}
               f(φ,θ,ψ) · P^l_{nm}(cos θ) · e^{i(nφ + mψ)} · sin θ  dφ dθ dψ
```

for −l ≤ m, n ≤ l (half-integer steps allowed). Here `P^l_{nm}` are the Wigner
d-matrix entries, which for m = n = 0 reduce to the standard Legendre polynomials
`P_l`.

### 1.2 Sampling grid (lines 684–691)

```
φ_{j1} = −π + j1 · (2π / (N−1)),   j1 = 0, …, N−1
ψ_{j2} = −π + j2 · (2π / (N−1)),   j2 = 0, …, N−1
θ_k    =       k  · (π  / (N−1)),   k  = 0, …, N−1
```

Total grid size: N³ points.

---

## 2. Step 3 — The 2-D FFT over (φ, ψ)

**Proposition (lines 695–700, equation labelled Core_eq at line 717):**

Substituting the Riemann-sum approximation and separating variables gives

```
f̂(l)_{mn}  ≈  (1 / 8π²)  Σ_{k=0}^{N-1}  P^l_{nm}(cos θ_k) sin θ_k
                           · [ Σ_{j1} e^{i n φ_{j1}}  Σ_{j2} e^{i m ψ_{j2}}  f(φ_{j1}, θ_k, ψ_{j2}) ]
```

The bracket is a **2-D DFT** in (j1, j2) for fixed k.

**Define** `f2(n, m, θ_k)` = 2-D DFT of the slice `f(·, θ_k, ·)` at frequency (n, m).

Then:

```
f̂(l)_{mn}  ≈  (1 / 8π²)  Σ_{k=0}^{N-1}  [s]_k  ·  P^l_{nm}(cos θ_k)
```

where `[s]_k = f2(n, m, θ_k) · sin θ_k`  is a length-N vector **s** (one per (n,m) pair).

This is equation `Core_eq` (line 717–719):

```
f̂(l)_{mn}  ≈  Σ_{k=0}^{N-1}  [s]_k  P^l_{nm}(cos θ_k)
            =  < s , P^l_{nm} >
```

**Implementation note:** For each of the N values of k, compute the full N×N 2-D
FFT of the slice. Store all (n, m) outputs. The factor 1/(8π²) is applied at the
end. This is entirely standard; use `acb_dft` from FLINT's Arb library.

---

## 3. Step 4 — Divide-and-Conquer over θ (the β-stage)

The rest of section 4 is devoted to computing `< s , P^l_{nm} >` for all l efficiently.
The paper derives this for m = n = 0 (Legendre polynomials) and states the Jacobi
generalisation is identical in structure (lines 1291–1295).

### 3.1 Three-term recurrence — Lemma (lines 737–766, eq. P_l at line 739)

Standard Legendre recurrence (`eq:rodrigues`, line 748):

```
(n+1) P_{n+1}(x) − (2n+1) x P_n(x) + n P_{n-1}(x) = 0
```

Rearranged (`eq:P_l`, line 739–741):

```
P_{l+1}(cos θ)  =  ((2l+1)/(l+1)) cos θ · P_l(cos θ)  −  (l/(l+1)) P_{l-1}(cos θ)
```

Initial conditions: P_0 = 1, P_{−1} = 0.

### 3.2 Shifted coefficients A_r^L, B_r^L — Proposition (lines 771–826, eq. CAP6-8 at line 773)

For any fixed base level L and step r ≥ 1, define trigonometric polynomials
A_r^L(cos θ) and B_r^L(cos θ) by:

```
P_{L+r}(cos θ)  =  A_r^L(cos θ) · P_L(cos θ)  +  B_r^L(cos θ) · P_{L-1}(cos θ)
```

(equation `CAP6-8` / `eq-eq`, lines 773–784).

**Initial conditions** (lines 776, 804–816):
```
A_0^L = 1,   B_0^L = 0   (r = 0: identity)
A_{-1}^L = 0, B_{-1}^L = 1  (r = −1: shift back one)
```

**Matrix recurrence** for A, B — Proposition (lines 830–870, eq. MATRIZ_LEGEN1 at line 880):

The three-term recurrence for P_{L+r} gives the matrix step equation
(`MATRIZ_LEGEN1`, lines 880–895):

```
| P_{L+r+1} |   | (2L+2r+1)/(L+r+1) · cos θ ,  −(L+r)/(L+r+1) |   | P_{L+r}   |
|            | = |                                                 | · |           |
| P_{L+r}   |   |           1                ,          0         |   | P_{L+r-1} |
```

Substituting `P_{L+k} = A_k^L P_L + B_k^L P_{L-1}` yields (`EQmatrix1`, lines 904–923):

```
| A_{r+1}^L   B_{r+1}^L |   | (2L+2r+1)/(L+r+1)·cosθ  −(L+r)/(L+r+1) |   | A_r^L    B_r^L   |
|                        | = |                                           | · |                  |
| A_r^L       B_r^L     |   |          1               ,       0        |   | A_{r-1}^L B_{r-1}^L |
```

In scalar form (lines 860–868):

```
A_{r+1}^L  =  ((2L+2r+1)/(L+r+1)) cos θ · A_r^L  −  ((L+r)/(L+r+1)) A_{r-1}^L
B_{r+1}^L  =  ((2L+2r+1)/(L+r+1)) cos θ · B_r^L  −  ((L+r)/(L+r+1)) B_{r-1}^L
```

So A and B satisfy **the same three-term recurrence as P**, but started at a
different base level L.

**Boundary case L = 0** (lines 998–999, 1067–1081):

```
A_r^0 = P_r,   B_r^0 = 0
```

Meaning: when L = 0, the shifted coefficients reduce to ordinary Legendre polynomials.

### 3.3 Composition rule — Proposition Prop-P_l (lines 972–1081, eq. composition at line 981)

The central algebraic fact enabling divide-and-conquer. Advancing r+s steps from L
equals advancing r steps then s steps from L+r. In matrix form (`eq:composition`,
lines 981–994):

```
| A_{r+s}^L     B_{r+s}^L   |   | A_s^{L+r}    B_s^{L+r}   |   | A_r^L    B_r^L   |
|                             | = |                            | · |                  |
| A_{r+s-1}^L   B_{r+s-1}^L |   | A_{s-1}^{L+r} B_{s-1}^{L+r}|   | A_{r-1}^L B_{r-1}^L|
```

In scalar form (`eq:rs-product`, lines 1025–1059):

```
A_{r+s}^L   = A_s^{L+r} · A_r^L   + B_s^{L+r} · A_{r-1}^L
B_{r+s}^L   = A_s^{L+r} · B_r^L   + B_s^{L+r} · B_{r-1}^L
A_{r+s-1}^L = A_{s-1}^{L+r} · A_r^L + B_{s-1}^{L+r} · A_{r-1}^L
B_{r+s-1}^L = A_{s-1}^{L+r} · B_r^L + B_{s-1}^{L+r} · B_{r-1}^L
```

This is the FFT's "butterfly" analogue: it lets you build the coefficient matrices
for a large step by composing those for two half-steps.

### 3.4 Main theorem — Theorem thm:final_legendre_fft (lines 1107–1237)

**Statement (eq:basic_projection, line 1118):**

If vectors `s^j = s ⊙ P_j` (pointwise products, precomputed for j ≤ L), then:

```
< s , P_{L+r} >  =  < s^L , A_r^L >  +  < s^{L-1} , B_r^L >
```

This is the key step: the inner product against a high-degree polynomial decomposes
into two inner products against lower-degree objects A_r^L and B_r^L.

**Divide-and-conquer expansion (eq:general_decomposition, line 1138):**

When r = n_1 + n_2 + … + n_k, the inner product expands to 2^{k-1} terms:

```
< s , P_{L+r} >  =  Σ_{τ1 ∈ {A,B}}  Σ_{ε1=0}^1  …  Σ_{ε_{k-1}=0}^1
                     < s^{L*(τ1)} , V_{τ1,ε1,…,ε_{k-1}} >
```

where:
- `L_0 = L`,  `L_{i-1} = L + n_1 + … + n_{i-1}`
- `s^{L*(A)} = s^L`,  `s^{L*(B)} = s^{L-1}`
- `V_{τ1,ε1,…,ε_{k-1}} = τ1_{n1-ε1}^L  ⊙  τ2(ε1)_{n2}^{L1}  ⊙  τ3(ε2)_{n3}^{L2}  ⊙  …`
- `τ_i(ε_{i-1}) = A if ε_{i-1}=0, else B`
- `⊙` is the Hadamard (pointwise) product of the N-vectors

**k = 2 example** (lines 1240–1283): For r = n1 + n2 the expansion gives exactly
4 terms:

```
< s , P_{L+r} >  =  < s^L , A_{n2}^{L+n1} ⊙ A_{n1}^L >
                  +  < s^L , B_{n2}^{L+n1} ⊙ A_{n1-1}^L >
                  +  < s^{L-1} , A_{n2}^{L+n1} ⊙ B_{n1}^L >
                  +  < s^{L-1} , B_{n2}^{L+n1} ⊙ B_{n1-1}^L >
```

---

## 4. Recursion Structure

**What is recursed over:** The total degree shift r from a base level L.

**How to halve the problem:** Split r = n1 + n2 (e.g. n1 = n2 = r/2). Compute the
shifted coefficient matrices for the two half-shifts, then combine via the
composition rule (matrix product of 2×2 matrices whose entries are N-vectors).

**Base case:** r = 1. Then A_1^L(cos θ) = ((2L+1)/(L+1)) cos θ and
B_1^L(cos θ) = −L/(L+1) (constants at each grid point), obtained directly from
the scalar recurrence. No further splitting needed.

**How to combine:** Use the four scalar formulas from eq:rs-product (lines 1025–1059).
Each formula is a pointwise (Hadamard) product of two N-vectors, followed by
pointwise addition — O(N) work per combination step.

**Full recursion depth:** log2(r) levels, each with O(N) work per node, and O(r)
nodes at the bottom. For all l from 0 to N−1, the total β-stage cost is O(N² log N)
instead of the naive O(N³).

**Generalisation to Jacobi (m, n ≠ 0):** Identical structure. Replace P_l with
P^l_{nm} (Wigner d-matrix entries = Jacobi polynomials with parameters depending on
m and n). The recurrence for P^l_{nm} has the same three-term form, so A_r^L and
B_r^L become Jacobi variants satisfying the same composition rule.

---

## 5. Complete Pseudocode (C / FLINT implementer)

```
/*
 * Inputs:
 *   f[j1][k][j2]  – function values on the N×N×N grid
 *                   j1 indexes φ, k indexes θ, j2 indexes ψ
 *   N             – bandwidth (grid size in each dimension)
 *
 * Outputs:
 *   fhat[l][m][n] – Fourier coefficients, l = 0, …, N-1,
 *                   −l ≤ m,n ≤ l (half-integer steps if needed)
 */

// ─── STAGE 1: 2-D FFT over (φ, ψ) for each θ-slice ──────────────────────────

for k = 0 to N-1:
    // slice2d[j1][j2] = f[j1][k][j2]
    F2[k][n][m] = 2D_FFT( slice2d )   // complex N×N output, all (n,m) pairs
                                        // uses acb_dft or fftw

// ─── STAGE 2: form the weighted vector s^{(n,m)}_k for each (n,m) ────────────

// Grid points:  theta_k = k * pi / (N-1)
for k = 0 to N-1:
    w[k] = sin(theta_k)                // quadrature weight

for each (n, m) with |n|,|m| < N:
    for k = 0 to N-1:
        s[n][m][k] = F2[k][n][m] * w[k] / (8 * pi^2)
    // s[n][m] is now the length-N vector fed into the β-stage

// ─── STAGE 3: β-stage — discrete Legendre/Jacobi transform ───────────────────
//
// For each (n, m) pair, compute:
//   fhat[l][m][n] = < s[n][m] , P^l_{nm}(cos θ_·) >   for l = 0, …, N-1
//
// Using the divide-and-conquer scheme below.

for each (n, m):
    s_vec = s[n][m]             // length-N input
    legendre_fft(s_vec, n, m, N, fhat[·][m][n])


// ─── SUBROUTINE: legendre_fft ─────────────────────────────────────────────────
//
// Computes < s_vec , P^l_{nm}(cos θ_k) > for l = 0, …, N-1
// Base case uses direct dot products; recursive case uses the composition rule.

procedure legendre_fft(s_vec, alpha, beta, N, out):

    // Precompute grid: x[k] = cos(theta_k) for k = 0..N-1

    // --- Direct pass for small l (base case) ---
    // For l = 0, 1: evaluate P^l_{nm}(x[k]) directly, form dot product.
    // (For general Jacobi: use FLINT's arb_hypgeom_jacobi_p or manual recurrence.)

    // Initialise the "current pair" of P-vectors at level 0:
    //   P_prev[k] = P^{-1}_{nm}(x[k]) = 0   (by convention)
    //   P_curr[k] = P^0_{nm}(x[k])           (known closed form)

    for l = 0 to N-1:
        out[l] = dot(s_vec, P_curr)           // < s , P^l_{nm} >   O(N) each
        // advance P using three-term recurrence eq:P_l (line 739):
        //   P_next[k] = ((2l+1)/(l+1)) x[k] P_curr[k]  -  (l/(l+1)) P_prev[k]
        //   (replace (2l+1)/(l+1) etc. with Jacobi coefficients for m,n ≠ 0)
        swap P_prev ↔ P_curr; P_curr ← P_next

    // This naive O(N²) pass is the direct algorithm.
    // ─────────────────────────────────────────────────────────────────────────
    // The FFT-accelerated version replaces the above with the following
    // divide-and-conquer scheme based on Theorem thm:final_legendre_fft:


// ─── DIVIDE-AND-CONQUER β-stage (the FFT proper) ─────────────────────────────
//
// Goal: compute fhat[l] = < s , P_{l} > for ALL l = 0..N-1
//       in O(N log²N) instead of O(N²).
//
// Key objects (all are length-N vectors over the θ-grid):
//   s^j[k]     = s[k] * P_j(x[k])           ("modulated" input)
//   A_r^L[k]   = A_r^L(x[k])                (shifted coefficient, type A)
//   B_r^L[k]   = B_r^L(x[k])                (shifted coefficient, type B)
//
// Composition rule (eq:rs-product, line 1025):
//   A_{r+s}^L = A_s^{L+r} ⊙ A_r^L  +  B_s^{L+r} ⊙ A_{r-1}^L
//   B_{r+s}^L = A_s^{L+r} ⊙ B_r^L  +  B_s^{L+r} ⊙ B_{r-1}^L
//   (and analogously for the (r+s-1) row)
//
// Basic projection (eq:basic_projection, line 1118):
//   < s , P_{L+r} >  =  < s^L , A_r^L >  +  < s^{L-1} , B_r^L >


// Top-level call: compute ALL projections < s , P_l > for l = 0..N-1.
procedure beta_fft(s_vec, x[], N, out[]):

    // Step A: precompute s^j = s ⊙ P_j for j = 0..N-1
    //   (store as N×N matrix S[j][k] = s[k] * P_j(x[k]))
    //   Can reuse the sequential Legendre pass to build these.

    for k = 0..N-1:
        P_prev[k] = 0; P_curr[k] = P^0(x[k])
    S[0] = s_vec ⊙ P_curr   // s^0

    for j = 1 to N-1:
        compute P_next from P_curr, P_prev via recurrence eq:P_l
        S[j] = s_vec ⊙ P_next
        P_prev ← P_curr; P_curr ← P_next
    // Cost: O(N²) but each step is O(N). This pre-pass is unavoidable.

    // Step B: use the composition rule to compute all < s , P_l >
    //
    // Recursive procedure: compute_proj(L, r, s^L, s^{L-1}) → < s , P_{L+r} >
    //
    // Base case (r == 1):
    //   A_1^L[k] = ((2L+1)/(L+1)) x[k]     (scalar × x[k])
    //   B_1^L[k] = −L/(L+1)                  (constant)
    //   return  dot(s^L, A_1^L) + dot(s^{L-1}, B_1^L)
    //
    // Recursive case (r > 1):
    //   split r = n1 + n2  (e.g. n1 = floor(r/2), n2 = r - n1)
    //   Recursively compute the 2×2 matrix of coefficient vectors:
    //     [ A_{n1}^L     B_{n1}^L    ]
    //     [ A_{n1-1}^L   B_{n1-1}^L ]
    //   and
    //     [ A_{n2}^{L+n1}     B_{n2}^{L+n1}    ]
    //     [ A_{n2-1}^{L+n1}   B_{n2-1}^{L+n1}  ]
    //   Combine via eq:rs-product (pointwise, O(N) per entry):
    //     A_r^L = A_{n2}^{L+n1} ⊙ A_{n1}^L  +  B_{n2}^{L+n1} ⊙ A_{n1-1}^L
    //     B_r^L = A_{n2}^{L+n1} ⊙ B_{n1}^L  +  B_{n2}^{L+n1} ⊙ B_{n1-1}^L
    //   Then apply eq:basic_projection:
    //     return  dot(s^L, A_r^L) + dot(s^{L-1}, B_r^L)
    //
    // To fill out[] for all l simultaneously, call this for each target l,
    // reusing intermediate A/B matrices that were computed on the way.
    // A memoisation table keyed on (L, r) avoids recomputation.

    // Step C: write results
    for l = 0 to N-1:
        out[l] = compute_proj(0, l, S[0], zero_vector)
        // At L=0: s^0 = S[0], s^{-1} = 0 (P_{-1} = 0 by convention)
        //         A_r^0 = P_r, B_r^0 = 0  (lines 998–999)
        //  so  < s , P_l >  =  < s^0 , P_l >  =  direct sum  (correct as expected)


// ─── ASSEMBLY ─────────────────────────────────────────────────────────────────

// After beta_fft returns fhat[l][m][n] for all (l, m, n), the SU(2) FFT is done.
// Apply the 1/(8π²) prefactor (already folded into s above).
// For half-integer l, the Jacobi parameters α = |m−n|, β = |m+n| change; the
// recursion structure is identical.
```

---

## 6. Key Implementation Notes for FLINT/Arb

1. **2-D DFT (Stage 1):** Use `acb_dft` on each θ-slice. For N up to a few thousand,
   FLINT's arbitrary-precision DFT is appropriate; for large N you may prefer FFTW
   and convert.

2. **Jacobi polynomial recurrence (general m, n):** The three-term recurrence for
   `P^l_{nm}` has the same structure as eq:P_l (line 739) with coefficients that
   depend on l, m, n. FLINT provides `arb_hypgeom_jacobi_p` for individual
   evaluations; for the recurrence you can compute coefficients analytically.

3. **Shifted coefficient vectors A_r^L, B_r^L:** Each is a length-N vector of
   real numbers (evaluated at `x[k] = cos(theta_k)`). Store as `arb_ptr` arrays.
   The composition step (eq:rs-product) is four Hadamard products and two
   pointwise sums, all O(N).

4. **Memoisation:** The recursive splitting of r visits O(r log r) distinct
   (L, r) pairs. Cache the 2×2 matrix of N-vectors for each visited pair.

5. **Precision:** The paper works with exact formulas. In Arb, carry enough bits
   (say, 2 × log2(N) extra guard bits) to ensure the final dot products are
   accurate.

6. **Half-integer l:** Wigner d-matrix entries `P^l_{nm}` for half-integer l are
   Jacobi polynomials `P_k^{(α,β)}` with α = |m−n|, β = |m+n|, k = l − max(|m|,|n|).
   The recurrence and composition rules carry over verbatim.

---

## 7. Equation / Label Reference Table

| Label | Location | Content |
|-------|----------|---------|
| `Core_eq` | line 717 | Main DLT sum: `f̂(l)_{mn} ≈ Σ_k [s]_k P^l_{nm}(cos θ_k)` |
| `eq:P_l` | line 739 | Three-term Legendre recurrence |
| `eq:rodrigues` | line 748 | Classical Bonnet formula `(n+1)P_{n+1} − (2n+1)xP_n + nP_{n-1} = 0` |
| `CAP6-8` | line 773 | Shifted representation `P_{L+r} = A_r^L P_L + B_r^L P_{L-1}` |
| `eq-eq` | line 781 | Same as CAP6-8 (proof body) |
| `MATRIZ_LEGEN1` | line 880 | Single-step matrix form of the Legendre recurrence |
| `EQmatrix1` | line 904 | Matrix recurrence for the (A, B) coefficient matrices |
| `eq:composition` | line 981 | Composition rule (2×2 matrix product of A/B matrices) |
| `eq:rs-product` | line 1025 | Scalar expansion of the matrix product |
| `eq:basic_projection` | line 1118 | `< s, P_{L+r} > = < s^L, A_r^L > + < s^{L-1}, B_r^L >` |
| `eq:general_decomposition` | line 1138 | Full binary-tree expansion for r = n_1+…+n_k |
