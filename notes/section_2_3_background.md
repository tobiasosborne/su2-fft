# Implementation-Oriented Summary: Sections 2–3 of "On The Fast Fourier Transform on SU(2)" (arXiv 2605.23923)

Source lines cited are from `/home/tobias/Projects/su2-fft/paper.tex`.

---

## 1. Section 2: Preliminaries

### 1.1 Torus Fourier Transform

The n-dimensional torus is T^n = S^1 × … × S^1 (n copies), identified with [0,1)^n  (lines 380–385).

Irreducible representations of T^n are the characters indexed by k ∈ Z^n (lines 390–394):

    χ_k(x) = exp(2πi k·x),   x ∈ T^n,  k ∈ Z^n.

The continuous Fourier transform and its inversion (lines 396–408):

    f(x)    = Σ_{k ∈ Z^n}  f̂(k) exp(2πi <k, x>)
    f̂(k)   = ∫_{T^n} f(x) exp(-2πi <k, x>) dx

### 1.2 DFT on T^n

Uniform grid with N points per dimension (lines 412–416):

    x_j = (j_1/N, …, j_n/N),   j = (j_1, …, j_{n-1}) ∈ {0, …, N-1}^n.

The DFT (lines 418–423, Definition):

    f̂(k) := Σ_{j ∈ {0,…,N-1}^n}  f_j  exp(-2πi <k, x_j>),   k ∈ {0,…,N-1}^n.

Direct evaluation costs O(N^{2n}); the n-dimensional FFT (Cooley–Tukey, applied axis-by-axis) reduces this to O(N^n log N)  (lines 426–446).

---

## 2. Section 3: Fourier Analysis on SU(2)

### 2.1 SU(2) as a Lie Group

Definition (lines 457–461):

    SU(2) = { U ∈ M_{2×2}(C) : U†U = I, det(U) = 1 }.

It is a compact, connected, non-abelian Lie group of real dimension 3, diffeomorphic to S^3 ⊂ R^4  (lines 464, 579).

### 2.2 Euler-Angle Parametrisation

Theorem, lines 500–509.  With Euler angles (φ, θ, ψ) — note the paper uses (φ, θ, ψ) and writes the element as u(φ, θ, ψ):

    u(φ, θ, ψ) = ( e^{i(φ+ψ)/2} cos(θ/2)          e^{i(φ-ψ)/2} i sin(θ/2) )
                 ( e^{-i(φ-ψ)/2} i sin(θ/2)         e^{-i(φ+ψ)/2} cos(θ/2) )

Angle ranges (inferred from the Haar integral written later, lines 666–667):
  φ ∈ [-π, π],   θ ∈ [0, π],   ψ ∈ [-π, π].

Later in Section 3 (line 664 onward) the paper switches notation to (φ, θ, ψ) for the same angles; elsewhere it also uses (α, β, γ) — all refer to the same parametrisation.

### 2.3 Irreducible Unitary Representations

Index set (lines 482–492, 532):

    l ∈ (1/2) N_0 = {0, 1/2, 1, 3/2, 2, …}.

For each l, the representation space V_l consists of homogeneous polynomials in (z_1, z_2) of degree 2l; dim(V_l) = 2l+1  (line 488).

Natural basis for V_l (lines 492–495, equation labelled `Polinomio_Plk`):

    p_{lk}(z) = z_1^k z_2^{2l-k},   k = 0, 1, …, 2l.         [label: Polinomio_Plk, line 493]

The representation T_l : SU(2) → GL(V_l) defined by (T_l(u)f)(z) = f(zu) is irreducible and unitary, and (up to equivalence) exhausts all irreducible unitary representations  (lines 488–492).

### 2.4 Matrix Coefficients t^l_{mn}

The matrix coefficients are (lines 510–513, Theorem):

    t^l_{mn}(u) = (d^{l-m}/dz_1^{l-m}) (d^{l+m}/dz_2^{l+m})
                  [ (z_1 a + z_2 c)^{l-n} (z_1 b + z_2 d)^{l+n} ]
                  / sqrt((l-m)!(l+m)!(l-n)!(l+n)!)

Indices run over -l ≤ m, n ≤ l with l-m, l-n ∈ Z  (line 530).  The matrix has size (2l+1)×(2l+1).

In Euler-angle form (lines 533–535):

    t^l_{nm}(φ, θ, ψ) = P^l_{nm}(cos θ) exp(-i(nφ + mψ))

NOTE: The paper uses index order (n,m) in t^l_{nm} on the left but (m,n) in the subscript of f̂(l)_{mn} on the right — this deliberate transposition is flagged at lines 546–546 as necessary for matrix multiplication to be consistent.

### 2.5 The Wigner-d Function / P^l_{nm}

The function P^l_{nm}(x) plays the role of the Wigner small-d matrix element d^l_{nm}(θ) evaluated at x = cos θ. Its explicit formula is (lines 537–543, equation labelled `Expresation P_mn^l`):

    P^l_{nm}(x) = c^l_{nm}  (1-x)^{(m-n)/2} / (1+x)^{(n+m)/2}
                  × (d/dx)^{l-n} [ (1-x)^{l-m} (1+x)^{l+m} ]      [label: Expresation P_mn^l, line 537]

with normalization constant (lines 541–543):

    c^l_{nm} = 2^{-l} (-1)^{l-m} i^{m-n} / sqrt((l-m)!(l+m)!)
               × sqrt((l+n)!/(l-n)!)

(An earlier version with indices (m,n) instead of (n,m) appears at lines 522–526; the (n,m) form at lines 537–543 is the one used throughout the algorithm.)

The case m = n = 0 reduces P^l_{00} to the standard Legendre polynomial P_l  (line 734).

### 2.6 Orthonormality and Peter–Weyl Basis

The normalized set {sqrt(2l+1) t^l_{nm}} forms an orthonormal basis of L^2(SU(2))  (lines 548–551).

Orthonormality relation (lines 562–565):

    <t^l_{nm}, t^l_{n'm'}> = δ_{mm'} δ_{nn'} / (2l+1).

Fourier series reconstruction (lines 552–554):

    f(x) = Σ_{l ∈ (1/2)N_0} (2l+1) Σ_{m,n} f̂(l)_{mn} t^l_{nm}(x).

### 2.7 Continuous Fourier Transform on SU(2) — Exact Definition

Definition (lines 569–575):

    f̂(l)_{mn} := ∫_{SU(2)} f(x)  conj(t^l_{nm}(x))  dx,

    -l ≤ m, n ≤ l,  l - m, l - n ∈ Z.

With Haar measure written in Euler angles (lines 664–675):

    f̂(l)_{mn} = (1/8π²) ∫_{-π}^{π} ∫_0^{π} ∫_{-π}^{π}
                    f(φ, θ, ψ)  P^l_{nm}(cos θ)  exp(i(nφ + mψ))
                    sin(θ)  dφ dθ dψ.

The Haar measure factor is sin(θ) dφ dθ dψ / 8π².

---

## 3. The Discrete FT Formula to Compute Fast (Section 3 / beginning of Section 4)

### 3.1 Sampling Grid

Definition (lines 684–691):

    φ_{j1} = -π + j1 · 2π/(N-1),   j1 = 0, …, N-1
    ψ_{j2} = -π + j2 · 2π/(N-1),   j2 = 0, …, N-1
    θ_k    = k · π/(N-1),           k  = 0, …, N-1

This is a uniform grid in all three angles; N is the bandwidth (f̂(l)_{mn} = 0 for l ≥ N, defined at line 678). There is no Gauss–Legendre quadrature; the θ nodes are uniformly spaced, and the integral over θ is approximated by a Riemann sum with weight sin(θ_k).

### 3.2 The Triple-Sum DFT Formula

Proposition, lines 695–700 (equation inline):

    f̂(l)_{mn} ≈ (1/8π²) Σ_{k=0}^{N-1}  P^l_{nm}(cos θ_k) sin(θ_k)
                 [ Σ_{j1=0}^{N-1} exp(inφ_{j1})  Σ_{j2=0}^{N-1} exp(imψ_{j2})  f(φ_{j1}, θ_k, ψ_{j2}) ]

The bracket is a 2D DFT over (j1, j2) for fixed k.  Denoting the FFT output as f_2(θ_k) [the (n,m)-th mode of the 2D FFT of the slice f(·, θ_k, ·)], this reduces to (lines 710–712):

    f̂(l)_{mn} ≈ (1/8π²) Σ_{k=0}^{N-1}  f_2(θ_k) sin(θ_k)  P^l_{nm}(cos θ_k).

The core inner-product to compute efficiently (equation labelled `Core_eq`, lines 717–720):

    f̂(l)_{mn} ≈ Σ_{k=0}^{N-1}  [s]_k  P^l_{nm}(cos θ_k)  =  <s, P^l_{nm}>   [label: Core_eq, line 717]

where [s]_k = f_2(θ_k) sin(θ_k)  (the 1/8π² is suppressed for notational convenience but must be restored in a full implementation).

---

## 4. Implementation Checklist — Critical Normalisations and Conventions

1. **Haar measure factor**: 1/8π² in front of every Fourier coefficient  (line 666).
2. **Weight in θ-sum**: sin(θ_k) (from Haar measure; NOT a quadrature weight — the grid is uniform)  (lines 697–698).
3. **Index order in t^l**: The paper writes t^l_{nm} (row n, column m), but f̂(l) is indexed as f̂(l)_{mn} (row m, column n). This swap is intentional for matrix-multiplication compatibility  (lines 546–546).
4. **Reconstruction factor**: (2l+1) multiplies each l-block in the synthesis sum  (line 554).
5. **Orthonormal basis**: sqrt(2l+1) t^l_{nm} is the ON basis; the factor (2l+1) in the synthesis compensates  (line 550).
6. **Convention for t^l_{nm}**: t^l_{nm}(φ, θ, ψ) = P^l_{nm}(cos θ) exp(-i(nφ + mψ))  (line 534) — note the minus sign in the exponent.
7. **Fourier coefficient uses conjugate**: f̂(l)_{mn} uses conj(t^l_{nm}) = P^l_{nm}(cos θ) exp(+i(nφ + mψ))  (line 674).
8. **Bandwidth N**: indices l run from 0 to N-1 (in half-integer steps), giving l ∈ {0, 1/2, 1, …, (N-1)/2} or {0, 1, …, N-1} depending on whether only integer or also half-integer representations are needed.
9. **Grid size**: (N-1+1) = N points in each of φ, ψ, θ; the endpoint −π + (N-1)·2π/(N-1) = π is included, so the grid is closed (not periodic-DFT standard). An implementer must decide whether to use this closed uniform grid or shift to the half-open FFT-compatible grid.
10. **Jacobi polynomial formula** [label: Expresation P_mn^l, line 537]: the Rodrigues-type formula involves a (d/dx)^{l-n} derivative of a binomial product; for half-integer l the factorial expressions require the Gamma function.
