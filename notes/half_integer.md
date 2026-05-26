# Half-Integer l Support — Scoping Brief

**Bead:** `su2fft-n8e`
**Status:** Scoping only. No code change in this document.

---

## 1. Mathematical Extension

The paper defines the basis functions for all $l \in \tfrac{1}{2}\mathbb{N}_0$
(paper lines 530–543). The Rodrigues-form $P^l_{nm}$ involves factorials of
$l \pm m$ and $l \pm n$; for half-integer $l$, $m$, $n$ those arguments are
half-integers, so $k!$ must be replaced by $\Gamma(k+1)$.

The de Moivre / Wigner sum (Sakurai convention, used throughout `su2_wigner.c`) becomes:

```
d^l_{n,m}(beta) = sum_t (-1)^{n-m+t}
    * sqrt( Gamma(l+n+1) Gamma(l-n+1) Gamma(l+m+1) Gamma(l-m+1) )
    / [ Gamma(l+m-t+1) Gamma(t+1) Gamma(n-m+t+1) Gamma(l-n-t+1) ]
    * cos(beta/2)^{2l+m-n-2t}
    * sin(beta/2)^{n-m+2t}
```

For half-integer $l$, $m$, $n$ (with $2l, 2m, 2n \in \mathbb{Z}$ and all of the
same parity): the summation index $t$ still steps by 1 (i.e., $t$ is an integer),
running over the range where all Gamma arguments are positive. The structure of the
sum is unchanged; only `fact()` becomes `tgamma()` (or `lgamma()` for stability).

The phase $i^{m-n}$ (paper line 542, `su2_wigner.c` header): for half-integer $m$,
$n$ with the same parity, $m - n$ is an integer, so `pow_i(m-n)` is well-defined.
No change needed there.

**Wigner recurrence (bead m21):** the Jacobi connection in `notes/wigner_recurrence.md`
§1 sets $a = |m-n|$ and $b = |m+n|$. For half-integer $m, n$ of matching parity,
$a$ and $b$ are non-negative integers. The DLMF 18.9.1–18.9.2 recurrence and the
lifted form in §2b are structurally identical. Only the normalisation factor $R(l)$
involves Gamma (via the factorial weights) rather than plain factorials. The seeded
recurrence therefore extends to half-integer $l$ without structural change.

**GL theta nodes (bead ega):** GL quadrature integrates polynomials in $\cos\theta$
exactly up to degree $2N-1$. The half-integer Wigner-d functions contain the same
$\sin(\beta/2)^a \cos(\beta/2)^b$ prefactor; after substitution $x = \cos\theta$,
the integrand is still a polynomial in $x$ up to a fixed trig factor. GL nodes
remain the correct quadrature for the theta direction.

---

## 2. Grid Change (the Hard Part)

**Integer case.** The closed phi/psi grid uses
$\phi_j = -\pi + j \cdot 2\pi/(N-1)$, $j \in [0, N-1]$.
For integer $n$, $e^{-in\phi}$ is $2\pi$-periodic and the closed-grid FFTW fold
trick in `su2_fft.c` Stage 1 works correctly.

**Half-integer case.** For $n = p + \tfrac{1}{2}$ ($p$ integer),
$e^{-in\phi}$ has period $4\pi$, not $2\pi$. The function takes opposite signs at
$\phi = 0$ and $\phi = 2\pi$: the grid is not periodic on $[0, 2\pi)$.
The existing $N$-point closed grid on $[0, 2\pi)$ cannot represent half-integer
modes without aliasing.

**Options:**

A. **Extend to $[0, 4\pi)$:** use a closed grid of $2N$ points in phi and psi.
   Doubles the phi/psi sample count. The FFTW plan size doubles to $(2N-1)^2$;
   the Stage 1 cost increases by $4\times$. The fold trick must be reworked for
   a $4\pi$-period.

B. **Half-shift reformulation:** represent half-integer $n$ as "integer $n$" on a
   shifted grid $\phi_j = j \cdot 2\pi/(2N-1)$. Equivalent to A in cost but
   more algebraically compact.

C. **Separate pure-integer and pure-half-integer APIs:** when $2l \in 2\mathbb{Z}$
   (integer $l$), all indices $m, n$ are integers — use the existing integer-l
   functions. When $2l \in 2\mathbb{Z}+1$ (half-integer $l$), all indices are
   half-integers. Implement a separate function family for the half-integer case
   with its own $4\pi$-period grid.

**Recommendation: Option C.** Pure separation is the cleanest first cut. A user
requesting the half-integer FFT calls `su2_fft_half`; integer users see no change.
No mixed-parity bookkeeping. The grid change is localised to the new function family.

---

## 3. API Design — Minimum Viable First Cut

Two tiers of scope, ordered from smallest to largest:

### Tier 1 (recommended first PR): Wigner-d only, ~100 LOC

Add two functions to `src/su2_wigner.c` / `include/su2.h`:

```c
/* Half-integer Wigner-d evaluation.
 * l, m, n encoded as 2l, 2m, 2n (always odd integers for half-integer values).
 * Physical values: l_phys = two_l / 2.0, etc.
 * Returns d^l_{n,m}(theta) (Sakurai convention, real).
 * Caller applies i^{m-n} phase to obtain paper's P^l_{nm}.
 */
double su2_wigner_d_half(int two_l, int two_n, int two_m, double theta);

/* Sequence via ascending-l recurrence; mirrors su2_wigner_d_seq.
 * two_l_min, two_l_max: both odd; two_l_max >= two_l_min >= max(|two_n|, |two_m|).
 * out_d: buffer of length (two_l_max - two_l_min)/2 + 1 doubles.
 */
void su2_wigner_d_seq_half(int two_l_min, int two_l_max,
                           int two_n, int two_m, double theta,
                           double *out_d);
```

The key changes in the implementation of `su2_wigner_d_half` vs `wigner_d_phys`:
- Replace `fact(k)` with `tgamma(k + 1)` (or precomputed log-gamma for speed).
- The "integer power" `ipow(c2, pc)` and `ipow(s2, ps)` exponents `pc`, `ps` become
  half-integers in general; replace with `pow(c2, pc)` and `pow(s2, ps)` (libm).
  (Integer `ipow` cannot handle non-integer exponents; this is a real cost regression
  for seeds, but seeds are $O(N^3)$ calls total post-m21, so it is acceptable.)
- The recurrence in `su2_wigner_d_seq_half` uses the same Jacobi DLMF 18.9.1–18.9.2
  coefficients but with $l, M, N_{min}$ taking half-integer values. Since $F_1, F_2$
  involve ratios of Gamma/factorial, replace the integer arithmetic with
  `tgamma` or half-integer-aware factored forms.

Concrete test (HANDOFF.md §7): for $l = 1/2$, $m = n = 1/2$,
$d^{1/2}_{1/2, 1/2}(\theta) = \cos(\theta/2)$. Check this to $10^{-12}$.
Similarly $d^{1/2}_{-1/2, 1/2}(\theta) = -\sin(\theta/2)$.

### Tier 2 (follow-on PR): Full half-integer FFT, ~500 LOC

```c
/* Half-integer FFT. Bandlimit: l in {1/2, 3/2, ..., (2N-1)/2}.
 * f: length (2N) * N * (2N) complex samples.
 *    phi, psi on [0, 4pi) closed grid with 2N points; theta: N GL nodes.
 * fhat: half-integer coefficient block, length su2_total_coeffs_half(N).
 */
void su2_fft_half(int N, const double _Complex *f, double _Complex *fhat);
void su2_fft_half_inv(int N, const double _Complex *fhat, double _Complex *f);
```

This requires: new grid functions for the $4\pi$-period phi/psi, a `(2N-1) \times (2N-1)$
FFTW plan (Stage 1), `su2_wigner_d_seq_half` in Stage 2, and new coefficient layout
helpers (since $(2l+1)$ is now an even integer). Coefficient offset arithmetic,
storage layout, and the `(-1)^n$ fold phase must all be rederived for the half-period
case. This is a substantial, error-prone change that deserves its own bead and test suite.

---

## 4. Scope Recommendation

**Ship Tier 1 (Wigner-d only) as bead `su2fft-n8e`.** Defer Tier 2 (full FFT) to a
new bead.

Rationale:
- The de Moivre sum change (fact -> tgamma) is mechanical and fully testable against
  known closed forms ($d^{1/2}$ is two-by-two, exact).
- The recurrence extension is structurally unchanged; verification is cheap.
- Tier 2 requires redesigning the phi/psi grid, the FFTW fold phase, and the
  coefficient storage layout — three independent invariants each easy to get subtly
  wrong. A full half-integer FFT is a 500–1000 LOC change across four files; that
  is too large for a reviewable single PR.
- The HANDOFF.md §7 acceptance criterion for `su2fft-n8e` (roundtrip on
  $u_{00}(g) = e^{i(\phi+\psi)/2}\cos(\theta/2)$ to $10^{-10}$) formally requires
  the full FFT. However, the Wigner-d tier already unblocks external users needing
  spin-$1/2$ matrix elements, and the FFT tier can follow immediately once the
  Wigner layer is verified.

---

## 5. Dependencies on Shipped Work

| Bead | Relevance to half-integer |
|------|--------------------------|
| m21  | Recurrence structurally unchanged for half-integer $l$; only $F_1, F_2$ need Gamma. |
| ega  | GL quadrature in $\cos\theta$ works identically; no change needed. |
| 0t1  | Phi/psi aliasing is equally present (and worse) for half-integer; fix is prerequisite for Tier 2. |
| 3lx  | Inverse FFT structure mirrors forward; Tier 2 extends both. |

---

## 6. Test Plan (when implemented)

Tier 1 (Wigner-d):
- $d^{1/2}_{1/2,1/2}(\theta) = \cos(\theta/2)$: verify to $10^{-12}$ at 100 random theta.
- $d^{1/2}_{-1/2,1/2}(\theta) = -\sin(\theta/2)$: same.
- Unitarity: $\sum_n [d^{1/2}_{n,m}(\theta)]^2 = 1$ for $m = \pm 1/2$.
- Recurrence agrees with direct de Moivre sum at $l = 1/2, 3/2, 5/2$ to $10^{-14}$.

Tier 2 (FFT):
- Spin-$1/2$ roundtrip: $f(g) = e^{i(\phi+\psi)/2}\cos(\theta/2)$ (paper line 503,
  the $(0,0)$ entry of the defining representation); verify $\hat{f}(1/2)_{00} = 1$
  to $\sim 10^{-10}$.
- Unitarity of the $(2\times2)$ coefficient block at $l=1/2$.
- Forward(inverse(fhat)) roundtrip at GL theta, subject to phi/psi aliasing floor.

---

## 7. Out-of-Scope Follow-Ups (file new beads when needed)

- Mixed integer/half-integer API (single call handles both parities).
- $4\pi$-native phi/psi grid that avoids the separate-function-family split.
- Half-integer arb-precision path (extend `su2_fft_arb` family).
- Julia bindings for half-integer functions.
