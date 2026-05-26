# Wigner Small-d Three-Term Recurrence in l

**Target:** replace per-(l,m,n,k) calls to `wigner_d_phys` in `su2_fft.c` Stage 2
with an O(1)-per-step ascending-l sweep; reduces Stage 2 from O(N^5) to O(N^4).

---

## 1. Jacobi Connection (primary source)

For fixed (m, n) and beta, set `M = max(|m|,|n|)`, `N = min(|m|,|n|)`,
`a = |m-n|`, `b = |m+n|`, `k = l - M`. Then (Wikipedia, "Wigner D-matrix",
§"Relation to Jacobi polynomials"):

```
d^l_{n,m}(beta) = sigma * sqrt[(l+M)!(l-M)! / ((l+N)!(l-N)!)]
                * sin(beta/2)^a * cos(beta/2)^b
                * P_{l-M}^{(a,b)}(cos beta)
```

`sigma = (-1)^{(m-n-|m-n|)/2}` is constant in l for fixed (m,n). The trig
prefactor and sign are l-independent; only the factorial weight R(l) and the
Jacobi index k = l-M change with l.

---

## 2. Recurrence Formula

### 2a. Jacobi recurrence (DLMF 18.9.1–18.9.2, exact)

```
P_{k+1}^{(a,b)}(x) = (A_k x + B_k) P_k^{(a,b)}(x) - C_k P_{k-1}^{(a,b)}(x)

A_k = (2k+a+b+1)(2k+a+b+2) / [2(k+1)(k+a+b+1)]
B_k = (a^2 - b^2)(2k+a+b+1) / [2(k+1)(k+a+b+1)(2k+a+b)]   [= 0 when |m|=|n|]
C_k = (k+a)(k+b)(2k+a+b+2) / [(k+1)(k+a+b+1)(2k+a+b)]
```

`P_0 = 1`, `P_1 = ((a+b+2)/2) x + (a-b)/2`.

### 2b. Lifted recurrence for d^l_{n,m}(beta)

Multiply the Jacobi step by the ratio of normalisation weights:

```
F1 = sqrt[(l+1+M)(l+1-M) / ((l+1+N)(l+1-N))]         = R(l+1)/R(l)
F2 = F1 * sqrt[(l+M)(l-M) / ((l+N)(l-N))]             = R(l+1)/R(l-1)
```

The recurrence for k = l-M (derived from DLMF 18.9.2 + Jacobi connection):

```
d^{l+1}_{n,m}(beta) = (A_k cos(beta) + B_k) * F1 * d^l_{n,m}(beta)
                     -  C_k * F2 * d^{l-1}_{n,m}(beta)
```

**Verification m=n=0:** a=b=0, M=N=0, F1=F2=1. Coefficients reduce to
`A_k = (2l+1)/(l+1)`, `B_k=0`, `C_k = l/(l+1)`, giving
`(l+1) P_{l+1} = (2l+1) cos P_l - l P_{l-1}` — matches paper.tex `eq:P_l` (line 739). ✓

### 2c. Edmonds / Varshalovich form (for reference; primary text not accessed)

Edmonds (1957) §4.5.4 and Varshalovich et al. (1988) §4.8.2 state:

```
(j+1) sqrt[j^2-m^2] sqrt[j^2-n^2] * d^{j+1}_{n,m}
= (2j+1)(j^2 cos(beta) - mn) * d^j_{n,m}
- j sqrt[(j+1)^2-m^2] sqrt[(j+1)^2-n^2] * d^{j-1}_{n,m}
```

The form in §2b (derived from accessible sources) is equivalent and preferred
for implementation.

---

## 3. Base Cases

`lmin = M = max(|m|,|n|)`. At k=0 (l=M): `P_0^{(a,b)} = 1`, so

```
d^M_{n,m}(beta) = sigma * sqrt[(2M)! / ((M+N)!(M-N)!)] * sin(beta/2)^a * cos(beta/2)^b
```

One trig term; O(1). At k=1 (l=M+1): two terms. In both cases `wigner_d_phys`
computes the correct value with O(1) loop iterations (tmin=tmax or tmin=tmax-1).
Recommended: seed with two `wigner_d_phys` calls, then run the §2b recurrence.

---

## 4. Singularity Analysis

Three potential singularities in §2b:

1. **`(l+N)(l-N)` in F2 denominator, zeroes at l=N=min(|m|,|n|):** loop starts
   at l=M+1 >= N+1, so l=N is never reached. Safe.

2. **`(l+M)(l-M)` in F2 numerator, zeroes at l=M:** F2=0, zeroing the d^{l-1}
   term. Correct — d^{M-1} does not exist. Non-singular.

3. **`s = 2k+a+b` in B_k, C_k denominators, zeroes at k=0, a=b=0 (m=n=0):**
   loop starts at k=1 (l=1=M+1 for M=0) where s=2. The k=0 step is absorbed
   into the seed `d_curr = wigner_d_phys(1, 0, 0, beta) = cos(beta)`. Safe.

**Conclusion:** non-singular for all l >= M+1 when seeded at l = M and M+1.

---

## 5. Numerical Stability

Forward ascending recurrence is the stable direction for Jacobi polynomials:
the dominant solution grows with k (DLMF §18.2(iii)), so rounding errors do
not amplify. Same regime as the standard Legendre forward recurrence.

- Risbo (1996), *J. Geodesy* 70:383–396, §3: explicitly states the ascending-l
  recurrence for D-functions is numerically stable.
- Edmonds (1957) §4.5: uses ascending recurrence without instability caveat.
- `su2_wigner.c` notes de Moivre stable to l~25; recurrence inherits similar range.

---

## 6. Paper Context

`paper.tex` (arXiv 2605.23923) derives the recurrence only for m=n=0 (lines
747–766, `eq:P_l`). Line 1295 states Jacobi polynomials "also satisfy a
recurrence relation" but gives no coefficients. The paper's P^l_{nm} differs
from d^l_{n,m} by the phase i^{m-n} (`su2_wigner.c` header); the recurrence
operates on the real d^l_{n,m} and the phase is applied once per (m,n) sweep.

---

## 7. Implementation Sketch

```c
/* Per (m, n, beta): sweep l from M to N-1 */
int M = MAX(abs(m),abs(n)), Nm = MIN(abs(m),abs(n));
int a = abs(m-n), b = abs(m+n);
double x = cos(beta);
double d_prev = wigner_d_phys(M,   n, m, beta);  /* seed l=M   */
double d_curr = wigner_d_phys(M+1, n, m, beta);  /* seed l=M+1 */
use(M, d_prev); use(M+1, d_curr);
for (int l = M+1; l <= N-2; ++l) {
    int k = l-M;  double s = 2*k+a+b;
    double Ak = (s+1)*(s+2)/(2.*(k+1)*(k+a+b+1));
    double Bk = (a*a-b*b)*(s+1)/(2.*(k+1)*(k+a+b+1)*s);  /* 0 if a==b */
    double Ck = (k+a)*(k+b)*(s+2)/((k+1.)*(k+a+b+1)*s);
    double F1sq = (double)((l+1+M)*(l+1-M))/((l+1+Nm)*(l+1-Nm));
    double F2sq = F1sq * (double)((l+M)*(l-M))/((l+Nm)*(l-Nm));
    double d_next = (Ak*x+Bk)*sqrt(F1sq)*d_curr - Ck*sqrt(F2sq)*d_prev;
    d_prev = d_curr; d_curr = d_next;
    use(l+1, d_curr);
}
```

Cost: O(1) per l step; O(N-M) per (m,n,beta) triple.
