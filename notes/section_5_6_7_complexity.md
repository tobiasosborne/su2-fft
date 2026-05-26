# Complexity Analysis: Sections 5–7 of "On The Fast Fourier Transform on SU(2)"

Source: `/home/tobias/Projects/su2-fft/paper.tex` (arXiv 2605.23923).
All line numbers refer to that file.

---

## 0. What "N" means

N is the **bandlimit**: the function f : SU(2) → C is assumed bandlimited to degree N,
meaning all Fourier coefficients ĥf(l)_{mn} with l ≥ N vanish. Concretely:

- l runs from 0 to N−1.
- The Euler-angle grid is taken to be N × N × N (each angle discretised with N points).

Both uses coincide, so the same symbol N controls both the truncation level and the sample
count per angle.

---

## 1. Direct FT Complexity (Section 5, lines 1299–1326)

**Theorem [Direct Computation Complexity]** (line 1304–1306)

> "If the Fourier coefficients ĥf(l)_{mn} are approximated via numerical quadrature on a
> discretized grid of size N³, the total computational complexity of the direct method is
> O(N⁶)."

### Factor breakdown (proof, lines 1308–1323)

**Number of coefficients** (lines 1309–1311):

    sum_{l=0}^{N-1} (2l+1)^2 = sum_{l=0}^{N-1} (4l² + 4l + 1) = O(N³)

The (2l+1)² factor counts the (m,n) index pairs for representation l (matrix of size
(2l+1)×(2l+1)), summed over l = 0, …, N−1.

**Cost per coefficient** (lines 1314–1318):

    ĥf(l)_{mn} ≈ sum_{i=1}^{N³} f(g_i) · conj(t^l_{mn}(g_i))

This is a sum over the N³ quadrature points on the Euler-angle grid (φ, θ, ψ each with N
points), costing O(N³) per coefficient (assuming O(1) per basis function evaluation).

**Combined** (lines 1320–1323):

    O(N³) [num. coeffs] × O(N³) [cost per coeff] = O(N⁶)

The paper gives no explicit leading constant; the result is purely big-O.

---

## 2. FFT Complexity (Section 6, lines 1330–1496)

The algorithm splits into two stages.

### Stage 1 — 2D FFT over (φ, ψ) [Proposition P_1, lines 1335–1358]

The key intermediate quantity (line 1347):

    f₂(θ_k; m, n) := sum_{j1=0}^{N-1} sum_{j2=0}^{N-1} f(φ_{j1}, θ_k, ψ_{j2}) e^{inφ_{j1}} e^{imψ_{j2}}

For each of the N values of θ_k this is a 2D FFT on an N×N grid, costing O(N² log N).
Running over all N values of θ:

    Cost_Stage1 = N × O(N² log N) = **O(N³ log N)**   (lines 1352–1357)

### Stage 1 reduced problem — eq. EQ_OP_1 (lines 1361–1365)

After Stage 1 the problem reduces to computing inner products:

    ĥf(l)_{mn} ≈ sum_{k=0}^{N-1} [s]_k P^l_{nm}(cos θ_k) = <s, P^l_{nm}>

where [s]_k = f₂(θ_k) sin(θ_k). This must be done for all (l, m, n).

### Stage 2 — Recursive Legendre Transform [Proposition P_2, lines 1369–1451]

The recurrence expands each inner product via a shift of r steps decomposed into k
sub-shifts n₁, …, n_k (line 1370):

    <s, P_{L+r}> = sum_{τ1 ∈ {A,B}} sum_{ε1..ε_{k-1} ∈ {0,1}} <s^{L*(τ1)}, V_{τ1,ε1,…,ε_{k-1}}>

Three-step cost analysis (lines 1387–1449):

1. **Number of terms**: 2 · 2^{k-1} = **2^k**  (line 1392)
2. **Cost per term**: each V is formed by (k−1) Hadamard products of length-N vectors
   (cost O(kN)), plus an O(N) inner product → **O(kN)** per term  (lines 1412–1439)
3. **Cost per coefficient**: 2^k · O(kN) = O(2^k k N)

Multiplied over O(N³) total coefficients (line 1447):

    Cost_Stage2 = **O(2^k k N⁴)**

### Combined — Theorem TEO_FIN [Optimal Algorithm Complexity] (lines 1455–1486)

    T(k) = C₁ N³ log N + C₂ · 2^k k N⁴       (line 1470)

The factor h(k) = k·2^k is strictly increasing for k > 0 (logarithmic derivative
ln2 + 1/k > 0, line 1474). Hence the minimum is at **k = 1**:

    T(1) = O(N³ log N) + O(2N⁴) = **O(N⁴)**   (line 1484)

### The logarithmic-factor variant — Corollary (lines 1490–1496)

Choosing k ≈ log₂ log N (so that 2^k ≈ log N) gives:

    2^k k N⁴ ≈ (log N)(log₂ log N) N⁴ = **O(N⁴ log N log log N)**

This is **worse** than O(N⁴) and is presented as a cautionary result, not the headline.

**Headline result**: O(N⁴) at k = 1.

---

## 3. Comparison Table and Figure (Section 7, lines 1506–1620)

### Table `tab:comparison` (lines 1520–1539)

Caption (line 1522): "Comparison between the number of operations required by the direct
FT on SU(2) (O(N⁶)) and the proposed FFT-based method for various values of k."

| Bandlimit N       | FT ops O(N⁶)          | FFT ops O(N⁴), k=1    | FFT ops, k=log₂ log₂ N |
|-------------------|-----------------------|-----------------------|------------------------|
| 2^10 = 1024       | ≈ 1.15 × 10^18        | ≈ 1.07 × 10^12        | ≈ 3.55 × 10^13         |
| 2^12 = 4096       | ≈ 4.72 × 10^21        | ≈ 2.82 × 10^14        | ≈ 1.21 × 10^16         |
| 2^14 = 16384      | ≈ 1.94 × 10^25        | ≈ 7.44 × 10^16        | ≈ 3.96 × 10^18         |
| 2^16 = 65536      | ≈ 7.92 × 10^28        | ≈ 1.96 × 10^19        | ≈ 1.25 × 10^21         |

(lines 1531–1537)

### Wall-clock illustration at N = 1024 (lines 1541–1563), assuming 1 GFlop/s:

- Direct FT: 1.15 × 10^9 seconds ≈ **36.5 years**
- FFT k=1:  1.07 × 10^3 seconds ≈ **18 minutes**
- FFT k=log₂ log₂ N: 3.55 × 10^4 seconds ≈ **9.86 hours**

### Figure `fig:ftvsfft` (lines 1575–1613)

A log-log plot of theoretical FLOPs vs. bandwidth N (domain N = 4 to 64), three curves:

- **Red solid** (mark=square): Direct FT, O(N⁶), plotted as x^6
- **Blue solid** (mark=circle): Base complexity O(N⁴), plotted as x^4
- **Green dashed** (mark=triangle): O(N⁴ log N log log N), plotted as
  x^4 · (ln x / ln 2) · (ln(ln x / ln 2) / ln 2)

Caption (line 1612): "The red line shows the prohibitive scaling of the direct computation
(O(N⁶)). The blue line represents the asymptotic O(N⁴) complexity. The green dashed line
accounts for the logarithmic factors in the recursive steps, following O(N⁴ log N log log N),
which remains significantly more efficient than the direct method for large N."

---

## 4. Summary

| Quantity              | Result                                | Key line(s)  |
|-----------------------|---------------------------------------|--------------|
| Direct FT             | O(N⁶)                                | 1305, 1322   |
| Stage 1 (2D FFT)      | O(N³ log N)                          | 1356         |
| Stage 2 (rec. Legendre) at depth k | O(2^k k N⁴)           | 1449         |
| Combined T(k)         | C₁ N³ log N + C₂ · 2^k k N⁴         | 1470         |
| Optimal (k=1)         | O(N⁴)   ← headline result            | 1484         |
| Suboptimal (k~log log N) | O(N⁴ log N log log N)             | 1493         |

No explicit leading constant is given for any formula; all bounds are big-O.
The O(N⁴ log N log log N) appearing in the figure caption (line 1612) is the k = log₂ log N
variant, explicitly inferior to the k=1 headline result O(N⁴).
