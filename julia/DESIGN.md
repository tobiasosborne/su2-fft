# Julia bindings for libsu2 — design brief
# Bead: su2fft-t6z

## 1. Directory layout

```
su2-fft/
  julia/
    Project.toml          # package metadata; deps: stdlib + Test only
    DESIGN.md             # this file
    src/
      SU2FFT.jl           # module: ccall wrappers, accessors
    deps/
      build.jl            # shells out to `make libsu2.so`, writes deps.jl
    test/
      runtests.jl         # gold-standard cross-check suite
```

## 2. Library linkage strategy

Build target `build/libsu2.so` (added to Makefile, see Deliverable B).
Object files compiled with `-fPIC` so they can be linked into a shared
library.

`deps/build.jl`:
- Runs `make -C <project_root> libsu2.so` via `Cmd`.
- `<project_root>` is computed as `dirname(dirname(@__DIR__))` (two levels
  up from `julia/deps/`).
- Writes the absolute path to `deps/deps.jl`:
  ```julia
  const LIBSU2 = "/abs/path/to/su2-fft/build/libsu2.so"
  ```

`src/SU2FFT.jl`:
- `include("../deps/deps.jl")` near the top of the module.
- All ccalls reference `LIBSU2`.

Portability caveat: this design is Linux-only (`.so`, local build).
Portable distribution requires BinaryBuilder.jl + Yggdrasil. File as a
follow-up bead; do not attempt in this cut.

## 3. Memory layout marshalling

### Samples `f`

C storage (su2.h:42):
```
f[j1*N*N + k*N + j2]    -- row-major in (j1, k, j2)
```
Linear index for C element at (j1, k, j2):
```
idx_C = j1*N^2 + k*N + j2
```

Julia `Array{ComplexF64,3}` with shape `(N, N, N)` is column-major.
Linear index for Julia element `A[i1, i2, i3]` (1-based):
```
idx_J = (i1-1) + N*(i2-1) + N^2*(i3-1)
```

Matching `idx_C == idx_J` with `i1 = j2+1, i2 = k+1, i3 = j1+1`:
```
idx_C = j1*N^2 + k*N + j2
      = (i3-1)*N^2 + (i2-1)*N + (i1-1)
      = (i1-1) + N*(i2-1) + N^2*(i3-1) = idx_J  -- check
```

Convention: the Julia caller constructs the array with
```julia
f[j2+1, k+1, j1+1] = sample at Euler-grid index (j1, k, j2)
```
This passes without any permutation or copy to the C ccall.

Alternative considered: `permutedims(f, (3,2,1))` before each ccall.
Rejected: doubles memory traffic on every call; callers can build the
array in the correct layout natively.

### Coefficients `fhat`

C storage: flat `Vector{ComplexF64}` of length `su2_total_coeffs(N)`.

Degree-l block starts at `su2_coeff_offset(l)` (su2.h:30).
Within block, element at (m, n) with m, n in [-l, l]:
```
offset = su2_coeff_offset(l) + (m+l)*(2l+1) + (n+l)    # su2.h:38
```
Block size: `(2l+1)^2`. Total: `sum_{l=0}^{N-1} (2l+1)^2`.

Julia representation: a flat `Vector{ComplexF64}` is passed directly.
Accessor and block functions handle index arithmetic.

`fftw_complex` is layout-compatible with `double _Complex`, which is
layout-compatible with Julia `ComplexF64` (HANDOFF.md §6 item 4).
No reinterpretation needed.

## 4. Public API

Functions exported from module `SU2FFT`:

```julia
SU2FFT.fft(f::Array{ComplexF64,3}) -> Vector{ComplexF64}
    # O(N^4) fast path; wraps su2_fft

SU2FFT.ft_direct(f::Array{ComplexF64,3}) -> Vector{ComplexF64}
    # O(N^6) reference; wraps su2_ft_direct

SU2FFT.wigner_d(l::Int, n::Int, m::Int, theta::Float64) -> ComplexF64
    # Returns paper's P^l_{n,m}(cos theta) = i^{m-n} * d^l_{n,m}(theta)
    # wraps su2_wigner_d

SU2FFT.total_coeffs(N::Int) -> Int
    # wraps su2_total_coeffs

SU2FFT.coeff_offset(l::Int) -> Int
    # wraps su2_coeff_offset

SU2FFT.fhat_at(fhat::Vector{ComplexF64}, l::Int, m::Int, n::Int) -> ComplexF64
    # returns fhat at degree l, indices m, n in [-l, l]
    # pure Julia arithmetic, no ccall

SU2FFT.fhat_block(fhat::Vector{ComplexF64}, l::Int) -> AbstractMatrix{ComplexF64}
    # returns (2l+1) x (2l+1) view of the degree-l block
    # implemented as reshape(view(fhat, off+1:off+(2l+1)^2), 2l+1, 2l+1)
    # row-major in C, so column j of the Julia matrix = row j of C matrix;
    # document this transposition in docstring

SU2FFT.libsu2_path() -> String
    # returns LIBSU2 for debugging linkage
```

Not exposed in this cut (file follow-up beads for each):
- `su2_fft_arb`, `su2_ft_direct_arb` — require Arblib.jl.
- `su2_wigner_d_seq` — internal recurrence helper, not user-facing.

## 5. ccall signatures

`ComplexF64` is layout-compatible with C `double _Complex` (both are
two contiguous `double` values). `Cint` maps to C `int`. `Csize_t`
maps to C `size_t`.

```julia
function fft(f::Array{ComplexF64,3})
    N = size(f, 1)
    @assert size(f) == (N, N, N) "samples array must be (N, N, N)"
    fhat = Vector{ComplexF64}(undef, total_coeffs(N))
    ccall((:su2_fft, LIBSU2), Cvoid,
          (Cint, Ptr{ComplexF64}, Ptr{ComplexF64}),
          N, f, fhat)
    return fhat
end

function ft_direct(f::Array{ComplexF64,3})
    N = size(f, 1)
    @assert size(f) == (N, N, N) "samples array must be (N, N, N)"
    fhat = Vector{ComplexF64}(undef, total_coeffs(N))
    ccall((:su2_ft_direct, LIBSU2), Cvoid,
          (Cint, Ptr{ComplexF64}, Ptr{ComplexF64}),
          N, f, fhat)
    return fhat
end

function wigner_d(l::Int, n::Int, m::Int, theta::Float64)::ComplexF64
    # C signature: double _Complex su2_wigner_d(int l, int n, int m, double theta)
    ccall((:su2_wigner_d, LIBSU2), ComplexF64,
          (Cint, Cint, Cint, Cdouble),
          l, n, m, theta)
end

function total_coeffs(N::Int)::Int
    # C signature: size_t su2_total_coeffs(int N)
    Int(ccall((:su2_total_coeffs, LIBSU2), Csize_t, (Cint,), N))
end

function coeff_offset(l::Int)::Int
    # C signature: size_t su2_coeff_offset(int l)
    Int(ccall((:su2_coeff_offset, LIBSU2), Csize_t, (Cint,), l))
end
```

`fhat_at` and `fhat_block` are pure Julia — no ccall needed:
```julia
function fhat_at(fhat::Vector{ComplexF64}, l::Int, m::Int, n::Int)::ComplexF64
    off = coeff_offset(l)
    fhat[off + (m + l) * (2l + 1) + (n + l) + 1]   # +1: 1-based
end

function fhat_block(fhat::Vector{ComplexF64}, l::Int)
    off = coeff_offset(l)
    d = 2l + 1
    reshape(view(fhat, off+1:off+d*d), d, d)
    # note: C block is row-major; this Julia view is column-major.
    # fhat_block(fhat, l)[col, row] == C fhat[l][(row-1)*(2l+1)+(col-1)]
    # Document in docstring; transpose if the caller wants row-major semantics.
end
```

## 6. Test strategy

Entry point: `test/runtests.jl`, run via `Pkg.test("SU2FFT")`.

1. **Coefficient count.**
   `total_coeffs(N) == sum((2l+1)^2 for l in 0:N-1)` for N = 1, 2, 4, 8.
   Explicit: N=4 -> 1+9+25+49 = 84.

2. **Zero input.**
   `fft(zeros(ComplexF64, N, N, N)) ≈ zeros(ComplexF64, total_coeffs(N))`
   at N=4 (fast, ~1ms).

3. **Gold-standard cross-check.**
   Random `f = randn(ComplexF64, N, N, N)` (memory layout: `f[j2+1, k+1, j1+1]`
   is the sample at grid point (j1, k, j2)).
   `norm(fft(f) - ft_direct(f)) / norm(ft_direct(f)) < 1e-10` at N=6 and N=8.
   Mirrors `tests/test_fft.c::test_fft_matches_direct_random`.

4. **Wigner invariant.**
   `wigner_d(0, 0, 0, 0.5) ≈ 1.0 + 0.0im` (P^0_{0,0} = 1 for all theta).

5. **Offset consistency.**
   `coeff_offset(0) == 0`.
   `coeff_offset(l+1) == coeff_offset(l) + (2l+1)^2` for l = 0..4.

## 7. Out of scope — file as new beads later

Do not file these now; they are recorded here for the next step.

- Arblib.jl bindings for `su2_fft_arb` and `su2_ft_direct_arb`.
- BinaryBuilder.jl / Yggdrasil for portable `.so`/`.dylib` distribution.
- macOS `.dylib` support (`.so` -> `.dylib`, different rpath).
- Registration to Julia General registry.
- Inverse FFT bindings — blocked on C bead `su2fft-3lx`.
- Half-integer l support — blocked on C bead `su2fft-n8e`.
