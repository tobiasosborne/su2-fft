"""
    SU2FFT

Julia bindings for the libsu2 C library — the SU(2) Fast Fourier Transform
of Delgado et al., arXiv 2605.23923 (May 2026).

Provides ccall wrappers around:
- `su2_fft`         — O(N^4) fast FFT on SU(2)
- `su2_ft_direct`   — O(N^6) reference direct transform
- `su2_wigner_d`    — paper's matrix coefficient P^l_{n,m}(cos theta)
- `su2_total_coeffs`, `su2_coeff_offset` — coefficient layout helpers

Plus pure-Julia accessors `fhat_at`, `fhat_block` over the flat coefficient
storage. Sample storage convention: `f[j2+1, k+1, j1+1]` is the sample at
Euler-grid point (j1, k, j2); see DESIGN.md §3.

The bead corresponding to this binding is `su2fft-t6z`.
"""
module SU2FFT

using Libdl

# Load LIBSU2 (set by build.jl).
const DEPS_FILE = joinpath(@__DIR__, "..", "deps", "deps.jl")
if !isfile(DEPS_FILE)
    error("SU2FFT: deps.jl not found at $DEPS_FILE. Run `Pkg.build(\"SU2FFT\")` first.")
end
include(DEPS_FILE)

# Module-private handle to the dlopen'd libsu2.so. We use RTLD_DEEPBIND so
# libsu2's transitive dependencies (libflint, libgmp, libfftw3) are resolved
# using the system's library search order, not Julia's pre-loaded symbol
# table. This works around the Julia 1.12 bundled-libgmp ABI mismatch
# documented in bead su2fft-e5z.
const _LIBSU2_HANDLE     = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_FFT           = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_FFT_INV       = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_FT_DIRECT     = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_WIGNER_D      = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_TOTAL_COEFFS  = Ref{Ptr{Cvoid}}(C_NULL)
const _SYM_COEFF_OFFSET  = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    # Fallback for the Julia 1.12 bundled-libgmp ABI mismatch: pre-load the
    # system libgmp.so.10 with RTLD_GLOBAL so its symbols are visible before
    # libsu2 and libflint are mapped. On this system RTLD_DEEPBIND on libsu2
    # alone was insufficient because libflint resolves __gmpn_modexact_1_odd
    # against the already-loaded Julia-bundled libgmp; pre-loading the
    # system libgmp keeps its definitions in the global symbol table first.
    try
        Libdl.dlopen("/lib/x86_64-linux-gnu/libgmp.so.10",
                     Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL)
    catch
        # silently ignore on systems without /lib/x86_64-linux-gnu/libgmp.so.10
    end
    flags = Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL | Libdl.RTLD_DEEPBIND
    _LIBSU2_HANDLE[]    = Libdl.dlopen(LIBSU2, flags)
    _SYM_FFT[]          = Libdl.dlsym(_LIBSU2_HANDLE[], :su2_fft)
    _SYM_FFT_INV[]      = Libdl.dlsym(_LIBSU2_HANDLE[], :su2_fft_inv)
    _SYM_FT_DIRECT[]    = Libdl.dlsym(_LIBSU2_HANDLE[], :su2_ft_direct)
    _SYM_WIGNER_D[]     = Libdl.dlsym(_LIBSU2_HANDLE[], :su2_wigner_d)
    _SYM_TOTAL_COEFFS[] = Libdl.dlsym(_LIBSU2_HANDLE[], :su2_total_coeffs)
    _SYM_COEFF_OFFSET[] = Libdl.dlsym(_LIBSU2_HANDLE[], :su2_coeff_offset)
end

export fft, ft_direct, fft_inv, wigner_d, total_coeffs, coeff_offset,
       fhat_at, fhat_block, libsu2_path

# ----- Coefficient layout helpers (ccall to C) -----

"""
    total_coeffs(N::Integer) -> Int

Total number of complex coefficients in a bandlimit-N transform, i.e.
`sum_{l=0..N-1} (2l+1)^2`. Wraps `su2_total_coeffs`.
"""
function total_coeffs(N::Integer)::Int
    Int(ccall(_SYM_TOTAL_COEFFS[], Csize_t, (Cint,), N))
end

"""
    coeff_offset(l::Integer) -> Int

Offset of the degree-l block in a flat coefficient array. Wraps
`su2_coeff_offset`. Equals `sum_{l'=0..l-1} (2l'+1)^2`.
"""
function coeff_offset(l::Integer)::Int
    Int(ccall(_SYM_COEFF_OFFSET[], Csize_t, (Cint,), l))
end

# ----- Wigner small-d / matrix coefficient -----

"""
    wigner_d(l, n, m, theta) -> ComplexF64

Evaluate the paper's matrix coefficient `P^l_{n,m}(cos theta)`. Wraps
`su2_wigner_d`. Note: this is `i^{m-n} * d^l_{n,m}(theta)` (Sakurai d);
see paper.tex line 537 and HANDOFF.md §2 item 2.
"""
function wigner_d(l::Integer, n::Integer, m::Integer, theta::Real)::ComplexF64
    ccall(_SYM_WIGNER_D[], ComplexF64,
          (Cint, Cint, Cint, Cdouble),
          l, n, m, theta)
end

# ----- Transforms -----

"""
    fft(f::Array{ComplexF64,3}) -> Vector{ComplexF64}

O(N^4) fast Fourier transform on SU(2). Wraps `su2_fft`.

Input `f` must be a cubic `(N, N, N)` `Array{ComplexF64,3}` with layout
`f[j2+1, k+1, j1+1] = sample at Euler-grid (j1, k, j2)`; this matches the
C library's row-major `f[j1*N*N + k*N + j2]` without permutation. See
DESIGN.md §3.

Returns a flat `Vector{ComplexF64}` of length `total_coeffs(N)`.
"""
function fft(f::Array{ComplexF64,3})::Vector{ComplexF64}
    N = size(f, 1)
    size(f) == (N, N, N) || throw(ArgumentError("samples must be (N,N,N), got $(size(f))"))
    N >= 2 || throw(ArgumentError("N must be >= 2, got $N"))
    fhat = Vector{ComplexF64}(undef, total_coeffs(N))
    ccall(_SYM_FFT[], Cvoid,
          (Cint, Ptr{ComplexF64}, Ptr{ComplexF64}),
          N, f, fhat)
    return fhat
end

"""
    ft_direct(f::Array{ComplexF64,3}) -> Vector{ComplexF64}

O(N^6) reference (direct) Fourier transform on SU(2). Wraps `su2_ft_direct`.
Same layout convention as `fft`. Use to cross-check `fft` (they must agree
to ~1e-10).
"""
function ft_direct(f::Array{ComplexF64,3})::Vector{ComplexF64}
    N = size(f, 1)
    size(f) == (N, N, N) || throw(ArgumentError("samples must be (N,N,N), got $(size(f))"))
    N >= 2 || throw(ArgumentError("N must be >= 2, got $N"))
    fhat = Vector{ComplexF64}(undef, total_coeffs(N))
    ccall(_SYM_FT_DIRECT[], Cvoid,
          (Cint, Ptr{ComplexF64}, Ptr{ComplexF64}),
          N, f, fhat)
    return fhat
end

"""
    fft_inv(fhat::AbstractVector{ComplexF64}, N::Integer) -> Array{ComplexF64,3}

Peter-Weyl synthesis (inverse of `fft`). Given a flat coefficient vector
`fhat` of length `total_coeffs(N)`, reconstructs samples `f` on the
Euler-angle grid as a Julia `Array{ComplexF64,3}` of shape `(N, N, N)`
with the same layout convention as `fft`:
`f[j2+1, k+1, j1+1] = sample at Euler-grid (j1, k, j2)`.

Wraps `su2_fft_inv`.

**Tolerance note.** The synthesis is the mathematically exact discrete
Peter-Weyl sum. However, the spectrum roundtrip `fft(fft_inv(fhat))`
does NOT recover `fhat` to machine precision under the current
closed-grid Riemann theta quadrature: the relative error is O(1) for
random full-bandwidth `fhat`. The analytical synthesis (e.g.
`fft_inv` of a single coefficient) does match the closed-form to ~1e-12.
Exact roundtrip requires Gauss-Legendre theta nodes (bead `su2fft-ega`).
"""
function fft_inv(fhat::AbstractVector{ComplexF64}, N::Integer)::Array{ComplexF64,3}
    length(fhat) == total_coeffs(N) || throw(ArgumentError(
        "fhat must have length total_coeffs(N) = $(total_coeffs(N)); got $(length(fhat))"))
    N >= 2 || throw(ArgumentError("N must be >= 2, got $N"))
    f = Array{ComplexF64,3}(undef, N, N, N)
    ccall(_SYM_FFT_INV[], Cvoid,
          (Cint, Ptr{ComplexF64}, Ptr{ComplexF64}),
          N, fhat, f)
    return f
end

# ----- Pure-Julia accessors over the flat coefficient array -----

"""
    fhat_at(fhat::AbstractVector{ComplexF64}, l::Integer, m::Integer, n::Integer) -> ComplexF64

Read coefficient at degree `l`, indices `m, n` in `[-l, l]`, from the flat
storage. Layout: `fhat[su2_coeff_offset(l) + (m+l)*(2l+1) + (n+l) + 1]`
(the +1 converts the C 0-based offset to a Julia 1-based index).
"""
function fhat_at(fhat::AbstractVector{ComplexF64}, l::Integer, m::Integer, n::Integer)::ComplexF64
    abs(m) <= l || throw(ArgumentError("|m| must be <= l, got m=$m, l=$l"))
    abs(n) <= l || throw(ArgumentError("|n| must be <= l, got n=$n, l=$l"))
    off = coeff_offset(l)
    fhat[off + (m + l) * (2l + 1) + (n + l) + 1]
end

"""
    fhat_block(fhat::AbstractVector{ComplexF64}, l::Integer) -> AbstractMatrix{ComplexF64}

Return a `(2l+1) x (2l+1)` view of the degree-l block. The C library
stores this block row-major as `fhat[(m+l)*(2l+1) + (n+l)]` for m, n in
`[-l, l]`; reshaping a column-major Julia view of that flat storage
yields a matrix whose Julia-index `[i, j]` equals C-index
`[(j-1)*(2l+1) + (i-1)]`. In other words: `fhat_block(fhat, l)[i, j]`
equals the C value at row `(j-1)`, column `(i-1)` — Julia's `[row, col]`
maps to C's `[col, row]`. Caller should `transpose` if a `[m+l+1, n+l+1]`
indexing is desired.
"""
function fhat_block(fhat::AbstractVector{ComplexF64}, l::Integer)
    off = coeff_offset(l)
    d = 2l + 1
    reshape(view(fhat, off+1 : off + d*d), d, d)
end

# ----- Diagnostics -----

"""
    libsu2_path() -> String

Absolute path to the dynamically-loaded `libsu2.so`, as recorded by
`deps/build.jl`. For debugging linkage.
"""
libsu2_path() = LIBSU2

end # module SU2FFT
