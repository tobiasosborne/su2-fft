"""su2fft -- minimal Python ctypes bindings to libsu2 (sphere FFT only).

Bead: su2fft-cce.  Loads build/libsu2.so from the project root.

Public API:
    P(N)                            -> int (= 2*N - 1)
    sphere_total_coeffs(N)          -> int (= N**2)
    sphere_resolved_total_samples(N) -> int (= P(N) * N)
    gl_nodes_weights(N)             -> (numpy.ndarray, numpy.ndarray), shape (N,)
    fft_sphere_resolved(f)          -> fhat (complex128, length N**2)
    fft_sphere_resolved_inv(fhat, N) -> f   (complex128, length P*N)

Conventions:
    f is shape (P, N) row-major: f[j1, k] is the sample at phi[j1], theta_k.
    fhat is a 1-D length-N**2 complex array indexed by sum_{l'<l}(2l'+1)+(n+l).

    phi[j1]  = -pi + j1 * 2pi/P,   j1 in [0, P-1],   P = 2N-1.
    theta_k = arccos(x_k), x_k the k-th N-point Gauss-Legendre node on [-1, 1].

Returns numpy arrays at full precision.  Input arrays are coerced to
contiguous complex128 if needed.
"""
# bead: su2fft-cce
from __future__ import annotations

import ctypes
import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------
_here: pathlib.Path = pathlib.Path(__file__).resolve().parent
_root: pathlib.Path = _here.parent
_libpath: pathlib.Path = _root / "build" / "libsu2.so"
if not _libpath.exists():
    raise RuntimeError(
        f"libsu2.so not found at {_libpath}. Run `make lib` in the project root."
    )

# No libgmp preload hack needed for Python (only Julia hits that, see
# julia/src/SU2FFT.jl::__init__).  CPython's dlopen does not pre-bind libgmp.
_lib: ctypes.CDLL = ctypes.CDLL(str(_libpath))

# Treat `double _Complex *` as `double *` -- numpy.complex128 is a 16-byte
# pair of doubles (real, imag) in memory, which is ABI-compatible with C99
# `double _Complex` on Linux/x86-64 (see gcc/glibc complex.h).
_DPtr = ctypes.POINTER(ctypes.c_double)

# ---------------------------------------------------------------------------
# C entry-point signatures
# ---------------------------------------------------------------------------
_lib.su2_sphere_total_coeffs.argtypes = [ctypes.c_int]
_lib.su2_sphere_total_coeffs.restype = ctypes.c_size_t

# NB: su2_resolved_P is `static inline` in include/su2.h -- NOT a symbol in
# libsu2.so.  Implement P(N) in pure Python below.  Likewise the matching
# su2_resolved_total_samples / su2_resolved_sample_index helpers.

_lib.su2_sphere_resolved_total_samples.argtypes = [ctypes.c_int]
_lib.su2_sphere_resolved_total_samples.restype = ctypes.c_size_t

_lib.su2_gl_nodes_weights.argtypes = [ctypes.c_int, _DPtr, _DPtr]
_lib.su2_gl_nodes_weights.restype = None

_lib.su2_fft_sphere_resolved.argtypes = [ctypes.c_int, _DPtr, _DPtr]
_lib.su2_fft_sphere_resolved.restype = None

_lib.su2_fft_sphere_inv_resolved.argtypes = [ctypes.c_int, _DPtr, _DPtr]
_lib.su2_fft_sphere_inv_resolved.restype = None


# ---------------------------------------------------------------------------
# Pure-Python layout helpers
# ---------------------------------------------------------------------------
def P(N: int) -> int:
    """Return the open-grid phi/psi sample count, P = 2N - 1.

    This mirrors `su2_resolved_P` from include/su2.h, which is a
    `static inline` (not exported as a symbol).  Implemented directly
    in Python.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    return 2 * N - 1


def sphere_total_coeffs(N: int) -> int:
    """Number of spherical-harmonic coefficients at bandlimit N.

    Equals N**2.  Wraps `su2_sphere_total_coeffs`.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    return int(_lib.su2_sphere_total_coeffs(ctypes.c_int(N)))


def sphere_resolved_total_samples(N: int) -> int:
    """Number of sphere samples on the resolved grid.

    Equals P(N) * N = (2N - 1) * N.  Wraps
    `su2_sphere_resolved_total_samples`.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    return int(_lib.su2_sphere_resolved_total_samples(ctypes.c_int(N)))


# ---------------------------------------------------------------------------
# Gauss-Legendre nodes / weights
# ---------------------------------------------------------------------------
def gl_nodes_weights(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return N-point Gauss-Legendre nodes and weights on [-1, 1].

    Ascending in (-1, 1).  Theta grid: theta_k = arccos(x_k).
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    x = np.empty(N, dtype=np.float64)
    w = np.empty(N, dtype=np.float64)
    _lib.su2_gl_nodes_weights(
        ctypes.c_int(N),
        x.ctypes.data_as(_DPtr),
        w.ctypes.data_as(_DPtr),
    )
    return x, w


# ---------------------------------------------------------------------------
# Sphere FFT (resolved variant)
# ---------------------------------------------------------------------------
def _as_c128(arr: np.ndarray) -> np.ndarray:
    """Coerce to a C-contiguous complex128 array (copy only if needed)."""
    return np.ascontiguousarray(arr, dtype=np.complex128)


def fft_sphere_resolved(f: np.ndarray, N: int | None = None) -> np.ndarray:
    """Forward S^2 FFT on the resolved (open-P phi, GL theta) grid.

    Wraps `su2_fft_sphere_resolved` (bead su2fft-9qk).

    Parameters
    ----------
    f : array_like, complex
        Samples on the resolved grid.  Either shape (P, N) with P = 2N-1
        (row-major (phi_index, theta_index)) or a flat length-P*N vector.
    N : int, optional
        Bandlimit.  Required if `f` is 1-D; inferred from shape if 2-D.

    Returns
    -------
    fhat : ndarray, complex128, shape (N**2,)
        Flat spectrum indexed by sum_{l'<l}(2l'+1) + (n+l).
    """
    f = _as_c128(f)
    if N is None:
        if f.ndim == 2:
            P_in, N_inferred = f.shape
            if P_in != 2 * N_inferred - 1:
                raise ValueError(
                    f"f.shape={f.shape}: expected (P, N) with P = 2N-1, "
                    f"got P={P_in}, N={N_inferred}"
                )
            N = N_inferred
        else:
            raise ValueError("Pass N explicitly when f is 1-D")
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    expected = sphere_resolved_total_samples(N)
    if f.size != expected:
        raise ValueError(
            f"f has {f.size} samples; expected sphere_resolved_total_samples({N}) = {expected}"
        )
    fhat = np.empty(sphere_total_coeffs(N), dtype=np.complex128)
    _lib.su2_fft_sphere_resolved(
        ctypes.c_int(N),
        f.ctypes.data_as(_DPtr),
        fhat.ctypes.data_as(_DPtr),
    )
    return fhat


def fft_sphere_resolved_inv(fhat: np.ndarray, N: int) -> np.ndarray:
    """Inverse S^2 FFT on the resolved grid (Peter-Weyl synthesis, m=0).

    Wraps `su2_fft_sphere_inv_resolved` (bead su2fft-9qk).

    Parameters
    ----------
    fhat : array_like, complex
        Flat spectrum of length N**2.
    N : int
        Bandlimit.

    Returns
    -------
    f : ndarray, complex128, shape (P, N) with P = 2N - 1
        Samples on the resolved grid, row-major (phi_index, theta_index).
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    fhat = _as_c128(fhat)
    expected = sphere_total_coeffs(N)
    if fhat.size != expected:
        raise ValueError(
            f"fhat has {fhat.size} entries; expected sphere_total_coeffs({N}) = {expected}"
        )
    P_val = P(N)
    f = np.empty((P_val, N), dtype=np.complex128)
    _lib.su2_fft_sphere_inv_resolved(
        ctypes.c_int(N),
        fhat.ctypes.data_as(_DPtr),
        f.ctypes.data_as(_DPtr),
    )
    return f


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(0xC0FFEE)
    N = 8
    P_val = P(N)
    nsamp = sphere_resolved_total_samples(N)
    ncoeff = sphere_total_coeffs(N)
    print(f"N={N}  P={P_val}  samples={nsamp}  coeffs={ncoeff}")

    passes = 0
    fails = 0

    def _report(name: str, ok: bool, value: float) -> None:
        global passes, fails
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}: {value:.3e}")
        if ok:
            passes += 1
        else:
            fails += 1

    # ---- Test 1: forward of constant 1.0 -> fhat[0] ~ 1, rest ~ 0.
    f_const = np.ones((P_val, N), dtype=np.complex128)
    fhat_const = fft_sphere_resolved(f_const)
    fhat0_err = abs(fhat_const[0] - 1.0)
    other_max = float(np.max(np.abs(fhat_const[1:])))
    _report("forward(constant=1) fhat[0]=1", fhat0_err < 1e-12, fhat0_err)
    _report("forward(constant=1) |other|<1e-12", other_max < 1e-12, other_max)

    # ---- Test 2: inverse of (1, 0, ...) -> f ~ 1.0 everywhere.
    fhat_l0 = np.zeros(ncoeff, dtype=np.complex128)
    fhat_l0[0] = 1.0
    f_recon = fft_sphere_resolved_inv(fhat_l0, N)
    l0_err = float(np.max(np.abs(f_recon - 1.0)))
    _report("inverse([1,0,...]) ~ constant 1", l0_err < 1e-12, l0_err)

    # ---- Test 3: random spectrum roundtrip forward(inverse(fhat)) == fhat.
    fhat_rand = (rng.standard_normal(ncoeff) + 1j * rng.standard_normal(ncoeff))
    fhat_rand = fhat_rand.astype(np.complex128)
    f_rand = fft_sphere_resolved_inv(fhat_rand, N)
    fhat_back = fft_sphere_resolved(f_rand)
    abs_err = float(np.max(np.abs(fhat_back - fhat_rand)))
    rel_err = abs_err / float(np.max(np.abs(fhat_rand)))
    _report("roundtrip forward(inverse(random)) (abs)", abs_err < 1e-10, abs_err)
    _report("roundtrip forward(inverse(random)) (rel)", rel_err < 1e-12, rel_err)

    print(f"summary: {passes} passed, {fails} failed")
    sys.exit(0 if fails == 0 else 1)
