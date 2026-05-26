"""Spherical-harmonic spectrum of the interpolated weather field (bead su2fft-cce, Step F).

Computes fhat = su2_fft_sphere_resolved(f_sph), decomposes into per-l blocks,
plots the angular power spectrum and the |fhat(l, n)| heatmap, and writes a
CSV with the per-l power so the plot can be re-rendered later without re-
running the full pipeline.

Output:

    examples/figures/weather_spectrum.png   (2-panel figure)
    examples/figures/weather_spectrum.csv   (per-l power, l, power, log10_power)

The script also prints:

    - total power,
    - per-l power for l = 0..8,
    - sample-space inverse(forward(f)) residual (expected ~1e-12),
    - fitted exponent p in power(l) ~ l^p over l in [3, N-3].

Grid conventions match examples/weather/fetch_interpolate.py:

    phi[j]   = -pi + j * 2pi/P,  P = 2N - 1   (open)
    theta_k  = arccos(x_k),      x_k = ascending GL nodes on [-1, 1]

Run:

    .venv/bin/python examples/weather/spectrum.py

NOTE on a denser bandlimited reconstruction: one could synthesise from the
spectrum at higher angular resolution by evaluating the Peter-Weyl m=0 sum
on a denser (lon, lat) grid -- this requires a Python Wigner-d (or piping
fhat back through a higher-N inverse on a re-sampled mesh).  Skipped to
keep this file focused; see notes/sphere.md for the formula.
"""
# bead: su2fft-cce
from __future__ import annotations

import csv
import pathlib
import sys
from typing import List, Tuple

import numpy as np

# Repo root on sys.path so we can import the sibling examples.weather module
# and the python/ su2fft wrapper.
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "python"))

import su2fft  # noqa: E402

FIGDIR = _ROOT / "examples" / "figures"


# ---------------------------------------------------------------------------
# Spectrum decomposition
# ---------------------------------------------------------------------------
def per_l_blocks(fhat_sph: np.ndarray, N: int) -> List[np.ndarray]:
    """Return a list of N arrays; entry l has length 2l+1 holding fhat(l, n).

    Indexing follows src/su2_sphere.c: flat offset sum_{l'<l}(2l'+1) + (n+l)
    with l in [0, N-1] and n in [-l, l].
    """
    blocks: List[np.ndarray] = []
    off = 0
    for l in range(N):
        d = 2 * l + 1
        blocks.append(fhat_sph[off:off + d])
        off += d
    if off != N * N:
        raise RuntimeError(f"layout invariant broken: off={off}, expected {N*N}")
    return blocks


def per_l_power(fhat_sph: np.ndarray, N: int) -> np.ndarray:
    """power[l] = sum_n |fhat(l, n)|^2.  Returned as a length-N real array."""
    blocks = per_l_blocks(fhat_sph, N)
    return np.array([float(np.sum(np.abs(b) ** 2)) for b in blocks])


def fit_loglog_slope(
    l_arr: np.ndarray, power: np.ndarray, l_min: int, l_max: int
) -> Tuple[float, float]:
    """Fit log10(power) = a + p*log10(l) over l in [l_min, l_max] inclusive.

    Returns (p, a).  Zero-power degrees are dropped from the fit.
    """
    mask = (l_arr >= l_min) & (l_arr <= l_max) & (power > 0.0)
    if mask.sum() < 2:
        raise RuntimeError(
            f"too few points to fit slope: {int(mask.sum())} in [{l_min}, {l_max}]"
        )
    x = np.log10(l_arr[mask].astype(np.float64))
    y = np.log10(power[mask])
    p, a = np.polyfit(x, y, 1)
    return float(p), float(a)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_spectrum(
    power: np.ndarray,
    fhat_sph: np.ndarray,
    N: int,
    p_slope: float,
    a_intercept: float,
    fit_lmin: int,
    fit_lmax: int,
    title: str,
    out_png: pathlib.Path,
) -> None:
    """Two-panel figure: per-l power (semilogy) + |fhat(l, n)| heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    l_arr = np.arange(N)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8), dpi=160, constrained_layout=True
    )

    # ---- Top: per-l power on log y --------------------------------------
    ax_top.semilogy(l_arr, power, marker="o", linestyle="-", color="C0", label="data")
    # Fitted reference line over the fit window:
    l_fit = np.arange(max(1, fit_lmin), fit_lmax + 1)
    fit_line = (10.0 ** a_intercept) * (l_fit.astype(np.float64) ** p_slope)
    ax_top.semilogy(
        l_fit, fit_line,
        linestyle="--", color="C3",
        label=f"fit: power ~ l^p, p = {p_slope:.2f}  (l in [{fit_lmin}, {fit_lmax}])",
    )
    ax_top.set_xlabel("degree l")
    ax_top.set_ylabel(r"per-l power $\sum_n |\hat f(l, n)|^2$")
    ax_top.set_title(title, fontsize=11)
    ax_top.text(
        0.5, 1.01,
        "computed via su2_fft_sphere_resolved on the (P=2N-1) x N GL sphere grid; bead su2fft-cce.",
        transform=ax_top.transAxes,
        ha="center", va="bottom",
        fontsize=8, color="0.35",
    )
    ax_top.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_top.legend(loc="upper right", fontsize=9)

    # ---- Bottom: |fhat(l, n)| heatmap -----------------------------------
    H = np.full((N, 2 * N - 1), np.nan, dtype=np.float64)
    blocks = per_l_blocks(fhat_sph, N)
    centre = N - 1  # n=0 column
    for l, b in enumerate(blocks):
        for idx, val in enumerate(b):
            n = idx - l
            H[l, centre + n] = abs(complex(val))

    positive = H[np.isfinite(H) & (H > 0.0)]
    if positive.size == 0:
        vmin, vmax = 1e-16, 1.0
    else:
        vmin = float(positive.min())
        vmax = float(positive.max())
        if vmin <= 0.0 or not np.isfinite(vmin):
            vmin = 1e-16
        if vmax <= vmin:
            vmax = vmin * 10.0

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="white")
    im = ax_bot.imshow(
        H,
        origin="lower",
        extent=[-(N - 1) - 0.5, (N - 1) + 0.5, -0.5, N - 0.5],
        aspect="auto",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap,
        interpolation="nearest",
    )
    ax_bot.set_xlabel("order n")
    ax_bot.set_ylabel("degree l")
    ax_bot.set_title(r"$|\hat f(l, n)|$ on the bandlimited triangle  (|n| $\leq$ l)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax_bot, shrink=0.9)
    cbar.set_label(r"$|\hat f(l, n)|$")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def write_csv(power: np.ndarray, out_csv: pathlib.Path) -> None:
    """Tiny per-l power CSV: l, power, log10_power."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["l", "power", "log10_power"])
        for l, p in enumerate(power):
            log10_p = float(np.log10(p)) if p > 0.0 else float("-inf")
            w.writerow([l, f"{p:.17e}", f"{log10_p:.17e}"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from examples.weather.fetch_interpolate import (
        fetch_noaa_air_temperature,
        pick_month,
        interpolate_to_sphere_grid,
    )

    N = 32
    path = fetch_noaa_air_temperature()
    field = pick_month(path, year=2024, month=1)
    f_sph, meta = interpolate_to_sphere_grid(field, N=N)

    P = meta["P"]
    print(
        f"N={meta['N']} P={P} month={meta['year']}-{meta['month']:02d}  "
        f"T mean={f_sph.real.mean():.2f} K  "
        f"min={f_sph.real.min():.2f} K  max={f_sph.real.max():.2f} K"
    )

    # ---- Forward FFT ------------------------------------------------------
    fhat_sph = su2fft.fft_sphere_resolved(f_sph)
    if fhat_sph.shape != (N * N,):
        raise RuntimeError(f"fhat_sph shape {fhat_sph.shape}, expected ({N*N},)")

    # ---- Sanity check via inverse + reconstruction error -----------------
    # Two distinct residuals to print:
    #   (a) inverse(forward(f)) - f         -- bandlimit-truncation residual:
    #       f is NOT bandlimited at N, so this is NOT a machine-precision
    #       quantity.  It measures power lost to l >= N modes.
    #   (b) forward(inverse(fhat)) - fhat   -- the operator-identity residual:
    #       this IS at machine precision on the resolved (GL) grid.
    f_reconstructed = su2fft.fft_sphere_resolved_inv(fhat_sph, N)
    bandlimit_err = np.abs(f_reconstructed - f_sph)
    fhat_back = su2fft.fft_sphere_resolved(f_reconstructed)
    op_err = np.abs(fhat_back - fhat_sph)
    print(
        f"sample-space inverse(forward(f)) residual (bandlimit truncation): "
        f"max={bandlimit_err.max():.2e}, mean={bandlimit_err.mean():.2e}"
    )
    print(
        f"spectrum-space forward(inverse(fhat)) residual (operator identity): "
        f"max={op_err.max():.2e}, mean={op_err.mean():.2e}"
    )

    # ---- Per-l power + summary -------------------------------------------
    power = per_l_power(fhat_sph, N)
    total_power = float(power.sum())
    print(f"total power sum_l sum_n |fhat(l,n)|^2 = {total_power:.6e}")
    print("per-l power table (l = 0..8):")
    print(f"  {'l':>3}  {'power':>14}  {'log10(power)':>14}")
    for l in range(min(9, N)):
        log10_p = np.log10(power[l]) if power[l] > 0.0 else float("-inf")
        print(f"  {l:>3}  {power[l]:>14.6e}  {log10_p:>14.6f}")

    # ---- Slope fit over l in [3, N-3] ------------------------------------
    l_arr = np.arange(N)
    fit_lmin = 3
    fit_lmax = N - 3
    p_slope, a_intercept = fit_loglog_slope(l_arr, power, fit_lmin, fit_lmax)
    print(
        f"loglog fit over l in [{fit_lmin}, {fit_lmax}]: "
        f"power ~ l^p, p = {p_slope:.3f}  (intercept log10(a) = {a_intercept:.3f})"
    )

    # ---- Plot + CSV ------------------------------------------------------
    month_label = f"{meta['year']}-{meta['month']:02d}"
    title = (
        f"Spherical-harmonic spectrum of {month_label} surface temperature "
        f"(NCEP/NCAR, N={N})"
    )
    out_png = FIGDIR / "weather_spectrum.png"
    out_csv = FIGDIR / "weather_spectrum.csv"

    plot_spectrum(
        power=power,
        fhat_sph=fhat_sph,
        N=N,
        p_slope=p_slope,
        a_intercept=a_intercept,
        fit_lmin=fit_lmin,
        fit_lmax=fit_lmax,
        title=title,
        out_png=out_png,
    )
    write_csv(power, out_csv)

    print(f"  saved {out_png}  ({out_png.stat().st_size/1024:.1f} KB)")
    print(f"  saved {out_csv}  ({out_csv.stat().st_size/1024:.1f} KB)")

    # ---- Comparison to typical geophysical 1/l^2 .. 1/l^3 spectra --------
    if -3.5 <= p_slope <= -1.5:
        comment = (
            f"fitted slope p = {p_slope:.2f} sits inside the typical "
            f"geophysical band [-3, -2]."
        )
    elif p_slope > -1.5:
        comment = (
            f"fitted slope p = {p_slope:.2f} is shallower than the typical "
            f"geophysical band [-3, -2]."
        )
    else:
        comment = (
            f"fitted slope p = {p_slope:.2f} is steeper than the typical "
            f"geophysical band [-3, -2]."
        )
    print(comment)
