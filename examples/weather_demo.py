#!/usr/bin/env python3
"""End-to-end weather visualisation demo for the resolved-grid SU(2) FFT.

Bead: su2fft-cce.  Run after `make lib` and after installing the venv deps
(`.venv/bin/pip install -r examples/requirements.txt`).

Pipeline:
    1. fetch NOAA NCEP/NCAR Reanalysis surface temperature (cached under data/)
    2. pick a target month
    3. interpolate the 2.5 deg lat/lon field onto the resolved sphere grid
       (P=2N-1 open phi, N Gauss-Legendre theta)
    4. render the 3D sphere with matplotlib, pyvista, plotly (N=N_HERO)
    5. compute the spherical-harmonic spectrum via su2_fft_sphere_resolved
    6. plot per-l power decay + |fhat(l, n)| heatmap (N=N_SPEC)

Run: .venv/bin/python examples/weather_demo.py
"""
from __future__ import annotations

import sys
import time
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples.weather.fetch_interpolate import (
    fetch_noaa_air_temperature, pick_month, interpolate_to_sphere_grid,
)
from examples.weather.render_3d import (
    render_matplotlib, render_pyvista, render_plotly,
)
from examples.weather.spectrum import (
    per_l_power, fit_loglog_slope, plot_spectrum, write_csv,
)
from python import su2fft


N_HERO = 64           # 3D sphere renders
N_SPEC = 32           # spectrum panel
YEAR, MONTH = 2024, 1

FIG_DIR = ROOT / "examples" / "figures"


def main() -> int:
    print("=== Weather sphere FFT demo (bead su2fft-cce) ===\n")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. fetch + 2. pick month
    t0 = time.time()
    nc_path = fetch_noaa_air_temperature()
    field_2d = pick_month(nc_path, year=YEAR, month=MONTH)
    print(f"fetched + picked: {nc_path.name}, month={YEAR}-{MONTH:02d} "
          f"({time.time()-t0:.1f}s)\n")

    # 3a. interpolate at N_HERO for renders
    t0 = time.time()
    f_hero, meta_hero = interpolate_to_sphere_grid(field_2d, N=N_HERO)
    P_hero = 2 * N_HERO - 1
    print(f"interpolated to ({P_hero}, {N_HERO}) sphere grid for renders "
          f"({time.time()-t0:.2f}s)")
    print(f"  T range: mean={f_hero.real.mean():.2f}K "
          f"min={f_hero.real.min():.2f}K max={f_hero.real.max():.2f}K\n")

    title = (f"Surface air temperature, {meta_hero['year']}-"
             f"{meta_hero['month']:02d} (NOAA NCEP/NCAR Reanalysis)")

    # 4. three renders
    t0 = time.time()
    p_mpl = FIG_DIR / "weather_matplotlib.png"
    render_matplotlib(f_hero, N_HERO, title, p_mpl)
    print(f"  matplotlib  -> {p_mpl.name}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    p_pv_png = FIG_DIR / "weather_pyvista.png"
    p_pv_html = FIG_DIR / "weather_pyvista.html"
    render_pyvista(f_hero, N_HERO, title, p_pv_png, p_pv_html)
    print(f"  pyvista     -> {p_pv_png.name}, {p_pv_html.name}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    p_pl = FIG_DIR / "weather_plotly.html"
    render_plotly(f_hero, N_HERO, title, p_pl)
    print(f"  plotly      -> {p_pl.name}  ({time.time()-t0:.1f}s)\n")

    # 3b. interpolate again at N_SPEC for spectrum (smaller, fits two panels)
    t0 = time.time()
    f_spec, meta_spec = interpolate_to_sphere_grid(field_2d, N=N_SPEC)
    print(f"interpolated to ({2*N_SPEC-1}, {N_SPEC}) sphere grid for spectrum "
          f"({time.time()-t0:.2f}s)")

    # 5. forward FFT + spectrum
    t0 = time.time()
    fhat_sph = su2fft.fft_sphere_resolved(f_spec)
    power = per_l_power(fhat_sph, N_SPEC)
    fit_lmin, fit_lmax = 3, N_SPEC - 3
    l_arr = np.arange(N_SPEC)
    p_slope, a_intercept = fit_loglog_slope(l_arr, power, fit_lmin, fit_lmax)
    print(f"  forward FFT + per-l power  ({time.time()-t0:.2f}s)")
    print(f"  fit slope:  power(l) ~ l^{p_slope:.3f}  (l in [{fit_lmin}, {fit_lmax}])")

    # operator-identity residual: forward(inverse(fhat)) - fhat
    f_back = su2fft.fft_sphere_resolved_inv(fhat_sph, N_SPEC)
    fhat_back = su2fft.fft_sphere_resolved(f_back)
    identity_res = float(np.abs(fhat_back - fhat_sph).max())
    print(f"  identity:   max|fhat - forward(inverse(fhat))| = {identity_res:.2e}\n")

    # 6. spectrum plot + CSV
    spec_title = (f"Spherical-harmonic spectrum of {meta_spec['year']}-"
                  f"{meta_spec['month']:02d} surface temperature "
                  f"(NCEP/NCAR, N={N_SPEC})")
    p_spec_png = FIG_DIR / "weather_spectrum.png"
    p_spec_csv = FIG_DIR / "weather_spectrum.csv"
    plot_spectrum(power, fhat_sph, N_SPEC, p_slope, a_intercept,
                  fit_lmin, fit_lmax, spec_title, p_spec_png)
    write_csv(power, p_spec_csv)
    print(f"  spectrum    -> {p_spec_png.name}, {p_spec_csv.name}\n")

    print("=== done ===")
    print(f"Outputs in: {FIG_DIR}/")
    print(f"  Static PNG:  weather_matplotlib.png, weather_pyvista.png, weather_spectrum.png")
    print(f"  Interactive: weather_pyvista.html, weather_plotly.html")
    print(f"  Data:        weather_spectrum.csv  (per-l power)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
