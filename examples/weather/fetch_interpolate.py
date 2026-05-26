"""NOAA reanalysis -> sphere-FFT grid interpolation (bead su2fft-cce, Step D).

Fetches the NOAA NCEP/NCAR Reanalysis monthly-mean surface air temperature
(``air.sig995.mon.mean``) into ``data/`` and interpolates a chosen month onto
the resolved SU(2) sphere FFT grid:

    phi[j1] = -pi + j1 * 2pi/P,   j1 in [0, P-1],   P = 2N-1   (open, ascending)
    theta_k = arccos(x_k),  x_k = N-point Gauss-Legendre nodes on [-1, 1]
                              (k=0 -> theta near pi, k=N-1 -> theta near 0
                               because GL nodes are ascending in x).

Convention choices (printed by the smoke run for reproducibility):

  * Longitude wrap: NOAA's lon is in [0, 360). We map to phi in [-pi, pi)
    by  phi = ((lon + 180) mod 360 - 180) * pi/180,  then sort. This puts
    Greenwich at phi=0 and the dateline at +/- pi, which is the natural
    convention matching the sphere FFT layout.
  * Colatitude: theta = pi/2 - lat * pi/180. NOAA lat includes +/- 90;
    RectSphereBivariateSpline requires u in the open interval (0, pi),
    so we DROP the two pole rows on the input side. The GL query nodes
    are strictly interior, so the spline is evaluated only in (0, pi).
  * Interpolation: scipy.interpolate.RectSphereBivariateSpline with s=0
    (pure interpolating tensor-product spline; respects spherical wrap
    in longitude).

Programmatic interface
----------------------
    fetch_noaa_air_temperature(cache_dir="data") -> Path
    pick_month(path, year=2024, month=1)         -> xarray.DataArray (lat, lon)
    interpolate_to_sphere_grid(field_2d, N=32)   -> (np.ndarray complex128
                                                     of shape (P, N), meta dict)

Running this file as a script downloads (or reuses) the cache, picks
2024-01, interpolates to N=32, and prints a one-line summary.
"""
# bead: su2fft-cce
from __future__ import annotations

import pathlib
import sys
from typing import Tuple

import numpy as np
import requests
import xarray as xr
from scipy.interpolate import RectSphereBivariateSpline

# Allow `import su2fft` from the project's python/ directory.
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "python"))
import su2fft  # noqa: E402  (after sys.path tweak)

_PRIMARY_URL = (
    "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/"
    "surface/air.sig995.mon.mean.nc"
)
_FALLBACK_URL = (
    "https://psl.noaa.gov/thredds/fileServer/Datasets/"
    "ncep.reanalysis.derived/surface/air.sig995.mon.mean.nc"
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _stream_download(url: str, dest: pathlib.Path) -> int:
    """Stream `url` to `dest`, printing one progress line. Returns bytes written."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or 0)
        written = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                if total > 0:
                    pct = 100.0 * written / total
                    print(
                        f"  downloading {url.rsplit('/', 1)[-1]}: "
                        f"{written/1e6:7.2f} MB / {total/1e6:7.2f} MB ({pct:5.1f}%)",
                        end="\r",
                        flush=True,
                    )
                else:
                    print(
                        f"  downloading {url.rsplit('/', 1)[-1]}: "
                        f"{written/1e6:7.2f} MB",
                        end="\r",
                        flush=True,
                    )
        print()  # newline after progress
    tmp.replace(dest)
    return written


def fetch_noaa_air_temperature(cache_dir: str | pathlib.Path = "data") -> pathlib.Path:
    """Download (or reuse cached) NOAA surface-air monthly-mean reanalysis.

    Cache file: ``<cache_dir>/air.sig995.mon.mean.nc`` relative to the repo root
    (or to the user's CWD if `cache_dir` is absolute / relative).

    Idempotent: returns immediately if the file already exists and opens
    successfully.
    """
    cache_dir = pathlib.Path(cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = _ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "air.sig995.mon.mean.nc"

    if dest.exists():
        # Validate it opens; if corrupt, force re-download.
        try:
            with xr.open_dataset(dest) as ds:
                _ = ds["air"].shape
            return dest
        except Exception:
            dest.unlink(missing_ok=True)

    errors: list[str] = []
    for url in (_PRIMARY_URL, _FALLBACK_URL):
        try:
            _stream_download(url, dest)
            with xr.open_dataset(dest) as ds:
                _ = ds["air"].shape
            return dest
        except Exception as exc:  # noqa: BLE001 -- we re-raise below if both fail
            errors.append(f"  {url}\n    -> {type(exc).__name__}: {exc}")
            dest.unlink(missing_ok=True)
    raise RuntimeError(
        "Failed to fetch NOAA NCEP/NCAR Reanalysis surface air temperature "
        "(air.sig995.mon.mean.nc) from both:\n"
        + "\n".join(errors)
        + f"\nManual download: save the file to {dest} and re-run."
    )


# ---------------------------------------------------------------------------
# Pick month
# ---------------------------------------------------------------------------
def pick_month(path: pathlib.Path, year: int = 2024, month: int = 1) -> xr.DataArray:
    """Return the 2-D (lat, lon) air-temperature slice for the requested month.

    Units are inherited from the file (typically Kelvin -- NCEP stores degK).
    """
    ds = xr.open_dataset(path)
    air = ds["air"]  # (time, lat, lon)
    # NCEP monthly means are mid-month timestamps; pick by year/month equality.
    sel = air.sel(time=f"{year:04d}-{month:02d}")
    if sel.ndim == 3:
        # If multiple matches (shouldn't happen with month resolution), take first.
        sel = sel.isel(time=0)
    sel = sel.load()
    ds.close()
    # NCEP files often store as int16 packed -> degC + offset; xarray auto-decodes
    # but the result may be in degC depending on the file. Force Kelvin.
    units = str(sel.attrs.get("units", "")).strip().lower()
    if units in ("degc", "degrees_c", "celsius", "c"):
        sel = sel + 273.15
        sel.attrs["units"] = "K"
    elif units in ("", "k", "kelvin", "degk", "degrees_k"):
        sel.attrs.setdefault("units", "K")
    return sel


# ---------------------------------------------------------------------------
# Interpolate to sphere grid
# ---------------------------------------------------------------------------
def interpolate_to_sphere_grid(
    field_2d: xr.DataArray, N: int = 32
) -> Tuple[np.ndarray, dict]:
    """Interpolate a (lat, lon) field onto the resolved SU(2) sphere grid.

    Output array has shape (P, N) row-major with P = 2N-1; entry [j1, k]
    is the interpolated value at (phi[j1], theta_k).  Returned as complex128
    (imag = 0) because the sphere FFT C entry-point takes complex input.
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")

    # ---- Input grid (NOAA) --------------------------------------------------
    lat_deg = np.asarray(field_2d["lat"].values, dtype=np.float64)
    lon_deg = np.asarray(field_2d["lon"].values, dtype=np.float64)
    data = np.asarray(field_2d.values, dtype=np.float64)  # (lat, lon)

    # NCEP lat is typically descending 90 -> -90; RectSphereBivariateSpline
    # needs ascending colatitude in (0, pi). Sort lat ascending first.
    lat_order = np.argsort(lat_deg)  # ascending: -90 -> 90
    lat_deg = lat_deg[lat_order]
    data = data[lat_order, :]
    # colatitude = pi/2 - lat in radians; lat ascending -> colat descending.
    # We want colat ascending, so reverse:
    data = data[::-1, :]
    lat_deg = lat_deg[::-1]  # now descending lat -> ascending colat
    colat_rad = (np.pi / 2.0) - lat_deg * (np.pi / 180.0)

    # Drop pole rows (lat = +/- 90 -> colat = 0 or pi). RectSphereBivariateSpline
    # requires u strictly in the open interval (0, pi).  Rationale: the GL
    # query nodes are strictly interior, so dropping the poles on the input
    # side loses no useful information for the interior evaluation.
    interior = (colat_rad > 1e-12) & (colat_rad < np.pi - 1e-12)
    colat_rad = colat_rad[interior]
    data = data[interior, :]
    if not np.all(np.diff(colat_rad) > 0):
        raise RuntimeError("Internal: colatitude not strictly ascending")

    # NOAA lon is in [0, 360) ascending. Wrap to [-pi, pi):
    #   phi_deg = ((lon + 180) mod 360) - 180
    # then sort ascending. This matches the FFT grid convention
    # (phi[0] = -pi, phi[P-1] = -pi + (P-1)*2pi/P).
    phi_deg = ((lon_deg + 180.0) % 360.0) - 180.0
    phi_order = np.argsort(phi_deg)
    phi_deg = phi_deg[phi_order]
    data = data[:, phi_order]
    phi_rad_in = phi_deg * (np.pi / 180.0)
    if not np.all(np.diff(phi_rad_in) > 0):
        raise RuntimeError("Internal: input longitude not strictly ascending")

    # ---- Output grid (sphere FFT) ------------------------------------------
    P = su2fft.P(N)
    x_gl, _w_gl = su2fft.gl_nodes_weights(N)  # ascending in (-1, 1)
    theta_q = np.arccos(x_gl)  # descending in (0, pi): x ascending -> theta desc
    # All theta_q in (0, pi) -- safe for RectSphereBivariateSpline.

    phi_q = -np.pi + np.arange(P, dtype=np.float64) * (2.0 * np.pi / P)

    # ---- Spline + evaluate --------------------------------------------------
    spline = RectSphereBivariateSpline(colat_rad, phi_rad_in, data, s=0.0)
    # Build meshgrid of query points with axis order (theta, phi) since the
    # spline expects (u, v) = (colat, lon). Evaluate flat, then reshape.
    TT, PP = np.meshgrid(theta_q, phi_q, indexing="ij")  # shape (N, P)
    vals = spline.ev(TT.ravel(), PP.ravel()).reshape(N, P)
    # Transpose to (P, N) row-major to match the sphere FFT layout.
    out = np.ascontiguousarray(vals.T, dtype=np.float64)
    out_c = out.astype(np.complex128)

    units = str(field_2d.attrs.get("units", "K"))
    # Recover year/month from the field's time coordinate if present.
    year = month = None
    if "time" in field_2d.coords:
        t = np.datetime64(field_2d["time"].values)
        year = int(str(t)[0:4])
        month = int(str(t)[5:7])
    meta = {
        "N": N,
        "P": P,
        "year": year,
        "month": month,
        "units": units,
        "interp": "RectSphereBivariateSpline(s=0)",
        "phi_convention": "phi in [-pi, pi), phi[j1] = -pi + j1*2pi/P",
        "theta_convention": "theta_k = arccos(x_k), x_k = ascending GL nodes",
        "lon_wrap": "((lon_deg + 180) mod 360) - 180",
        "poles_dropped": "lat = +/- 90 input rows excluded (open colat interval)",
    }
    return out_c, meta


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Convention choices for this smoke run:")
    print("  longitude wrap : ((lon_deg + 180) mod 360) - 180  ->  phi in [-pi, pi)")
    print("  colatitude     : theta = pi/2 - lat (radians); poles (+/-90) dropped")
    print("  output grid    : phi[j1] = -pi + j1 * 2pi/P (P=2N-1, open),")
    print("                   theta_k = arccos(x_k), x_k ascending GL nodes")
    print("  interpolation  : scipy RectSphereBivariateSpline(s=0)")
    print()

    path = fetch_noaa_air_temperature()
    size_mb = path.stat().st_size / 1e6
    print(f"  cache         : {path}  ({size_mb:.2f} MB)")

    field = pick_month(path, year=2024, month=1)
    f_sph, meta = interpolate_to_sphere_grid(field, N=32)
    print(
        f"N={meta['N']} month={meta['year']}-{meta['month']:02d} "
        f"shape={f_sph.shape} mean={f_sph.real.mean():.2f}K "
        f"min={f_sph.real.min():.2f}K max={f_sph.real.max():.2f}K"
    )
