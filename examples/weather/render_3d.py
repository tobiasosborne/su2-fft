"""3D-sphere renderings of the interpolated weather field (bead su2fft-cce, Step E).

Three side-by-side renders of the same field f(phi_j, theta_k) on the unit
sphere, one per library:

    matplotlib  -> examples/figures/weather_matplotlib.png   (static PNG)
    pyvista     -> examples/figures/weather_pyvista.png      (static PNG)
                   examples/figures/weather_pyvista.html     (interactive HTML)
    plotly      -> examples/figures/weather_plotly.html      (interactive HTML)

This step renders the RAW interpolated input -- not the bandlimited FFT
reconstruction.  The FFT-roundtrip render is the next step (Step F).

Grid conventions (matching examples/weather/fetch_interpolate.py):

    phi[j]  = -pi + j * 2pi / P,        P = 2N - 1,  j in [0, P-1]   (open)
    theta_k = arccos(x_k),              x_k = ascending GL nodes
                                        => theta_k descending in k

    Cartesian on the unit sphere:
        X = sin(theta) cos(phi)
        Y = sin(theta) sin(phi)
        Z = cos(theta)

    T has shape (P, N): T[j, k] = field at (phi_j, theta_k).
    XYZ meshes are built with the same (P, N) shape.

Run:

    .venv/bin/python examples/weather/render_3d.py
"""
# bead: su2fft-cce
from __future__ import annotations

import pathlib
import sys

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
# Shared mesh + colour scaling
# ---------------------------------------------------------------------------
def _sphere_mesh(N: int):
    """Return (X, Y, Z, theta, phi) for the (P, N) sphere mesh with phi wrap.

    The j = P column from `fetch_interpolate` is duplicated (= the j = 0
    column shifted by 2pi) so the surface closes seamlessly at the dateline.
    Returned arrays have shape (P+1, N).
    """
    P = su2fft.P(N)
    x_gl, _w = su2fft.gl_nodes_weights(N)
    theta = np.arccos(x_gl)  # descending in k (x ascending)
    phi = -np.pi + np.arange(P, dtype=np.float64) * (2.0 * np.pi / P)
    phi_wrap = np.concatenate([phi, [phi[0] + 2.0 * np.pi]])  # close the seam
    Phi, Theta = np.meshgrid(phi_wrap, theta, indexing="ij")  # (P+1, N)
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)
    return X, Y, Z, theta, phi_wrap


def _wrap_field(T: np.ndarray) -> np.ndarray:
    """Append the j = 0 column at j = P to seal the dateline seam.

    Input  (P, N) -> output (P+1, N).
    """
    return np.concatenate([T, T[0:1, :]], axis=0)


# ---------------------------------------------------------------------------
# 1) matplotlib
# ---------------------------------------------------------------------------
def render_matplotlib(f: np.ndarray, N: int, title: str, out_png: pathlib.Path) -> None:
    """Static PNG via matplotlib 3D (mplot3d, plot_surface)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib import cm, colormaps

    T = _wrap_field(f.real)  # (P+1, N)
    X, Y, Z, _theta, _phi = _sphere_mesh(N)
    if T.shape != X.shape:
        raise RuntimeError(f"mesh/field shape mismatch: {T.shape} vs {X.shape}")

    tmin, tmax = float(T.min()), float(T.max())
    norm = Normalize(vmin=tmin, vmax=tmax)
    cmap = colormaps["RdYlBu_r"]

    # plot_surface facecolors expects cell-centred colours of shape (M-1, N-1).
    T_face = 0.25 * (T[:-1, :-1] + T[1:, :-1] + T[:-1, 1:] + T[1:, 1:])
    facecolors = cmap(norm(T_face))

    fig = plt.figure(figsize=(8, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        shade=False,
        rstride=1,
        cstride=1,
        antialiased=True,
        linewidth=0.0,
    )
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=25, azim=-60)
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    # Colorbar on its own axes so the 3D ax stays square.
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cax = fig.add_axes([0.86, 0.20, 0.025, 0.6])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Surface air temperature [K]", fontsize=10)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2) pyvista
# ---------------------------------------------------------------------------
def render_pyvista(
    f: np.ndarray,
    N: int,
    title: str,
    out_png: pathlib.Path,
    out_html: pathlib.Path,
) -> None:
    """Static PNG + interactive HTML via pyvista StructuredGrid."""
    import pyvista as pv

    pv.OFF_SCREEN = True  # headless safe; flipped per-plotter below for HTML
    T = _wrap_field(f.real)  # (P+1, N)
    X, Y, Z, _theta, _phi = _sphere_mesh(N)

    grid = pv.StructuredGrid(X, Y, Z)
    # StructuredGrid points are flattened in Fortran order over (dim0, dim1,
    # dim2) = (P+1, N, 1).  pyvista's `dimensions` reflects that, so ravelling
    # T in Fortran order matches the point layout.
    if grid.n_points != T.size:
        raise RuntimeError(
            f"pyvista point count {grid.n_points} != field size {T.size}"
        )
    grid["temperature_K"] = T.ravel(order="F")

    tmin, tmax = float(T.min()), float(T.max())
    scalar_bar_args = dict(
        title="Surface air temperature [K]",
        title_font_size=12,
        label_font_size=10,
        n_labels=5,
        fmt="%.0f",
    )

    # Equator line: theta = pi/2 isoline on the unit sphere.
    eq_phi = np.linspace(-np.pi, np.pi, 361)
    equator = pv.lines_from_points(
        np.column_stack([np.cos(eq_phi), np.sin(eq_phi), np.zeros_like(eq_phi)])
    )

    # ---- Static PNG ---------------------------------------------------------
    p_png = pv.Plotter(off_screen=True, window_size=(800, 800))
    p_png.add_mesh(
        grid,
        scalars="temperature_K",
        cmap="RdYlBu_r",
        clim=(tmin, tmax),
        smooth_shading=True,
        scalar_bar_args=scalar_bar_args,
    )
    p_png.add_mesh(equator, color="black", line_width=1.0)
    p_png.add_text(title, font_size=10, position="upper_edge")
    p_png.set_background("white")
    p_png.camera_position = "iso"
    p_png.camera.azimuth = -60
    p_png.camera.elevation = 25
    out_png.parent.mkdir(parents=True, exist_ok=True)
    p_png.screenshot(str(out_png), transparent_background=False)
    p_png.close()

    # ---- Interactive HTML ---------------------------------------------------
    p_html = pv.Plotter(off_screen=True, window_size=(800, 800))
    p_html.add_mesh(
        grid,
        scalars="temperature_K",
        cmap="RdYlBu_r",
        clim=(tmin, tmax),
        smooth_shading=True,
        scalar_bar_args=scalar_bar_args,
    )
    p_html.add_mesh(equator, color="black", line_width=1.0)
    p_html.add_text(title, font_size=10, position="upper_edge")
    p_html.set_background("white")
    p_html.camera_position = "iso"
    p_html.camera.azimuth = -60
    p_html.camera.elevation = 25
    out_html.parent.mkdir(parents=True, exist_ok=True)
    p_html.export_html(str(out_html))
    p_html.close()


# ---------------------------------------------------------------------------
# 3) plotly
# ---------------------------------------------------------------------------
def render_plotly(f: np.ndarray, N: int, title: str, out_html: pathlib.Path) -> None:
    """Interactive HTML via plotly graph_objects.Surface."""
    import plotly.graph_objects as go

    T = _wrap_field(f.real)  # (P+1, N)
    X, Y, Z, _theta, _phi = _sphere_mesh(N)
    tmin, tmax = float(T.min()), float(T.max())

    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=T,
        colorscale="RdBu_r",
        cmin=tmin,
        cmax=tmax,
        colorbar=dict(title="K"),
        showscale=True,
        lighting=dict(ambient=0.7, diffuse=0.4, specular=0.1),
    )
    fig = go.Figure(data=[surface])
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="cube",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        width=800,
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", auto_open=False)


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
    f, meta = interpolate_to_sphere_grid(field, N=N)

    T = f.real
    print(
        f"N={meta['N']} P={meta['P']} month={meta['year']}-{meta['month']:02d}  "
        f"T mean={T.mean():.2f} K  min={T.min():.2f} K  max={T.max():.2f} K"
    )

    title = (
        f"Surface air temperature, {meta['year']}-{meta['month']:02d} "
        f"(NOAA NCEP/NCAR Reanalysis)"
    )

    out_mpl = FIGDIR / "weather_matplotlib.png"
    out_pv_png = FIGDIR / "weather_pyvista.png"
    out_pv_html = FIGDIR / "weather_pyvista.html"
    out_pl_html = FIGDIR / "weather_plotly.html"

    print("rendering matplotlib ...")
    render_matplotlib(f, N, title, out_mpl)
    print(f"  saved {out_mpl}  ({out_mpl.stat().st_size/1024:.1f} KB)")

    print("rendering pyvista ...")
    render_pyvista(f, N, title, out_pv_png, out_pv_html)
    print(f"  saved {out_pv_png}  ({out_pv_png.stat().st_size/1024:.1f} KB)")
    print(f"  saved {out_pv_html} ({out_pv_html.stat().st_size/1024:.1f} KB)")

    print("rendering plotly ...")
    render_plotly(f, N, title, out_pl_html)
    print(f"  saved {out_pl_html} ({out_pl_html.stat().st_size/1024:.1f} KB)")
