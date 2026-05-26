#!/usr/bin/env python3
"""Plot an SU(2) FFT spectrum dumped by bench/vis_dump.

Usage:  python3 bench/visualize.py spectrum.txt output.png

Top panel:  ||fhat(l)||_F  (Frobenius norm of the (2l+1)x(2l+1) matrix)
            vs l, on a log scale, paper Figure 1 style.

Bottom row: per-l heat-maps of |fhat(l)[m,n]| for l in [0, L_max].
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_dump(path):
    rows = []
    expected = []
    N = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith("# N"):
                N = int(line.split()[2])
            elif line.startswith("#   l=") and "expected_fhat=" in line:
                expected.append(line.strip("# ").strip())
            elif line.startswith("#") or not line:
                continue
            else:
                parts = line.split()
                rows.append((int(parts[0]), int(parts[1]), int(parts[2]),
                             float(parts[3]), float(parts[4])))
    return N, rows, expected


def into_matrices(N, rows):
    fhat = {l: np.zeros((2*l+1, 2*l+1), dtype=complex) for l in range(N)}
    for (l, m, n, re, im) in rows:
        fhat[l][m + l, n + l] = re + 1j*im
    return fhat


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    txt_path, out_path = sys.argv[1], sys.argv[2]
    N, rows, expected = load_dump(txt_path)
    fhat = into_matrices(N, rows)

    ls = list(range(N))
    fro = [np.linalg.norm(fhat[l], 'fro') for l in ls]

    L_show = min(N, 8)               # heat-maps for l = 0 .. L_show-1
    fig = plt.figure(figsize=(10, 6.5), constrained_layout=True)
    gs  = fig.add_gridspec(2, L_show, height_ratios=[1.2, 1])

    # Top: Frobenius spectrum
    ax0 = fig.add_subplot(gs[0, :])
    bars = ax0.bar(ls, fro, width=0.7, color='#2266aa')
    ax0.set_yscale('log')
    ax0.set_xlabel('l')
    ax0.set_ylabel(r'$\|\hat f(l)\|_F$')
    ax0.set_title(f'SU(2) Fourier spectrum (N={N})')
    ax0.grid(True, axis='y', alpha=0.3, which='both')
    ax0.set_xticks(ls)
    # Annotate where the spectrum is "real".
    floor = max(min([f for f in fro if f > 0], default=1e-20), 1e-20)
    ax0.set_ylim(bottom=max(floor * 0.3, 1e-20))

    # Bottom: heat-maps of |fhat(l)[m,n]|
    vmax = max(np.abs(fhat[l]).max() for l in range(L_show))
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    for l in range(L_show):
        ax = fig.add_subplot(gs[1, l])
        size = 2*l + 1
        im   = ax.imshow(np.abs(fhat[l]), cmap=cmap, norm=norm,
                         extent=[-l-0.5, l+0.5, l+0.5, -l-0.5])
        ax.set_title(f'l = {l}\n({size}x{size})', fontsize=9)
        ticks = list(range(-l, l+1)) if size <= 7 else [-l, 0, l]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xlabel('n'); ax.set_ylabel('m')
        ax.tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=fig.axes[1:], shrink=0.8, label=r'$|\hat f(l)_{m,n}|$')

    # Footer: list expected modes (rendered as plain text below the figure)
    if expected:
        footer = "Input modes (synthesised f):\n" + "\n".join(expected)
        fig.text(0.01, -0.02, footer, fontsize=7, family='monospace',
                 verticalalignment='top')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
