"""Drawing layer (matplotlib 3D) for fig_modes. Reads only the cached npz.
Each panel is a surface plot of the change of the Wigner function, W_Delta = g W_0, of one Hermite-Gauss mode (n, l).
Palette TURBO centred at zero, as in Fig. 1a (red: positive, blue: negative); the height carries the same information
as the colour, so the figure is readable in grey scale."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from data_fig_modes import load
import figstyle as fs
fs.apply()
import matplotlib.pyplot as plt
from matplotlib import cm
z = load(); x = z["x"]; st = 3; X, P = np.meshgrid(x[::st], x[::st]); Rm = 3.6
fig = plt.figure(figsize=(fs.COL1, 2.25)); NC, NR = 4, 2; w, h = 1.0 / NC, 0.5
t = np.linspace(0, 2 * np.pi, 121)
for i, (n, l) in enumerate(z["modes"]):
    r, c = divmod(i, NC)
    ax = fig.add_axes([c * w - 0.035, (NR - 1 - r) * h - 0.075, w + 0.07, h + 0.09], projection="3d", computed_zorder=False)
    W = z[f"W_{n}_{l}"][::st, ::st]; m = np.abs(W).max(); out = X ** 2 + P ** 2 > Rm ** 2
    Z = np.where(out, np.nan, W / m); fc = cm.turbo(0.5 + 0.5 * np.nan_to_num(Z))
    ax.plot(Rm * np.cos(t), Rm * np.sin(t), -1.0, color=fs.FAINT, lw=0.5, zorder=0)                      # floor: edge of the plotted disk
    ax.plot([-Rm, Rm], [0, 0], [-1.0, -1.0], color=fs.FAINT, lw=0.4, zorder=0); ax.plot([0, 0], [-Rm, Rm], [-1.0, -1.0], color=fs.FAINT, lw=0.4, zorder=0)
    ax.plot_surface(X, P, Z, rstride=1, cstride=1, facecolors=fc, edgecolor="none", linewidth=0, antialiased=False, shade=False, rasterized=True, zorder=2)
    ax.plot_wireframe(X, P, Z, rstride=9, cstride=9, color=fs.INK, linewidth=0.18, alpha=0.55, zorder=3)
    ax.set_xlim(-Rm, Rm); ax.set_ylim(-Rm, Rm); ax.set_zlim(-1.0, 1.0); ax.set_box_aspect((1, 1, 0.62)); ax.view_init(elev=32, azim=-58)
    ax.set_axis_off(); ax.patch.set_alpha(0)
    fig.text((c + 0.5) * w, (NR - r) * h - 0.035, rf"$({n},{l})$", ha="center", va="top", fontsize=9)
fig.savefig(f"{HERE}/fig_modes.pdf", dpi=600)
