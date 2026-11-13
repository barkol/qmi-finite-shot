"""Drawing layer (matplotlib 3D) for fig_gauss_overview (Fig. 1). Reads only the cached npz.
Heterodyne: Wong vermillion, filled markers, solid lines; homodyne: Wong blue, open markers, dashed lines (BW-safe)."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from dane_fig_gauss_overview import load
import figstyle as fs
fs.apply()
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
z = load(); HET, HOM = fs.ORANGE, fs.BLUE
fig = plt.figure(figsize=(fs.COL2, 2.95))  # two-column figure
def clean(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor("white"); a.pane.set_edgecolor("white"); a._axinfo["grid"]["linewidth"] = 0.25; a._axinfo["grid"]["color"] = fs.FAINT
    ax.tick_params(pad=-3, labelsize=6.5); ax.patch.set_alpha(0)
# ------------------------------------------------------------------ (a) state, change, measurement
ax = fig.add_axes([-0.085, -0.06, 0.37, 1.06], projection="3d", computed_zorder=False)
x = z["a_x"]; X, P = np.meshgrid(x, x); W0, WD, V = z["a_W0"], z["a_WD"], z["a_V"]
s0 = 1.0 / W0.max(); zf, z1, z2 = 0.0, 1.0, 2.55                         # floor, base of W0, base of W_Delta
Rm = 3.3; out = X ** 2 + P ** 2 > Rm ** 2; W0 = np.where(out, np.nan, W0); WD = np.where(out, np.nan, WD)
ax.plot_surface(X, P, z1 + 0.9 * s0 * W0, rstride=5, cstride=5, color="#f2f2f2", edgecolor=fs.GREY, linewidth=0.22, alpha=0.8, shade=False, zorder=3)
wm = np.nanmax(np.abs(WD)); nrm = colors.TwoSlopeNorm(vmin=-wm, vcenter=0, vmax=wm); fc = cm.PuOr_r(nrm(np.nan_to_num(WD))); fc[..., 3] = np.clip(0.25 + 2.5 * np.abs(np.nan_to_num(WD)) / wm, 0, 1)
ax.plot_surface(X, P, z2 + 0.6 * WD / wm, rstride=3, cstride=3, facecolors=fc, edgecolor="none", linewidth=0, antialiased=True, shade=False, zorder=5)
t = np.linspace(0, 2 * np.pi, 200)
ax.plot(Rm * np.cos(t), Rm * np.sin(t), zf, color=fs.FAINT, lw=0.6, zorder=0)
ax.plot(np.sqrt(V[0, 0]) * np.cos(t), np.sqrt(V[1, 1]) * np.sin(t), zf, color=fs.INK, lw=1.0, zorder=1)                    # state: V
ax.plot(np.sqrt(0.5) * np.cos(t), np.sqrt(0.5) * np.sin(t), zf, color=HET, lw=1.0, zorder=1)                                 # heterodyne noise
th = np.deg2rad(35); L = 3.0
ax.plot([-L * np.cos(th), L * np.cos(th)], [-L * np.sin(th), L * np.sin(th)], [zf, zf], color=HOM, lw=1.0, ls=(0, (4, 2)), zorder=1)  # homodyne quadrature
ax.add_artist(fs.Arrow3D([0.82 * L * np.cos(th), L * np.cos(th)], [0.82 * L * np.sin(th), L * np.sin(th)], [zf, zf], mutation_scale=6, arrowstyle="-|>", color=HOM, lw=0.8, zorder=2))
ax.text(0.2, -2.25, zf, r"$V$", color=fs.INK, fontsize=8, ha="center"); ax.text(-1.25, -0.2, zf, r"$\Sigma$", color=HET, fontsize=8, ha="center")
ax.text(-0.2, 1.3, zf, r"$\mathbf{u}$", color=HOM, fontsize=8, zorder=20)
ax.text(3.4, -1.0, z1 + 0.2, r"$W_0$", fontsize=8, zorder=20); ax.text(3.4, -1.0, z2 + 0.2, r"$W_\Delta$", fontsize=8, zorder=20)
ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6); ax.set_zlim(0, 3.2); ax.set_box_aspect((1, 1, 1.25)); ax.view_init(elev=20, azim=-58); ax.set_axis_off()
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); clean(ax)
ax.text(3.55, -0.5, zf, r"$x$", fontsize=8, zorder=20); ax.text(-0.3, 3.6, zf, r"$p$", fontsize=8, zorder=20)
ax.plot([-Rm, Rm], [0, 0], [zf, zf], color=fs.FAINT, lw=0.5, zorder=0); ax.plot([0, 0], [-Rm, Rm], [zf, zf], color=fs.FAINT, lw=0.5, zorder=0)
fig.text(0.012, 0.93, r"(a)", fontsize=8)
# ------------------------------------------------------------------ (b) mode lattice near a thermal state
ax = fig.add_axes([0.155, -0.05, 0.47, 1.02], projection="3d", computed_zorder=False)
M = z["b_modes"]; lam = float(z["b_lambda"]); nmax = int(M[:, 0].max())
tri = [(-nmax - 0.6, nmax + 0.6), (nmax + 0.6, nmax + 0.6), (0, 0.4)]
for h, c, a in ((0.5, HOM, 0.10), (lam, HET, 0.10)):
    ax.add_collection3d(Poly3DCollection([[(l, n, h) for l, n in tri]], facecolor=c, edgecolor=c, linewidths=0.5, alpha=a, zorder=1))
for n in range(1, nmax + 1): ax.plot([-n, n], [n, n], [0, 0], color=fs.FAINT, lw=0.5, zorder=0)
ax.plot([0, -nmax], [0, nmax], [0, 0], color=fs.GREY, lw=0.5, zorder=0); ax.plot([0, nmax], [0, nmax], [0, 0], color=fs.GREY, lw=0.5, zorder=0)
dx = 0.17
for n, l, eq, ew in M[np.argsort(-M[:, 0])]:                      # far rows first
    ax.plot([l - dx, l - dx], [n, n], [0, eq], color=HET, lw=1.1, zorder=3); ax.plot([l + dx, l + dx], [n, n], [0, ew], color=HOM, lw=0.9, ls=(0, (2.5, 1.2)), zorder=3)
    ax.scatter([l - dx], [n], [eq], s=9, color=HET, edgecolor=HET, linewidth=0.5, depthshade=False, zorder=4)
    ax.scatter([l + dx], [n], [ew], s=9, facecolor="white", edgecolor=HOM, linewidth=0.7, depthshade=False, zorder=4)
ax.text(-nmax - 1.9, nmax + 0.6, lam, r"$\lambda$", color=HET, fontsize=8, va="center"); ax.text(-nmax - 1.9, nmax + 0.6, 0.5, r"$\frac12$", color=HOM, fontsize=8, va="center")
ax.set_xlim(-nmax - 0.6, nmax + 0.6); ax.set_ylim(0.4, nmax + 0.6); ax.set_zlim(0, 0.8); ax.set_box_aspect((1.75, 1.25, 0.95)); ax.view_init(elev=22, azim=-64)
ax.set_xticks(range(-8, 9, 4)); ax.set_yticks([2, 4, 6, 8]); ax.set_zticks([0, 0.25, 0.5, 0.75]); clean(ax)
ax.set_xlabel(r"angular momentum $l$", labelpad=-7); ax.set_ylabel(r"order $n$", labelpad=-7); ax.set_zlabel(r"efficiency $F/F_{\mathrm{SLD}}$", labelpad=-5)
fig.text(0.225, 0.93, r"(b)", fontsize=8)
# ------------------------------------------------------------------ (c) squeezed thermal state
ax = fig.add_axes([0.625, -0.05, 0.36, 1.02], projection="3d", computed_zorder=False)
R, Nb = np.meshgrid(z["c_r"], z["c_nb"])
ax.plot_surface(R, Nb, 0 * R + 0.5, color=HOM, alpha=0.08, edgecolor="none", shade=False, zorder=1)
ax.plot_wireframe(R, Nb, z["c_hom"], rstride=4, cstride=5, color=HOM, linewidth=0.6, linestyle=(0, (3, 1.5)), zorder=2)
ax.plot_surface(R, Nb, z["c_het"], color=HET, alpha=0.22, edgecolor="none", shade=False, zorder=3)
ax.plot_wireframe(R, Nb, z["c_het"], rstride=4, cstride=5, color=HET, linewidth=0.7, zorder=4)
ax.plot(z["c_r"], 0 * z["c_r"], z["c_hom"][0], color=fs.INK, lw=1.6, zorder=6)                                    # nbar = 0: the bounds coincide
ax.plot(z["c_r"], 0 * z["c_r"], 0.5 + 0 * z["c_r"], color=HOM, lw=0.8, ls=":", zorder=5); ax.plot(z["c_r"], 2 + 0 * z["c_r"], 0.5 + 0 * z["c_r"], color=HOM, lw=0.8, ls=":", zorder=1)
ax.text(0.75, 2.0, 0.53, r"$\frac12$", color=HOM, fontsize=8, va="center", zorder=20); ax.text(0.02, 2.0, 0.89, r"$\lambda_r$", color=HET, fontsize=8); ax.text(0.75, 0, 0.60, r"$\frac{1+\tanh|r|}{2}$", color=fs.INK, fontsize=8, zorder=20)
ax.set_xlim(0, 1.5); ax.set_ylim(0, 2); ax.set_zlim(0.4, 1.0); ax.set_box_aspect((1.25, 1.25, 1.0), zoom=0.76); ax.view_init(elev=22, azim=-52)
ax.set_xticks([0, 0.5, 1, 1.5]); ax.set_yticks([0, 1, 2]); ax.set_zticks([0.5, 0.75, 1.0]); clean(ax)
ax.set_xlabel(r"squeezing $|r|$", labelpad=-7); ax.set_ylabel(r"occupation $\bar n$", labelpad=-7); ax.set_zlabel(r"largest efficiency", labelpad=-5)
fig.text(0.675, 0.93, r"(c)", fontsize=8)
# legend
from matplotlib.lines import Line2D
h = [Line2D([], [], color=HET, lw=1.1, marker="o", ms=3.2, label="heterodyne"), Line2D([], [], color=HOM, lw=0.9, ls=(0, (2.5, 1.2)), marker="o", ms=3.2, mfc="white", label="homodyne")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, 1.0), ncol=2, columnspacing=1.6, handlelength=2.2)
fig.savefig(os.path.join(HERE, "fig_gauss_overview.pdf")); fig.savefig(os.path.join(HERE, "fig_gauss_overview.png"), dpi=170)
