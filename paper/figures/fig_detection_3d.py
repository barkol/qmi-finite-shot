"""3D surfaces (each panel has its own z scale) of phase-space estimates from finite heterodyne and homodyne records.

Row 1: heterodyne Q estimate (histogram of N events drawn from the exact Q image, qmi.shots.heterodyne_image).
Row 2: homodyne W estimate (N events over K=36 phases + filtered back-projection, qmi.shots.homodyne_wigner).
Columns: N = 1e3, 1e5, N -> inf (exact Q; homodyne_wigner_exact with K=36).
State: even cat |alpha> + |-alpha>, alpha = 2 (real), Fock cutoff 64. Grid 128 x 128 on [-12, 12]^2 (QMI_XMAX=12),
QuTiP convention alpha = (x + i p)/sqrt(2). The display crops the estimator's grid (no re-gridding).

Data separate from drawing: arrays are computed only if figures/data/fig_detection_3d.npz is missing (or --recompute);
the drawing layer only loads the npz.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "fig_detection_3d.npz")
QMI_REPO = os.environ.get("QMI_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
STYLE_DIR = os.environ.get("FIGSTYLE_DIR", os.path.dirname(os.path.abspath(__file__)))   # read-only: figstyle.py template

GRID, XMAX, ALPHA, CUTOFF, K = 128, 12.0, 2.0, 64, 36
NS = (1000, 100000)                       # finite records; third column = N -> inf
SEEDS = {("Q", 1000): 1103, ("Q", 100000): 1105, ("W", 1000): 2103, ("W", 100000): 2105}   # one fixed seed per panel
LDISP = 4.0                               # display window [-LDISP, LDISP]^2


def build():
    os.environ["QMI_XMAX"] = str(int(XMAX))
    sys.path.insert(0, QMI_REPO)
    import qmi                            # must precede any qsdl import (sets the phase-space window)
    import qutip as qt
    from qmi import shots
    from qsdl.wigner import apply_wigner
    from qsdl.husimi import apply_husimi
    assert shots._XMAX == XMAX, shots._XMAX
    psi = (qt.coherent(CUTOFF, ALPHA) + qt.coherent(CUTOFF, -ALPHA)).unit()
    rho = qt.ket2dm(psi)
    W = apply_wigner(rho, GRID, XMAX).astype(np.float64)
    Q = apply_husimi(rho, GRID, XMAX).astype(np.float64)
    d = {"x": np.linspace(-XMAX, XMAX, GRID), "Q_inf": Q, "W_inf_exact": W,
         "W_inf": shots.homodyne_wigner_exact(W, K).astype(np.float64)}
    for N in NS:
        d[f"Q_{N}"] = shots.heterodyne_image(Q, N, np.random.default_rng(SEEDS["Q", N])).astype(np.float64)
        d[f"W_{N}"] = shots.homodyne_wigner(W, N, np.random.default_rng(SEEDS["W", N]), n_angles=K).astype(np.float64)
    d["seeds"] = np.array([[SEEDS["Q", N] for N in NS], [SEEDS["W", N] for N in NS]])
    d["meta"] = np.array(f"even cat alpha={ALPHA}, cutoff={CUTOFF}, grid={GRID}, xmax={XMAX}, K={K}")
    return d


def load():
    if os.path.exists(DATA) and "--recompute" not in sys.argv:
        return dict(np.load(DATA, allow_pickle=True))
    d = build(); os.makedirs(os.path.dirname(DATA), exist_ok=True); np.savez(DATA, **d); return d


# ----------------------------------------------------------------------------- drawing (loads npz only)
def draw(d, cmap_name="cividis"):
    sys.path.insert(0, STYLE_DIR)
    import figstyle as fs
    fs.apply()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, Normalize, ListedColormap
    from matplotlib.ticker import FuncFormatter
    zfmt = FuncFormatter(lambda v, _: "$" + fs._strip_zeros(v) + "$")     # math minus sign
    civ = plt.get_cmap(cmap_name)   # --cmap (domyslnie cividis; np. jet)
    civ_q = ListedColormap(civ(np.linspace(0.22, 1.0, 256))) if cmap_name == "cividis" else civ   # Q: lift the zero level off the darkest blue
    sys.path.insert(0, HERE)
    from figlook import big2mm                                            # digits/capitals >= 2 mm in print
    big2mm(plt, {"xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10})

    x = d["x"]; m = np.abs(x) <= LDISP + 1e-9; xs = x[m]
    X, P = np.meshgrid(xs, xs)            # qt.wigner/qfunc: rows = p, columns = x
    crop = lambda A: A[np.ix_(m, m)]
    cols = [f"{N}" for N in NS] + ["inf"]
    rows = [("Q", r"heterodyne, $Q$", civ_q), ("W", r"homodyne, $W$ (FBP)", civ)]

    def nice(v):                          # largest 1/2/2.5/5 x 10^k not exceeding v
        k = np.floor(np.log10(v)); c = [b * 10 ** k for b in (1, 2, 2.5, 5, 10) if b * 10 ** k <= v * (1 + 1e-9)]
        return round(max(c), 6)

    def zscale(Z, rep):                   # per-panel z range fitted to the data shown in that panel
        hi = float(Z.max()); lo = float(Z.min())
        if rep == "Q":
            return (0.0, 1.06 * hi), [0, nice(hi)]
        return (1.06 * lo, 1.06 * hi), [-nice(-lo), 0, nice(hi)]
    ctitle = [r"$N=10^3$", r"$N=10^5$", r"$N\to\infty$"]
    ls = LightSource(azdeg=315, altdeg=50)
    elev, azim = 35, -30

    fig = plt.figure(figsize=(fs.COL2, 4.1))
    L, R, B, T, HG, VG = 0.03, 0.975, 0.015, 0.975, -0.01, 0.0
    w = (R - L - 2 * HG) / 3; h = (T - B - VG) / 2
    letters = iter("abcdef")
    for i, (rep, rlab, cmap) in enumerate(rows):
        for j, c in enumerate(cols):
            ax = fig.add_axes([L + j * (w + HG), T - (i + 1) * h - i * VG, w, h], projection="3d")
            Z = crop(d[f"{rep}_{c}"]); zl, zt = zscale(Z, rep); norm = Normalize(*zl)
            Xs, Ps = X, P
            if rep == "W":                # kontur obszarow W < -0.1 max|W| MALOWANY na teksturze powierzchni, zeby zaslanial go
                # ten sam grzbiet co powierzchnie (linie 3D w mpl nie maja testu glebokosci); siatka 3x gestsza (biliniowo, bez nowych ekstremow)
                from scipy.ndimage import zoom, binary_erosion
                lev = -0.1 * float(np.abs(Z).max())
                Z = zoom(Z, 3, order=1); Xs = zoom(X, 3, order=1); Ps = zoom(P, 3, order=1)
            rgb = ls.shade(Z, cmap=cmap, norm=norm, blend_mode="soft", vert_exag=6.0 / (zl[1] - zl[0]))
            if rep == "W":
                neg = Z < lev
                ii, jj = np.indices(Z.shape); hatch = ((ii + jj) // 2) % 2 == 0   # ukosne paski co 2 komorki w obszarze ujemnym
                rgb[neg & hatch, :3] *= 0.45
            ax.plot_surface(Xs, Ps, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False,
                            shade=False, rasterized=True)
            ax.set_proj_type("persp", focal_length=0.6)
            ax.set_xlim(-LDISP, LDISP); ax.set_ylim(-LDISP, LDISP); ax.set_zlim(*zl)
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((1, 1, 0.7), zoom=0.9); ax.set_facecolor((1, 1, 1, 0))
            for a in (ax.xaxis, ax.yaxis, ax.zaxis):
                a.set_pane_color((0.96, 0.96, 0.96, 1.0)); a.pane.set_edgecolor(fs.FAINT)
                a._axinfo["grid"].update(linewidth=0.25, color=(0.85, 0.85, 0.85, 1.0))
                a._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.25)
                a.line.set_linewidth(0.4); a.line.set_color(fs.GREY)
            ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3]); ax.set_zticks(zt)
            ax.zaxis.set_major_formatter(zfmt); ax.xaxis.set_major_formatter(zfmt); ax.yaxis.set_major_formatter(zfmt)
            ax.tick_params(pad=-3.5, length=1.5, width=0.4)
            ax.tick_params(axis="z", pad=0.0)
            ax.set_xlabel(r"$x$", labelpad=-5); ax.set_ylabel(r"$p$", labelpad=-5)
            ax.text2D(0.07, 0.97, rf"\textbf{{({next(letters)})}}", transform=ax.transAxes, ha="left", va="top", fontsize=10)
            if i == 0:
                fig.text(L + j * (w + HG) + 0.5 * w, 0.995, ctitle[j], ha="center", va="top", fontsize=10)
        fig.text(0.012, T - (i + 0.5) * h - i * VG, rlab, rotation=90, ha="center", va="center", fontsize=10)
    return fig, plt


if __name__ == "__main__":
    cm = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--cmap=")), "cividis")
    d = load()
    fig, plt = draw(d, cm)
    out = os.path.join(HERE, "fig_detection_3d" + ("" if cm == "cividis" else "_" + cm))
    fig.savefig(out + ".pdf", dpi=400); fig.savefig(out + ".png", dpi=300)
    from PIL import Image
    Image.open(out + ".png").convert("L").save(out + "_gray.png")
    print(out + ".{pdf,png}", "+ _gray.png")
