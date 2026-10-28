"""Figury z out/maps.npz + out/summary.json (bez obliczen).  python analyses/shap/shap_plot.py [--small]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ.get("FIGSTYLE_DIR", str(Path(__file__).resolve().parents[2] / "paper" / "figures")))
import figstyle as fs
fs.apply()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

HERE = Path(__file__).resolve().parent
FAMS = ("fock", "coherent", "sqvac", "thermal", "cat", "gkp", "binomial")
SHOW = ("cat", "fock", "sqvac", "gkp", "binomial")
LAB = {"cat": "cat", "fock": "Fock", "sqvac": "sq. vacuum", "gkp": "GKP", "binomial": "binomial", "coherent": "coherent", "thermal": "thermal"}
G, XM, K = 128, 12.0, 36; x = np.linspace(-XM, XM, G); CROP = 7.5; ix = np.where(np.abs(x) <= CROP)[0]
# rozbiezna Wong: niebieski - bialy - pomaranczowy (znak dodatkowo kodowany stylem konturu)
DIV = LinearSegmentedColormap.from_list("wongdiv", [fs.BLUE, "#ffffff", fs.ORANGE])
SEQ = LinearSegmentedColormap.from_list("wongseq", ["#ffffff", fs.SKY, fs.BLUE, "#0b2545"])   # monotoniczna w luminancji, biala w zerze

def crop(M, rep):
    return M[:, ix] if rep == "S" else M[np.ix_(ix, ix)]

def show(ax, M, rep, state=None, signed=True):
    M = crop(M, rep); ext = (-CROP, CROP, 0, 180) if rep == "S" else (-CROP, CROP, -CROP, CROP)
    v = np.abs(M).max() or 1
    if signed:
        ax.imshow(M, cmap=DIV, vmin=-v, vmax=v, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
        X, Y = np.meshgrid(np.linspace(ext[0], ext[1], M.shape[1]), np.linspace(ext[2], ext[3], M.shape[0]))
        ax.contour(X, Y, M, levels=[0.3 * v], colors="k", linewidths=0.45, linestyles="solid")
        ax.contour(X, Y, M, levels=[-0.3 * v], colors="k", linewidths=0.45, linestyles="dashed")
    else:
        ax.imshow(M / v, cmap=SEQ, vmin=0, vmax=1, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
        X, Y = np.meshgrid(np.linspace(ext[0], ext[1], M.shape[1]), np.linspace(ext[2], ext[3], M.shape[0]))
    if state is not None:
        S = crop(state, rep); s = np.abs(S).max()
        ax.contour(X, Y, np.abs(S), levels=[0.15 * s], colors=[fs.GREEN], linewidths=0.6, linestyles="dotted")
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_visible(True); sp_.set_linewidth(0.4)

def gray(png):
    Image.open(png).convert("L").save(str(png).replace(".png", "_gray.png"))

def fig_maps(D, R, out, cell="clean", kind="rec", base="other"):
    fig = plt.figure(figsize=(7.0, 2.55))
    gl = fig.add_gridspec(4, 5, left=0.045, right=0.585, top=0.93, bottom=0.03, wspace=0.06, hspace=0.08)
    rows = [("Q_s0", "Q", r"$Q$ (het.)"), ("W_s0", "W", r"$W$ (FBP)"), ("S_s0", "S", r"$S$ (sino.)"), ("HQ", "Q", r"$H_Q$ (ideal)")]
    for r, (nm, rep, lab) in enumerate(rows):
        for c, fam in enumerate(SHOW):
            ci = FAMS.index(fam); ax = fig.add_subplot(gl[r, c])
            st = D[f"{cell}|state_{'S' if rep == 'S' else ('Q' if nm != 'W_s0' else 'W')}"][ci]
            if nm == "HQ": show(ax, D[f"{cell}|HQ"][ci], "Q", state=st, signed=False)
            else: show(ax, D[f"{cell}|{nm}|{kind}|{base}|signed"][ci], rep, state=st)
            if r == 0: ax.set_title(LAB[fam], pad=2)
            if c == 0: ax.set_ylabel(lab, fontsize=7, labelpad=2)
    fs.panel(fig.axes[0], "(a)", x=-0.25, y=1.12)
    # (b) udzial |SHAP| poza nosnikiem stanu; (c) Spearman z mapa idealnego obserwatora
    A = R["attr"]; San = R.get("sanity", {})
    ax = fig.add_axes([0.665, 0.60, 0.325, 0.33]); xs = np.arange(len(FAMS)); w = 0.26
    for k, (nm, col, mk, hatch) in enumerate((("Q_s0", fs.BLUE, "o", ""), ("W_s0", fs.ORANGE, "s", "////"), ("S_s0", fs.GREEN, "^", "...."))):
        v = [A[f"{cell}|{nm}|{kind}|{base}"]["out_support"][f] for f in FAMS]
        ax.bar(xs + (k - 1) * w, v, w, color=col, edgecolor=fs.INK, linewidth=0.4, hatch=hatch, label=nm[0])
        rnd = San.get(f"{nm[0]}|{kind}", {}).get("out_support")
        if rnd: ax.axhline(rnd[-1], color=col, lw=0.8, ls=(0, (2, 1.5)))
    ax.set_xticks(xs); ax.set_xticklabels([LAB[f][:5] for f in FAMS], rotation=45, fontsize=6, ha="right")
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"]); ax.set_ylabel(r"$|\mathrm{SHAP}|$ off-support", fontsize=7)
    ax.legend(ncol=3, loc="upper left", fontsize=6, handlelength=1.0, columnspacing=0.6, borderaxespad=0.1)
    fs.panel(ax, "(b)", x=-0.17)
    ax = fig.add_axes([0.665, 0.12, 0.325, 0.30])
    for k, (nm, col, mk, ls) in enumerate((("Q_s0", fs.BLUE, "o", "-"), ("S_s0", fs.GREEN, "^", "--"))):
        v = [A[f"{cell}|{nm}|{kind}|{base}"]["spear_hell_supp"][f] for f in FAMS]
        ax.plot(xs, v, color=col, marker=mk, ms=3.2, lw=0.9, ls=ls, label={"Q_s0": r"$Q$ vs $H_Q$", "S_s0": r"$S$ vs $H_S$"}[nm])
    X_ = R.get("extra", {}).get(f"{cell}|stateQ_vs_HQ")
    if X_: ax.plot(xs, [X_[f] for f in FAMS], color=fs.GREY, marker="x", ms=3, lw=0.7, ls=":", label=r"state $Q$ vs $H_Q$")
    ax.axhline(0, color=fs.GREY, lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([LAB[f][:5] for f in FAMS], rotation=45, fontsize=6, ha="right")
    ax.set_ylim(-0.2, 1.25); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"]); ax.set_ylabel(r"Spearman $\rho$", fontsize=7)
    ax.legend(loc="upper left", fontsize=5.6, ncol=3, handlelength=1.3, columnspacing=0.6, borderaxespad=0.1)
    fs.panel(ax, "(c)", x=-0.17)
    for ext in ("pdf", "png"): fig.savefig(out.with_suffix("." + ext))
    gray(out.with_suffix(".png")); plt.close(fig)

def fig_sanity(D, R, out, kind="rec"):
    fig = plt.figure(figsize=(fs.COL1, 2.5))
    ax = fig.add_axes([0.14, 0.50, 0.83, 0.45]); st = np.arange(len(R["cascade"]))
    for nm, col, mk in (("Q", fs.BLUE, "o"), ("W", fs.ORANGE, "s"), ("S", fs.GREEN, "^")):
        s = R["sanity"].get(f"{nm}|{kind}"); g = R["sanity"].get(f"{nm}|{kind}_G")
        if s: ax.plot(st, s["abs"], color=col, marker=mk, ms=2.8, lw=0.9, ls="--")
        if s: ax.plot(st, s["signed"], color=col, marker=mk, ms=2.8, lw=0.9, ls="-", label=nm)
        if g: ax.plot(st, g["abs"], color=col, marker=mk, ms=2.8, lw=0.9, ls=":", mfc="white")
    ax.plot([], [], "k--", lw=0.9, label=r"$|\mathrm{SHAP}|$"); ax.plot([], [], "k-", lw=0.9, label="signed")
    ax.plot([], [], "k:", lw=0.9, label=r"$|\mathrm{EG}|$")
    ax.axhline(0, color=fs.GREY, lw=0.5); ax.set_ylim(-0.6, 1.05); ax.set_yticks([-0.5, 0, 0.5, 1]); ax.set_yticklabels(["-0.5", "0", "0.5", "1"])
    ax.set_xticks(st); ax.set_xticklabels(["none", "fc2", "fc1", "b4", "b3", "b2", "b1"], fontsize=6)
    ax.set_xlabel("randomized up to (cascade, top first)", fontsize=7, labelpad=1); ax.set_ylabel(r"rank corr. with trained", fontsize=7)
    ax.legend(ncol=3, fontsize=5.8, loc="lower left", handlelength=1.6, columnspacing=0.6, borderaxespad=0.1)
    fs.panel(ax, "(a)", x=-0.10)
    # (b) przyklad: kot, siec W, rekord N=100: wytrenowana / gora losowa / calosc losowa
    j = FAMS.index("cat")
    for k, step in enumerate((0, 2, 6)):
        a = fig.add_axes([0.20 + k * 0.26, 0.015, 0.22, 0.22 * fs.COL1 / 2.5])
        M = D[f"sanity|W|{kind}|{step}"][j]; show(a, M, "W")
        a.set_title(["trained", "head rand.", "all rand."][k], fontsize=6, pad=1)
    fs.panel(fig.axes[1], "(b)", x=-0.12, y=0.85)
    for ext in ("pdf", "png"): fig.savefig(out.with_suffix("." + ext))
    gray(out.with_suffix(".png")); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--small", action="store_true"); a = ap.parse_args()
    O = HERE / ("out_small" if a.small else "out"); D = dict(np.load(O / "maps.npz")); R = json.load(open(O / "summary.json"))
    if (O / "extra.json").exists(): R["extra"] = json.load(open(O / "extra.json"))
    tag = "_small" if a.small else ""
    fig_maps(D, R, HERE / f"fig_shap_maps{tag}")
    fig_maps(D, R, HERE / f"fig_shap_maps_deph20{tag}", cell="deph20")
    fig_maps(D, R, HERE / f"fig_shap_maps_EG{tag}", base="otherG")
    fig_sanity(D, R, HERE / f"fig_shap_sanity{tag}")
    print("PLOT_DONE")
