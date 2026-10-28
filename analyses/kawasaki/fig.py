"""Figura szkicowa: rekordy PSS Kawasaki+ 2024. Dane -> out/fig_data.npz (licz-jesli-brak, z out/summary.json i out/*/ml.npz), potem rysunek.
  python -I analyses/kawasaki/fig.py [--przelicz]"""
import json, os, sys
from pathlib import Path
import numpy as np

OUT = Path(str(Path(__file__).resolve().parent / "out")); CACHE = OUT / "fig_data.npz"
LOSS = ["01mW_00dB_2", "01mW_01dB_2", "01mW_02dB_2", "01mW_05dB_2", "01mW_10dB_2", "01mW_15dB_2", "01mW_20dB_2"]
DB = np.array([0, 1, 2, 5, 10, 15, 20.0])
MAPS = [("01mW_00dB_2", r"1 mW, 0 dB"), ("01mW_10dB_2", r"1 mW, 10 dB"), ("01mW_20dB_2", r"1 mW, 20 dB"), ("25mW_00dB", r"25 mW, 0 dB")]
NN = ("100", "1000", "10000")

if "--przelicz" in sys.argv or not CACHE.exists():
    S = json.load(open(OUT / "summary.json"))["pliki"]; XM, G = 12.0, 128
    y = np.linspace(-XM, XM, G); m = np.abs(y) <= 3.6
    d = dict(y=y[m], db=DB)
    for i, (t, _) in enumerate(MAPS): d[f"map{i}"] = np.load(OUT / t / "ml.npz")["W"][np.ix_(m, m)]
    d["W0"] = np.array([S[t]["ml"]["W0"] for t in LOSS]); d["W0_sd"] = np.array([S[t]["ml"]["boot_sd"]["W0"] for t in LOSS])
    d["Wmin"] = np.array([S[t]["ml"]["Wmin"] for t in LOSS]); d["Wmin_sd"] = np.array([S[t]["ml"]["boot_sd"]["Wmin"] for t in LOSS])
    for N in NN:
        d[f"fock_{N}"] = np.array([S[t]["klasyf_baza"][N]["frac"]["exp_los"]["fock"] for t in LOSS])
        d[f"fock_sym_{N}"] = np.array([S[t]["klasyf_baza"][N]["frac"]["sym"]["fock"] for t in LOSS])
        d[f"neg_{N}"] = np.array([S[t]["negN"][f"N{N}"]["frac_W0_neg"] for t in LOSS])
    d["fock_full"] = np.array([np.mean([c == "fock" for c in S[t]["klasyf_baza"]["0"]["full_class"]]) for t in LOSS])
    d["kurt"] = np.array([np.mean(S[t]["kurt"]) for t in LOSS])
    np.savez(CACHE, **d)

D = dict(np.load(CACHE))
sys.path.insert(0, os.environ.get("FIGSTYLE_DIR", str(Path(__file__).resolve().parents[2] / "paper" / "figures"))); import figstyle as fs   # tylko odczyt wzorca
fs.apply(); import matplotlib.pyplot as plt
fig = plt.figure(figsize=(fs.COL2, 3.9)); gs = fig.add_gridspec(2, 12, height_ratios=[1, 1.05], hspace=0.55, wspace=2.2)
vm = 0.20; ext = [D["y"][0], D["y"][-1], D["y"][0], D["y"][-1]]
for i, (t, lab) in enumerate(MAPS):
    ax = fig.add_subplot(gs[0, 3 * i:3 * i + 3]); W = D[f"map{i}"]
    im = ax.imshow(W, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="bilinear")
    Yx, Yp = np.meshgrid(D["y"], D["y"]); Wc = np.where(Yx ** 2 + Yp ** 2 < 2.0 ** 2, W, np.nan)   # kontury tylko w rdzeniu (ogony ~0 to szum)
    ax.contour(D["y"], D["y"], Wc, levels=[0], colors=fs.INK, linewidths=0.6)                     # W=0: czytelne w skali szarosci
    ax.contour(D["y"], D["y"], Wc, levels=[-0.05], colors=fs.INK, linewidths=0.6, linestyles="--")
    ax.set_title(lab, pad=2); ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3]); ax.set_xlabel(r"$x$", labelpad=1)
    if i == 0: ax.set_ylabel(r"$p$", labelpad=1)
    fs.panel(ax, f"({'abcd'[i]})")
cb = fig.colorbar(im, ax=fig.axes[:4], fraction=0.015, pad=0.01); cb.set_ticks([-0.2, 0, 0.2]); cb.ax.tick_params(labelsize=6)

ax = fig.add_subplot(gs[1, 0:4]); fs.zero_axes(ax, x=False)
ax.errorbar(D["db"], D["W0"], 2 * D["W0_sd"], color=fs.BLUE, marker="o", ms=3, ls="-", lw=1, capsize=1.5, label=r"$W(0)$")
ax.errorbar(D["db"], D["Wmin"], 2 * D["Wmin_sd"], color=fs.ORANGE, marker="s", ms=3, ls="--", lw=1, capsize=1.5, label=r"$\min W$")
ax.set_xlabel("added loss (dB)"); ax.set_ylabel(r"$W_{\mathrm{ML}}$"); ax.legend(loc="upper left"); fs.panel(ax, "(e)")
ax = fig.add_subplot(gs[1, 4:8]); sty = {"100": ("o", ":", fs.SKY), "1000": ("s", "--", fs.BLUE), "10000": ("^", "-", fs.GREEN)}
for N in NN:
    mk, ls, c = sty[N]; ax.plot(D["db"], D[f"fock_{N}"], marker=mk, ls=ls, color=c, ms=3, lw=1, label=rf"$N={int(N)}$")
    ax.plot(D["db"], D[f"fock_sym_{N}"], marker=mk, ls="none", mfc="none", color=c, ms=5, mew=0.6)
ax.plot(D["db"], D["fock_full"], marker="D", ls="-.", color=fs.INK, ms=3, lw=1, label=r"full record")
ax.set_ylim(-0.03, 1.05); ax.set_xlabel("added loss (dB)"); ax.set_ylabel("Fock fraction", labelpad=1); ax.legend(loc="lower left", ncol=1); fs.panel(ax, "(f)")
ax = fig.add_subplot(gs[1, 8:12])
for N in NN:
    mk, ls, c = sty[N]; ax.plot(D["db"], D[f"neg_{N}"], marker=mk, ls=ls, color=c, ms=3, lw=1, label=rf"$N={int(N)}$")
ax.plot(D["db"], (D["W0"] < 0).astype(float), marker="D", ls="-.", color=fs.INK, ms=3, lw=1, label=r"$N=3\times10^4$")
ax.set_ylim(-0.03, 1.05); ax.set_xlabel("added loss (dB)"); ax.set_ylabel(r"fraction $W_{\mathrm{ML}}(0)<0$", labelpad=1); ax.legend(loc="lower left"); fs.panel(ax, "(g)")
fig.savefig(OUT.parent / "fig_kawasaki.pdf", bbox_inches="tight"); fig.savefig(OUT.parent / "fig_kawasaki.png", bbox_inches="tight", dpi=250)
from PIL import Image
Image.open(OUT.parent / "fig_kawasaki.png").convert("L").save(OUT.parent / "fig_kawasaki_gray.png"); print("FIG_DONE")
