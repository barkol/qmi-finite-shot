"""Fig. sweeps: error rate 1 - accuracy of the exact-image classifiers versus noise strength (three channels), W vs Q.
Line = mean over training runs, band = range over runs. Physical x axes: loss 1-eta, dephasing gamma = 3p/100,
Gaussian noise sigma^2 = p/100 (vacuum variance 1/2). Triangles mark the cells of fig:rule (50 %, 80 % loss, 20 % dephasing).

  python figures/fig_sweeps_col.py        results/eval_sweep_clf_{W,Q}_exact_s*.json -> data/fig_sweeps.npz -> plot
  python figures/fig_sweeps_col.py --replot  plot from data/fig_sweeps.npz only
RES=<dir> overrides the results directory."""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import exact_runs, R                    # results reader shared with the other figures (unchanged)
from figlook import REP, CHANCE_ERR, need, load_or_build, style, err_axis, to_err, save

CH = ("loss", "dephase", "gauss")
TITLE = {"loss": "(a) Photon loss", "dephase": "(b) Dephasing", "gauss": "(c) Gaussian noise"}
XMAP = {"loss": lambda p: p, "dephase": lambda p: 3.0 * p / 100.0, "gauss": lambda p: p / 100.0}
XLAB = {"loss": r"loss $1-\eta$ (\%)", "dephase": r"dephasing rate $\gamma$",
        "gauss": r"noise variance $\sigma^2$ (vacuum: $1/2$)"}
XTICKS = {"loss": [0, 25, 50, 75, 100], "dephase": [0, 1, 2, 3], "gauss": [0, 0.25, 0.5, 0.75, 1]}
RULE = {"loss": [50, 80], "dephase": [20], "gauss": []}          # strengths p of the fig:rule cells


def build():
    d = {}
    for rep in "WQ":
        runs = exact_runs(rep)
        need(runs, f"brak plikow {R}/eval_sweep_clf_{rep}_exact_s*.json")
        d[f"{rep}_nruns"] = np.array(len(runs))
        for c in CH:
            need(all(c in r for r in runs), f"kanal '{c}' brakuje w eval_sweep_clf_{rep}_exact_s*.json")
            lv = sorted(int(k) for k in runs[0][c])
            for i, r in enumerate(runs):
                need(sorted(int(k) for k in r[c]) == lv, f"{rep} {c}: rozne poziomy w przebiegu {i}")
                for l in lv:
                    need("acc" in r[c][str(l)] and "n" in r[c][str(l)], f"{rep} {c} {l}: brak 'acc' lub 'n'")
            A = np.array([[r[c][str(l)]["acc"] for l in lv] for r in runs])
            n = np.array([[r[c][str(l)]["n"] for l in lv] for r in runs])
            d.update({f"{rep}_{c}_lvl": np.array(lv), f"{rep}_{c}_m": A.mean(0), f"{rep}_{c}_lo": A.min(0),
                      f"{rep}_{c}_hi": A.max(0), f"{rep}_{c}_nmin": n.min()})
    return d


def plot(d):
    for rep in "WQ":
        for c in CH:
            for s in ("lvl", "m", "lo", "hi", "nmin"):
                need(f"{rep}_{c}_{s}" in d, f"klucz {rep}_{c}_{s} w cache; uruchom bez --replot")
    fs, plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    # zero errors cannot sit on a log axis: below half the resolution of the run-averaged error -> "0" floor
    res = min(1.0 / (float(d[f"{r}_nruns"]) * float(d[f"{r}_{c}_nmin"])) for r in "WQ" for c in CH)
    floor = 0.6 * res
    fig, ax0 = plt.subplots(1, 1, figsize=(3.37, 2.1)); axs = [ax0]
    fig.subplots_adjust(left=0.175, right=0.945, bottom=0.19, top=0.80)   # 2mm: right margin for the "100" tick
    for a, c in zip(axs, CH):
        err_axis(a, floor)
        a.axhline(CHANCE_ERR, color=fs.GREY, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        f = XMAP[c]
        xmax = f(max(d[f"W_{c}_lvl"].max(), d[f"Q_{c}_lvl"].max()))
        for p in RULE[c]:
            a.plot([f(p)], [1.0], marker="v", ms=3.6, color=fs.INK, clip_on=False, zorder=6)
            a.axvline(f(p), color=fs.FAINT, lw=0.6, zorder=0)
        if c == "gauss":                                   # W * G_{1/2} = Q of the noiseless state: no negative values
            a.axvline(0.5, color=fs.INK, lw=0.6, zorder=1)
            a.text(0.515, 0.55, r"$W\geq0$ $\rightarrow$", fontsize=9, ha="left", va="bottom",   # y 0.012 -> 0.55, napis lezal na linii danych
                   transform=a.get_xaxis_transform(), color=fs.INK)
        for rep in "WQ":
            s = REP[rep]; x = f(d[f"{rep}_{c}_lvl"].astype(float))
            a.fill_between(x, to_err(d[f"{rep}_{c}_hi"], floor), to_err(d[f"{rep}_{c}_lo"], floor),
                           color=s["c"], alpha=0.17, lw=0, zorder=2)
            a.plot(x, to_err(d[f"{rep}_{c}_m"], floor), color=s["c"], ls=s["ls"], lw=1.2, zorder=4,
                   marker=s["m"], ms=3.0, mfc=s["mfc"], mec=s["c"], mew=0.8, markevery=2)
        a.set_xlim(f(0), xmax); a.set_xticks(XTICKS[c]); a.set_xticklabels([f"{v:g}" for v in XTICKS[c]])
        a.set_xlabel(XLAB[c], labelpad=2)
        a.text(0.03, 0.42, r"dephasing ($\gamma\le3$) and Gaussian" "\n" r"noise ($\sigma^2\le1$): no errors", transform=a.transAxes, fontsize=9, va="center", ha="left", color=fs.INK)
    axs[0].set_ylabel(r"error rate $1-$accuracy", labelpad=2)
    axs[0].text(0.02, CHANCE_ERR * 0.83, "chance", fontsize=9, color=fs.GREY, ha="left", va="top",
                transform=axs[0].get_yaxis_transform())
    hs = [Line2D([], [], color=REP[r]["c"], ls=REP[r]["ls"], lw=1.2, marker=REP[r]["m"], ms=3.0,
                 mfc=REP[r]["mfc"], mec=REP[r]["c"], mew=0.8) for r in "WQ"]
    hs += [Patch(color=fs.GREY, alpha=0.3, lw=0),
           Line2D([], [], color=fs.INK, marker="v", ms=3.6, ls="none")]
    # 2mm: 9 pt labels, shortened so two columns fit in 3.37 in (caption explains range and triangles)
    labs = [REP["W"]["lab"], REP["Q"]["lab"], "range over runs", "cells on records"]
    fig.legend([hs[0], hs[2], hs[1], hs[3]], [labs[0], labs[2], labs[1], labs[3]], loc="upper center", ncol=2,
               bbox_to_anchor=(0.54, 1.0), frameon=False, handlelength=2.0, handletextpad=0.5,
               columnspacing=0.9, borderaxespad=0.15)
    save(fig, "fig_sweeps_col")


if __name__ == "__main__":
    plot(load_or_build("fig_sweeps", build))
