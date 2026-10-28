"""Fig. shots: error rate 1 - accuracy versus number of recorded events N (log-log), W vs Q.
Thick lines (straight segments in log-log): networks trained on finite records (mean over runs, band = range). The point at N = infinity is the error
of the same networks on exact images (long-record limit), with a faint dotted guide across the panel.
Vertical tick on the curve: N95, the N at which the mean accuracy reaches 95 % of the exact-image accuracy (linear interpolation in
log10 N, identical to scripts/make_numbers.py; omitted when below 10^2 or above 10^6).
Thin faint lines: networks trained on exact images and tested on records (training-measurement mismatch).

  python figures/fig_shots.py            results/eval_shots.json + eval_shots_s[0-9].json -> data/fig_shots.npz -> plot
  python figures/fig_shots.py --replot   plot from data/fig_shots.npz only
RES=<dir> overrides the results directory."""
import os, sys, re
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load_shots, R                    # results reader shared with the other figures (unchanged)
from figlook import REP, CHANCE_ERR, need, load_or_build, style, err_axis, to_err, save

PANELS = (("loss", 0), ("loss", 50), ("loss", 80), ("dephase", 20))   # 60 -> 50 %, jak w fig_rule i tekscie
NS = (100, 1000, 10000, 100000, 1000000)
N95TAG = {("loss", 0): "loss0", ("loss", 50): "loss50", ("loss", 80): "loss80", ("dephase", 20): "deph20"}
XINF = 10 ** 7.0                                    # where N = infinity is drawn


def cross(y, level):
    """make_numbers.cross: first N (linear in log10 N) where y(N) >= level; 'lt' below NS[0], None above NS[-1]."""
    x = np.log10(NS)
    if y[0] >= level: return "lt"
    for i in range(1, len(NS)):
        if y[i] >= level:
            return x[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
    return None


def build():
    need(os.path.exists(os.path.join(R, "eval_shots.json")), f"brak {R}/eval_shots.json")
    sh = load_shots()
    d = {"N": np.array(NS)}
    for rep in "WQ":
        for kind in ("shots", "exact"):
            names = sorted(n for n in sh if f"clf_{rep}_{kind}_s" in n)
            need(names, f"brak sieci clf_{rep}_{kind}_s* w {R}/eval_shots*.json")
            d[f"{rep}_{kind}_nruns"] = np.array(len(names))
            for ch, L in PANELS:
                def cell(n, N):
                    try:
                        c = sh[n][ch][str(N)][str(L)]
                    except KeyError:
                        need(False, f"{n}: brak komorki {ch}/N={N}/{L}")
                    need("acc" in c and "n" in c, f"{n} {ch}/N={N}/{L}: brak 'acc' lub 'n'")
                    return c
                A = np.array([[cell(n, N)["acc"] for N in NS] for n in names])
                nmin = min(cell(n, N)["n"] for n in names for N in (0,) + NS)
                inf = np.array([cell(n, 0)["acc"] for n in names])
                k = f"{rep}_{kind}_{ch}{L}"
                d.update({k + "_m": A.mean(0), k + "_lo": A.min(0), k + "_hi": A.max(0), k + "_inf": inf.mean(),
                          k + "_inflo": inf.min(), k + "_infhi": inf.max(), k + "_nmin": np.array(nmin)})
                if kind == "shots":
                    c = cross(A.mean(0), 0.95 * inf.mean())
                    d[k + "_n95"] = np.array(np.nan if c in (None, "lt") else c)
                    d[k + "_n95kind"] = np.array({None: "gt", "lt": "lt"}.get(c, "eq") if not isinstance(c, float) else "eq")
    check_n95(d)
    return d


def check_n95(d):
    """compare with the N95 macros of numbers.tex in RES (warn only: numbers.tex may be older than the JSON files)."""
    cand = [os.path.join(R, "numbers.tex"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "numbers.tex")]
    p = next((c for c in cand if os.path.exists(c)), None)
    if p is None:
        print("  (brak numbers.tex: N95 nie porownane z makrami)"); return
    print(f"  N95 porownane z {os.path.abspath(p)}")
    mac = {}
    for line in open(p, encoding="utf-8"):
        m = re.match(r"^\\expandafter\\def\\csname res@(N95_\S+?)\\endcsname\{(.*)\}\s*$", line)
        if m: mac[m.group(1)] = m.group(2)
    for rep in "WQ":
        for ch, L in PANELS:
            tag = N95TAG.get((ch, L))
            if not tag or f"N95_{tag}_{rep}" not in mac: continue
            k = f"{rep}_shots_{ch}{L}"; kind = str(d[k + "_n95kind"])
            if kind == "eq":
                c = float(d[k + "_n95"]); e = int(np.floor(c))
                mine = f"\\ensuremath{{{10 ** (c - e):.1f}\\times10^{{{e}}}}}"
            else:
                mine = "\\ensuremath{<10^{2}}" if kind == "lt" else "\\ensuremath{>10^{6}}"
            ok = mine == mac[f"N95_{tag}_{rep}"]
            print(f"  N95 {rep} {tag}: {mine} {'== makro' if ok else '!= makro ' + mac[f'N95_{tag}_{rep}'] + '  <-- UWAGA'}")


def plot(d):
    for rep in "WQ":
        for kind in ("shots", "exact"):
            for ch, L in PANELS:
                for s in ("m", "lo", "hi", "inf", "inflo", "infhi", "nmin") + (("n95", "n95kind") if kind == "shots" else ()):
                    need(f"{rep}_{kind}_{ch}{L}_{s}" in d, f"klucz {rep}_{kind}_{ch}{L}_{s} w cache; uruchom bez --replot")
    fs, plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator
    N = d["N"].astype(float)
    res = min(1.0 / (float(d[f"{r}_{k}_nruns"]) * float(d[f"{r}_{k}_{c}{L}_nmin"]))
              for r in "WQ" for k in ("shots", "exact") for c, L in PANELS)
    floor = 0.6 * res
    fig, axs = plt.subplots(1, len(PANELS), figsize=(7.0, 2.2), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.83, wspace=0.10)
    ttl = {"loss": r"%d\,\%% loss", "dephase": r"%d\,\%% dephasing"}
    for i, (a, (ch, L)) in enumerate(zip(axs, PANELS)):
        err_axis(a, floor)
        a.set_xscale("log"); a.set_xlim(10 ** 1.8, 10 ** 7.3)
        a.xaxis.set_major_locator(FixedLocator(list(N) + [XINF]))
        a.set_xticklabels([r"$10^{%d}$" % round(np.log10(v)) if round(np.log10(v)) % 2 == 0 else "" for v in N] + [r"$\infty$"])
        a.xaxis.set_minor_locator(NullLocator())
        xb = 10 ** 6.62                                 # break between 10^6 and infinity (2mm: clear of the 9 pt 10^6 label)
        kw = dict(transform=a.get_xaxis_transform(), clip_on=False, color=fs.INK, lw=0.6)
        for dx in (0.92, 1.09):
            a.plot([xb * dx / 1.12, xb * dx * 1.12], [-0.025, 0.025], **kw)
        a.plot([xb * 0.92, xb * 1.09], [0, 0], transform=a.get_xaxis_transform(), clip_on=False, color="white", lw=1.4)
        a.axhline(CHANCE_ERR, color=fs.GREY, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        for rep in "WQ":
            s = REP[rep]
            ke = f"{rep}_exact_{ch}{L}"                  # mismatch: trained on exact images, tested on records
            a.plot(N, to_err(d[ke + "_m"], floor), color=s["c"], ls=s["ls"], lw=0.75, alpha=0.45, zorder=2)
            k = f"{rep}_shots_{ch}{L}"
            a.fill_between(N, to_err(d[k + "_hi"], floor), to_err(d[k + "_lo"], floor),
                           color=s["c"], alpha=0.17, lw=0, zorder=2)
            a.plot(N, to_err(d[k + "_m"], floor), color=s["c"], ls=s["ls"], lw=1.25, zorder=4)
            a.plot(N, to_err(d[k + "_m"], floor), ls="none", marker=s["m"], ms=3.2, mfc=s["mfc"], mec=s["c"], mew=0.8, zorder=5)
            einf = float(to_err(d[k + "_inf"], floor))
            a.plot([N[-1], XINF], [einf, einf], color=s["c"], lw=0.6, ls=(0, (1, 1.3)), alpha=0.8, zorder=3)
            elo, ehi = to_err(d[k + "_infhi"], floor), to_err(d[k + "_inflo"], floor)
            a.plot([XINF, XINF], [elo, ehi], color=s["c"], lw=0.8, zorder=4)
            a.plot([XINF], [einf], marker=s["m"], ms=3.4, mfc=s["mfc"], mec=s["c"], mew=0.8, color=s["c"], zorder=5,
                   clip_on=False)
            if str(d[k + "_n95kind"]) == "eq":
                # x = N95 (make_numbers rule); y = on the drawn curve (straight segments in log-log)
                x95 = 10 ** float(d[k + "_n95"])
                y95 = 10 ** np.interp(np.log10(x95), np.log10(N), np.log10(to_err(d[k + "_m"], floor)))
                a.plot([x95], [y95], marker="|", ms=8, mew=1.1, color=fs.INK, zorder=6)
        a.set_title(("(%s) " % "abcd"[i]) + ("noiseless" if (ch, L) == ("loss", 0) else ttl[ch] % L), loc="left", pad=3)
        a.set_xlabel(r"events $N$", labelpad=1)
    axs[0].set_ylabel(r"error rate $1-$accuracy", labelpad=2)
    axs[0].text(10 ** 3.1, CHANCE_ERR * 0.80, "chance", fontsize=9, color=fs.GREY, ha="center", va="top")   # 2mm: off the thin lines
    hs = [Line2D([], [], color=REP[r]["c"], ls=REP[r]["ls"], lw=1.25, marker=REP[r]["m"], ms=3.2,
                 mfc=REP[r]["mfc"], mec=REP[r]["c"], mew=0.8) for r in "WQ"]
    hs += [Line2D([], [], color=fs.GREY, lw=0.75), Line2D([], [], color=fs.INK, marker="|", ms=7, mew=1.1, ls="none")]
    labs = [REP["W"]["lab"], REP["Q"]["lab"], "trained on exact images", r"$N_{95}$"]
    fig.legend(hs, labs, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0), frameon=False,
               handlelength=2.4, columnspacing=1.6, borderaxespad=0.2)
    save(fig, "fig_shots")


if __name__ == "__main__":
    plot(load_or_build("fig_shots", build))
