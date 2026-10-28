"""Rule for choosing the measurement (replaces tab:rule and tab:ahm).
(a) base set, (b) Ahmed set: paired difference Q-W (points, 95% CI) at N = 1e2, 1e3, 1e4 for four noise conditions;
    S-W (homodyne without reconstruction minus FBP Wigner) as a second marker when rule_*_dSW keys exist.
    Right margin: accuracies Q / W in % (nothing from the table is lost).
(c) N95 of W and Q for the four conditions, both sets (log N), factor = larger / smaller N95.
(d) remaining rows of tab:ahm: exact-image accuracies and N=1e4 at 60% loss.

Data separate from drawing:
  collect  : results/ci.json, results/ci_ahm.json (scripts/make_numbers.py) + results/numbers{,_ahm}.tex
             -> figures/data/fig_rule.json; every plotted value is checked against its \\RES macro after rounding
             exactly as make_numbers.py formats it.
  plot     : reads only figures/data/fig_rule.json.
  python figures/fig_rule.py            collect + plot (default; picks up new numbers with no code change)
  python figures/fig_rule.py --replot   plot from the existing cache only
Missing keys stop the script, unless FIG_PLACEHOLDER=1 (then: draw what exists, gray "n/a", diagonal watermark).
RES=<dir> overrides the results directory."""
import os, sys, re, json, math, argparse, datetime

sys.dont_write_bytecode = True                      # never write __pycache__ into OneDrive
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RES", os.path.join(HERE, "..", "results"))
CACHE = os.path.join(HERE, "data", "fig_rule.json")
PLACEHOLDER = os.environ.get("FIG_PLACEHOLDER", "") == "1"

SETS = [("base", "", "ci.json", "numbers.tex", "base set"),
        ("ahm", "ahm_", "ci_ahm.json", "numbers_ahm.tex", r"Ahmed \emph{et al.}")]
NS = (100, 1000, 10000)
CONDS = [("clean", "loss0"), ("loss50", "loss50"), ("loss80", "loss80"), ("deph20", "deph20")]   # (rule tag, N95 tag)
EXTRA = ["acc_clean", "acc_loss50", "acc_loss80", "deph_acc20", "acc_gauss100", "mm_l60_shots"]


# ======================================================================= collect
def read_macros(path):
    out = {}
    rx = re.compile(r"^\\expandafter\\def\\csname res@(\S+?)\\endcsname\{(.*)\}\s*$")
    for line in open(path, encoding="utf-8"):
        m = rx.match(line)
        if m: out[m.group(1)] = m.group(2).replace(r"\ensuremath{-}", "-")   # make_numbers pisze minus jako \ensuremath{-}
    return out


def pct(x): return f"{100 * x:.1f}\\%"                         # make_numbers.pct
def pp(x): return f"{100 * x:.1f}"                              # make_numbers put_diff
def ci_tex(c): return f"\\ensuremath{{[{100 * c['lo']:.1f},\\,{100 * c['hi']:.1f}]}}"


def parse_n95(s):
    """make_numbers fmt(): '\\ensuremath{3.6\\times10^{2}}', '\\ensuremath{<10^{2}}', '\\ensuremath{>10^{6}}'"""
    t = s.replace("\\ensuremath{", "", 1)[:-1] if s.startswith("\\ensuremath{") else s
    m = re.fullmatch(r"([0-9.]+)\\times10\^\{(-?\d+)\}", t)
    if m: return dict(val=float(m.group(1)) * 10 ** int(m.group(2)), kind="eq", tex=t)
    m = re.fullmatch(r"([<>])10\^\{(\d+)\}", t)
    if m: return dict(val=10.0 ** int(m.group(2)), kind="lt" if m.group(1) == "<" else "gt", tex=t)
    raise SystemExit(f"BLAD: nie umiem odczytac N95 z makra: {s!r}")


def collect():
    missing, used, sets = [], [], {}
    for sid, pfx, fci, fnum, _ in SETS:
        pci, pnum = os.path.join(RES, fci), os.path.join(RES, fnum)
        CI = json.load(open(pci)) if os.path.exists(pci) else {}
        MAC = read_macros(pnum) if os.path.exists(pnum) else {}
        if not CI: missing.append(f"{fci} (file)")
        if not MAC: missing.append(f"{fnum} (file)")

        def acc(k):
            """accuracy key -> % value, checked against macro"""
            if k not in CI: missing.append(pfx + k); return None
            c = CI[k]
            if pfx + k not in MAC: missing.append(pfx + k + " (macro)"); return None
            if MAC[pfx + k] != pct(c["mean"]):
                raise SystemExit(f"BLAD: {pfx}{k}: ci={pct(c['mean'])} makro={MAC[pfx + k]} (ci.json i numbers.tex z roznych przebiegow?)")
            used.append(pfx + k); return 100 * c["mean"]

        def diff(k, required=True):
            if k not in CI:
                if required: missing.append(pfx + k)
                return None
            c = CI[k]
            for kk, want in ((k, pp(c["mean"])), (k + "_ci", ci_tex(c))):
                if pfx + kk not in MAC: missing.append(pfx + kk + " (macro)"); return None
                if MAC[pfx + kk] != want:
                    raise SystemExit(f"BLAD: {pfx}{kk}: ci={want} makro={MAC[pfx + kk]}")
            used.append(pfx + k)
            return dict(mean=100 * c["mean"], lo=100 * c["lo"], hi=100 * c["hi"])

        rows = []
        for N in NS:
            for tag, _ in CONDS:
                b = f"rule_N{N}_{tag}"
                rows.append(dict(N=N, cond=tag, Q=acc(b + "_Q"), W=acc(b + "_W"),
                                 dQW=diff(b + "_dQW"), dSW=diff(b + "_dSW", required=False)))
        n95 = {}
        for tag, nt in CONDS:
            n95[tag] = {}
            for rep in "WQ":
                k = f"N95_{nt}_{rep}"
                if pfx + k in MAC: n95[tag][rep] = parse_n95(MAC[pfx + k]); used.append(pfx + k)
                else: n95[tag][rep] = None; missing.append(pfx + k + " (macro)")
        extra = {k: {rep: acc(f"{k}_{rep}") for rep in "WQ"} for k in EXTRA}
        seeds = sorted({CI[f"rule_N{N}_{t}_{r}"]["k"] for N in NS for t, _ in CONDS for r in "QW"
                        if f"rule_N{N}_{t}_{r}" in CI})
        sets[sid] = dict(rows=rows, n95=n95, extra=extra, seeds=seeds,
                         src={f: datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")
                              for f, p in ((fci, pci), (fnum, pnum)) if os.path.exists(p)})
    if missing and not PLACEHOLDER:
        raise SystemExit("BLAD: brak kluczy (FIG_PLACEHOLDER=1 rysuje to, co jest):\n  " + "\n  ".join(missing))
    dsw = any(r["dSW"] for s in sets.values() for r in s["rows"])
    data = dict(placeholder=bool(PLACEHOLDER), missing=missing, n_used=len(used), have_dSW=dsw, sets=sets,
                collected=datetime.datetime.now().isoformat(timespec="seconds"), res=os.path.relpath(RES, os.path.join(HERE, "..", "..")))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f: json.dump(data, f, indent=1)
    print(f"cache -> {CACHE}: {len(used)} kluczy, brak {len(missing)}, dSW={'tak' if dsw else 'nie'}")
    for m in missing: print("  brak:", m)
    return data


# ======================================================================= plot
def plot(D):
    STYLE_DIR = os.environ.get("FIGSTYLE_DIR", os.path.dirname(os.path.abspath(__file__)))   # read-only template
    sys.path.insert(0, STYLE_DIR)
    import figstyle as fs                                                  # noqa: E402
    sys.path.remove(STYLE_DIR)
    import numpy as np                                                     # noqa: E402
    import matplotlib.pyplot as plt                                        # noqa: E402
    from matplotlib.lines import Line2D                                    # noqa: E402
    from matplotlib.transforms import blended_transform_factory as blend  # noqa: E402
    from figlook import big2mm                                             # noqa: E402  (2 mm rule)
    fs.apply(); big2mm(plt)
    # every label >= 9 pt -> digits >= 2.04 mm before the 1.008 LaTeX scale; scripts floored at 9 pt
    FS, FL = 9.0, 9.0
    REP = {"W": dict(c=fs.BLUE, m="o"), "Q": dict(c=fs.ORANGE, m="s"), "S": dict(c=fs.GREEN, m="^")}
    DIFF = {"dQW": dict(c=fs.INK, m="D", ms=3.6), "dSW": dict(c=fs.GREEN, m="^", ms=4.0)}
    CLAB = {"clean": "noiseless", "loss50": r"$50\,\%$ loss", "loss80": r"$80\,\%$ loss", "deph20": r"$20\,\%$ deph."}
    NLAB = {100: r"$N=10^2$", 1000: r"$N=10^3$", 10000: r"$N=10^4$"}
    ELAB = {"acc_clean": "noiseless", "acc_loss50": r"$50\,\%$ loss", "acc_loss80": r"$80\,\%$ loss",
            "deph_acc20": r"$20\,\%$ deph.", "acc_gauss100": r"$\sigma^2{=}1$",
            "mm_l60_shots": r"$60\,\%$ loss"}
    SHORT = {"base": "base", "ahm": "Ahmed"}
    NA = dict(color=fs.GREY, fontsize=FS, ha="center", va="center", style="italic")
    have_dsw = D["have_dSW"]

    W_IN, H_IN = 7.0, 2.86   # 2.45 -> 2.86, fonts 6.5 -> 9 pt
    fig = plt.figure(figsize=(W_IN, H_IN))
    def axin(x0, y0, w, h): return fig.add_axes([x0 / W_IN, y0 / H_IN, w / W_IN, h / H_IN])

    # ---------------- (a), (b): forest plots ----------------
    GAP = 0.6
    ypos = {}
    for g, N in enumerate(NS):
        for c, (tag, _) in enumerate(CONDS):
            ypos[(N, tag)] = -(g * (len(CONDS) + GAP) + c)
    ymin, ymax = min(ypos.values()) - 0.55, 1.35
    vals = [v for s in D["sets"].values() for r in s["rows"] for d in ("dQW", "dSW") if r[d] for v in (r[d]["lo"], r[d]["hi"])]
    lim = max(10.0, 5 * math.ceil((max(abs(v) for v in vals) + 1.0) / 5)) if vals else 10.0
    step = 10 if lim > 12 else 5
    tmax = 2 * step * math.floor(lim / (2 * step))         # 2 mm: only -2s, 0, 2s (5 labels do not fit at 9 pt)
    YB, HF = 0.42, 1.98                                   # bottom and height of the forest axes (in)
    XA, WF, TXT = 0.93, 1.18, 0.75                        # left of (a), plot width, text-column width
    XB = XA + WF + TXT + 0.06
    axes_f = []
    for i, ((sid, _, _, _, sname), x0) in enumerate(zip(SETS, (XA, XB))):
        ax = axin(x0, YB, WF, HF); axes_f.append(ax)
        S = D["sets"][sid]
        ax.set_xlim(-lim, lim); ax.set_ylim(ymin, ymax)
        ax.axvline(0, color=fs.GREY, lw=0.6, zorder=1)
        ax.set_xticks(np.arange(-tmax, tmax + 0.1, tmax)); ax.set_xticks(np.arange(-tmax, tmax + 0.1, step), minor=True)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "$" + fs._strip_zeros(v) + "$"))
        ax.tick_params(axis="x", labelsize=FS, length=2.4, pad=1.5); ax.tick_params(axis="x", which="minor", length=1.4)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlabel(r"$Q-W$, $S-W$ (points)", fontsize=FL, labelpad=1.5)
        for g in range(1, len(NS)):                       # light separators between N groups
            ys = -(g * (len(CONDS) + GAP)) + 0.5 + GAP / 2
            ax.plot([-lim, lim], [ys, ys], color=fs.FAINT, lw=0.5, zorder=0, clip_on=False)
        ax.text(-lim * 0.98, ymax - 0.05, r"$W$ better", fontsize=FS, color=fs.GREY, ha="left", va="top")
        ax.text(lim * 0.98, ymax - 0.05, r"$Q$ better", fontsize=FS, color=fs.GREY, ha="right", va="top")
        ax.text(0.0, 1.0 + 0.04 / HF, r"\textbf{(%s) %s}" % ("ab"[i], sname),   # 2 mm: run counts (5, 3) moved to the caption
                transform=ax.transAxes, fontsize=FL, ha="left", va="bottom")
        tr = blend(ax.transAxes, ax.transData)
        xq, xs, xw = 1.0 + 0.37 / WF, 1.0 + 0.40 / WF, 1.0 + 0.43 / WF
        ax.text(xs, 1.0 + 0.04 / HF, r"$Q$\,/\,$W$ (\%)", transform=ax.transAxes, fontsize=FS, ha="center", va="bottom")
        for r in S["rows"]:
            y = ypos[(r["N"], r["cond"])]
            dd = [("dQW", y + (0.17 if have_dsw else 0.0))] + ([("dSW", y - 0.17)] if have_dsw else [])
            for d, yy in dd:
                c = r[d]
                if c is None:
                    if d == "dQW" or D["placeholder"]: ax.text(0.5 * step * (1 if d == "dQW" else -1), yy, "n/a", **NA)
                    continue
                sig = c["lo"] > 0 or c["hi"] < 0
                st = DIFF[d]
                ax.plot([c["lo"], c["hi"]], [yy, yy], color=st["c"], lw=0.9, solid_capstyle="butt", zorder=2)
                for e in (c["lo"], c["hi"]):
                    ax.plot([e, e], [yy - 0.13, yy + 0.13], color=st["c"], lw=0.6, zorder=2)
                ax.plot(c["mean"], yy, st["m"], ms=st["ms"], mec=st["c"], mew=0.7,
                        mfc=st["c"] if sig else "white", zorder=3)
            for rep, x, ha in (("Q", xq, "right"), ("W", xw, "left")):
                v = r[rep]
                ax.text(x, y, f"{v:.1f}" if v is not None else "n/a", transform=tr, fontsize=FS, ha=ha, va="center",
                        color=REP[rep]["c"] if v is not None else fs.GREY)
            ax.text(xs, y, "/", transform=tr, fontsize=FS, ha="center", va="center", color=fs.GREY)
        if i == 0:                                        # shared y labels
            trl = blend(fig.transFigure, ax.transData)
            for (N, tag), y in ypos.items():
                ax.text(-0.04, y, CLAB[tag], transform=blend(ax.transAxes, ax.transData), fontsize=FS, ha="right", va="center")
            for N in NS:
                ys = [ypos[(N, t)] for t, _ in CONDS]
                xl = 0.08 / W_IN
                ax.text(xl, np.mean(ys), NLAB[N], transform=trl, fontsize=FL, rotation=90, ha="center", va="center")
                ax.plot([xl + 0.09 / W_IN] * 2, [min(ys) - 0.3, max(ys) + 0.3], transform=trl, color=fs.GREY, lw=0.5, clip_on=False)

    # ---------------- (c) N95 dumbbells, (d) remaining tab:ahm rows ----------------
    # 2 mm: x labels moved into the panel headers "(c) N95", "(d) accuracy (%)"; set names only above (c)
    XC, WC, GC = 5.69, 0.52, 0.17                         # left of mini-axes, width, gap between base | Ahmed
    YD, HD = 0.24, 0.98                                   # (d)
    YC, HC = 1.60, 0.64                                   # (c)
    XL = XC - 0.79                                        # left edge of the (c)/(d) label column
    for sid_i, (sid, _, _, _, sname) in enumerate(SETS):
        S = D["sets"][sid]
        x0 = XC + sid_i * (WC + GC)
        # (c)
        ax = axin(x0, YC, WC, HC)
        ax.set_xscale("log"); ax.set_xlim(10 ** 1.6, 10 ** 6.6); ax.set_ylim(-len(CONDS) + 0.55, 0.5)
        ax.set_xticks([1e2, 1e6]); ax.set_xticks([1e3, 1e4, 1e5], minor=True)   # 2 mm: two labels per mini-axis
        ax.tick_params(axis="x", labelsize=FS, length=2.4, pad=1.5); ax.tick_params(axis="x", which="minor", length=1.4); ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.text(0.5, 1.0 + 0.04 / HC, SHORT[sid], transform=ax.transAxes, fontsize=FS, ha="center", va="bottom")
        for c, (tag, _) in enumerate(CONDS):
            y = -c
            ax.axhline(y, color=fs.FAINT, lw=0.4, zorder=0)
            p = S["n95"][tag]
            if sid_i == 0:
                ax.text(-0.05, y, CLAB[tag], transform=blend(ax.transAxes, ax.transData), fontsize=FS, ha="right", va="center")
            if p["W"] is None or p["Q"] is None:
                ax.text(10 ** 4, y, "n/a", **NA)
            if p["W"] and p["Q"]:
                a, b = p["W"]["val"], p["Q"]["val"]
                ax.plot([a, b], [y, y], color=fs.GREY, lw=1.6, solid_capstyle="butt", zorder=1)
                f = max(a, b) / min(a, b)
                bound = p["W"]["kind"] != "eq" or p["Q"]["kind"] != "eq"
                ft = (r"$\times%.0f$" if f >= 9.95 else r"$\times%.1f$") % f
                if bound: ft = (r"$>$" if f > 1 else "") + ft
                ax.text(max(a, b) * 10 ** 0.5, y, ft, fontsize=FS, ha="left", va="center")   # 2 mm: factor right of the bar
            for rep in "WQ":
                q = p[rep]
                if q is None: continue
                big = rep == "W"
                ax.plot(q["val"], y, REP[rep]["m"], ms=4.4 if big else 3.4, mfc=REP[rep]["c"], mec=fs.INK, mew=0.35,
                        zorder=3 if big else 4)
                if q["kind"] != "eq":
                    dx = 1.8 if q["kind"] == "gt" else 1 / 1.8
                    ax.annotate("", xy=(q["val"] * dx, y), xytext=(q["val"], y),
                                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=REP[rep]["c"], mutation_scale=5), zorder=2)
        if sid_i == 0:
            fig.text(XL / W_IN, (YC + HC + 0.04) / H_IN, r"\textbf{(c)} $N_{95}$", fontsize=FL, ha="left", va="bottom")
        # (d)
        ax = axin(x0, YD, WC, HD)
        ev = [v for s in D["sets"].values() for e in s["extra"].values() for v in e.values() if v is not None]
        lo = 10 * math.floor((min(ev) - 2) / 10) if ev else 50
        ax.set_xlim(lo, 101.5); ax.set_ylim(-len(EXTRA) + 0.45, 0.55)
        ax.set_xticks(np.arange(lo, 100.1, 10 if 100 - lo <= 40 else 20))
        ax.tick_params(axis="x", labelsize=FS, length=2.4, pad=1.5)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        for j, k in enumerate(EXTRA):
            y = -j
            ax.axhline(y, color=fs.FAINT, lw=0.4, zorder=0)
            e = S["extra"][k]
            if j == len(EXTRA) - 1:
                ax.plot([lo, 101.5], [y + 0.5, y + 0.5], color=fs.GREY, lw=0.4, ls=(0, (2, 1.5)), clip_on=False)
            if sid_i == 0:
                ax.text(-0.05, y, ELAB[k], transform=blend(ax.transAxes, ax.transData), fontsize=FS, ha="right", va="center")
            if e["W"] is None or e["Q"] is None:
                ax.text((lo + 100) / 2, y, "n/a", **NA)
            if e["W"] is not None and e["Q"] is not None:
                ax.plot([e["W"], e["Q"]], [y, y], color=fs.GREY, lw=1.6, solid_capstyle="butt", zorder=1)
            for rep in "WQ":
                if e[rep] is None: continue
                big = rep == "W"
                ax.plot(e[rep], y, REP[rep]["m"], ms=4.4 if big else 3.4, mfc=REP[rep]["c"], mec=fs.INK, mew=0.35,
                        zorder=3 if big else 4)
        if sid_i == 0:
            trd = blend(fig.transFigure, ax.transData)
            xg = (XL + 0.05) / W_IN
            ax.text(xg, -2.0, r"exact", transform=trd, fontsize=FS, rotation=90, ha="center", va="center")
            ax.plot([xg + 0.09 / W_IN] * 2, [-4.3, 0.3], transform=trd, color=fs.GREY, lw=0.5, clip_on=False)
            fig.text(XL / W_IN, (YD + HD + 0.04) / H_IN, r"\textbf{(d)} accuracy (\%)", fontsize=FL, ha="left", va="bottom")

    # ---------------- legend ----------------
    h = [Line2D([], [], ls="-", color=fs.INK, lw=0.9, marker="D", ms=3.6, mfc=fs.INK, mec=fs.INK),
         Line2D([], [], ls="-", color=fs.INK, lw=0.9, marker="D", ms=3.6, mfc="white", mec=fs.INK)]
    lab = [r"$Q-W$, CI excludes 0", r"$Q-W$, CI includes 0"]
    if have_dsw:
        h.append(Line2D([], [], ls="-", color=fs.GREEN, lw=0.9, marker="^", ms=4.0, mfc=fs.GREEN, mec=fs.GREEN))
        lab.append(r"$S-W$")
    h += [Line2D([], [], ls="none", marker="o", ms=4.4, mfc=fs.BLUE, mec=fs.INK, mew=0.35),
          Line2D([], [], ls="none", marker="s", ms=3.4, mfc=fs.ORANGE, mec=fs.INK, mew=0.35)]
    lab += [r"$W$ (homodyne)", r"$Q$ (heterodyne)"]
    fig.legend(h, lab, ncol=len(h), fontsize=FS, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.5,
               columnspacing=1.0, handletextpad=0.35, borderaxespad=0.03)

    if D["placeholder"]:
        fig.text(0.5, 0.5, r"PLACEHOLDER (old x8 numbers)", fontsize=34, color="0.80", alpha=0.45, rotation=22,
                 ha="center", va="center", zorder=100)
    from PIL import Image
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "fig_rule." + ext), dpi=600 if ext == "png" else None)
    Image.open(os.path.join(HERE, "fig_rule.png")).convert("L").save(os.path.join(HERE, "fig_rule_gray.png"))
    print("fig_rule.pdf/.png/_gray.png", W_IN, H_IN, "placeholder" if D["placeholder"] else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="plot from figures/data/fig_rule.json, skip collection")
    a = ap.parse_args()
    if a.replot:
        if not os.path.exists(CACHE): raise SystemExit(f"BLAD: brak cache {CACHE}; uruchom bez --replot")
        D = json.load(open(CACHE))
    else:
        D = collect()
    plot(D)
