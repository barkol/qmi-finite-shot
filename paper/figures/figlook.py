"""Shared look for fig_sweeps_col.py and fig_shots.py (error rate on a log axis, W/Q styles, cache handling).

Data separate from drawing:
  python figures/fig_X.py            collect from results/*.json -> figures/data/fig_X.npz, then plot (default;
                                     new results are picked up with no code change)
  python figures/fig_X.py --replot   plot from the existing cache only
RES=<dir> overrides the results directory (as in common.py). Missing keys stop the script with a clear message."""
import os, sys, glob, datetime
import numpy as np

sys.dont_write_bytecode = True                       # never write __pycache__ next to the template in OneDrive
HERE = os.path.dirname(os.path.abspath(__file__))
STYLE_DIR = os.environ.get("FIGSTYLE_DIR", os.path.dirname(os.path.abspath(__file__)))   # read-only template

# W = homodyne / Wigner, Q = heterodyne / Husimi Q (as in fig_rule.py and fig_design.py)
REP = {"W": dict(c="#0072B2", m="o", ls="-", mfc="#0072B2", lab=r"homodyne ($W$)"),
       "Q": dict(c="#D55E00", m="s", ls=(0, (3.2, 1.6)), mfc="white", lab=r"heterodyne ($Q$)")}
CHANCE_ERR = 6.0 / 7.0


class MissingData(SystemExit):
    pass


def need(cond, msg):
    if not cond:
        raise MissingData("BLAD (brak danych): " + msg)


def replot():
    return "--replot" in sys.argv


def load_or_build(name, build):
    """default: build from results and write the cache; --replot: read the cache only."""
    p = os.path.join(HERE, "data", name + ".npz")
    if replot():
        need(os.path.exists(p), f"{p} nie istnieje; uruchom bez --replot")
        d = dict(np.load(p, allow_pickle=True))
        print(f"cache <- {p} (zebrane {d.get('collected', '?')})")
        return d
    d = build()
    d["collected"] = np.array(datetime.datetime.now().isoformat(timespec="seconds"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, **d)
    print(f"cache -> {p}")
    return d


def style():
    sys.path.insert(0, STYLE_DIR)
    import figstyle as fs                            # noqa: E402  (read-only import)
    sys.path.remove(STYLE_DIR)
    fs.apply()
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
                         "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 7,
                         "ytick.minor.size": 1.4, "xtick.minor.size": 1.4,
                         "ytick.minor.width": 0.4, "xtick.minor.width": 0.4})
    big2mm(plt)                                      # 2 mm rule (overrides the sizes above)
    return fs, plt


# ── smallest capitals and numerals >= 2 mm in print ─────────
# Computer Modern digits are 0.6444 em high (TFM), so 2 mm = 5.67 bp needs >= 8.8 pt text; we use >= 9 pt.
# Sub/superscripts: LaTeX would set them at 0.7x (6.3 pt at 9 pt). TEX_2MM puts a 9 pt floor under the script
# and scriptscript sizes (never above the base size), also for the standard sizes predefined in the kernel.
MIN_PT = 9.0
TEX_2MM = r"""\makeatletter
\def\calculate@math@sizes{\dimen@\f@size\p@\@tempdimb.7\dimen@\ifdim\@tempdimb<9\p@\@tempdimb9\p@\fi
\ifdim\@tempdimb>\dimen@\@tempdimb\dimen@\fi\@tempdima.5\dimen@\ifdim\@tempdima<9\p@\@tempdima9\p@\fi
\ifdim\@tempdima>\dimen@\@tempdima\dimen@\fi
\expandafter\xdef\csname S@\f@size\endcsname{\gdef\noexpand\tf@size{\f@size}\gdef\noexpand\sf@size{\strip@pt\@tempdimb}\gdef\noexpand\ssf@size{\strip@pt\@tempdima}\noexpand\math@fontstrue}\csname S@\f@size\endcsname}
\@for\x:={5,6,7,8,9,10,10.95,11,12,14.4,17.28,20.74,24.88}\do{\expandafter\let\csname S@\x\endcsname\relax}
\makeatother"""
RC_2MM = {"font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
          "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9}


def big2mm(plt, rc=None):
    """Apply the 2 mm rule on top of figstyle (call after fs.apply()); figstyle itself stays untouched."""
    pre = plt.rcParams.get("text.latex.preamble", "")
    if "calculate@math@sizes" not in pre:
        plt.rcParams["text.latex.preamble"] = pre + "\n" + TEX_2MM
    plt.rcParams.update(RC_2MM)
    if rc:
        plt.rcParams.update(rc)


def err_axis(ax, floor, top=1.0):
    """log error axis in %, with the floor tick standing for zero errors (all test states correct)."""
    from matplotlib.ticker import FixedLocator, NullFormatter, LogLocator
    ax.set_yscale("log")
    lo = floor / 1.6
    ax.set_ylim(lo, top)
    dec = [10.0 ** k for k in range(int(np.ceil(np.log10(floor * 1.9))), 1)]
    lab = {1e-4: r"$0.01\%$", 1e-3: r"$0.1\%$", 1e-2: r"$1\%$", 1e-1: r"$10\%$", 1.0: r"$100\%$"}
    ax.yaxis.set_major_locator(FixedLocator([floor] + dec))
    ax.set_yticklabels(["0"] + [lab.get(v, f"{100 * v:g}") for v in dec])
    minor = [m * 10.0 ** k for k in range(-6, 0) for m in range(2, 10) if floor * 1.9 < m * 10.0 ** k < top]
    ax.yaxis.set_minor_locator(FixedLocator(minor)); ax.yaxis.set_minor_formatter(NullFormatter())
    # axis break between the zero tick and the first decade
    yb = np.sqrt(floor * dec[0]) if dec else floor * 1.5
    kw = dict(transform=ax.get_yaxis_transform(), clip_on=False, color="#1a1a1a", lw=0.6)
    for dy in (0.93, 1.07):
        ax.plot([-0.025, 0.025], [yb * dy / 1.06, yb * dy * 1.06], **kw)
    ax.plot([0, 0], [yb * 0.93, yb * 1.07], transform=ax.get_yaxis_transform(), clip_on=False, color="white", lw=1.2)


def to_err(acc, floor):
    """1 - accuracy, with exact zeros mapped onto the floor line."""
    e = 1.0 - np.asarray(acc, float)
    e = np.where(e < 0.5 * floor, floor, np.maximum(e, floor))
    return e


def save(fig, name):
    from PIL import Image
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, name + "." + ext), dpi=600 if ext == "png" else None)
    Image.open(os.path.join(HERE, name + ".png")).convert("L").save(os.path.join(HERE, name + "_gray.png"))
    w, h = fig.get_size_inches()
    print(f"{name}.pdf/.png/_gray.png  {w:.2f} x {h:.2f} in")
