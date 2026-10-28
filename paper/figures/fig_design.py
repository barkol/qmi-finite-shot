"""Methods figure: study design. (a) training-data design (same noise sequence and same <n> cycle in every family),
(b) the 135 networks (groups x representation x independent training runs).
Data separate from drawing: reads only figures/data/fig_design.json (no recomputation)."""
import os, sys, json
from collections import Counter

sys.dont_write_bytecode = True                      # never write __pycache__ into OneDrive
STYLE_DIR = os.environ.get("FIGSTYLE_DIR", os.path.dirname(os.path.abspath(__file__)))   # read-only template
sys.path.insert(0, STYLE_DIR)
import figstyle as fs                               # noqa: E402
import numpy as np                                  # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Rectangle, ConnectionPatch  # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data", "fig_design.json")))

fs.apply()
sys.path.insert(0, HERE)
from figlook import big2mm                          # noqa: E402  smallest digits/capitals >= 2 mm (9 pt floor, also scripts)
big2mm(plt)
plt.rcParams.update({"hatch.linewidth": 0.45})
FS = 9.0                                            # smallest text: CM digit 0.644 em -> 2.04 mm (x1.008 in print)
FT = 9.5                                            # panel titles, legend and lines with super/subscripts

# ---------------- channel encoding: colour + luminance step + geometry (bar height / hatch / marker) -------------
CH = {"loss":    dict(c="#555555", lab=r"loss $1-\eta$"),   # szary i fiolet zamiast niebieskiego i zoltego (kolidowaly z kodem W/Q)
      "dephase": dict(c="#CC79A7", lab=r"dephasing"),
      "gauss":   dict(c="#BBE1F6", lab=r"Gaussian $\sigma^2$"),   # light sky tint (Wong SKY + white): luminance step
      "combo":   dict(c="white",   lab=r"combined"),
      "clean":   dict(c="white",   lab=r"noiseless")}
FAM = {"fock": "Fock", "coherent": "coherent", "sqvac": "sq.\\ vacuum", "thermal": "thermal",
       "cat": "cat", "gkp": "GKP", "binomial": "binomial"}
REP = {"W": dict(c=fs.BLUE, m="o"), "Q": dict(c=fs.ORANGE, m="s"), "S": dict(c=fs.GREEN, m="^")}

# 2.70 -> H_IN; text 7 -> 9 pt, panel (b) widened, long titles moved to the caption
W_IN, H_IN = 7.0, 3.00
fig = plt.figure(figsize=(W_IN, H_IN))
fxi = lambda x: x / W_IN                            # inches -> figure fraction
fyi = lambda y: y / H_IN

# ===================================== panel (a) =====================================
cells = D["train_cells"]; nf = D["files_per_class"]; fams = D["families"]; nfam = len(fams)
assert len(cells) == nf
AX0, AW = 0.80, 2.50                                # bar area, inches
axa = fig.add_axes([fxi(AX0), fyi(1.26), fxi(AW), fyi(1.12)])
BH = 0.80                                           # bar box height within a row
for r, f in enumerate(fams):
    y0 = nfam - 1 - r
    axa.add_patch(Rectangle((0, y0), nf, BH, fc="none", ec=fs.FAINT, lw=0.4, zorder=0))
    for j, (ch, lv) in enumerate(cells):
        if ch == "clean":
            axa.plot(j + 0.5, y0 + 0.10, "o", ms=2.2, mfc="white", mec=fs.INK, mew=0.5, zorder=3)
        elif ch == "combo":
            axa.add_patch(Rectangle((j + 0.06, y0), 0.88, BH, fc="white", ec=fs.INK, lw=0.25,
                                    hatch="//////", zorder=2))
        else:
            axa.add_patch(Rectangle((j + 0.06, y0), 0.88, BH * lv / 100, fc=CH[ch]["c"],
                                    ec=fs.INK if ch == "gauss" else "none", lw=0.2, zorder=2))
axa.set_xlim(0, nf); axa.set_ylim(-0.05, nfam)
axa.set_yticks([nfam - 1 - r + BH / 2 for r in range(nfam)]); axa.set_yticklabels([FAM[f] for f in fams], fontsize=FS)
axa.tick_params(axis="y", length=0, pad=2)
xt = [1, 40, 80, nf]
axa.set_xticks([x - 0.5 for x in xt]); axa.set_xticklabels([str(x) for x in xt], fontsize=FS)
axa.tick_params(axis="x", length=2, pad=1.5)
for s_ in ("left", "bottom"): axa.spines[s_].set_visible(False)
axa.set_xlabel(r"training file (bar height: noise, 0--100\,\%)", fontsize=FS, labelpad=1.0, loc="right")
fig.text(fxi(0.02), fyi(H_IN - 0.03), r"\textbf{(a)}~~\textbf{Training data}", fontsize=FT, va="top", ha="left")
ntr = nfam * nf * D["samples_per_file"]
assert ntr == D["n_train_base"]

# zoom: one file = 25 states, <n> cycles over the grid from the start of each file
nb = D["nbar"]; spf = D["samples_per_file"]
seq = [nb[i % len(nb)] for i in range(spf)]; cnt = Counter(seq)
ZW = 2.75                                           # strip width, inches (0.11 in per state)
axz = fig.add_axes([fxi(AX0 - 0.20), fyi(0.70), fxi(ZW), fyi(0.155)])
gl = {2: "1.0", 3: "0.86", 4: "0.70", 5: "0.54"}    # luminance steps for <n>
for i, v in enumerate(seq):
    axz.add_patch(Rectangle((i + 0.05, 0), 0.9, 1, fc=gl[v], ec=fs.INK, lw=0.35))
    axz.text(i + 0.5, 0.45, str(v), ha="center", va="center", fontsize=FS)
axz.set_xlim(0, spf); axz.set_ylim(0, 1); axz.axis("off")
fig.text(fxi(AX0 - 0.24), fyi(0.70 + 0.0775), r"file 1", ha="right", va="center", fontsize=FS)
fig.text(fxi(AX0 - 0.20), fyi(0.645), r"$\langle n\rangle$ of its %d states (%s states per value)"
         % (spf, "/".join(str(cnt[v]) for v in nb)), ha="left", va="top", fontsize=FS)
axa.add_patch(Rectangle((0, -0.05), 1, nfam + 0.05 - 0.2, fc="none", ec=fs.GREY, lw=0.4, zorder=4))

# legend of channels (two rows)
hd = [Rectangle((0, 0), 1, 1, fc=CH["loss"]["c"], ec="none"),
      Rectangle((0, 0), 1, 1, fc=CH["dephase"]["c"], ec="none"),
      Rectangle((0, 0), 1, 1, fc=CH["gauss"]["c"], ec="none"),
      Rectangle((0, 0), 1, 1, fc="white", ec=fs.INK, lw=0.3, hatch="//////"),
      Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=fs.INK, mew=0.5)]
fig.legend(hd, [CH[k]["lab"] for k in ("loss", "dephase", "gauss", "combo", "clean")], ncol=3, fontsize=FT,
           loc="upper left", bbox_to_anchor=(fxi(0.02), fyi(H_IN - 0.24)), handlelength=1.1, handleheight=0.8,
           columnspacing=0.9, handletextpad=0.4, labelspacing=0.25, borderaxespad=0)
# ===================================== panel (b) =====================================
ROWS = [  # (group in base set, group in Ahmed set, question answered by the group)
    ("exact",    "exact",    r"exact images ($N\to\infty$)"),
    ("shots",    "shots",    r"records: $Q$ vs $W$"),
    ("sino",     "sino",     r"no reconstruction"),
    ("cvexact",  None,       r"5-fold CV, exact"),
    ("cvshots",  None,       r"5-fold CV, records"),
    ("drift",    "drift",    r"phase drift"),
    ("shotsK6",  None,       r"$K=6$"),
    ("shotsK12", "shotsK12", r"$K=12$"),
    ("shotsK18", None,       r"$K=18$"),
    ("shotsK24", None,       r"$K=24$"),
    ("shotsK72", None,       r"$K=72$"),
    ("hoexact",  "hoexact",  r"held out, exact"),
    ("hoshots",  "hoshots",  r"held out, records"),
    (None,       "hoexactS", r"held out$'$, exact"),
    (None,       "hoshotsS", r"held out$'$, records"),
    ("narrow",   "narrow",   r"narrow noise range"),
    ("gkpxK6",   None,       r"squeezed GKP"),
]
BRACK = [(6, 10, r"homodyne", r"phases")]
nets = D["nets"]
C = Counter((n["set"], n["group"], n["rep"]) for n in nets)
cnt_set = Counter(n["set"] for n in nets)
assert sum(C.values()) == len(nets) == 135
used = set()
BX0 = 3.48; WB = W_IN - BX0 - 0.01                   # panel (b), inches; x data unit = 1 inch
nr = len(ROWS)
PITCH = 0.142                                        # row pitch, inches
YLO, YHI = -0.6, nr + 1.25                           # rows (data units)
axb = fig.add_axes([fxi(BX0), fyi(0.04), fxi(WB), fyi(PITCH * (YHI - YLO))])
cols = [("base", "W"), ("base", "Q"), ("base", "S"), ("ahmed", "W"), ("ahmed", "Q"), ("ahmed", "S")]
XQ = 1.48                                            # right edge of the question column
DX = 0.30; GAP = 0.12
XC = [XQ + 0.20 + i * DX + (GAP if i >= 3 else 0) for i in range(6)]
XL = XQ + 0.04                                       # left edge of the matrix
axb.set_xlim(0, WB); axb.set_ylim(YLO, YHI)
for i, (gb, ga, q) in enumerate(ROWS):
    y = nr - 1 - i
    if i % 2 == 0:
        axb.add_patch(Rectangle((XL, y - 0.5), WB - XL, 1.0, fc="0.93", ec="none", zorder=0))
    axb.text(XQ, y, q, ha="right", va="center", fontsize=FS)
    for (st, rp), xc in zip(cols, XC):
        g = gb if st == "base" else ga
        k = C.get((st, g, rp), 0) if g else 0
        if k == 0:
            continue
        used.add((st, g, rp))
        xs = xc + (np.arange(k) - (k - 1) / 2) * 0.052
        axb.scatter(xs, np.full(k, y), s=11.0, marker=REP[rp]["m"], c=REP[rp]["c"], edgecolors=fs.INK,
                    linewidths=0.35, zorder=3)
assert used == set(C), set(C) - used               # every network is drawn exactly once
for a, b, t1, t2 in BRACK:
    ya, yb = nr - 1 - a, nr - 1 - b
    xb0 = XQ - 0.50
    axb.plot([xb0 + 0.03, xb0, xb0, xb0 + 0.03], [ya + 0.35, ya + 0.35, yb - 0.35, yb - 0.35], color=fs.INK, lw=0.5)
    axb.text(xb0 - 0.05, (ya + yb) / 2 + 0.55, t1, ha="right", va="center", fontsize=FS)
    axb.text(xb0 - 0.05, (ya + yb) / 2 - 0.55, t2, ha="right", va="center", fontsize=FS)
for after in (2, 5, 10, 14):                         # separators between blocks of questions
    axb.plot([XL, WB], [nr - 1 - after - 0.5] * 2, color=fs.GREY, lw=0.4)
yh = nr - 0.45                                       # headers
for (st, rp), xc in zip(cols, XC):
    axb.text(xc, yh, "$" + rp + "$", ha="center", va="bottom", fontsize=FS)
for xs_, t, n_ in ((XC[:3], r"base set", cnt_set["base"]), (XC[3:], r"Ahmed \emph{et al.}", cnt_set["ahmed"])):
    axb.text(np.mean(xs_), yh + 0.95, t + r" (%d)" % n_, ha="center", va="bottom", fontsize=FS)
    axb.plot([xs_[0] - 0.12, xs_[-1] + 0.12], [yh + 0.85] * 2, color=fs.INK, lw=0.5)
xm = (XC[2] + XC[3]) / 2
axb.plot([xm, xm], [nr - 1 - 16 - 0.5, nr - 0.5], color=fs.GREY, lw=0.4)
axb.axis("off")
fig.text(fxi(BX0 + 0.02), fyi(H_IN - 0.03), r"\textbf{(b)}~~\textbf{Networks} ($%d+%d=%d$)"
         % (cnt_set["base"], cnt_set["ahmed"], len(nets)), fontsize=FT, va="top", ha="left")
tN = D["test_N"]
e0, e1 = int(round(np.log10(tN[0]))), int(round(np.log10(tN[-1])))
th = lambda v: f"{v:,}".replace(",", r"\,")
fig.text(fxi(0.02), fyi(0.02), r"per set: %d families $\times$ %d files $\times$ %d states $=$ %s" "\n"
         r"training, %s validation and %s test states;" "\n" r"test records of $N=10^{%d}\ldots10^{%d}$ events"
         % (nfam, nf, spf, th(ntr), th(D["n_val_base"]), th(D["n_test_sweep"]), e0, e1),
         ha="left", va="bottom", fontsize=FS, linespacing=1.12)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "fig_design." + ext), dpi=600 if ext == "png" else None)
from PIL import Image                                # noqa: E402
Image.open(os.path.join(HERE, "fig_design.png")).convert("L").save(os.path.join(HERE, "fig_design_gray.png"))
print("fig_design.pdf/.png/_gray.png", W_IN, H_IN)
