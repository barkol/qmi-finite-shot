"""Decision flowchart "homodyne or heterodyne?" (discussion, rule paragraph), column width.

Leaves are never hand-written. They follow from the paired differences Q-W in
results/ci.json (base set) and results/ci_ahm.json (Ahmed set), keys
rule_N{N}_{cond}_dQW = {mean, lo, hi} (fractions; plotted in points = 100x):

  heterodyne (Q)     95% CI of Q-W excludes zero with positive sign in BOTH sets
  homodyne (W)       ... negative sign in BOTH sets
  no difference      CI includes zero in BOTH sets
  Q / W (one set)    CI excludes zero in ONE set and includes it in the other
  family-dependent   CI excludes zero in both sets with OPPOSITE signs

Data separate from drawing: step 1 writes figures/data/fig_rule_flow.json (leaves +
supporting numbers), step 2 writes the TikZ fragment figures/fig_rule_flow.tex from
that JSON only.  Adjacent noise rows with the same leaf share one leaf box; when all
noise rows of an N give the same leaf, the "Noise?" question is dropped for that N.

Nodes kept: event budget N, dominant noise (clean / 50% loss / 80% loss / 20% dephasing).
Nodes dropped (no paired CI of Q-W in the results files):
phase-reference stability (drift tested only at N=1e4, no CI, sign differs between
sets), fine interference structure (per-family numbers only for exact images / drift),
Gaussian noise (tested only on exact images; stated in the caption instead).

Usage:  python fig_rule_flow.py                 (missing key -> error, exit 1)
        python fig_rule_flow.py --replot        (draw from figures/data/fig_rule_flow.json only)
        FIG_PLACEHOLDER=1 python fig_rule_flow.py   (fill what exists, red PLACEHOLDER mark)
Env RES (results dir), CI_AHM (Ahmed CI file), FIG_OUT_JSON / FIG_OUT_TEX (outputs) are for tests only.
"""
import json, os, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RES", os.path.join(HERE, "..", "results"))
CI_BASE = os.path.join(RES, "ci.json")
CI_AHM = os.environ.get("CI_AHM", os.path.join(RES, "ci_ahm.json"))
OUT_JSON = os.environ.get("FIG_OUT_JSON", os.path.join(HERE, "data", "fig_rule_flow.json"))
OUT_TEX = os.environ.get("FIG_OUT_TEX", os.path.join(HERE, "fig_rule_flow.tex"))
PLACEHOLDER = os.environ.get("FIG_PLACEHOLDER", "") not in ("", "0")

NS = [100, 1000, 10000]
NLAB = {100: r"$100$", 1000: r"$1000$", 10000: r"$10\,000$"}   # written out: no superscripts (2 mm rule)
CONDS = ["clean", "loss50", "loss80", "deph20"]
CLAB = {"clean": "noiseless", "loss50": r"50\,\% loss", "loss80": r"80\,\% loss", "deph20": r"20\,\% deph."}
LEAF_TXT = {"Q": r"heterodyne", "W": r"homodyne", "none": r"no\\significant\\difference",
            "Q1": r"$Q$ in\\one set", "W1": r"$W$ in\\one set",
            "dep": r"family-\\dependent", "missing": "?"}


# ----------------------------------------------------------------------------- step 1: data
def load(path):
    if not os.path.exists(path):
        if PLACEHOLDER:
            print(f"[placeholder] missing file {path}", file=sys.stderr); return {}
        sys.exit(f"ERROR: missing results file {path}")
    return json.load(open(path))


def sign(ci):
    """+1 / -1 if the 95% CI excludes zero, 0 if it includes zero, None if absent."""
    if ci is None: return None
    if ci["lo"] > 0: return 1
    if ci["hi"] < 0: return -1
    return 0


def leaf(sb, sa):
    if sb is None or sa is None: return "missing"
    if sb == 1 and sa == 1: return "Q"
    if sb == -1 and sa == -1: return "W"
    if sb == 0 and sa == 0: return "none"
    if 0 in (sb, sa): return "Q1" if sb + sa > 0 else "W1"   # one set significant, the other not
    return "dep"                                              # significant with opposite signs


def build():
    base, ahm = load(CI_BASE), load(CI_AHM)
    missing, groups = [], []
    for N in NS:
        rows = []
        for c in CONDS:
            key = f"rule_N{N}_{c}_dQW"
            cell = {"N": N, "cond": c, "key": key}
            for tag, src in (("base", base), ("ahm", ahm)):
                v = src.get(key)
                if v is None or not all(k in v for k in ("mean", "lo", "hi")):
                    missing.append(f"{tag}:{key}"); cell[tag] = None
                else:
                    cell[tag] = {k: 100.0 * v[k] for k in ("mean", "lo", "hi")}   # points
                    cell[tag]["sig"] = sign(v)
            cell["leaf"] = leaf(*(None if cell[t] is None else cell[t]["sig"] for t in ("base", "ahm")))
            rows.append(cell)
        groups.append({"N": N, "rows": rows, "collapsed": len({r["leaf"] for r in rows}) == 1})
    if missing and not PLACEHOLDER:
        sys.exit("ERROR: missing keys (set FIG_PLACEHOLDER=1 to draw anyway):\n  " + "\n  ".join(missing))
    out = {"rule": "leaf Q/W if the 95% CI of Q-W excludes zero with the same sign in both sets; "
                   "no difference if it includes zero in both; Q/W in one set if it excludes zero in one set only; "
                   "family-dependent if it excludes zero in both with opposite signs",
           "units": "points (100 x accuracy difference)",
           "sources": {"base": os.path.relpath(CI_BASE, HERE), "ahm": os.path.relpath(CI_AHM, HERE)},
           "source_mtime": {t: (os.path.getmtime(p) if os.path.exists(p) else None)
                            for t, p in (("base", CI_BASE), ("ahm", CI_AHM))},
           "placeholder": PLACEHOLDER, "missing": missing,
           "nodes_kept": ["event budget N", "dominant noise"],
           "nodes_dropped": {"phase reference (drift)": "no paired CI of Q-W; only N=1e4; sign differs between sets",
                             "fine interference structure (cat, GKP)": "per-family data only for exact images and drift",
                             "Gaussian noise": "tested only on exact images"},
           "groups": groups}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    return out


# ----------------------------------------------------------------------------- step 2: drawing
def fmt(cell):
    if cell is None: return r"?"
    s = f"{cell['mean']:+.1f}"
    if s in ("+0.0", "-0.0"): s = "0.0"
    s = s.replace("+", "{+}") if s.startswith("+") else s
    return (r"$\mathbf{" + s + "}$") if cell["sig"] != 0 else ("$" + s + "$")


def draw(d):
    # smallest capitals and numerals >= 2 mm in print -> all text at \small (9 pt, digits 2.05 mm), no
    # sub/superscripts (N values written out, 10\,000 instead of $10^4$). Horizontal positions in pt (column 246 pt),
    # vertical in mm.
    P, G, TOP = 4.4, 2.0, 0.0          # row pitch, gap between N groups, y of first row [mm]
    X = dict(root0=0.0, root1=11.0, nq0=40.0, nq1=68.0, fan=71.0, lab=73.5, db=147.5, da=177.5,
             arr0=180.0, leaf0=188.0, leaf1=242.5)                     # [pt]
    L = []
    a = L.append
    a("% Generated by figures/fig_rule_flow.py from " + d["sources"]["base"] + " and " + d["sources"]["ahm"]
      + ". Do not edit by hand.")
    a("% Needs \\usepackage{tikz} with \\usetikzlibrary{positioning,arrows.meta,shapes}.")
    a(r"\definecolor{flowQ}{HTML}{D55E00}\definecolor{flowW}{HTML}{0072B2}\definecolor{flowK}{HTML}{4D4D4D}")
    a(r"\begin{tikzpicture}[x=1pt,y=1mm,font=\small,")
    a(r" ques/.style={anchor=center,draw=flowK,rounded corners=1.6pt,fill=flowK!8,line width=0.5pt,align=center,inner sep=1pt},")
    a(rf" leaf/.style={{anchor=center,rectangle,minimum width={X['leaf1'] - X['leaf0']:.1f}pt,inner sep=0.8pt,align=center}},")
    a(r" lQ/.style={leaf,draw=flowQ,fill=flowQ!28,line width=1.0pt},")
    a(r" lW/.style={leaf,draw=flowW,fill=flowW!16,line width=0.5pt,double=white,double distance=0.7pt},")
    a(r" lnone/.style={leaf,draw=flowK,fill=white,line width=0.5pt,dash pattern=on 1.6pt off 1pt},")
    a(r" lQ1/.style={leaf,draw=flowQ,fill=flowQ!9,line width=0.5pt},")
    a(r" lW1/.style={leaf,draw=flowW,fill=flowW!6,line width=0.4pt,double=white,double distance=0.5pt},")
    a(r" ldep/.style={leaf,draw=flowK,fill=flowK!14,line width=0.5pt,dash pattern=on 0.5pt off 0.8pt},")
    a(r" lmissing/.style={leaf,draw=red,fill=white,line width=0.5pt},")
    a(r" ed/.style={draw=flowK,line width=0.4pt},")
    a(r" ar/.style={ed,-{Latex[length=1.3mm,width=1mm]}}]")
    # row coordinates
    y, ys, gy = TOP, [], []
    for g in d["groups"]:
        yy = []
        for _ in g["rows"]:
            yy.append(y); y -= P
        ys.append(yy); gy.append(sum(yy) / len(yy)); y -= G
    ybot = ys[-1][-1]
    # header
    yh = TOP + 3.6
    a(rf"\node[anchor=base east,inner sep=0] at ({X['db']},{yh}) {{base}};")
    a(rf"\node[anchor=base east,inner sep=0] at ({X['da']},{yh}) {{\cite{{Ahmed2021}}}};")
    a(rf"\node[anchor=south,inner sep=0] at ({(X['db'] + X['da']) / 2 - 12},{yh + 4.4}) {{$Q-W$ (points)}};")
    a(rf"\node[anchor=base,inner sep=0] at ({(X['leaf0'] + X['leaf1']) / 2},{yh}) {{choose}};")
    a(rf"\node[anchor=base west,inner sep=0] at ({X['lab']},{yh}) {{noise}};")
    a(rf"\draw[flowK,line width=0.3pt] ({X['lab']},{yh - 1.0}) -- ({X['leaf1']},{yh - 1.0});")
    # root question (rotated)
    ymid = (TOP + ybot) / 2
    a(rf"\node[ques,rotate=90,minimum width={TOP - ybot + 3.4:.2f}mm,minimum height={X['root1'] - X['root0']}pt] "
      rf"(root) at ({(X['root0'] + X['root1']) / 2},{ymid:.2f}) {{Event budget $N$?}};")
    for gi, g in enumerate(d["groups"]):
        rows, yy = g["rows"], ys[gi]
        if g["collapsed"]:
            # noise question dropped: N edge runs straight into the fan
            a(rf"\draw[ed] ({X['root1']},{gy[gi]:.2f}) -- ({X['fan']},{gy[gi]:.2f});")
            a(rf"\node[anchor=north,inner sep=0.8pt,text=flowK,font=\small\itshape] at ({(X['root1'] + X['fan']) / 2 + 2},{gy[gi]:.2f}) {{any noise}};")
        else:
            a(rf"\draw[ed] ({X['root1']},{gy[gi]:.2f}) -- ({X['nq0']},{gy[gi]:.2f});")
            a(rf"\node[ques,minimum width={X['nq1'] - X['nq0']}pt,minimum height=6.4mm] at "
              rf"({(X['nq0'] + X['nq1']) / 2},{gy[gi]:.2f}) {{Noise?}};")
            a(rf"\draw[ed] ({X['nq1']},{gy[gi]:.2f}) -- ({X['fan']},{gy[gi]:.2f});")
        a(rf"\node[anchor=south,inner sep=0.8pt] at ({(X['root1'] + X['nq0']) / 2 + 1.5},{gy[gi]:.2f}) {{{NLAB[g['N']]}}};")
        a(rf"\draw[ed] ({X['fan']},{yy[0]:.2f}) -- ({X['fan']},{yy[-1]:.2f});")
        for r, yr in zip(rows, yy):
            a(rf"\draw[ed] ({X['fan']},{yr:.2f}) -- ({X['lab']},{yr:.2f});")
            a(rf"\node[anchor=west,inner sep=0.5pt] at ({X['lab']},{yr:.2f}) {{{CLAB[r['cond']]}}};")
            a(rf"\node[anchor=east,inner sep=0.5pt] at ({X['db']},{yr:.2f}) {{{fmt(r['base'])}}};")
            a(rf"\node[anchor=east,inner sep=0.5pt] at ({X['da']},{yr:.2f}) {{{fmt(r['ahm'])}}};")
            a(rf"\draw[ar] ({X['arr0']},{yr:.2f}) -- ({X['leaf0']},{yr:.2f});")
        # leaf boxes: runs of adjacent rows with the same leaf
        i = 0
        while i < len(rows):
            j = i
            while j + 1 < len(rows) and rows[j + 1]["leaf"] == rows[i]["leaf"]: j += 1
            lf = rows[i]["leaf"]
            yc, h = (yy[i] + yy[j]) / 2, (j - i) * P + P - 0.8
            a(rf"\node[l{lf},minimum height={h:.2f}mm] at ({(X['leaf0'] + X['leaf1']) / 2},{yc:.2f}) {{{LEAF_TXT[lf]}}};")
            i = j + 1
    if d["placeholder"]:
        a(rf"\node[anchor=south west,draw=red,text=red,font=\small\bfseries,inner sep=1pt] at "
          rf"({X['root0']},{yh + 4.4}) {{PLACEHOLDER}};")
    a(r"\end{tikzpicture}")
    open(OUT_TEX, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if "--replot" in sys.argv:                       # draw from the cached JSON only (no results read)
        if not os.path.exists(OUT_JSON): sys.exit(f"ERROR: --replot needs {OUT_JSON}")
        data = json.load(open(OUT_JSON))
    else:
        data = build()
    draw(json.load(open(OUT_JSON)))
    for g in data["groups"]:
        print(f"N={g['N']:>5}" + (" (collapsed)" if g["collapsed"] else ""))
        for r in g["rows"]:
            b, h = r["base"], r["ahm"]
            fb = "--" if b is None else f"{b['mean']:+5.1f} [{b['lo']:+5.1f},{b['hi']:+5.1f}]"
            fa = "--" if h is None else f"{h['mean']:+5.1f} [{h['lo']:+5.1f},{h['hi']:+5.1f}]"
            print(f"   {r['cond']:7s} base {fb:24s} ahm {fa:24s} -> {r['leaf']}")
    print("wrote", OUT_JSON, "and", OUT_TEX, "(PLACEHOLDER)" if PLACEHOLDER else "")
