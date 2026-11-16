"""Drawing layer (GLE) for fig_tmsv. Reads only the cached npz. Needs GLE 4.3 with LaTeX.
Orders are distinguished by line style (BW-safe); the second order, which contains the two physical directions, is drawn in the heterodyne colour of the TURBO palette."""
import os, subprocess, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_fig_tmsv import load
z = load(); here = os.path.dirname(os.path.abspath(__file__)); G = f"{here}/gle"; os.makedirs(G, exist_ok=True)
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); import palette as _p
C2, CK = _p.role("het", "gle"), "black"; W, H = 8.6, 5.0
np.savetxt(f"{G}/a.dat", np.c_[z["r"], *[z[f"up{n}"] for n in (1, 2, 3, 4)], *[z[f"lo{n}"] for n in (1, 2, 3, 4)]], fmt="%.8e")
np.savetxt(f"{G}/m.dat", np.c_[z["rm"], z["sq"], z["ph"]], fmt="%.8e")
sty = {1: (1, CK), 2: (1, C2), 3: (2, CK), 4: (3, CK)}
L = [f"size {W} {H + 0.75}", "set font texcmr", "set hei .34", "set lwidth .02",
     "amove 0 0.1", "begin graph", f"  size {W} {H}", "  scale auto", "  nobox",
     "  xaxis min 0 max 2 dticks 0.5 hei .34", "  yaxis min 0 max 1 dticks 0.25 hei .34",
     '  xtitle "\\tex{squeezing parameter $r$}" hei .34 dist .1', '  ytitle "\\tex{efficiency $F/F_{\\mathrm{SLD}}$}" hei .34 dist .1',
     '  data "gle/a.dat" d1=c1,c2 d2=c1,c3 d3=c1,c4 d4=c1,c5 d5=c1,c6 d6=c1,c7 d7=c1,c8 d8=c1,c9', '  data "gle/m.dat" d9=c1,c2 d10=c1,c3']
for n in (1, 2, 3, 4):
    ls, col = sty[n]; lw = ".03" if n == 2 else ".018"
    L += [f"  d{n} line lstyle {ls} lwidth {lw} color {col}", f"  d{n + 4} line lstyle {ls} lwidth {lw} color {col}"]
L += [f"  d9 marker fcircle msize .22 color {C2}", f"  d10 marker wcircle msize .22 color {C2}", "end graph"]
ly = H + 0.45; x = 0.9
for n in (1, 2, 3, 4):
    ls, col = sty[n]
    L += [f"set color {col}", f"set lstyle {ls}", f"set lwidth {'.03' if n == 2 else '.018'}", f"amove {x} {ly}", f"rline 0.6 0", "set lstyle 1",
          "set color black", "set just lc", "set hei .33", f"amove {x + 0.7} {ly}", f'write "\\tex{{$n={n}$}}"']
    x += 1.75
L += ["set hei .33", "set just lc", f"set color {C2}", "amove 1.75 4.62", "set hei .2", "marker fcircle", "set hei .33", "amove 2.0 4.62", 'write "squeezing parameter"',
      "amove 1.75 1.12", "set hei .2", "marker wcircle", "set hei .33", "amove 2.0 1.12", 'write "phase shift of one arm"', "set color black"]
open(f"{here}/fig_tmsv.gle", "w").write("\n".join(L) + "\n")
res = subprocess.run(["gle", "-d", "pdf", "fig_tmsv.gle"], cwd=here, capture_output=True, text=True); print(res.stdout[-400:], res.stderr[-400:])
