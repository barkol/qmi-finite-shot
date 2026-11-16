"""Drawing layer (GLE) for fig_disp_chernoff. Reads only the cached npz. Needs GLE 4.3 with LaTeX.
Heterodyne: TURBO orange-red, solid, filled markers; homodyne: TURBO blue, dashed, open markers; nbar by marker shape (BW-safe)."""
import os, subprocess, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_fig_disp_chernoff import load
z = load(); here = os.path.dirname(os.path.abspath(__file__)); G = f"{here}/gle"; os.makedirs(G, exist_ok=True)
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); import palette as _p
HET, HOM = _p.role("het", "gle"), _p.role("hom", "gle"); W, H = 8.6, 4.8
a = z["alpha"]; nbs = [float(v) for v in z["nb"]]; sub = slice(12, None, 36)
cols = [a] + [z[f"{k}_{nb}"] for nb in nbs for k in ("het", "hom")]
np.savetxt(f"{G}/a.dat", np.c_[tuple(cols)], fmt="%.8e"); np.savetxt(f"{G}/am.dat", np.c_[tuple(cols)][sub], fmt="%.8e")
L = [f"size {W} {H + 0.75}", "set font texcmr", "set hei .34", "set lwidth .02",
     "amove 0 0.1", "begin graph", f"  size {W} {H}", "  scale auto", "  nobox",
     "  xaxis min 0 max 12 dticks 2 hei .34", "  yaxis min 0 max 1 dticks 0.25 hei .34",
     '  xtitle "\\tex{displacement $|\\alpha|$}" hei .34 dist .1', '  ytitle "\\tex{$\\xi/\\xi_{\\mathrm{QC}}$}" hei .34 dist .1',
     '  data "gle/a.dat" d1=c1,c2 d2=c1,c3 d3=c1,c4 d4=c1,c5', '  data "gle/am.dat" d5=c1,c2 d6=c1,c3 d7=c1,c4 d8=c1,c5',
     f"  d1 line lstyle 1 lwidth .018 color {HET}", f"  d2 line lstyle 2 lwidth .018 color {HOM}",
     f"  d3 line lstyle 1 lwidth .018 color {HET}", f"  d4 line lstyle 2 lwidth .018 color {HOM}",
     f"  d5 marker fsquare msize .16 color {HET}", f"  d6 marker wsquare msize .16 color {HOM}",
     f"  d7 marker ftriangle msize .16 color {HET}", f"  d8 marker wtriangle msize .16 color {HOM}", "end graph"]
items = [("fsquare", HET, f"\\tex{{het., $\\bar n={nbs[0]:g}$}}"), ("ftriangle", HET, f"\\tex{{${nbs[1]:g}$}}"),
         ("wsquare", HOM, f"\\tex{{hom., $\\bar n={nbs[0]:g}$}}"), ("wtriangle", HOM, f"\\tex{{${nbs[1]:g}$}}")]
for x, (mk, col, txt) in zip([0.9, 3.45, 4.5, 7.15], items):
    L += [f"set color {col}", f"amove {x} {H + 0.45}", f"marker {mk} .5", "set color black", "set just lc", "set hei .33", f"amove {x + 0.25} {H + 0.45}", f'write "{txt}"']
open(f"{here}/fig_disp_chernoff.gle", "w").write("\n".join(L) + "\n")
res = subprocess.run(["gle", "-d", "pdf", "fig_disp_chernoff.gle"], cwd=here, capture_output=True, text=True); print(res.stdout[-400:], res.stderr[-400:])
