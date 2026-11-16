"""Drawing layer (GLE) for fig_observer. Reads only the cached npz. Needs GLE 4.3 with LaTeX.
One row per channel; marker = difference of accuracies (heterodyne minus homodyne), bar = 95% bootstrap interval; black triangle = the same difference for the neural networks Q and W.
Positive values (heterodyne better): TURBO orange-red, filled; negative (homodyne better): TURBO blue, open (BW-safe)."""
import os, subprocess, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_fig_observer import load
z = load(); here = os.path.dirname(os.path.abspath(__file__))
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); import palette as _p
HET, HOM = _p.role("het", "gle"), _p.role("hom", "gle")
LAB = {"clean": "no noise", "deph20": "$20\\%$ dephasing", "loss50": "$50\\%$ loss", "loss80": "$80\\%$ loss", "gauss20": "Gaussian, $\\sigma^2=0.2$", "gauss50": "Gaussian, $\\sigma^2=0.5$"}
W, X0, X1, DY = 8.6, 3.75, 8.35, 0.46; lo, hi = -5.0, 15.5; n = len(z["cells"]); Y0 = 0.95; H = Y0 + n * DY + 0.15
gx = lambda v: X0 + (X1 - X0) * (v - lo) / (hi - lo)
L = [f"size {W} {H + 0.1}", "set font texcmr", "set hei .33", "set lwidth .018", f"amove {X0} {Y0 - 0.1}", f"aline {X1} {Y0 - 0.1}"]
for t in range(-5, 16, 5):
    L += [f"amove {gx(t)} {Y0 - 0.1}", "rline 0 0.09", "set just tc", f"amove {gx(t)} {Y0 - 0.2}", f'write "\\tex{{${t}$}}"']
L += ["set just tc", f"amove {(X0 + X1) / 2} {Y0 - 0.62}", 'write "\\tex{accuracy difference (points)}"',
      "set lstyle 2", "set color gray50", f"amove {gx(0)} {Y0 - 0.1}", f"aline {gx(0)} {H - 0.1}", "set lstyle 1", "set color black"]
for i, (c, N, (g, a, b)) in enumerate(zip(z["cells"], z["N"], z["gap"])):
    y = Y0 + (n - 1 - i) * DY + 0.2; col, mk = (HET, "fcircle") if g > 0 else (HOM, "wcircle")
    txt = LAB[str(c)] + (", $N=10^3$" if N == 1000 else "")
    L += ["set just rc", f"amove {X0 - 0.15} {y}", f'write "\\tex{{{txt}}}"', f"set color {col}", "set lwidth .03",
          f"amove {gx(a)} {y}", f"aline {gx(b)} {y}", f"amove {gx(g)} {y}", "set hei .2", f"marker {mk}", "set hei .33", "set lwidth .018", "set color black"]
    if np.isfinite(z["net"][i]): L += [f"amove {gx(float(z['net'][i]))} {y}", "set hei .2", "marker ftriangle", "set hei .33"]
L += ["set hei .33", "set just bl", f"amove {gx(0) + 0.12} {H - 0.28}", f"set color {HET}", 'write "heterodyne better"',
      "set just br", f"amove {gx(0) - 0.12} {H - 0.28}", f"set color {HOM}", 'write "homodyne better"', "set color black"]
KY = H + 0.2                                                                   # key above the panel, outside the axes
L[0] = f"size {W} {H + 0.5}"
for x, mk, t in ((gx(-4.0), "fcircle", "observer"), (gx(6.0), "ftriangle", "networks")):
    L += ["set color black", f"amove {x} {KY}", "set hei .2", f"marker {mk}", "set hei .33", "set just lc", f"amove {x + 0.18} {KY}", f'write "{t}"']
open(f"{here}/fig_observer.gle", "w").write("\n".join(L) + "\n")
res = subprocess.run(["gle", "-d", "pdf", "fig_observer.gle"], cwd=here, capture_output=True, text=True); print(res.stdout[-400:], res.stderr[-400:])
