"""Drawing layer (GLE) for fig_mode_efficiency (Fig. 2). Reads only the cached npz. Needs GLE 4.3 with LaTeX.
Heterodyne: Wong vermillion, filled markers, solid lines; homodyne: Wong blue, open markers, dashed lines (BW-safe);
marker shape = family of modes (circle edge, square |l| = n-2, triangle central)."""
import os, subprocess, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dane_fig_mode_efficiency import load
z = load(); here = os.path.dirname(os.path.abspath(__file__)); G = f"{here}/gle"; os.makedirs(G, exist_ok=True)
HET, HOM = "rgb255(213,94,0)", "rgb255(0,114,178)"
MK = {"edge": "circle", "next": "square", "central": "triangle"}
W, H = 8.6, 5.0
L = [f"size {W} {2 * H + 2.05}", "set font texcmr", "set hei .3", "set lwidth .02"]
def panel(nb, y0, letter):
    s = [f"amove 0 {y0}", "begin graph", f"  size {W} {H}", "  scale auto", "  nobox",
         "  xaxis min 0 max 21 dticks 5 hei .3", "  yaxis min 0.001 max 1 log hei .3",
         '  xtitle "\\tex{order $n$ of the mode}" hei .3 dist .1', '  ytitle "\\tex{efficiency $F/F_{\\mathrm{SLD}}$}" hei .3 dist .1']
    k = 1
    for fam in ("edge", "next", "central"):
        a = z[f"nb{nb}_{fam}"]
        np.savetxt(f"{G}/nb{nb}_{fam}.dat", a, fmt="%.8e")
        s += [f'  data "gle/nb{nb}_{fam}.dat" d{k}=c1,c2 d{k+1}=c1,c3',
              f"  d{k} marker f{MK[fam]} msize .17 line lstyle 1 lwidth .018 color {HET}",
              f"  d{k+1} marker w{MK[fam]} msize .17 line lstyle 3 lwidth .018 color {HOM}"]
        k += 2
    s += ["end graph", "set just bl", "set hei .3", f"amove 0.1 {y0 + H + 0.12}",
          f'write "\\tex{{({letter}) $\\bar n={nb:g}$}}"']
    return s
L += panel(3.0, 0.2, "b") + panel(0.5, H + 1.0, "a")
# legend: one row above both panels (never overlaps data)
ly = 2 * H + 1.78
items = [("w" + MK["edge"], "black", "\\tex{edge}"), ("w" + MK["next"], "black", "\\tex{$|l|=n-2$}"),
         ("w" + MK["central"], "black", "\\tex{central}"), ("fcircle", HET, "\\tex{heterodyne}"), ("wcircle", HOM, "\\tex{homodyne}")]
xs = [0.2, 1.3, 3.05, 4.6, 6.65]
for x, (mk, col, txt) in zip(xs, items):
    L += [f"set color {col}", f"amove {x} {ly}", f"marker {mk} .5", "set just lc", "set hei .27", f"amove {x + 0.25} {ly}", f'write "{txt}"', "set color black"]
open(f"{here}/fig_mode_efficiency.gle", "w").write("\n".join(L) + "\n")
r = subprocess.run(["gle", "-d", "pdf", "fig_mode_efficiency.gle"], cwd=here, capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-600:])
