"""Drawing layer (GLE) for fig_squeezed_bounds. Reads only the cached npz. Needs GLE 4.3 with LaTeX.
Heterodyne: Wong vermillion, solid, filled markers (nbar = 0 coincides with the homodyne curve and is not drawn); homodyne at a uniform laboratory phase: Wong blue, dashed, open circles;
homodyne at the phase adapted to the squeezing: Wong blue, dotted, open triangles (BW-safe)."""
import os, subprocess, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dane_fig_squeezed_bounds import load
z = load(); here = os.path.dirname(os.path.abspath(__file__)); G = f"{here}/gle"; os.makedirs(G, exist_ok=True)
HET, HOM = "rgb255(213,94,0)", "rgb255(0,114,178)"
W, H = 8.6, 4.6
r = z["r"]; sub = slice(None, None, 6)
np.savetxt(f"{G}/a.dat", np.c_[r, z["het_nb0.0"], z["het_nb0.5"], z["het_nb3.0"], z["hom_lab"], z["hom_adapted"]], fmt="%.8e")
np.savetxt(f"{G}/am.dat", np.c_[r, z["het_nb0.0"], z["het_nb0.5"], z["het_nb3.0"], z["hom_lab"], z["hom_adapted"]][sub], fmt="%.8e")
np.savetxt(f"{G}/b.dat", z["orders"], fmt="%.8e")
L = [f"size {W} {2 * H + 1.9}", "set font texcmr", "set hei .3", "set lwidth .02"]
L += [f"amove 0 {H + 0.9}", "begin graph", f"  size {W} {H}", "  scale auto", "  nobox",
      "  xaxis min 0 max 1.5 dticks 0.5 hei .3", "  yaxis min 0.4 max 1 dticks 0.2 hei .3",
      '  xtitle "\\tex{squeezing parameter $|r|$}" hei .3 dist .1', '  ytitle "\\tex{largest efficiency}" hei .3 dist .1',
      '  data "gle/a.dat" d1=c1,c2 d2=c1,c3 d3=c1,c4 d4=c1,c5 d5=c1,c6',
      '  data "gle/am.dat" d6=c1,c2 d7=c1,c3 d8=c1,c4 d9=c1,c5 d10=c1,c6',
      f"  d2 line lstyle 1 lwidth .018 color {HET}", f"  d3 line lstyle 1 lwidth .018 color {HET}",
      f"  d4 line lstyle 3 lwidth .018 color {HOM}", f"  d5 line lstyle 2 lwidth .018 color {HOM}",
      f"  d7 marker fsquare msize .15 color {HET}", f"  d8 marker ftriangle msize .15 color {HET}",
      f"  d9 marker wcircle msize .15 color {HOM}", f"  d10 marker wtriangle msize .15 color {HOM}",
      "end graph", "set just bl", f"amove 0.1 {2 * H + 1.02}", 'write "\\tex{(a)}"']
L += ["amove 0 0.2", "begin graph", f"  size {W} {H}", "  scale auto", "  nobox",
      "  xaxis min 0 max 13 dticks 2 hei .3", "  yaxis min 0 max 1 dticks 0.2 hei .3",
      '  xtitle "\\tex{order $n$}" hei .3 dist .1', '  ytitle "\\tex{largest efficiency in order $n$}" hei .3 dist .1',
      '  data "gle/b.dat" d1=c1,c2 d2=c1,c3 d3=c1,c4',
      f"  d1 marker fsquare msize .17 line lstyle 1 lwidth .018 color {HET}",
      f"  d2 marker wcircle msize .17 line lstyle 3 lwidth .018 color {HOM}",
      f"  d3 marker wtriangle msize .17 line lstyle 2 lwidth .018 color {HOM}",
      "end graph", "set just bl", f"amove 0.1 {H + 0.32}", f'write "\\tex{{(b) $\\bar n={float(z["nb_b"]):g}$, $r={float(z["r_b"]):g}$}}"']
ly = 2 * H + 1.62
items = [("fsquare", HET, "\\tex{het., $\\bar n=0.5$}"), ("ftriangle", HET, "\\tex{$3$}"),
         ("wcircle", HOM, "\\tex{hom., uniform phase}"), ("wtriangle", HOM, "\\tex{adapted}")]
xs = [0.2, 2.55, 3.55, 6.6]
for x, (mk, col, txt) in zip(xs, items):
    L += [f"set color {col}", f"amove {x} {ly}", f"marker {mk} .5", "set just lc", "set hei .27", f"amove {x + 0.25} {ly}", f'write "{txt}"', "set color black"]
open(f"{here}/fig_squeezed_bounds.gle", "w").write("\n".join(L) + "\n")
res = subprocess.run(["gle", "-d", "pdf", "fig_squeezed_bounds.gle"], cwd=here, capture_output=True, text=True)
print(res.stdout[-600:], res.stderr[-600:])
