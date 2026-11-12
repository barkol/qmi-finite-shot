"""Drawing layer (GLE) for fig_fisher_overview (Fig. 1). Reads only the cached npz.
Needs GLE 4.3 (glx.sourceforge.io) with LaTeX for the \\tex labels; writes fig_fisher_overview.pdf here.
Colours: Wong vermillion (heterodyne) and blue (homodyne); every series also differs by line style or marker fill (BW-safe)."""
import os, subprocess, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dane_fig_fisher_overview import load
z = load(); here = os.path.dirname(os.path.abspath(__file__)); D = f"{here}/gle"; os.makedirs(D, exist_ok=True)
HET, HOM = "rgb255(213,94,0)", "rgb255(0,114,178)"
def dat(name, arr): np.savetxt(f"{D}/{name}.dat", arr, fmt="%.6f"); return f'"gle/{name}.dat"'
nb = z["nb"]
for k in ("het_all", "het_l3", "hom_l1", "hom_l0"): dat(k, np.c_[nb, z[k]])
mQ, mW = z["med_Q"], z["med_W"]
dat("medQ", np.c_[mQ[:, 0] + 0.07, mQ[:, 1], mQ[:, 1] - mQ[:, 2], mQ[:, 3] - mQ[:, 1]])
dat("medW", np.c_[mW[:, 0] - 0.07, mW[:, 1], mW[:, 1] - mW[:, 2], mW[:, 3] - mW[:, 1]])
e3 = z["edge_l3"]; dat("e3Q", e3[:, [0, 1]]); dat("e3W", e3[:, [0, 2]])
for k in ("central", "next", "edge"): dat(f"s_{k}", z[f"star_{k}"])
W, H = 8.6, 5.0
L = [f"size {W} {2 * H + 0.9}", "set font texcmr", "set hei .3", "set lwidth .022"]
# ---- panel (a)
L += [f"amove 0 {H + 0.6}", "begin graph", f"  size {W} {H}", "  scale auto", "  nobox",
      "  xaxis min 0 max 5 dticks 1 hei .3", "  yaxis min 0 max 1 dticks 0.25 hei .3",
      '  xtitle "\\tex{thermal occupation $\\bar n$}" hei .3 dist .1',
      '  ytitle "\\tex{$F/F_{\\mathrm{SLD}}$}" hei .3 dist .1',
      '  data "gle/het_all.dat" d1=c1,c2', f"  d1 line lstyle 1 lwidth .055 color {HET}",
      '  data "gle/het_l3.dat" d2=c1,c2', f"  d2 line lstyle 2 lwidth .03 color {HET}",
      '  data "gle/hom_l1.dat" d3=c1,c2', f"  d3 line lstyle 1 lwidth .022 color {HOM}",
      '  data "gle/hom_l0.dat" d4=c1,c2', f"  d4 line lstyle 3 lwidth .03 color {HOM}",
      '  data "gle/e3Q.dat" d5=c1,c2', f"  d5 marker fcircle msize .22 color {HET}",
      '  data "gle/e3W.dat" d6=c1,c2', f"  d6 marker wcircle msize .22 color {HOM}",
      '  data "gle/medQ.dat" d7=c1,c2 d8=c1,c3 d9=c1,c4', f"  d7 marker fsquare msize .2 color {HET} errdown d8 errup d9 errwidth .1",
      '  data "gle/medW.dat" d10=c1,c2 d11=c1,c3 d12=c1,c4', f"  d10 marker wsquare msize .2 color {HOM} errdown d11 errup d12 errwidth .1",
      "end graph"]
gx = lambda x: 1.45 + (W - 1.45 - 0.25) * x / 5      # approximate graph-to-page mapping for labels (checked visually)
gy = lambda y: H + 0.6 + 0.85 + (H - 0.85 - 0.25) * y
labs = [(2.3, 0.955, "\\tex{heterodyne, $\\lambda$}", HET), (3.3, 0.655, "\\tex{heterodyne, $l=3$}", HET),
        (2.15, 0.548, "\\tex{homodyne, $l\\geq1$}", HOM), (2.15, 0.415, "\\tex{homodyne, $l=0$}", HOM)]
L += ["set just lc", "set hei .27"]
for x, y, t, c in labs: L += [f"set color {c}", f"amove {gx(x)} {gy(y)}", f'write "{t}"']
L += ["set color black", f"amove {gx(1.45)} {gy(0.335)}", 'write "\\tex{filled: heterodyne; open: homodyne}"',
      f"amove {gx(1.45)} {gy(0.265)}", 'write "\\tex{circles: $l=3$; squares: random $\\Delta$}"',
      "set just bl", "set hei .3", f"amove 0.1 {2 * H + 0.55}", 'write "(a)"']
# ---- panel (b)
L += ["amove 0 0.2", "begin graph", f"  size {W} {H}", "  scale auto", "  nobox",
      "  xaxis min 0 max 21 dticks 5 hei .3", "  yaxis min 0 max 5.5 dticks 1 hei .3",
      '  xtitle "\\tex{order $n$ of the Hermite--Gauss mode}" hei .3 dist .1',
      '  ytitle "\\tex{threshold $\\bar n^\\ast$}" hei .3 dist .1',
      '  data "gle/s_central.dat" d1=c1,c2', "  d1 marker fcircle msize .2 line lstyle 1 lwidth .02 color black",
      '  data "gle/s_next.dat" d2=c1,c2', "  d2 marker wsquare msize .2 line lstyle 2 lwidth .02 color gray50",
      '  data "gle/s_edge.dat" d3=c1,c2', f"  d3 marker ftriangle msize .2 color {HET}",
      "end graph"]
hx = lambda x: 1.45 + (W - 1.45 - 0.25) * x / 21
hy = lambda y: 0.2 + 0.85 + (H - 0.85 - 0.25) * y / 5.5
leg = [(4.9, "fcircle", "black", "\\tex{central, $|l|\\leq1$}"), (4.45, "wsquare", "gray50", "\\tex{$|l|=n-2$}"),
       (4.0, "ftriangle", HET, "\\tex{edge, $|l|=n$}")]
for y, mk, c, txt in leg:
    L += [f"set color {c}", f"amove {hx(1.2)} {hy(y)}", "set hei .3", f"marker {mk} .7", "set color black",
          "set just lc", "set hei .27", f"amove {hx(1.9)} {hy(y)}", f'write "{txt}"']
L += ["set just lc", "set hei .27", f"amove {hx(13.2)} {hy(2.6)}", 'write "\\tex{homodyne better}"',
      f"amove {hx(2.2)} {hy(3.0)}", 'write "\\tex{heterodyne better}"',
      "set just bl", "set hei .3", f"amove 0.1 {H + 0.25}", 'write "(b)"']
open(f"{here}/fig_fisher_overview.gle", "w").write("\n".join(L) + "\n")
r = subprocess.run(["gle", "-d", "pdf", "fig_fisher_overview.gle"], cwd=here, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
