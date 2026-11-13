"""Drawing layer (GLE) for fig_dephasing_sectors. Reads only the cached npz.
Needs GLE 4.3 (glx.sourceforge.io) with LaTeX for the \tex labels; writes fig_dephasing_sectors.pdf here."""
import os, subprocess, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dane_fig_dephasing_sectors import load
z = load(); here = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{here}/gle", exist_ok=True)
STY = {  # family: (label, marker, rgb). Light colours only on FILLED markers, open markers dark -> BW-safe
    "fock": ("Fock", "fcircle", (0, 0, 0)), "coherent": ("coherent", "fsquare", (230, 159, 0)),
    "thermal": ("thermal", "ftriangle", (86, 180, 233)), "sqvac": ("sq.\\ vacuum", "fdiamond", (0, 158, 115)),
    "cat": ("cat", "wsquare", (0, 114, 178)), "gkp": ("GKP", "wcircle", (213, 94, 0)),
    "binomial": ("binomial", "wtriangle", (90, 90, 90)), "num": ("num.\\ code", "cross", (0, 0, 0))}
order = ["fock", "coherent", "thermal", "sqvac", "cat", "gkp", "binomial", "num"]
def panel(fam, x0, letter, title, y0=1.2):
    fh, dl, lab, names = z[f"{fam}_fh"], z[f"{fam}_dlog"], z[f"{fam}_lab"], list(z[f"{fam}_names"])
    s = [f"amove {x0} {y0}", "begin graph", "  size 7.2 5.4", "  fullsize",
         "  xaxis min 0 max 1 dticks 0.25 hei .3", "  yaxis min -0.8 max 1.6 dticks 0.4 hei .3",
         '  xtitle "\\tex{$f_{|l|\\geq2}$}" hei .32 dist .1',
         '  ytitle "\\tex{$\\Delta\\ln(\\xi_W/\\xi_Q)$}" hei .32 dist .1']
    k = 1
    for nm in order:
        if nm not in names: continue
        m = lab == names.index(nm)
        np.savetxt(f"{here}/gle/{fam}_{nm}.dat", np.c_[fh[m], dl[m]], fmt="%.5f")
        lbl, mk, (r, g, b) = STY[nm]
        s += [f'  data "gle/{fam}_{nm}.dat" d{k}=c1,c2', f"  d{k} marker {mk} msize .21 color rgb255({r},{g},{b})"]
        k += 1
    s += ["end graph", "set lstyle 2", "set lwidth .01", "set color gray50",
          f"amove {x0} {y0 + 5.4 * 0.8 / 2.4}", f"aline {x0 + 7.2} {y0 + 5.4 * 0.8 / 2.4}", "set lstyle 1", "set color black",
          "set just bl", "set hei .3", f"amove {x0} {y0 + 5.4 + 0.15}", f'write "({letter}) {title}"',
          "set just tr", f"amove {x0 + 7.2 - 0.2} {y0 + 5.4 - 0.2}",
          f'write "\\tex{{$\\rho_S={float(z[fam + "_rho"]):.2f}$}}"']
    return s
def panel_pred(fam, x0, letter, title):
    """observed change against the change predicted by the theory; dashed diagonal = equality"""
    pr, dl, lab, names = z[f"{fam}_pred"], z[f"{fam}_dlog"], z[f"{fam}_lab"], list(z[f"{fam}_names"])
    lo, hi = -0.8, 2.0
    s = [f"amove {x0} 1.2", "begin graph", "  size 7.2 5.4", "  fullsize",
         f"  xaxis min {lo} max {hi} dticks 0.4 hei .3", "  yaxis min -0.8 max 1.6 dticks 0.4 hei .3",
         '  xtitle "\\tex{predicted $\\Delta\\ln(F_W/F_Q)$}" hei .32 dist .1',
         '  ytitle "\\tex{$\\Delta\\ln(\\xi_W/\\xi_Q)$}" hei .32 dist .1']
    k = 1; inr = z[f"{fam}_inrange"]
    for sel, tag in ((~inr, "out"), (inr, "in")):                      # pairs outside the range of the theory first, in grey
        for nm in order:
            if nm not in names: continue
            m = (lab == names.index(nm)) & sel
            if not m.any(): continue
            np.savetxt(f"{here}/gle/{fam}_{nm}_pred_{tag}.dat", np.c_[pr[m], dl[m]].reshape(-1, 2), fmt="%.5f")
            lbl, mk, (r, g, b) = STY[nm]
            col = f"rgb255({r},{g},{b})" if tag == "in" else "rgb255(185,185,185)"
            s += [f'  data "gle/{fam}_{nm}_pred_{tag}.dat" d{k}=c1,c2', f"  d{k} marker {mk} msize .21 color {col}"]
            k += 1
    gx = lambda v: x0 + 7.2 * (v - lo) / (hi - lo); gy = lambda v: 1.2 + 5.4 * (v + 0.8) / 2.4
    s += ["end graph", "set lstyle 2", "set lwidth .02", "set color gray50",
          f"amove {gx(-0.8)} {gy(-0.8)}", f"aline {gx(1.6)} {gy(1.6)}", "set lstyle 1", "set lwidth .022", "set color black",
          "set just bl", "set hei .3", f"amove {x0} {1.2 + 5.4 + 0.15}", f'write "({letter}) {title}"',
          "set just tl", f"amove {x0 + 0.2} {1.2 + 5.4 - 0.2}",
          f'write "\\tex{{$\\rho_S={float(z[fam + "_rho_pred_in"]):.2f}$ (colored), ${float(z[fam + "_rho_pred"]):.2f}$ (all)}}"']
    return s
L = ["size 18 15.4", "set font texcmr", "set hei .3", "set lwidth .022"]
top = panel("eqn", 1.4, "a", "base set", 8.2) + panel("ahm", 10.4, "b", "set of Ahmed et al.", 8.2)
L += top + panel_pred("eqn", 1.4, "c", "base set, theory") + panel_pred("ahm", 10.4, "d", "set of Ahmed et al., theory")
# shared legend: one row above the panels (manual, so it never overlaps data)
xs = [1.5 + 2.15 * k for k in range(8)]
for x, nm in zip(xs, order):
    lbl, mk, (r, g, b) = STY[nm]
    L += [f"set color rgb255({r},{g},{b})", f"amove {x} 14.85", "set hei .3", f"marker {mk}", "set color black",
          "set just lc", "set hei .27", f"amove {x + 0.25} 14.85", f'write "{lbl}"']
open(f"{here}/fig_dephasing_sectors.gle", "w").write("\n".join(L) + "\n")
r = subprocess.run(["gle", "-d", "pdf", "fig_dephasing_sectors.gle"], cwd=here, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
