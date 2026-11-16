"""Drawing layer (GLE) for Fig. 7 (fig_dephasing_prediction): single-column figure with two panels, the observed change of
ln(xi_W/xi_Q) under dephasing against the prediction of the theory. Reads only the cache data/fig_dephasing_sectors.npz.
Needs GLE 4.3 with LaTeX for the \\tex labels; writes fig_dephasing_prediction.pdf here."""
import os, subprocess, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_fig_dephasing_sectors import load
z = load(); here = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{here}/gle", exist_ok=True)
import palette as _p
STY = {nm: (lbl, mk, _p.role(nm, "rgb")) for nm, lbl, mk in (   # colours: TURBO; light colours only on FILLED markers, open markers dark -> BW-safe
    ("fock", "Fock", "fcircle"), ("coherent", "coherent", "fsquare"), ("thermal", "thermal", "ftriangle"), ("sqvac", "sq.\\ vacuum", "fdiamond"),
    ("cat", "cat", "wsquare"), ("gkp", "GKP", "wcircle"), ("binomial", "binomial", "wtriangle"), ("num", "num.\\ code", "cross"))}
order = ["fock", "coherent", "thermal", "sqvac", "cat", "gkp", "binomial", "num"]
PW, PH, X0 = 7.2, 4.4, 1.45
def panel_pred(fam, y0, letter, title, xtitle):
    pr, dl, lab, names = z[f"{fam}_pred"], z[f"{fam}_dlog"], z[f"{fam}_lab"], list(z[f"{fam}_names"])
    lo, hi = -0.8, 2.0
    s = [f"amove {X0} {y0}", "begin graph", f"  size {PW} {PH}", "  fullsize",
         f"  xaxis min {lo} max {hi} dticks 0.4 hei .34", "  yaxis min -0.8 max 1.6 dticks 0.4 hei .34",
         '  ytitle "\\tex{observed $\\Delta\\ln(\\xi_W/\\xi_Q)$}" hei .34 dist .1']
    if xtitle: s += ['  xtitle "\\tex{predicted $\\Delta\\ln(F_W/F_Q)$}" hei .34 dist .1']
    k = 1; inr = z[f"{fam}_inrange"]
    for sel, tag in ((~inr, "out"), (inr, "in")):
        for nm in order:
            if nm not in names: continue
            m = (lab == names.index(nm)) & sel
            if not m.any(): continue
            np.savetxt(f"{here}/gle/{fam}_{nm}_pred_{tag}.dat", np.c_[pr[m], dl[m]].reshape(-1, 2), fmt="%.5f")
            lbl, mk, (r, g, b) = STY[nm]
            col = f"rgb255({r},{g},{b})" if tag == "in" else "rgb255(185,185,185)"
            s += [f'  data "gle/{fam}_{nm}_pred_{tag}.dat" d{k}=c1,c2', f"  d{k} marker {mk} msize {'.24' if tag == 'in' else '.12'} color {col}"]   # in range: large and coloured; out of range: small and grey (also distinct in grey scale)
            k += 1
    gx = lambda v: X0 + PW * (v - lo) / (hi - lo); gy = lambda v: y0 + PH * (v + 0.8) / 2.4
    s += ["end graph", "set lstyle 2", "set lwidth .02", "set color gray50",
          f"amove {gx(-0.8)} {gy(-0.8)}", f"aline {gx(1.6)} {gy(1.6)}", "set lstyle 1", "set lwidth .022", "set color black",
          "set just br", "set hei .34", f"amove {X0 + PW - 0.2} {y0 + 0.2}", f'write "({letter}) {title}"',
          "set just tl", f"amove {X0 + 0.2} {y0 + PH - 0.2}",
          f'write "\\tex{{$\\rho_S={float(z[fam + "_rho_pred_in"]):.2f}$ (colored), ${float(z[fam + "_rho_pred"]):.2f}$ (all)}}"']
    return s
YB, YT = 1.2, 1.2 + PH + 0.45
L = [f"size {X0 + PW + 0.25} {YT + PH + 1.25}", "set font texcmr", "set hei .34", "set lwidth .022"]
L += panel_pred("eqn", YT, "a", "base set", False) + panel_pred("ahm", YB, "b", "set of Ahmed et al.", True)
for i, nm in enumerate(order):                                   # legend: two rows of four above the panels
    x = X0 + 0.1 + 1.85 * (i % 4); y = YT + PH + 0.85 - 0.45 * (i // 4)
    lbl, mk, (r, g, b) = STY[nm]
    L += [f"set color rgb255({r},{g},{b})", f"amove {x} {y}", "set hei .34", f"marker {mk}", "set color black",
          "set just lc", "set hei .33", f"amove {x + 0.25} {y}", f'write "{lbl}"']
open(f"{here}/fig_dephasing_prediction.gle", "w").write("\n".join(L) + "\n")
r = subprocess.run(["gle", "-d", "pdf", "fig_dephasing_prediction.gle"], cwd=here, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
