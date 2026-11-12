"""C1: mode-weight formula (eq. modes, app. A) vs direct integrals int delta^2/p.
Writes out/c1b.json after each case (resumable: done cases are skipped)."""
import json, os, sys, numpy as np
from fisherlib import *

OUT = "out/c1b.json"; os.makedirs("out", exist_ok=True)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, "w"), indent=1)

rng = np.random.default_rng(20261112)
DIM = 5
cases = [(nb, i) for nb in (0.0, 0.5, 1.0, 2.0, 5.0) for i in range(6)]
Deltas = {c: random_traceless(DIM, rng) for c in cases}   # fixed draw order -> reproducible

for nb, i in cases:
    key = f"nb{nb}_i{i}"
    if key in res: continue
    D = Deltas[(nb, i)]
    g = Grid(nb + 0.5, DIM)
    FWd, FQd = g.F_hom_random(D), g.F_het(D)
    FWf, FQf, _ = formula_F2(D, nb + 0.5, nmax=200 if nb >= 2 else 120)
    res[key] = dict(nbar=nb, FW_direct=FWd, FW_formula=FWf, FQ_direct=FQd, FQ_formula=FQf,
                    relW=abs(FWf - FWd) / FWd, relQ=abs(FQf - FQd) / FQd)
    save(); print(key, f"FW {FWd:.6g}/{FWf:.6g}  FQ {FQd:.6g}/{FQf:.6g}", flush=True)

# convergence of the truncation at nbar=5
if "conv_nb5" not in res:
    D = Deltas[(5.0, 0)]; g = Grid(5.5, DIM)
    res["conv_nb5"] = {str(n): formula_F2(D, 5.5, nmax=n)[:2] for n in (40, 80, 160, 240)}
    save(); print("conv", res["conv_nb5"])

# displaced reference: covariance (F unchanged when both reference and Delta are displaced)
for nb, al in ((0.0, 0.8 + 0.3j), (1.0, 0.8 + 0.3j)):
    key = f"displ_nb{nb}"
    if key in res: continue
    D = Deltas[(nb, 1)]; big = 45
    Dp = np.zeros((big, big), complex); Dp[:DIM, :DIM] = D
    U = displace_op(al, big); Dd = U @ Dp @ U.conj().T
    g0 = Grid(nb + 0.5, DIM)
    gd = Grid(nb + 0.5, big, shift=(np.sqrt(2) * al.real, np.sqrt(2) * al.imag), nth=360)
    res[key] = dict(FW0=g0.F_hom_random(D), FWd=gd.F_hom_random(Dd), FQ0=g0.F_het(D), FQd=gd.F_het(Dd))
    save(); print(key, res[key], flush=True)

# K equally spaced phases: equal to random-phase value iff K > max|m-m'| (here DIM-1 = 4)
if "Kphases" not in res:
    out = {}
    for nb in (0.0, 1.0):
        g = Grid(nb + 0.5, DIM)
        for i in range(3):
            D = Deltas[(nb, i)]; FW = g.F_hom_random(D)
            out[f"nb{nb}_i{i}"] = {str(K): g.F_hom_K(D, K) / FW for K in (2, 3, 4, 5, 6, 36)}
    res["Kphases"] = out; save(); print("K", out)
print("done")
