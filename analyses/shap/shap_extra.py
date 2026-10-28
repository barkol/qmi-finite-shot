"""Odniesienia zerowe dla (c): Spearman map klasowych z mapa Hellingera dla (i) samego stanu, (ii) sieci wytrenowanej i
calkowicie przelosowanej na tym samym podzbiorze sanity (10/klase, czyste, wyrownane). -> out/extra.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shap_compute as sc
from shap_summary import align, norm1, sp, FAMS
OUT = sc.OUT; sg = sc.conv(); E = {}
D = np.load(OUT / "maps.npz")
for cell in sc.CELLS:
    for rep, H, st in (("Q", "HQ", "state_Q"), ("S", "HS", "state_S")):
        Hm, Sm = D[f"{cell}|{H}"], D[f"{cell}|{st}"]
        E[f"{cell}|state{rep}_vs_{H}"] = {FAMS[c]: sp(np.abs(Sm[c]), Hm[c], np.abs(Sm[c]) > 0.01 * np.abs(Sm[c]).max()) for c in range(7)}
inp = dict(np.load(OUT / "inputs_clean.npz")); lab = inp["label"]
for rep, H, st in (("Q", "HQ", "state_Q"), ("W", "HQ", "state_Q"), ("S", "HS", "state_S")):
    for kind in ("rec", "ex"):
        for G in ("", "_G"):
            z = np.load(OUT / f"sanity_{rep}_s0_{kind}_0{G}.npz"); sel = z["sel"]; ls = lab[sel]; ph = inp["phi"][sel]
            for step in (0, 6):
                A = np.load(OUT / f"sanity_{rep}_s0_{kind}_{step}{G}.npz")["A"]
                al = np.abs(align(norm1(A), ph, "S" if rep == "S" else "Q", sg))
                Hm, Sm = D[f"clean|{H}"], D[f"clean|{st}"]
                E[f"clean|{rep}|{kind}{G}|step{step}_vs_{H}"] = {FAMS[c]: sp(al[ls == c].mean(0), Hm[c], np.abs(Sm[c]) > 0.01 * np.abs(Sm[c]).max()) for c in range(7)}
json.dump(E, open(OUT / "extra.json", "w"), indent=1)
for k, v in E.items(): print(f"{k:40s}", " ".join(f"{v[f]:.2f}" for f in FAMS), " med", f"{np.median(list(v.values())):.2f}")

# Iloraz wiarygodnosci: idealny detektor zliczen heterodynowych sumuje log p(x)/q(x) po zdarzeniach, wiec jego czulosc na
# piksel to |log p/q| (z podloga eps*max, bo poza nosnikami obu stanow iloraz jest nieokreslony). Porownanie z EG sieci Q.
L = {}
for cell in sc.CELLS:
    inp = dict(np.load(OUT / f"inputs_{cell}.npz")); lab = inp["label"]; hz = np.load(OUT / f"hell_{cell}.npz")
    P = np.clip(inp["Qtrue"].astype(np.float64), 0, None); P /= P.sum((1, 2), keepdims=True)
    for eps in (1e-4, 1e-3):
        fl = eps * P.max((1, 2), keepdims=True)
        LLR = np.abs(np.log(P + fl) - np.log(P[hz["nn"]] + fl))
        Lm = align(LLR / LLR.sum((1, 2), keepdims=True), inp["phi"], "Q", sg); Lc = np.stack([Lm[lab == c].mean(0) for c in range(7)])
        for nm in ("Q_s0", "W_s0"):
            for kind in ("rec", "ex"):
                for b in ("otherG", "other"):
                    m = D[f"{cell}|{nm}|{kind}|{b}|abs"]
                    L[f"{cell}|{nm}|{kind}|{b}|LLR_eps{eps:g}"] = {FAMS[c]: sp(m[c], Lc[c]) for c in range(7)}
        # i siec losowa (podzbior sanity, tylko czyste)
        if cell == "clean":
            for G in ("_G", ""):
                z = np.load(OUT / f"sanity_Q_s0_rec_0{G}.npz"); sel = z["sel"]; ls = lab[sel]
                for step in (0, 6):
                    A = np.load(OUT / f"sanity_Q_s0_rec_{step}{G}.npz")["A"]
                    al = np.abs(align(norm1(A), inp["phi"][sel], "Q", sg))
                    L[f"clean|Q_s0|rec{G}|sanity_step{step}|LLR_eps{eps:g}"] = {FAMS[c]: sp(al[ls == c].mean(0), Lc[c]) for c in range(7)}
E.update(L); json.dump(E, open(OUT / "extra.json", "w"), indent=1)
for k, v in L.items(): print(f"{k:46s}", " ".join(f"{v[f]:.2f}" for f in FAMS), " med", f"{np.median(list(v.values())):.2f}")
