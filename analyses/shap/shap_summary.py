"""Metryki z atrybucji (bez obliczen sieci): out/attr_*.npz, hell_*.npz, sanity_*.npz -> out/summary.json, out/maps.npz.
  python analyses/shap/shap_summary.py [--small]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shap_compute as sc

FAMS = sc.FAMS; x = sc.x
XX, PP = np.meshgrid(x, x)                  # obraz [p, x] (qutip: wiersze = p)
RR = np.hypot(XX, PP)

def sp(a, b, m=None):
    a, b = a.ravel(), b.ravel()
    if m is not None: a, b = a[m.ravel()], b[m.ravel()]
    return float(spearmanr(a, b)[0])

def norm1(A):
    s = np.abs(A).sum(axis=(-2, -1), keepdims=True); return A / np.where(s > 0, s, 1)

def align(A, phi, rep, sg):
    """A: (n, ..., H, W) -> wyrownane wzgledem fazy stanu."""
    out = np.empty_like(A)
    for i in range(len(A)):
        p = None if np.isnan(phi[i]) else phi[i]
        if A[i].ndim == 3:
            out[i] = np.stack([sc.align_sino(a, p, sg[1]) if rep == "S" else sc.align_img(a, p, sg[0]) for a in A[i]])
        else:
            out[i] = sc.align_sino(A[i], p, sg[1]) if rep == "S" else sc.align_img(A[i], p, sg[0])
    return out

def masks(inp, rep, name=""):
    """Nosnik stanu (1% max) i obszar ujemnego W (fizyczna struktura nieklasyczna) na tej samej siatce co wejscie."""
    if rep == "S":
        S = inp["Sx"]; m = np.abs(S) > 0.01 * np.abs(S).max(axis=(1, 2), keepdims=True); return m, None
    if rep == "Q":
        Q = inp["Qtrue"]; sup = Q > 0.01 * Q.max(axis=(1, 2), keepdims=True)
    else:
        W = inp["Wtrue"]; sup = np.abs(W) > 0.01 * np.abs(W).max(axis=(1, 2), keepdims=True)
    W = inp["Wtrue"]; neg = W < -0.01 * np.abs(W).max(axis=(1, 2), keepdims=True)
    return sup, neg

def main(OUT):
    sc.OUT = OUT; sg = sc.conv(); R = {}; MAPS = {}
    for cell in sc.CELLS:
        inp = dict(np.load(OUT / f"inputs_{cell}.npz")); lab = inp["label"]; phi = inp["phi"]
        hz = np.load(OUT / f"hell_{cell}.npz")
        H = align(norm1(hz["H"]), phi, "Q", sg); HS = align(norm1(hz["HS"]), phi, "S", sg)
        MAPS[f"{cell}|HQ"] = np.stack([H[lab == c].mean(0) for c in range(7)])
        MAPS[f"{cell}|HS"] = np.stack([HS[lab == c].mean(0) for c in range(7)])
        for rep, key in (("Q", "Qtrue"), ("W", "Wtrue"), ("S", "Sx")):
            st = align(inp[key][:, None] if False else inp[key], phi, rep, sg)
            MAPS[f"{cell}|state_{rep}"] = np.stack([st[lab == c].mean(0) for c in range(7)])
        R.setdefault("C_half", {})[cell] = {FAMS[c]: dict(het=float(np.median(hz["C_half"][lab == c])), hom=float(np.median(hz["C_halfS"][lab == c]))) for c in range(7)}
        for f in sorted(OUT.glob(f"attr_{cell}_*.npz")):
            if ".tmp" in f.name: continue
            _, _, name, mod_seed, kind, base = f.stem.split("_"); name = f"{name}_{mod_seed}"
            rep = sc.MOD_REP[name.split("_")[0]]
            z = np.load(f); A = z["A"].astype(np.float32); Lg = z["logits"]                 # (n,R,H,W)
            a = norm1(A); al = align(a, phi, rep, sg)
            signed = np.stack([al[lab == c].mean((0, 1)) for c in range(7)])
            absm = np.stack([np.abs(al[lab == c]).mean((0, 1)) for c in range(7)])
            tag = f"{cell}|{name}|{kind}|{base}"; MAPS[tag + "|signed"] = signed; MAPS[tag + "|abs"] = absm
            sup, neg = masks(inp, "W" if name.startswith("Wex") else rep)
            aa = np.abs(a)
            out_share = 1 - (aa * sup[:, None]).sum((-2, -1))                                  # (n,R)
            r = dict(acc={FAMS[c]: float((Lg[lab == c].argmax(-1) == c).mean()) for c in range(7)},
                     out_support={FAMS[c]: float(np.median(out_share[lab == c])) for c in range(7)},
                     out_support_all=float(np.median(out_share)),
                     area_out={FAMS[c]: float(1 - sup[lab == c].mean()) for c in range(7)})
            if neg is not None:
                nsh = (aa * neg[:, None]).sum((-2, -1)); narea = neg.mean((1, 2))
                r["neg_share"] = {FAMS[c]: float(np.median(nsh[lab == c])) for c in range(7) if narea[lab == c].mean() > 0.002}
                r["neg_area"] = {FAMS[c]: float(narea[lab == c].mean()) for c in range(7) if narea[lab == c].mean() > 0.002}
                r["neg_enrich"] = {k: r["neg_share"][k] / r["neg_area"][k] for k in r["neg_share"]}
                # sciśnieta proznia: ogony = nosnik poza r > 2 (szerokosc prozni 0.71)
                tail = sup & (RR[None] > 2.0); c = 2
                tsh = (aa[lab == c] * tail[lab == c][:, None]).sum((-2, -1))
                r["sqvac_tail_share"] = float(np.median(tsh)); r["sqvac_tail_area"] = float(tail[lab == c].mean())
                r["sqvac_tail_share_of_support"] = float(np.median(tsh / np.maximum((aa[lab == c] * sup[lab == c][:, None]).sum((-2, -1)), 1e-12)))
                r["sqvac_tail_area_of_support"] = float(tail[lab == c].sum() / sup[lab == c].sum())
            # zgodnosc z idealnym obserwatorem: Q-siec vs Hellinger heterodyny, S-siec vs Hellinger homodyny
            ref = MAPS[f"{cell}|HQ"] if rep in ("Q", "W") else MAPS[f"{cell}|HS"]
            st = MAPS[f"{cell}|state_{'S' if rep == 'S' else 'Q'}"]
            r["spear_hell_full"] = {FAMS[c]: sp(absm[c], ref[c]) for c in range(7)}
            r["spear_hell_supp"] = {FAMS[c]: sp(absm[c], ref[c], st[c] > 0.01 * st[c].max() if rep != "S" else np.abs(st[c]) > 0.01 * np.abs(st[c]).max()) for c in range(7)}
            if rep != "S":                                    # wzgledem samego stanu: |W| i Q (czy mapa to tylko "gdzie jest stan")
                r["spear_absW"] = {FAMS[c]: sp(absm[c], np.abs(MAPS[f"{cell}|state_W"][c])) for c in range(7)}
                r["spear_Q"] = {FAMS[c]: sp(absm[c], MAPS[f"{cell}|state_Q"][c]) for c in range(7)}
            R.setdefault("attr", {})[tag] = r
        # porownania map: baseline, seed, dokladne vs rekordy
        A_ = lambda t: MAPS.get(t + "|abs")
        for rep in ("Q", "W", "S"):
            for kind in ("ex", "rec"):
                g0, g1 = A_(f"{cell}|{rep}_s0|{kind}|otherG"), A_(f"{cell}|{rep}_s1|{kind}|otherG")
                if g0 is not None and g1 is not None: R.setdefault("seed_stab_G", {})[f"{cell}|{rep}|{kind}"] = {FAMS[c]: sp(g0[c], g1[c]) for c in range(7)}
                o = A_(f"{cell}|{rep}_s0|{kind}|other")
                if o is None: continue
                for b in ("vac", "zero"):
                    v = A_(f"{cell}|{rep}_s0|{kind}|{b}")
                    if v is not None: R.setdefault("baseline_sens", {})[f"{cell}|{rep}|{kind}|other_vs_{b}"] = {FAMS[c]: sp(o[c], v[c]) for c in range(7)}
                v = A_(f"{cell}|{rep}_s1|{kind}|other")
                if v is not None: R.setdefault("seed_stab", {})[f"{cell}|{rep}|{kind}"] = {FAMS[c]: sp(o[c], v[c]) for c in range(7)}
            e, rr = A_(f"{cell}|{rep}_s0|ex|other"), A_(f"{cell}|{rep}_s0|rec|other")
            if e is not None and rr is not None: R.setdefault("ex_vs_rec", {})[f"{cell}|{rep}"] = {FAMS[c]: sp(e[c], rr[c]) for c in range(7)}
        # spojnosc wewnatrzklasowa: mediana Spearmana |a| pojedynczego stanu z mapa klasowa (rekordy, s0)
    # sanity
    inp = dict(np.load(OUT / "inputs_clean.npz")); lab = inp["label"]
    for rep in ("Q", "W", "S"):
      for G in ("", "_G"):
        for kind in ("rec", "ex"):
            f0 = OUT / f"sanity_{rep}_s0_{kind}_0{G}.npz"
            if not f0.exists(): continue
            z0 = np.load(f0); A0 = z0["A"]; sel = z0["sel"]
            X = sc.get_X(inp, f"{rep}_s0", kind)[sel, 0]
            res = {"abs": [], "signed": [], "abs_vs_input": [], "out_support": [], "neg_enrich": [], "spear_hell": [], "classmean_signed": []}
            sup, neg = masks(inp, rep); sup, neg = sup[sel], (neg[sel] if neg is not None else None)
            hz = np.load(OUT / "hell_clean.npz"); Href = (hz["HS"] if rep == "S" else hz["H"])[sel]
            phs = inp["phi"][sel]; ls = lab[sel]
            cm0 = None
            for step in range(len(sc.CASCADE) + 1):
                As = np.load(OUT / f"sanity_{rep}_s0_{kind}_{step}{G}.npz")["A"]
                res["abs"].append(float(np.median([sp(np.abs(A0[i]), np.abs(As[i])) for i in range(len(sel))])))
                res["signed"].append(float(np.median([sp(A0[i], As[i]) for i in range(len(sel))])))
                res["abs_vs_input"].append(float(np.median([sp(np.abs(As[i]), np.abs(X[i])) for i in range(len(sel))])))
                aa = np.abs(norm1(As)); res["out_support"].append(float(np.median(1 - (aa * sup).sum((1, 2)))))
                if neg is not None:
                    nf = neg.mean((1, 2)) > 0.002
                    res["neg_enrich"].append(float(np.median(((aa * neg).sum((1, 2)) / np.maximum(neg.mean((1, 2)), 1e-9))[nf])))
                res["spear_hell"].append(float(np.median([sp(aa[i], Href[i], sup[i]) for i in range(len(sel))])))
                al = align(norm1(As), phs, rep, sg); cm = np.stack([al[ls == c].mean(0) for c in range(7)])
                if cm0 is None: cm0 = cm
                res["classmean_signed"].append(float(np.median([sp(cm0[c], cm[c]) for c in range(7)])))
                MAPS[f"sanity|{rep}|{kind}{G}|{step}"] = As[:: max(1, len(sel) // 7)][:7]
            MAPS[f"sanity|{rep}|{kind}|X"] = X[:: max(1, len(sel) // 7)][:7]
            R.setdefault("sanity", {})[f"{rep}|{kind}{G}"] = res
    R["cascade"] = ["trained"] + sc.CASCADE; R["conv"] = list(sg)
    json.dump(R, open(OUT / "summary.json", "w"), indent=1)
    np.savez_compressed(OUT / "maps.npz", **{k: v.astype(np.float32) for k, v in MAPS.items()})
    print("SUMMARY_DONE", OUT)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--small", action="store_true"); a = ap.parse_args()
    main(sc.HERE / ("out_small" if a.small else "out"))
