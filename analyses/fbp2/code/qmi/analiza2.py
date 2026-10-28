"""Q-W sparowane (jak make_numbers.paired_diff) dla W starego (runs_eq/pary) i W z nowym FBP (results/pary2), te same stany i sieci Q.
  python -m qmi.analiza2 --old <runs_eq/pary> --new <results/pary2> --variants kak2,hann2 --out <json>"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import t as T
CELLS = [("loss", 0), ("loss", 50), ("loss", 80), ("dephase", 20)]; NS = (100, 1000, 10000)

def pdiff(MQ, MW, y):
    """MQ, MW: (siec, stan) poprawnosci; zwraca Q-W w pp z 95% CI (skladnik stanow + skladnik ziaren Welcha jesli >=2 ziarna po obu stronach)"""
    aQ, aW = MQ.mean(1), MW.mean(1); di = MQ.mean(0) - MW.mean(0); n = len(y)
    se_st = np.sqrt(sum((y == c).sum() * di[y == c].var(ddof=1) for c in np.unique(y))) / n
    D = float(aQ.mean() - aW.mean()); r = dict(diff=100 * D, accQ=100 * float(aQ.mean()), accW=100 * float(aW.mean()), nQ=len(aQ), nW=len(aW), se_states=100 * float(se_st))
    if len(aQ) > 1 and len(aW) > 1:
        vq, vw = aQ.var(ddof=1) / len(aQ), aW.var(ddof=1) / len(aW)
        df = (vq + vw) ** 2 / (vq ** 2 / (len(aQ) - 1) + vw ** 2 / (len(aW) - 1)) if vq + vw > 0 else 1e9
        h = float(np.hypot(1.959964 * se_st, T.ppf(0.975, df) * np.sqrt(vq + vw))); r.update(se_seeds=100 * float(np.sqrt(vq + vw)), df=float(df))
    else: h = 1.959964 * float(se_st); r["note"] = "tylko skladnik stanow (1 ziarno W)"
    r.update(lo=100 * (D - h) if False else 100 * D - 100 * h, hi=100 * D + 100 * h, ci_half=100 * h); return r

def load(p):
    d = np.load(p); return d["y"], [str(x) for x in d["names"]], d["correct"].astype(float)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--old", required=True); ap.add_argument("--new", required=True); ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--note", default=""); ap.add_argument("--seeds", default=None, help="podzbior ziaren nowych sieci, np. 0"); a = ap.parse_args()
    res = {"_note": a.note, "per_seed": {}}
    for ch, lv in CELLS:
        for N in NS:
            po = Path(a.old) / f"{ch}_{lv}_N{N}.npz"
            if not po.exists(): continue
            y, on, Mo = load(po); q = [i for i, n in enumerate(on) if n.startswith("clf_Q_shots")]; w = [i for i, n in enumerate(on) if n.startswith("clf_W_shots")]
            key = f"{ch}_{lv}_N{N}"; res["per_seed"].setdefault("paper_W36", {})[key] = [100 * float(Mo[i].mean()) for i in w]; res["per_seed"].setdefault("Q", {})[key] = [100 * float(Mo[i].mean()) for i in q]; res[key] = {"paper_W36_5seeds": pdiff(Mo[q], Mo[w], y)}
            for v in a.variants.split(","):
                pn = Path(a.new) / f"{v}_{ch}_{lv}_N{N}.npz"
                if not pn.exists(): continue
                yn, nn, Mn = load(pn); assert np.array_equal(y, yn)
                sel = [i for i, n in enumerate(nn) if a.seeds is None or n.rsplit("_s", 1)[1] in a.seeds.split(",")]
                res[key][v] = pdiff(Mo[q], Mn[sel], y); res[key][v]["W_seeds"] = [nn[i] for i in sel]
                sd = [nn[i].rsplit("_s", 1)[1] for i in sel]; wi = [i for i in w if on[i].rsplit("_s", 1)[1] in sd]
                res["per_seed"].setdefault(v, {})[key] = [100 * float(Mn[i].mean()) for i in sel]; res[key][v + "_vs_paperW_sameseeds"] = pdiff(Mo[q], Mo[wi], y)   # stare W na tych samych ziarnach (kontrola)
                res[key][v + "_Wnew_minus_Wold_pp"] = 100 * float(Mn[sel].mean() - Mo[wi].mean())
    json.dump(res, open(a.out, "w"), indent=1)
    for k, r in res.items():
        if k in ("_note", "per_seed"): continue
        print(k, " | ".join(f"{n}: {x['diff']:.1f} (W {x['accW']:.1f}, Q {x['accQ']:.1f}) [{x['lo']:.1f},{x['hi']:.1f}]" for n, x in r.items() if isinstance(x, dict) and "diff" in x))
