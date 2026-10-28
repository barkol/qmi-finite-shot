"""Blad wzgledny L2 rekonstrukcji FBP z dokladnych rzutow (N=inf) wzgledem dokladnego W, K=36, stany bez szumu zbioru testowego (okno 12).
Warianty: stary (rzad 1, ramp), nowy (rzad 2, Kak), oraz ablacje (rzad 2 + ramp, rzad 1 + Kak).
  QMI_XMAX=12 python -m qmi.fbp_err --test data_eq/qmi/test/sweep --out results/fbp_err.json"""
import argparse, glob, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from qmi.shots2 import exact_o
V = {"old_o1_ramp": (1, "freq", None), "o2_ramp": (2, "freq", None), "o1_kak": (1, "kak", None), "new_o2_kak": (2, "kak", None), "o2_kak_hann": (2, "kak", "hann"), "o1_ramp_hann": (1, "freq", "hann")}
def one(f):
    d = np.load(f, allow_pickle=True); out = {k: [] for k in V}; fam = str(d["family"][0]) if "family" in d.files else f.split("_")[-1][:-4]
    for W in d["W"]:
        for k, v in V.items(): out[k].append(float(np.linalg.norm(exact_o(W, 36, *v) - W) / np.linalg.norm(W)))
    return fam, out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=20); a = ap.parse_args()
    fs = sorted(glob.glob(f"{a.test}/part_loss_000_*.npz"))
    with ProcessPoolExecutor(a.workers) as ex: res = list(ex.map(one, fs))
    o = {"n": sum(len(r[1]["old_o1_ramp"]) for r in res), "by_variant": {}, "by_family": {}}
    for k in V:
        e = np.concatenate([r[1][k] for r in res]); o["by_variant"][k] = dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)), mean=float(e.mean()))
    for fam, r in res: o["by_family"][fam] = {k: float(np.median(r[k])) for k in V}
    json.dump(o, open(a.out, "w"), indent=1); print(json.dumps(o, indent=1)); print("FBP_ERR_DONE")
