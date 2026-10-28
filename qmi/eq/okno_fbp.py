"""Kontrola okna i bledu FBP na stanach bez szumu zbioru testowego:
masa Q w oknie (min, ulamek < 0.99) i wzgledny blad L2 obrazu FBP z dokladnych rzutow (K=36) wzgledem dokladnego W.
   QMI_XMAX=12 python -m qmi.eq.okno_fbp --test data_eq/qmi/test/sweep --out runs_eq/okno_fbp.json"""
import argparse, glob, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from qsdl.config import XMAX, GRID
from qmi.shots import homodyne_wigner_exact

def one(f):
    d = np.load(f, allow_pickle=True); A = (2 * XMAX / (GRID - 1)) ** 2; m, e = [], []
    for W, Q in zip(d["W"], d["Q"]):
        m.append(float(np.clip(Q, 0, None).sum() * A))
        R = homodyne_wigner_exact(W, 36); e.append(float(np.linalg.norm(R - W) / np.linalg.norm(W)))
    return m, e

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); fs = sorted(glob.glob(f"{a.test}/part_loss_000_*.npz"))
    if not fs: raise SystemExit("brak plikow part_loss_000_*")
    with ProcessPoolExecutor(a.workers) as ex: res = list(ex.map(one, fs))
    m = np.concatenate([r[0] for r in res]); e = np.concatenate([r[1] for r in res])
    o = dict(n=int(len(m)), xmax=XMAX, qmass_min=float(m.min()), qmass_frac_below_099=float((m < 0.99).mean()),
             fbp_err_med=float(np.median(e)), fbp_err_p90=float(np.percentile(e, 90)))
    json.dump(o, open(a.out, "w"), indent=1); print(o); print("OKNO_FBP_DONE")
