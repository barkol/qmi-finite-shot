"""MaxLik dla plikow zdarzen (wynik gen_events.py / gen_train_events.py) na GPU.
  test : <ev>/<tryb>/<komorka>.npz -> <out>/<tryb>/<komorka>.npz (rho complex64, ll, nan_fix)
  train: <trev>/<split>/part_*.npz -> <out>/<split>/part_*.npz w formacie qmi.data.load_dir (W = obraz W_ML, N=0 -> W dokladne)
Stany z NaN w fp32 liczone ponownie w complex128; jesli nadal NaN -> stop. Plik = punkt kontrolny; log do pliku.
  QMI_XMAX=12 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python analyses/maxlik/run_ml.py test --inp analyses/maxlik/ev_base --out analyses/maxlik/ml_base"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlcore import povm, ml_torch, GRID
from common import wimg

LOG = None
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(map(str, a)); print(s, flush=True)
    if LOG: open(LOG, "a").write(s + "\n")


def ml_safe(H, Pi, iters, bs=700):
    rhos, lls, nfix = [], [], 0
    for i in range(0, len(H), bs):
        r, ll, _ = ml_torch(H[i:i + bs], Pi, iters)
        bad = ~np.isfinite(r.reshape(len(r), -1)).all(1)
        if bad.any():
            r2, ll2, _ = ml_torch(H[i:i + bs][bad], Pi, iters, dtype="c128"); r[bad] = r2; ll[bad] = ll2; nfix += int(bad.sum())
            if not np.isfinite(r2).all(): sys.exit(f"BLAD: NaN takze w complex128 ({int(bad.sum())} stanow)")
        rhos.append(r.astype(np.complex64)); lls.append(ll)
    return np.concatenate(rhos), np.concatenate(lls), nfix


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("test", "train")); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--D", type=int, default=40); ap.add_argument("--iters", type=int, default=1000); ap.add_argument("--glob", default="*/*.npz"); a = ap.parse_args()
    if os.environ.get("QMI_XMAX") != "12": sys.exit("BLAD: QMI_XMAX=12")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); LOG = str(out / "run_ml.log")
    Pi = povm(a.D); t0 = time.time(); Bt = None
    files = sorted(Path(a.inp).glob(a.glob)); log(a.cmd, len(files), "plikow, D", a.D, "iters", a.iters)
    if a.cmd == "train": Bt = torch.as_tensor(np.load(Path(__file__).parent / f"wbasis_D{a.D}.npy"), device="cuda")
    for f in files:
        dst = out / f.relative_to(a.inp)
        if dst.exists(): continue
        dst.parent.mkdir(parents=True, exist_ok=True); d = np.load(f)
        H = d["H"].reshape(len(d["H"]), -1).astype(np.float32)
        if a.cmd == "test":
            rho, ll, nfix = ml_safe(H, Pi, a.iters)
            np.savez(str(dst) + ".tmp.npz", rho=rho, ll=ll, y=d["y"], nan_fix=nfix, D=a.D, iters=a.iters)
        else:
            ex = np.zeros(len(H), bool); ex[d["iex"]] = True; Wm = np.zeros((len(H), GRID, GRID), np.float32); nfix = 0
            if (~ex).any():
                rho, ll, nfix = ml_safe(H[~ex], Pi, a.iters); Wm[~ex] = wimg(rho, Bt)
            if ex.any(): Wm[ex] = d["Wex"]
            np.savez(str(dst) + ".tmp.npz", W=Wm, label=d["label"], channel=d["channel"], level=d["level"], N=d["N"])
        os.replace(str(dst) + ".tmp.npz", dst); log(f"{dst.relative_to(out)} n={len(H)} nan_fix={nfix} {time.time()-t0:.0f}s")
    log("RUN_ML_DONE")
