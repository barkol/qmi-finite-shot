"""Klasyfikacja obrazow W_ML (z rho MaxLik) sieciami; wynik: <out>/<tryb>/<komorka>.npz (y, names, correct[model, stan], W_stats).
  QMI_XMAX=12 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python analyses/maxlik/classify.py --ml analyses/maxlik/ml_base --runs analyses/maxlik/runs \
      --match 're:^clf_W_(exact|shots)_s\d+$' --out analyses/maxlik/cls_base_ref"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import wimg, predict_all, load_models

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ml", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--match", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--nmin", type=int, default=3); ap.add_argument("--glob", default="*/*.npz"); a = ap.parse_args()
    models = load_models(a.runs, a.match); names = sorted(models); print("modele:", names, flush=True)
    if len(names) < a.nmin: sys.exit(f"BLAD: {len(names)} modeli")
    D = None; Bt = None
    for f in sorted(Path(a.ml).glob(a.glob)):
        d = np.load(f); dst = Path(a.out) / f.relative_to(a.ml)
        if dst.exists(): continue
        if D != int(d["D"]): D = int(d["D"]); Bt = torch.as_tensor(np.load(Path(__file__).parent / f"wbasis_D{D}.npy"), device="cuda")
        X = wimg(d["rho"], Bt); y = d["y"]; p = predict_all(models, X)
        M = np.stack([p[n] == y for n in names]); dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(dst) + ".tmp.npz", y=y, names=np.array(names), correct=M, pred=np.stack([p[n] for n in names]))
        os.replace(str(dst) + ".tmp.npz", dst); print(dst, " ".join(f"{n}:{M[i].mean():.3f}" for i, n in enumerate(names)), flush=True)
    print("CLASSIFY_DONE", flush=True)
