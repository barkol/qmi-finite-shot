"""Klasyfikacja eksperymentalnych rekordow kwadraturowych GKP (Konno+ 2024, Dryad) sieciami W z K=6 katami.
   python -m qmi.exp_gkp --runs runs --match shotsK6_s --out runs/exp_gkp_K6.json [--offset -0.26] [--flip 0]"""
import argparse, json, numpy as np
from pathlib import Path
from qmi.shots import radon, fbp
from qmi.eval_shots import load_models, pred
from qsdl.config import GRID, XMAX   # z QMI_XMAX (bylo na sztywno 128, 5.0)
FILES = {0: 0, 30: 30, 60: 60, -30: 150, -60: 120, -90: 90}      # kat pliku -> kat w [0,180); ujemne: +180 i odwrocenie znaku
TH = sorted(FILES.values())
CLASSES = ("fock", "coherent", "vacuum", "thermal", "cat", "gkp", "binomial")

def load(d, offset, flip):
    rec = {}
    for a, th in FILES.items():
        x = np.load(Path(d) / f"quad_{a}deg.npy").astype(float) - offset
        if a < 0: x = -x
        if flip: x = -x
        rec[th] = x
    return rec

def mass():
    import qutip as qt
    y = np.linspace(-XMAX, XMAX, GRID); W = qt.wigner(qt.coherent_dm(40, 0.0), y, y)
    return float(radon(W, [0])[0].sum())

def image(rec, per, rng, M):
    edges = np.linspace(-XMAX - (XMAX / (GRID - 1)), XMAX + (XMAX / (GRID - 1)), GRID + 1)
    rows = []
    for th in TH:
        x = rec[th]; s = x if per is None or per >= len(x) else rng.choice(x, per, replace=False)
        c = np.histogram(s, bins=edges)[0].astype(float)
        rows.append(c / max(1, len(s)) * M)
    return fbp(np.stack(rows), TH, GRID)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/dryad_gkp"); ap.add_argument("--runs", default="runs")
    ap.add_argument("--match", default="shotsK6_s"); ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=float, default=-0.26); ap.add_argument("--flip", type=int, default=0); ap.add_argument("--draws", type=int, default=40)
    a = ap.parse_args(); rng = np.random.default_rng(7); M = mass()
    rec = load(a.data, a.offset, a.flip); models = load_models(a.runs, a.match); print("modele:", list(models), "M =", M, flush=True)
    res = {}
    for Ntot in (600, 6000, 60000, 120000):
        per = Ntot // 6; full = per >= min(len(v) for v in rec.values())
        imgs = np.stack([image(rec, None if full else per, rng, M) for _ in range(1 if full else a.draws)])
        for name, (m, rep, sc) in models.items():
            p = pred(m, imgs, sc); res.setdefault(name, {})[str(Ntot)] = np.bincount(p, minlength=7).tolist()
        print(Ntot, {n: dict(zip(CLASSES, r[str(Ntot)])) for n, r in res.items()}, flush=True)
    np.save(a.out.replace(".json", "_img.npy"), imgs[0]); json.dump(res, open(a.out, "w"))
