"""Kontrola energetyczna: ile klasy da sie odgadnac z samej oszacowanej sredniej liczby fotonow.

Z rekordu N zdarzen liczymy jedna liczbe, estymator <n>:
  heterodyna  n_het = mean(|beta|^2) - 1,          beta = (x + i p)/sqrt(2) z pikseli obrazu Q (konwencja qutip, g = sqrt 2)
  homodyna    n_hom = mean_theta mean(x_theta^2) - 1/2,  zdarzenia z K = 36 rzutow Radona obrazu W (jak sinogram_counts)
N = 0 oznacza wartosc dokladna z obrazu. Klasyfikator: k-NN (k = 50) na tej jednej cesze, trenowany na zbiorze treningowym
przy tym samym N, oceniany na zbiorze testowym sweep po kanalach i poziomach. W zbiorze eqn powinien dawac 1/7.

Cechy kazdego pliku zapisywane osobno w <out>/feat/ (wznawialne), wynik w <out>/energy_control.json.
  python -m qmi.energy_control --train data_eq/qmi/train/mixed --test data_eq/qmi/test/sweep --out runs_eq/energy_control
"""
import argparse, json, os, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from qsdl.config import XMAX, GRID
from qmi.shots import radon, split_events
from qmi.data import wilson, CH_NAMES

NS = (0, 100, 1000, 10000)
K = 36


def feats(W, Q, rng):
    x = np.linspace(-XMAX, XMAX, GRID); X, P = np.meshgrid(x, x, indexing="xy"); r2 = ((X ** 2 + P ** 2) / 2).ravel()
    q = np.clip(Q.astype(np.float64), 0, None).ravel(); q /= q.sum()
    th = np.linspace(0, 180, K, endpoint=False); sino = np.clip(radon(W.astype(np.float64), th), 0, None)
    sino /= sino.sum(1, keepdims=True); x2 = x ** 2
    out = []
    for N in NS:
        if N == 0:
            het = q @ r2 - 1; hom = float((sino @ x2).mean()) - 0.5
        else:
            het = rng.multinomial(N, q) @ r2 / N - 1
            per = split_events(N, K); hom = float(sum(rng.multinomial(per[k], sino[k]) @ x2 for k in range(K)) / N) - 0.5
        out.append((het, hom))
    return np.array(out, np.float32)                      # (len(NS), 2)


def run_file(job):
    f, dst, seed = job
    if os.path.exists(dst): return dst
    d = np.load(f, allow_pickle=True); rng = np.random.default_rng([seed, zlib.crc32(Path(f).name.encode())])
    F = np.stack([feats(w, q, rng) for w, q in zip(d["W"], d["Q"])])
    tmp = dst + ".tmp.npz"; np.savez(tmp, F=F, y=d["label"], channel=d["channel"], level=d["level"]); os.replace(tmp, dst)
    return dst


def load_feats(root, out, tag, workers, seed):
    fd = Path(out) / "feat" / tag; fd.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(fd / f.name), seed) for f in sorted(Path(root).glob("part_*.npz"))]
    if not jobs: raise FileNotFoundError(root)
    with ProcessPoolExecutor(workers) as ex: list(ex.map(run_file, jobs))
    z = [np.load(j[1]) for j in jobs]
    return {k: np.concatenate([a[k] for a in z]) for k in ("F", "y", "channel", "level")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=20); ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    tr = load_feats(a.train, a.out, "train", a.workers, a.seed); te = load_feats(a.test, a.out, "test", a.workers, a.seed + 1)
    res = {"chance": 1 / len(np.unique(tr["y"])), "n_train": int(len(tr["y"]))}
    for i, N in enumerate(NS):
        for j, det in enumerate(("het", "hom")):
            m = KNeighborsClassifier(50).fit(tr["F"][:, i, j:j + 1], tr["y"]); p = m.predict(te["F"][:, i, j:j + 1])
            r = res[f"{det}_N{N}"] = {}
            for c in sorted(set(te["channel"].tolist())):
                r[CH_NAMES[int(c)]] = {}
                for L in sorted(set(te["level"][te["channel"] == c].tolist())):
                    mk = (te["channel"] == c) & (te["level"] == L); k = int((p[mk] == te["y"][mk]).sum()); n = int(mk.sum())
                    lo, hi = wilson(k, n); r[CH_NAMES[int(c)]][int(L)] = dict(acc=k / n, lo=lo, hi=hi, n=n)
            cl = r.get("loss", {}).get(0); print(f"{det} N={N}: clean acc {cl['acc']:.3f} [{cl['lo']:.3f}, {cl['hi']:.3f}]" if cl else det, flush=True)
    nb = {int(c): [float(te["F"][te["y"] == c, 0, 0].mean()), float(te["F"][te["y"] == c, 0, 0].std())] for c in np.unique(te["y"])}
    res["nbar_het_exact_by_class"] = nb
    json.dump(res, open(Path(a.out) / "energy_control.json", "w"), indent=1); print("ENERGY_CONTROL_DONE")
