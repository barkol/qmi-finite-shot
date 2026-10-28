"""Wplyw dryfu fazy LO na dokladnosc sieci wytrenowanych na rekordach BEZ dryfu (odpornosc na niemodelowany efekt).
  python -m qmi.eval_drift --runs runs --test data/qmi/test/sweep --out runs/eval_drift.json
Wznawialne: czesciowe wyniki co 40 zadan w <out>.partial.json (klucze zadan), restart pomija wykonane."""
import argparse, json, zlib, time, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
from qmi.clf import wilson, DEV
from qmi.eval_shots import load_models, pred
from qmi.drift import make

SEED = 515151; K = 36
LV = {"loss": (0, 20, 50), "dephase": (20, 50)}
CH = {"loss": 1, "dephase": 2}
NS = (1000, 10000, 100000)
CFG = [("walk", 0), ("walk", 3), ("walk", 10), ("walk", 30), ("jitter", 3), ("jitter", 10), ("jitter", 30)]


def gen(args):
    f, N, mode, sigma = args; d = np.load(f, allow_pickle=True)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(Path(f).name.encode()), N, 0 if mode == "walk" else 1, sigma]))
    Xw, Xq = [], []
    for W, Q in zip(d["W"], d["Q"]):
        w, q = make(W, Q, N, rng, mode, float(sigma), K); Xw.append(w); Xq.append(q)
    return args, int(d["channel"][0]), int(d["level"][0]), np.stack(Xw), np.stack(Xq), d["label"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=24); ap.add_argument("--match", default="_shots_s"); a = ap.parse_args()
    torch.set_num_threads(2); models = load_models(a.runs, a.match); print("modele:", list(models), flush=True)
    files = []
    for f in sorted(Path(a.test).glob("part_*.npz")):
        p = f.name.split("_")
        if p[1] in LV and int(p[2]) in LV[p[1]]: files.append(f)
    part = Path(a.out + ".partial.json"); st = json.load(open(part)) if part.exists() else {"done": [], "acc": {}}
    done = set(st["done"]); acc = {k: [v[0], v[1], np.array(v[2])] for k, v in st["acc"].items()}
    tasks = [(str(f), N, m, s) for f in files for N in NS for (m, s) in CFG if "|".join(map(str, (f.name, N, m, s))) not in done]
    print(len(files), "plikow;", len(tasks), "zadan do wykonania (wznowione:", len(done), ")", flush=True)
    inv = {v: k for k, v in CH.items()}; t0 = time.time()
    def save():
        json.dump({"done": sorted(done), "acc": {k: [v[0], v[1], v[2].tolist()] for k, v in acc.items()}}, open(str(part) + ".tmp", "w")); os.replace(str(part) + ".tmp", part)
    with Pool(a.workers) as pool:
        for i, (args, ch, lv, Xw, Xq, y) in enumerate(pool.imap_unordered(gen, tasks, chunksize=1)):
            f, N, mode, sigma = args
            for name, (m, rep, sc) in models.items():
                p = pred(m, Xw if rep == "W" else Xq, sc); key = f"{name}|{inv[ch]}|{lv}|{mode}|{sigma}|{N}"
                r = acc.setdefault(key, [0, 0, np.zeros((7, 7), int)]); r[0] += int((p == y).sum()); r[1] += len(y); np.add.at(r[2], (y, p), 1)
            done.add("|".join(map(str, (Path(f).name, N, mode, sigma))))
            if i % 40 == 0: save(); print(f"{i}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    save(); out = {}
    for key, (k, n, cm) in acc.items():
        name, ch, lv, mode, sigma, N = key.split("|"); lo, hi = wilson(k, n)
        out.setdefault(name, {}).setdefault(ch, {}).setdefault(lv, {}).setdefault(f"{mode}{sigma}", {})[N] = dict(acc=k / n, lo=lo, hi=hi, n=n, cm=cm.tolist())
    json.dump(out, open(a.out, "w")); print("EVAL_DRIFT_DONE", flush=True)
