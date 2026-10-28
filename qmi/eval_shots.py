"""Ewaluacja modeli na obrazach z szumem zliczen generowanych w locie (test sweep: straty i dephasing).
Jedno przejscie po obrazach, wszystkie modele naraz.   python -m qmi.eval_shots --runs runs --test data/qmi/test/sweep --out runs/eval_shots.json"""
import argparse, json, re, zlib, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
from qsdl.cnn.classifier import StateClassifierCNNv2
from qmi.clf import wilson, DEV
from qmi.shots import heterodyne_image, homodyne_wigner, homodyne_wigner_exact, sinogram_counts

NS = (100, 1000, 10000, 100000, 1000000, 0); K = 36; SEED = 424242
CHN = {"loss": 1, "dephase": 2}


def gen(args):
    f, N, K, mode = args; d = np.load(f, allow_pickle=True)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(Path(f).name.encode()), N]))
    Xw, Xq = [], []
    for W, Q in zip(d["W"], d["Q"]):
        if mode == "sino": Xw.append(sinogram_counts(W, N, rng, K)); continue
        if N == 0: Xq.append(Q); Xw.append(homodyne_wigner_exact(W, K))
        else: Xq.append(heterodyne_image(Q, N, rng)); Xw.append(homodyne_wigner(W, N, rng, K))
    return int(d["channel"][0]), int(d["level"][0]), N, np.stack(Xw), (np.stack(Xq) if Xq else None), d["label"]


def load_models(runs, match="clf_"):
    ms = {}
    for p in sorted(Path(runs).glob("clf_*/best.pt")):
        nm = p.parent.name
        if (re.search(match[3:], nm) is None) if match.startswith("re:") else (match not in nm): continue   # "re:<regex>" albo podciag
        s = torch.load(p, map_location=DEV, weights_only=False); m = StateClassifierCNNv2().to(DEV)
        m.load_state_dict(s["model"]); m.eval(); ms[p.parent.name] = (m, s["rep"], s["scale"])
    return ms


@torch.no_grad()
def pred(m, X, scale):
    return m(torch.from_numpy(X / scale).unsqueeze(1).to(DEV)).argmax(1).cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=24); ap.add_argument("--K", type=int, default=36); ap.add_argument("--match", default="clf_"); ap.add_argument("--mode", default="img", choices=("img", "sino")); a = ap.parse_args()
    torch.set_num_threads(2); models = load_models(a.runs, a.match); print("modele:", list(models), flush=True)
    files = [f for f in sorted(Path(a.test).glob("part_*.npz")) if f.name.split("_")[1] in CHN]
    tasks = [(str(f), N, a.K, a.mode) for f in files for N in NS]; print(len(tasks), "zadan", flush=True)
    acc = {}; t0 = time.time(); inv = {v: k for k, v in CHN.items()}
    with Pool(a.workers) as pool:
        for i, (ch, lv, N, Xw, Xq, y) in enumerate(pool.imap_unordered(gen, tasks, chunksize=1)):
            for name, (m, rep, sc) in models.items():
                p = pred(m, Xw if rep == "W" else Xq, sc); key = f"{name}|{inv[ch]}|{lv}|{N}"
                r = acc.setdefault(key, [0, 0, np.zeros((7, 7), int)]); r[0] += int((p == y).sum()); r[1] += len(y); np.add.at(r[2], (y, p), 1)
            if i % 50 == 0: print(f"{i}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    out = {}
    for key, (k, n, cm) in acc.items():
        name, ch, lv, N = key.split("|"); lo, hi = wilson(k, n)
        out.setdefault(name, {}).setdefault(ch, {}).setdefault(N, {})[lv] = dict(acc=k / n, lo=lo, hi=hi, n=n, cm=cm.tolist())
    json.dump(out, open(a.out, "w")); print("EVAL_SHOTS_DONE", flush=True)
