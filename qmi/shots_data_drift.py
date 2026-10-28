"""Zbiory z szumem zliczen I dryfem fazy LO (do treningu sieci odpornych na dryf). Jak qmi.shots_data, ale kazda probka dostaje
losowy model dryfu: z prawdop. 0.25 brak, inaczej (walk|jitter) x sigma in {3,10,30} stopni. N=0 (N->inf) bez dryfu.
  python -m qmi.shots_data_drift --src data/qmi/train/mixed --dst data/qmi/train/shots_drift --copies 2 --workers 22"""
import argparse, os, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from qmi.shots import homodyne_wigner_exact, heterodyne_image, homodyne_wigner
from qmi.drift import make

NS = (100, 1000, 10000, 100000, 1000000, 0); K = 36; GAUSS = 3
_save = np.savez_compressed if os.environ.get("QMI_NPZ_COMPRESS") == "1" else np.savez   # kompresja zbiorow _v2 (miejsce na dysku)
CFG = [("walk", 3), ("walk", 10), ("walk", 30), ("jitter", 3), ("jitter", 10), ("jitter", 30)]


def one(args):
    src, dst, seed, copies, K = args; dst = Path(dst)
    if dst.exists(): return "skip"
    d = np.load(src, allow_pickle=True)
    if int(d["channel"][0]) == GAUSS: return "gauss-pominiety"
    rng = np.random.default_rng(np.random.SeedSequence([seed, 77, zlib.crc32(Path(src).name.encode())]))
    W, Q = d["W"], d["Q"]; n = len(W); Ws, Qs, Nn, idx = [], [], [], []
    for _ in range(copies):
        for i in range(n):
            N = int(rng.choice(NS))
            if N == 0: Qs.append(Q[i]); Ws.append(homodyne_wigner_exact(W[i], K))
            elif rng.random() < 0.25: Qs.append(heterodyne_image(Q[i], N, rng)); Ws.append(homodyne_wigner(W[i], N, rng, K))
            else:
                mode, sig = CFG[int(rng.integers(len(CFG)))]; w, q = make(W[i], Q[i], N, rng, mode, float(sig), K); Ws.append(w); Qs.append(q)
            Nn.append(N); idx.append(i)
    idx = np.array(idx); tmp = dst.with_name(dst.name + ".tmp.npz")
    _save(tmp, W=np.stack(Ws), Q=np.stack(Qs), N=np.array(Nn, np.int64), label=d["label"][idx], channel=d["channel"][idx], level=d["level"][idx])
    tmp.replace(dst); return "done"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--copies", type=int, default=2); ap.add_argument("--seed", type=int, default=31337); ap.add_argument("--workers", type=int, default=22); a = ap.parse_args()
    Path(a.dst).mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(Path(a.dst) / f.name), a.seed, a.copies, K) for f in sorted(Path(a.src).glob("part_*.npz"))]
    print(len(jobs), "plikow zrodlowych", flush=True); done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            f.result(); done += 1
            if done % 10 == 0 or done == len(jobs): print(f"{done}/{len(jobs)}", flush=True)
    print("SHOTS_DATA_DONE", flush=True)
