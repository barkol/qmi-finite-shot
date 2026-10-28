"""Zbiory sinogramow (homodyna bez rekonstrukcji): jak shots_data, ale wejscie = histogramy K projekcji. Zapisuje klucz W (K x n)."""
import argparse, os, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from qmi.shots import sinogram_counts

NS = (100, 1000, 10000, 100000, 1000000, 0); GAUSS = 3
_save = np.savez_compressed if os.environ.get("QMI_NPZ_COMPRESS") == "1" else np.savez   # kompresja zbiorow _v2 (miejsce na dysku)

def one(args):
    src, dst, seed, copies, K = args; dst = Path(dst)
    if dst.exists(): return "skip"
    d = np.load(src, allow_pickle=True)
    if int(d["channel"][0]) == GAUSS: return "gauss-pominiety"
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(Path(src).name.encode())]))
    W = d["W"]; n = len(W); X, Nn, idx = [], [], []
    for _ in range(copies):
        for i in range(n):
            N = int(rng.choice(NS)); X.append(sinogram_counts(W[i], N, rng, K)); Nn.append(N); idx.append(i)
    idx = np.array(idx); tmp = dst.with_name(dst.name + ".tmp.npz")
    _save(tmp, W=np.stack(X), N=np.array(Nn, np.int64), label=d["label"][idx], channel=d["channel"][idx], level=d["level"][idx])
    tmp.replace(dst); return "done"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--copies", type=int, default=2); ap.add_argument("--seed", type=int, default=5150); ap.add_argument("--K", type=int, default=36); ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(); Path(a.dst).mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(Path(a.dst) / f.name), a.seed, a.copies, a.K) for f in sorted(Path(a.src).glob("part_*.npz"))]
    print(len(jobs), "plikow", flush=True); done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            f.result(); done += 1
            if done % 10 == 0 or done == len(jobs): print(f"{done}/{len(jobs)}", flush=True)
    print("SINO_DATA_DONE", flush=True)
