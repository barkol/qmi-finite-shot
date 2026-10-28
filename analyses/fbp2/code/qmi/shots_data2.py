"""Rekordy treningowe/walidacyjne W z nowym FBP (rzad 2 + Kak). Ten sam protokol co qmi.shots_data (seed 31337, copies 2, NS, K=36),
ten sam strumien N i Q co w pracy. Zapisuje tylko W (Q nie jest potrzebne). Wznawialne: gotowe pliki sa pomijane.
  python -m qmi.shots_data2 --src data_eq/qmi/train/mixed --dst fbp2/data/train --workers 20"""
import argparse, os, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from qmi.shots import heterodyne_image
from qmi.shots2 import homodyne_new, exact_o, VARIANTS
NS = (100, 1000, 10000, 100000, 1000000, 0); GAUSS = 3

def one(args):
    src, dst, seed, copies, K, var = args; dst = Path(dst)
    if dst.exists(): return "skip"
    d = np.load(src, allow_pickle=True)
    if int(d["channel"][0]) == GAUSS: return "gauss-pominiety"
    nm = zlib.crc32(Path(src).name.encode())
    rng = np.random.default_rng(np.random.SeedSequence([seed, nm])); rng2 = np.random.default_rng(np.random.SeedSequence([seed, nm, 99]))
    W, Q = d["W"], d["Q"]; n = len(W); Ws, Nn, idx = [], [], []
    for _ in range(copies):
        for i in range(n):
            N = int(rng.choice(NS))
            if N != 0: heterodyne_image(Q[i], N, rng)
            Ws.append(exact_o(W[i], K, *VARIANTS[var]) if N == 0 else homodyne_new(W[i], N, rng, rng2, K, var))
            Nn.append(N); idx.append(i)
    idx = np.array(idx); tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez_compressed(tmp, W=np.stack(Ws), N=np.array(Nn, np.int64), label=d["label"][idx], channel=d["channel"][idx], level=d["level"][idx])
    tmp.replace(dst); return "done"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--copies", type=int, default=2); ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--workers", type=int, default=20); ap.add_argument("--K", type=int, default=36); ap.add_argument("--variant", default="kak2"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    Path(a.dst).mkdir(parents=True, exist_ok=True)
    fs = sorted(Path(a.src).glob("part_*.npz")); fs = fs[:a.limit] if a.limit else fs
    jobs = [(str(f), str(Path(a.dst) / f.name), a.seed, a.copies, a.K, a.variant) for f in fs]; print(len(jobs), "plikow", flush=True); done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            f.result(); done += 1
            if done % 20 == 0 or done == len(jobs): print(f"{done}/{len(jobs)}", flush=True)
    print("SHOTS_DATA2_DONE", flush=True)
