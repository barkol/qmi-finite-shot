"""Zliczenia homodynowe rekordow treningowych/walidacyjnych, te same zdarzenia co qmi.shots_data (copies=2, seed=31337, K=36):
na kazdy rekord losowane N z NS, heterodyna (zuzywa generator), potem homodyna. Dla N=0 zapisujemy dokladny obraz W
(granica N->inf dla MaxLik = sam stan; FBP w shots_data uzywa wtedy homodyne_wigner_exact).
Straznik: dla pierwszego pliku FBP(zliczenia) == W z qmi.shots_data.one (wynik do katalogu tymczasowego), wzgl. <= 1e-5.
Wynik: <dst>/<plik>.npz (H uint16 [rek,K,GRID], N, label, channel, level, Wex [rek z N=0], iex); plik = punkt kontrolny.
  QMI_XMAX=12 PYTHONPATH=. python analyses/maxlik/gen_train_events.py --src data_eq/qmi/train/mixed --dst analyses/maxlik/trev/train"""
import argparse, os, sys, tempfile, time, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from qmi.shots import heterodyne_image, radon, split_events, fbp
from qmi.shots_data import NS, GAUSS, one as sd_one

K = 36; TH = np.linspace(0, 180, K, endpoint=False)


def counts_hom(W, N, rng):
    sino = radon(np.asarray(W, np.float64), TH); C = np.zeros(sino.shape, np.int64); per = split_events(N, K)
    for k in range(K):
        p = np.clip(sino[k], 0, None); s = p.sum()
        if per[k] == 0 or s <= 0: continue
        C[k] = rng.multinomial(per[k], p / s)
    return C, sino.sum(1)


def work(args):
    src, dst, seed, copies = args
    if Path(dst).exists(): return "skip", 0.0
    d = np.load(src, allow_pickle=True)
    if int(d["channel"][0]) == GAUSS: return "gauss", 0.0
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(Path(src).name.encode())]))
    W, Q = d["W"], d["Q"]; n = len(W); Hs, Ns, idx, Wex, iex, M = [], [], [], [], [], []
    for _ in range(copies):
        for i in range(n):
            N = int(rng.choice(NS))
            if N == 0:
                Hs.append(np.zeros((K, W.shape[1]), np.int64)); Wex.append(W[i]); iex.append(len(Ns)); M.append(np.zeros(K))
            else:
                heterodyne_image(Q[i], N, rng); C, m = counts_hom(W[i], N, rng); Hs.append(C); M.append(m)
            Ns.append(N); idx.append(i)
    H = np.stack(Hs); assert H.max() < 65535
    idx = np.array(idx); tmp = str(dst) + ".tmp.npz"
    np.savez(tmp, H=H.astype(np.uint16), N=np.array(Ns, np.int64), label=d["label"][idx], channel=d["channel"][idx], level=d["level"][idx],
             Wex=np.array(Wex, np.float32).reshape(-1, W.shape[1], W.shape[2]), iex=np.array(iex, np.int64), mass=np.array(M))
    os.replace(tmp, dst); return "done", 0.0


def guard(src, seed, copies):
    with tempfile.TemporaryDirectory() as td:
        sd_one((src, str(Path(td) / "ref.npz"), seed, copies, K)); ref = np.load(Path(td) / "ref.npz")
        work((src, str(Path(td) / "me.npz"), seed, copies)); me = np.load(Path(td) / "me.npz")
        assert (ref["N"] == me["N"]).all() and (ref["label"] == me["label"]).all()
        err = 0.0; iex = set(me["iex"].tolist())
        for j in range(len(me["N"])):
            if j in iex: continue
            per = split_events(me["N"][j], K); w = fbp(me["H"][j] / np.maximum(per, 1)[:, None] * me["mass"][j][:, None], TH, ref["W"].shape[1])
            err = max(err, float(np.abs(w - ref["W"][j]).max() / np.abs(ref["W"][j]).max()))
        return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=31337); ap.add_argument("--copies", type=int, default=2); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if os.environ.get("QMI_XMAX") != "12": sys.exit("BLAD: QMI_XMAX=12")
    files = sorted(Path(a.src).glob("part_*.npz")); Path(a.dst).mkdir(parents=True, exist_ok=True)
    g = next(f for f in files if int(np.load(f)["channel"][0]) != GAUSS)
    e = guard(str(g), a.seed, a.copies); print(f"straznik {g.name}: FBP(zliczenia) vs shots_data {e:.1e}", flush=True)
    if e > 1e-5: sys.exit("BLAD straznika")
    t0 = time.time(); done = 0
    with Pool(a.workers) as pool:
        for r, _ in pool.imap_unordered(work, [(str(f), str(Path(a.dst) / f.name), a.seed, a.copies) for f in files]):
            done += 1
            if done % 50 == 0: print(f"{done}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    print("GEN_TRAIN_DONE", flush=True)
