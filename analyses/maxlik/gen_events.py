"""Zliczenia homodynowe (K=36 x GRID pikseli, int) dla komorek reguly, odtworzone bitowo jak w ewaluacji sparowanej.
  tryb ev : jak qmi.eval_shots.gen (SEED=424242; na kazdy stan najpierw heterodyna, potem homodyna z tego samego generatora)
            -> te same zdarzenia, z ktorych powstaja obrazy W(FBP) w runs_eq/pary (para z Q i W).
  tryb sw : jak qmi.eq.pary_sw.records (SEED=7121; sinogram_counts) -> te same zdarzenia co sieci S w runs_eq/pary_sw.
Straznik (kazdy plik): FBP(zliczenia) == obraz W z oryginalnej funkcji (ev: qmi.eval_shots.gen; sw: pary_sw.records),
wzgledna roznica <= 1e-5; inaczej stop. Wynik: <out>/<tryb>/<kanal>_<poziom>_N<N>.npz (H uint16 [stan,K,GRID], y, rho_ok);
plik na komorke = punkt kontrolny.
  QMI_XMAX=12 PYTHONPATH=. python analyses/maxlik/gen_events.py --test data_eq/qmi/test/sweep --out analyses/maxlik/ev_base"""
import argparse, os, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from qmi.shots import heterodyne_image, radon, split_events, fbp

K = 36; TH = np.linspace(0, 180, K, endpoint=False)
CELLS = [("loss", 0), ("loss", 50), ("loss", 80), ("dephase", 20)]


def counts_hom(W, N, rng):
    """kopia petli z qmi.shots.homodyne_wigner / sinogram_counts, ale zwraca tez calkowite zliczenia"""
    sino = radon(np.asarray(W, np.float64), TH); out = np.zeros_like(sino); C = np.zeros(sino.shape, np.int64); per = split_events(N, K)
    for k in range(K):
        p = np.clip(sino[k], 0, None); s = p.sum()
        if per[k] == 0 or s <= 0: continue
        C[k] = rng.multinomial(per[k], p / s); out[k] = C[k] / per[k] * sino[k].sum()
    return C, out


def one(args):
    f, N, mode = args; d = np.load(f, allow_pickle=True); Hs = []; err = 0.0
    if mode == "ev":
        from qmi.eval_shots import gen, SEED
        rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(Path(f).name.encode()), N]))
        for W, Q in zip(d["W"], d["Q"]):
            heterodyne_image(Q, N, rng); C, out = counts_hom(W, N, rng); Hs.append(C)
        _, _, _, Xw, _, y = gen((f, N, K, "img"))
        mine = np.stack([fbp(c / np.maximum(split_events(N, K), 1)[:, None] * radon(np.asarray(W, np.float64), TH).sum(1)[:, None], TH, W.shape[0])
                         for c, W in zip(Hs, d["W"])])
    else:
        from qmi.eq.pary_sw import records, SEED
        rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(Path(f).name.encode()), N]))
        for W in d["W"]:
            C, out = counts_hom(W, N, rng); Hs.append(C)
        _, Xw, y = records(f, N)
        mine = np.stack([fbp(c / np.maximum(split_events(N, K), 1)[:, None] * radon(np.asarray(W, np.float64), TH).sum(1)[:, None], TH, W.shape[0])
                         for c, W in zip(Hs, d["W"])])
    err = float(np.abs(mine - Xw).max() / np.abs(Xw).max())
    H = np.stack(Hs); assert H.max() < 65535 and (H.sum((1, 2)) == N).all(), "suma zliczen != N"
    return f, N, H.astype(np.uint16), np.asarray(y), err


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="ev,sw"); ap.add_argument("--ns", default="100,1000"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cells", default="all"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if os.environ.get("QMI_XMAX") != "12": sys.exit("BLAD: QMI_XMAX=12")
    cells = CELLS if a.cells == "all" else [c for c in CELLS if f"{c[0]}_{c[1]}" in a.cells.split(",")]
    t0 = time.time()
    for mode in a.modes.split(","):
        od = Path(a.out) / mode; od.mkdir(parents=True, exist_ok=True)
        for ch, lv in cells:
            files = sorted(Path(a.test).glob(f"part_{ch}_{lv:03d}_*.npz"))
            if len(files) != 7: sys.exit(f"BLAD: {ch} {lv}: {len(files)} plikow")
            for N in map(int, a.ns.split(",")):
                dst = od / f"{ch}_{lv}_N{N}.npz"
                if dst.exists(): print("jest", dst, flush=True); continue
                with Pool(a.workers) as pool: res = pool.map(one, [(str(f), N, mode) for f in files])
                res.sort(key=lambda r: r[0]); errs = [r[4] for r in res]
                if max(errs) > 1e-5: sys.exit(f"BLAD straznika {mode} {ch} {lv} N={N}: FBP(zliczenia) != oryginal, {max(errs):.2e}")
                H = np.concatenate([r[2] for r in res]); y = np.concatenate([r[3] for r in res])
                np.savez(str(dst) + ".tmp.npz", H=H, y=y, files=np.array([Path(r[0]).name for r in res])); os.replace(str(dst) + ".tmp.npz", dst)
                print(f"{mode} {dst.name} stany={len(y)} straznik max {max(errs):.1e}  {time.time()-t0:.0f}s", flush=True)
    print("GEN_EVENTS_DONE", flush=True)
