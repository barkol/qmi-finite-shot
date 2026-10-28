"""Sparowana ewaluacja sieci W z nowym FBP na TYCH SAMYCH rekordach Q co w pracy. Straznik: trafnosc stanowa sieci Q (runs_eq) na zregenerowanych
rekordach Q musi byc identyczna z runs_eq/pary/*.npz (dowod, ze strumien rng/N/Q jest bitowo ten sam). Wynik: results/pary2/<wariant>_<kanal>_<lv>_N<N>.npz,
punkt kontrolny na komorke (wznowienie pomija gotowe).
  QMI_XMAX=12 python -m qmi.pary2 --variant kak2 --test <data_eq/qmi/test/sweep> --runsnew ../runs --runsold <runs_eq> --out ../results/pary2 --N 100 --cells loss:0,dephase:20"""
import argparse, os, sys, time, zlib
from pathlib import Path
import numpy as np
from qmi.shots import heterodyne_image
from qmi.shots2 import homodyne_new
from qmi.eval_shots import load_models, pred, SEED

def gen2(f, N, variant):
    d = np.load(f, allow_pickle=True); nm = zlib.crc32(Path(f).name.encode())
    rng = np.random.default_rng(np.random.SeedSequence([SEED, nm, N])); rng2 = np.random.default_rng(np.random.SeedSequence([SEED, nm, N, 99]))
    Xw, Xq = [], []
    for W, Q in zip(d["W"], d["Q"]):
        Xq.append(heterodyne_image(Q, N, rng)); Xw.append(homodyne_new(W, N, rng, rng2, 36, variant))
    return np.stack(Xw), np.stack(Xq), d["label"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--runsnew", required=True); ap.add_argument("--runsold", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--N", default="100,1000,10000"); ap.add_argument("--cells", default="loss:0,loss:50,loss:80,dephase:20")
    ap.add_argument("--match", default=None); a = ap.parse_args()
    assert os.environ.get("QMI_XMAX") == "12"
    import torch; torch.set_num_threads(4)
    wn = load_models(a.runsnew, a.match or f"re:^clf_W_{a.variant}_s\\d+$"); qn = load_models(a.runsold, "re:^clf_Q_shots_s\\d+$")
    names = sorted(wn); qnames = sorted(qn); print("nowe W:", names, "Q:", qnames, flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for cell in a.cells.split(","):
        ch, lv = cell.split(":"); lv = int(lv)
        files = sorted(Path(a.test).glob(f"part_{ch}_{lv:03d}_*.npz"))
        if len(files) != 7: sys.exit(f"BLAD {ch} {lv}: {len(files)} plikow")
        for N in map(int, a.N.split(",")):
            dst = out / f"{a.variant}_{ch}_{lv}_N{N}.npz"
            if dst.exists() and set(np.load(dst)["names"]) >= set(names): print("jest", dst.name, flush=True); continue
            ys, CW, CQ = [], {n: [] for n in names}, {n: [] for n in qnames}
            for f in files:
                Xw, Xq, y = gen2(str(f), N, a.variant); ys.append(y)
                for n in names: m, rep, sc = wn[n]; CW[n].append(pred(m, Xw, sc) == y)
                for n in qnames: m, rep, sc = qn[n]; CQ[n].append(pred(m, Xq, sc) == y)
            y = np.concatenate(ys); old = np.load(Path(a.runsold) / "pary" / f"{ch}_{lv}_N{N}.npz"); on = list(old["names"])
            assert np.array_equal(old["y"], y)
            for n in qnames:
                if not np.array_equal(old["correct"][on.index(n)].astype(bool), np.concatenate(CQ[n])): sys.exit(f"BLAD: rekordy Q rozne od pracy: {n} {ch} {lv} N={N}")
            np.savez(str(dst) + ".tmp.npz", y=y, names=np.array(names), correct=np.stack([np.concatenate(CW[n]) for n in names]))
            os.replace(str(dst) + ".tmp.npz", dst); print(dst.name, "Q zgodne z praca;", f"{time.time()-t0:.0f}s", flush=True)
    print("PARY2_DONE", flush=True)
