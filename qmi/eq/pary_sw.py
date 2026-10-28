"""Wplyw samej rekonstrukcji: sieci W (FBP) i S (histogramy homodynowe bez rekonstrukcji) na DOKLADNIE tych samych zdarzeniach
Dla kazdego stanu: zliczenia S = sinogram_counts(W, N, K=36), obraz W = FBP tych samych zliczen
(to samo co homodyne_wigner przy tym samym stanie generatora; N=0: dokladne rzuty). Roznica S-W zalezy wiec tylko od rekonstrukcji
(i od tego, jak siec czyta dany ksztalt wejscia), a nie od losowania zdarzen.
Komorki: tabela reguly (straty 0/50/80%, defazowanie 20%) x N = 1e2, 1e3, 1e4, 1e5 i N=0 (dokladne rzuty).
Wynik: <runs>/pary_sw/<kanal>_<poziom>_N<N>.npz (y, names, correct[model, stan]); plik na komorke = punkt kontrolny.
Straznik na starcie: FBP(sinogram_counts) == homodyne_wigner przy tym samym stanie generatora (bitowo do float32) i
FBP(dokladne rzuty) == homodyne_wigner_exact.
   QMI_XMAX=12 python -m qmi.eq.pary_sw --runs runs_eq --test data_eq/qmi/test/sweep"""
import argparse, copy, os, sys, time, zlib
from pathlib import Path
import numpy as np
from qmi.eval_shots import load_models, pred
from qmi.shots import sinogram_counts, fbp, homodyne_wigner, homodyne_wigner_exact

CELLS = [("loss", 0), ("loss", 50), ("loss", 80), ("dephase", 20)]; NS = (100, 1000, 10000, 100000, 0); K = 36; SEED = 7121
TH = np.linspace(0, 180, K, endpoint=False)


def records(f, N):
    d = np.load(f, allow_pickle=True)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(Path(f).name.encode()), N]))
    Xs, Xw = [], []
    for W in d["W"]:
        S = sinogram_counts(W, N, rng, K); Xs.append(S); Xw.append(fbp(S.astype(np.float64), TH, W.shape[0]))
    return np.stack(Xs), np.stack(Xw).astype(np.float32), d["label"]


def guard(f):
    W = np.load(f, allow_pickle=True)["W"][0]
    r1 = np.random.default_rng(5); r2 = copy.deepcopy(r1)
    a = fbp(sinogram_counts(W, 1000, r1, K).astype(np.float64), TH, W.shape[0]); b = homodyne_wigner(W, 1000, r2, K)
    c = fbp(sinogram_counts(W, 0, None, K).astype(np.float64), TH, W.shape[0]); e = homodyne_wigner_exact(W, K)
    da, dc = float(np.abs(a - b).max() / np.abs(b).max()), float(np.abs(c - e).max() / np.abs(e).max())
    if da > 1e-5 or dc > 1e-5: sys.exit(f"BLAD straznika: FBP(sinogram) != homodyne_wigner ({da:.2e}, {dc:.2e})")
    print(f"straznik: wzgledne roznice {da:.1e} (N=1000) i {dc:.1e} (N=0)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--wmatch", default=r"re:^clf_W_shots_s\d+$"); ap.add_argument("--smatch", default=r"re:^clf_W_sino_s\d+$")
    a = ap.parse_args()
    if os.environ.get("QMI_XMAX") not in ("8", "12"): sys.exit("BLAD: QMI_XMAX musi byc 8 albo 12")
    mw = load_models(a.runs, a.wmatch); ms = load_models(a.runs, a.smatch)
    if len(mw) < 3 or len(ms) < 3: sys.exit(f"BLAD: za malo sieci W {sorted(mw)} / S {sorted(ms)}")
    names = sorted(mw) + sorted(ms); print("W:", sorted(mw), "S:", sorted(ms), flush=True)
    out = Path(a.runs) / "pary_sw"; out.mkdir(exist_ok=True); t0 = time.time()
    guard(sorted(Path(a.test).glob("part_loss_000_*.npz"))[0])
    for ch, lv in CELLS:
        files = sorted(Path(a.test).glob(f"part_{ch}_{lv:03d}_*.npz"))
        if len(files) != 7: sys.exit(f"BLAD: {ch} {lv}: {len(files)} plikow zamiast 7")
        for N in NS:
            dst = out / f"{ch}_{lv}_N{N}.npz"
            if dst.exists(): print("jest", dst.name, flush=True); continue
            ys, C = [], {n: [] for n in names}
            for f in files:
                Xs, Xw, y = records(str(f), N); ys.append(y)
                for n in names:
                    m, rep, sc = (mw.get(n) or ms.get(n)); C[n].append(pred(m, Xs if n in ms else Xw, sc) == y)
            y = np.concatenate(ys); M = np.stack([np.concatenate(C[n]) for n in names])
            np.savez(str(dst) + ".tmp.npz", y=y, names=np.array(names), correct=M); os.replace(str(dst) + ".tmp.npz", dst)
            print(f"{dst.name}  W {M[:len(mw)].mean():.3f}  S {M[len(mw):].mean():.3f}  {time.time()-t0:.0f}s", flush=True)
    print("PARY_SW_DONE", flush=True)
