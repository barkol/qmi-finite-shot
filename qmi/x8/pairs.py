r"""Paired evaluation of Q against W on a common test set: correctness of every test state for every network.
The test records are deterministic (seed = file, N; qmi.eval_shots.gen), so all networks see the same records and the
difference Q-W can be computed state by state. Cells: loss 0/50/80%, dephasing 20%, N = 1e2/1e3/1e4.
Output: <runs>/pairs/<channel>_<level>_N<N>.npz (y, network names, correctness matrix [network, state]); one file per cell is
the checkpoint, a restart skips finished cells. Check: the number of correct decisions must agree with eval_shots*.json.
  QMI_XMAX=12 PYTHONPATH=. python -m qmi.x8.pairs --runs runs_eq --test data_eq/qmi/test/sweep \
      --ref runs_eq/eval_shots_all_s0-4.json
--sino <regex> adds the homodyne networks without reconstruction (input: histograms from qmi.eval_shots.gen in mode
"sino", the same states in the same order as the image records); the output goes to the directory --out (e.g. pairs_sino),
separate from the files in pairs/:
  ... -m qmi.x8.pairs --runs runs_eq --test data_eq/qmi/test/sweep --match 're:^clf_Q_shots_s\d+$' \
      --sino 're:^clf_W_sino_s\d+$' --out pairs_sino --ref runs_eq/eval_shots_all_s0-4.json runs_eq/eval_shots_sino_s0-4.json"""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np
from qmi.eval_shots import gen, load_models, pred

CELLS = [("loss", 0), ("loss", 50), ("loss", 80), ("dephase", 20)]; NS = (100, 1000, 10000)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--ref", nargs="+", required=True); ap.add_argument("--match", default=r"re:^clf_[WQ]_shots_s\d+$")
    ap.add_argument("--sino", default=None); ap.add_argument("--out", default="pairs")
    a = ap.parse_args()
    if os.environ.get("QMI_XMAX") not in ("8", "12"): sys.exit("ERROR: QMI_XMAX must be 8 or 12; otherwise the records are on the grid +-5")
    ref = {}
    for f in a.ref: ref.update(json.load(open(f)))
    models = load_models(a.runs, a.match); sino = set()
    if a.sino:
        ms = load_models(a.runs, a.sino); sino = set(ms); models.update(ms)
    names = sorted(models); print("networks:", names, "histogram input:", sorted(sino), flush=True)
    if not names or any(n not in ref for n in names): sys.exit(f"ERROR: networks without a reference in --ref: {[n for n in names if n not in ref]}")
    out = Path(a.runs) / a.out; out.mkdir(exist_ok=True); t0 = time.time()
    for ch, lv in CELLS:
        files = sorted(Path(a.test).glob(f"part_{ch}_{lv:03d}_*.npz"))
        if len(files) != 7: sys.exit(f"ERROR: {ch} {lv}: {len(files)} files instead of 7")
        for N in NS:
            dst = out / f"{ch}_{lv}_N{N}.npz"
            if dst.exists(): print("exists", dst.name, flush=True); continue
            ys, C = [], {n: [] for n in names}
            for f in files:
                _, _, _, Xw, Xq, y = gen((str(f), N, 36, "img")); ys.append(y)
                if sino:
                    _, _, _, Xs, _, ys_ = gen((str(f), N, 36, "sino"))
                    if not np.array_equal(ys_, y): sys.exit(f"ERROR: different labels img/sino in {f.name}")
                for n in names:
                    m, rep, sc = models[n]; C[n].append(pred(m, Xs if n in sino else (Xw if rep == "W" else Xq), sc) == y)
            y = np.concatenate(ys); M = np.stack([np.concatenate(C[n]) for n in names])
            for i, n in enumerate(names):   # check: the same number of correct decisions as in eval_shots
                r = ref[n][ch][str(N)][str(lv)]; k = int(round(r["acc"] * r["n"]))
                if r["n"] != len(y) or k != int(M[i].sum()):
                    sys.exit(f"ERROR: mismatch {n} {ch} {lv} N={N}: {int(M[i].sum())}/{len(y)} against {k}/{r['n']}")
            np.savez(str(dst) + ".tmp.npz", y=y, names=np.array(names), correct=M); os.replace(str(dst) + ".tmp.npz", dst)
            print(f"{dst.name}  agrees with eval_shots  {time.time()-t0:.0f}s", flush=True)
    print("PAIRS_DONE", flush=True)
