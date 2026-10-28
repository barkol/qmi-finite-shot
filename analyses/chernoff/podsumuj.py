"""Podsumowanie wykladnikow Chernoffa (out/M24_*.npz) -> out/podsumowanie.json + tabela na stdout.
xi_X(rho) = min po sigma z innych klas C_X; P_X(N) ~ srednia_rho 1/2 exp(-N xi_X) (granica Chernoffa dla najblizszej pary);
Q-W [pkt] = 100 (P_W - P_Q). Najtrudniejsza para = klasa sigma, w ktorej osiagniete minimum."""
import json, sys
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent / "out"; TAG = sys.argv[1] if len(sys.argv) > 1 else "M24"
NS = (100, 1000, 10000); NG = np.logspace(1, 5, 81)
res = {}
for f in sorted(OUT.glob(f"{TAG}_*.npz")):
    d = np.load(f, allow_pickle=True); lab = d["lab"]; names = list(d["names"]); other = lab[:, None] != lab[None, :]
    row = {}
    for X in ("CQ", "CW", "CQCB"):
        C = np.where(other, d[X], np.inf); row[X] = C.min(axis=1); row[X + "_arg"] = lab[C.argmin(axis=1)]
    P = {X: lambda N, X=X: 0.5 * np.exp(-np.outer(N, row[X])).mean(axis=1) for X in ("CQ", "CW", "CQCB")}
    dQW = 100 * (P["CW"](NG) - P["CQ"](NG)); sgn = np.sign(np.round(dQW, 2)); cross = [float(NG[i + 1]) for i in range(len(NG) - 1) if sgn[i] * sgn[i + 1] < 0]
    pairs = {}
    for i in range(len(lab)):
        k = " / ".join(sorted((names[lab[i]], names[row["CQ_arg"][i]]))); pairs[k] = pairs.get(k, 0) + 1
    fam, cell = f.stem.split("_", 2)[1:]
    res.setdefault(fam, {})[cell] = dict(
        xiQ_med=float(np.median(row["CQ"])), xiW_med=float(np.median(row["CW"])), xiQCB_med=float(np.median(row["CQCB"])),
        frac_W_gt_Q=float((row["CW"] > row["CQ"]).mean()), ratio_WQ_med=float(np.median(row["CW"] / row["CQ"])),
        effQ_med=float(np.median(row["CQ"] / row["CQCB"])), effW_med=float(np.median(row["CW"] / row["CQCB"])),
        dQW_pts={str(N): float(100 * (P["CW"](np.array([N])) - P["CQ"](np.array([N])))[0]) for N in NS},
        PeQ={str(N): float(P["CQ"](np.array([N]))[0]) for N in NS}, PeW={str(N): float(P["CW"](np.array([N]))[0]) for N in NS},
        crossover_N=cross, hardest_pairs_Q=dict(sorted(pairs.items(), key=lambda t: -t[1])[:3]),
        per_class={names[c]: dict(xiQ=float(np.median(row["CQ"][lab == c])), xiW=float(np.median(row["CW"][lab == c])))
                   for c in range(len(names))})
(OUT / f"podsumowanie_{TAG}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
for fam, cells in res.items():
    print(f"== {fam}")
    print(f"{'komorka':10s} {'xiQ':>8s} {'xiW':>8s} {'xiQCB':>8s} {'W>Q':>5s} {'W/Q':>5s}  Q-W[pkt] N=1e2   1e3   1e4   przejscie N")
    for c, r in cells.items():
        dq = r["dQW_pts"]
        print(f"{c:10s} {r['xiQ_med']:8.4f} {r['xiW_med']:8.4f} {r['xiQCB_med']:8.4f} {r['frac_W_gt_Q']:5.2f} {r['ratio_WQ_med']:5.2f}"
              f"  {dq['100']:+8.1f} {dq['1000']:+6.1f} {dq['10000']:+6.1f}   {', '.join(f'{x:.0f}' for x in r['crossover_N']) or '-'}")
        print(f"{'':10s} najtrudniejsze pary (Q): {r['hardest_pairs_Q']}")
