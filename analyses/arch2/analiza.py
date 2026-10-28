"""Analiza drugiej architektury: komorki reguly, trafnosci, sparowane roznice Q-W, Q-S, r18-v2.
Metoda CI: funkcje paired_diff, Acc, ci_of (z _wilson, _hs) wyjete doslownie (ast) z
paper/scripts/make_numbers.py, wiec przedzialy licza sie tak samo jak w pracy.
Lisc wg fig_rule_flow.py (tylko zbior bazowy): Q gdy CI(Q-W) > 0, W gdy < 0, none gdy zawiera zero.
Straznik: v2 (5 ziaren) musi odtworzyc results/ci.json (rule_N*_*_{Q,W,S,dQW,dQS}) co do 1e-9.
Krok 1 (dane): wyniki.json; krok 2 (tylko z JSON-a): tabela.md.
  python analyses/arch2/analiza.py"""
import ast, json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import t as _tdist

HERE = Path(__file__).resolve().parent
MN = (Path(os.environ.get("QMI_PAPER", Path(__file__).resolve().parents[2] / "paper")) / "scripts" / "make_numbers.py")
CIJ = (Path(os.environ.get("QMI_PAPER", Path(__file__).resolve().parents[2] / "paper")) / "results") / "ci.json"
src = MN.read_text(); tree = ast.parse(src); ns = {"np": np, "_tdist": _tdist}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ("Acc", "_wilson", "_hs", "ci_of", "paired_diff"):
        exec(compile(ast.Module([node], []), str(MN), "exec"), ns)
Acc, ci_of, paired_diff = ns["Acc"], ns["ci_of"], ns["paired_diff"]

CELLS = [("clean", "loss", 0), ("loss50", "loss", 50), ("loss80", "loss", 80), ("deph20", "dephase", 20)]; NS = (100, 1000, 10000)
GROUPS = {   # architektura -> rep -> prefiks nazw
    "r18": {"Q": "r18_Q_shots_s", "W": "r18_W_shots_s", "S": "r18_W_sino_s"},
    "v2": {"Q": "clf_Q_shots_s", "W": "clf_W_shots_s", "S": "clf_W_sino_s"},
    "v2_s012": {"Q": "clf_Q_shots_s", "W": "clf_W_shots_s", "S": "clf_W_sino_s"},
}
SEEDS = {"r18": {0, 1, 2}, "v2": {0, 1, 2, 3, 4}, "v2_s012": {0, 1, 2}}


def rows(names, arch, rep):
    p = GROUPS[arch][rep]
    return [i for i, n in enumerate(names) if n.startswith(p) and int(n[len(p):]) in SEEDS[arch]]


def sign(c): return 1 if c["lo"] > 0 else (-1 if c["hi"] < 0 else 0)
LEAF = {1: "Q", -1: "W", 0: "none"}


def diff(M, y, names, ia, ib):
    sub = np.vstack([M[ia], M[ib]]); nm = [names[i] for i in ia] + [names[i] for i in ib]; ga = set(names[i] for i in ia)
    return paired_diff(None, grp=lambda s: s in ga, M=sub, y=y, names=nm)


def compute():
    ref = json.load(open(CIJ)); out = {"cells": {}, "guard_v2_vs_ci_json": []}
    for tag, ch, lv in CELLS:
        for N in NS:
            d = np.load(HERE / "pary" / f"{ch}_{lv}_N{N}.npz"); M = d["correct"].astype(float); y = d["y"]; names = [str(s) for s in d["names"]]
            cell = {}
            for arch in GROUPS:
                r = {k: rows(names, arch, k) for k in "QWS"}
                if any(len(v) != len(SEEDS[arch]) for v in r.values()): sys.exit(f"BLAD: {arch} {tag} N={N}: brak sieci {r}")
                a = {k: ci_of(Acc(M[v].mean(1), [len(y)] * len(v))) | {"seeds": M[v].mean(1).tolist()} for k, v in r.items()}
                dqw = diff(M, y, names, r["Q"], r["W"]); dqs = diff(M, y, names, r["Q"], r["S"]); dsw = diff(M, y, names, r["S"], r["W"])
                cell[arch] = dict(acc=a, dQW=dqw, dQS=dqs, dSW=dsw, leaf_QW=LEAF[sign(dqw)], leaf_QS=LEAF[sign(dqs)])
            for k in "QWS":   # r18 - v2 (5 ziaren) w tej samej reprezentacji
                ia, ib = rows(names, "r18", k), rows(names, "v2", k)
                cell.setdefault("r18_minus_v2", {})[k] = diff(M, y, names, ia, ib)
            # straznik: v2 5 ziaren == ci.json
            for key, val in (("Q", cell["v2"]["acc"]["Q"]), ("W", cell["v2"]["acc"]["W"]), ("S", cell["v2"]["acc"]["S"]),
                             ("dQW", cell["v2"]["dQW"]), ("dQS", cell["v2"]["dQS"])):
                rk = f"rule_N{N}_{tag}_{key}"; dev = max(abs(ref[rk][f] - val[f]) for f in ("mean", "lo", "hi"))
                out["guard_v2_vs_ci_json"].append([rk, dev])
                if dev > 1e-9: sys.exit(f"BLAD: v2 nie odtwarza ci.json {rk}: odchylenie {dev}")
            out["cells"][f"{tag}_N{N}"] = cell
    json.dump(out, open(HERE / "wyniki.json", "w"), indent=1); print("wyniki.json zapisany; straznik ci.json OK,", len(out["guard_v2_vs_ci_json"]), "kluczy")


def render():
    w = json.load(open(HERE / "wyniki.json"))["cells"]; f = lambda c: f"{100*c['mean']:+.1f} [{100*c['lo']:+.1f}, {100*c['hi']:+.1f}]"
    p = lambda c: f"{100*c['mean']:.1f}"
    L = ["| N | komorka | r18 Q / W / S (%) | v2 Q / W / S (%) | r18 Q−W (pp, 95% CI) | v2 Q−W (pp, 95% CI) | lisc r18 / v2 | r18 Q−S (pp) | v2 Q−S (pp) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for N in NS:
        for tag, _, _ in CELLS:
            c = w[f"{tag}_N{N}"]; r, v = c["r18"], c["v2"]
            L.append(f"| {N} | {tag} | {p(r['acc']['Q'])} / {p(r['acc']['W'])} / {p(r['acc']['S'])} | {p(v['acc']['Q'])} / {p(v['acc']['W'])} / {p(v['acc']['S'])} | "
                     f"{f(r['dQW'])} | {f(v['dQW'])} | **{r['leaf_QW']}** / **{v['leaf_QW']}** | {f(r['dQS'])} | {f(v['dQS'])} |")
    L += ["", "| N | komorka | r18−v2 Q (pp) | r18−v2 W (pp) | r18−v2 S (pp) | v2 (s0–2) Q−W (pp) | lisc v2 (s0–2) |", "|---|---|---|---|---|---|---|"]
    for N in NS:
        for tag, _, _ in CELLS:
            c = w[f"{tag}_N{N}"]; d = c["r18_minus_v2"]
            L.append(f"| {N} | {tag} | {f(d['Q'])} | {f(d['W'])} | {f(d['S'])} | {f(c['v2_s012']['dQW'])} | {c['v2_s012']['leaf_QW']} |")
    (HERE / "tabela.md").write_text("\n".join(L) + "\n"); print("tabela.md zapisana")


if __name__ == "__main__":
    if "--render" not in sys.argv: compute()
    render()
