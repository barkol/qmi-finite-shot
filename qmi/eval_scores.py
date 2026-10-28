"""Krzywe ROC, AUC i kalibracja. Te same rekordy testowe co qmi.eval_shots (to samo ziarno), ale
zapisujemy prawdopodobienstwa klas (softmax), nie tylko argmax. N=-1 oznacza dokladne obrazy (W i Q liczone wprost),
N=0 granice N->inf dla rekordow (Q dokladne, W = dokladne FBP), N>0 rekordy skonczone.
Wyjscie: <out>.npz (prawdopodobienstwa float16 + etykiety, klucz model|kanal|poziom|N) oraz <out>.json z metrykami:
AUC jedna-klasa-przeciw-reszcie dla kazdej rodziny, makro-AUC, ECE (15 przedzialow), Brier, dokladnosc, krzywe ROC
(TPR na siatce 101 punktow FPR) dla kazdej rodziny. Wznawialny na poziomie pliku wyjsciowego.
   python -m qmi.eval_scores --runs runs --match '_(exact|shots)_s' --test data/qmi/test/sweep --out runs/scores"""
import argparse, json, re, time, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score, roc_curve
from qmi.eval_shots import load_models, CHN, SEED
from qmi.shots import heterodyne_image, homodyne_wigner, homodyne_wigner_exact
from qmi.clf import DEV

NS = (-1, 100, 1000, 10000, 100000, 1000000, 0)
FPR = np.linspace(0, 1, 101)


def gen(args):
    f, N, K = args; d = np.load(f, allow_pickle=True)
    if N == -1: return int(d["channel"][0]), int(d["level"][0]), N, d["W"], d["Q"], d["label"]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(Path(f).name.encode()), N]))   # jak w eval_shots
    Xw, Xq = [], []
    for W, Q in zip(d["W"], d["Q"]):
        if N == 0: Xq.append(Q); Xw.append(homodyne_wigner_exact(W, K))
        else: Xq.append(heterodyne_image(Q, N, rng)); Xw.append(homodyne_wigner(W, N, rng, K))
    return int(d["channel"][0]), int(d["level"][0]), N, np.stack(Xw), np.stack(Xq), d["label"]


@torch.no_grad()
def proba(m, X, scale, bs=256):
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.softmax(m(torch.from_numpy((X[i:i + bs] / scale).astype(np.float32)).unsqueeze(1).to(DEV)), 1).cpu())
    return torch.cat(out).numpy()


def metrics(P, y, nb=15):
    P = P.astype(np.float64); P /= P.sum(1, keepdims=True); pred = P.argmax(1); conf = P.max(1)
    r = {"n": int(len(y)), "acc": float((pred == y).mean()), "brier": float(((P - np.eye(7)[y]) ** 2).sum(1).mean())}
    auc, roc = [], []
    for c in range(7):
        t = (y == c).astype(int)
        if 0 < t.sum() < len(t):
            auc.append(float(roc_auc_score(t, P[:, c]))); fp, tp, _ = roc_curve(t, P[:, c]); roc.append(np.interp(FPR, fp, tp).round(4).tolist())
        else: auc.append(None); roc.append(None)
    r["auc"] = auc; v = [a for a in auc if a is not None]; r["auc_macro"] = float(np.mean(v)) if v else None; r["roc_tpr"] = roc
    edges = np.linspace(0, 1, nb + 1); ece = 0.0; rel = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
            rel.append([float((lo + hi) / 2), float(conf[m].mean()), float((pred[m] == y[m]).mean()), int(m.sum())])
    r["ece"] = float(ece); r["reliability"] = rel
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--match", default=r"_(exact|shots)_s")
    ap.add_argument("--workers", type=int, default=20); ap.add_argument("--K", type=int, default=36); ap.add_argument("--ns", default=",".join(map(str, NS))); a = ap.parse_args()
    out = Path(a.out)
    if out.with_suffix(".json").exists(): print("juz jest", out.with_suffix(".json")); print("EVAL_SCORES_DONE"); raise SystemExit
    torch.set_num_threads(2)
    models = load_models(a.runs, "re:" + a.match)   # laduje tylko pasujace (nie czyta best.pt trenowanych wlasnie sieci)
    print("modele:", list(models), flush=True)
    if not models: raise SystemExit("brak modeli")
    files = [f for f in sorted(Path(a.test).glob("part_*.npz")) if f.name.split("_")[1] in CHN]
    tasks = [(str(f), N, a.K) for f in files for N in map(int, a.ns.split(","))]; print(len(tasks), "zadan", flush=True)
    acc = {}; t0 = time.time(); inv = {v: k for k, v in CHN.items()}
    with Pool(a.workers) as pool:
        for i, (ch, lv, N, Xw, Xq, y) in enumerate(pool.imap_unordered(gen, tasks, chunksize=1)):
            for name, (m, rep, sc) in models.items():
                P = proba(m, Xw if rep == "W" else Xq, sc); key = f"{name}|{inv[ch]}|{lv}|{N}"
                r = acc.setdefault(key, [[], []]); r[0].append(P.astype(np.float16)); r[1].append(np.asarray(y, np.int8))
            if i % 50 == 0: print(f"{i}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    arrs = {}; res = {}
    for key, (Ps, ys) in acc.items():
        P = np.concatenate(Ps); y = np.concatenate(ys).astype(int); arrs[key + "|P"] = P; arrs[key + "|y"] = y.astype(np.int8)
        name, ch, lv, N = key.split("|"); res.setdefault(name, {}).setdefault(ch, {}).setdefault(N, {})[lv] = metrics(P, y)
    np.savez_compressed(out.with_suffix(".npz"), **arrs)
    json.dump(res, open(out.with_suffix(".json.tmp"), "w")); out.with_suffix(".json.tmp").replace(out.with_suffix(".json"))
    print("EVAL_SCORES_DONE", flush=True)
