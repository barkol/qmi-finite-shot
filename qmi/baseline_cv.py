"""Bazy niesieciowe z doborem hiperparametrow 5-krotna walidacja krzyzowa na zbiorze treningowym.
Te same modele co qmi.baseline (obrazy 128x128 -> 32x32 usrednione blokami, StandardScaler, PCA), ale liczba skladowych PCA,
regularyzacja C (regresja logistyczna) i k (k-NN) sa wybierane przez StratifiedGroupKFold(5) z grupa = plik czesciowy
(stany jednego pliku nie trafiaja jednoczesnie do treningu i walidacji). Model z najlepszymi parametrami jest uczony na
calym zbiorze treningowym i oceniany na tescie. Format wyniku jak w qmi.baseline, plus wybrane parametry i wyniki CV.
   python -m qmi.baseline_cv --train data/qmi/train/mixed --test data/qmi/test/sweep --out runs/baseline_cv.json"""
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from qmi.data import load_dir, wilson
from qmi.baseline import pool

GRIDS = {
    "logreg": (lambda: Pipeline([("sc", StandardScaler()), ("pca", PCA(random_state=0)), ("clf", LogisticRegression(max_iter=5000))]),
               {"pca__n_components": [50, 100, 200], "clf__C": [0.01, 0.1, 1.0, 10.0]}),
    "knn": (lambda: Pipeline([("sc", StandardScaler()), ("pca", PCA(random_state=0)), ("clf", KNeighborsClassifier())]),
            {"pca__n_components": [20, 50, 100], "clf__n_neighbors": [1, 5, 15, 31]}),
}


def load_groups(root, rep):
    X, y, g = [], [], []
    for i, f in enumerate(sorted(Path(root).glob("part_*.npz"))):
        d = np.load(f, allow_pickle=True)
        X.append(pool(d[rep].astype(np.float32))); y.append(d["label"]); g.append(np.full(len(d["label"]), i))
    if not X: raise FileNotFoundError(root)
    return np.concatenate(X), np.concatenate(y).astype(np.int64), np.concatenate(g)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=10); a = ap.parse_args()
    out = Path(a.out); part = out.with_name(out.name + ".partial.json")
    res = json.load(open(part)) if part.exists() else {}          # wznawianie: gotowe modele pomijamy
    for rep in ("W", "Q"):
        Xt, yt, gt = load_groups(a.train, rep)
        test = {chn: load_dir(a.test, rep, channels=[chn]) for chn in ("loss", "dephase", "gauss")}
        for name, (make, grid) in GRIDS.items():
            key = f"{name}_{rep}"
            if key in res: continue
            t0 = time.time()
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
            gs = GridSearchCV(make(), grid, cv=cv, scoring="accuracy", n_jobs=a.jobs, refit=True)
            gs.fit(Xt, yt, groups=gt)
            r = {"cv": {"best_params": gs.best_params_, "best_mean": float(gs.best_score_),
                        "best_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
                        "grid": [{"params": p, "mean": float(m), "std": float(s)} for p, m, s in
                                 zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"], gs.cv_results_["std_test_score"])]}}
            for chn, d in test.items():
                p = gs.predict(pool(d["X"])); r[chn] = {}
                for L in sorted(set(d["level"].tolist())):
                    mk = d["level"] == L; k = int((p[mk] == d["y"][mk]).sum()); n = int(mk.sum()); lo, hi = wilson(k, n)
                    r[chn][int(L)] = dict(acc=k / n, lo=lo, hi=hi, n=n)
            res[key] = r; json.dump(res, open(part, "w"))
            print(key, gs.best_params_, "cv %.3f +- %.3f" % (gs.best_score_, r["cv"]["best_std"]),
                  {c: round(r[c][min(r[c])]["acc"], 3) for c in test if r[c]}, "%.0f s" % (time.time() - t0), flush=True)
    json.dump(res, open(out, "w")); part.unlink(missing_ok=True); print("BASELINE_CV_DONE")
