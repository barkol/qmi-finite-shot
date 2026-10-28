"""Bazy niesieciowe na dokladnych obrazach: regresja logistyczna i k-NN (obrazy 128x128 -> 32x32 usrednione blokami, PCA).
Pokazuje, ile daje siec ponad prosty klasyfikator.   python -m qmi.baseline --train data/qmi/train/mixed --test data/qmi/test/sweep --out runs/baseline.json"""
import argparse, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from qmi.data import load_dir, wilson

pool = lambda X: X.reshape(len(X), 32, 4, 32, 4).mean((2, 4)).reshape(len(X), -1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); res = {}
    for rep in ("W", "Q"):
        tr = load_dir(a.train, rep); Xt = pool(tr["X"]); yt = tr["y"]
        models = {"logreg": make_pipeline(StandardScaler(), PCA(100, random_state=0), LogisticRegression(max_iter=2000)),
                  "knn": make_pipeline(StandardScaler(), PCA(50, random_state=0), KNeighborsClassifier(5))}
        for name, m in models.items():
            m.fit(Xt, yt); res[f"{name}_{rep}"] = {}
            for chn in ("loss", "dephase", "gauss"):
                d = load_dir(a.test, rep, channels=[chn]); p = m.predict(pool(d["X"])); res[f"{name}_{rep}"][chn] = {}
                for L in sorted(set(d["level"].tolist())):
                    mk = d["level"] == L; k = int((p[mk] == d["y"][mk]).sum()); n = int(mk.sum()); lo, hi = wilson(k, n)
                    res[f"{name}_{rep}"][chn][int(L)] = dict(acc=k / n, lo=lo, hi=hi, n=n)
            print(name, rep, {c: round(res[f"{name}_{rep}"][c][0]["acc"], 3) for c in res[f"{name}_{rep}"]}, flush=True)
    json.dump(res, open(a.out, "w")); print("BASELINE_DONE")
