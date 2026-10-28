"""Zestawienie: trafnosci i sparowane roznice Q-W_ML, S-W_ML, W_ML-W(FBP) z 95% CI (paired_diff jak w make_numbers.py).
Pary:  Q, W(FBP)  <- <runs>/pary/<komorka>.npz       (zdarzenia ev; W_ML z tych samych zdarzen: cls_*/ev)
       S          <- <runs>/pary_sw/<komorka>.npz    (zdarzenia sw; W_ML z tych samych zdarzen: cls_*/sw)
Straznik: te same etykiety stanow w kazdej parze; trafnosci Q, W, S odtworzone z plikow pary zgodne z ci*.json pracy.
  python analyses/maxlik/analyze.py --set base  (albo ahm) -> analyses/maxlik/wyniki_<set>.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import t as _tdist

CELLS = [("clean", "loss", 0), ("loss50", "loss", 50), ("loss80", "loss", 80), ("deph20", "dephase", 20)]
FAM = {"ML": r"clf_W_ml_s", "MLexact": r"clf_W_exact_s", "MLfbpnet": r"clf_W_shots_s"}   # sieci czytajace obrazy W_ML


def paired_diff(M, y, isA):
    """kopia qmi make_numbers.paired_diff (grupa A minus grupa B): skladnik stanow warstwowo po rodzinach + skladnik ziaren (Welch)"""
    aQ, aW = M[isA].mean(1), M[~isA].mean(1); di = M[isA].mean(0) - M[~isA].mean(0); n = len(y)
    se_st = np.sqrt(sum((y == c).sum() * di[y == c].var(ddof=1) for c in np.unique(y))) / n
    vq, vw = aQ.var(ddof=1) / len(aQ), aW.var(ddof=1) / len(aW)
    df = (vq + vw) ** 2 / (vq ** 2 / (len(aQ) - 1) + vw ** 2 / (len(aW) - 1)) if vq + vw > 0 else 1e9
    D = float(aQ.mean() - aW.mean()); h = float(np.hypot(1.959964 * se_st, _tdist.ppf(0.975, df) * np.sqrt(vq + vw)))
    return dict(mean=D, lo=D - h, hi=D + h, kA=int(isA.sum()), kB=int((~isA).sum()), n=n, se_states=float(se_st), se_seeds=float(np.sqrt(vq + vw)), df=float(df))


def rows(d, pat):
    nm = [str(n) for n in d["names"]]; i = [k for k, n in enumerate(nm) if pat in n]
    return d["correct"][i].astype(float), [nm[k] for k in i]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="base"); ap.add_argument("--cls", default=None); ap.add_argument("--tag", default=None); a = ap.parse_args()
    A = Path("analyses/maxlik"); runs = Path("runs_eq" if a.set == "base" else "runs_ahm_eq"); cls = Path(a.cls) if a.cls else A / ("cls_base" if a.set == "base" else "cls_ahm")
    ci = json.load(open((Path(os.environ.get("QMI_PAPER", Path(__file__).resolve().parents[2] / "paper")) / "results") / ("ci.json" if a.set == "base" else "ci_ahm.json")))
    out = {}
    for tag, ch, lv in CELLS:
        for N in (100, 1000):
            c = f"{ch}_{lv}_N{N}"; P = np.load(runs / "pary" / f"{c}.npz"); SW = np.load(runs / "pary_sw" / f"{c}.npz")
            fe, fs = cls / "ev" / f"{c}.npz", cls / "sw" / f"{c}.npz"
            if not fe.exists(): print("brak", fe); continue
            E, S_ = np.load(fe), np.load(fs); y = P["y"]
            if not (np.array_equal(y, E["y"]) and np.array_equal(SW["y"], S_["y"]) and np.array_equal(y, SW["y"])): sys.exit(f"BLAD: etykiety {c}")
            Q, _ = rows(P, "clf_Q_shots_s"); W, _ = rows(P, "clf_W_shots_s"); S, _ = rows(SW, "_sino_s"); Wsw, _ = rows(SW, "clf_W_shots_s")
            r = dict(acc=dict(Q=Q.mean(), W=W.mean(), S=S.mean(), W_sw=Wsw.mean()), seeds=dict(Q=Q.mean(1).tolist(), W=W.mean(1).tolist(), S=S.mean(1).tolist()), diff={})
            for key, ref in (("Q", f"rule_N{N}_{tag}_Q"), ("W", f"rule_N{N}_{tag}_W")):   # straznik zgodnosci z praca (S z pary_sw ma inne zdarzenia niz rule_*_S, sprawdzane przez dSWe nizej)
                if ref in ci and abs(ci[ref]["mean"] - r["acc"][key]) > 1e-9: sys.exit(f"BLAD: {c} {key} {r['acc'][key]} != {ref} {ci[ref]['mean']}")
            for fam, pat in FAM.items():
                Me, _ = rows(E, pat); Ms, _ = rows(S_, pat)
                if len(Me) < 2: continue
                r["acc"][fam] = Me.mean(); r["acc"][fam + "_sw"] = Ms.mean(); r["seeds"][fam] = Me.mean(1).tolist()
                r["diff"][f"Q-{fam}"] = paired_diff(np.vstack([Q, Me]), y, np.r_[np.ones(len(Q), bool), np.zeros(len(Me), bool)])
                r["diff"][f"S-{fam}"] = paired_diff(np.vstack([S, Ms]), y, np.r_[np.ones(len(S), bool), np.zeros(len(Ms), bool)])
                r["diff"][f"{fam}-W"] = paired_diff(np.vstack([Me, W]), y, np.r_[np.ones(len(Me), bool), np.zeros(len(W), bool)])
            r["diff"]["Q-W"] = paired_diff(np.vstack([Q, W]), y, np.r_[np.ones(len(Q), bool), np.zeros(len(W), bool)])
            r["diff"]["S-W_sw"] = paired_diff(np.vstack([S, Wsw]), y, np.r_[np.ones(len(S), bool), np.zeros(len(Wsw), bool)])
            for k, ref in (("Q-W", f"rule_N{N}_{tag}_dQW"), ("S-W_sw", f"rule_N{N}_{tag}_dSWe")):
                if ref in ci and abs(ci[ref]["mean"] - r["diff"][k]["mean"]) > 1e-9: sys.exit(f"BLAD: {c} {k} != {ref}")
            out[f"{tag}_N{N}"] = r
            f = lambda d: f"{100*d['mean']:+.1f} [{100*d['lo']:+.1f},{100*d['hi']:+.1f}]"
            print(f"{tag:7s} N={N:5d} Q {100*r['acc']['Q']:.1f} W {100*r['acc']['W']:.1f} S {100*r['acc']['S']:.1f} "
                  + " ".join(f"{fam} {100*r['acc'][fam]:.1f}" for fam in FAM if fam in r["acc"])
                  + " | " + " ".join(f"{k} {f(v)}" for k, v in r["diff"].items() if k.split('-')[0] in ("Q", "S", "ML") or k.startswith("ML-")), flush=True)
    json.dump(out, open(A / f"wyniki_{a.tag or a.set}.json", "w"), indent=1, default=float); print("ANALYZE_DONE")
