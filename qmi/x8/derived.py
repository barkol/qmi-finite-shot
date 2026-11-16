"""Derived files without a script of their own (recomputed from the results):
  scores_summary.json   = scores.json (qmi.eval_scores) without the curves: per model/channel/N/level only acc, auc_macro, ece, brier, auc.
  drift_perclass_base.json = accuracy per class from eval_drift.json: confusion matrices summed over all networks clf_{W,Q}_shots_s*,
                          channels loss0/loss20/loss50, configurations walk0/walk10/walk30/jitter30, N 1e3/1e4/1e5; rounded to 3 digits.
  python -m qmi.x8.derived scores runs_eq/scores.json runs_eq/scores_summary.json
  python -m qmi.x8.derived drift  runs_eq/eval_drift.json runs_eq/drift_perclass_base.json  (also works on *.partial.json)"""
import json, sys, re, os, numpy as np
KEEP = ("acc", "auc_macro", "ece", "brier", "auc")
LABELS = ["fock", "coherent", "vacuum", "thermal", "cat", "gkp", "binomial"]

def scores(src, dst):
    d = json.load(open(src))
    out = {m: {ch: {N: {L: {k: r[k] for k in KEEP} for L, r in v3.items()} for N, v3 in v2.items()} for ch, v2 in v1.items()} for m, v1 in d.items()}
    json.dump(out, open(dst + ".tmp", "w")); os.replace(dst + ".tmp", dst)

def _cms(src):
    """(model, channel, level, cfg, N) -> confusion matrix; from the final result or from a .partial.json."""
    d = json.load(open(src)); out = {}
    if "done" in d and "acc" in d:
        for key, (k, n, cm) in d["acc"].items():
            m, ch, L, mode, sig, N = key.split("|"); out[(m, ch, L, f"{mode}{sig}", N)] = np.array(cm, float)
    else:
        for m, v1 in d.items():
            for ch, v2 in v1.items():
                for L, v3 in v2.items():
                    for cfg, v4 in v3.items():
                        for N, r in v4.items(): out[(m, ch, L, cfg, N)] = np.array(r["cm"], float)
    return out

def drift(src, dst):
    c = _cms(src); acc = {}
    for rep in "WQ":
        for ch, L in (("loss", "0"), ("loss", "20"), ("loss", "50")):
            for cfg in ("walk0", "walk10", "walk30", "jitter30"):
                for N in ("1000", "10000", "100000"):
                    cms = [v for (m, c_, l_, g, n), v in c.items() if re.match(rf"clf_{rep}_shots_s\d+$", m) and (c_, l_, g, n) == (ch, L, cfg, N)]
                    if not cms: raise SystemExit(f"missing matrix {rep} {ch}{L} {cfg} {N}")
                    cm = sum(cms); acc[f"{rep}|{ch}{L}|{cfg}|{N}"] = (np.diag(cm) / cm.sum(1)).round(3).tolist()
    json.dump({"labels": LABELS, "acc": acc}, open(dst + ".tmp", "w")); os.replace(dst + ".tmp", dst)

if __name__ == "__main__":
    {"scores": scores, "drift": drift}[sys.argv[1]](sys.argv[2], sys.argv[3]); print("DERIVED_DONE", sys.argv[1:])
