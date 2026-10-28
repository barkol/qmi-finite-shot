"""Wspolne: wczytywanie wynikow, agregacja po ziarnach, styl (LaTeX, Wong). Dane osobno od rysowania:
kazdy skrypt fig_*.py buduje cache figures/data/<nazwa>.npz (licz-jesli-brak, --recompute wymusza), rysuje tylko z cache."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.environ.get("RES", os.path.join(HERE, "..", "results")); D = os.path.join(HERE, "data")
WONG = {"W": "#0072B2", "Q": "#D55E00"}; NAME = {"W": "Wigner", "Q": r"Husimi $Q$"}
os.makedirs(D, exist_ok=True)

def recompute(): return "--recompute" in sys.argv

def exact_runs(rep, kind="exact"):
    out = []
    for f in sorted(glob.glob(os.path.join(R, f"eval_sweep_clf_{rep}_{kind}_s*.json"))): out.append(json.load(open(f)))
    return out

def band(runs, ch):
    """(poziomy, srednia, min, max) po ziarnach dla kanalu ch."""
    lv = sorted(int(k) for k in runs[0][ch]); A = np.array([[r[ch][str(l)]["acc"] for l in lv] for r in runs])
    return np.array(lv), A.mean(0), A.min(0), A.max(0)

def style():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ok = False
    try:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 9}); fig = plt.figure(); plt.title(r"$x$"); fig.canvas.draw(); plt.close(fig); ok = True
    except Exception:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 9})
    return plt, ok

def cached(name, build):
    p = os.path.join(D, name + ".npz")
    if os.path.exists(p) and not recompute(): return dict(np.load(p, allow_pickle=True))
    d = build(); np.savez(p, **d); return d


def load_shots():
    """eval_shots.json + eval_shots_s3/s4 (dodatkowe ziarna sieci na rekordach), polaczone po nazwach modeli."""
    out = {}
    for f in [os.path.join(R, "eval_shots.json")] + sorted(glob.glob(os.path.join(R, "eval_shots_s[0-9].json"))):
        out.update(json.load(open(f)))
    return out
