"""Data layer for fig_observer: difference of the accuracies of the likelihood-ratio observer on heterodyne and homodyne (K = 36)
records, in percentage points, with 95% bootstrap intervals over test states, from analyses/lr_observer/lr_results.json,
and the corresponding difference of the neural networks Q and W (net_gap_pts of the same file; absent for Gaussian noise).
Compute-if-missing -> paper/figures/data/fig_observer.npz."""
import json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "paper" / "figures" / "data" / "fig_observer.npz"
ROWS = [("clean", 100), ("deph20", 100), ("loss50", 100), ("loss80", 100), ("loss80", 1000), ("gauss20", 100), ("gauss50", 100)]
def build():
    c = json.load(open(ROOT / "analyses" / "lr_observer" / "lr_results.json"))["cells"]
    g = np.array([[c[f"{k}|gap_W36|{N}"][q] for q in ("obs_gap_pts", "lo", "hi")] for k, N in ROWS])
    e = np.array([[100 * c[f"{k}|{rep}|{N}"]["obs_err"] for rep in ("Q", "W36")] for k, N in ROWS])
    net = np.array([c[f"{k}|gap_W36|{N}"].get("net_gap_pts") if c[f"{k}|gap_W36|{N}"].get("net_gap_pts") is not None else np.nan for k, N in ROWS], dtype=float)
    CACHE.parent.mkdir(parents=True, exist_ok=True); np.savez(CACHE, gap=g, err=e, net=net, cells=np.array([k for k, _ in ROWS]), N=np.array([N for _, N in ROWS]))
def load():
    if not CACHE.exists(): build()
    return np.load(CACHE)
if __name__ == "__main__":
    z = load(); print(np.c_[z["gap"].round(2), z["err"].round(2)])
