"""Data layer for fig_tmsv: efficiencies (1 +- tanh^n r)/2 of heterodyne detection of both modes and of homodyne detection of
both modes at uniform phases near the two-mode squeezed vacuum (Eq. tmsveff of the paper), orders n = 1..4, and the two
physical directions of the second order (squeezing parameter: upper branch, phase shift of one arm: lower branch).
Compute-if-missing -> paper/figures/data/fig_tmsv.npz."""
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "paper" / "figures" / "data" / "fig_tmsv.npz"
def build():
    r = np.linspace(0, 2, 201); t = np.tanh(r); d = {"r": r}
    for n in (1, 2, 3, 4): d[f"up{n}"] = (1 + t ** n) / 2; d[f"lo{n}"] = (1 - t ** n) / 2
    rm = np.array([0.5, 1.0]); d["rm"] = rm; d["sq"] = (1 + np.tanh(rm) ** 2) / 2; d["ph"] = (1 - np.tanh(rm) ** 2) / 2
    CACHE.parent.mkdir(parents=True, exist_ok=True); np.savez(CACHE, **d)
def load():
    if not CACHE.exists(): build()
    return np.load(CACHE)
if __name__ == "__main__":
    z = load(); print(z["sq"], z["ph"])
