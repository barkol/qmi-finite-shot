"""Data layer for fig_squeezed_bounds: efficiencies near a squeezed thermal state (Theorems 1-3 of the paper, symlib).
(a) largest efficiency over all directions against the squeezing parameter r: heterodyne lambda_p for nbar = 0, 0.5, 3;
    homodyne at a uniform laboratory phase (1 + tanh r)/2; homodyne at a phase uniform in the whitened coordinates 1/2.
(b) largest efficiency in each order n at nbar = 0.5, r = 0.5 for the same three measurements (eigenvalues of the blocks).
Compute-if-missing -> paper/figures/data/fig_squeezed_bounds.npz."""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analyses" / "fisher"))
CACHE = ROOT / "paper" / "figures" / "data" / "fig_squeezed_bounds.npz"
def build():
    import symlib as sl
    r = np.linspace(0, 1.5, 61); d = {"r": r, "hom_lab": (1 + np.tanh(r)) / 2, "hom_adapted": 0.5 + 0 * r}
    for nb in (0.0, 0.5, 3.0):
        d[f"het_nb{nb}"] = np.exp(2 * r) * (2 * nb + 1) / (np.exp(2 * r) * (2 * nb + 1) + 1)
    nb, r0 = 0.5, 0.5; v = nb + 0.5; t = np.tanh(r0); N = np.arange(1, 13)
    ph = (np.arange(400) + 0.5) * np.pi / 400; w = (1 - t ** 2) / (1 + t ** 2 + 2 * t * np.cos(2 * ph)); w /= w.sum()
    Lh = sl.lam(v, 0.5 * np.diag([np.exp(2 * r0), np.exp(-2 * r0)]))
    rows = []
    for n in N:
        P = [sl.sym(sl.proj(p), n) for p in ph]
        rows.append((n, sl.eff(sl.sym(Lh, n), v, n).max(), sl.eff(sum(wi * Pi for wi, Pi in zip(w, P)), v, n).max(), sl.eff(sum(P) / len(P), v, n).max()))
    d["orders"] = np.array(rows); d["nb_b"], d["r_b"] = nb, r0
    CACHE.parent.mkdir(parents=True, exist_ok=True); np.savez(CACHE, **d)
def load():
    if not CACHE.exists(): build()
    return np.load(CACHE)
if __name__ == "__main__":
    z = load(); print(z["orders"].round(4))
