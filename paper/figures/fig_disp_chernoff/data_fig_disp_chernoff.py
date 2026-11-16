"""Data layer for fig_disp_chernoff: Chernoff exponents of heterodyne and homodyne detection (uniform recorded phase) for a
displacement of a thermal state by a coherent amplitude alpha (Eq. dispchernoff of the paper), relative to the quantum Chernoff
exponent of a collective measurement, for nbar = 0.5 and 3. Compute-if-missing -> paper/figures/data/fig_disp_chernoff.npz."""
import numpy as np
from pathlib import Path
from scipy.special import ive
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "paper" / "figures" / "data" / "fig_disp_chernoff.npz"
NB = (0.5, 3.0)
def build():
    a = np.linspace(0.02, 12, 300); d = {"alpha": a, "nb": np.array(NB)}
    for nb in NB:
        xq = a ** 2 / (4 * (nb + 1)); u = a ** 2 / (4 * (2 * nb + 1)); xw = -np.log(ive(0, u))       # u - ln I0(u)
        xqc = a ** 2 * (np.sqrt(nb + 1) - np.sqrt(nb)) ** 2
        d[f"het_{nb}"] = xq / xqc; d[f"hom_{nb}"] = xw / xqc; d[f"lim_{nb}"] = u[:1] * 0 + 1 / (4 * (2 * nb + 1) * (np.sqrt(nb + 1) - np.sqrt(nb)) ** 2)
    CACHE.parent.mkdir(parents=True, exist_ok=True); np.savez(CACHE, **d)
def load():
    if not CACHE.exists(): build()
    return np.load(CACHE)
if __name__ == "__main__":
    z = load(); print({k: np.round(z[k][[0, -1]], 4) for k in z.files if k[:3] in ("het", "hom")})
