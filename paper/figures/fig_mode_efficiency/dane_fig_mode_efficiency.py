"""Data layer for fig_mode_efficiency (Fig. 2): efficiencies of the Hermite-Gauss mode (n, l) near a thermal state,
Theorem 1 of the paper (closed form, no numerics):
  heterodyne  F_Q/F_SLD = [P(N0 >= k) + P(N0 >= k+|l|)]/2 = (1 + q^|l|) q^k / 2,   N0 thermal, q = nbar/(nbar+1)
  homodyne    F_W/F_SLD = [P(X = k) + P(X = k+|l|)]/2,  X ~ Bin(n, kappa/2), kappa/2 = nbar/(2 nbar + 1),
k = (n-|l|)/2, for edge modes (|l| = n), |l| = n - 2 and central modes (|l| <= 1).
Compute-if-missing -> paper/figures/data/fig_mode_efficiency.npz."""
import numpy as np
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "paper" / "figures" / "data" / "fig_mode_efficiency.npz"
def eff(nb, n, l):
    q = nb / (nb + 1); p = nb / (2 * nb + 1); k = (n - abs(l)) // 2
    pb = lambda j: comb(n, j) * p ** j * (1 - p) ** (n - j)
    het = (1 + q ** abs(l)) * q ** k / 2
    hom = (pb(k) + pb(k + abs(l))) / 2
    return het, hom
def build():
    d = {}
    for nb in (0.5, 3.0):
        for fam, lf in (("edge", lambda n: n), ("next", lambda n: n - 2), ("central", lambda n: n % 2)):
            rows = []
            for n in range(1, 21):
                l = lf(n)
                if l < 0 or (fam != "edge" and l == n): continue
                if n == 0: continue
                rows.append((n, *eff(nb, n, l)))
            d[f"nb{nb}_{fam}"] = np.array(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True); np.savez(CACHE, **d)
def load():
    if not CACHE.exists(): build()
    return np.load(CACHE)
if __name__ == "__main__":
    z = load()
    for k in z.files: print(k, z[k][:3].round(4).tolist())
