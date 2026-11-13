"""Data layer for fig_gauss_overview (Fig. 1): the setting and the main consequences of Theorems 1-3 in three 3D panels.
(a) phase space: Wigner function W0 of a squeezed thermal reference state (nbar = 0.5, r = 0.4), a change W_Delta = g W0 in one
    Hermite-Gauss mode (order 3, angular momentum 1), the covariance ellipse of the state and the noise of the two detections;
(b) efficiencies F/F_SLD of all modes (n, l), n <= 8, near a thermal state with nbar = 1 (closed forms of Corollary 2):
    heterodyne (1 + q^|l|) q^k / 2, homodyne [P(X = k) + P(X = k + |l|)]/2, X ~ Bin(n, nbar/(2 nbar + 1)), k = (n - |l|)/2;
(c) largest efficiency over all directions near a squeezed thermal state against the squeezing r and the occupation nbar:
    heterodyne lambda_r, homodyne at a uniform phase (1 + tanh r)/2, homodyne at the adapted phase 1/2.
Compute-if-missing -> paper/figures/data/fig_gauss_overview.npz."""
import numpy as np
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "paper" / "figures" / "data" / "fig_gauss_overview.npz"
def build():
    d = {}
    # (a)
    nb, r = 0.5, 0.4; v = nb + 0.5; V = np.diag([v * np.exp(-2 * r), v * np.exp(2 * r)])
    x = np.linspace(-3.6, 3.6, 181); X, P = np.meshgrid(x, x)
    W0 = np.exp(-0.5 * (X ** 2 / V[0, 0] + P ** 2 / V[1, 1])) / (2 * np.pi * v)
    z = (X / np.sqrt(V[0, 0]) + 1j * P / np.sqrt(V[1, 1])) / np.sqrt(2)          # whitened coordinates
    g = np.real(z * (2 - np.abs(z) ** 2))                                           # zeta L_1^(1)(|zeta|^2): order 3, l = 1
    d.update(a_x=x, a_W0=W0, a_WD=g * W0, a_V=V, a_nb=nb, a_r=r)
    # (b)
    nb = 1.0; q = nb / (nb + 1); p = nb / (2 * nb + 1); rows = []
    for n in range(1, 9):
        for l in range(-n, n + 1, 2):
            k = (n - abs(l)) // 2; pb = lambda j: comb(n, j) * p ** j * (1 - p) ** (n - j)
            rows.append((n, l, (1 + q ** abs(l)) * q ** k / 2, (pb(k) + pb(k + abs(l))) / 2))
    d.update(b_modes=np.array(rows), b_nb=nb, b_lambda=(2 * nb + 1) / (2 * nb + 2))
    # (c)
    r = np.linspace(0, 1.5, 31); n_ = np.linspace(0, 2, 21); Rr, Nn = np.meshgrid(r, n_)
    d.update(c_r=r, c_nb=n_, c_het=np.exp(2 * Rr) * (2 * Nn + 1) / (np.exp(2 * Rr) * (2 * Nn + 1) + 1), c_hom=(1 + np.tanh(Rr)) / 2)
    CACHE.parent.mkdir(parents=True, exist_ok=True); np.savez(CACHE, **d)
def load():
    if not CACHE.exists(): build()
    return np.load(CACHE)
if __name__ == "__main__":
    z = load(); print(z["b_modes"][:6].round(4))
