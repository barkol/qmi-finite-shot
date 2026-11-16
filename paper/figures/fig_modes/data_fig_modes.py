"""Data layer for fig_modes: changes of the Wigner function, W_Delta = g W_0, for the Hermite-Gauss modes (n, l) of one mode
near a thermal state, in the normalized quadratures z = x / sqrt(v). Real part of rho^|l| L_k^{|l|}(rho^2/2) e^{i l theta},
k = (n - |l|)/2, normalized to unit norm under W_0. Compute-if-missing -> paper/figures/data/fig_modes.npz."""
import numpy as np
from pathlib import Path
from math import factorial
from scipy.special import eval_genlaguerre
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "paper" / "figures" / "data" / "fig_modes.npz"
MODES = [(1, 1), (2, 0), (2, 2), (3, 1), (3, 3), (4, 0), (4, 2), (4, 4)]
def build():
    x = np.linspace(-4, 4, 241); Z1, Z2 = np.meshgrid(x, x); r2 = Z1 ** 2 + Z2 ** 2; th = np.arctan2(Z2, Z1)
    W0 = np.exp(-r2 / 2) / (2 * np.pi); d = {"x": x, "modes": np.array(MODES)}
    for n, l in MODES:
        k = (n - l) // 2
        norm = np.sqrt(2.0 ** l * factorial(k + l) / factorial(k))            # norm of the complex mode under W_0
        g = r2 ** (l / 2) * eval_genlaguerre(k, l, r2 / 2) * np.cos(l * th) / norm * (np.sqrt(2) if l else 1)
        xc = np.linspace(-9, 9, 721); A, B = np.meshgrid(xc, xc); c2 = A ** 2 + B ** 2      # wider grid for the check of the norm
        gc = c2 ** (l / 2) * eval_genlaguerre(k, l, c2 / 2) * np.cos(l * np.arctan2(B, A)) / norm * (np.sqrt(2) if l else 1)
        d[f"norm_{n}_{l}"] = float((gc ** 2 * np.exp(-c2 / 2) / (2 * np.pi)).sum() * (xc[1] - xc[0]) ** 2)   # check: 1
        d[f"W_{n}_{l}"] = g * W0
    CACHE.parent.mkdir(parents=True, exist_ok=True); np.savez(CACHE, **d)
def load():
    if not CACHE.exists(): build()
    return np.load(CACHE)
if __name__ == "__main__":
    z = load(); print({k: round(float(z[k]), 4) for k in z.files if k.startswith("norm")})
