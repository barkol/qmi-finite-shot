"""Data layer for fig_fisher_overview (Fig. 1): efficiencies F/F_SLD of homodyne (random phase) and heterodyne
detection near a thermal state, and the mode criterion of Eq. (criterion).
(a) bounds of Theorems 2 and 3 and the sharp homodyne supremum of sector l = 3 (Eq. (sectorsup), from Theorem 1) versus nbar (closed form), medians over random directions (analyses/fisher/out/c4.json,
    Fock support d = 4, 6, 8) and the edge directions (a^dag)^l rho0 + h.c., l = 1, 3 (c9.json, largest support).
(b) occupation nbar* above which heterodyne detection carries more information on the mode (n, |l|):
    binom(n, (n-|l|)/2) = ((2 nbar + 1)/(nbar + 1))^n  ->  nbar* = (c - 1)/(2 - c),  c = binom^(1/n).
Compute-if-missing -> paper/figures/data/fig_fisher_overview.npz."""
import json, os, numpy as np
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "paper" / "figures" / "data" / "fig_fisher_overview.npz"
FOUT = ROOT / "analyses" / "fisher" / "out"

def nbar_star(n, l):
    c = comb(n, (n - l) // 2) ** (1.0 / n)
    return (c - 1) / (2 - c)

def build():
    nb = np.linspace(0, 5, 501); q = nb / (nb + 1)
    d = dict(nb=nb, hom_l1=0.5 + 0 * nb, hom_l0=2 * nb * (nb + 1) / (2 * nb + 1) ** 2,
             het_all=(2 * nb + 1) / (2 * nb + 2), het_l3=(1 + q ** 3) / 2, het_l0=q)
    # Theorem 3: sharp homodyne supremum in sector l, (a^l+b^l)/2 max_k binom(l+2k,k)(ab)^k
    A, B = (nb + 1) / (2 * nb + 1), nb / (2 * nb + 1)
    d["hom_sup_l3"] = 0.5 * (A ** 3 + B ** 3) * np.max([comb(3 + 2 * k, k) * (A * B) ** k for k in range(60)], axis=0)
    c4 = json.load(open(FOUT / "c4.json")); nbs = (0.5, 1.0, 2.0, 5.0)
    for X in ("Q", "W"):
        med = np.array([[c4[f"thermal_nb{x}_d{s}"][f"eta{X}_median"] for s in (4, 6, 8)] for x in nbs])
        d[f"med_{X}"] = np.c_[nbs, np.median(med, 1), med.min(1), med.max(1)]
    c9 = json.load(open(FOUT / "c9.json"))
    for l in (1, 3):
        rows = [(x, c9[f"nb{x}_l{l}"]["rows"][-1][1], c9[f"nb{x}_l{l}"]["rows"][-1][2]) for x in (0.5, 1.0, 3.0)]
        d[f"edge_l{l}"] = np.array(rows)               # nbar, eta_Q, eta_W
    n = np.arange(1, 21)
    for name, off in (("edge", 0), ("next", 2), ("central", None)):
        pts = []
        for k in n:
            l = (k % 2) if off is None else k - off
            if l < 0 or (name != "edge" and l == k): continue          # n = 1 is an edge mode
            pts.append((k, nbar_star(k, l)))
        d[f"star_{name}"] = np.array(pts)
    os.makedirs(CACHE.parent, exist_ok=True); np.savez(CACHE, **d)

def load():
    if not CACHE.exists(): build()
    return np.load(CACHE)

if __name__ == "__main__":
    z = load()
    for k in z.files: print(k, z[k].shape)
    print("central n*:", z["star_central"][[0, 2, 6, 18]]); print("next:", z["star_next"][:4]); print(z["med_Q"], z["med_W"], z["edge_l3"], sep="\n")
