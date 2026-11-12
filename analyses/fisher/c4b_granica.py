"""C4b: conjecture eta_W = F_W/F_SLD <= 1/2 (random-phase homodyne, thermal reference); vacuum tangent eta = 1/2."""
import json, os, numpy as np
from scipy.linalg import eigh
from c4lib import *

OUT = "out/c4b.json"; res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, "w"), indent=1, default=float)
if "vac_tangent_random" not in res:
    rng = np.random.default_rng(5); d = 7; g = Grid(0.5, d, nth=64); rows = []
    for _ in range(20):
        psi = rng.normal(size=d - 1) + 1j * rng.normal(size=d - 1)
        D = np.zeros((d, d), complex); D[1:, 0] = psi; D[0, 1:] = psi.conj()
        FS = F_SLD(D, 0.0); rows.append((g.F_het(D) / FS, g.F_hom_random(D) / FS))
    rows = np.array(rows); res["vac_tangent_random"] = dict(etaQ=[rows[:, 0].min(), rows[:, 0].max()], etaW=[rows[:, 1].min(), rows[:, 1].max()])
    save(); print(res["vac_tangent_random"], flush=True)
for nb in (0.1, 0.5, 1.0, 5.0):
    for d in (10, 12):
        key = f"nb{nb}_d{d}"
        if key in res: continue
        B = herm_basis(d); g = Grid(nb + 0.5, d, nth=2 * d + 8)
        MQ, MW = fisher_matrices(g, B); MS = sld_matrix(B, nb, "sld")
        eQ = eigh(MQ, MS, eigvals_only=True); eW = eigh(MW, MS, eigvals_only=True)
        res[key] = dict(etaW_max=eW.max(), etaQ_max=eQ.max()); save(); print(key, res[key], flush=True)
print("done")
