"""Sharpness: displacement of the thermal state attains both bounds; higher sectors via Delta = (a^dag)^l rho0 + h.c."""
import json, numpy as np
from c4lib import *
out = {}
for nb in (0.5, 1.0, 3.0):
    q = nb / (nb + 1); v = nb + 0.5
    for l in (1, 2, 3):
        rows = []
        for d in (12, 20, 30):
            p = pth(nb, d); a = np.diag(np.sqrt(np.arange(1, d)), 1); ad = a.T
            G = np.linalg.matrix_power(ad, l) @ np.diag(p); D = (G + G.T).astype(complex)
            g = Grid(v, d, nth=4 * l + 8, nq=301)
            FS = F_SLD(D, nb); rows.append((d, g.F_het(D) / FS, g.F_hom_random(D) / FS))
        out[f"nb{nb}_l{l}"] = dict(rows=rows, bound_het=(1 + q ** l) / 2, bound_hom=0.5)
        print(nb, l, [(r[0], round(r[1], 6), round(r[2], 6)) for r in rows], "het bound", round((1 + q ** l) / 2, 6), flush=True)
json.dump(out, open("out/c9.json", "w"), indent=1, default=float)
