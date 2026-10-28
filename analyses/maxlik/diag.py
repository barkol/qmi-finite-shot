"""Diagnostyka fizyczna rho_ML (zdarzenia ev): wiernosc z prawdziwym rho (Uhlmann), czystosc, <n> ML wobec prawdziwego; wg rodzin.
  QMI_XMAX=12 PYTHONPATH=. python analyses/maxlik/diag.py base -> analyses/maxlik/diag_<set>.json"""
import sys, json, glob
from pathlib import Path
import numpy as np
from scipy.linalg import sqrtm
st = sys.argv[1]; test = "data_eq/qmi/test/sweep" if st == "base" else "data_eq_ahm/ahm/test/sweep"; mld = f"analyses/maxlik/ml_{st}/ev"
out = {}
for f in sorted(glob.glob(mld + "/*.npz")):
    c = Path(f).stem; ch, lv, _ = c.split("_"); d = np.load(f); rm = d["rho"].astype(complex); D = rm.shape[1]
    files = sorted(Path(test).glob(f"part_{ch}_{int(lv):03d}_*.npz")); R = np.concatenate([np.load(x)["rho"] for x in files]).astype(complex)
    fam = np.concatenate([[x.stem.split("_")[-1]] * len(np.load(x)["label"]) for x in files])
    F, pu, nm, nt = [], [], [], []
    for a, b in zip(rm, R):
        bt = b[:D, :D]; s = sqrtm(bt); F.append(float(np.real(np.trace(sqrtm(s @ a @ s))) ** 2))
        pu.append(float(np.real(np.trace(a @ a)))); nm.append(float(np.real(np.trace(a @ np.diag(np.arange(D)))))); nt.append(float(np.real(np.trace(b @ np.diag(np.arange(len(b)))))))
    F, pu, nm, nt = map(np.array, (F, pu, nm, nt)); ptrue = np.real(np.einsum("nij,nji->n", R, R))
    out[c] = {k: dict(F=float(np.median(F[fam == k])), purity_ml=float(pu[fam == k].mean()), purity_true=float(ptrue[fam == k].mean()),
                      n_ml=float(nm[fam == k].mean()), n_true=float(nt[fam == k].mean())) for k in np.unique(fam)}
    out[c]["all"] = dict(F=float(np.median(F)), purity_ml=float(pu.mean()), purity_true=float(ptrue.mean()), n_ml=float(nm.mean()), n_true=float(nt.mean()))
    print(c, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out[c].items() if k == "all"}, flush=True)
json.dump(out, open(f"analyses/maxlik/diag_{st}.json", "w"), indent=1)
