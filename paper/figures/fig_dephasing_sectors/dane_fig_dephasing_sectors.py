"""Data layer for fig_dephasing_sectors: per state i (both sets), its clean nearest-Q partner j,
f_high = share of ||rho_j - rho_i||^2 in Fock coherences |m-n|>=2, dlog = ln[(xi_W/xi_Q)_deph / (xi_W/xi_Q)_clean] for the same pair.
pred = the same change predicted by the theory of Sec. II (analyses/fisher/c18_dephasing_prediction.py).
Compute-if-missing -> paper/figures/data/fig_dephasing_sectors.npz. Rebuild the cache: delete it and run this file."""
import sys, os, numpy as np, importlib.util
from pathlib import Path
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[3]                      # repository root
CACHE = str(ROOT / "paper" / "figures" / "data" / "fig_dephasing_sectors.npz")
OUT = str(ROOT / "analyses" / "chernoff" / "out")                # M24_*.npz from chernoff.py --M 24
def build():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("ch", str(ROOT / "analyses" / "chernoff" / "chernoff.py"))
    ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)
    data = {}
    for fam in ("eqn", "ahm"):
        c0 = np.load(f"{OUT}/M24_{fam}_clean0.npz", allow_pickle=True); c1 = np.load(f"{OUT}/M24_{fam}_dephase20.npz", allow_pickle=True)
        lab = c0["lab"]; names = c0["names"]
        M = c0["CQ"].copy(); M[lab[:, None] == lab[None, :]] = np.inf; j = M.argmin(1); i = np.arange(len(lab))
        dlog = np.log((c1["CW"][i, j] / c1["CQ"][i, j]) / (c0["CW"][i, j] / c0["CQ"][i, j]))
        rhos, lab2, _, _ = ch.samples(fam, 24, 2210); assert (lab2 == lab).all()
        R = [ch.dm(r) for r in rhos]; d = R[0].shape[0]; mm = np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
        fh = np.array([(np.abs(R[b] - R[a]) ** 2)[mm >= 2].sum() / (np.abs(R[b] - R[a]) ** 2).sum() for a, b in zip(i, j)])
        rs = spearmanr(fh, dlog)
        data.update({f"{fam}_fh": fh, f"{fam}_dlog": dlog, f"{fam}_lab": lab, f"{fam}_names": names,
                     f"{fam}_rho": rs.correlation, f"{fam}_p": rs.pvalue})
    os.makedirs(os.path.dirname(CACHE), exist_ok=True); np.savez(CACHE, **data)
SHIFT_MAX = 0.15   # a pair is in the range of the theory if dephasing changes its mean heterodyne distribution by at most this total-variation distance
def add_prediction():
    """Merge the prediction of the theory (analyses/fisher/out/c18.json, made by c18_dephasing_prediction.py) into the cache."""
    import json
    c = json.load(open(str(ROOT / "analyses" / "fisher" / "out" / "c18.json"))); z = dict(np.load(CACHE, allow_pickle=True))
    for fam in ("eqn", "ahm"):
        assert np.abs(np.array(c[fam]["obs"]) - z[f"{fam}_dlog"]).max() < 1e-12            # same pairs, same order
        z[f"{fam}_pred"] = np.array(c[fam]["pred"]); z[f"{fam}_rho_pred"] = c[fam]["spearman"]
    c19 = json.load(open(str(ROOT / "analyses" / "fisher" / "out" / "c19.json")))       # range of validity of the prediction (c19_dephasing_reference.py)
    for fam in ("eqn", "ahm"):
        sh = np.array([x["shiftQ"] for x in c19["pairs"][fam]]); z[f"{fam}_shift"] = sh; z[f"{fam}_inrange"] = sh <= SHIFT_MAX
        m = z[f"{fam}_inrange"]; z[f"{fam}_rho_pred_in"] = spearmanr(z[f"{fam}_pred"][m], z[f"{fam}_dlog"][m]).correlation
    z["shift_max"] = SHIFT_MAX
    np.savez(CACHE, **z)
def load():
    if not os.path.exists(CACHE): build()
    if "eqn_inrange" not in np.load(CACHE, allow_pickle=True).files: add_prediction()
    return np.load(CACHE, allow_pickle=True)
if __name__ == "__main__":
    z = load(); print({k: z[k].shape for k in z.files})
    for f in ("eqn", "ahm"): print(f, float(z[f + "_rho"]), float(z[f + "_p"]), z[f + "_dlog"].min(), z[f + "_dlog"].max())
