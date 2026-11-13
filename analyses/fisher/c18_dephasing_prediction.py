"""C18 (Sec. III A, Fig. 5): prediction of the theory of Sec. II for the effect of dephasing on fixed pairs of states.
For each test state rho_i and its closest state rho_j of another family (the pairs of fig_dephasing_sectors), the difference
Delta = rho_j - rho_i is taken as a direction at the thermal state rho0 with the mean photon number of (rho_i + rho_j)/2.
By Corollary 2 and Eq. (noisemodes), dephasing of variance gamma multiplies the modes of angular momentum l by exp(-gamma l^2/2)
and leaves rho0 unchanged, so the predicted change is
    dlog_pred = ln[F_W(D_gamma Delta)/F_Q(D_gamma Delta)] - ln[F_W(Delta)/F_Q(Delta)],  (D_gamma Delta)_{mm'} = exp(-gamma (m-m')^2/2) Delta_{mm'},
with F_Q and F_W (K = 36 phases, as in the test) the chi^2 forms of heterodyne and homodyne detection relative to rho0, computed
by direct integrals (fisherlib.Grid). Compared with the change of ln(xi_W/xi_Q) of the Chernoff exponents (analyses/chernoff).
Needs qsdl on PYTHONPATH (states) and QMI_XMAX=12. Resumable: partial results in out/c18_part.json.  Output: out/c18.json."""
import os, sys, json, importlib.util, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from fisherlib import Grid
GAMMA, K = 0.6, 36
PART = HERE / "out" / "c18_part.json"
def main():
    spec = importlib.util.spec_from_file_location("ch", str(ROOT / "analyses" / "chernoff" / "chernoff.py"))
    ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)
    part = json.load(open(PART)) if PART.exists() else {}
    grids = {}
    for fam in ("eqn", "ahm"):
        c0 = np.load(ROOT / "analyses" / "chernoff" / "out" / f"M24_{fam}_clean0.npz", allow_pickle=True)
        c1 = np.load(ROOT / "analyses" / "chernoff" / "out" / f"M24_{fam}_dephase20.npz", allow_pickle=True)
        lab = c0["lab"]; M = c0["CQ"].copy(); M[lab[:, None] == lab[None, :]] = np.inf; j = M.argmin(1); i = np.arange(len(lab))
        obs = np.log((c1["CW"][i, j] / c1["CQ"][i, j]) / (c0["CW"][i, j] / c0["CQ"][i, j]))
        rhos, lab2, _, _ = ch.samples(fam, 24, 2210); assert (lab2 == lab).all()
        R = [ch.dm(r) for r in rhos]; d = R[0].shape[0]; m = np.arange(d); mm = np.subtract.outer(m, m); damp = np.exp(-GAMMA * mm ** 2 / 2)
        for a, b in zip(i, j):
            key = f"{fam}_{a}"
            if key in part: continue
            D = R[b] - R[a]; nb = float(np.real(np.trace((R[a] + R[b]) @ np.diag(m)))) / 2
            g = grids.setdefault(round(nb, 2), Grid(round(nb, 2) + 0.5, d, nth=K))
            FQ0, FW0, FQ1, FW1 = g.F_het(D), g.F_hom_K(D, K), g.F_het(D * damp), g.F_hom_K(D * damp, K)
            part[key] = dict(nbar=nb, FQ=FQ0, FW=FW0, FQd=FQ1, FWd=FW1, pred=float(np.log((FW1 / FQ1) / (FW0 / FQ0))), obs=float(obs[a]), lab=int(lab[a]),
                             ratio_local=float(FW0 / FQ0), ratio_obs=float(c0["CW"][a, b] / c0["CQ"][a, b]))
            if len(part) % 12 == 0: json.dump(part, open(PART, "w")); print(len(part), key, round(part[key]["pred"], 3), round(part[key]["obs"], 3), flush=True)
        json.dump(part, open(PART, "w"))
    out = {"gamma": GAMMA, "K": K}
    for fam in ("eqn", "ahm"):
        r = [v for k, v in sorted(part.items(), key=lambda kv: int(kv[0].split("_")[1])) if k.startswith(fam)]
        p, o = np.array([x["pred"] for x in r]), np.array([x["obs"] for x in r])
        out[fam] = dict(pred=p.tolist(), obs=o.tolist(), lab=[x["lab"] for x in r], nbar=[x["nbar"] for x in r],
                        spearman=float(spearmanr(p, o).correlation), spearman_p=float(spearmanr(p, o).pvalue), pearson=float(pearsonr(p, o)[0]),
                        sign_agree=float(np.mean(np.sign(p) == np.sign(o))), median_pred=float(np.median(p)), median_obs=float(np.median(o)),
                        ratio_local=[x["ratio_local"] for x in r], ratio_obs=[x["ratio_obs"] for x in r],
                        spearman_ratio=float(spearmanr([x["ratio_local"] for x in r], [x["ratio_obs"] for x in r]).correlation))
        print(fam, {k: (round(v, 3) if isinstance(v, float) else None) for k, v in out[fam].items() if isinstance(v, float)})
    json.dump(out, open(HERE / "out" / "c18.json", "w"), indent=1)
if __name__ == "__main__": main()
