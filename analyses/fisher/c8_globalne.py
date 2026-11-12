"""Global section: why xi_W/xi_Q drops under 20% dephasing. Uses existing pair matrices (audit/chernoff/out/M24_*.npz),
regenerates the same states (seed 2210, M=24) only for the sector decomposition of Delta = rho_j - rho_i."""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]          # repository root
sys.path.insert(0, str(ROOT))
OUT = ROOT / "analyses" / "chernoff" / "out"        # M24_*.npz, see MANIFEST_LARGE_FILES.md
def load(fam, cell): return np.load(f"{OUT}/M24_{fam}_{cell}.npz", allow_pickle=True)
def nearest(C, lab):
    M = C.copy(); M[lab[:, None] == lab[None, :]] = np.inf
    j = M.argmin(1); return j, M[np.arange(len(lab)), j]
res = {}
for fam in ("eqn", "ahm"):
    c0, c1 = load(fam, "clean0"), load(fam, "dephase20"); lab = c0["lab"]; names = list(c0["names"])
    jQ0, xQ0 = nearest(c0["CQ"], lab); jW0, xW0 = nearest(c0["CW"], lab)
    jQ1, xQ1 = nearest(c1["CQ"], lab); jW1, xW1 = nearest(c1["CW"], lab)
    r0, r1 = xW0 / xQ0, xW1 / xQ1
    fam_tab = {}
    for c, nm in enumerate(names):
        m = lab == c
        fam_tab[nm] = dict(ratio_clean=float(np.median(r0[m])), ratio_deph=float(np.median(r1[m])),
                           nearest_changed=float(np.mean(lab[jQ0[m]] != lab[jQ1[m]])))
    # fixed pairs: clean nearest-Q partner, ratio of the SAME pair after dephasing
    i = np.arange(len(lab)); rp0 = c0["CW"][i, jQ0] / c0["CQ"][i, jQ0]; rp1 = c1["CW"][i, jQ0] / c1["CQ"][i, jQ0]
    res[fam] = dict(median_ratio_clean=float(np.median(r0)), median_ratio_deph=float(np.median(r1)),
                    fixed_pair_median_clean=float(np.median(rp0)), fixed_pair_median_deph=float(np.median(rp1)),
                    nearest_family_changed=float(np.mean(lab[jQ0] != lab[jQ1])), families=fam_tab)
    print(fam, json.dumps({k: v for k, v in res[fam].items() if k != "families"}, indent=0))
    for nm, t in fam_tab.items(): print(f"   {nm:9s} clean {t['ratio_clean']:.3f} deph {t['ratio_deph']:.3f} nearest changed {t['nearest_changed']:.2f}")
    # sector content of Delta for fixed pairs (regenerate states)
    sys.argv = ["x"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("ch", str(ROOT / "analyses" / "chernoff" / "chernoff.py"))
    ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)
    rhos, lab2, _, _ = ch.samples(fam, 24, 2210); assert (lab2 == lab).all()
    R = [ch.dm(r) for r in rhos]; d = R[0].shape[0]; mm = np.subtract.outer(np.arange(d), np.arange(d))
    fh, f0, f1 = [], [], []
    for a in i:
        D = R[jQ0[a]] - R[a]; w = np.abs(D) ** 2; tot = w.sum()
        f0.append(w[mm == 0].sum() / tot); f1.append(w[np.abs(mm) == 1].sum() / tot); fh.append(w[np.abs(mm) >= 2].sum() / tot)
    fh, f0, f1 = map(np.array, (fh, f0, f1)); dlog = np.log(rp1 / rp0)
    from scipy.stats import spearmanr
    rs = spearmanr(fh, dlog)
    res[fam]["sector"] = dict(median_f0=float(np.median(f0)), median_f1=float(np.median(f1)), median_fhigh=float(np.median(fh)),
                              spearman_fhigh_vs_dlogratio=float(rs.correlation), p=float(rs.pvalue),
                              dlog_median_fhigh_top_tercile=float(np.median(dlog[fh >= np.quantile(fh, 2 / 3)])),
                              dlog_median_fhigh_bottom_tercile=float(np.median(dlog[fh <= np.quantile(fh, 1 / 3)])))
    print("   sector", res[fam]["sector"], flush=True)
json.dump(res, open("out/c8.json", "w"), indent=1)
