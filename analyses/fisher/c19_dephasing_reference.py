"""C19 (Sec. III A, Fig. 5): origin of the offset between the prediction of C18 and the observed effect of dephasing.
For each pair (as in C18 and fig_dephasing_sectors), without noise and after 20% dephasing, on the pixel grid of analyses/chernoff:
  xi   Chernoff exponent,
  Fth  quadratic form sum (p_j - p_i)^2 / p_0 with the outcome distribution p_0 of the thermal state of the same mean photon number
       (the chi^2 form of Sec. II; homodyne: mean over the K phases),
  T    the same form with the mean distribution (p_i + p_j)/2 of the pair in place of p_0,
  tv   total-variation distance of the mean distribution of the pair from p_0 (how far the pair is from the Gaussian reference),
  shift  total-variation distance between the mean distributions of the pair without noise and after dephasing; the theory of
         Sec. II assumes a reference state that dephasing leaves unchanged, so shift = 0 is its range of validity.
Result: the change of ln(T_W/T_Q) reproduces the observed change of ln(xi_W/xi_Q); the offset of the thermal form comes from the
reference state, mostly through heterodyne detection.  Needs qsdl on PYTHONPATH and QMI_XMAX=12.  Output: out/c19.json (about 9 min)."""
import sys, json, importlib.util, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ch", str(ROOT / "analyses" / "chernoff" / "chernoff.py")); ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)
from qsdl.noise import apply_channel_noise
from qmi.gen import level_params_eqn
par = level_params_eqn("dephase", 20); K = ch.K; out = {}
def forms(pi, pj, p0, hom):
    if hom:
        pi, pj, p0 = pi.reshape(K, -1), pj.reshape(K, -1), p0.reshape(K, -1)
        Fth = np.mean(np.sum((pj - pi) ** 2 / np.maximum(p0, 1e-300), axis=1)); T = np.mean(np.sum((pj - pi) ** 2 / np.maximum((pi + pj) / 2, 1e-300), axis=1))
        xi = ch.chern_rows(np.log(np.maximum(pi.ravel(), 1e-300)), np.log(np.maximum(pj.ravel(), 1e-300))[None], K)[0]
    else:
        Fth = np.sum((pj - pi) ** 2 / np.maximum(p0, 1e-300)); T = np.sum((pj - pi) ** 2 / np.maximum((pi + pj) / 2, 1e-300))
        xi = ch.chern_rows(np.log(np.maximum(pi, 1e-300)), np.log(np.maximum(pj, 1e-300))[None])[0]
    return float(xi), float(Fth), float(T)
for fam in ("eqn", "ahm"):
    c0 = np.load(ROOT / "analyses" / "chernoff" / "out" / f"M24_{fam}_clean0.npz", allow_pickle=True)
    lab = c0["lab"]; M = c0["CQ"].copy(); M[lab[:, None] == lab[None, :]] = np.inf; j = M.argmin(1)
    rhos, _, _, _ = ch.samples(fam, 24, 2210); R0 = [ch.dm(r) for r in rhos]; R1 = [ch.dm(apply_channel_noise(r, par)) for r in rhos]
    d = R0[0].shape[0]; m = np.arange(d); rows = []
    for a, b in enumerate(j):
        nb = float(np.real(np.trace((R0[a] + R0[b]) @ np.diag(m)))) / 2; q = nb / (nb + 1); th = np.diag((1 - q) * q ** m)
        row = {"lab": int(lab[a]), "nbar": nb}; mean = {}
        for tag, R in (("c", R0), ("d", R1)):
            for det, f, hom in (("Q", ch.q_probs, False), ("W", lambda r: ch.hom_probs(r).ravel(), True)):
                pi, pj, p0 = f(R[a]), f(R[b]), f(th)
                row[f"{det}{tag}"] = forms(pi, pj, p0, hom)
                row[f"tv{det}{tag}"] = float(0.5 * np.abs((pi + pj) / 2 - p0).sum() / (K if hom else 1))     # distance of the pair's mean distribution from the thermal one
                mean[det + tag] = (pi + pj) / 2
        for det, hom in (("Q", False), ("W", True)):                                                          # change of the pair's mean distribution under dephasing
            row[f"shift{det}"] = float(0.5 * np.abs(mean[det + "c"] - mean[det + "d"]).sum() / (K if hom else 1))
        rows.append(row)
        if a % 24 == 0: print(fam, a, flush=True)
    out[fam] = rows
summ = {}
for fam in ("eqn", "ahm"):
    r = out[fam]; g = lambda key, i: np.array([x[key][i] for x in r]); D = {}
    for det in "QW":
        for i, nm in enumerate(("xi", "Fth", "T")): D[det + nm] = np.log(g(det + "d", i) / g(det + "c", i))
    obs, th, own = D["Wxi"] - D["Qxi"], D["WFth"] - D["QFth"], D["WT"] - D["QT"]
    summ[fam] = dict(spearman_thermal=float(spearmanr(obs, th).correlation), spearman_own=float(spearmanr(obs, own).correlation),
                     median_obs=float(np.median(obs)), median_thermal=float(np.median(th)), median_own=float(np.median(own)),
                     **{"median_dln_" + k: float(np.median(v)) for k, v in D.items()})
    print(fam, {k: round(v, 3) for k, v in summ[fam].items()})
json.dump({"summary": summ, "pairs": out}, open(Path(__file__).resolve().parent / "out" / "c19.json", "w"))
