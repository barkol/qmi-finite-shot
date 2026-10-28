"""Coverage of the paper's interval for the paired difference Q-W, on synthetic data.
This self-contained simulation estimates the coverage of the interval used in the paper (z for the test-state term, t for the run term).

Model (logit scale).  A test set of 700 states (7 families x 100) is shared by all networks.  Network of type T
(Q or W), run j, state i of family c succeeds with probability sigmoid(a_Tc + u_i + e_Ti + b_Tj) and the outcome is
a Bernoulli draw: u_i = difficulty common to Q and W (sd s_u), e_Ti = state-specific part (sd s_e), b_Tj = run effect
(sd s_b), a_Tc = family intercept.  The true difference is the population mean over states and runs (Gauss-Hermite).
Interval constructions, applied exactly as in make_numbers.paired_diff to the 5 (or 3) runs per type:
  paper : hypot(1.96 * se_states, t_{0.975,df} * se_runs), se_states from the per-state difference of run-averaged
          outcomes, stratified by family; se_runs and Welch df from the run accuracies;
  single_t : t_{0.975,df} * hypot(se_states, se_runs) (one t quantile for the sum, Welch df of the run term).
Writes results/ci_coverage.json.  Usage: python ci_coverage.py [n_rep] [seed]"""
import json, os, sys, itertools
import numpy as np
from scipy.stats import t as tdist
from numpy.polynomial.hermite_e import hermegauss

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results", "ci_coverage.json")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 20261028)
NF, NPF = 7, 100; Y = np.repeat(np.arange(NF), NPF); NS = NF * NPF
sig = lambda x: 1 / (1 + np.exp(-x))
GX, GW = hermegauss(80); GW = GW / GW.sum()
S_U, S_E = 1.0, 0.7
FAMOFF = np.array([-1.2, -0.6, -0.2, 0.0, 0.3, 0.7, 1.1])      # family spread, same for Q and W
# centre logit of (Q, W) for the four noise cells of the paper at N=100 (accuracies about 96/91, 81/75, 58/49, 83/69)
CENTRES = {"clean": (3.2, 2.4), "loss50": (1.6, 1.2), "loss80": (0.5, 0.0), "deph20": (1.7, 0.9)}


def truth(aq, aw, s_b):
    s = np.sqrt(S_U ** 2 + S_E ** 2 + s_b ** 2)
    f = lambda a: np.mean([(GW * sig(a + FAMOFF[c] + s * GX)).sum() for c in range(NF)])
    return f(aq) - f(aw)


def interval(aq, aw, mq, mw, y):
    """aq, aw: run accuracies; mq, mw: (runs, states) outcomes"""
    di = mq.mean(0) - mw.mean(0)
    se_st = np.sqrt(sum(NPF * di[y == c].var(ddof=1) for c in range(NF))) / NS
    vq, vw = aq.var(ddof=1) / len(aq), aw.var(ddof=1) / len(aw); se_r = np.sqrt(vq + vw)
    df = (vq + vw) ** 2 / (vq ** 2 / (len(aq) - 1) + vw ** 2 / (len(aw) - 1)) if vq + vw > 0 else 1e9
    D = aq.mean() - aw.mean(); tq = tdist.ppf(0.975, df)
    return D, np.hypot(1.959964 * se_st, tq * se_r), tq * np.hypot(se_st, se_r), 1.959964 * np.hypot(se_st, se_r)


res = []
for (cell, (cq, cw)), s_b, k in itertools.product(CENTRES.items(), (0.1, 0.3, 0.5), (5, 3)):
    D0 = truth(cq, cw, s_b); hit = np.zeros(3); width = np.zeros(3)
    for _ in range(NREP):
        u = rng.normal(0, S_U, NS)
        m = []
        for a in (cq, cw):
            lg = a + FAMOFF[Y] + u + rng.normal(0, S_E, NS) + rng.normal(0, s_b, (k, 1))
            m.append((rng.random((k, NS)) < sig(lg)).astype(float))
        D, h1, h2, h3 = interval(m[0].mean(1), m[1].mean(1), m[0], m[1], Y)
        for i, h in enumerate((h1, h2, h3)): hit[i] += abs(D - D0) <= h; width[i] += h
    r = dict(cell=cell, seed_sd_logit=s_b, runs=k, true_diff_pct=100 * D0, n_rep=NREP,
             **{f"cov_{n}": float(hit[i] / NREP) for i, n in enumerate(("paper", "single_t", "all_z"))},
             **{f"halfwidth_pct_{n}": float(100 * width[i] / NREP) for i, n in enumerate(("paper", "single_t", "all_z"))})
    res.append(r); print(f"{cell:7s} sb={s_b} k={k}  true={r['true_diff_pct']:.1f}  cov paper={r['cov_paper']:.3f} single_t={r['cov_single_t']:.3f} z={r['cov_all_z']:.3f}", flush=True)
summ = {f"k{k}": {n: [float(min(r[f"cov_{n}"] for r in res if r["runs"] == k)), float(np.mean([r[f"cov_{n}"] for r in res if r["runs"] == k])), float(max(r[f"cov_{n}"] for r in res if r["runs"] == k))] for n in ("paper", "single_t", "all_z")} for k in (5, 3)}
json.dump(dict(model=__doc__.split("Writes")[0], summary_min_mean_max=summ, scenarios=res), open(OUT, "w"), indent=1)
print(json.dumps(summ))
