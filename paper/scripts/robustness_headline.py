#!/usr/bin/env python3
"""Robustness of the headline Q-W comparison.

Reads  results/ci.json, results/ci_ahm.json         (paper's seed-level intervals, qmi/x8/pary.py)
       results/pary/*.npz, results_ahm/pary/*.npz   (per-seed, per-state correctness, 5 and 3 seeds)
       results/eval_shots_cv5.json                  (5-fold cross-validation, base set)
Writes results/robustness.json and results/numbers_robust.tex (keys rob_*).
Run from anywhere:  python scripts/robustness_headline.py [results_dir]

Step 0 reproduces numbers_holm.tex (holm_p_*, holm_nsig_*) and aborts if they differ.
Analyses: (a) paper, (b) CV folds, (c) exact seed permutation, (d) single t on total se,
(e) larger family, plus equivalence (TOST) for N = 1e3, 1e4. The margin of 5 points is a choice."""
import itertools, json, math, os, re, sys
import numpy as np
from scipy import stats, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
R = os.path.abspath(R)
RA = os.path.join(R, "..", "results_ahm")
MARGIN = 0.05
CONDS = [("clean", "loss", 0), ("deph20", "dephase", 20), ("loss50", "loss", 50), ("loss80", "loss", 80)]
NS = (100, 1000, 10000)

# ---------------------------------------------------------------- helpers
def pval(c, mode="paper"):
    """Smallest alpha at which the interval around mean excludes zero (CI inversion, as holm_rule.py)."""
    m = abs(c["mean"]); s1, s2, df = c["se_states"], c["se_seeds"], c["df"]
    if m == 0: return 1.0
    if mode == "paper": f = lambda a: math.hypot(stats.norm.ppf(1 - a / 2) * s1, stats.t.ppf(1 - a / 2, df) * s2) - m
    elif mode == "tall": f = lambda a: stats.t.ppf(1 - a / 2, df) * math.hypot(s1, s2) - m
    if f(1e-14) < 0: return 1e-14
    if f(0.999999) > 0: return 1.0
    return optimize.brentq(f, 1e-14, 0.999999)

def holm(ps, m_total=None):
    """Step-down Holm; ps dict key->p. m_total > len(ps) pads the family with p=1 cells."""
    items = sorted(ps.items(), key=lambda kv: kv[1]); m = m_total or len(items); adj = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); adj[k] = run
    return adj

def perm_p(a, b):
    """Exact two-sided permutation p of the difference of means (all C(n,k) relabelings)."""
    v = np.r_[a, b]; k = len(a); D = a.mean() - b.mean(); c = t = 0
    for idx in itertools.combinations(range(len(v)), k):
        msk = np.zeros(len(v), bool); msk[list(idx)] = True
        d = v[msk].mean() - v[~msk].mean(); t += 1; c += abs(d) >= abs(D) - 1e-12
    return c / t

def tost(c, margin=MARGIN):
    """TOST with the paper's mixed interval. Returns (p_tost, lo90, hi90). Equivalent iff p_tost < 0.05."""
    d, s1, s2, df = c["mean"], c["se_states"], c["se_seeds"], c["df"]
    if s1 == 0 and s2 == 0: return (0.0 if abs(d) < margin else 1.0), d, d
    hw = lambda a: math.hypot(stats.norm.ppf(1 - a) * s1, stats.t.ppf(1 - a, df) * s2)   # one-sided level a
    def one(side):  # alpha at which d + side*hw(a) just reaches +-margin
        gap = margin - side * d
        if gap <= 0: return 1.0
        f = lambda a: hw(a) - gap
        if f(1e-14) < 0: return 1e-14
        if f(0.5) > 0: return 0.5
        return optimize.brentq(f, 1e-14, 0.5)
    return max(one(+1), one(-1)), d - hw(0.05), d + hw(0.05)

def fp(p):  # p value as LaTeX text, three decimals as in numbers_holm.tex
    return "$<0.001$" if p < 0.0005 else f"{p:.3f}"
def fpp(x):  # signed percentage points
    return f"{100 * x:+.1f}".replace("-", "$-$")
def fpp_abs(x): return f"{100 * x:.1f}"

# ---------------------------------------------------------------- load
A = json.load(open(os.path.join(R, "ci.json"))); B = json.load(open(os.path.join(R, "ci_ahm.json")))
SETS = {"base": A, "ahm": B}
def ci(set_, N, tag, diff="dQW"): return SETS[set_][f"rule_N{N}_{tag}_{diff}"]

def seed_acc(set_, ch, lv, N):
    d = np.load(os.path.join(R if set_ == "base" else RA, "pary", f"{ch}_{lv}_N{N}.npz"))
    M = d["correct"].astype(float); nm = [str(s) for s in d["names"]]
    q = np.array([s.startswith("clf_Q") for s in nm]); w = np.array([s.startswith("clf_W") for s in nm])
    return M[q].mean(1), M[w].mean(1)

cv = json.load(open(os.path.join(R, "eval_shots_cv5.json")))
def cv_diff(ch, lv, N):
    q = np.array([cv[f"clf_Q_cvshots_f{f}"][ch][str(N)][str(lv)]["acc"] for f in range(5)])
    w = np.array([cv[f"clf_W_cvshots_f{f}"][ch][str(N)][str(lv)]["acc"] for f in range(5)])
    return q, w

keys, out = {}, {}
def K(name, val): keys["rob_" + name] = str(val)

# ---------------------------------------------------------------- step 0: reproduce paper
fam = {}  # 24-cell Q-W family
for s in ("base", "ahm"):
    for N in NS:
        for tag, _, _ in CONDS: fam[(s, N, tag)] = ci(s, N, tag)
p_paper = {k: pval(c) for k, c in fam.items()}
adj_paper = holm(p_paper)
ref = dict(re.findall(r"csname res@(\w+)\\endcsname\{([^}]*)\}", open(os.path.join(R, "numbers_holm.tex")).read()))
chk = {"holm_p_deph20": adj_paper[("base", 100, "deph20")], "holm_p_loss80": adj_paper[("base", 100, "loss80")],
       "holm_p_loss50": adj_paper[("base", 100, "loss50")], "holm_p_clean": adj_paper[("base", 100, "clean")]}
for k, v in chk.items():
    assert f"{v:.3f}" == ref[k], (k, v, ref[k])
assert str(sum(v < .05 for v in adj_paper.values())) == ref["holm_nsig_QW"]
print("step 0 OK: reproduced", {k: f"{v:.3f}" for k, v in chk.items()}, "nsig", ref["holm_nsig_QW"])
K("repro_ok", "yes")

# ---------------------------------------------------------------- (b) cross-validation, base set, 12 cells
cvp, cvstat = {}, {}
for N in NS:
    for tag, ch, lv in CONDS:
        q, w = cv_diff(ch, lv, N); d = q - w
        if np.allclose(d, 0): p = 1.0; pw = 1.0
        else:
            p = stats.ttest_1samp(d, 0).pvalue; pw = stats.ttest_ind(q, w, equal_var=False).pvalue
            if np.isnan(p): p = 1.0
            if np.isnan(pw): pw = 1.0
        cvp[("base", N, tag)] = p
        cvstat[("base", N, tag)] = dict(mean=d.mean(), sd=d.std(ddof=1), nfold_pos=int((d > 0).sum()), nfold_neg=int((d < 0).sum()),
                                        p_paired=p, p_welch=pw, diffs=d.tolist())
cv_adj12 = holm(cvp)
# family of 24: Ahmed has no CV; its cells enter with the paper's raw p (hybrid), or padded with p = 1 (stricter)
hyb = dict(cvp); hyb.update({k: v for k, v in p_paper.items() if k[0] == "ahm"})
cv_adj24 = holm(hyb)
cv_adj24pad = holm(cvp, m_total=24)
cvw = {k: v["p_welch"] for k, v in cvstat.items()}
cvw_adj12 = holm(cvw); cvw_adj24pad = holm(cvw, m_total=24)

# ---------------------------------------------------------------- (c) exact permutation over seeds, 24 cells
perm = {}; permstat = {}
for s in ("base", "ahm"):
    for N in NS:
        for tag, ch, lv in CONDS:
            a, b = seed_acc(s, ch, lv, N)
            c = ci(s, N, tag)
            if a.std() == 0 and b.std() == 0 and a.mean() == b.mean(): p = 1.0
            else: p = perm_p(a, b)
            perm[(s, N, tag)] = p
            permstat[(s, N, tag)] = dict(k=len(a), pmin=2 / math.comb(2 * len(a), len(a)),
                                         separated=bool(a.min() > b.max() or a.max() < b.min()), aQ=a.tolist(), aW=b.tolist())
perm_adj = holm(perm)

# ---------------------------------------------------------------- (d) single t on total se, 24 cells
p_tall = {k: pval(c, "tall") for k, c in fam.items()}
tall_adj = holm(p_tall)

# ---------------------------------------------------------------- (e) larger family: all pairwise contrasts
big_paper, big_tall = {}, {}
pat = re.compile(r"rule_N(100|1000|10000)_[a-z0-9]+_d(QW|QS|SW|SWe)$|rule_Ninf_[a-z0-9]+_dSWe$")
for s, d in SETS.items():
    for k, c in d.items():
        if isinstance(c, dict) and "se_states" in c and pat.match(k):
            big_paper[(s, k)] = pval(c); big_tall[(s, k)] = pval(c, "tall")
bigA, bigT = holm(big_paper), holm(big_tall)
nbig = len(big_paper)

# ---------------------------------------------------------------- sign consistency at N=100
sign = {}
for s in ("base", "ahm"):
    for tag, ch, lv in CONDS:
        a, b = seed_acc(s, ch, lv, 100)
        sign[(s, tag)] = dict(sep=bool(a.min() > b.max()), minQ=float(a.min()), maxW=float(b.max()))

# ---------------------------------------------------------------- (2) equivalence, N = 1e3, 1e4
equiv = {}
for s in ("base", "ahm"):
    for N in (1000, 10000):
        for tag, _, _ in CONDS:
            c = ci(s, N, tag); p, lo90, hi90 = tost(c)
            equiv[(s, N, tag)] = dict(mean=c["mean"], lo95=c["lo"], hi95=c["hi"], lo90=lo90, hi90=hi90, p_tost=p,
                                      equiv=bool(p < 0.05), within95=bool(c["lo"] > -MARGIN and c["hi"] < MARGIN))
n_eq = sum(v["equiv"] for v in equiv.values()); n_eq95 = sum(v["within95"] for v in equiv.values())

# ---------------------------------------------------------------- keys
for tag in ("clean", "deph20", "loss50", "loss80"):
    for s, pre in (("base", ""), ("ahm", "ahm_")):
        k = (s, 100, tag); c = ci(s, 100, tag)
        K(f"{pre}{tag}_mean", fpp(c["mean"]))
        K(f"{pre}{tag}_lo", fpp(c["lo"])); K(f"{pre}{tag}_hi", fpp(c["hi"]))
        K(f"{pre}{tag}_p_paper", fp(adj_paper[k]))
        K(f"{pre}{tag}_praw_paper", f"{p_paper[k]:.1e}")
        K(f"{pre}{tag}_p_perm_raw", f"{perm[k]:.4f}")
        K(f"{pre}{tag}_p_perm_adj", fp(perm_adj[k]))
        K(f"{pre}{tag}_p_tall", fp(tall_adj[k]))
        K(f"{pre}{tag}_p_big_paper", fp(bigA[(s, f"rule_N100_{tag}_dQW")]))
        K(f"{pre}{tag}_p_big_tall", fp(bigT[(s, f"rule_N100_{tag}_dQW")]))
        K(f"{pre}{tag}_sep", "yes" if sign[(s, tag)]["sep"] else "no")
    st = cvstat[("base", 100, tag)]
    K(f"{tag}_cv_mean", fpp(st["mean"])); K(f"{tag}_cv_sd", fpp_abs(st["sd"]))
    K(f"{tag}_cv_praw", f"{st['p_paired']:.3f}")
    K(f"{tag}_cv_p12", fp(cv_adj12[("base", 100, tag)]))
    K(f"{tag}_cv_p24", fp(cv_adj24[("base", 100, tag)]))
    K(f"{tag}_cv_p24pad", fp(cv_adj24pad[("base", 100, tag)]))
    K(f"{tag}_cv_welch_p12", fp(cvw_adj12[("base", 100, tag)]))
    K(f"{tag}_cv_welch_p24pad", fp(cvw_adj24pad[("base", 100, tag)]))
    K(f"{tag}_cv_npos", st["nfold_pos"])
K("nseeds_base", 5); K("nseeds_ahm", 3); K("nfold", 5)
K("pmin_perm_base", f"{2 / math.comb(10, 5):.4f}"); K("pmin_perm_ahm", f"{2 / math.comb(6, 3):.2f}")
K("nsig_perm24", sum(v < .05 for v in perm_adj.values()))
K("nsig_tall24", sum(v < .05 for v in tall_adj.values()))
K("nsig_cv12", sum(v < .05 for v in cv_adj12.values()))
K("nsig_cv24", sum(v < .05 for v in cv_adj24.values()))
K("nsig_big_paper", sum(v < .05 for v in bigA.values()))
K("nsig_big_tall", sum(v < .05 for v in bigT.values()))
K("nbig", nbig); K("ncell", 24); K("ncell_base", 12)
assert all(sign[("base", t)]["sep"] for t, _, _ in CONDS), "text claims complete separation of seeds at N=100 (base set)"
assert all(cvstat[("base", 100, t)]["nfold_pos"] == 5 for t, _, _ in CONDS), "text claims Q-W>0 in every fold at N=100"
K("sep_all_base", "yes" if all(sign[("base", t)]["sep"] for t, _, _ in CONDS) else "no")
K("cv_allpos_all", "yes" if all(cvstat[("base", 100, t)]["nfold_pos"] == 5 for t, _, _ in CONDS) else "no")
# permutation: largest family at which the headline stays below 0.05 at the minimal p
K("perm_mmax", int(0.05 // (2 / math.comb(10, 5))))
# equivalence
K("margin_pp", int(MARGIN * 100))
K("equiv_n", n_eq); K("equiv_ncell", len(equiv)); K("equiv95_n", n_eq95)
for s in ("base", "ahm"):
    for N in (1000, 10000):
        e = [t for t, _, _ in CONDS if equiv[(s, N, t)]["equiv"]]
        K(f"equiv_{s}_N{N}_n", len(e))
        for tag, _, _ in CONDS:
            v = equiv[(s, N, tag)]; pre = "" if s == "base" else "ahm_"
            K(f"{pre}N{N}_{tag}_ci", f"[{fpp(v['lo95'])}, {fpp(v['hi95'])}]")
            K(f"{pre}N{N}_{tag}_mean", fpp(v["mean"]))
            K(f"{pre}N{N}_{tag}_ptost", fp(v["p_tost"]))
            K(f"{pre}N{N}_{tag}_equiv", "yes" if v["equiv"] else "no")
# worst-case half-width among the not-equivalent cells
ne = [(k, v) for k, v in equiv.items() if not v["equiv"]]
_ne = sorted((k[0], k[1], k[2]) for k, v in equiv.items() if not v["equiv"])
assert _ne == sorted([("base", 1000, "loss80"), ("ahm", 1000, "loss50"), ("ahm", 1000, "loss80"), ("ahm", 10000, "loss80")]), _ne
assert equiv[("base", 10000, "loss80")]["equiv"] and not equiv[("ahm", 10000, "loss80")]["equiv"]
assert [k for k, v in cv_adj12.items() if v < .05] == [("base", 100, "loss50")], "text: only 50% loss passes in CV-12"
assert not any(v < .05 for v in perm_adj.values()) and tall_adj[("base", 100, "deph20")] < .05
K("noteq_list", ", ".join(f"{'A' if k[0]=='ahm' else 'B'}-$10^{int(math.log10(k[1]))}$-{k[2]}" for k, v in ne))

# ---------------------------------------------------------------- write
sk = lambda k: "|".join(map(str, k))
out = dict(
    note="all p adjusted by Holm unless 'raw'; family 24 = 2 sets x N in {1e2,1e3,1e4} x 4 conditions; margin %.2f" % MARGIN,
    reproduced_holm={k: float(v) for k, v in chk.items()},
    paper=dict(raw={sk(k): v for k, v in p_paper.items()}, holm24={sk(k): v for k, v in adj_paper.items()}),
    cv=dict(stats={sk(k): v for k, v in cvstat.items()}, holm12={sk(k): v for k, v in cv_adj12.items()},
            holm24_hybrid_ahmed_paper_p={sk(k): v for k, v in cv_adj24.items()},
            holm24_padded_p1={sk(k): v for k, v in cv_adj24pad.items()},
            welch_holm12={sk(k): v for k, v in cvw_adj12.items()}, welch_holm24_padded={sk(k): v for k, v in cvw_adj24pad.items()}),
    permutation=dict(raw={sk(k): v for k, v in perm.items()}, holm24={sk(k): v for k, v in perm_adj.items()},
                     stats={sk(k): v for k, v in permstat.items()}),
    single_t=dict(raw={sk(k): v for k, v in p_tall.items()}, holm24={sk(k): v for k, v in tall_adj.items()}),
    big_family=dict(m=nbig, holm_paper={sk(k): v for k, v in bigA.items()}, holm_tall={sk(k): v for k, v in bigT.items()}),
    sign_N100={sk(k): v for k, v in sign.items()},
    equivalence=dict(margin=MARGIN, cells={sk(k): v for k, v in equiv.items()}),
)
json.dump(out, open(os.path.join(R, "robustness.json"), "w"), indent=1)
with open(os.path.join(R, "numbers_robust.tex"), "w") as fh:
    fh.write("% generated by scripts/robustness_headline.py\n")
    for k, v in sorted(keys.items()):
        fh.write("\\expandafter\\def\\csname res@%s\\endcsname{%s}\n" % (k, v))

hl = ("base", 100, "deph20")
print("headline  paper %.4f | CV12 %.3f CV24hyb %.3f CV24pad %.3f | perm raw %.4f adj %.3f | tall %.4f | big paper %.4f tall %.4f (m=%d)" % (
    adj_paper[hl], cv_adj12[hl], cv_adj24[hl], cv_adj24pad[hl], perm[hl], perm_adj[hl], tall_adj[hl],
    bigA[("base", "rule_N100_deph20_dQW")], bigT[("base", "rule_N100_deph20_dQW")], nbig))
for tag, _, _ in CONDS:
    k = ("base", 100, tag)
    print(f"{tag:7s} paper {adj_paper[k]:.4f} cv12 {cv_adj12[k]:.3f} cv24 {cv_adj24[k]:.3f} pad {cv_adj24pad[k]:.3f} perm {perm_adj[k]:.3f} tall {tall_adj[k]:.4f} bigP {bigA[('base','rule_N100_'+tag+'_dQW')]:.4f} bigT {bigT[('base','rule_N100_'+tag+'_dQW')]:.4f} cv npos {cvstat[k]['nfold_pos']} sep {sign[('base',tag)]['sep']}")
for tag, _, _ in CONDS:
    k = ("ahm", 100, tag)
    print(f"AHM {tag:7s} paper {adj_paper[k]:.4f} perm raw {perm[k]:.3f} tall {tall_adj[k]:.4f} bigP {bigA[('ahm','rule_N100_'+tag+'_dQW')]:.4f} bigT {bigT[('ahm','rule_N100_'+tag+'_dQW')]:.4f} sep {sign[('ahm',tag)]['sep']}")
print("nsig: perm24", sum(v < .05 for v in perm_adj.values()), "tall24", sum(v < .05 for v in tall_adj.values()), "cv12", sum(v < .05 for v in cv_adj12.values()), "big", sum(v < .05 for v in bigA.values()), sum(v < .05 for v in bigT.values()))
print("equiv", n_eq, "of", len(equiv), "; within95", n_eq95)
for k, v in equiv.items(): print(" ", k, "mean %+.4f 95%%[%+.4f,%+.4f] 90%%[%+.4f,%+.4f] pTOST %.3g %s" % (v["mean"], v["lo95"], v["hi95"], v["lo90"], v["hi90"], v["p_tost"], v["equiv"]))
