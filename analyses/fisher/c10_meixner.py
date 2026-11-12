"""Check of the sector spectrum: in sector l, directions c_n = pi_n sqrt((n+l)!/n!) M_k(n; l+1, q) (Meixner) are
simultaneously orthogonal for F_SLD, F_Q, F_W, with
  F_Q/F_SLD = (1+q^l)/2 q^k,   F_W/F_SLD = (a^l+b^l)/2 C(l+2k,k) (ab)^k,  a=(nb+1)/(2nb+1), b=nb/(2nb+1)."""
import json, sys, numpy as np
from math import comb
from scipy.special import hyp2f1
from c10_diag import mats
def meixner(k, n, beta, c):           # M_k(n; beta, c) = 2F1(-k, -n; beta; 1 - 1/c)
    return np.array([hyp2f1(-k, -x, beta, 1 - 1 / c) for x in n])
out = {}; worst = dict(diag=0.0, off=0.0)
for nb in (0.1, 0.5, 1.0, 2.0, 5.0):
    q = nb / (nb + 1); a, b = (nb + 1) / (2 * nb + 1), nb / (2 * nb + 1)
    N = int(np.ceil(np.log(1e-18) / np.log(q))) + 30 if nb > 0.2 else 60
    N = min(N, 260)
    for l in range(0, 6):
        IW, MQ, S = mats(nb, l, N) if l > 0 else (None, None, None)
        if l == 0: continue       # l = 0 is Lemma 1 of the paper (checked by c5)
        n = np.arange(N); pi = (1 - q) * q ** n
        from scipy.special import gammaln
        base = pi[:N] * np.exp(0.5 * (gammaln(n + l + 1) - gammaln(n + 1)))
        K = 6; C = np.array([base * meixner(k, n, l + 1, q) for k in range(K)])
        GS = C @ np.diag(S) @ C.T; GQ = C @ MQ @ C.T; GW = C @ IW @ C.T
        d = np.sqrt(np.diag(GS)); nS = GS / np.outer(d, d); nQ = GQ / np.outer(d, d); nW = GW / np.outer(d, d)
        off = max(np.abs(nS - np.eye(K)).max(), np.abs(nQ - np.diag(np.diag(nQ))).max(), np.abs(nW - np.diag(np.diag(nW))).max())
        pQ = np.array([(1 + q ** l) / 2 * q ** k for k in range(K)])
        pW = np.array([(a ** l + b ** l) / 2 * comb(l + 2 * k, k) * (a * b) ** k for k in range(K)])
        dg = max(np.abs(np.diag(nQ) / pQ - 1).max(), np.abs(np.diag(nW) / pW - 1).max())
        worst["diag"] = max(worst["diag"], dg); worst["off"] = max(worst["off"], off)
        out[f"nb{nb}_l{l}"] = dict(N=N, FQ_ratio=list(np.diag(nQ)), FW_ratio=list(np.diag(nW)), pred_Q=list(pQ), pred_W=list(pW), max_rel_diag=dg, max_offdiag=off)
        print(f"nb={nb} l={l} N={N}: max rel dev {dg:.1e}, max off-diag {off:.1e}", flush=True)
print("worst", worst); json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "out/c10_meixner.json", "w"), indent=1, default=float)
