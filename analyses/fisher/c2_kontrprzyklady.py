"""C2: try to break the claims of app. A and the discussion paragraph 'Fisher information near thermal states'.
Resumable: each test stored in out/c2.json as soon as it finishes."""
import json, os, numpy as np
from math import comb, factorial
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh
from fisherlib import *

OUT = "out/c2.json"; os.makedirs("out", exist_ok=True)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, "w"), indent=1, default=float)
def done(k): return k in res

def herm_basis(d):
    B = []
    for i in range(d):
        for j in range(i, d):
            if i == j:
                if i == 0: continue
                M = np.zeros((d, d), complex); M[i, i] = 1; M[0, 0] = -1; B.append(M)   # traceless diagonal
            else:
                M = np.zeros((d, d), complex); M[i, j] = M[j, i] = 1; B.append(M)
                M = np.zeros((d, d), complex); M[i, j] = 1j; M[j, i] = -1j; B.append(M)
    return B

def fisher_matrices(g, B, K=None):
    """Gram matrices of the direct Fisher forms on a basis of traceless Hermitian operators."""
    n = len(B)
    dQ = [g.qfun(b).ravel() for b in B]
    th = g.th if K is None else np.arange(K) * np.pi / K
    dW = [np.concatenate([g.marginal(b, t) / np.sqrt(g.p0_theta(t)) for t in th]) for b in B]
    q0 = g.Q0.ravel()
    MQ = np.array([[np.sum(dQ[i] * dQ[j] / q0) for j in range(n)] for i in range(n)]) * g.dq ** 2
    MW = np.array([[np.dot(dW[i], dW[j]) for j in range(n)] for i in range(n)]) * g.dx / len(th)
    return MQ, MW

# (a) nbar = 0: F_W >= F_Q for every Delta, equality only on span{|d><0|+h.c., i|d><0|+h.c.}
for K in (None, 7, 36, 4, 3):
    key = f"a_vac_K{K}"
    if done(key): continue
    d = 6; B = herm_basis(d); g = Grid(0.5, d, nth=360)
    MQ, MW = fisher_matrices(g, B, K)
    mu, V = eigh(MW, MQ)                      # MW v = mu MQ v
    ent = dict(min_ratio=mu.min(), max_ratio=mu.max(), n_eq=int(np.sum(np.abs(mu - 1) < 1e-6)))
    # check the equality eigenspace: components only on (0,d) and (d,0) entries
    Veq = V[:, np.abs(mu - 1) < 1e-6]
    if Veq.size:
        leak = 0.0
        for c in Veq.T:
            M = sum(ci * b for ci, b in zip(c, B)); M = M / np.linalg.norm(M)
            mask = np.zeros((d, d), bool); mask[0, 1:] = mask[1:, 0] = True
            leak = max(leak, np.linalg.norm(M[~mask]))
        ent["leak_outside_d0"] = leak
    res[key] = ent; save(); print(key, ent, flush=True)

# (b) mode criterion for nbar > 0: heterodyne wins on (n,l) iff C(n,(n-|l|)/2) < ((2nb+1)/(nb+1))^n
if not done("b_criterion"):
    out = {}
    for nb in (1.0, 2.0, 5.0):
        q = (2 * nb + 1) / (nb + 1)
        # central modes |l|<=1: smallest n from which heterodyne never wins again
        wins = [n for n in range(1, 200) if comb(n, (n - (n % 2)) // 2) < q ** n]
        last = max(wins) if wins else 0
        out[str(nb)] = dict(central_het_wins_up_to_n=last, first_n_never=last + 1,
                            edge_always=all(1 < q ** n for n in range(1, 200)))
    # mode (2,0): threshold nbar
    f = lambda nb: ((2 * nb + 1) / (nb + 1)) ** 2 - 2
    from scipy.optimize import brentq
    out["mode20_threshold_nbar"] = brentq(f, 0.01, 5)
    out["vac_no_mode_favours_het"] = all(comb(n, (n - l) // 2) / 2 ** n >= 0.5 ** n
                                         for n in range(1, 60) for l in range(-n, n + 1, 2))
    res["b_criterion"] = out; save(); print("b", out, flush=True)

# (b') global extremes of F_W/F_Q for nbar>0 (no claim, just the range)
for nb in (0.5, 1.0, 2.0):
    key = f"b_range_nb{nb}"
    if done(key): continue
    d = 6; B = herm_basis(d); g = Grid(nb + 0.5, d, nth=360)
    MQ, MW = fisher_matrices(g, B)
    mu = eigh(MW, MQ, eigvals_only=True)
    res[key] = dict(min_ratio=mu.min(), max_ratio=mu.max()); save(); print(key, res[key], flush=True)

# (c) leading order: Chernoff exponent / (eps^2 F / 8) -> 1
def chernoff(p, q, w):
    f = lambda s: np.log(np.sum(w * p ** s * q ** (1 - s)))
    r = minimize_scalar(f, bounds=(1e-4, 1 - 1e-4), method="bounded")
    return -r.fun, r.x

if not done("c_local"):
    rng = np.random.default_rng(7); out = {}
    for nb in (0.0, 1.0):
        d = 5; g = Grid(nb + 0.5, d, nth=72)
        if nb == 0:
            tau = random_traceless(d, rng); tau = tau @ tau; tau /= np.trace(tau)
            rho0 = np.zeros((d, d)); rho0[0, 0] = 1; D = tau - rho0
        else:
            D = random_traceless(d, rng); rho0 = None
        FQ, FW = g.F_het(D), g.F_hom_random(D)
        rowsQ, rowsW = [], []
        for eps in (0.2, 0.05, 0.0125):
            q0 = g.Q0.ravel(); q1 = q0 + eps * g.qfun(D).ravel()
            if q1.min() < 0: rowsQ.append(None); continue
            xi, s = chernoff(q0, q1, np.full(q0.size, g.dq ** 2))
            rowsQ.append((eps, xi / (eps ** 2 * FQ / 8), s))
            # homodyne at random known phase: joint density over (theta, x)
            P0 = np.concatenate([g.p0_theta(t) for t in g.th]); P1 = P0 + eps * np.concatenate([g.marginal(D, t) for t in g.th])
            xiW, sW = chernoff(P0, P1, np.full(P0.size, g.dx / len(g.th)))
            rowsW.append((eps, xiW / (eps ** 2 * FW / 8), sW))
        out[str(nb)] = dict(het=rowsQ, hom=rowsW)
    res["c_local"] = out; save(); print("c", out, flush=True)

# (c') range of the leading order near the vacuum: the same direction as in (c) at small eps.
# Precise exponent: normalized discrete distributions and log1p/expm1, so that xi ~ eps^2 is not lost to round-off.
def chernoff_precise(p, q):
    p = p / p.sum(); q = q / q.sum(); lr = np.log1p(q / p - 1)
    f = lambda s: np.log1p(np.sum(p * np.expm1((1 - s) * lr)))
    r = minimize_scalar(f, bounds=(1e-4, 1 - 1e-4), method="bounded", options=dict(xatol=1e-8))
    return -r.fun, r.x

if not done("c_local_vac"):
    rng = np.random.default_rng(7); d = 5; g = Grid(0.5, d, nth=72)          # same draw as nb=0 in (c)
    tau = random_traceless(d, rng); tau = tau @ tau; tau /= np.trace(tau)
    rho0 = np.zeros((d, d)); rho0[0, 0] = 1; D = tau - rho0
    FQ, FW = g.F_het(D), g.F_hom_random(D); rowsQ, rowsW = [], []
    for eps in (0.0125, 1e-3, 1e-4, 1e-5, 1e-6):
        q0 = g.Q0.ravel(); q1 = q0 + eps * g.qfun(D).ravel()
        xi, s = chernoff_precise(q0, q1); rowsQ.append((eps, xi / (eps ** 2 * FQ / 8), s))
        P0 = np.concatenate([g.p0_theta(t) for t in g.th]); P1 = P0 + eps * np.concatenate([g.marginal(D, t) for t in g.th])
        xiW, sW = chernoff_precise(P0, P1); rowsW.append((eps, xiW / (eps ** 2 * FW / 8), sW))
        print("c'", eps, rowsQ[-1], rowsW[-1], flush=True)
    res["c_local_vac"] = dict(het=rowsQ, hom=rowsW); save()

# (d) squeezed states, closed-form Gaussian Chernoff
def logC(a, b, s):  # ln int N(0,a)^s N(0,b)^(1-s)
    return 0.5 * ((1 - s) * np.log(a) + s * np.log(b)) - 0.5 * np.log(s * b + (1 - s) * a)
def xi_minimize(fun):
    r = minimize_scalar(fun, bounds=(1e-6, 1 - 1e-6), method="bounded"); return -r.fun
def sq_var(r, phi, th):  # quadrature variance at angle th of state squeezed along angle phi
    return 0.5 * (np.exp(-2 * r) * np.cos(th - phi) ** 2 + np.exp(2 * r) * np.sin(th - phi) ** 2)
if not done("d_squeezed"):
    out = {}; K = 36; th = np.arange(K) * np.pi / K
    vac = []
    for r in np.linspace(0.05, 2.0, 40):
        xiQ = xi_minimize(lambda s: logC(1.0, (np.exp(-2 * r) + 1) / 2, s) + logC(1.0, (np.exp(2 * r) + 1) / 2, s))
        xiW = xi_minimize(lambda s: np.mean([logC(0.5, sq_var(r, 0, t), s) for t in th]))
        vac.append((r, xiQ, xiW))
    out["vac_vs_sqvac_Q_gt_W_all"] = all(q > w for _, q, w in vac)
    out["vac_vs_sqvac_min_margin"] = min(q - w for _, q, w in vac)
    xp = {}
    for r in (2.0, 4.0, 8.0, 16.0):
        a, b = (np.exp(-2 * r) + 1) / 2, (np.exp(2 * r) + 1) / 2
        xiQ = xi_minimize(lambda s: logC(a, b, s) + logC(b, a, s))
        xiW = xi_minimize(lambda s: np.mean([logC(sq_var(r, 0, t), sq_var(r, np.pi / 2, t), s) for t in th]))
        th2 = np.array([np.pi / 4, 3 * np.pi / 4])
        xiW2 = xi_minimize(lambda s: np.mean([logC(sq_var(r, 0, t), sq_var(r, np.pi / 2, t), s) for t in th2]))
        xp[str(r)] = dict(ratio_K36=xiW / xiQ, ratio_K2_45deg=xiW2 / xiQ)
    out["x_vs_p"] = xp
    res["d_squeezed"] = out; save(); print("d", out, flush=True)

# (e) even vs odd cat: xi_W/xi_Q vs e^{|alpha|^2}
if not done("e_cats"):
    out = {}
    for al in (1.0, 1.5, 2.0, 2.5):
        d = 50
        c = np.array([al ** k / np.sqrt(float(factorial(k))) for k in range(d)])
        ev = c * (np.arange(d) % 2 == 0); od = c * (np.arange(d) % 2 == 1)
        ev /= np.linalg.norm(ev); od /= np.linalg.norm(od)
        g = Grid(0.5, d, L=12, nx=2001, nq=401, nth=36)
        R1, R2 = np.outer(ev, ev), np.outer(od, od)
        q1, q2 = np.clip(g.qfun(R1).ravel(), 1e-300, None), np.clip(g.qfun(R2).ravel(), 1e-300, None)
        xiQ, _ = chernoff(q1, q2, np.full(q1.size, g.dq ** 2))
        P1 = np.clip(np.concatenate([g.marginal(R1, t) for t in g.th]), 1e-300, None)
        P2 = np.clip(np.concatenate([g.marginal(R2, t) for t in g.th]), 1e-300, None)
        xiW, _ = chernoff(P1, P2, np.full(P1.size, g.dx / len(g.th)))
        out[str(al)] = dict(xiQ=xiQ, xiW=xiW, ratio=xiW / xiQ, ratio_over_e_a2=xiW / xiQ / np.exp(al ** 2))
        res["e_cats"] = out; save(); print("e", al, out[str(al)], flush=True)
print("done")
