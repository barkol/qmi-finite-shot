"""Numerical checks of a direct proof of the homodyne bound 1/2 of Corollary 2: (i) int W_{|n><n|} L_k(u) dxdp = kappa^k M_k(n),
(ii) l=0 sector: F_W = sum_k w_k rho^k mu_k^2, (iii) l!=0 sector identity 4 int|g|^2/p0 = F_x(G_R)+F_x(G_I) <= F_SLD,
(iv) sector additivity of F_W."""
import json, numpy as np
from math import comb, factorial
from scipy.special import eval_laguerre, poch
from c4lib import *
res = {}
def meixner(k, n, q):  # M_k(n; beta=1, c=q) = 2F1(-k,-n;1;1-1/q)
    z = 1 - 1 / q
    return sum(poch(-k, j) * poch(-n, j) / (factorial(j) ** 2) * z ** j for j in range(min(k, n) + 1))
for nb in (0.5, 1.0, 3.0):
    v = nb + 0.5; q = nb / (nb + 1); kap = 2 * nb / (2 * nb + 1)
    t, w = np.polynomial.hermite.hermgauss(120); X, P = np.meshgrid(t, t, indexing="ij")
    err = 0.0
    for n in range(6):
        A = np.zeros((n + 1, n + 1)); A[n, n] = 1
        Wpoly = np.real(wigner_poly(A, X, P))          # W * e^{x^2+p^2}
        u = (X ** 2 + P ** 2) / (2 * v)
        for k in range(6):
            val = np.sum(np.outer(w, w) * Wpoly * eval_laguerre(k, u))
            err = max(err, abs(val - kap ** k * meixner(k, n, q)))
    res[f"meixner_identity_nb{nb}"] = err; print("identity max err", nb, err, flush=True)
    # (ii) diagonal sector
    rng = np.random.default_rng(1); d = 7
    dn = rng.normal(size=d); dn -= dn.mean(); D = np.diag(dn)
    p = pth(nb, d); g = Grid(v, d, nth=4)
    FW = g.F_hom_random(D); FS = F_SLD(D, nb)
    rho = 4 * nb * (nb + 1) / (2 * nb + 1) ** 2
    # mu_k = q^{k/2} sum_n d_n M_k(n)   (orthonormal Meixner coefficients of t_n = d_n/p_n)
    mu = [q ** (k / 2) * sum(dn[n] * meixner(k, n, q) for n in range(d)) for k in range(0, 60)]
    Fform = sum(comb(2 * k, k) / 4 ** k * rho ** k * mu[k] ** 2 for k in range(1, 60))
    res[f"diag_nb{nb}"] = dict(FW_direct=FW, FW_formula=Fform, FSLD=FS, sum_mu2=sum(m ** 2 for m in mu[1:]), ratio=FW / FS, bound=0.5 * rho)
    print("diag", nb, res[f"diag_nb{nb}"], flush=True)
    # (iii) l != 0 sector, complex coefficients
    for l in (1, 2, 3):
        c = rng.normal(size=d - l) + 1j * rng.normal(size=d - l)
        G = np.zeros((d, d), complex)
        for n in range(d - l): G[n + l, n] = c[n]
        Da = G + G.conj().T; GR = np.real(G) + np.real(G).T; GI = np.imag(G) + np.imag(G).T
        lhs = 2 * g.F_hom_random(Da)                   # = 4 int |g|^2 / p0
        Fx = lambda A: g.F_theta(A, 0.0)
        res[f"l{l}_nb{nb}"] = dict(two_FW=lhs, FxR_plus_FxI=Fx(GR) + Fx(GI), FSLD=F_SLD(Da, nb),
                                   FSLD_R_plus_I=F_SLD(GR.astype(complex), nb) + F_SLD(GI.astype(complex), nb))
        print("l", l, nb, res[f"l{l}_nb{nb}"], flush=True)
    # (iv) sector additivity
    Dfull = random_traceless(d, rng); tot = g.F_hom_random(Dfull); parts = 0.0
    for l in range(0, d):
        S = np.zeros((d, d), complex)
        for n in range(d - l): S[n + l, n] = Dfull[n + l, n]
        S = S + S.conj().T if l > 0 else S
        parts += g.F_hom_random(S)
    res[f"additivity_nb{nb}"] = dict(total=tot, sum_sectors=parts); print("add", nb, tot, parts, flush=True)
json.dump(res, open("out/c5.json", "w"), indent=1, default=float)
