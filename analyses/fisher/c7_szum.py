"""C3: noise channels act diagonally on Hermite-Gauss modes (n,l). Direct integrals vs formulas."""
import json, numpy as np
from math import factorial, comb
from scipy.signal import fftconvolve
from c4lib import *
out = {}
def loss_op(A, eta, dmax):
    d = A.shape[0]; big = dmax
    Ab = np.zeros((big, big), complex); Ab[:d, :d] = A
    a = np.diag(np.sqrt(np.arange(1, big)), 1); N = np.arange(big)
    res = np.zeros_like(Ab)
    for k in range(big):
        Ek = np.sqrt((1 - eta) ** k / factorial(k)) * np.diag(eta ** (N / 2)) @ np.linalg.matrix_power(a, k)
        res += Ek @ Ab @ Ek.conj().T
    return res
def dephase(A, gam):
    m = np.arange(A.shape[0]); return A * np.exp(-gam * np.subtract.outer(m, m) ** 2 / 2)
def formula_from_b(b, v_old, v_new, mu, nmax):
    lam = v_new / (v_new + 0.5); FW = FQ = 0.0
    for n in range(nmax + 1):
        bn = np.array([b[j, n - j] for j in range(n + 1)])
        FW += mu ** n * bn @ u_avg(n) @ bn; FQ += (lam * mu) ** n * bn @ bn
    return FW, FQ
rng = np.random.default_rng(42); d = 5
for nb in (0.0, 1.0):
    D = random_traceless(d, rng); v = nb + 0.5
    b = hermite_coeffs(D, v, 120)
    # --- dephasing: sector weights e^{-gamma l^2}
    for gam in (0.3, 1.5):
        g = Grid(v, d, nth=360); Dd = dephase(D, gam)
        FWd, FQd = g.F_hom_random(Dd), g.F_het(Dd)
        FWs = FQs = 0.0
        for l in range(d):
            S = np.zeros((d, d), complex)
            for n in range(d - l): S[n + l, n] = D[n + l, n]
            if l: S = S + S.conj().T
            FWs += np.exp(-gam * l * l) * g.F_hom_random(S); FQs += np.exp(-gam * l * l) * g.F_het(S)
        out[f"deph_nb{nb}_g{gam}"] = dict(FW_direct=FWd, FW_sectors=FWs, FQ_direct=FQd, FQ_sectors=FQs)
        print("deph", nb, gam, out[f"deph_nb{nb}_g{gam}"], flush=True)
    # --- loss
    for eta in (0.5, 0.2):
        Dl = loss_op(D, eta, d); vn = eta * nb + 0.5; mu = eta * v / vn
        g = Grid(vn, d, nth=360)
        FWd, FQd = g.F_hom_random(Dl), g.F_het(Dl)
        FWf, FQf = formula_from_b(b, v, vn, mu, 120)
        out[f"loss_nb{nb}_eta{eta}"] = dict(FW_direct=FWd, FW_formula=FWf, FQ_direct=FQd, FQ_formula=FQf)
        print("loss", nb, eta, out[f"loss_nb{nb}_eta{eta}"], flush=True)
    # --- Gaussian noise sigma^2 (phase space convolution)
    for s2 in (0.25, 1.0):
        vn = v + s2; mu = v / vn
        g = Grid(v, d, nx=4001, nq=401, nth=180, L=14)
        ker = np.exp(-g.x ** 2 / (2 * s2)); ker /= ker.sum()
        p0n = lambda: np.exp(-g.x ** 2 / (2 * vn)) / np.sqrt(2 * np.pi * vn)
        FWd = np.mean([np.sum(np.convolve(g.marginal(D, t), ker, mode="same") ** 2 / p0n()) * g.dx for t in g.th])
        q = g.qfun(D); k2 = np.exp(-(g.X ** 2 + g.P ** 2) / (2 * s2)); k2 /= k2.sum()
        qn = fftconvolve(q, k2, mode="same"); Q0n = np.exp(-(g.X ** 2 + g.P ** 2) / (2 * (vn + 0.5))) / (2 * np.pi * (vn + 0.5))
        # the tails of Q0n underflow far outside the state; there qn is convolution round-off, and qn^2/Q0n would
        # blow up (2.9e13 at nbar=0, s2=0.25), so the integral runs over the grid points where Q0n is representable
        m = Q0n > 1e-20 * Q0n.max()
        FQd = np.sum(qn[m] ** 2 / Q0n[m]) * g.dq ** 2
        FWf, FQf = formula_from_b(b, v, vn, mu, 120)
        out[f"gauss_nb{nb}_s2{s2}"] = dict(FW_direct=FWd, FW_formula=FWf, FQ_direct=FQd, FQ_formula=FQf)
        print("gauss", nb, s2, out[f"gauss_nb{nb}_s2{s2}"], flush=True)
json.dump(out, open("out/c7.json", "w"), indent=1, default=float)
