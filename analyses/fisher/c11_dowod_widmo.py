"""Step-by-step checks of the proof of the sector spectrum (Theorem 3 candidate).
(1) E[M_k(n; l+1, q) | s] = k!/(l+1)_k L_k^(l)(s/nbar),  n|s ~ Poisson(s).
(2) <beta|Gamma_{l,k}|beta>/Q0(beta) = C * conj(beta)^l L_k^(l)(|beta|^2/(nbar+1))  (single LG mode of order l+2k).
(3) W_Delta/W0 of Delta_{l,k} has Hermite coefficients only of order n = l+2k (fisherlib.hermite_coeffs, independent).
(4) the binomial form and the bounds F_W <= F_SLD/2 (l>=1), with equality only for l=1, k=0."""
import sys, json, numpy as np
from math import comb, factorial, lgamma, exp, log
from scipy.special import hyp2f1, eval_genlaguerre, poch, gammaln
sys.path.insert(0, ".")
from fisherlib import hermite_coeffs
M = lambda k, n, b, c: hyp2f1(-k, -n, b, 1 - 1 / c)
out = {}; w1 = w2 = w3 = 0.0
for nb in (0.3, 1.0, 2.5):
    q = nb / (nb + 1)
    for l in (0, 1, 2, 3, 4):
        for k in (0, 1, 2, 3):
            # (1)
            nn = np.arange(400)
            for s in (0.2, 1.3, 4.0):
                logp = -s + nn * np.log(s) - gammaln(nn + 1)
                lhs = np.sum(np.exp(logp) * np.array([M(k, x, l + 1, q) for x in nn]))
                rhs = factorial(k) / poch(l + 1, k) * eval_genlaguerre(k, l, s / nb)
                w1 = max(w1, abs(lhs - rhs) / max(1, abs(rhs)))
            # (2) direct coherent-state matrix element, Fock cutoff D
            D = 160; n = np.arange(D - l); pi = (1 - q) * q ** np.arange(D)
            c = pi[:D - l] * np.exp(0.5 * (gammaln(n + l + 1) - gammaln(n + 1))) * np.array([M(k, x, l + 1, q) for x in n])
            vals = []
            for beta in (0.4 + 0.3j, 1.1 - 0.7j, 2.0 + 0.5j):
                r2 = abs(beta) ** 2
                # <beta|n+l><n|beta> = e^{-|beta|^2} conj(beta)^{n+l} beta^n / sqrt((n+l)! n!)
                el = np.sum(c * np.exp(-r2 + (n + l) * np.log(abs(beta)) + n * np.log(abs(beta)) - 0.5 * (gammaln(n + l + 1) + gammaln(n + 1)))) * np.exp(-1j * l * np.angle(beta))
                Q0 = (1 - q) * np.exp(-(1 - q) * r2)
                vals.append(el / Q0 / (np.conj(beta) ** l * eval_genlaguerre(k, l, r2 / (nb + 1))))
            vals = np.array(vals); w2 = max(w2, np.abs(vals / vals[0] - 1).max())
            # (3) pure Hermite order of W_Delta/W0 (small cutoff where hermite_coeffs is exact)
            if nb <= 1.0 and l + 2 * k <= 8:
                d = 40; nn2 = np.arange(d - l); pi2 = (1 - q) * q ** np.arange(d)
                cc = pi2[:d - l] * np.exp(0.5 * (gammaln(nn2 + l + 1) - gammaln(nn2 + 1))) * np.array([M(k, x, l + 1, q) for x in nn2])
                G = np.zeros((d, d), complex)
                for i, ci in enumerate(cc): G[i + l, i] = ci
                Dl = G + G.conj().T
                if l == 0: Dl = Dl - np.trace(Dl) * np.diag(pi2) / pi2.sum()   # traceless (k>=1 already is)
                b = hermite_coeffs(Dl, nb + 0.5, 14)
                orders = {}
                for (j, m), val in np.ndenumerate(b):
                    if j + m <= 14: orders[j + m] = orders.get(j + m, 0) + abs(val) ** 2
                tot = sum(orders.values()); main = orders.get(l + 2 * k, 0)
                if tot > 1e-30 and not (l == 0 and k == 0): w3 = max(w3, 1 - main / tot)
print(f"(1) Meixner-Laguerre conditional expectation: max rel err {w1:.1e}")
print(f"(2) heterodyne density ratio is one LG mode: max rel dev {w2:.1e}")
print(f"(3) W_Delta/W0 outside order l+2k (share of |b|^2): max {w3:.1e}")
# (4)
w4 = 0.0; eq = []
for nb in np.linspace(0.01, 10, 60):
    a, b = (nb + 1) / (2 * nb + 1), nb / (2 * nb + 1)
    for l in range(1, 12):
        for k in range(0, 40):
            r = 0.5 * (a ** l + b ** l) * comb(l + 2 * k, k) * (a * b) ** k
            w4 = max(w4, r - 0.5)
            if abs(r - 0.5) < 1e-12: eq.append((l, k))
print(f"(4) max(F_W/F_SLD - 1/2) over l>=1: {w4:.2e}; equality cases: {sorted(set(eq))}")
json.dump(dict(meixner_laguerre=w1, lg_mode=w2, pure_order=w3, bound=w4, equality=sorted(set(eq))), open("out/c11.json", "w"), indent=1)
