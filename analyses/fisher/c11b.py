import sys, numpy as np
from scipy.special import hyp2f1, gammaln
sys.path.insert(0, ".")
from fisherlib import hermite_coeffs
M = lambda k, n, b, c: hyp2f1(-k, -n, b, 1 - 1 / c)
for nb, l, k in ((1.0, 2, 3), (1.0, 4, 2), (0.3, 3, 1)):
    q = nb / (nb + 1)
    for d in (20, 30, 40, 50):
        n = np.arange(d - l); pi = (1 - q) * q ** np.arange(d)
        cc = pi[:d - l] * np.exp(0.5 * (gammaln(n + l + 1) - gammaln(n + 1))) * np.array([M(k, x, l + 1, q) for x in n])
        G = np.zeros((d, d), complex)
        for i, ci in enumerate(cc): G[i + l, i] = ci
        b = hermite_coeffs(G + G.conj().T, nb + 0.5, 14)
        o = np.zeros(15)
        for (j, m), val in np.ndenumerate(b):
            if j + m <= 14: o[j + m] += abs(val) ** 2
        print(nb, l, k, d, f"outside share {1 - o[l + 2 * k] / o.sum():.1e}")
