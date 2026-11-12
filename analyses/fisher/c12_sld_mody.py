"""Check: in the Hermite-Gauss expansion of Eq. (modes), F_SLD is diagonal too:
F_SLD = sum_{n,l} sigma_{n,l} |b_{n,l}|^2,  sigma = 2 lambda^n / [(1+q^|l|) q^k],  k=(n-|l|)/2.
For a random direction inside one phase sector l all modes have angular momentum +-l, so the sum over one order n
of |b_jk|^2 (Cartesian Hermite coefficients) is |b_{n,l}|^2+|b_{n,-l}|^2."""
import json, numpy as np
from fisherlib import hermite_coeffs, random_traceless
from c4lib import F_SLD
rng = np.random.default_rng(12); out = {}; worst = 0.0
for nb in (0.3, 1.0, 2.0):
    q = nb / (nb + 1); lam = (2 * nb + 1) / (2 * nb + 2); d = 7; nmax = {0.3: 80, 1.0: 240, 2.0: 400}[nb]   # series converges slowly at large q
    for l in range(0, 5):
        for rep in range(3):
            D = np.zeros((d, d), complex)
            if l == 0:
                x = rng.normal(size=d); pi = (1 - q) * q ** np.arange(d); x -= (pi @ x) / pi.sum() * 0  # diagonal
                D = np.diag(x).astype(complex); D -= np.trace(D) / d * np.eye(d) * 0
                D = np.diag(x - x.mean()).astype(complex)
            else:
                c = rng.normal(size=d - l) + 1j * rng.normal(size=d - l)
                for i in range(d - l): D[i + l, i] = c[i]
                D = D + D.conj().T
            b = hermite_coeffs(D, nb + 0.5, nmax)
            orders = np.zeros(nmax + 1)
            for (j, m), val in np.ndenumerate(b):
                if j + m <= nmax: orders[j + m] += abs(val) ** 2
            pred = sum(orders[n] * 2 * lam ** n / ((1 + q ** l) * q ** ((n - l) // 2)) for n in range(l, nmax + 1, 2) if n >= max(l, 1))
            if l == 0: pred = sum(orders[n] * lam ** n / q ** (n // 2) for n in range(2, nmax + 1, 2))   # l=0: (1+q^0)/2 = 1
            ex = F_SLD(D, nb); worst = max(worst, abs(pred / ex - 1))
            out[f"nb{nb}_l{l}_{rep}"] = dict(FSLD=ex, modes=pred)
            print(nb, l, rep, f"F_SLD {ex:.10g}  sum sigma|b|^2 {pred:.10g}  rel {abs(pred/ex-1):.1e}", flush=True)
print("worst", worst); json.dump(dict(cases=out, worst=worst), open("out/c12.json", "w"), indent=1)
