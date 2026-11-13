"""Checks of the bounds of the paper on the order-n blocks (symlib; whitened coordinates, V = v 1, which is general up to a
symplectic change of the quadratures).
(A) Theorem 2: one general-dyne setting, largest efficiency in every order <= lambda_max(Lambda), equality in order 1.
(B) Theorem 3: recorded random settings with the shape (mu1, mu2) of Lambda independent of its phase:
    largest efficiency in order n <= mbar + rho_n dbar, rho_n = max_{1<=m<=n} |w_m|.
(C) homodyne with a phase distribution: <= eta_n + rho_n (1 - eta_n) <= (1 + rho_n)/2.
(D) K equally spaced phases at a thermal state: largest efficiency in order n = max_r sum_{k = r mod K} e_k.
(E) squeezed thermal state, uniform laboratory phase: |w_m| = tanh^m |r|, supremum (1 + tanh|r|)/2 attained in order 1;
    heterodyne: lambda_p.
(F) the bound lambda_max(mean Lambda) is false: homodyne of x and p at the vacuum, order 2."""
import os, json, numpy as np
from math import comb
import symlib as sl
rng = np.random.default_rng(20261113); out = {}
rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
def rand_sigma():
    s, a = rng.uniform(-1.5, 1.5), rng.uniform(0, np.pi)
    Sg = 0.5 * rot(a) @ np.diag([np.exp(2 * s), np.exp(-2 * s)]) @ rot(a).T
    if rng.random() < 0.5: B = rng.normal(size=(2, 2)); Sg = Sg + 0.3 * B @ B.T      # mixed seed (noisy general-dyne)
    return Sg
NMAX = 8
# (A)
exc, gap1 = -1, 0
for _ in range(300):
    v = 0.5 + rng.exponential(1.0); L = sl.lam(v, rand_sigma()); top = np.linalg.eigvalsh(L).max()
    for n in range(1, NMAX + 1):
        e = sl.eff(sl.sym(L, n), v, n).max(); exc = max(exc, e - top)
        if n == 1: gap1 = max(gap1, abs(e - top))
out["A_max_excess"], out["A_order1_gap"] = float(exc), float(gap1)
# (B), (C)
def ebin(v, n):
    p = (1 - 1 / (2 * v)) / 2; b = np.array([comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]); return 0.5 * (b + b[::-1])
excB, tightB, excC = -1, 0, -1
for _ in range(400):
    v = 0.5 + rng.exponential(1.0); J, K = rng.integers(1, 4), rng.integers(1, 6)
    hom = rng.random() < 0.4
    shapes = [(1.0, 0.0)] if hom else [tuple(np.sort(np.linalg.eigvalsh(sl.lam(v, rand_sigma())))[::-1]) for _ in range(J)]
    ps = rng.dirichlet(np.ones(len(shapes))); ph = rng.uniform(0, np.pi, K); pw = rng.dirichlet(np.ones(K))
    mbar = sum(p * (a + b) / 2 for p, (a, b) in zip(ps, shapes)); dbar = sum(p * (a - b) / 2 for p, (a, b) in zip(ps, shapes))
    for n in range(1, NMAX + 1):
        M = sum(p * w * sl.sym(rot(f) @ np.diag(sh) @ rot(f).T, n) for p, sh in zip(ps, shapes) for f, w in zip(ph, pw))
        rho = max(abs(np.sum(pw * np.exp(2j * m * ph))) for m in range(1, n + 1)); e = sl.eff(M, v, n).max()
        excB = max(excB, e - (mbar + rho * dbar)); tightB += abs(e - (mbar + rho * dbar)) < 1e-9
        if hom: eta = ebin(v, n).max(); excC = max(excC, e - (eta + rho * (1 - eta)))
out["B_max_excess"], out["B_tight_cases"], out["C_max_excess"] = float(excB), int(tightB), float(excC)
# (D)
devD = 0
for nb in (0.2, 0.5, 1.0, 3.0):
    v = nb + 0.5
    for K in range(1, 7):
        for n in range(1, 11):
            M = sum(sl.sym(sl.proj(0.3 + j * np.pi / K), n) for j in range(K)) / K; e = ebin(v, n)
            pred = max(e[r::K].sum() for r in range(K)); devD = max(devD, abs(sl.eff(M, v, n).max() - pred))
out["D_max_dev"] = float(devD)
# (E)
devE, excE, devEh = 0, -1, 0
for nb in (0.0, 0.5, 3.0):
    v = nb + 0.5
    for r in (0.2, 0.5, 1.0):
        t = np.tanh(r); ph = (np.arange(2000) + 0.5) * np.pi / 2000; w = (1 - t ** 2) / (1 + t ** 2 + 2 * t * np.cos(2 * ph)); w /= w.sum()
        devE = max(devE, max(abs(abs(np.sum(w * np.exp(2j * m * ph))) - t ** m) for m in range(1, 9)))
        for n in range(1, NMAX + 1):
            e = sl.eff(sum(wi * sl.sym(sl.proj(p), n) for p, wi in zip(ph[::10], w[::10] / w[::10].sum())), v, n).max()
            excE = max(excE, e - (1 + t) / 2)
        lp = np.exp(2 * r) * (2 * nb + 1) / (np.exp(2 * r) * (2 * nb + 1) + 1)
        Lh = sl.lam(v, 0.5 * np.diag([np.exp(2 * r), np.exp(-2 * r)]))       # heterodyne seen in the whitened coordinates
        devEh = max(devEh, abs(max(sl.eff(sl.sym(Lh, n), v, n).max() for n in range(1, NMAX + 1)) - lp))
out["E_fourier_dev"], out["E_hom_excess"], out["E_het_dev"] = float(devE), float(excE), float(devEh)
# (F)
M = 0.5 * (sl.sym(sl.proj(0.0), 2) + sl.sym(sl.proj(np.pi / 2), 2))
out["F_order2_eff"] = float(sl.eff(M, 0.5, 2).max()); out["F_mean_lambda"] = 0.5
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "c17.json"), "w"), indent=1)
for k, x in out.items(): print(k, x)
