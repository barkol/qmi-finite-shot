"""Heterodyne Fisher per phase sector l, thermal reference, as a generalized eigenproblem against F_SLD.
Sector l>0:  F_Q = c^dag M c,  M_nm = (2/(1-q)) (n+m+l)! /((1+q)^{n+m+l+1} sqrt(n!(n+l)! m!(m+l)!))
             F_SLD = sum_n 4|c_n|^2/(p_n+p_{n+l}),  p_n=(1-q)q^n
Sector 0:    F_Q = d^T M0 d, M0_nm = (1/(1-q)) (n+m)!/((1+q)^{n+m+1} n! m!),  F_SLD = sum d_n^2/p_n, constraint sum d_n = 0.
Checked against the grid integrals of fisherlib for small cases. Uses mpmath for large N."""
import json, numpy as np, mpmath as mp
from c4lib import *
mp.mp.dps = 60
def sector_matrices(q, l, N):
    M = mp.matrix(N, N); S = []
    for n in range(N):
        S.append(4 / ((1 - q) * q ** n * (1 + q ** l)) if l > 0 else 1 / ((1 - q) * q ** n))
        for m in range(N):
            if l > 0:
                M[n, m] = 2 / (1 - q) * mp.factorial(n + m + l) / ((1 + q) ** (n + m + l + 1) * mp.sqrt(mp.factorial(n) * mp.factorial(n + l) * mp.factorial(m) * mp.factorial(m + l)))
            else:
                M[n, m] = 1 / (1 - q) * mp.factorial(n + m) / ((1 + q) ** (n + m + 1) * mp.factorial(n) * mp.factorial(m))
    return M, S
def top_ratio(q, l, N):
    M, S = sector_matrices(mp.mpf(q), l, N)
    Sh = mp.diag([1 / mp.sqrt(s) for s in S]); A = Sh * M * Sh
    if l == 0:   # restrict to sum d_n = 0  <=>  sum sqrt(S_n)^{-1}... : constraint vector in scaled coords
        v = mp.matrix([1 / mp.sqrt(s) for s in S]); v = v / mp.norm(v)
        P = mp.eye(N) - v * v.T; A = P * A * P
    ev = mp.eigsy(A)[0]
    return float(max(ev))
# sanity vs grid (nbar=1, l=2, N=4 random c)
nb = 1.0; q = nb / (nb + 1); rng = np.random.default_rng(0); d = 7
c = rng.normal(size=4) + 1j * rng.normal(size=4); D = np.zeros((d, d), complex)
for n in range(4): D[n + 2, n] = c[n]
D = D + D.conj().T; g = Grid(nb + 0.5, d)
M, S = sector_matrices(mp.mpf(q), 2, 4); Mn = np.array(M.tolist(), dtype=complex)
print("sanity l=2:", g.F_het(D), np.real(c.conj() @ Mn @ c), " SLD", F_SLD(D, nb), sum(S[n] * abs(c[n]) ** 2 for n in range(4)))
dd = rng.normal(size=5); dd -= dd.mean(); D0 = np.diag(np.r_[dd, 0, 0]).astype(complex)
M0, S0 = sector_matrices(mp.mpf(q), 0, 5); M0n = np.array(M0.tolist(), dtype=float)
print("sanity l=0:", g.F_het(D0), dd @ M0n @ dd)
out = {}
for nb in (0.1, 0.5, 1.0, 3.0):
    q = nb / (nb + 1); lam = (1 + q) / 2
    row = {}
    for l in range(0, 6):
        row[str(l)] = [top_ratio(q, l, N) for N in (10, 25, 50)]
    out[str(nb)] = dict(lam=lam, sectors=row); print(nb, "lam", round(lam, 5), {k: [round(x, 5) for x in v] for k, v in row.items()}, flush=True)
json.dump(out, open("out/c6.json", "w"), indent=1)
