"""Sup over directions in phase sector l of F_W/F_SLD (random-phase homodyne) near a thermal state.
Delta = Gamma + Gamma^dag, Gamma = sum_n c_n |n+l><n|.  Phase average: F_W = 2 int |g|^2/p0, g = sum c_n psi_{n+l} psi_n,
p0 = N(0, v), v = nbar + 1/2.  F_SLD = sum_n 4|c_n|^2/(pi_n + pi_{n+l}).  Generalized eigenproblem, real symmetric."""
import numpy as np, json, sys
from scipy.linalg import eigh
def psis(nmax, x):
    P = np.zeros((nmax + 1, x.size)); P[0] = np.pi ** -0.25 * np.exp(-x * x / 2)
    if nmax: P[1] = np.sqrt(2) * x * P[0]
    for n in range(1, nmax): P[n + 1] = np.sqrt(2 / (n + 1)) * x * P[n] - np.sqrt(n / (n + 1)) * P[n - 1]
    return P
def sup_hom(nb, l, N, dx=0.004):
    v = nb + 0.5; X = np.sqrt(2 * (N + l)) + 9; x = np.arange(-X, X + dx / 2, dx)
    P = psis(N + l, x); U = P[l:l + N] * P[:N] * np.exp(x * x / (4 * v))       # u_n = psi_{n+l} psi_n / sqrt(p0) * (2 pi v)^{-1/4}
    I = 2 * np.sqrt(2 * np.pi * v) * (U @ U.T) * dx
    q = nb / (nb + 1); pi = (1 - q) * q ** np.arange(N + l)
    S = np.diag(4 / (pi[:N] + pi[l:l + N]))
    w, V = eigh(I, S)
    return w[-1], V[:, -1]
if __name__ == "__main__":
    out = {}
    # check against grid results of the probe: nb=1, l=2, N=12 -> 0.277160 ; l=3 N=12 -> 0.180271
    print("check", sup_hom(1.0, 2, 12)[0], sup_hom(1.0, 3, 12)[0], sup_hom(0.5, 4, 18)[0])
    for nb in (0.1, 0.5, 1.0, 2.0, 5.0):
        a, b = (nb + 1) / (2 * nb + 1), nb / (2 * nb + 1)
        for l in range(1, 7):
            vals = [sup_hom(nb, l, N)[0] for N in (20, 40, 80, 160)]
            out[f"nb{nb}_l{l}"] = dict(sup=vals, edge=0.5 * (a ** l + b ** l), het=(1 + (nb / (nb + 1)) ** l) / 2)
            print(f"nb={nb} l={l}", " ".join(f"{s:.6f}" for s in vals), f"| edge {0.5*(a**l+b**l):.6f} het {(1+(nb/(nb+1))**l)/2:.4f}", flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "out/c10.json", "w"), indent=1)
