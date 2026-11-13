"""Commutation test: in sector l the homodyne form is diagonal in the heterodyne eigenbasis (Corollary 2 of the paper).
Truncated Fock basis (N = 40): exact only for small nbar; at nbar = 2 and 5 the residual commutator is a truncation
effect that decreases with N (nbar = 5, l = 3: 0.139, 0.068, 0.030 for N = 40, 80, 140). See c10_meixner.py for large nbar."""
import numpy as np, mpmath as mp
from math import comb, lgamma, exp, log
from scipy.linalg import eigh
from c10_sektory import psis
def mats(nb, l, N, dx=0.004):
    v = nb + 0.5; X = np.sqrt(2 * (N + l)) + 9; x = np.arange(-X, X + dx / 2, dx)
    P = psis(N + l, x); U = P[l:l + N] * P[:N] * np.exp(x * x / (4 * v))
    IW = 2 * np.sqrt(2 * np.pi * v) * (U @ U.T) * dx
    q = nb / (nb + 1); pi = (1 - q) * q ** np.arange(N + l)
    S = 4 / (pi[:N] + pi[l:l + N])
    n = np.arange(N)
    LG = lambda k: lgamma(k + 1)
    MQ = np.array([[2 / (1 - q) * exp(LG(a + b + l) - (a + b + l + 1) * log(1 + q) - 0.5 * (LG(a) + LG(a + l) + LG(b) + LG(b + l))) for b in n] for a in n])
    return IW, MQ, S
if __name__ == "__main__":
    for nb, l in ((1.0, 3), (2.0, 4), (0.5, 2), (5.0, 3)):
        q = nb / (nb + 1); lam = (2 * nb + 1) / (2 * nb + 2); N = 40
        IW, MQ, S = mats(nb, l, N)
        Sh = np.diag(S ** -0.5); A = Sh @ IW @ Sh; Bq = Sh @ MQ @ Sh
        comm = np.abs(A @ Bq - Bq @ A).max() / max(np.abs(A).max(), np.abs(Bq).max()) ** 2
        wq, V = np.linalg.eigh(Bq); wq = wq[::-1]; V = V[:, ::-1]
        wW = np.array([V[:, i] @ A @ V[:, i] for i in range(8)])     # homodyne in the heterodyne eigenbasis
        k = np.arange(8); nn = l + 2 * k
        pred = np.array([comb(m, kk) / 2 ** m / lam ** m for m, kk in zip(nn, k)])
        print(f"nb={nb} l={l} rel.commutator={comm:.1e}")
        print("   het eig  :", np.round(wq[:8], 6))
        print("   hom in it:", np.round(wW, 6))
        print("   ratio/pred:", np.round(wW / wq[:8] / pred, 6))
