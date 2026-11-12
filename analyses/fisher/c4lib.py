import numpy as np
from fisherlib import *
def pth(nb, d):
    m = np.arange(d)
    return (1 / (nb + 1)) * (nb / (nb + 1)) ** m if nb > 0 else (m == 0).astype(float)

def F_SLD(D, nb):
    p = pth(nb, D.shape[0]); S = p[:, None] + p[None, :]
    m = S > 0
    if np.any(np.abs(D[~m]) > 1e-12): return np.inf     # non-regular direction at a pure state
    return 2 * np.sum(np.abs(D[m]) ** 2 / S[m])

def F_QC(D, nb):
    p = pth(nb, D.shape[0]); S = (np.sqrt(p)[:, None] + np.sqrt(p)[None, :]) ** 2
    m = S > 0
    if np.any(np.abs(D[~m]) > 1e-12): return np.inf
    return 4 * np.sum(np.abs(D[m]) ** 2 / S[m])

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

def sld_matrix(B, nb, fn):
    n = len(B); M = np.zeros((n, n))
    p = pth(nb, B[0].shape[0])
    S = (p[:, None] + p[None, :]) if fn == "sld" else (np.sqrt(p)[:, None] + np.sqrt(p)[None, :]) ** 2
    c = 2.0 if fn == "sld" else 4.0
    for i in range(n):
        for j in range(n):
            M[i, j] = c * np.real(np.sum(np.conj(B[i]) * B[j] / S))
    return M

