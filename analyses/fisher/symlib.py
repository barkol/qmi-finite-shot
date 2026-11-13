"""Order-n blocks of the Fisher forms as symmetric powers.  Normal (Williamson) coordinates: the Gaussian reference state has
covariance V = v 1, v = nbar + 1/2 (hbar = 1, vacuum 1/2).  The Hermite-Gauss modes of order n span Sym^n(C^2).
  measurement with Gaussian noise covariance Sigma (heterodyne: 1/2; general-dyne: any Sigma with Sigma + i Omega/2 >= 0;
  homodyne along u: Lambda = |u><u|):      M_n = Sym^n(Lambda),   Lambda = v (v + Sigma)^-1
  quantum (SLD):                            S_n^-1 = [Sym^n(G_+) + Sym^n(G_-)]/2,   G_+- = 1 +- i Omega/(2v)
  recorded mixture of settings:             M_n = E_s Sym^n(Lambda_s)
Efficiencies in order n = eigenvalues of M_n S_n^-1."""
import numpy as np
from math import comb
OM = np.array([[0, 1], [-1, 0]], float)
def sym(A, n):
    """Sym^n(A) in the orthonormal basis e_a = sqrt(C(n,a)) x^a y^(n-a), a = 0..n."""
    A = np.asarray(A, complex); S = np.zeros((n + 1, n + 1), complex)
    for b in range(n + 1):                      # image of x^b y^(n-b): (A00 x + A10 y)^b (A01 x + A11 y)^(n-b)
        p = np.array([1.0 + 0j])
        for _ in range(b): p = np.convolve(p, [A[1, 0], A[0, 0]])        # coefficients in powers of x, ascending
        for _ in range(n - b): p = np.convolve(p, [A[1, 1], A[0, 1]])
        for a in range(n + 1): S[a, b] = p[a] * np.sqrt(comb(n, b) / comb(n, a))
    return S
def Sinv(v, n): return 0.5 * (sym(np.eye(2) + 1j * OM / (2 * v), n) + sym(np.eye(2) - 1j * OM / (2 * v), n))
def lam(v, Sigma): return v * np.linalg.inv(v * np.eye(2) + np.asarray(Sigma, float))
def proj(phi): u = np.array([np.cos(phi), np.sin(phi)]); return np.outer(u, u)
def dgamma(L, n):
    """P (1/n) sum_i L^(i) P on Sym^n: derivative of sym(1 + tL) at t = 0, divided by n."""
    e = 1e-6; return (sym(np.eye(2) + e * np.asarray(L), n) - sym(np.eye(2) - e * np.asarray(L), n)) / (2 * e * n)
def eff(M, v, n):
    w = np.linalg.eigvals(M @ Sinv(v, n)); assert np.abs(w.imag).max() < 1e-8; return np.sort(w.real)
if __name__ == "__main__":
    # thermal reference: Corollary 2 of the paper
    for nb in (0.5, 3.0):
        v = nb + 0.5; q = nb / (nb + 1); kap = 2 * nb / (2 * nb + 1); err = 0
        for n in range(1, 9):
            l = np.abs(n - 2 * np.arange(n + 1)); k = (n - l) // 2
            het = np.sort((1 + q ** l) * q ** k / 2)
            hom = np.sort([(comb(n, int(kk)) * (kap / 2) ** kk * (1 - kap / 2) ** (n - kk) + comb(n, int(kk + ll)) * (kap / 2) ** (kk + ll) * (1 - kap / 2) ** (n - kk - ll)) / 2 for kk, ll in zip(k, l)])
            Mhom = sum(sym(proj(p), n) for p in np.arange(256) * np.pi / 256) / 256
            err = max(err, np.abs(eff(sym(lam(v, 0.5 * np.eye(2)), n), v, n) - het).max(), np.abs(eff(Mhom, v, n) - hom).max())
        print("thermal nb", nb, "max deviation from Corollary 2 (n <= 8):", err)
