"""Direct (formula-free) Fisher information of homodyne and heterodyne detection.

Convention: x = (a + a^dag)/sqrt2, vacuum variance 1/2, thermal variance v = nbar + 1/2.
Homodyne at phase theta: marginal P_theta(x) = <x_theta|A|x_theta>.
Heterodyne: Q(x,p) = <alpha|A|alpha>/(2 pi), alpha = (x + i p)/sqrt2, density in (x,p).
"""
import numpy as np
from math import factorial, comb

def hermite_functions(mmax, x):
    """psi_m(x), m=0..mmax, |psi_0|^2 = N(0,1/2)."""
    psi = np.zeros((mmax + 1, x.size))
    psi[0] = np.pi ** -0.25 * np.exp(-x ** 2 / 2)
    if mmax >= 1:
        psi[1] = np.sqrt(2) * x * psi[0]
    for m in range(1, mmax):
        psi[m + 1] = np.sqrt(2 / (m + 1)) * x * psi[m] - np.sqrt(m / (m + 1)) * psi[m - 1]
    return psi

def hermite_norm(nmax, z):
    """h_n(z) = He_n(z)/sqrt(n!), orthonormal under N(0,1)."""
    h = np.zeros((nmax + 1, z.size))
    h[0] = 1.0
    if nmax >= 1:
        h[1] = z
    for n in range(1, nmax):
        h[n + 1] = (z * h[n] - np.sqrt(n) * h[n - 1]) / np.sqrt(n + 1)
    return h

class Grid:
    def __init__(self, v, dim, L=None, nx=1201, nq=301, nth=720, shift=(0.0, 0.0)):
        self.v, self.dim = v, dim
        L = L or (9.0 * np.sqrt(v) + 6.0 + abs(shift[0]) + abs(shift[1]))
        self.x = np.linspace(-L, L, nx); self.dx = self.x[1] - self.x[0]
        self.psi = hermite_functions(dim - 1, self.x)
        self.th = np.arange(nth) * np.pi / nth
        Lq = 7.0 * np.sqrt(v + 0.5) + 5.0 + abs(shift[0]) + abs(shift[1])
        q = np.linspace(-Lq, Lq, nq); self.dq = q[1] - q[0]
        X, P = np.meshgrid(q, q, indexing="ij")
        self.alpha = (X + 1j * P) / np.sqrt(2)
        self.shift = shift
        x0, p0 = shift
        # reference densities: (displaced) thermal
        self.Q0 = np.exp(-((X - x0) ** 2 + (P - p0) ** 2) / (2 * (v + 0.5))) / (2 * np.pi * (v + 0.5))
        self.X, self.P = X, P

    def p0_theta(self, th):
        x0, p0 = self.shift
        mu = x0 * np.cos(th) + p0 * np.sin(th)
        return np.exp(-(self.x - mu) ** 2 / (2 * self.v)) / np.sqrt(2 * np.pi * self.v)

    def marginal(self, A, th):
        m = np.arange(self.dim)
        ph = np.exp(-1j * np.subtract.outer(m, m) * th)
        M = (A * ph)
        return np.real(np.einsum("ij,ix,jx->x", M, self.psi, self.psi))

    def qfun(self, A):
        a = self.alpha
        m = np.arange(self.dim)
        pw = np.array([a ** k / np.sqrt(float(factorial(k))) for k in m])  # <k|alpha> without gaussian
        val = np.einsum("ij,ixy,jxy->xy", A, np.conj(pw), pw)
        return np.real(val * np.exp(-np.abs(a) ** 2)) / (2 * np.pi)

    def F_theta(self, A, th):
        d = self.marginal(A, th)
        return np.sum(d ** 2 / self.p0_theta(th)) * self.dx

    def F_hom_random(self, A):
        return np.mean([self.F_theta(A, t) for t in self.th])

    def F_hom_K(self, A, K):
        return np.mean([self.F_theta(A, k * np.pi / K) for k in range(K)])

    def F_het(self, A):
        d = self.qfun(A)
        return np.sum(d ** 2 / self.Q0) * self.dq ** 2

def formula_F(grid, A, nmax=80, nth_fourier=None):
    """Mode-weight formula: Hermite moments of the marginals -> circular components.
    F_W = sum_{n,l} 2^-n C(n,k) |b_{n,l}|^2, F_Q = sum_{n,l} lam^n |b_{n,l}|^2, l = n - 2k.
    b_{n,l} obtained from Fourier coefficients of m_n(theta) = sum_k 2^{-n/2} sqrt(C(n,k)) e^{i(n-2k)th} b_{n,k}.
    Only for undisplaced reference."""
    v = grid.v
    lam = v / (v + 0.5)
    nth = nth_fourier or 4 * nmax + 8
    th = np.arange(nth) * 2 * np.pi / nth           # full period for Fourier in theta
    z = grid.x / np.sqrt(v)
    h = hermite_norm(nmax, z)
    m = np.zeros((nmax + 1, nth))
    for t, tt in enumerate(th):
        d = grid.marginal(A, tt)
        m[:, t] = h @ d * grid.dx
    FW = FQ = 0.0
    for n in range(nmax + 1):
        c = np.fft.fft(m[n]) / nth                   # c[q] coefficient of e^{i q th}
        for k in range(n + 1):
            l = n - 2 * k
            coef = c[l % nth]
            wt = float(comb(n, k)) * 2.0 ** (-n)          # mode weight of homodyne
            b = coef / np.sqrt(wt)
            FW += wt * abs(b) ** 2
            FQ += lam ** n * abs(b) ** 2
    return FW, FQ

def random_traceless(dim, rng):
    G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    H = (G + G.conj().T) / 2
    H -= np.trace(H) / dim * np.eye(dim)
    return H / np.linalg.norm(H)

def displace_op(alpha, dim):
    from scipy.linalg import expm
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    return expm(alpha * a.conj().T - np.conj(alpha) * a)

# ---------- stable route: Wigner function -> Hermite coefficients by Gauss-Hermite ----------
from scipy.special import eval_genlaguerre, roots_hermite

def wigner_poly(A, x, p):
    """P(x,p) = W_A(x,p) * exp(x^2+p^2) (a polynomial), W normalized over dx dp."""
    dim = A.shape[0]; r2 = x ** 2 + p ** 2; z = np.sqrt(2) * (x - 1j * p)
    out = np.zeros(np.broadcast(x, p).shape, complex)
    for m in range(dim):
        for n in range(dim):
            if A[m, n] == 0: continue
            if m >= n:
                w = (-1) ** n / np.pi * np.sqrt(factorial(n) / factorial(m)) * z ** (m - n) * eval_genlaguerre(n, m - n, 2 * r2)
            else:
                w = np.conj((-1) ** m / np.pi * np.sqrt(factorial(m) / factorial(n)) * z ** (n - m) * eval_genlaguerre(m, n - m, 2 * r2))
            out = out + A[m, n] * w
    return out

def wigner(A, x, p):
    return wigner_poly(A, x, p) * np.exp(-(x ** 2 + p ** 2))

def hermite_coeffs(A, v, nmax, ngh=None):
    """b_jk = int W_A h_j(x/sqrt v) h_k(p/sqrt v) dx dp, exact by Gauss-Hermite (weight e^{-x^2})."""
    ngh = ngh or (nmax + 2 * A.shape[0] + 20)
    t, w = roots_hermite(ngh)
    X, P = np.meshgrid(t, t, indexing="ij")
    Pol = np.real(wigner_poly(A, X, P))
    h = hermite_norm(nmax, t / np.sqrt(v))          # (nmax+1, ngh)
    Hw = h * w                                       # absorb weights
    return Hw @ Pol @ Hw.T                           # b[j,k]

def u_avg(n, K=None, nth=None):
    """Phase average of u_n u_n^T, u_{n,j} = sqrt(C(n,j)) c^j s^(n-j); K phases in [0,pi) or continuous."""
    if K is None:
        nth = nth or 4 * n + 8; th = np.arange(nth) * np.pi / nth
    else:
        th = np.arange(K) * np.pi / K
    j = np.arange(n + 1)
    sq = np.sqrt(np.array([float(comb(n, jj)) for jj in j]))
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    U = sq * c ** j * s ** (n - j)                   # (nth, n+1)
    return U.T @ U / len(th)

def formula_F2(A, v, nmax, K=None):
    b = hermite_coeffs(A, v, nmax)
    lam = v / (v + 0.5)
    FW = FQ = 0.0; perorder = []
    for n in range(nmax + 1):
        bn = np.array([b[jj, n - jj] for jj in range(n + 1)])
        fw = bn @ u_avg(n, K) @ bn; fq = lam ** n * bn @ bn
        FW += fw; FQ += fq; perorder.append((fw, fq))
    return FW, FQ, perorder
