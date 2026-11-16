"""Order-n blocks for several modes (Sym^n for d x d matrices) and helpers for the two-mode squeezed vacuum.
Conventions: x = (x_a, p_a, x_b, p_b), hbar = 1, vacuum variance 1/2,
TMSV with V = 1/2 [[c 1, s Z],[s Z, c 1]], i.e. |psi_r> = exp[r(a^dag b^dag - a b)]|00> = sech r sum_n tanh^n r |n,n>."""
import itertools
import numpy as np
from functools import lru_cache
from math import factorial

def omega(N):
    return np.kron(np.eye(N), np.array([[0., 1.], [-1., 0.]]))

@lru_cache(None)
def basis(n, d):
    """multi-indices alpha (tuples of length d) with |alpha| = n; orthonormal basis sqrt(n!/alpha!) z^alpha of Sym^n(C^d)."""
    out = [a for a in itertools.product(range(n + 1), repeat=d) if sum(a) == n]
    return out, {a: i for i, a in enumerate(out)}

@lru_cache(None)
def iso(n, d):
    """isometry W_n: Sym^n -> Sym^(n-1) (x) C^d,  |alpha> -> sum_k sqrt(alpha_k/n) |alpha - e_k> (x) |k>."""
    bn, _ = basis(n, d); bm, im = basis(n - 1, d)
    W = np.zeros((len(bm) * d, len(bn)))
    for j, a in enumerate(bn):
        for k in range(d):
            if a[k]:
                b = list(a); b[k] -= 1
                W[im[tuple(b)] * d + k, j] = np.sqrt(a[k] / n)
    return W

def symn(A, n):
    """Sym^n(A): restriction of A^(x)n to the symmetric subspace, for a d x d matrix A."""
    A = np.asarray(A); d = A.shape[0]
    S = np.ones((1, 1), dtype=A.dtype)
    for k in range(1, n + 1):
        W = iso(k, d)
        S = W.T @ np.kron(S, A) @ W
    return S

def sqrtm_sym(V):
    w, U = np.linalg.eigh(V); return (U * np.sqrt(w)) @ U.T

def tmsv_S(r):
    Z = np.diag([1., -1.]); C, S = np.cosh(r), np.sinh(r)
    return np.block([[C * np.eye(2), S * Z], [S * Z, C * np.eye(2)]])

def tmsv_V(r):
    S = tmsv_S(r); return S @ S.T / 2

def lam_full(V, H, Sigma, L=None):
    """Lambda = L^T H^T (H V H^T + Sigma)^-1 H L with L L^T = V (default: symmetric root)."""
    if L is None: L = sqrtm_sym(V)
    return L.T @ H.T @ np.linalg.inv(H @ V @ H.T + Sigma) @ H @ L

def G_pm(V, L=None):
    N = V.shape[0] // 2
    if L is None: L = sqrtm_sym(V)
    Li = np.linalg.inv(L)
    K = Li @ (0.5j * omega(N)) @ Li.T
    return np.eye(2 * N) + K, np.eye(2 * N) - K

def J_n(V, n, L=None):
    Gp, Gm = G_pm(V, L)
    return 0.5 * (symn(Gp, n) + symn(Gm, n))

def eff_full(Mn, Jn, tol=1e-8):
    """eigenvalues of E_n = J^(1/2) M J^(1/2) on the range of J_n; returns (eigs sorted, rank of J_n)."""
    w, U = np.linalg.eigh((Jn + Jn.conj().T) / 2)
    keep = w > tol * w.max()
    R = U[:, keep] * np.sqrt(w[keep])
    E = R.conj().T @ Mn @ R
    return np.sort(np.linalg.eigvalsh((E + E.conj().T) / 2)), int(keep.sum())

def hom_H(tha, thb):
    return np.array([[np.cos(tha), np.sin(tha), 0, 0], [0, 0, np.cos(thb), np.sin(thb)]])

# ---------- reduced description for a pure reference state: E_n = 1/2 [[1, T_n],[T_n^dag, 1]], T_n = E Sym^n(tau)
EP = np.array([1, -1j]) / np.sqrt(2)          # i Omega e_+ = e_+
def tau_of_lambda(Lam):
    """tau_jk = 2 <e_+^(j)| Lambda |e_-^(k)> in the frame where G_+- = 1 +- i Omega (L = S/sqrt2)."""
    N = Lam.shape[0] // 2
    Ep = np.zeros((2 * N, N), complex)
    for j in range(N): Ep[2 * j:2 * j + 2, j] = EP
    return 2 * Ep.conj().T @ Lam @ Ep.conj(), 2 * Ep.conj().T @ Lam @ Ep     # tau, 2A

def tau_local(za, zb, t):
    """closed form for product seeds with Siegel parameters za, zb (heterodyne 0, homodyne exp(2 i theta))."""
    return np.array([[(1 - t * t) * za, t * (1 - za * zb)], [t * (1 - za * zb), (1 - t * t) * zb]]) / (1 - t * t * za * zb)

def eff_from_T(T):
    s = np.linalg.svd(T, compute_uv=False)
    return np.sort(np.concatenate([(1 - s) / 2, (1 + s) / 2]))

def pure_seed_sigma(s, th):
    """single-mode pure seed: variance e^{-2s}/2 along the quadrature at angle th; Siegel parameter tanh(s) e^{2 i th}."""
    u = np.array([np.cos(th), np.sin(th)]); w = np.array([-np.sin(th), np.cos(th)])
    return 0.5 * (np.exp(-2 * s) * np.outer(u, u) + np.exp(2 * s) * np.outer(w, w))
