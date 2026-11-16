"""Independent check of the symmetric-power theorem by direct integration on grids (Fock basis, no formulas of the theory).
Reference: S(r) rho_th S(r)^dag.  Complete orders n = 1..4 (14 real modes S Upsilon S^dag).
(T1) heterodyne;  (T2) homodyne at a uniformly random phase;  (T3) homodyne with phase density proportional to
1 + c cos 2(theta - theta0) (thermal reference).  Compared: block structure across orders and the efficiencies
(eigenvalues of S^-1/2 G S^-1/2) per order against symlib."""
import os, sys, json, numpy as np
from scipy.linalg import expm
from scipy.special import hyp2f1, gammaln
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisherlib import Grid
import symlib as sl
def basis(nb, d, nmax=4):
    q = nb / (nb + 1); pi = (1 - q) * q ** np.arange(d); ops, lab = [], []
    for l in range(0, nmax + 1):
        for k in range(0, nmax // 2 + 1):
            if l + 2 * k > nmax or l + 2 * k == 0: continue
            m = np.arange(d - l)
            c = pi[:d - l] * np.exp(0.5 * (gammaln(m + l + 1) - gammaln(m + 1))) * np.array([hyp2f1(-k, -x, l + 1, 1 - 1 / q) for x in m])
            G = np.zeros((d, d), complex); G[np.arange(d - l) + l, np.arange(d - l)] = c
            ops.append(G + G.conj().T); lab.append(l + 2 * k)
            if l > 0: ops.append(1j * (G - G.conj().T)); lab.append(l + 2 * k)
    return ops, np.array(lab), pi
nb = 0.5; v = nb + 0.5; d = 64; NTH = 120; th = np.arange(NTH) * np.pi / NTH
ops, order, pi = basis(nb, d); rho0 = np.diag(pi).astype(complex)
a = np.diag(np.sqrt(np.arange(1, d + 70)), 1).astype(complex); ad = a.conj().T
g = Grid(v, d, L=17.0, nx=2401, nq=401, nth=NTH)
FS = np.array([2 * np.real(np.sum(np.abs(A) ** 2 / (pi[:, None] + pi[None, :]))) for A in ops]); N = np.sqrt(np.outer(FS, FS))
def grams(r, wth=None):
    S = expm(0.5 * r * (a @ a - ad @ ad))
    def tr(A): B = np.zeros((d + 70, d + 70), complex); B[:d, :d] = A; return (S @ B @ S.conj().T)[:d, :d]
    r0 = tr(rho0); O = [tr(A) for A in ops]
    Q0 = np.real(g.qfun(r0)).ravel(); wQ = g.dq ** 2 / np.sum(Q0 * g.dq ** 2); Q0 = Q0.clip(1e-300)
    DQ = np.array([np.real(g.qfun(A)).ravel() for A in O]); GQ = (DQ * wQ / Q0) @ DQ.T
    wt = np.ones(NTH) if wth is None else wth
    GW = 0
    for t, w in zip(th, wt):
        p0 = np.real(g.marginal(r0, t)).clip(1e-300); D = np.array([np.real(g.marginal(A, t)) for A in O]); GW = GW + w * (D * g.dx / p0) @ D.T / NTH
    return GQ / N, GW / N
def compare(E, pred):
    cross = max(np.abs(E[np.ix_(order == n, order != n)]).max() for n in range(1, 5))
    dev = max(np.abs(np.sort(np.linalg.eigvalsh(E[np.ix_(order == n, order == n)])) - pred(n)).max() for n in range(1, 5))
    return float(cross), float(dev)
out = {}
for r in (0.3, 0.6):
    EQ, EW = grams(r)
    predQ = lambda n: sl.eff(sl.sym(sl.lam(v, 0.5 * np.diag([np.exp(2 * r), np.exp(-2 * r)])), n), v, n)
    t = np.tanh(r)
    def predW(n):   # uniform lab phase = Poisson-kernel density in normal coordinates
        ph = np.arange(720) * np.pi / 720; w = (1 - t ** 2) / (1 + t ** 2 + 2 * t * np.cos(2 * ph))
        return sl.eff(sum(wi * sl.sym(sl.proj(p), n) for p, wi in zip(ph, w)) / 720, v, n)
    out[f"T1_het_r{r}"] = compare(EQ, predQ); out[f"T2_hom_r{r}"] = compare(EW, predW)
    print(f"r={r}: heterodyne cross-order {out[f'T1_het_r{r}'][0]:.1e}, dev {out[f'T1_het_r{r}'][1]:.1e} | homodyne cross-order {out[f'T2_hom_r{r}'][0]:.1e}, dev {out[f'T2_hom_r{r}'][1]:.1e}", flush=True)
c, th0 = 0.7, 0.4
_, EW = grams(0.0, 1 + c * np.cos(2 * (th - th0)))
out["T3_hom_weighted"] = compare(EW, lambda n: sl.eff(sum((1 + c * np.cos(2 * (p - th0))) * sl.sym(sl.proj(p), n) for p in np.arange(720) * np.pi / 720) / 720, v, n))
print("weighted phases, thermal: cross-order %.1e, dev %.1e" % tuple(out["T3_hom_weighted"]), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "c16.json"), "w"), indent=1)
