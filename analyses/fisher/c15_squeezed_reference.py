"""C15 (Sec. II F, squeezed thermal reference by direct integrals; not quoted in the paper): squeezed thermal reference state rho0' = S(r) rho_th S(r)^dag, nbar = 0.5.
Any zero-mean one-mode Gaussian state has this form, so this covers squeezed vacuum with thermal noise as well.
F_SLD is unitarily covariant and stays diagonal in the transformed modes S Upsilon S^dag with the same values.
Heterodyne detection and homodyne detection at a uniformly random phase are not covariant under squeezing: their Fisher
forms acquire off-diagonal elements in these modes and the bounds lambda and 1/2 of the thermal state fail.
Nine modes (l <= 2, k <= 1).  Output: out/c15.json.  Run time about 1 min."""
import json, numpy as np
from scipy.linalg import expm
from scipy.special import hyp2f1, gammaln
from fisherlib import Grid

def basis(nb, d, L=2, K=1):
    """Upsilon_{l,k} = Gamma + Gamma^dag and i(Gamma - Gamma^dag), Gamma = (a^dag)^l M_k(n; l+1, q) rho0."""
    q = nb / (nb + 1); pi = (1 - q) * q ** np.arange(d); ops, lab = [], []
    for l in range(0, L + 1):
        for k in range(0, K + 1):
            if l == 0 and k == 0: continue
            m = np.arange(d - l)
            c = pi[:d - l] * np.exp(0.5 * (gammaln(m + l + 1) - gammaln(m + 1))) * np.array([hyp2f1(-k, -x, l + 1, 1 - 1 / q) for x in m])
            G = np.zeros((d, d), complex); G[np.arange(d - l) + l, np.arange(d - l)] = c
            ops.append(G + G.conj().T); lab.append((l, k, "re"))
            if l > 0: ops.append(1j * (G - G.conj().T)); lab.append((l, k, "im"))
    return ops, lab, pi

nb = 0.5; d = 60; NTH = 96; th = np.arange(NTH) * np.pi / NTH
ops, lab, pi = basis(nb, d); rho0 = np.diag(pi).astype(complex)
a = np.diag(np.sqrt(np.arange(1, d + 60)), 1).astype(complex); ad = a.conj().T
g = Grid(nb + 0.5, d, L=16.0, nx=2001, nq=361, nth=NTH)
FS = np.array([2 * np.real(np.sum(np.abs(A) ** 2 / (pi[:, None] + pi[None, :]))) for A in ops])
def gram(r):
    S = expm(0.5 * r * (a @ a - ad @ ad))
    def tr(A): B = np.zeros((d + 60, d + 60), complex); B[:d, :d] = A; return (S @ B @ S.conj().T)[:d, :d]
    r0 = tr(rho0); O = [tr(A) for A in ops]
    Q0 = np.real(g.qfun(r0)).ravel(); wQ = g.dq ** 2 / np.sum(Q0 * g.dq ** 2); Q0 = Q0.clip(1e-300)
    P0 = np.array([np.real(g.marginal(r0, t)) for t in th]).ravel().clip(1e-300); wP = g.dx / NTH
    DQ = np.array([np.real(g.qfun(A)).ravel() for A in O]); DP = np.array([np.array([np.real(g.marginal(A, t)) for t in th]).ravel() for A in O])
    GQ = (DQ * wQ / Q0) @ DQ.T; GW = (DP * wP / P0) @ DP.T; N = np.sqrt(np.outer(FS, FS))
    off = lambda G: float(np.abs(G / np.sqrt(np.outer(np.diag(G), np.diag(G))) - np.eye(len(G))).max())
    return dict(offQ=off(GQ), offW=off(GW), effQ=(np.diag(GQ) / FS).tolist(), effW=(np.diag(GW) / FS).tolist(),
                maxeffQ=float(np.linalg.eigvalsh(GQ / N).max()), maxeffW=float(np.linalg.eigvalsh(GW / N).max()))
out = {"nbar": nb, "labels": [list(map(str, x)) for x in lab], "lambda": (2 * nb + 1) / (2 * nb + 2)}
for r in (0.0, 0.2, 0.5):
    o = out[str(r)] = gram(r)
    print(f"r={r}: offdiag Q {o['offQ']:.3f} W {o['offW']:.3f} | largest efficiency on the span: Q {o['maxeffQ']:.3f} W {o['maxeffW']:.3f}", flush=True)
json.dump(out, open("out/c15.json", "w"), indent=1, default=float)
