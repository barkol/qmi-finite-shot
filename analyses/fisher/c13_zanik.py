"""C13 (thermalization, Sec. II H):  decay of the Fisher information under thermal damping toward rho0 (same nbar), generator
L = A[(nbar+1) D[a] + nbar D[a^dag]].  Claims: (i) Phi_t(Upsilon_{l,k}) = exp(-(k+l/2) A t) Upsilon_{l,k};
(ii) for Delta = sum c_j Upsilon_j,  F_X(Phi_t Delta) = sum c_j^2 exp(-2(k_j+l_j/2) A t) F_X(Upsilon_j) for X = SLD, Q, W
(no cross terms), so each detection loses information mode by mode and at late times the slowest mode sets F_W/F_Q."""
import sys, json, numpy as np
from scipy.special import hyp2f1, gammaln
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import kron, identity, csr_matrix
sys.path.insert(0, ".")
from fisherlib import Grid

def basis(nb, d, L=3, K=3):
    q = nb / (nb + 1); n = np.arange(d); pi = (1 - q) * q ** n
    ops, lab = [], []
    for l in range(0, L + 1):
        for k in range(0, K + 1):
            if l == 0 and k == 0: continue
            m = np.arange(d - l)
            c = pi[:d - l] * np.exp(0.5 * (gammaln(m + l + 1) - gammaln(m + 1))) * np.array([hyp2f1(-k, -x, l + 1, 1 - 1 / q) for x in m])
            G = np.zeros((d, d), complex)
            for i, ci in enumerate(c): G[i + l, i] = ci
            ops.append(G + G.conj().T); lab.append((l, k, "re"))
            if l > 0: ops.append(1j * (G - G.conj().T)); lab.append((l, k, "im"))
    return ops, lab, pi

def sld_gram(ops, pi):
    S = pi[:, None] + pi[None, :]
    return np.array([[2 * np.real(np.sum(np.conj(A) * B / S)) for B in ops] for A in ops])


def liouvillian(nb, d):
    a = csr_matrix(np.diag(np.sqrt(np.arange(1, d)), 1)); ad = a.T.tocsr(); I = identity(d, format="csr")
    def D(c):   # vec(c rho c^dag - 1/2 {c^dag c, rho}), row-major vec: vec(A rho B) = kron(A, B^T) vec(rho)
        cdc = (c.T.conj() @ c)
        return kron(c, c.conj()) - 0.5 * kron(cdc, I) - 0.5 * kron(I, cdc.T)
    return (nb + 1) * D(a) + nb * D(ad)

out = {}
for nb in (0.5, 1.0):
    d = {0.5: 40, 1.0: 55}[nb]
    ops, lab, pi = basis(nb, d)
    L = liouvillian(nb, d)
    idx = {x: i for i, x in enumerate(lab)}
    # (i) eigen-relation
    eig = []
    for i, (l, k, part) in enumerate(lab):
        v = ops[i].ravel(); w = (L @ v).reshape(d, d)
        lam = -(k + l / 2)
        m = np.abs(ops[i]) > 1e-8 * np.abs(ops[i]).max()
        eig.append(float(np.abs(w - lam * ops[i])[:d - 8, :d - 8].max() / np.abs(ops[i]).max()))
    # (ii) combination
    comb = {(0, 1, "re"): 1.0, (1, 0, "re"): 0.8, (1, 1, "im"): 0.5, (2, 1, "re"): 0.6, (3, 0, "re"): 0.4, (2, 2, "im"): 0.3}
    Delta = sum(c * ops[idx[key]] for key, c in comb.items())
    g = Grid(nb + 0.5, d, nth=96, nq=161)
    def FX(A): return dict(SLD=2 * np.real(np.sum(np.abs(A) ** 2 / (pi[:, None] + pi[None, :]))), Q=g.F_het(A), W=g.F_hom_random(A))
    single = {key: FX(ops[idx[key]]) for key in comb}
    rows = []
    for t in (0.0, 0.5, 1.0, 2.0, 4.0):
        Dt = expm_multiply(L * t, Delta.ravel()).reshape(d, d) if t > 0 else Delta
        F = FX(Dt)
        pred = {X: sum(c ** 2 * np.exp(-2 * (key[1] + key[0] / 2) * t) * single[key][X] for key, c in comb.items()) for X in F}
        rows.append(dict(t=t, **{f"{X}": F[X] for X in F}, **{f"{X}_pred": pred[X] for X in F},
                         **{f"{X}_rel": abs(F[X] / pred[X] - 1) for X in F}, W_over_Q=F["W"] / F["Q"]))
        print(nb, t, {X: f"{F[X]:.6g}/{pred[X]:.6g}" for X in F}, f"W/Q={F['W']/F['Q']:.4f}", flush=True)
    slow = min(comb, key=lambda key: key[1] + key[0] / 2)
    out[str(nb)] = dict(eigen_max_rel=max(eig), rows=rows, slowest_mode=str(slow),
                        slowest_W_over_Q=single[slow]["W"] / single[slow]["Q"])
    print(nb, "eigen max rel", max(eig), "slowest", slow, "W/Q of slowest", single[slow]["W"] / single[slow]["Q"], flush=True)
out["max_rel_dev"] = max(max(r[k] for k in ("SLD_rel", "Q_rel", "W_rel")) for v in out.values() for r in v["rows"])
out["max_eigen_rel"] = max(v["eigen_max_rel"] for k, v in out.items() if k != "max_rel_dev")
json.dump(out, open("out/c13.json", "w"), indent=1, default=float)
