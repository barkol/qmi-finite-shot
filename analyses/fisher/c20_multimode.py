"""Checks of the multimode statements of the paper (Theorems 1 and 2 for M modes, Sec. "Two-mode squeezed vacuum").
Conventions: hbar = 1, vacuum variance 1/2, x = (x_1, p_1, ..., x_M, p_M), Omega = direct sum of [[0, 1], [-1, 0]].
A Gaussian measurement is a pair (H, Sigma): outcome Y = H X + N, N ~ N(0, Sigma), with Sigma +- (i/2) H Omega H^T >= 0.
Order-n blocks in the whitened basis z = V^(-1/2) x:
  M_n = Sym^n(Lambda), Lambda = V^(1/2) H^T (H V H^T + Sigma)^(-1) H V^(1/2);  J_n = [Sym^n(G_+) + Sym^n(G_-)]/2,
  G_+- = V^(-1/2) (V +- i Omega/2) V^(-1/2);  efficiencies = eigenvalues of J_n^(1/2) M_n J_n^(1/2) on the range of J_n.
(A) spectrum and rank of J_n against the normal-mode formula (mixed, partly pure and pure states, M = 2, 3).
(B) Theorem 2 for M modes: largest efficiency of every order <= lambda_max(Lambda), equality in order 1; the multiplicity
    of the efficiency 1 for noiseless commuting quadratures; the counterexample without the physicality condition.
(C) pure Gaussian states of two modes: heterodyne detection of all modes, efficiencies (1 +- prod tanh^k_j r_j)/2.
(D) two-mode squeezed vacuum: closed-form spectra (heterodyne of both modes, homodyne of both modes at independent uniform
    phases and at one common uniform phase, one arm, fixed phases, Bell-type measurement, seed equal to the state).
(E) directions at the two-mode squeezed vacuum: squeezing parameter and phase of one mode from the Gaussian formula
    F = tr[(K^-1 dK)^2]/2 for the outcome covariance K (no Hermite orders); Kerr nonlinearity from the two-mode Fock
    expansion of the direction; loss in one arm.
(F) two-mode squeezed thermal state with equal occupations: first order and the change of the common temperature.
Writes out/c20.json. Runs in about one minute."""
import os, json, itertools, numpy as np
from math import comb, factorial
from scipy.linalg import expm
from scipy.optimize import brentq
rng = np.random.default_rng(20261114); out = {}

def om(M): return np.kron(np.eye(M), np.array([[0.0, 1.0], [-1.0, 0.0]]))
_W = {}
def isom(d, n):
    """Isometry of Sym^n(C^d) into (C^d)^(tensor n): columns = normalized symmetrized products of basis vectors."""
    if (d, n) not in _W:
        idx = list(itertools.combinations_with_replacement(range(d), n)); W = np.zeros((d ** n, len(idx)))
        for j, t in enumerate(idx):
            perms = set(itertools.permutations(t))
            for p in perms: W[np.ravel_multi_index(p, (d,) * n), j] = 1 / np.sqrt(len(perms))
        _W[(d, n)] = W
    return _W[(d, n)]
def symn(A, n):
    A = np.asarray(A, complex); d = A.shape[0]; W = isom(d, n); T = W.astype(complex).reshape((d,) * n + (W.shape[1],))
    for ax in range(n): T = np.moveaxis(np.tensordot(A, T, axes=(1, ax)), 0, ax)
    return W.T @ T.reshape(d ** n, -1)
def psqrt(V):
    w, U = np.linalg.eigh(V); return (U * np.sqrt(w)) @ U.T, (U / np.sqrt(w)) @ U.T
def G(V):
    R, Ri = psqrt(V); X = Ri @ om(V.shape[0] // 2) @ Ri; I = np.eye(V.shape[0]); return I + 0.5j * X, I - 0.5j * X
def Jn(V, n): Gp, Gm = G(V); return 0.5 * (symn(Gp, n) + symn(Gm, n))
def Lam(V, H, Sig):
    R, _ = psqrt(V); H = np.atleast_2d(H); return R @ H.T @ np.linalg.inv(H @ V @ H.T + Sig) @ H @ R
def eff(Mn, J, tol=1e-9):
    """Efficiencies on the range of J and the rank of J."""
    w, U = np.linalg.eigh(J); k = w > tol * w.max(); B = U[:, k] * np.sqrt(w[k])
    E = B.conj().T @ Mn @ B; return np.sort(np.linalg.eigvalsh((E + E.conj().T) / 2)), int(k.sum())
def rsymp(M, s=0.5):
    A = rng.normal(size=(2 * M, 2 * M)); return expm(om(M) @ (s * (A + A.T) / 2))
def rstate(M, kind):
    v = 0.5 + rng.exponential(0.6, M)
    if kind == "pure": v[:] = 0.5
    if kind == "partly": v[0] = 0.5
    S = rsymp(M); return S @ np.diag(np.repeat(v, 2)) @ S.T, v
def physical(H, Sig, M): return np.linalg.eigvalsh(Sig + 0.5j * H @ om(M) @ H.T).min() > -1e-10
def rsetting(M):
    """Random physical Gaussian setting (H, Sigma, number of noiseless commuting quadratures or None)."""
    kind = rng.integers(4)
    if kind == 0:                                   # general-dyne detection of all modes, pure or mixed seed
        S = rsymp(M, 0.7); Sig = S @ S.T / 2
        if rng.random() < 0.5: B = rng.normal(size=(2 * M, 2 * M)); Sig = Sig + 0.2 * B @ B.T
        return np.eye(2 * M), Sig, None
    if kind == 1:                                   # d commuting quadratures without noise
        d = rng.integers(1, M + 1); H = rsymp(M, 0.7)[0:2 * d:2, :]; return H, np.zeros((d, d)), d
    if kind == 2:                                   # heterodyne detection of a subset of the modes
        k = rng.integers(1, M + 1); H = np.eye(2 * M)[:2 * k, :]; return H, np.eye(2 * k) / 2, None
    H = np.zeros((M, 2 * M))                        # homodyne detection of every mode at random phases
    for k in range(M): a = rng.uniform(0, np.pi); H[k, 2 * k:2 * k + 2] = np.cos(a), np.sin(a)
    return H, np.zeros((M, M)), M

# (A) spectrum and rank of J_n
devJ, rank_ok = 0.0, True
for M, nmax in ((2, 5), (3, 3)):
    for kind in ("mixed", "partly", "pure"):
        for _ in range(4):
            V, v = rstate(M, kind); tau = 1 / (2 * v); P = int(np.sum(v == 0.5))
            for n in range(1, nmax + 1):
                w = np.sort(np.linalg.eigvalsh(Jn(V, n))); f = []
                for t in itertools.combinations_with_replacement(range(2 * M), n):          # label 2k: plus, 2k+1: minus
                    a = np.prod([1 + tau[i // 2] if i % 2 == 0 else 1 - tau[i // 2] for i in t])
                    b = np.prod([1 - tau[i // 2] if i % 2 == 0 else 1 + tau[i // 2] for i in t]); f.append((a + b) / 2)
                devJ = max(devJ, np.abs(w - np.sort(f)).max())
                rk = 2 * comb(n + 2 * M - P - 1, n) - (comb(n + 2 * M - 2 * P - 1, n) if P < M else 0) if P else comb(n + 2 * M - 1, n)
                rank_ok &= int(np.sum(w > 1e-9)) == rk
out["A_J_dev"], out["A_rank_ok"] = float(devJ), bool(rank_ok)

# (B) Theorem 2 for M modes
exc, gap1, mult_ok, ncase = -1.0, 0.0, True, 0
for M, nmax in ((2, 4), (3, 3)):
    for kind in ("mixed", "partly", "pure"):
        for _ in range(40):
            V, _v = rstate(M, kind); H, Sig, d = rsetting(M); assert physical(H, Sig, M)
            L = Lam(V, H, Sig); top = np.linalg.eigvalsh(L).max(); ncase += 1
            for n in range(1, nmax + 1):
                e, _rk = eff(symn(L, n), Jn(V, n)); exc = max(exc, e.max() - top)
                if n == 1: gap1 = max(gap1, abs(e.max() - top))
                if d is not None: mult_ok &= int(np.sum(e > 1 - 1e-8)) == comb(n + d - 1, n) and bool(np.all((e > 1 - 1e-8) | (e < 1e-8)))
out["B_max_excess"], out["B_order1_gap"], out["B_mult_ok"], out["B_cases"] = float(exc), float(gap1), bool(mult_ok), ncase
V1 = np.eye(2) / 2; L1 = Lam(V1, np.eye(2), 0.1 * np.eye(2))                      # violates the physicality condition
out["B_unphysical_e2"] = float(eff(symn(L1, 2), Jn(V1, 2))[0].max()); assert abs(out["B_unphysical_e2"] - 25 / 18) < 1e-12

# (C) pure two-mode Gaussian states, heterodyne detection of all modes
devC = 0.0
for _ in range(6):
    S = rsymp(2, 0.6); V = S @ S.T / 2; rj = 0.5 * np.log(np.sort(np.linalg.eigvalsh(2 * V))[2:]); t = np.tanh(rj)
    L = Lam(V, np.eye(4), np.eye(4) / 2)
    for n in range(1, 5):
        e, rk = eff(symn(L, n), Jn(V, n)); s = np.array([t[0] ** k * t[1] ** (n - k) for k in range(n + 1)])
        devC = max(devC, np.abs(e - np.sort(np.concatenate([(1 + s) / 2, (1 - s) / 2]))).max()); assert rk == 2 * (n + 1)
out["C_dev"] = float(devC)

# (D) two-mode squeezed vacuum
Z = np.diag([1.0, -1.0])
def Vtmsv(r, v=0.5):
    c, s = np.cosh(2 * r), np.sinh(2 * r); return v * np.block([[c * np.eye(2), s * Z], [s * Z, c * np.eye(2)]])
def Hhom(ta, tb): return np.array([[np.cos(ta), np.sin(ta), 0, 0], [0, 0, np.cos(tb), np.sin(tb)]])
def binom_pair(n, p): b = np.array([comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]); return (b + b[::-1]) / 2
devD, devForm = {}, 0.0
for r in (0.5, 1.0):
    V = Vtmsv(r); t = np.tanh(r); q = t ** 2; nb = np.sinh(r) ** 2; KI, KC = 48, 256; dv = {}
    for n in range(1, 5):
        J = Jn(V, n); pair = np.sort(np.concatenate([np.full(n + 1, (1 + t ** n) / 2), np.full(n + 1, (1 - t ** n) / 2)]))
        Mhet = symn(Lam(V, np.eye(4), np.eye(4) / 2), n)
        ph = np.arange(KI) * np.pi / KI
        Mind = sum(symn(Lam(V, Hhom(a, b), np.zeros((2, 2))), n) for a in ph for b in ph) / KI ** 2
        Mcom = sum(symn(Lam(V, Hhom(0.3 + a, 1.1 + a), np.zeros((2, 2))), n) for a in np.arange(KC) * np.pi / KC) / KC
        Marm = symn(Lam(V, np.eye(4)[:2], np.eye(2) / 2), n)
        Marmh = sum(symn(Lam(V, np.array([[np.cos(a), np.sin(a), 0, 0]]), np.zeros((1, 1))), n) for a in np.arange(KC) * np.pi / KC) / KC
        Mfix = symn(Lam(V, Hhom(0.4, 1.3), np.zeros((2, 2))), n)
        Hb = np.array([[1, 0, -1, 0], [0, 1, 0, 1]]) / np.sqrt(2); Mbell = symn(Lam(V, Hb, np.zeros((2, 2))), n)
        Mself = symn(Lam(V, np.eye(4), V), n)
        zero = np.zeros(n + 1); one = np.ones(n + 1)
        ref = {"het": pair, "hom_indep": pair, "hom_common": pair,
               "arm_het": np.sort(np.concatenate([zero, [(q ** m + q ** (n - m)) / 2 for m in range(n + 1)]])),
               "arm_hom": np.sort(np.concatenate([zero, binom_pair(n, nb / (2 * nb + 1))])),
               "hom_fixed": np.sort(np.concatenate([zero, one])), "bell": np.sort(np.concatenate([zero, one])),
               "seed_state": np.full(2 * n + 2, 0.5)}
        for key, Mn in (("het", Mhet), ("hom_indep", Mind), ("hom_common", Mcom), ("arm_het", Marm), ("arm_hom", Marmh),
                        ("hom_fixed", Mfix), ("bell", Mbell), ("seed_state", Mself)):
            e, rk = eff(Mn, J); assert rk == 2 * (n + 1); dv[key] = max(dv.get(key, 0.0), float(np.abs(e - ref[key]).max()))
        w, U = np.linalg.eigh(J); B = U[:, w > 1e-9 * w.max()]                     # equality of the forms on the range of J_n
        for Mn in (Mind, Mcom): devForm = max(devForm, float(np.abs(B.conj().T @ (Mn - Mhet) @ B).max()))
    devD[f"r{r}"] = dv
out["D_spectra_dev"], out["D_form_dev"] = devD, devForm
out["D_max_dev"] = float(max(max(d.values()) for d in devD.values()))

# (E) directions at the two-mode squeezed vacuum
def Fgauss(V, dV, H, Sig): K = H @ V @ H.T + Sig; X = np.linalg.inv(K) @ (H @ dV @ H.T); return 0.5 * np.trace(X @ X)
def Fhom_uniform(V, dV, K=96):
    ph = np.arange(K) * np.pi / K; return float(np.mean([Fgauss(V, dV, Hhom(a, b), np.zeros((2, 2))) for a in ph for b in ph]))
E = {}; devE = 0.0
for r in (0.5, 1.0):
    V = Vtmsv(r); c, s, t = np.cosh(2 * r), np.sinh(2 * r), np.tanh(r); nb = np.sinh(r) ** 2; I4 = np.eye(4)
    dVr = np.block([[s * np.eye(2), c * Z], [c * Z, s * np.eye(2)]])
    Oa = np.zeros((4, 4)); Oa[:2, :2] = om(1); dVp = Oa @ V + V @ Oa.T
    Pa = np.diag([1.0, 1, 0, 0]); dVl = -0.5 * (Pa @ V + V @ Pa) + 0.5 * Pa
    d = {"r_het": Fgauss(V, dVr, I4, I4 / 2) / 4, "r_hom": Fhom_uniform(V, dVr) / 4, "r_formula": (1 + t ** 2) / 2,
         "r_bell": Fgauss(V, dVr, np.array([[1, 0, -1, 0], [0, 1, 0, 1]]) / np.sqrt(2), np.zeros((2, 2))) / 4,
         "phase_het": Fgauss(V, dVp, I4, I4 / 2) / s ** 2, "phase_hom": Fhom_uniform(V, dVp) / s ** 2, "phase_formula": (1 - t ** 2) / 2,
         "phase_fixed_quarter": Fgauss(V, dVp, Hhom(0.2, np.pi / 2 - 0.2), np.zeros((2, 2))) / s ** 2, "phase_fixed_formula": 1 / np.cosh(2 * r) ** 2,
         "loss_het": Fgauss(V, dVl, I4, I4 / 2), "loss_het_formula": nb / 2,
         "loss_hom": Fhom_uniform(V, dVl), "loss_hom_formula": nb * (1 - 1 / (2 * c)),
         "loss_hom_fixed_xx": Fgauss(V, dVl, Hhom(0, 0), np.zeros((2, 2))), "loss_hom_fixed_formula": nb * (3 * nb + 1)}
    devE = max(devE, abs(d["r_het"] - d["r_formula"]), abs(d["r_hom"] - d["r_formula"]), abs(d["r_bell"] - 1),
               abs(d["phase_het"] - d["phase_formula"]), abs(d["phase_hom"] - d["phase_formula"]),
               abs(d["phase_fixed_quarter"] - d["phase_fixed_formula"]),
               abs(d["loss_het"] / d["loss_het_formula"] - 1), abs(d["loss_hom"] / d["loss_hom_formula"] - 1),
               abs(d["loss_hom_fixed_xx"] / d["loss_hom_fixed_formula"] - 1))
    # Kerr nonlinearity of the mode a: expansion of the direction in the Fock states of the Bogoliubov modes
    NC = 60 if r < 0.75 else 150; C, S = np.cosh(r), np.sinh(r); sq = np.sqrt(np.arange(1, NC))
    def a_(x):  o = np.zeros_like(x); o[:-1, :] = x[1:, :] * sq[:, None]; return o
    def ad_(x): o = np.zeros_like(x); o[1:, :] = x[:-1, :] * sq[:, None]; return o
    def b_(x):  o = np.zeros_like(x); o[:, :-1] = x[:, 1:] * sq[None, :]; return o
    def bd_(x): o = np.zeros_like(x); o[:, 1:] = x[:, :-1] * sq[None, :]; return o
    psi0 = np.zeros((NC, NC), complex); psi0[np.arange(NC), np.arange(NC)] = t ** np.arange(NC) / C
    def bog(ma, mb):                                # U|ma, mb> = (A^dag)^ma (B^dag)^mb |Psi_r>/sqrt(ma! mb!), A^dag = C a^dag - S b
        x = psi0
        for _ in range(ma): x = C * ad_(x) - S * b_(x)
        for _ in range(mb): x = C * bd_(x) - S * a_(x)
        return x / np.sqrt(factorial(ma) * factorial(mb))
    na = np.arange(NC)[:, None]; chi = -1j * (na ** 2 * psi0); chi = chi - np.vdot(psi0, chi) * psi0
    coef = {(ma, n - ma): np.vdot(bog(ma, n - ma), chi) for n in range(1, 5) for ma in range(n + 1)}
    norm2 = float(np.vdot(chi, chi).real); cap = sum(abs(x) ** 2 for x in coef.values())
    num = sum(t ** n * sum((np.conj(coef[(m, n - m)]) * np.conj(coef[(n - m, m)])).real for m in range(n + 1)) for n in range(1, 5))
    w2, w4 = (2 * c - 1) ** 2, s ** 2
    d.update({"kerr_eff": 0.5 + num / (2 * norm2), "kerr_formula": 0.5 - (t ** 2 * w2 + t ** 4 * w4) / (2 * (w2 + w4)),
              "kerr_fsld": 4 * norm2, "kerr_fsld_formula": s ** 2 * (w2 + w4), "kerr_captured": cap / norm2})
    devE = max(devE, abs(d["kerr_eff"] - d["kerr_formula"]), abs(d["kerr_fsld"] / d["kerr_fsld_formula"] - 1), abs(d["kerr_captured"] - 1))
    E[f"r{r}"] = {k: float(np.real(x)) for k, x in d.items()}
out["E_directions"], out["E_max_dev"] = E, float(devE)

# (F) two-mode squeezed thermal state with equal occupations of the normal modes
Fd = {}; devF, order1_min = 0.0, 1.0
def temp_eff(r, nb, K=96):
    v = nb + 0.5; V = Vtmsv(r, v); dV = Vtmsv(r, 1.0); R, Ri = psqrt(V); A = Ri @ dV @ Ri / np.sqrt(2); Gp, Gm = G(V)
    Jfull = 0.5 * (np.kron(Gp, Gp) + np.kron(Gm, Gm)); fs = float(np.real(A.reshape(-1) @ np.linalg.solve(Jfull, A.reshape(-1).astype(complex))))
    return Fgauss(V, dV, np.eye(4), np.eye(4) / 2) / fs, Fhom_uniform(V, dV, K) / fs
for r, nb in ((0.5, 0.1), (0.5, 1.0), (1.0, 0.1), (1.0, 1.0)):
    v = nb + 0.5; V = Vtmsv(r, v); tau = 1 / (2 * v); t = np.tanh(r); mu = np.array([v * np.exp(2 * r) / (v * np.exp(2 * r) + 0.5), v * np.exp(-2 * r) / (v * np.exp(-2 * r) + 0.5)])
    Lh = Lam(V, np.eye(4), np.eye(4) / 2); ph = np.arange(64) * np.pi / 64
    Lw = sum(Lam(V, Hhom(a, b), np.zeros((2, 2))) for a in ph for b in ph) / 64 ** 2
    eh, ew = np.linalg.eigvalsh(Lh), np.linalg.eigvalsh(Lw)
    devF = max(devF, np.abs(eh - np.sort(np.repeat(mu, 2))).max(), np.abs(ew - np.sort(np.repeat([(1 - t) / 2, (1 + t) / 2], 2))).max())
    order1_min = min(order1_min, float(np.linalg.eigvalsh(Lh - Lw).min()))
    th, tw = temp_eff(r, nb)
    devF = max(devF, abs(th - 0.5 * (1 - tau ** 2) * (mu ** 2).sum()), abs(tw - 0.5 * (1 - tau ** 2)))
    Fd[f"r{r}_nb{nb}"] = {"disp_het": float(mu.max()), "disp_hom": float((1 + t) / 2), "temp_het": float(th), "temp_hom": float(tw)}
def thr(r):
    f = lambda nb: sum((((nb + 0.5) * np.exp(sg * 2 * r)) / ((nb + 0.5) * np.exp(sg * 2 * r) + 0.5)) ** 2 for sg in (1, -1)) - 1
    return float(brentq(f, 1e-6, 50))
out["F_cases"], out["F_max_dev"], out["F_order1_min_gap"] = Fd, float(devF), order1_min
out["F_threshold"] = {"r0": thr(0.0), "r0.5": thr(0.5), "r1": thr(1.0)}; assert abs(out["F_threshold"]["r0"] - 1 / np.sqrt(2)) < 1e-9

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "c20.json"), "w"), indent=1)
for k in ("A_J_dev", "A_rank_ok", "B_max_excess", "B_order1_gap", "B_mult_ok", "B_cases", "C_dev", "D_max_dev", "D_form_dev", "E_max_dev", "F_max_dev", "F_order1_min_gap", "F_threshold"):
    print(k, out[k])
print(json.dumps(out["D_spectra_dev"])); print(json.dumps(out["E_directions"], indent=0)); print(json.dumps(out["F_cases"]))
