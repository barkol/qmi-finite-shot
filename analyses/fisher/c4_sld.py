"""C4: efficiency of heterodyne and homodyne detection relative to the quantum limits, locally around
(thermal) reference states. Resumable: results in out/c4.json, saved after each block.

F_SLD(Delta) = 2 sum_{ij} |Delta_ij|^2 / (p_i + p_j)            (thermal rho0 diagonal in Fock, p_i>0 terms)
Best single-copy local Chernoff exponent  = eps^2 F_SLD / 8.
Collective (quantum Chernoff) local exponent = (eps^2/2) sum_{ij} |Delta_ij|^2/(sqrt p_i + sqrt p_j)^2
    =: eps^2 F_QC / 8  (checked numerically below against -min_s ln Tr rho^s sigma^(1-s)).
"""
import json, os, numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from fisherlib import *

OUT = "out/c4.json"; os.makedirs("out", exist_ok=True)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, "w"), indent=1, default=float)

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

def mpow(H, s):
    w, V = np.linalg.eigh(H); w = np.clip(w, 0, None)
    return (V * w ** s) @ V.conj().T

def qcb(rho, sig):
    f = lambda s: np.log(np.real(np.trace(mpow(rho, s) @ mpow(sig, 1 - s))))
    r = minimize_scalar(f, bounds=(1e-4, 1 - 1e-4), method="bounded"); return -r.fun

# --- 0. sanity: displacement of the vacuum, both detections at 1/2 of F_SLD
if "sanity_displacement" not in res:
    d = 4; D = np.zeros((d, d)); D[1, 0] = D[0, 1] = 1.0
    g = Grid(0.5, d)
    res["sanity_displacement"] = dict(FQ=g.F_het(D), FW=g.F_hom_random(D), FSLD=F_SLD(D, 0.0), FQC=F_QC(D, 0.0))
    save(); print("sanity", res["sanity_displacement"], flush=True)

# --- 1. check the local quantum Chernoff formula numerically (nbar=1, random Delta)
if "check_qc_formula" not in res:
    rng = np.random.default_rng(3); d = 5; big = 80; nb = 1.0
    D = random_traceless(d, rng); Db = np.zeros((big, big), complex); Db[:d, :d] = D
    rho = np.diag(pth(nb, big)); rho /= np.trace(rho)
    rows = []
    for eps in (0.05, 0.0125, 0.003):
        rows.append((eps, qcb(rho, rho + eps * Db) / (eps ** 2 * F_QC(D, nb) / 8)))
    res["check_qc_formula"] = rows; save(); print("qc formula", rows, flush=True)

# --- 2. vacuum, tangent directions |d><0|+h.c. and i|d><0|+h.c.: efficiencies per d
if "vac_edge" not in res:
    out = {}
    for d in range(1, 9):
        D = np.zeros((d + 1, d + 1), complex); D[d, 0] = D[0, d] = 1.0
        g = Grid(0.5, d + 1, nth=4 * d + 8)
        FQ, FW, FS, FC = g.F_het(D), g.F_hom_random(D), F_SLD(D, 0.0), F_QC(D, 0.0)
        out[str(d)] = dict(FQ=FQ, FW=FW, FSLD=FS, eta_Q=FQ / FS, eta_W=FW / FS, eta_Q_vs_QC=FQ / FC)
        print("vac d", d, out[str(d)], flush=True)
    res["vac_edge"] = out; save()

# --- 3. thermal references: extreme efficiencies over all directions (generalized eigenproblem)
from c4lib import herm_basis, fisher_matrices, sld_matrix   # no exec of other scripts (side effects!)

for nb in (0.5, 1.0, 2.0, 5.0):
    for d in (4, 6, 8):
        key = f"thermal_nb{nb}_d{d}"
        if key in res: continue
        B = herm_basis(d); g = Grid(nb + 0.5, d, nth=120)
        MQ, MW = fisher_matrices(g, B)
        MS = sld_matrix(B, nb, "sld"); MC = sld_matrix(B, nb, "qc")
        eQ = eigh(MQ, MS, eigvals_only=True); eW = eigh(MW, MS, eigvals_only=True)
        eQc = eigh(MQ, MC, eigvals_only=True); eWc = eigh(MW, MC, eigvals_only=True)
        # random directions: typical efficiencies
        rng = np.random.default_rng(11); typ = []
        for _ in range(200):
            c = rng.normal(size=len(B))
            typ.append(((c @ MQ @ c) / (c @ MS @ c), (c @ MW @ c) / (c @ MS @ c)))
        typ = np.array(typ)
        res[key] = dict(etaQ_min=eQ.min(), etaQ_max=eQ.max(), etaW_min=eW.min(), etaW_max=eW.max(),
                        etaQ_QC_max=eQc.max(), etaW_QC_max=eWc.max(),
                        etaQ_median=float(np.median(typ[:, 0])), etaW_median=float(np.median(typ[:, 1])))
        save(); print(key, res[key], flush=True)
print("done")
