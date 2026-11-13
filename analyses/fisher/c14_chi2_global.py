"""C14 (Sec. II G, arbitrary states against a thermal state): the quadratic forms of the paper are chi^2 divergences from the
thermal state rho0 for ANY state rho, not only for a small change.
(A) constants of the elementary inequalities for sqrt(1+x), x >= -1, used in the Bhattacharyya sandwich;
(B) displacement of rho0 by alpha of any size: heterodyne xi_Q = |a|^2/(4(nb+1)); homodyne at a random recorded phase
    xi_W = u - ln I0(u), u = |a|^2/(4(2nb+1)); quantum Chernoff exponent |a|^2 (sqrt(nb+1)-sqrt(nb))^2;
(C) mixtures rho_eps = (1-eps) rho0 + eps sigma with nine pure states sigma: chi^2(P_eps||P_0) = eps^2 F(sigma-rho0) exactly,
    Bhattacharyya sandwich 1 - chi2/8 - (3/8) E0|X|^3 <= BC <= 1 - chi2/8 + E0 X^3/16 (X = delta/f0), xi <= 2B, and the
    ratio xi_W/xi_Q of the Chernoff exponents along the path against its limit chi2_W/chi2_Q;
    lower bound on the number of events N >= ln(1+4(1-2 delta)^2)/ln(1+chi2).
Output: out/c14.json.  Run time about 2 min."""
import json, numpy as np
from scipy.special import i0e
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from fisherlib import Grid
out = {}
x = np.concatenate([np.linspace(-1, -1e-6, 200001), np.linspace(1e-6, 50, 200001)])
r2 = np.sqrt(1 + x) - (1 + x / 2 - x ** 2 / 8)
out["A"] = dict(lower_c=float(np.max(-r2 / np.abs(x) ** 3)), upper_ok=bool(np.all(r2 - x ** 3 / 16 <= 1e-15)),
                weak_ok=bool(np.all(np.sqrt(1 + x) - (1 + x / 2 - x ** 2 / 2) >= -1e-15)))
print("A", out["A"], flush=True)

def thermal(nb, d): q = nb / (nb + 1); return np.diag((1 - q) * q ** np.arange(d)).astype(complex)
def lad(d): a = np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex); return a, a.conj().T
def chern(f0, f1, w):
    g = lambda s: np.log(np.sum(w * f0 ** s * f1 ** (1 - s)))
    return -minimize_scalar(g, bounds=(1e-3, 1 - 1e-3), method="bounded").fun
def qchern(r0, r1):
    e0, U0 = np.linalg.eigh(r0); e1, U1 = np.linalg.eigh(r1); e0 = e0.clip(1e-300); e1 = e1.clip(1e-300)
    O = np.abs(U0.conj().T @ U1) ** 2
    g = lambda s: np.log(np.sum((e0[:, None] ** s) * (e1[None, :] ** (1 - s)) * O))
    return -minimize_scalar(g, bounds=(1e-3, 1 - 1e-3), method="bounded").fun
def dens(g, rho, th):
    return np.real(g.qfun(rho)).ravel().clip(1e-300), np.array([np.real(g.marginal(rho, t)) for t in th]).ravel().clip(1e-300)

# (B) displacement of any size
B = []; NTH = 96; th = np.arange(NTH) * np.pi / NTH
for nb in (0.0, 0.5, 1.0):
    for al in (0.3, 1.0, 2.0, 3.0):
        d = int(50 + 25 * nb + 3 * al ** 2); v = nb + 0.5; a, ad = lad(d + 40)
        g = Grid(v, d, L=np.sqrt(2) * al + 9 * np.sqrt(v) + 3, nx=2401, nq=401, nth=NTH)
        D = expm(al * ad - np.conj(al) * a); r0 = thermal(nb, d); r1 = (D @ thermal(nb, d + 40) @ D.conj().T)[:d, :d]
        Q0, P0 = dens(g, r0, th); Q1, P1 = dens(g, r1, th)
        u = al ** 2 / (4 * (2 * nb + 1))
        row = dict(nb=nb, al=al, xiQ=chern(Q0, Q1, g.dq ** 2 / (np.sum(Q0) * g.dq ** 2)), xiQ_formula=al ** 2 / (4 * (nb + 1)),
                   xiW=chern(P0, P1, g.dx / NTH), xiW_formula=float(-np.log(i0e(u))),
                   xiQC=qchern(r0, r1), xiQC_formula=al ** 2 * (np.sqrt(nb + 1) - np.sqrt(nb)) ** 2)
        row["max_rel"] = max(abs(row[k] / row[k + "_formula"] - 1) for k in ("xiQ", "xiW", "xiQC"))
        B.append(row); print("B", nb, al, f"{row['max_rel']:.1e}", flush=True)
out["B"] = B; out["B_max_rel"] = max(r["max_rel"] for r in B)

# (C) mixtures with distant states
d = 36; NTH = 72; th = np.arange(NTH) * np.pi / NTH; a, ad = lad(d + 30)
def ket(v): v = np.asarray(v, complex)[:d]; v = v / np.linalg.norm(v); return np.outer(v, v.conj())
def coh(al): return expm(al * ad - np.conj(al) * a)[:, 0]
def sq(r): return expm(0.5 * r * (a @ a - ad @ ad))[:, 0]
def fock(n): v = np.zeros(d + 30, complex); v[n] = 1; return v
SIG = {"fock1": ket(fock(1)), "fock2": ket(fock(2)), "fock3": ket(fock(3)), "cat_even_1.5": ket(coh(1.5) + coh(-1.5)),
       "cat_odd_1.5": ket(coh(1.5) - coh(-1.5)), "squeezed_0.5": ket(sq(0.5)), "coherent_1": ket(coh(1.0)),
       "binomial_0+4": ket(fock(0) + fock(4)), "fock01": ket(fock(0) + fock(1))}
EPS = (0.01, 0.05, 0.2, 0.5, 0.8, 1.0); C = {}
for nb in (0.5, 1.0):
    v = nb + 0.5; g = Grid(v, d, L=13.0, nx=1601, nq=321, nth=NTH)
    r0 = thermal(nb, d); Q0, P0 = dens(g, r0, th); wQ = g.dq ** 2 / np.sum(Q0 * g.dq ** 2); wP = g.dx / NTH; pi = np.real(np.diag(r0))
    for name, s in SIG.items():
        Dl = s - r0; dQ = np.real(g.qfun(Dl)).ravel(); dP = np.array([np.real(g.marginal(Dl, t)) for t in th]).ravel()
        FQ = np.sum(wQ * dQ ** 2 / Q0); FW = np.sum(wP * dP ** 2 / P0); FS = 2 * np.real(np.sum(np.abs(Dl) ** 2 / (pi[:, None] + pi[None, :])))
        rows = []
        for eps in EPS:
            res = {"eps": eps}
            for X, f0, dl, w, F in (("Q", Q0, dQ, wQ, FQ), ("W", P0, dP, wP, FW)):
                f1 = (f0 + eps * dl).clip(1e-300); xx = eps * dl / f0
                chi2 = np.sum(w * (f1 - f0) ** 2 / f0); BC = np.sum(w * np.sqrt(f0 * f1)); xi = chern(f0, f1, w)
                lo = 1 - chi2 / 8 - 0.375 * np.sum(w * f0 * np.abs(xx) ** 3); hi = 1 - chi2 / 8 + np.sum(w * f0 * xx ** 3) / 16
                res[X] = dict(chi2_rel=abs(chi2 / (eps ** 2 * F) - 1), xi=xi, BC=BC, lo=lo, hi=hi,
                              ok=bool(lo <= BC + 1e-9 and BC <= hi + 1e-9 and xi <= -2 * np.log(BC) + 1e-9))
            res["ratio_xi"] = res["W"]["xi"] / res["Q"]["xi"]; rows.append(res)
        C[f"nb{nb}_{name}"] = dict(FQ=FQ, FW=FW, FSLD=FS, ratio_chi2=FW / FQ, rows=rows)
        print("C", nb, name, f"{FW/FQ:.3f}", " ".join(f"{r['ratio_xi']:.3f}" for r in rows), flush=True)
out["C"] = C
out["C_chi2_max_rel"] = max(r[X]["chi2_rel"] for c in C.values() for r in c["rows"] for X in "QW")
out["C_inequalities_ok"] = all(r[X]["ok"] for c in C.values() for r in c["rows"] for X in "QW")
clear = {k: c for k, c in C.items() if abs(c["ratio_chi2"] - 1) > 0.05}
def same(c, emax): return all((r["ratio_xi"] > 1) == (c["ratio_chi2"] > 1) for r in c["rows"] if r["eps"] <= emax)
out["C_clear"] = len(clear); out["C_same_to_0.5"] = sum(same(c, 0.5) for c in clear.values()); out["C_same_to_1"] = sum(same(c, 1.0) for c in clear.values())
c = C["nb0.5_fock1"]; k = np.log(1 + 4 * (1 - 2 * 0.05) ** 2)
out["N_example"] = {X: int(np.ceil(k / np.log(1 + 0.2 ** 2 * c[F]))) for X, F in (("Q", "FQ"), ("W", "FW"), ("SLD", "FSLD"))}
print({k: v for k, v in out.items() if k not in "ABC"}, flush=True)
json.dump(out, open("out/c14.json", "w"), indent=1, default=float)
