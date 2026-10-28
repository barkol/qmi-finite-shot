"""Rekordy eksperymentalne PSS (Kawasaki+ 2024, Nat. Commun., Dryad doi:10.5061/dryad.9p8cz8wqn, CC0) tym samym protokolem
co rekordy GKP Konno+ (qmi/x8/konno.py; tu kopia z adaptacja, oryginal bez zmian).
Dane: 11 plikow quad_01GHz_{01,03,10,25}mW_{00..20}dB[_2].mat, zmienne p60deg..p210deg (6 faz co 30 st.), 5000 zdarzen kazda.
Mapowanie na katy potoku (jak qmi.exp_gkp dla Konno): 60,90,120,150 bez zmian; 180 -> 0 i 210 -> 30 ze zmiana znaku (x_{th+180} = -x_th).
Etapy (punkt kontrolny npz na plik i etap w <out>/<tag>/, wznowienie pomija gotowe; stop na pierwszym bledzie):
  kalib : srednia na kat = c + x0 cos a + p0 sin a (jak konno), <n> z drugich momentow, wariancje, kurtoza
  skala : skan skali kwadratur s (x_potok = s x_dane) - logL MaxLik z jakobianem N log s; s=1 <=> proznia 1/2, s=sqrt2 <=> proznia 1/4 (README)
  stac  : 7 testow kolejnosci (jak konno.stac, R permutacji)
  iid   : chi2 jednorodnosci 10 blokow + odleglosci blok/losowy podzbior (jak konno.iid)
  ml    : MaxLik R rho R (D=40, jak konno.ml), oba kierunki obrotu; czystosc, <n>, W(0), min W, bootstrap
  pss   : dopasowanie modelu PSS: strata eta po a S(r,psi)|0>, wiarygodnosc multinomialna; wiernosci z rho_ML
  klasyf: sieci W K=6 (bazowe shotsK6, gkpxK6, Ahmed K6 z analyses/kawasaki/runs_ahmK6): losowe podzbiory, kolejne bloki, rekordy z rho_ML
  QMI_XMAX=12 PYTHONPATH=. python -I analyses/kawasaki/kawa.py --out analyses/kawasaki/out"""
import argparse, json, os, sys, time, re
from pathlib import Path
import numpy as np

if os.environ.get("QMI_XMAX") != "12": sys.exit("BLAD: QMI_XMAX musi byc 12")
sys.path.insert(0, os.environ.get("QMI_ROOT", str(Path(__file__).resolve().parents[2])))   # -I wycina PYTHONPATH
from scipy import stats, optimize
import qmi  # noqa: F401  sets qsdl.config.XMAX from QMI_XMAX (pinned qsdl has XMAX=5.0); must precede qsdl imports
from qsdl.config import GRID, XMAX, CUTOFF
from qmi.shots import radon, fbp, split_events

RAW = Path(os.path.join(os.environ.get("QMI_ROOT", str(Path(__file__).resolve().parents[2])), "dane_eksp/kawasaki2024/raw"))
ANG = (60, 90, 120, 150, 180, 210)
MAP = {60: 60, 90: 90, 120: 120, 150: 150, 180: 0, 210: 30}
TH = sorted(MAP.values())
CLASSES = ("fock", "coherent", "vacuum", "thermal", "cat", "gkp", "binomial")
NS = (100, 1000, 10000)
DX = 2 * XMAX / (GRID - 1); EDGES = np.linspace(-XMAX - DX / 2, XMAX + DX / 2, GRID + 1)
D = 40
LOG = None


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True)
    if LOG: open(LOG, "a").write(s + "\n")


def save(p, **kw):
    np.savez(str(p) + ".tmp.npz", **kw); os.replace(str(p) + ".tmp.npz", p)


def files():
    fs = sorted(RAW.glob("quad_01GHz_*mW_*dB*.mat"))
    if len(fs) != 11: raise RuntimeError(f"{len(fs)} plikow .mat zamiast 11")
    return fs


def tag(f):
    m = re.match(r"quad_01GHz_(\d+)mW_(\d+)dB(_2)?\.mat$", f.name)
    return f"{m.group(1)}mW_{m.group(2)}dB" + ("_2" if m.group(3) else "")


def raw(f):
    """Tylko odczyt (scipy.io.loadmat); straznik wejscia: 6 faz, po 5000 skonczonych liczb."""
    from scipy.io import loadmat
    d = loadmat(str(f)); out = {}
    for a in ANG:
        x = np.asarray(d[f"p{a}deg"], float).ravel()
        if x.size != 5000 or not np.all(np.isfinite(x)) or np.abs(x).max() > 0.9 * XMAX: raise RuntimeError(f"{f.name} p{a}deg: zle dane")
        out[a] = x
    return out


def records(rr, c, s=1.0):
    return {MAP[a]: s * (-(x - c) if a >= 180 else (x - c)) for a, x in rr.items()}


# ---------- kalibracja (jak konno.kalib bez pikow GKP) + <n> z drugich momentow
def kalib(rr, rng):
    a = np.array(ANG); m = np.array([rr[k].mean() for k in a]); se = np.array([rr[k].std(ddof=1) / np.sqrt(len(rr[k])) for k in a])
    th = np.deg2rad(a); w = 1 / se
    def fit(A):
        coef, *_ = np.linalg.lstsq(A * w[:, None], m * w, rcond=None); res = (m - A @ coef) / se
        cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None])); return coef, np.sqrt(np.diag(cov)), float(res @ res)
    full = fit(np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)); disp = fit(np.stack([np.cos(th), np.sin(th)], 1)); off = fit(np.ones((6, 1)))
    rec = records(rr, full[0][0])
    # 6 rownych katow w [0,180): srednia <x_th^2> = (<x^2>+<p^2>)/2 dokladnie -> <n> = sr - v0 (v0 = wariancja prozni w jednostkach danych)
    def nb(rec, v0): return float(np.mean([np.mean(rec[t] ** 2) for t in TH]) / (2 * v0) * 1.0 - 0.5)   # <n> = (<x^2>+<p^2>)/(2*2v0) - 1/2
    boot = np.array([[nb({t: rng.choice(rec[t], len(rec[t])) for t in TH}, v0) for v0 in (0.5, 0.25)] for _ in range(500)])
    v = np.array([rec[t].var() for t in TH]); ku = np.array([stats.kurtosis(rec[t]) for t in TH])
    return dict(ang=a, mean=m, se=se, c=full[0][0], c_se=full[1][0], x0=full[0][1], x0_se=full[1][1], p0=full[0][2], p0_se=full[1][2],
                chi2_full=full[2], chi2_disp=disp[2], chi2_off=off[2], th=np.array(TH), var=v, kurt=ku,
                nbar_v05=nb(rec, 0.5), nbar_v05_se=boot[:, 0].std(), nbar_v025=nb(rec, 0.25), nbar_v025_se=boot[:, 1].std(),
                vmin=v.min(), vmax=v.max(), vprod=v.min() * v.max())


# ---------- MaxLik (jak konno.ml, wektorowo); Pi zalezy tylko od D i sgn
_PI = {}
def povm(sgn, sub=8):
    if sgn in _PI: return _PI[sgn]
    from numpy.polynomial.hermite import hermval
    from math import factorial, pi
    xs = (EDGES[:-1, None] + (np.arange(sub)[None, :] + 0.5) / sub * DX).ravel()
    psi = np.stack([np.exp(-xs ** 2 / 2) * hermval(xs, [0] * n + [1]) / np.sqrt(2 ** n * factorial(n) * np.sqrt(pi)) for n in range(D)], 1)
    Pis = []
    for t in TH:
        v = psi * np.exp(-1j * sgn * np.arange(D) * np.deg2rad(t))[None, :]
        Pis.append(np.einsum("xa,xb->xab", v.conj(), v).reshape(GRID, sub, D, D).sum(1) * DX / sub)
    Pi = np.concatenate(Pis); _PI[sgn] = (Pi.reshape(-1, D * D), np.transpose(Pi, (0, 2, 1)).reshape(-1, D * D).copy())
    return _PI[sgn]


def hist(rec):
    H = np.stack([np.histogram(rec[t], EDGES)[0] for t in TH]).astype(float)
    if H.sum() != sum(len(rec[t]) for t in TH): raise RuntimeError("zdarzenia poza oknem siatki")
    return H


def mlrho(H, sgn, iters, rho0=None):
    P, PT = povm(sgn); h = H.ravel(); rho = np.eye(D, dtype=complex) / D if rho0 is None else rho0.copy(); llh = []
    for it in range(iters):
        pr = (PT @ rho.ravel()).real
        R = (np.where(pr > 1e-300, h / np.maximum(pr, 1e-300), 0) @ P).reshape(D, D)
        rho = R @ rho @ R; rho = (rho + rho.conj().T) / 2; rho /= np.trace(rho).real
        if it % 500 == 499 or it == iters - 1: llh.append(float(np.sum(h * np.log(np.maximum((PT @ rho.ravel()).real, 1e-300)))))
    return rho, np.array(llh)


def probs_pi(rho, sgn):
    return (povm(sgn)[1] @ rho.ravel()).real.reshape(6, GRID)


def wig(rho, y):
    import qutip as qt
    return qt.wigner(qt.Qobj(rho), y, y)


def marg(rho):
    """Marginesy potoku (radon obrazu W na siatce, jak konno.marg) -> model zliczen rekordow symulowanych."""
    import qutip as qt
    r = np.zeros((CUTOFF, CUTOFF), complex); r[:D, :D] = rho
    y = np.linspace(-XMAX, XMAX, GRID); W = qt.wigner(qt.Qobj(r), y, y); S = radon(W, TH)
    P = np.clip(S, 0, None); return P / P.sum(1, keepdims=True), W


def wstats(rho):
    yy = np.linspace(-3, 3, 241); W = wig(rho, yy); i = np.unravel_index(np.argmin(W), W.shape)
    return float(wig(rho, np.array([0.0]))[0, 0]), float(W.min()), float(yy[i[1]]), float(yy[i[0]])


def summ(rho):
    n = np.arange(D); w0, wmin, xm, pm = wstats(rho)
    return dict(purity=float(np.trace(rho @ rho).real), nbar=float(np.real(np.sum(n * np.diag(rho)))), W0=w0, Wmin=wmin, Wmin_x=xm, Wmin_p=pm,
                podd=float(np.real(np.diag(rho)[1::2].sum())), p0=float(np.real(rho[0, 0])), p1=float(np.real(rho[1, 1])), p2=float(np.real(rho[2, 2])))


def ml(rec, rng, iters, nboot, ibo):
    H = hist(rec); best = None
    for sgn in (+1, -1):
        rho, llh = mlrho(H, sgn, iters); P = probs_pi(rho, sgn); Pe = H / H.sum(1, keepdims=True)
        # logL po Pi jest taka sama dla obu sgn (rho* przy -sgn daje te same rozklady); kierunek wybiera zgodnosc z marginesami
        # potoku (radon obrazu W, jak w konno.ml), bo z nich losujemy rekordy symulowane i z nich sa obrazy sieci
        Pm, W = marg(rho); llm = float(np.sum(H * np.log(Pm + 1e-12)))
        r = dict(sgn=sgn, rho=rho, ll=llh[-1], ll_hist=llh, tvd_pi=0.5 * np.abs(P - Pe).sum(1), ll_marg=llm, P=Pm, W=W, tvd=0.5 * np.abs(Pm - Pe).sum(1))
        log(f"   ml sgn={sgn}: logL(Pi)={llh[-1]:.1f} (zmiana w ostatnich 500 it.: {np.round(np.diff(llh)[-1:], 3)}) TVD(Pi) {np.round(r['tvd_pi'], 3)}; logL(marg potoku)={llm:.1f} TVD {np.round(r['tvd'], 3)}")
        if best is None or r["ll_marg"] > best["ll_marg"]: best = dict(r, ll_marg_other=best["ll_marg"] if best else np.nan)
        else: best["ll_marg_other"] = r["ll_marg"]
    rho = best["rho"]; Pe = H / H.sum(1, keepdims=True)
    half = np.array([0.5 * np.abs(np.histogram(rec[t][::2], EDGES)[0] / len(rec[t][::2]) - np.histogram(rec[t][1::2], EDGES)[0] / len(rec[t][1::2])).sum() for t in TH])
    out = dict(best, tvd_half=half, **summ(rho))
    bs = []                                       # bootstrap nieparametryczny (losowanie zdarzen ze zwracaniem w kazdym kacie), start z rho_ML
    for b in range(nboot):
        Hb = hist({t: rng.choice(rec[t], len(rec[t])) for t in TH}); rb, _ = mlrho(Hb, best["sgn"], ibo, rho0=0.9 * rho + 0.1 * np.eye(D) / D)
        s = summ(rb); bs.append([s[k] for k in ("purity", "nbar", "W0", "Wmin")])
    bs = np.array(bs).reshape(-1, 4); out["boot"] = bs; out["boot_sd"] = bs.std(0, ddof=1) if len(bs) > 1 else np.full(4, np.nan)
    return out


# ---------- skala kwadratur
SC = (0.707, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.414)
def skala(rr, c, sgn, iters):
    out = []
    for s in SC:
        rec = records(rr, c, s); H = hist(rec); rho, llh = mlrho(H, sgn, iters); su = summ(rho)
        out.append([s, llh[-1] + H.sum() * np.log(s), su["purity"], su["nbar"], su["W0"], su["Wmin"]])
        if True:   # model fizyczny PSS+strata (eta<=1) przy tej skali: -nll + N log s
            b = pss_fit(H, sgn); out[-1] += [-b.fun + H.sum() * np.log(s), *b.x]
        else: out[-1] += [np.nan] * 4
        log(f"   skala s={s}: logL_ML+N log s={out[-1][1]:.1f} logL_PSS+N log s={out[-1][6]:.1f} purity={su['purity']:.3f} nbar={su['nbar']:.2f} W0={su['W0']:.3f} PSS(r,psi,eta)={np.round(out[-1][7:], 3)}")
    return dict(tab=np.array(out), cols=np.array(["s", "logLj", "purity", "nbar", "W0", "Wmin", "logLj_pss", "pss_r", "pss_psi", "pss_eta"]))


# ---------- PSS: eta-strata po a S(r,psi)|0>
HB = 100; _A = np.diag(np.sqrt(np.arange(1, HB)), 1)
def pss_ket(r, psi):
    from scipy.linalg import expm
    z = r * np.exp(2j * psi); k = expm(0.5 * (np.conj(z) * _A @ _A - z * _A.T @ _A.T))[:, 0]; k = _A @ k
    k = k[:D]; return k / np.linalg.norm(k)


def cat_ket(al, phi):
    from scipy.special import gammaln
    n = np.arange(D); a = al * np.exp(1j * phi)
    c = np.exp(-abs(a) ** 2 / 2 + n * np.log(abs(a) + 1e-300) - 0.5 * gammaln(n + 1)) * np.exp(1j * phi * n)
    k = c - (-1) ** n * c; return k / np.linalg.norm(k)


def loss(rho, eta):
    from scipy.special import gammaln
    n = np.arange(D); out = np.zeros_like(rho)
    for k in range(D):
        m = n[k:]; c = np.exp(0.5 * (gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1) + (m - k) * np.log(eta) + k * np.log1p(-eta + 1e-300)))
        E = np.zeros((D, D)); E[m - k, m] = c; out += E @ rho @ E.T
    return out


def fid(rho, sig):
    from scipy.linalg import sqrtm
    s = sqrtm(rho); return float(np.real(np.trace(sqrtm(s @ sig @ s))) ** 2)


def pss_model(p):
    r, psi, eta = p; k = pss_ket(r, psi); return loss(np.outer(k, k.conj()), eta)


def pss_fit(H, sgn):
    def nll(p):
        r, psi, eta = p
        if not (0 <= r < 1.5 and 0.05 < eta <= 1.0): return 1e12
        return -float(np.sum(H * np.log(np.maximum(probs_pi(pss_model(p), sgn), 1e-12))))
    best = None
    for x0 in ([0.3, 0.0, 0.7], [0.3, np.pi / 2, 0.7], [0.6, np.pi / 4, 0.6], [0.6, -np.pi / 4, 0.6]):
        r = optimize.minimize(nll, x0, method="Nelder-Mead", options=dict(maxiter=600, xatol=1e-4, fatol=0.05))
        if best is None or r.fun < best.fun: best = r
    return best


def pss(rec, G):
    H = hist(rec); sgn = int(G["sgn"]); rho_ml = G["rho"]; model = pss_model
    best = pss_fit(H, sgn)
    rm, pm, em = best.x; sig = model(best.x); P = probs_pi(sig, sgn); Pe = H / H.sum(1, keepdims=True)
    # najlepsze stany czyste: PSS (r, psi) i kot nieparzysty (|alpha|, phi) wobec rho_ML
    fp = optimize.minimize(lambda p: -np.real(np.vdot(pss_ket(*p), rho_ml @ pss_ket(*p))), [rm, pm], method="Nelder-Mead")
    fc = max((optimize.minimize(lambda p: -np.real(np.vdot(cat_ket(*p), rho_ml @ cat_ket(*p))), [a0, f0], method="Nelder-Mead")
              for a0 in (0.8, 1.3) for f0 in (pm, pm + np.pi / 2)), key=lambda r: -r.fun)
    su = summ(sig)
    return dict(r=rm, psi=pm, eta=em, sqdb=8.686 * rm, nll=best.fun, ll_ml=float(G["ll"]), tvd=0.5 * np.abs(P - Pe).sum(1), F_model_ml=fid(rho_ml, sig),
                F_purePSS=-fp.fun, purePSS_r=fp.x[0], F_oddcat=-fc.fun, cat_alpha=abs(fc.x[0]), model_W0=su["W0"], model_nbar=su["nbar"], model_purity=su["purity"])


# ---------- stacjonarnosc i iid (kopie konno.stac / konno.iid; katy potoku)
LAGS = (1, 2, 5, 10, 100); KB = (10, 50, 200)
def stac(rec, rng, R=200):
    xs = [rec[k].astype(float) for k in TH]
    def ac(x): return np.mean([np.corrcoef(x[:-k], x[k:])[0, 1] for k in LAGS])
    def stats7(xs):
        s = [float(np.mean([ac(x) for x in xs]))]
        for f in (stats.f_oneway, stats.levene):
            for K in KB: s.append(float(-sum(np.log(max(f(*np.array_split(x, K)).pvalue, 1e-300)) for x in xs)))
        return np.array(s)
    obs = stats7(xs); nul = np.array([stats7([rng.permutation(x) for x in xs]) for _ in range(R)])
    p = (1 + (nul >= obs).sum(0)) / (R + 1)
    th = np.deg2rad(np.array(TH)); v = np.array([rec[k].var() for k in TH])
    M = np.stack([np.cos(th) ** 2, np.sin(th) ** 2, 2 * np.sin(th) * np.cos(th)], 1); ca, cb, cc = np.linalg.lstsq(M, v, rcond=None)[0]
    dV = (cb - ca) * np.sin(2 * th) + 2 * cc * np.cos(2 * th); ph = np.full(6, np.nan)
    for i, k in enumerate(TH):
        x = rec[k]; B = np.array_split(x, 10); vv = np.array([q.var() for q in B])
        kurt = np.mean((x - x.mean()) ** 4) / x.var() ** 2; se = x.var() * np.sqrt((kurt - 1) / len(B[0]))
        if abs(dV[i]) >= 0.9: ph[i] = np.rad2deg(np.sqrt(max(0.0, vv.var(ddof=1) - se ** 2)) / abs(dV[i]))
    return dict(names=np.array(["autokor"] + [f"anova_K{K}" for K in KB] + [f"levene_K{K}" for K in KB]), obs=obs, null_mean=nul.mean(0),
                null_sd=nul.std(0), pval=p, p_bonf=np.minimum(1, 7 * p), R=R, ac_z=float((obs[0] - nul[:, 0].mean()) / nul[:, 0].std()),
                ang=np.array(TH), dV=dV, phase_sd=ph, var_abc=np.array([ca, cb, cc]))


def iid(rec, rng):
    out = {}
    for th, x in rec.items():
        n = len(x); q = np.quantile(x, np.linspace(0, 1, 21)[1:-1]); b = np.digitize(x, q)
        T = np.stack([np.bincount(blk, minlength=20) for blk in np.array_split(b, 10)])
        chi2, p, dof, _ = stats.chi2_contingency(T)
        out[f"th{th}_chi2"] = chi2; out[f"th{th}_p"] = p; out[f"th{th}_dof"] = dof
        out[f"th{th}_lag"] = np.array([np.corrcoef(x[:-k], x[k:])[0, 1] for k in LAGS])
        hall = np.histogram(x, EDGES)[0] / n
        for N in NS:
            m = int(split_events(N, 6).max())
            blocks = [x[i:i + m] for i in range(0, n - m + 1, m)]
            dB = [np.sum((np.histogram(s, EDGES)[0] / m - hall) ** 2) for s in blocks]
            dR = [np.sum((np.histogram(rng.choice(x, m, replace=False), EDGES)[0] / m - hall) ** 2) for _ in range(max(200, len(blocks)))]
            out[f"th{th}_N{N}_dB"] = np.array(dB); out[f"th{th}_N{N}_dR"] = np.array(dR)
    out["fisher_p"] = stats.combine_pvalues([out[f"th{th}_p"] for th in rec]).pvalue
    for N in NS:
        B = [out[f"th{th}_N{N}_dB"] for th in rec]; R = [out[f"th{th}_N{N}_dR"] for th in rec]
        r0 = np.mean([b.mean() / r.mean() for b, r in zip(B, R)])
        bs = [np.mean([rng.choice(b, len(b)).mean() / rng.choice(r, len(r)).mean() for b, r in zip(B, R)]) for _ in range(2000)]
        out[f"ratio_N{N}"] = np.array([r0, *np.percentile(bs, [2.5, 97.5])]); out[f"nblocks_N{N}"] = np.array([len(b) for b in B])
    return out


# ---------- klasyfikacja (jak konno.klasyf; N=0 = caly rekord: 1 obraz + bootstrap + rekordy symulowane 5000/kat)
def img_counts(C, per, M):
    return fbp(np.stack([c / max(1, p) * M for c, p in zip(C, per)]), TH, GRID)


def klasyf(rec, Pm, models, M, rng, draws):
    from qmi.eval_shots import pred
    out = {}; nmin = min(len(v) for v in rec.values())
    for N in (*NS, 0):
        per = split_events(N, 6) if N else np.array([len(rec[t]) for t in TH])
        X = {"exp_los": [], "exp_blok": [], "sym": []} if N else {"exp_los": [], "exp_boot": [], "sym": []}
        for _ in range(draws):
            if N: X["exp_los"].append(img_counts([np.histogram(rng.choice(rec[t], p, replace=False), EDGES)[0] for t, p in zip(TH, per)], per, M))
            else: X["exp_boot"].append(img_counts([np.histogram(rng.choice(rec[t], p, replace=True), EDGES)[0] for t, p in zip(TH, per)], per, M))
            X["sym"].append(img_counts([rng.multinomial(p, Pm[k]) for k, p in enumerate(per)], per, M))
        if N:
            for b in range(nmin // per.max()):
                X["exp_blok"].append(img_counts([np.histogram(rec[t][b * p:(b + 1) * p], EDGES)[0] for t, p in zip(TH, per)], per, M))
        else: X["exp_los"].append(img_counts([np.histogram(rec[t], EDGES)[0] for t in TH], per, M))
        for src, imgs in X.items():
            imgs = np.stack(imgs).astype(np.float32)
            for nm, (m, rep, sc) in models.items(): out[f"{nm}|{src}|N{N}"] = np.bincount(pred(m, imgs, sc), minlength=7)
    return out


def negN(rec, G, rng, nsub=50, iters=1000):
    """Ujemnosc W przy skonczonym N: MaxLik z losowych podzbiorow N zdarzen (split_events(N,6)), start z rho maks. mieszanej."""
    sgn = int(G["sgn"]); out = {}
    for N in NS:
        per = split_events(N, 6); r = []
        for _ in range(nsub):
            H = np.stack([np.histogram(rng.choice(rec[t], p, replace=False), EDGES)[0] for t, p in zip(TH, per)]).astype(float)
            rho, _ = mlrho(H, sgn, iters); su = summ(rho); r.append([su["W0"], su["Wmin"], su["purity"], su["nbar"]])
        out[f"N{N}"] = np.array(r)
    return out


def cent(rec, G, iters):
    """Wariant czulosci (nie protokol Konno): kazda faza wycentrowana osobno (PSS ma <x_th>=0), ten sam sgn, MaxLik jak w ml."""
    r2 = {t: x - x.mean() for t, x in rec.items()}; H = hist(r2); rho, _ = mlrho(H, int(G["sgn"]), iters); Pm, _ = marg(rho)
    return dict(**summ(rho), ll_marg=float(np.sum(H * np.log(Pm + 1e-12))))


def rektest(rec, Pm, models, M, rng, B=40, nsub=50, nref=400):
    """Test na poziomie rekordu (dopasowana zaleznosc podzbiorow): statystyka T = TVD miedzy rozkladem klas nsub losowych podzbiorow
    JEDNEGO rekordu (5000 zdarzen/kat) a rozkladem referencyjnym (nref niezaleznych rekordow symulowanych z rho_ML, N zdarzen);
    rozklad zerowy T z B pelnych rekordow symulowanych (multinomial 5000/kat), z ktorych podzbiory losujemy tak samo
    (bez zwracania: wielowymiarowy rozklad hipergeometryczny na zliczeniach pikseli). p = (1 + #(T_sym >= T_exp)) / (B + 1)."""
    from qmi.eval_shots import pred
    n = np.array([len(rec[t]) for t in TH]); Hexp = hist(rec).astype(np.int64); out = {}
    def cls(imgs):
        imgs = np.stack(imgs).astype(np.float32); return {nm: np.bincount(pred(m, imgs, sc), minlength=7) for nm, (m, rep, sc) in models.items()}
    for N in NS:
        per = split_events(N, 6)
        ref = cls([img_counts([rng.multinomial(p, Pm[k]) for k, p in enumerate(per)], per, M) for _ in range(nref)])
        def T(H):
            c = cls([img_counts([rng.multivariate_hypergeometric(H[k], p) for k, p in enumerate(per)], per, M) for _ in range(nsub)])
            t = {nm: 0.5 * np.abs(c[nm] / nsub - ref[nm] / nref).sum() for nm in models}; t["suma"] = float(np.mean(list(t.values()))); return t, c
        te, ce = T(Hexp); ts = [T(np.stack([rng.multinomial(nk, Pm[k]) for k, nk in enumerate(n)]))[0] for _ in range(B)]
        for nm in (*models, "suma"):
            null = np.array([x[nm] for x in ts]); out[f"{nm}|N{N}|T_exp"] = np.array(te[nm]); out[f"{nm}|N{N}|T_null"] = null
            out[f"{nm}|N{N}|p"] = np.array((1 + (null >= te[nm]).sum()) / (B + 1))
        for nm in models: out[f"{nm}|N{N}|exp_counts"] = ce[nm]; out[f"{nm}|N{N}|ref_counts"] = ref[nm]
    return out


def work(args):
    """Etapy CPU dla jednego pliku (proces potomny, 1 watek BLAS)."""
    f, out, et, opt = args; f = Path(f); t = tag(f); d = Path(out) / t; d.mkdir(parents=True, exist_ok=True)
    global LOG; LOG = str(d / "log.txt"); rr = raw(f); seed = 20261022 + sum(map(ord, t))
    if not (d / "kalib.npz").exists(): save(d / "kalib.npz", **kalib(rr, np.random.default_rng(seed)))
    K = dict(np.load(d / "kalib.npz")); rec = records(rr, float(K["c"]))
    log(f"{t}: c={float(K['c']):+.4f}+-{float(K['c_se']):.4f} x0={float(K['x0']):+.4f} p0={float(K['p0']):+.4f} chi2 {float(K['chi2_full']):.1f}/3; "
        f"var {np.round(K['var'], 3)} <n>(v0=1/2)={float(K['nbar_v05']):.3f}")
    if "stac" in et and not (d / "stac.npz").exists(): save(d / "stac.npz", **stac(rec, np.random.default_rng(seed + 1), opt["R"]))
    if "iid" in et and not (d / "iid.npz").exists(): save(d / "iid.npz", **iid(rec, np.random.default_rng(seed + 2)))
    if "ml" in et and not (d / "ml.npz").exists(): save(d / "ml.npz", **ml(rec, np.random.default_rng(seed + 3), opt["iters"], opt["nboot"], opt["iboot"]))
    if "skala" in et and not (d / "skala.npz").exists():
        G = np.load(d / "ml.npz"); save(d / "skala.npz", **skala(rr, float(K["c"]), int(G["sgn"]), opt["iboot"]))
    if "cent" in et and not (d / "cent.npz").exists(): save(d / "cent.npz", **cent(rec, dict(np.load(d / "ml.npz")), opt["iters"]))
    if "negN" in et and not (d / "negN.npz").exists(): save(d / "negN.npz", **negN(rec, dict(np.load(d / "ml.npz")), np.random.default_rng(seed + 4)))
    if "pss" in et and not (d / "pss.npz").exists(): save(d / "pss.npz", **pss(rec, dict(np.load(d / "ml.npz"))))
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--etapy", default="kalib,stac,iid,ml,skala,pss,klasyf"); ap.add_argument("--pliki", default="")
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--R", type=int, default=200); ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--nboot", type=int, default=20); ap.add_argument("--iboot", type=int, default=1500); ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--runs", default="runs_eq,runs_eq,analyses/kawasaki/runs_ahmK6"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); LOG = str(out / "kawa.log"); t0 = time.time(); et = a.etapy.split(",")
    fs = [f for f in files() if not a.pliki or re.search(a.pliki, f.name)]
    log(f"start XMAX={XMAX} GRID={GRID} D={D} pliki={[tag(f) for f in fs]} etapy={et}")
    opt = dict(R=a.R, iters=a.iters, nboot=a.nboot, iboot=a.iboot)
    cpu = [e for e in et if e not in ("klasyf", "rektest")]
    if cpu:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(a.workers, len(fs))) as pool:
            for t in pool.imap_unordered(work, [(str(f), str(out), cpu, opt) for f in fs]): log(f"gotowe etapy CPU: {t} ({time.time()-t0:.0f}s)")
    if "klasyf" in et or "rektest" in et:
        import qutip as qt
        from qmi.eval_shots import load_models
        root = Path(os.environ.get("QMI_ROOT", str(Path(__file__).resolve().parents[2]))); rs = a.runs.split(",")
        sets = {"baza": (rs[0], r"re:^clf_W_shotsK6_s\d+$"), "gkpx": (rs[1], r"re:^clf_W_gkpxK6_s\d+$"), "ahm": (rs[2], r"re:^clf_W_shotsK6_s\d+$")}
        y = np.linspace(-XMAX, XMAX, GRID); M = float(radon(qt.wigner(qt.fock_dm(CUTOFF, 0), y, y), [0])[0].sum())
        for nm, (runs, rx) in sets.items():
            if not (root / runs).exists() or not list((root / runs).glob("clf_W_shotsK6_s*/best.pt") if nm != "gkpx" else (root / runs).glob("clf_W_gkpxK6_s*/best.pt")):
                log(f"klasyf {nm}: brak sieci w {runs} - pomijam"); continue
            models = load_models(str(root / runs), rx)
            if len(models) != 3: sys.exit(f"BLAD: {nm}: {len(models)} modeli zamiast 3: {sorted(models)}")
            for f in fs:
                t = tag(f); d = out / t; KO = d / f"klasyf_{nm}.json"
                if KO.exists() and ("rektest" not in et or (d / f"rektest_{nm}.npz").exists()): continue
                rr = raw(f); K = np.load(d / "kalib.npz"); rec = records(rr, float(K["c"])); G = np.load(d / "ml.npz")
                rng = np.random.default_rng(20261023 + sum(map(ord, t + nm)))
                if "klasyf" in et and not KO.exists():
                    res = klasyf(rec, G["P"], models, M, rng, a.draws)
                    json.dump({k: v.tolist() for k, v in res.items()}, open(str(KO) + ".tmp", "w")); os.replace(str(KO) + ".tmp", KO)
                    log(f"klasyf {nm} {t}: " + "; ".join(f"N{N}: " + " ".join(f"{s}:{CLASSES[int(np.argmax(sum(res[f'{m}|{s}|N{N}'] for m in models)))]}" for s in ("exp_los", "sym")) for N in (*NS, 0)))
                RT = d / f"rektest_{nm}.npz"
                if "rektest" in et and not RT.exists():
                    save(RT, **rektest(rec, G["P"], models, M, np.random.default_rng(20261024 + sum(map(ord, t + nm)))))
                    R_ = np.load(RT); log(f"rektest {nm} {t}: p(suma) " + " ".join(f"N{N}={float(R_[f'suma|N{N}|p']):.3f}" for N in NS))
    log(f"KAWA_DONE {time.time()-t0:.0f}s")
