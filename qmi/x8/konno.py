"""Rekordy eksperymentalne GKP (Konno+ 2024, Dryad) wobec modelu szumu zliczen.
Etapy (kazdy z punktem kontrolnym w <out>/, wznowienie pomija gotowe):
  1 kalib   : srednia na kat = c + x0 cos a + p0 sin a (wagi 1/se^2) -> staly offset c czy przesuniecie stanu; odstep pikow przy 0 st.
  2 iid     : stacjonarnosc (test jednorodnosci chi^2 10 blokow x 20 binow rownego prawdop.) oraz odleglosc histogramu
              n zdarzen od histogramu calego rekordu: kolejne bloki wobec losowych podzbiorow (przy iid stosunek ~1)
  3 dopas   : |1_L> (delta) po sciskaniu (r, psi) i stracie eta (konwencja i stan z Konno+ supl.); wiarygodnosc multinomialna rekordow na pikselach
              tej samej siatki i tych samych 6 rzutow co w potoku (radon/FBP z qmi.shots)
  3b ml     : rekonstrukcja MaxLik macierzy gestosci (jak u Konno+), rekordy symulowane w kroku 4 z niej (prosty model z 3 nie oddaje danych)
  4 klasyf  : sieci W K=6 (bazowe shotsK6 i gkpxK6), N = 1e2..1e4 oraz caly rekord: losowe podzbiory rekordu, kolejne bloki
              rekordu i rekordy symulowane z rekonstrukcji MaxLik (dokladnie model zliczen potoku) -> czestosci klas
  QMI_XMAX=12 PYTHONPATH=. ~/anaconda3/envs/ml/bin/python -m qmi.x8.konno --runs runs_eq --out runs_eq/konno"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, qutip as qt
from scipy import stats, optimize

if os.environ.get("QMI_XMAX") not in ("8", "12"): sys.exit("BLAD: QMI_XMAX musi byc 8 albo 12")
from qsdl.config import GRID, XMAX, CUTOFF
from qsdl.noise import photon_loss
from qmi.shots import radon, fbp, split_events

FILES = {0: 0, 30: 30, 60: 60, -30: 150, -60: 120, -90: 90}   # jak qmi.exp_gkp: ujemny kat -> +180 i odwrocenie znaku
TH = sorted(FILES.values())
CLASSES = ("fock", "coherent", "vacuum", "thermal", "cat", "gkp", "binomial")
NS = (100, 1000, 10000)
DX = 2 * XMAX / (GRID - 1); EDGES = np.linspace(-XMAX - DX / 2, XMAX + DX / 2, GRID + 1)
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    if LOG: open(LOG, "a").write(s + "\n")


def save(p, **kw):
    np.savez(str(p) + ".tmp.npz", **kw); os.replace(str(p) + ".tmp.npz", p)


def raw(d):
    return {a: np.load(Path(d) / f"quad_{a}deg.npy").astype(float) for a in FILES}


def records(rr, c):
    """Po odjeciu offsetu c: slownik kat potoku -> zdarzenia (kolejnosc w pliku zachowana)."""
    return {FILES[a]: (-(x - c) if a < 0 else x - c) for a, x in rr.items()}


# ---------- 1 kalibracja
def kalib(rr):
    a = np.array(sorted(rr)); m = np.array([rr[k].mean() for k in a]); se = np.array([rr[k].std(ddof=1) / np.sqrt(len(rr[k])) for k in a])
    th = np.deg2rad(a); w = 1 / se
    def fit(A):
        coef, *_ = np.linalg.lstsq(A * w[:, None], m * w, rcond=None); res = (m - A @ coef) / se
        cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None])); return coef, np.sqrt(np.diag(cov)), float(res @ res)
    full = fit(np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1))       # offset + przesuniecie stanu
    disp = fit(np.stack([np.cos(th), np.sin(th)], 1))                         # samo przesuniecie stanu
    off = fit(np.ones((len(th), 1)))                                          # sam offset
    # polozenia dwoch najwyzszych pikow gestosci (KDE) przy 0 st. i ich odstep; niepewnosc z bootstrapu
    from scipy.signal import find_peaks
    g = np.linspace(-5, 5, 2001)
    def pk(x):
        d = stats.gaussian_kde(x, bw_method=0.1)(g); i, pr = find_peaks(d, prominence=0.005)
        i = np.sort(i[np.argsort(pr["prominences"])[-2:]]); return g[i]
    x = rr[0] - full[0][0]; pos = pk(x); s = float(np.diff(pos)[0]) if len(pos) == 2 else np.nan
    rng = np.random.default_rng(1); sb = []
    for _ in range(100):
        q = pk(rng.choice(x, len(x))); sb.append(np.diff(q)[0] if len(q) == 2 else np.nan)
    return dict(ang=a, mean=m, se=se, var=np.array([rr[k].var() for k in a]), n=np.array([len(rr[k]) for k in a]),
                c=full[0][0], c_se=full[1][0], x0=full[0][1], x0_se=full[1][1], p0=full[0][2], p0_se=full[1][2],
                chi2_full=full[2], chi2_disp=disp[2], chi2_off=off[2], c_off=off[0][0], spacing=s, spacing_sd=float(np.nanstd(sb)), peaks=pos)


# ---------- 2b stacjonarnosc: czulsze testy kolejnosci zdarzen niz chi2 na 20 binach.
# Bateria ustalona z gory, wszystkie wyniki raportowane: (1) srednia autokorelacja dla opoznien 1,2,5,10,100 (usredniona po 6 katach),
# (2-4) ANOVA srednich w K=10,50,200 kolejnych blokach, (5-7) test Levene'a szerokosci w tych samych blokach; statystyka testow 2-7 =
# suma -log p po katach. p z R permutacji kolejnosci zdarzen (ten sam generator dla wszystkich), poprawka Bonferroniego na 7 testow.
# Rozrzut fazy: nadwyzka rozrzutu wariancji 10 blokow ponad szum probkowania, przeliczona przez dVar/dtheta z dopasowania
# Var(theta) = a cos^2 + b sin^2 + 2 c sin cos; tylko katy z |dVar/dtheta| >= 0.9 (przy 0 i 90 st. pochodna jest mala).
LAGS = (1, 2, 5, 10, 100); KB = (10, 50, 200)
def stac(rr, rng, R=200):
    xs = [rr[k].astype(float) for k in sorted(rr)]
    def ac(x): return np.mean([np.corrcoef(x[:-k], x[k:])[0, 1] for k in LAGS])
    def stats7(xs):
        s = [float(np.mean([ac(x) for x in xs]))]
        for f in (stats.f_oneway, stats.levene):
            for K in KB: s.append(float(-sum(np.log(max(f(*np.array_split(x, K)).pvalue, 1e-300)) for x in xs)))
        return np.array(s)
    obs = stats7(xs); nul = np.array([stats7([rng.permutation(x) for x in xs]) for _ in range(R)])
    p = (1 + (nul >= obs).sum(0)) / (R + 1)
    a = np.array(sorted(rr)); th = np.deg2rad(a); v = np.array([rr[k].var() for k in a])
    M = np.stack([np.cos(th) ** 2, np.sin(th) ** 2, 2 * np.sin(th) * np.cos(th)], 1); ca, cb, cc = np.linalg.lstsq(M, v, rcond=None)[0]
    dV = (cb - ca) * np.sin(2 * th) + 2 * cc * np.cos(2 * th); ph = np.full(len(a), np.nan)
    for i, k in enumerate(a):
        x = rr[k].astype(float); B = np.array_split(x, 10); vv = np.array([q.var() for q in B])
        kurt = np.mean((x - x.mean()) ** 4) / x.var() ** 2; se = x.var() * np.sqrt((kurt - 1) / len(B[0]))
        if abs(dV[i]) >= 0.9: ph[i] = np.rad2deg(np.sqrt(max(0.0, vv.var(ddof=1) - se ** 2)) / abs(dV[i]))
    return dict(names=np.array(["autokor"] + [f"anova_K{K}" for K in KB] + [f"levene_K{K}" for K in KB]), obs=obs, null_mean=nul.mean(0),
                null_sd=nul.std(0), pval=p, p_bonf=np.minimum(1, 7 * p), R=R, ac_z=float((obs[0] - nul[:, 0].mean()) / nul[:, 0].std()),
                ang=a, dV=dV, phase_sd=ph)


# ---------- 2 iid
def iid(rec, rng):
    out = {}
    for th, x in rec.items():
        n = len(x); q = np.quantile(x, np.linspace(0, 1, 21)[1:-1]); b = np.digitize(x, q)
        T = np.stack([np.bincount(blk, minlength=20) for blk in np.array_split(b, 10)])
        chi2, p, dof, _ = stats.chi2_contingency(T)
        lag = np.array([np.corrcoef(x[:-k], x[k:])[0, 1] for k in (1, 2, 5, 10, 100)])
        out[f"th{th}_chi2"] = chi2; out[f"th{th}_p"] = p; out[f"th{th}_dof"] = dof; out[f"th{th}_lag"] = lag
        hall = np.histogram(x, EDGES)[0] / n
        for N in NS:
            m = int(split_events(N, 6).max())
            blocks = [x[i:i + m] for i in range(0, n - m + 1, m)]
            dB = [np.sum((np.histogram(s, EDGES)[0] / m - hall) ** 2) for s in blocks]
            dR = [np.sum((np.histogram(rng.choice(x, m, replace=False), EDGES)[0] / m - hall) ** 2) for _ in range(max(200, len(blocks)))]
            out[f"th{th}_N{N}_dB"] = np.array(dB); out[f"th{th}_N{N}_dR"] = np.array(dR)
    ps = [out[f"th{th}_p"] for th in rec]; out["fisher_p"] = stats.combine_pvalues(ps).pvalue
    for N in NS:   # stosunek srednich odleglosci blok/losowe, zbiorczo po katach, bootstrap blokow
        B = [out[f"th{th}_N{N}_dB"] for th in rec]; R = [out[f"th{th}_N{N}_dR"] for th in rec]
        r0 = np.mean([b.mean() / r.mean() for b, r in zip(B, R)])
        bs = [np.mean([rng.choice(b, len(b)).mean() / rng.choice(r, len(r)).mean() for b, r in zip(B, R)]) for _ in range(2000)]
        out[f"ratio_N{N}"] = np.array([r0, *np.percentile(bs, [2.5, 97.5])]); out[f"nblocks_N{N}"] = np.array([len(b) for b in B])
    return out


# ---------- 3 dopasowanie
# Konno+ 2024 (arXiv:2309.02306, supl. II): stan docelowy |1_L>, hbar=1; surowe rekordy to ten stan po nieznanej operacji
# gaussowskiej (autorzy koryguja ja dopiero po rekonstrukcji). Model: |1_L> z qsdl.states.gkp_state (delta), sciskanie (r, os psi),
# strata (eta). Uwaga: qmi.gkpx._ket dla mu=1 nie daje |1_L> (brak mu w czynniku fazowym), wiec go tu nie uzywamy.
HB = 100


_A = np.diag(np.sqrt(np.arange(1, HB)), 1)


def gkp_fast(delta, mu, smax=4):
    """To samo co qsdl.states.gkp_state(HB, delta, mu, smax), ale expm z scipy (zgodnosc sprawdzana w tescie)."""
    from scipy.linalg import expm
    r = -np.log(delta); peak = expm(0.5 * r * (_A @ _A - _A.T @ _A.T))[:, 0]
    ket = sum(expm(np.sqrt(np.pi / 2) * (2 * s + mu) * (_A.T - _A)) @ peak for s in range(-smax, smax + 1))
    ket = np.exp(-delta ** 2 * np.arange(HB)) * ket; return ket / np.linalg.norm(ket)


def state(delta, r, psi, eta, mu=1):
    from scipy.linalg import expm
    z = r * np.exp(2j * psi); k = expm(0.5 * (np.conj(z) * _A @ _A - z * _A.T @ _A.T)) @ gkp_fast(delta, mu)   # = qt.squeeze(HB, z)
    k = k[:CUTOFF]; rho = qt.Qobj(np.outer(k, k.conj()) / np.vdot(k, k).real)
    return photon_loss(rho, eta) if eta < 1 else rho


def marg(rho):
    y = np.linspace(-XMAX, XMAX, GRID); W = qt.wigner(rho, y, y); S = radon(W, TH)
    P = np.clip(S, 0, None); return P / P.sum(1, keepdims=True), S, W


def dopas(rec):
    H = np.stack([np.histogram(rec[t], EDGES)[0] for t in TH]).astype(float)
    def nll(p, mu):
        delta, r, psi, eta = p
        if not (0.15 < delta < 0.95 and 0.0 <= r < 1.3 and 0.3 < eta <= 1.0): return 1e12
        P, _, _ = marg(state(delta, r, psi, eta, mu)); return -float(np.sum(H * np.log(P + 1e-12)))
    best = None
    for mu in (1, 0):
        for x0 in ([0.4, 0.7, 0.0, 0.8], [0.4, 0.7, np.pi / 2, 0.8], [0.6, 0.4, 0.3, 0.7]):
            r = optimize.minimize(nll, x0, args=(mu,), method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-3, fatol=0.5))
            log(f"  dopas mu={mu} start={x0} -> {np.round(r.x, 4)} nll={r.fun:.1f}")
            if best is None or r.fun < best[1]: best = (mu, r.fun, r.x)
    mu, f, (delta, r, psi, eta) = best
    P, S, W = marg(state(delta, r, psi, eta, mu)); Pe = H / H.sum(1, keepdims=True)
    tvd = 0.5 * np.abs(P - Pe).sum(1)
    # odniesienie: TVD miedzy dwiema polowami rekordu (sam szum zliczen przy tej liczbie zdarzen)
    half = np.array([0.5 * np.abs(np.histogram(rec[t][::2], EDGES)[0] / len(rec[t][::2]) - np.histogram(rec[t][1::2], EDGES)[0] / len(rec[t][1::2])).sum() for t in TH])
    return dict(mu=mu, delta=delta, r=r, psi=psi, eta=eta, nll=f, P=P, S=S, W=W, H=H, tvd=tvd, tvd_half=half)


# ---------- 3b rekonstrukcja MaxLik (jak u Konno+ 2024): iteracje R rho R na histogramach pikseli siatki potoku
def ml(rec, D=40, iters=3000, sub=8):
    from numpy.polynomial.hermite import hermval
    from math import factorial, pi
    H = np.stack([np.histogram(rec[t], EDGES)[0] for t in TH]).astype(float)
    xs = (EDGES[:-1, None] + (np.arange(sub)[None, :] + 0.5) / sub * DX).ravel()       # podsiatka do calkowania po pikselu
    psi = np.stack([np.exp(-xs ** 2 / 2) * hermval(xs, [0] * n + [1]) / np.sqrt(2 ** n * factorial(n) * np.sqrt(pi)) for n in range(D)], 1)
    best = None
    for sgn in (+1, -1):                                   # kierunek obrotu: <x_th|n> = exp(-i sgn n th) psi_n(x)
        Pis = []
        for t in TH:
            v = psi * np.exp(-1j * sgn * np.arange(D) * np.deg2rad(t))[None, :]       # (xs, n)
            Pis.append(np.einsum("xa,xb->xab", v.conj(), v).reshape(GRID, sub, D, D).sum(1) * DX / sub)
        Pi = np.concatenate(Pis)                           # (6*GRID, D, D), Pi[k][a,b] = <a|x><x|b>
        h = H.ravel(); rho = np.eye(D) / D
        for it in range(iters):
            pr = np.einsum("kab,ba->k", Pi, rho).real
            R = np.einsum("k,kab->ab", np.where(pr > 1e-300, h / np.maximum(pr, 1e-300), 0), Pi)
            rho = R @ rho @ R; rho = (rho + rho.conj().T) / 2; rho /= np.trace(rho).real
        r = np.zeros((CUTOFF, CUTOFF), complex); r[:D, :D] = rho              # Pi[a,b] = <a|x><x|b>, wiec rho jest w bazie |n>
        P, S, W = marg(qt.Qobj(r)); Pe = H / H.sum(1, keepdims=True); tvd = 0.5 * np.abs(P - Pe).sum(1)
        ll = float(np.sum(H * np.log(P + 1e-12)))
        log(f"  ml sgn={sgn}: TVD marginesy potoku-dane {np.round(tvd, 3)}, logL={ll:.1f}, purity={np.trace(rho @ rho).real:.3f}")
        if best is None or ll > best["ll"]: best = dict(sgn=sgn, rho=r, P=P, S=S, W=W, tvd=tvd, ll=ll, purity=np.trace(rho @ rho).real, nbar=float(np.real(np.trace(np.diag(np.arange(D)) @ rho))))
    return best


# ---------- 4 klasyfikacja
def img_counts(C, per, M):
    return fbp(np.stack([c / max(1, p) * M for c, p in zip(C, per)]), TH, GRID)


def klasyf(rec, fit, models, M, rng, draws):
    from qmi.eval_shots import pred
    out = {}; nmin = min(len(v) for v in rec.values())
    for N in (*NS, 0):                                   # 0 = caly rekord
        per = split_events(N, 6) if N else np.array([len(rec[t]) for t in TH])
        X = {"exp_los": [], "exp_blok": [], "sym": []}
        for _ in range(draws if N else 1):
            X["exp_los"].append(img_counts([np.histogram(rec[t] if not N else rng.choice(rec[t], p, replace=False), EDGES)[0] for t, p in zip(TH, per)], per, M))
            X["sym"].append(img_counts([rng.multinomial(p, fit["P"][k]) for k, p in enumerate(per)], per, M))
        nb = nmin // per.max() if N else 1
        for b in range(nb):
            X["exp_blok"].append(img_counts([np.histogram(rec[t][b * p:(b + 1) * p] if N else rec[t], EDGES)[0] for t, p in zip(TH, per)], per, M))
        for src, imgs in X.items():
            imgs = np.stack(imgs).astype(np.float32)
            for nm, (m, rep, sc) in models.items():
                out[f"{nm}|{src}|N{N}"] = np.bincount(pred(m, imgs, sc), minlength=7)
        log(f"  N={N or 'caly'}: " + "; ".join(f"{src} {dict(zip(CLASSES, sum(out[f'{nm}|{src}|N{N}'] for nm in models)))}" for src in X))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/dryad_gkp"); ap.add_argument("--runs", default="runs_x8")
    ap.add_argument("--out", default="runs_x8/konno"); ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--etapy", default="kalib,iid,dopas,ml,klasyf")
    ap.add_argument("--siec", default=r"re:^clf_W_(shots|gkpx)K6_s\d+$"); ap.add_argument("--nsieci", type=int, default=6)
    ap.add_argument("--klasyf-out", default="klasyf.json"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); LOG = str(out / "konno.log"); t0 = time.time(); et = a.etapy.split(",")
    rr = raw(a.data); log(f"start {time.ctime()} XMAX={XMAX} GRID={GRID} zdarzen={ {k: len(v) for k, v in rr.items()} }")
    if not (out / "kalib.npz").exists(): save(out / "kalib.npz", **kalib(rr))
    K = dict(np.load(out / "kalib.npz"))
    log(f"kalib: c={K['c']:.4f}+-{K['c_se']:.4f} x0={K['x0']:.4f}+-{K['x0_se']:.4f} p0={K['p0']:.4f}+-{K['p0_se']:.4f} "
        f"chi2: offset+przes {K['chi2_full']:.1f} (3 dof), samo przes {K['chi2_disp']:.1f} (4 dof), sam offset {K['chi2_off']:.1f} (5 dof); "
        f"piki 0st {np.round(K['peaks'], 3)}, odstep {K['spacing']:.4f}+-{K['spacing_sd']:.4f} (sqrt(pi)={np.sqrt(np.pi):.4f})")
    rec = records(rr, float(K["c"])); rng = np.random.default_rng(20261019)
    if "iid" in et and not (out / "iid.npz").exists(): save(out / "iid.npz", **iid(rec, rng)); log(f"iid gotowe {time.time()-t0:.0f}s")
    if (out / "iid.npz").exists():
        I = np.load(out / "iid.npz")
        log("iid: p chi2 blokow " + str({th: round(float(I[f'th{th}_p']), 4) for th in TH}) + f" Fisher p={float(I['fisher_p']):.3g}; "
            + "; ".join(f"N={N}: blok/los {np.round(I[f'ratio_N{N}'], 3)}" for N in NS))
    if "stac" in et and not (out / "stac.npz").exists():
        save(out / "stac.npz", **stac(rr, np.random.default_rng(20261020))); S = np.load(out / "stac.npz")
        log("stac: " + "; ".join(f"{nm} p={pp:.3f}" for nm, pp in zip(S["names"], S["pval"])) + f"; z autokor {float(S['ac_z']):.1f}; faza sd {np.round(S['phase_sd'], 1)} st")
    if "dopas" in et and not (out / "dopas.npz").exists(): save(out / "dopas.npz", **dopas(rec)); log(f"dopas gotowe {time.time()-t0:.0f}s")
    if (out / "dopas.npz").exists():
        F = dict(np.load(out / "dopas.npz"))
        log(f"dopas: mu={int(F['mu'])} delta={float(F['delta']):.3f} r={float(F['r']):.3f} psi={float(F['psi']):.3f} eta={float(F['eta']):.3f} "
            f"TVD model-dane {np.round(F['tvd'], 3)} TVD polowa-polowa {np.round(F['tvd_half'], 3)}")
    if "ml" in et and not (out / "ml.npz").exists(): save(out / "ml.npz", **ml(rec)); log(f"ml gotowe {time.time()-t0:.0f}s")
    if (out / "ml.npz").exists():
        G = dict(np.load(out / "ml.npz")); F = dict(F, P=G["P"])          # rekordy symulowane z rekonstrukcji MaxLik
        log(f"ml: sgn={int(G['sgn'])} TVD {np.round(G['tvd'], 3)} purity {float(G['purity']):.3f} nbar {float(G['nbar']):.2f}")
    KO = out / a.klasyf_out
    if "klasyf" in et and not KO.exists():
        from qmi.eval_shots import load_models
        models = load_models(a.runs, a.siec); log("modele:", sorted(models))
        if len(models) != a.nsieci: sys.exit(f"BLAD: {len(models)} modeli zamiast {a.nsieci}")
        y = np.linspace(-XMAX, XMAX, GRID); M = float(radon(qt.wigner(qt.fock_dm(CUTOFF, 0), y, y), [0])[0].sum())
        res = klasyf(rec, F, models, M, rng, a.draws)
        json.dump({k: v.tolist() for k, v in res.items()}, open(str(KO) + ".tmp", "w")); os.replace(str(KO) + ".tmp", KO)
    log(f"KONNO_DONE {time.time()-t0:.0f}s")
