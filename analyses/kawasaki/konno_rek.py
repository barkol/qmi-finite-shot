"""Rekordy GKP Konno+ 2024: test na poziomie rekordu (kawa.rektest, B=40) i dokladny test permutacyjny blokow (agg.exact)
dla sieci z pracy (bazowe K6, gkpx K6), tym samym kodem co dla Kawasakiego. Wejscie (tylko odczyt): data/dryad_gkp,
runs_eq/konno/{kalib,ml}.npz, runs_eq/konno/klasyf_{baza,gkpx}.json, sieci runs_eq. Wynik: out_konno/.
  QMI_XMAX=12 python -I analyses/kawasaki/konno_rek.py"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
import kawa
from kawa import rektest, radon, CUTOFF, GRID, XMAX, NS, save

ROOT = Path(os.environ.get("QMI_ROOT", str(Path(__file__).resolve().parents[2]))); K0 = ROOT / "runs_eq/konno"; OUT = Path(os.path.dirname(os.path.abspath(__file__))) / "out_konno"
OUT.mkdir(exist_ok=True); kawa.LOG = str(OUT / "konno_rek.log")
FILES = {0: 0, 30: 30, 60: 60, -30: 150, -60: 120, -90: 90}       # jak qmi.x8.konno


def exact(a, b, R=20000, seed=0):   # kopia agg.exact
    m = (a + b) > 0; a, b = a[m], b[m]
    if m.sum() < 2: return 1.0
    tot = a + b; na, nb = a.sum(), b.sum(); n = na + nb; e = tot / n
    def T(x): y = tot - x; return np.sum((x - na * e) ** 2 / (na * e) + (y - nb * e) ** 2 / (nb * e), axis=-1)
    sims = np.random.default_rng(seed).multivariate_hypergeometric(tot, na, size=R)
    return float((1 + np.sum(T(sims) >= T(a) - 1e-9)) / (R + 1))


def chi(a, b):
    m = (a + b) > 0
    return stats.chi2_contingency(np.stack([a[m], b[m]]))[1] if m.sum() > 1 else None


rr = {a: np.load(ROOT / "data/dryad_gkp" / f"quad_{a}deg.npy").astype(float) for a in FILES}
c = float(np.load(K0 / "kalib.npz")["c"]); rec = {FILES[a]: (-(x - c) if a < 0 else x - c) for a, x in rr.items()}
P = np.load(K0 / "ml.npz")["P"]
kawa.log(f"Konno: c={c:.4f} zdarzen {[len(rec[t]) for t in sorted(rec)]}")
res = {}
for nm, rx in (("baza", r"re:^clf_W_shotsK6_s\d+$"), ("gkpx", r"re:^clf_W_gkpxK6_s\d+$")):
    k = {a: np.array(b) for a, b in json.load(open(K0 / f"klasyf_{nm}.json")).items()}; nets = sorted({a.split("|")[0] for a in k})
    if len(nets) != 3: sys.exit(f"BLAD {nm}: {nets}")
    naive_los = [p for N in NS for n in nets if (p := chi(k[f"{n}|exp_los|N{N}"], k[f"{n}|sym|N{N}"])) is not None]
    naive_blok = [p for N in NS for n in nets if (p := chi(k[f"{n}|exp_blok|N{N}"], k[f"{n}|sym|N{N}"])) is not None]
    ex_blok = {f"{n}|N{N}": exact(k[f"{n}|exp_blok|N{N}"], k[f"{n}|sym|N{N}"]) for N in NS for n in nets}
    RT = OUT / f"rektest_{nm}.npz"
    if not RT.exists():
        import qutip as qt
        from qmi.eval_shots import load_models
        models = load_models(str(ROOT / "runs_eq"), rx)
        if sorted(models) != nets: sys.exit(f"BLAD {nm}: sieci {sorted(models)} != {nets}")
        y = np.linspace(-XMAX, XMAX, GRID); M = float(radon(qt.wigner(qt.fock_dm(CUTOFF, 0), y, y), [0])[0].sum())
        t0 = time.time(); save(RT, **rektest(rec, P, models, M, np.random.default_rng(20261025 + len(nm)))); kawa.log(f"rektest {nm} {time.time()-t0:.0f}s")
    R_ = dict(np.load(RT))
    rek = {f"{n}|N{N}": float(R_[f"{n}|N{N}|p"]) for N in NS for n in nets}; rek_suma = {f"N{N}": float(R_[f"suma|N{N}|p"]) for N in NS}
    res[nm] = dict(naive_los_n=len(naive_los), naive_los_pmin=min(naive_los), naive_blok_n=len(naive_blok), naive_blok_pmin=min(naive_blok),
                   exact_blok=ex_blok, exact_blok_pmin=min(ex_blok.values()), exact_blok_n=len(ex_blok), rek=rek, rek_pmin=min(rek.values()), rek_n=len(rek),
                   rek_suma=rek_suma, nblok={f"N{N}": int(k[f"{nets[0]}|exp_blok|N{N}"].sum()) for N in NS},
                   T_exp={f"N{N}": float(R_[f"suma|N{N}|T_exp"]) for N in NS}, T_null_med={f"N{N}": float(np.median(R_[f"suma|N{N}|T_null"])) for N in NS})
    kawa.log(f"{nm}: naiwny chi2 los n={len(naive_los)} pmin={min(naive_los):.3f}; bloki chi2 pmin={min(naive_blok):.3f}; "
             f"bloki dokladny pmin={min(ex_blok.values()):.3f}; rektest pmin={min(rek.values()):.3f} (n={len(rek)}), suma {rek_suma}")
json.dump(res, open(OUT / "konno_rek.json", "w"), indent=1); print("KONNO_REK_DONE")
