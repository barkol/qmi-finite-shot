"""Krok 3: rekordy JEDNEGO modu (hd1; hd2 zapisywane do kontroli) z kwadratur pakietow (krok 2) + testy jak qmi.x8.konno (iid, stac).
Rekord: co STRIDE-ty pakiet w ramce (STRIDE=40 pakietow = 1.6 ns; autokorelacja pakietow |C(m)| <= 0.003 dla m >= 10, przy m=1 -0.08
z odpowiedzi detektora, widoczna tez w szumie srutowym), ramki w kolejnosci czasu (co 400 us). x -> kat 0, p -> kat 90.
n = (Var x + Var p)/2 - 1/2 (zdarzenia: szum elektroniki zawarty, dziala jak strata); n_el = to samo po odjeciu szumu elektroniki.
Wznowienie: gotowe rekordy/<zestaw>.npz pomijane. Stop na pierwszym bledzie.
  QMI_XMAX=12 python -I krok3_rekordy.py"""
import json, os, sys, time
from pathlib import Path
import numpy as np
if os.environ.get("QMI_XMAX") != "12": sys.exit("BLAD: QMI_XMAX musi byc 12 (siatka eq)")
sys.path.insert(0, os.environ.get("QMI_ROOT", str(Path(__file__).resolve().parents[2])))
from scipy import stats
from qmi.x8 import konno          # iid, stac (bez zmian w oryginale)

OUT = Path(__file__).resolve().parent; D = OUT / "dane"; R = OUT / "rekordy"; LOG = OUT / "krok3.log"; STRIDE = 40


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); open(LOG, "a").write(s + "\n")


def rekord(q): return np.ascontiguousarray(q[:, ::STRIDE]).ravel().astype(np.float64)


def opis(x, p, meta, h, rng):
    vx, vp = x.var(), p.var(); nb = (vx + vp) / 2 - 0.5
    bs = []
    for _ in range(300):
        i = rng.integers(0, len(x), len(x)); j = rng.integers(0, len(p), len(p)); bs.append((x[i].var() + p[j].var()) / 2 - 0.5)
    sx = meta[f"var_el_corr_x{h}"] / meta[f"var_x{h}"]; sp = meta[f"var_el_corr_p{h}"] / meta[f"var_p{h}"]   # wspolczynnik korekty el.
    nbe = (vx * sx + vp * sp) / 2 - 0.5
    sk = {}
    for nm, z in (("x", x), ("p", p)):
        sk[f"kurt_{nm}"] = float(stats.kurtosis(z)); sk[f"skew_{nm}"] = float(stats.skew(z))
        sk[f"ks_p_{nm}"] = float(stats.kstest(z / z.std(), "norm").pvalue)
        sk[f"ad_{nm}"] = float(stats.anderson(z / z.std()).statistic)
    return dict(var_x=float(vx), var_p=float(vp), nbar=float(nb), nbar_se=float(np.std(bs)), nbar_el=float(nbe),
                aniso_db=float(10 * np.log10(vp / vx)), mean_x=float(x.mean()), mean_p=float(p.mean()), n_ev=int(len(x)), eta_el=float(meta[f"eta_el_hd{h}"]), **sk)


if __name__ == "__main__":
    R.mkdir(exist_ok=True); log(f"start {time.ctime()} STRIDE={STRIDE}")
    files = sorted(D.glob("*.npz"))
    if len(files) != 14: raise RuntimeError(f"{len(files)} zestawow zamiast 14")
    for f in files:
        o = R / f.name
        if o.exists(): log(f"pomijam (gotowe) {o.name}"); continue
        t0 = time.time(); d = np.load(f); meta = json.loads(str(d["meta"])); rng = np.random.default_rng(20261022)
        rec = {h: (rekord(d[f"x{h}"]), rekord(d[f"p{h}"])) for h in "12"}
        res = {h: opis(*rec[h], meta, h, rng) for h in "12"}
        x, p = rec["1"]; rr = {0: x, 90: p}
        I = konno.iid(rr, rng); S = konno.stac(rr, np.random.default_rng(20261020), R=200)
        res["iid"] = dict(fisher_p=float(I["fisher_p"]), p0=float(I["th0_p"]), p90=float(I["th90_p"]),
                          lag0=I["th0_lag"].tolist(), lag90=I["th90_lag"].tolist(), **{f"ratio_N{N}": I[f"ratio_N{N}"].tolist() for N in konno.NS})
        res["stac"] = dict(names=S["names"].tolist(), pval=S["pval"].tolist(), p_bonf=S["p_bonf"].tolist(), ac_z=float(S["ac_z"]), obs=S["obs"].tolist())
        res["meta"] = meta
        np.savez(str(o) + ".tmp.npz", x1=rec["1"][0].astype(np.float32), p1=rec["1"][1].astype(np.float32), x2=rec["2"][0].astype(np.float32),
                 p2=rec["2"][1].astype(np.float32), wyniki=json.dumps(res)); os.replace(str(o) + ".tmp.npz", o)
        json.dump(res, open(R / (f.stem + ".json"), "w"), indent=1)
        r1 = res["1"]
        log(f"{f.stem}: n={r1['nbar']:.3f}+-{r1['nbar_se']:.3f} (el-kor {r1['nbar_el']:.3f}; hd2 {res['2']['nbar']:.3f}) Vx {r1['var_x']:.3f} Vp {r1['var_p']:.3f} "
            f"aniz {r1['aniso_db']:.2f} dB kurt {r1['kurt_x']:.3f}/{r1['kurt_p']:.3f} | iid Fisher p={res['iid']['fisher_p']:.3g} "
            f"blok/los N1e4 {np.round(res['iid']['ratio_N10000'], 3)} | stac p " + " ".join(f"{a}:{b:.3f}" for a, b in zip(res['stac']['names'], res['stac']['pval']))
            + f" | {time.time()-t0:.0f}s")
    log(f"KROK3_DONE {time.ctime()}")
