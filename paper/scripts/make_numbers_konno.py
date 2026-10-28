"""Liczby do podsekcji o rekordach eksperymentalnych GKP (Konno+ 2024) -> results/numbers_konno.tex.
Zrodlo: kopia wynikow qmi.x8.konno w results/konno/ (kalib.npz, iid.npz, dopas.npz, ml.npz, klasyf_*.json).
  RES=results ~/anaconda3/bin/python scripts/make_numbers_konno.py"""
import json, os, sys
import numpy as np
from scipy import stats

R = os.path.join(os.environ.get("RES", os.path.join(os.path.dirname(__file__), "..", "results")), "konno")
OUT = os.path.join(os.environ.get("OUTDIR", os.path.dirname(R)), "numbers_konno.tex")
C = ("fock", "coherent", "vacuum", "thermal", "cat", "gkp", "binomial")
NS = (100, 1000, 10000)
K = {}


def put(k, v, fmt="{:.1f}", pct=False):
    K["konno_" + k] = (fmt.format(v) + (r"\%" if pct else "")) if not isinstance(v, str) else v


def num(x, d=2):
    return r"\ensuremath{" + f"{x:.{d}f}".replace("-", "-") + "}"


ka = np.load(os.path.join(R, "kalib.npz")); ii = np.load(os.path.join(R, "iid.npz"))
dp = np.load(os.path.join(R, "dopas.npz")); ml = np.load(os.path.join(R, "ml.npz"))

# kalibracja
n = ka["n"]; put("nmin", int(n.min()), "{:,}"); put("nmax", int(n.max()), "{:,}"); put("ntot", int(n.sum()), "{:,}")
K["konno_nmin"] = K["konno_nmin"].replace(",", "\\,"); K["konno_nmax"] = K["konno_nmax"].replace(",", "\\,"); K["konno_ntot"] = K["konno_ntot"].replace(",", "\\,")
put("offset", num(float(ka["c"]), 2)); put("offset_se", f"{float(ka['c_se']):.2f}")
put("meanlo", num(float(ka["mean"].min()), 2)); put("meanhi", num(float(ka["mean"].max()), 2))
put("disp", f"{max(abs(float(ka['x0'])), abs(float(ka['p0']))):.2f}")
# skladowe przesuniecia z niepewnosciami; chi2 dopasowania (3 st. swobody) i niepewnosc przeskalowana przez sqrt(chi2/3), gdy chi2>3
_s = max(1.0, np.sqrt(float(ka["chi2_full"]) / 3))
put("x0", num(float(ka["x0"]), 3)); put("x0_se", f"{float(ka['x0_se']):.3f}"); put("p0", num(float(ka["p0"]), 3)); put("p0_se", f"{float(ka['p0_se']):.3f}")
put("chi2_full", f"{float(ka['chi2_full']):.1f}"); put("chi2_full_p", f"{stats.chi2.sf(float(ka['chi2_full']), 3):.3f}")
put("p0_sig", f"{abs(float(ka['p0'])) / float(ka['p0_se']):.1f}"); put("p0_sig_resc", f"{abs(float(ka['p0'])) / (float(ka['p0_se']) * _s):.1f}")
put("p0_pix", f"{abs(float(ka['p0'])) / (16 / 127):.2f}")
put("dxpix", f"{16 / 127:.3f}")
put("peaklo", num(float(ka["peaks"][0]), 2)); put("peakhi", num(float(ka["peaks"][1]), 2))
put("spacing", f"{float(ka['spacing']):.2f}"); put("spacing_sd", f"{float(ka['spacing_sd']):.2f}")
put("sqrtpi", f"{np.sqrt(np.pi):.2f}")

# niezaleznosc zdarzen
put("iid_p", f"{float(ii['fisher_p']):.2f}")
put("iid_pmin", f"{min(float(ii[f'th{t}_p']) for t in (0, 30, 60, 90, 120, 150)):.2f}")
for N in NS:
    r = ii[f"ratio_N{N}"]; put(f"ratio_N{N}", f"{r[0]:.2f}"); put(f"ratio_N{N}_ci", r"\ensuremath{[" + f"{r[1]:.2f},\\,{r[2]:.2f}" + "]}")
    put(f"nblocks_N{N}", int(ii[f"nblocks_N{N}"].min()), "{:d}")

# stacjonarnosc (etap stac w qmi/x8/konno.py): 7 testow, p z permutacji, Bonferroni
_sp = os.path.join(R, "stac.npz")
if os.path.exists(_sp):
    st = np.load(_sp); nm = [str(x) for x in st["names"]]
    for k, pp, pb in zip(nm, st["pval"], st["p_bonf"]): put(f"stac_{k}_p", f"{float(pp):.3f}"); put(f"stac_{k}_pbonf", f"{float(pb):.3f}")
    put("stac_R", int(st["R"]), "{:d}"); put("stac_pmin_res", f"{1 / (int(st['R']) + 1):.3f}"); put("stac_ac_z", f"{float(st['ac_z']):.1f}")
    put("stac_nsig", int((st["p_bonf"] < 0.05).sum()), "{:d}")
    ph = st["phase_sd"][np.isfinite(st["phase_sd"])]; put("stac_phase_min", f"{ph.min():.0f}"); put("stac_phase_max", f"{ph.max():.0f}")
    put("stac_phase_nang", len(ph), "{:d}")

# prosty model i MaxLik
put("delta", f"{float(dp['delta']):.2f}"); put("r", f"{float(dp['r']):.2f}"); put("sqdb", f"{8.686 * float(dp['r']):.1f}")
put("eta", f"{float(dp['eta']):.2f}"); put("tvd_par_max", f"{float(dp['tvd'].max()):.2f}"); put("tvd_par_min", f"{float(dp['tvd'].min()):.2f}")
put("tvd_half_min", f"{float(dp['tvd_half'].min()):.2f}"); put("tvd_half_max", f"{float(dp['tvd_half'].max()):.2f}")
put("tvd_ml_min", f"{float(ml['tvd'].min()):.2f}"); put("tvd_ml_max", f"{float(ml['tvd'].max()):.2f}")
put("ml_purity", f"{float(ml['purity']):.2f}"); put("ml_nbar", f"{float(ml['nbar']):.2f}")

# klasyfikacja
def klas(fn, tag, nsieci):
    p = os.path.join(R, fn)
    if not os.path.exists(p): return
    d = {k: np.array(v) for k, v in json.load(open(p)).items()}; nets = sorted({k.split("|")[0] for k in d})
    if len(nets) != nsieci: sys.exit(f"BLAD: {fn}: {len(nets)} sieci zamiast {nsieci}")
    put(f"{tag}_nsieci", len(nets), "{:d}"); pv = []; pb_ = []
    for N in (*NS, 0):
        for src in ("exp_los", "sym"):
            t = sum(d[f"{nn}|{src}|N{N}"] for nn in nets); f = 100 * t / t.sum()
            for c, x in zip(C, f): put(f"{tag}_{src}_N{N}_{c}", float(x), "{:.0f}", pct=True)
        if N:
            for nn in nets:   # to samo dla kolejnych blokow (exp_blok) wobec sym; tu wchodzi kolejnosc zdarzen
                a, b = d[f"{nn}|exp_blok|N{N}"], d[f"{nn}|sym|N{N}"]; m = (a + b) > 0
                if m.sum() > 1: pb_.append(stats.chi2_contingency(np.stack([a[m], b[m]]))[1])
            for nn in nets:   # test jednorodnosci exp_los wobec sym (klasy z co najmniej jednym zliczeniem)
                a, b = d[f"{nn}|exp_los|N{N}"], d[f"{nn}|sym|N{N}"]; m = (a + b) > 0
                if m.sum() > 1: pv.append(stats.chi2_contingency(np.stack([a[m], b[m]]))[1])
            tv = max(0.5 * np.abs(d[f"{nn}|exp_los|N{N}"] / d[f"{nn}|exp_los|N{N}"].sum() - d[f"{nn}|sym|N{N}"] / d[f"{nn}|sym|N{N}"].sum()).sum() for nn in nets)
            put(f"{tag}_tvd_N{N}", f"{tv:.2f}")
    put(f"{tag}_draws", int(sum(d[f"{nets[0]}|exp_los|N100"])), "{:d}")
    put(f"{tag}_blok_ntest", len(pb_), "{:d}"); put(f"{tag}_blok_pmin", f"{min(pb_):.2f}")
    put(f"{tag}_ntest", len(pv), "{:d}"); put(f"{tag}_pmin", f"{min(pv):.2f}"); put(f"{tag}_pmin_bonf", f"{min(1, min(pv) * len(pv)):.2f}")


klas("klasyf_baza.json", "base", 3)

# dokladnosc sieci K=6 (bazowe i z rozszerzona klasa GKP) na tescie bazowym
for fn, tag in (("eval_shots_Kscan_K6.json", "base"), ("eval_shots_gkpxK6.json", "gkpx")):
    d = json.load(open(os.path.join(R, fn))); nets = sorted(d)
    if len(nets) != 3: sys.exit(f"BLAD: {fn}: {len(nets)} sieci")
    for N, lv in ((100, 0), (100, 80), (10000, 0), (10000, 80)):
        put(f"{tag}_acc_N{N}_loss{lv}", 100 * np.mean([d[m]["loss"][str(N)][str(lv)]["acc"] for m in nets]), "{:.1f}", pct=True)
klas("klasyf_gkpx.json", "gkpx", 3)

with open(OUT, "w") as f:
    f.write("% generated by scripts/make_numbers_konno.py -- do not edit by hand\n")
    for k in sorted(K): f.write(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{K[k]}}}\n")
print(f"{len(K)} kluczy -> {OUT}")
