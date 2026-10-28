"""Agregacja wynikow kawa.py -> out/summary.json + out/tabele.md. Testy roznic exp/sym jak scripts/make_numbers_konno.py:
chi2 jednorodnosci rozkladow klas (klasy z >=1 zliczeniem) dla kazdej sieci i N; dla calego rekordu: klasa obrazu z calego rekordu
wobec rozkladu klas 200 rekordow symulowanych (p = udzial rekordow symulowanych w tej klasie) oraz chi2 bootstrap rekordu wobec sym.
  python -I analyses/kawasaki/agg.py"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path(str(Path(__file__).resolve().parent / "out"))
C = ("fock", "coherent", "vacuum", "thermal", "cat", "gkp", "binomial")
NS = (100, 1000, 10000, 0)
LOSS = ["01mW_00dB_2", "01mW_01dB_2", "01mW_02dB_2", "01mW_05dB_2", "01mW_10dB_2", "01mW_15dB_2", "01mW_20dB_2"]
PUMP = ["01mW_00dB", "03mW_00dB", "10mW_00dB", "25mW_00dB"]
TAGS = PUMP + LOSS
PAPER_W = {"01mW_00dB": (-0.090, 0.004), "03mW_00dB": (-0.097, 0.005), "10mW_00dB": (-0.088, 0.005), "25mW_00dB": (-0.066, 0.006)}  # arXiv:2402.17408 (hbar=1)
f = lambda x: float(x)


def L(t, nm):
    p = OUT / t / f"{nm}.npz"
    if not p.exists(): sys.exit(f"BLAD: brak {p}")
    return dict(np.load(p))


def chi(a, b):
    m = (a + b) > 0
    return stats.chi2_contingency(np.stack([a[m], b[m]]))[1] if m.sum() > 1 else 1.0


def exact(a, b, R=20000, seed=0):
    """Dokladny test permutacyjny tabeli 2xK (warunkowy na brzegach, uogolnienie testu Fishera): wiersze = bloki exp i rekordy sym;
    przy H0 przydzial zliczen klas do wierszy jest wielowymiarowo hipergeometryczny; statystyka chi2 Pearsona; p = (1 + #(T >= T_obs)) / (R + 1)."""
    m = (a + b) > 0; a, b = a[m], b[m]
    if m.sum() < 2: return 1.0
    tot = a + b; na, nb = a.sum(), b.sum(); n = na + nb; e = tot / n
    def T(x): y = tot - x; return np.sum((x - na * e) ** 2 / (na * e) + (y - nb * e) ** 2 / (nb * e), axis=-1)
    rng = np.random.default_rng(seed); sims = rng.multivariate_hypergeometric(tot, na, size=R)
    return float((1 + np.sum(T(sims) >= T(a) - 1e-9)) / (R + 1))


S = {"pliki": {}, "testy": {}, "uwagi": []}
for t in TAGS:
    K, St, I, G, Sk, P = (L(t, n) for n in ("kalib", "stac", "iid", "ml", "skala", "pss"))
    tab = Sk["tab"]; cols = list(Sk["cols"]); s = tab[:, 0]
    d = dict(c=f(K["c"]), c_se=f(K["c_se"]), x0=f(K["x0"]), x0_se=f(K["x0_se"]), p0=f(K["p0"]), p0_se=f(K["p0_se"]), chi2_mean_fit=f(K["chi2_full"]),
             var=K["var"].round(4).tolist(), kurt=K["kurt"].round(3).tolist(), nbar_mom=f(K["nbar_v05"]), nbar_mom_se=f(K["nbar_v05_se"]),
             nbar_mom_if_vac025=f(K["nbar_v025"]),
             stac_p=dict(zip([str(x) for x in St["names"]], St["pval"].round(4).tolist())), stac_nsig_bonf=int((St["p_bonf"] < 0.05).sum()),
             stac_phase_sd_deg=[None if not np.isfinite(x) else round(f(x), 2) for x in St["phase_sd"]], stac_ac_z=f(St["ac_z"]),
             iid_fisher_p=f(I["fisher_p"]), iid_p=[f(I[f"th{th}_p"]) for th in (0, 30, 60, 90, 120, 150)],
             iid_ratio={str(N): I[f"ratio_N{N}"].round(3).tolist() for N in (100, 1000, 10000)},
             ml=dict(sgn=int(G["sgn"]), purity=f(G["purity"]), nbar=f(G["nbar"]), W0=f(G["W0"]), Wmin=f(G["Wmin"]), Wmin_at=[f(G["Wmin_x"]), f(G["Wmin_p"])],
                     p0=f(G["p0"]), p1=f(G["p1"]), p2=f(G["p2"]), podd=f(G["podd"]), boot_sd=dict(zip(("purity", "nbar", "W0", "Wmin"), G["boot_sd"].round(4).tolist())),
                     nboot=int(len(G["boot"])), tvd_max=f(G["tvd"].max()), tvd_half_max=f(G["tvd_half"].max()), ll_marg=f(G["ll_marg"]), ll_marg_other_sgn=f(G["ll_marg_other"]),
                     conv_last500=f(np.diff(G["ll_hist"])[-1])),
             skala=dict(s=s.tolist(), logL_ml_jac=tab[:, cols.index("logLj")].round(1).tolist(), logL_pss_jac=tab[:, cols.index("logLj_pss")].round(1).tolist(),
                        W0=tab[:, cols.index("W0")].round(4).tolist(), nbar=tab[:, cols.index("nbar")].round(3).tolist(),
                        pss_eta=tab[:, cols.index("pss_eta")].round(3).tolist(), pss_r=tab[:, cols.index("pss_r")].round(3).tolist(),
                        s_best_pss=f(s[np.nanargmax(tab[:, cols.index("logLj_pss")])]), s_best_ml=f(s[np.nanargmax(tab[:, cols.index("logLj")])])),
             pss={k: (f(P[k]) if P[k].ndim == 0 else P[k].round(4).tolist()) for k in P})
    if (OUT / t / "negN.npz").exists():
        Ng = L(t, "negN"); d["negN"] = {k: dict(frac_W0_neg=f((v[:, 0] < 0).mean()), frac_Wmin_lt_m0p01=f((v[:, 1] < -0.01).mean()), W0_med=f(np.median(v[:, 0])),
                                                Wmin_med=f(np.median(v[:, 1])), nsub=len(v)) for k, v in Ng.items()}
    if (OUT / t / "cent.npz").exists():
        Ce = L(t, "cent"); d["ml_centrowanie_per_faza"] = {k: f(Ce[k]) for k in ("W0", "Wmin", "purity", "nbar", "ll_marg")}
    if t in PAPER_W: d["paper_W0"] = PAPER_W[t]
    # klasyfikacja
    for nm in ("baza", "gkpx", "ahm"):
        p = OUT / t / f"klasyf_{nm}.json"
        if not p.exists(): continue
        k = {a: np.array(b) for a, b in json.load(open(p)).items()}; nets = sorted({a.split("|")[0] for a in k})
        if len(nets) != 3: sys.exit(f"BLAD {p}: {len(nets)} sieci")
        r = {}
        for N in NS:
            srcs = ("exp_los", "exp_blok", "sym") if N else ("exp_los", "exp_boot", "sym")
            fr = {src: (sum(k[f"{n}|{src}|N{N}"] for n in nets)) for src in srcs}
            e = dict(frac={src: dict(zip(C, (v / v.sum()).round(4).tolist())) for src, v in fr.items()},
                     n={src: int(v.sum()) for src, v in fr.items()})
            if N:
                e["p_los"] = [chi(k[f"{n}|exp_los|N{N}"], k[f"{n}|sym|N{N}"]) for n in nets]
                e["p_blok"] = [chi(k[f"{n}|exp_blok|N{N}"], k[f"{n}|sym|N{N}"]) for n in nets]
                e["p_blok_exact"] = [exact(k[f"{n}|exp_blok|N{N}"], k[f"{n}|sym|N{N}"]) for n in nets]
                e["tvd_los_sym"] = [0.5 * f(np.abs(k[f"{n}|exp_los|N{N}"] / k[f"{n}|exp_los|N{N}"].sum() - k[f"{n}|sym|N{N}"] / k[f"{n}|sym|N{N}"].sum()).sum()) for n in nets]
            else:
                cls = [int(np.argmax(k[f"{n}|exp_los|N0"])) for n in nets]
                e["full_class"] = [C[c] for c in cls]
                e["p_full"] = [f(k[f"{n}|sym|N0"][c] / k[f"{n}|sym|N0"].sum()) for n, c in zip(nets, cls)]
                e["p_boot"] = [chi(k[f"{n}|exp_boot|N0"], k[f"{n}|sym|N0"]) for n in nets]
            r[str(N)] = e
        pr = OUT / t / f"rektest_{nm}.npz"
        if pr.exists():
            R_ = dict(np.load(pr))
            for N in (100, 1000, 10000):
                r[str(N)]["rek_p_suma"] = f(R_[f"suma|N{N}|p"]); r[str(N)]["rek_p_sieci"] = [f(R_[f"{n}|N{N}|p"]) for n in nets]
                r[str(N)]["rek_T_exp"] = f(R_[f"suma|N{N}|T_exp"]); r[str(N)]["rek_T_null_med"] = f(np.median(R_[f"suma|N{N}|T_null"])); r[str(N)]["rek_B"] = len(R_[f"suma|N{N}|T_null"])
        d[f"klasyf_{nm}"] = r
    S["pliki"][t] = d

# zbiorcze testy (jak make_numbers_konno: wszystkie sieci x N, osobno losowe podzbiory i bloki)
for nm in ("baza", "gkpx", "ahm"):
    pl, pb, pf, prs, prn, pbe = [], [], [], [], [], []
    for t in TAGS:
        r = S["pliki"][t].get(f"klasyf_{nm}")
        if not r: continue
        for N in (100, 1000, 10000): pl += r[str(N)]["p_los"]; pb += r[str(N)]["p_blok"]; pbe.extend(r[str(N)]["p_blok_exact"])
        pf += r["0"]["p_boot"]
        for N in (100, 1000, 10000):
            if "rek_p_suma" in r[str(N)]: prs.append(r[str(N)]["rek_p_suma"]); prn += r[str(N)]["rek_p_sieci"]
    if pl:
        S["testy"][nm] = dict(n_los=len(pl), pmin_los=min(pl), pmin_los_bonf=min(1, min(pl) * len(pl)), n_sig05_los=int(sum(p < 0.05 for p in pl)),
                              n_blok=len(pb), pmin_blok=min(pb), pmin_blok_bonf=min(1, min(pb) * len(pb)), n_sig05_blok=int(sum(p < 0.05 for p in pb)),
                              n_boot=len(pf), pmin_boot=min(pf), n_sig05_boot=int(sum(p < 0.05 for p in pf)),
                              fisher_los=f(stats.combine_pvalues(pl).pvalue), fisher_blok=f(stats.combine_pvalues(pb).pvalue),
                              uwaga_los_boot="chi2 exp_los i exp_boot zaklada niezalezne losowania; podzbiory jednego rekordu 5000/kat sa zalezne -> p zanizone; poprawny test = rektest",
                              n_blok_exact=len(pbe), pmin_blok_exact=min(pbe), pmin_blok_exact_bonf=min(1, min(pbe) * len(pbe)), n_sig05_blok_exact=int(sum(p < 0.05 for p in pbe)),
                              n_rek=len(prs), pmin_rek=min(prs) if prs else None, n_sig05_rek=int(sum(p < 0.05 for p in prs)), n_rek_sieci=len(prn),
                              pmin_rek_sieci=min(prn) if prn else None, n_sig05_rek_sieci=int(sum(p < 0.05 for p in prn)),
                              n_full=sum(1 for t in TAGS if S["pliki"][t].get(f"klasyf_{nm}")) * 3,
                              pmin_full=min(min(S["pliki"][t][f"klasyf_{nm}"]["0"]["p_full"]) for t in TAGS if S["pliki"][t].get(f"klasyf_{nm}")))
for fn, nm in (("eval_shots_Kscan_K6.json", "baza"), ("eval_shots_gkpxK6.json", "gkpx")):
    p = Path(os.path.join(os.environ.get("QMI_ROOT", str(Path(__file__).resolve().parents[2])), "results_eq/konno")) / fn
    if p.exists():
        e = json.load(open(p)); S.setdefault("acc_test", {})[nm] = {f"N{N}_loss{lv}": float(np.mean([e[m]["loss"][str(N)][str(lv)]["acc"] for m in e])) for N in (100, 10000) for lv in (0, 80)}
p = OUT / "eval_shots_ahmK6.json"
if p.exists():
    e = json.load(open(p)); S.setdefault("acc_test", {})["ahm"] = {f"N{N}_loss{lv}": float(np.mean([e[m]["loss"][str(N)][str(lv)]["acc"] for m in e])) for N in (100, 10000) for lv in (0, 80)}
json.dump(S, open(OUT / "summary.json.tmp", "w"), indent=1); os.replace(OUT / "summary.json.tmp", OUT / "summary.json")

# tabele markdown
P = S["pliki"]; md = []
md.append("| zestaw | c | x0 | p0 | chi2/3 | Var min..max | <n> momenty | stac: istotne/7 | max sd fazy [st] | iid Fisher p | blok/los N=1e3 |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    d = P[t]; ph = [x for x in d["stac_phase_sd_deg"] if x is not None]
    md.append(f"| {t} | {d['c']:+.3f}({d['c_se']:.3f}) | {d['x0']:+.3f} | {d['p0']:+.3f} | {d['chi2_mean_fit']:.1f} | {min(d['var']):.3f}..{max(d['var']):.3f} | "
              f"{d['nbar_mom']:.3f}({d['nbar_mom_se']:.3f}) | {d['stac_nsig_bonf']} | {max(ph) if ph else float('nan'):.1f} | {d['iid_fisher_p']:.3f} | {d['iid_ratio']['1000'][0]:.3f} |")
md.append("")
md.append("| zestaw | czystosc | <n> | W(0) | min W | W(0) praca | TVD ML max (polowa/polowa max) | PSS r [dB] | PSS eta | F(model,ML) | F czysty PSS | F kot nieparz. (|a|) | s* (PSS) |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    d = P[t]; m = d["ml"]; b = m["boot_sd"]; ps = d["pss"]; pw = d.get("paper_W0")
    md.append(f"| {t} | {m['purity']:.3f}({b['purity']:.3f}) | {m['nbar']:.3f}({b['nbar']:.3f}) | {m['W0']:+.3f}({b['W0']:.3f}) | {m['Wmin']:+.3f}({b['Wmin']:.3f}) | "
              f"{'%+.3f(%.3f)' % pw if pw else '-'} | {m['tvd_max']:.3f} ({m['tvd_half_max']:.3f}) | {ps['r']:.3f} [{ps['sqdb']:.1f}] | {ps['eta']:.3f} | {ps['F_model_ml']:.3f} | "
              f"{ps['F_purePSS']:.3f} | {ps['F_oddcat']:.3f} ({ps['cat_alpha']:.2f}) | {d['skala']['s_best_pss']:.2f} |")
md.append("")
md.append("| zestaw | udzial W(0)<0 N=100 | N=1000 | N=10000 | mediana W(0) N=1000 | kurtoza sr. | ML centrowanie per faza: W(0) / min W |")
md.append("|---|---|---|---|---|---|---|")
for t in TAGS:
    d = P[t]; g = d.get("negN", {}); c = d.get("ml_centrowanie_per_faza", {})
    fr = " | ".join(f"{g[k]['frac_W0_neg']:.2f}" if k in g else "-" for k in ("N100", "N1000", "N10000"))
    med = f"{g['N1000']['W0_med']:+.3f}" if "N1000" in g else "-"; ce = f"{c['W0']:+.3f} / {c['Wmin']:+.3f}" if c else "-"
    md.append(f"| {t} | {fr} | {med} | {np.mean(d['kurt']):+.2f} | {ce} |")
md.append("")
for nm in ("baza", "gkpx", "ahm"):
    if f"klasyf_{nm}" not in P[TAGS[0]]: continue
    md.append(f"Sieci {nm}: dominujaca klasa (udzial, 3 sieci razem) losowe podzbiory exp / rekordy sym; p_rek = test na poziomie rekordu (B=40, srednia TVD 3 sieci); p_blok = min z 3 sieci, dokladny test permutacyjny 2xK kolejnych blokow vs sym; caly rekord: klasa (3 sieci) i udzial rekordow sym w tej klasie")
    md.append("| zestaw | N=100 | N=1000 | N=10000 | caly rekord (30000) |"); md.append("|---|---|---|---|---|")
    for t in TAGS:
        r = P[t][f"klasyf_{nm}"]; row = []
        for N in (100, 1000, 10000):
            e = r[str(N)]; fl, fs = e["frac"]["exp_los"], e["frac"]["sym"]; cl = max(fl, key=fl.get); cs = max(fs, key=fs.get)
            row.append(f"{cl} {fl[cl]:.2f} / {cs} {fs[cs]:.2f}; p_rek {e.get('rek_p_suma', float('nan')):.2f}; p_blok {min(e['p_blok_exact']):.2f}")
        e = r["0"]; row.append(f"{','.join(e['full_class'])}; sym w klasie {min(e['p_full']):.2f}..{max(e['p_full']):.2f}")
        md.append(f"| {t} | " + " | ".join(row) + " |")
    md.append("")
open(OUT / "tabele.md", "w").write("\n".join(md) + "\n"); print("\n".join(md)); print(json.dumps(S["testy"], indent=1))
