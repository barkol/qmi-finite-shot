"""Krok 5: zbiorcze liczby (fizyka, rekordy, klasyfikacja, testy istotnosci) -> wyniki.json. Tylko odczyt wynikow krokow 2-4.
  python -I krok5_agreguj.py"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats, optimize

OUT = Path(__file__).resolve().parent
ZAKRES = {"baza": (2.0, 5.0), "ahm": (1.0, 16.0)}     # n termiczne w treningu: eqn NBAR = {2,3,4,5}; Ahmed U(1, 16]
NS = (100, 1000, 10000)


def tmsv(nbar, vsq):
    """Model czystej straty dla TMSV: n = eta sinh^2 r, Var(x-) / Var_vac = eta e^{-2r} + 1 - eta. Zwraca (eta, r) albo None."""
    a = 1 - vsq
    if a <= 0: return None
    y = 1 / (1 + 4 * nbar / a)                        # z (1-y)/(4y) * a = n
    return a / (1 - y), -0.5 * np.log(y)


def main():
    W = {"zestawy": {}, "klasyf": {}}
    for f in sorted((OUT / "rekordy").glob("*.json")):
        r = json.load(open(f)); m = r["meta"]; z = f.stem
        sq = {q: min(m[f"{q}_minus_db"], m[f"{q}_plus_db"]) for q in "xp"}
        sq_el = {"x": m["x_minus_db_el"]}
        nb12 = 0.5 * (r["1"]["nbar"] + r["2"]["nbar"]); vsq = 10 ** (np.mean(list(sq.values())) / 10)
        e = tmsv(nb12, vsq)
        W["zestawy"][z] = dict(
            seria=m["seria"], zestaw=m["zestaw"], pump1_mW=int(m["zestaw"][:3]), pump2_mW=int(m["zestaw"][6:9]), loss_dB=int(m["zestaw"][12:14]),
            opoznienie_probek=m["L"], pakiety=m["n_pak"], shot_z=m["shot_z"],
            clearance_db=[m["clearance_db_hd1"], m["clearance_db_hd2"]], eta_el=[m["eta_el_hd1"], m["eta_el_hd2"]],
            shot_before_after_db=[m["shot_before_after_db_hd1"], m["shot_before_after_db_hd2"]],
            var_pakiety={k: m[f"var_{k}"] for k in ("x1", "p1", "x2", "p2")},
            epr_db=dict(x_minus=m["x_minus_db"], x_plus=m["x_plus_db"], p_minus=m["p_minus_db"], p_plus=m["p_plus_db"],
                        x_minus_el=m["x_minus_db_el"], p_minus_el=m["p_minus_db_el"], x_opt=m["x_opt_db"], p_opt=m["p_opt_db"], g_opt=[m["x_g_opt"], m["p_g_opt"]]),
            C_pakiety_x1=m["C_x1"], C_shot_hd1=m["C_shot_hd1"], cross_adj_x=m["x_cross_adj"],
            rekord_hd1=r["1"], rekord_hd2=r["2"], iid=r["iid"], stac=dict(names=r["stac"]["names"], pval=r["stac"]["pval"], p_bonf=r["stac"]["p_bonf"]),
            tmsv_eta_r=(None if e is None else [float(e[0]), float(e[1])]),
            w_zakresie={k: bool(lo <= r["1"]["nbar"] <= hi) for k, (lo, hi) in ZAKRES.items()})
    # seria pompy 1: r = a sqrt(P1), eta wspolne (model czystej straty), dopasowanie do n i Var(x-) czterech zestawow
    s1 = {k: v for k, v in W["zestawy"].items() if v["seria"].endswith("pump1")}
    P = np.array([v["pump1_mW"] for v in s1.values()], float)
    nb = np.array([0.5 * (v["rekord_hd1"]["nbar"] + v["rekord_hd2"]["nbar"]) for v in s1.values()])
    vs = np.array([10 ** (0.5 * (min(v["epr_db"]["x_minus"], v["epr_db"]["x_plus"]) + min(v["epr_db"]["p_minus"], v["epr_db"]["p_plus"])) / 10) for v in s1.values()])
    def res(t):
        eta, a = t; r = a * np.sqrt(P); return np.concatenate([(eta * np.sinh(r) ** 2 - nb) / nb, (eta * np.exp(-2 * r) + 1 - eta - vs) / vs])
    fit = optimize.least_squares(res, [0.7, 0.1], bounds=([0.01, 0.0], [1.0, 1.0]))
    eta, a = fit.x; r = a * np.sqrt(P)
    W["pompa1_fit"] = dict(P_mW=P.tolist(), nbar_zm=nb.tolist(), nbar_model=(eta * np.sinh(r) ** 2).tolist(),
                           vsq_db_zm=(10 * np.log10(vs)).tolist(), vsq_db_model=(10 * np.log10(eta * np.exp(-2 * r) + 1 - eta)).tolist(),
                           eta=float(eta), a_per_sqrt_mW=float(a), r_200mW=float(a * np.sqrt(200)), rms_wzgl=float(np.sqrt(np.mean(fit.fun ** 2))))
    # klasyfikacja
    for d in sorted((OUT / "klasyf").glob("*_*")):
        grupa, war = d.name.rsplit("_", 1); fam = grupa.split("_")[0]
        for f in sorted(d.glob("*.json")):
            J = json.load(open(f)); C = {k: np.array(v) for k, v in J["counts"].items()}; nets = sorted({k.split("|")[0] for k in C})
            lab = J["labels"]; ti = lab.index("thermal"); z = f.stem; o = {}
            for N in NS:
                e = {}
                for src in ("exp_los", "exp_blok", "sym_gauss", "sym_term"):
                    t = sum(C[f"{nn}|{src}|N{N}"] for nn in nets); e[src] = dict(zip(lab, (t / t.sum()).round(4).tolist())); e[src]["_n"] = int(t.sum())
                pv = {}
                for a_, b_ in (("exp_los", "sym_gauss"), ("exp_los", "sym_term"), ("exp_blok", "sym_gauss"), ("sym_gauss", "sym_term")):
                    ps = []
                    for nn in nets:
                        A, B = C[f"{nn}|{a_}|N{N}"], C[f"{nn}|{b_}|N{N}"]; msk = (A + B) > 0
                        ps.append(float(stats.chi2_contingency(np.stack([A[msk], B[msk]]))[1]) if msk.sum() > 1 else 1.0)
                    pv[f"{a_}_vs_{b_}"] = ps
                o[f"N{N}"] = dict(frakcje=e, p_chi2=pv, thermal_exp_los=e["exp_los"]["thermal"], thermal_sym_term=e["sym_term"]["thermal"])
            W["klasyf"].setdefault(f"{grupa}_{war}", {})[z] = dict(nets=nets, nbar=J["nbar"], w_zakresie=W["zestawy"][z]["w_zakresie"][fam], **o)
    # podsumowanie testow istotnosci w zakresie treningowym
    S = {}
    for gw, zz in W["klasyf"].items():
        for para in ("exp_los_vs_sym_gauss", "exp_los_vs_sym_term", "exp_blok_vs_sym_gauss", "sym_gauss_vs_sym_term"):
            ps = [p for z, v in zz.items() if v["w_zakresie"] for N in NS for p in v[f"N{N}"]["p_chi2"][para]]
            if ps: S[f"{gw}|{para}"] = dict(ntest=len(ps), pmin=min(ps), n_p_lt_005=int(sum(p < 0.05 for p in ps)), n_bonf=int(sum(p < 0.05 / len(ps) for p in ps)),
                                            oczek_p_lt_005=0.05 * len(ps))
    W["istotnosc_w_zakresie"] = S
    json.dump(W, open(OUT / "wyniki.json", "w"), indent=1)
    print(json.dumps(W["pompa1_fit"], indent=1)); [print(k, v) for k, v in S.items()]
    for z, v in W["zestawy"].items(): print(z, "tmsv eta,r", np.round(v["tmsv_eta_r"], 3) if v["tmsv_eta_r"] else None)


if __name__ == "__main__":
    main()
