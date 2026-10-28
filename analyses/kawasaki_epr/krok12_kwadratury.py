"""Krok 1-2: surowe przebiegi (Kawasaki+ 2024, Dryad) -> kwadratury modow czasowych, kalibracja na szumie srutowym, kontrole EPR.
Mod czasowy: Asavanant+ 2019 supl. (S1) przeskalowany 1e3 razy (Kawasaki+ 2024: 'only the time width is shortened by 10^3'):
Dt = 40 ps (10 probek po 3.906 ps), gamma = 2 pi x 10.5 GHz, pakiety rozlaczne. Kanal hd2 opozniony o L probek wzgledem hd1
(L z korelacji wzajemnej, wyznaczane dla kazdego zestawu). Normalizacja: x = q / sqrt(2 Var_shot), wiec proznia ma Var 1/2
(konwencja projektu x = (a + a^dag)/sqrt2); Var_shot = srednia z shot_before i shot_after (dla EPR1 bez wlasnych plikow shot: shot
z 200mW_800mW_00dB tej serii, nagrany przed pierwszym i po ostatnim zestawie). Szum elektroniki NIE jest odejmowany od zdarzen
(zawarty w Var_shot dziala jak dodatkowa strata eta_el = 1 - V_el/V_shot); wartosci z odjeciem podajemy osobno.
Wznowienie: zestaw z gotowym dane/<seria>__<zestaw>.npz jest pomijany. Stop na pierwszym bledzie.
  python -I krok12_kwadratury.py [--tylko SERIA/ZESTAW]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wspolne import RAW, load, mode_fn, quads

OUT = Path(__file__).resolve().parent; D = OUT / "dane"; LOG = OUT / "krok12.log"
NS, GAM = 10, 2 * np.pi * 10.5e9
F = mode_fn(NS, GAM)


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); open(LOG, "a").write(s + "\n")


def db(v): return float(10 * np.log10(v))


def delay(a, b, nseg=500, n=8192):
    a = a[:nseg].astype(np.float64); b = b[:nseg].astype(np.float64); a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
    c = np.fft.irfft((np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n))).mean(0), n) / np.sqrt((a * a).sum(1).mean() * (b * b).sum(1).mean())
    i = int(np.argmax(np.abs(c))); lag = i - n if i > n // 2 else i
    return -lag, float(c[i])          # b[t + L] ~ a[t]


def adj(q, ms=(1, 2, 3, 5)):
    q = q - q.mean(); return [float(np.mean(q[:, :-m] * q[:, m:]) / np.mean(q * q)) for m in ms]


def zestawy():
    out = []
    for s in sorted(Path(RAW).glob("20230807_EPR*")):
        for z in sorted(s.iterdir()): out.append((s.name, z.name))
    return out


def shot_dir(seria, zestaw):
    z = Path(RAW) / seria / zestaw
    if (z / "shot_before_hd1.h5").exists(): return z
    z2 = Path(RAW) / seria / "200mW_800mW_00dB"
    if seria.endswith("pump1") and (z2 / "shot_before_hd1.h5").exists(): return z2
    raise RuntimeError(f"brak szumu srutowego dla {seria}/{zestaw}")


def przetworz(seria, zestaw, el):
    z = Path(RAW) / seria / zestaw; sd = shot_dir(seria, zestaw); t0 = time.time()
    V = {f"{q}{h}": load(z / f"{q}_hd{h}.h5")[0] for q in "xp" for h in "12"}
    L, cx = delay(V["x1"], V["x2"]); Lp, cp = delay(V["p1"], V["p2"])
    if L != Lp or abs(cx) < 0.3 or abs(cp) < 0.3: raise RuntimeError(f"opoznienie x {L} ({cx:.2f}) vs p {Lp} ({cp:.2f})")
    cut = lambda A, h: A[:, :A.shape[1] - L] if h == "1" else A[:, L:]
    Q = {k: quads(cut(v, k[1]), F) for k, v in V.items()}; del V
    S = {}
    for h in "12":
        for w in ("before", "after"):
            S[f"{w}{h}"] = quads(cut(load(sd / f"shot_{w}_hd{h}.h5")[0], h), F)
    vs = {h: 0.5 * (S[f"before{h}"].var() + S[f"after{h}"].var()) for h in "12"}
    ve = {h: float(quads(el[h], F).var()) for h in "12"}
    r = dict(seria=seria, zestaw=zestaw, shot_z=sd.name, L=L, corr_peak_x=cx, corr_peak_p=cp, n_pak=int(Q["x1"].size), pak_na_ramke=int(Q["x1"].shape[1]))
    for h in "12":
        r[f"shot_before_after_db_hd{h}"] = db(S[f"before{h}"].var() / S[f"after{h}"].var())
        r[f"clearance_db_hd{h}"] = db(vs[h] / ve[h]); r[f"eta_el_hd{h}"] = 1 - ve[h] / vs[h]
        r[f"C_shot_hd{h}"] = adj(S[f"before{h}"])
    X = {k: (v / np.sqrt(2 * vs[k[1]])) for k, v in Q.items()}   # proznia: Var = 1/2
    for k, v in Q.items():
        r[f"mean_{k}"] = float(X[k].mean()); r[f"var_{k}"] = float(X[k].var())
        r[f"var_el_corr_{k}"] = float(0.5 * (v.var() - ve[k[1]]) / (vs[k[1]] - ve[k[1]]))
        r[f"C_{k}"] = adj(X[k]); r[f"kurt_ex_{k}"] = float(np.mean((X[k] - X[k].mean()) ** 4) / X[k].var() ** 2 - 3)
    for q in "xp":
        a, b = X[q + "1"], X[q + "2"]
        r[f"{q}_minus_db"] = db(np.mean((a - b) ** 2 / 2) / 0.5); r[f"{q}_plus_db"] = db(np.mean((a + b) ** 2 / 2) / 0.5)
        g = float(np.mean(a * b) / np.mean(b * b)); r[f"{q}_g_opt"] = g
        r[f"{q}_opt_db"] = db(np.mean((a - g * b) ** 2) / (0.5 * (1 + g * g)))   # Var(x1 - g x2) wobec prozni 1/2 (1 + g^2)
        r[f"{q}_corr"] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        # z odjeciem szumu elektroniki (oba kanaly niezalezne): Var_corr(a-b) = Var(a-b) - Vel1n - Vel2n
        e1 = ve["1"] / (2 * vs["1"]); e2 = ve["2"] / (2 * vs["2"]); sc1 = 0.5 / (0.5 - e1); sc2 = 0.5 / (0.5 - e2)
        vm = (np.var(a) - e1) * sc1 + (np.var(b) - e2) * sc2 - 2 * np.mean(a * b) * np.sqrt(sc1 * sc2)
        r[f"{q}_minus_db_el"] = db(vm / 2 / 0.5)
        # pakiety sasiednie miedzy detektorami (niezaleznosc pakietow)
        r[f"{q}_cross_adj"] = [float(np.mean(a[:, :-1] * b[:, 1:]) / np.sqrt(a.var() * b.var())), float(np.mean(a[:, 1:] * b[:, :-1]) / np.sqrt(a.var() * b.var()))]
    r["sek"] = time.time() - t0
    return r, {k: v.astype(np.float32) for k, v in X.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tylko", default=None); a = ap.parse_args()
    D.mkdir(exist_ok=True); log(f"start {time.ctime()} mod: Dt={NS} probek, gamma=2pi*{GAM/2/np.pi/1e9:.1f} GHz, f={np.round(F, 4).tolist()}")
    el = {h: load(Path(RAW) / "20230808_bEPR_circuitnoise" / f"hd{h}.h5")[0] for h in "12"}
    if el["1"].shape[1] < 5121: raise RuntimeError("szum elektroniki: krotszy przebieg")
    lista = zestawy()
    if a.tylko: lista = [x for x in lista if f"{x[0]}/{x[1]}" == a.tylko]
    if not lista: raise RuntimeError("pusta lista zestawow")
    for seria, zestaw in lista:
        p = D / f"{seria}__{zestaw}.npz"
        if p.exists(): log(f"pomijam (gotowe) {p.name}"); continue
        r, X = przetworz(seria, zestaw, el)
        np.savez(str(p) + ".tmp.npz", meta=json.dumps(r), **X); os.replace(str(p) + ".tmp.npz", p)
        json.dump(r, open(D / f"{seria}__{zestaw}.json", "w"), indent=1)
        log(f"{seria}/{zestaw}: L={L if (L:=r['L']) else 0} Var x1 {r['var_x1']:.3f} p1 {r['var_p1']:.3f} x2 {r['var_x2']:.3f} p2 {r['var_p2']:.3f} | "
            f"x- {r['x_minus_db']:.2f} dB (el-korekta {r['x_minus_db_el']:.2f}), p-/p+ {r['p_minus_db']:.2f}/{r['p_plus_db']:.2f} dB | "
            f"clear {r['clearance_db_hd1']:.1f}/{r['clearance_db_hd2']:.1f} dB, shot b/a {r['shot_before_after_db_hd1']:.2f}/{r['shot_before_after_db_hd2']:.2f} dB | "
            f"C1 x1 {r['C_x1'][0]:.3f} | {r['sek']:.0f}s")
    log(f"KROK12_DONE {time.ctime()}")
