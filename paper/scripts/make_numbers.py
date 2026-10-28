"""Wylicza liczby do manuskryptu z results/*.json i zapisuje results/numbers.tex (makra \\res@klucz).
Brakujace pliki nie psuja budowy: odpowiedni klucz po prostu nie jest zdefiniowany (w PDF czerwone [klucz]).
   python scripts/make_numbers.py            (z korzenia projektu manuskryptu)
Zmienne: RES (katalog JSON-ow), PFX (prefiks kluczy), OUTDIR (katalog wyjscia, domyslnie results).
Przedzialy ufnosci 95% dla dokladnosci usrednionej po ziarnach (rodziny acc, nar, mm, plat, a100/a1000,
rule, ks): polowa szerokosci = sqrt(h_W^2 + h_s^2), gdzie h_W to polowa przedzialu Wilsona dla jednego zbioru testowego
(n z JSON-a; ziarna dziela zbior testowy, wiec usrednianie go nie zmniejsza), a h_s = t_{k-1} s / sqrt(k) to rozrzut
k ziaren (s = odchylenie standardowe). Gora i dol osobno (Wilson jest niesymetryczny), obciete do [0, 1].
Dla roznic Q-W (rule_..._dQW) wariancje sie sumuja (zalozenie niezaleznosci, ostrozne). Klucze dodatkowe:
<klucz>_lo, <klucz>_hi (w %), <klucz>_ci (\\ensuremath{[lo,hi]}\\%), oraz OUTDIR/ci{_PFX}.json z pelnym zestawem."""
import glob, json, os, re
import numpy as np
from scipy.stats import t as _tdist

R = os.environ.get("RES", "results")
PFX = os.environ.get("PFX", "")   # np. RES=results_ahm PFX=ahm_ -> results/numbers_ahm.tex
OUTDIR = os.environ.get("OUTDIR", "results")
OUT = os.path.join(OUTDIR, "numbers%s.tex" % ("_" + PFX.rstrip("_") if PFX else ""))
keys = {}
CI = {}   # klucz -> dict(mean, lo, hi, sd, k, n)


class Acc(float):
    """srednia dokladnosc po ziarnach, z wartosciami poszczegolnych ziaren (v) i licznosciami zbioru testowego (n)"""
    def __new__(cls, v, n):
        o = float.__new__(cls, float(np.mean(v))); o.v = [float(x) for x in v]; o.n = [int(x) for x in n]; return o


def _wilson(p, n, z=1.959964):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _hs(a):
    k = len(a.v)
    return float(_tdist.ppf(0.975, k - 1) * np.std(a.v, ddof=1) / np.sqrt(k)) if k > 1 else 0.0


def ci_of(a):
    p = float(a); n = min(a.n); wl, wh = _wilson(p, n); hs = _hs(a)
    lo = max(0.0, p - np.hypot(p - wl, hs)); hi = min(1.0, p + np.hypot(wh - p, hs))
    return dict(mean=p, lo=lo, hi=hi, sd=float(np.std(a.v, ddof=1)) if len(a.v) > 1 else 0.0, k=len(a.v), n=n)


def put(k, a):
    """klucz z dokladnoscia; dla Acc takze przedzial ufnosci"""
    keys[k] = pct(a)
    if isinstance(a, Acc): CI[k] = ci_of(a)


def paired_diff(f, grp=None, M=None, y=None, names=None):
    """roznica Q-W ze sparowanej ewaluacji (qmi/x8/pary.py): ten sam zbior testowy dla wszystkich sieci.
    Skladnik zbioru testowego: roznica srednich po ziarnach dla kazdego stanu, warstwowo po rodzinach (z);
    skladnik ziaren: wariancje trafnosci ziaren Q i W, kwantyl t Welcha. Polowki sumowane w kwadraturze."""
    if M is None: d = np.load(f); M = d["correct"].astype(float); y = d["y"]; names = d["names"]
    q = np.array([(grp(s) if grp else s.startswith("clf_Q")) for s in names])   # grp = predykat grupy "Q" (np. sieci S wobec W)
    aQ, aW = M[q].mean(1), M[~q].mean(1); di = M[q].mean(0) - M[~q].mean(0); n = len(y)
    se_st = np.sqrt(sum((y == c).sum() * di[y == c].var(ddof=1) for c in np.unique(y))) / n
    vq, vw = aQ.var(ddof=1) / len(aQ), aW.var(ddof=1) / len(aW)
    df = (vq + vw) ** 2 / (vq ** 2 / (len(aQ) - 1) + vw ** 2 / (len(aW) - 1)) if vq + vw > 0 else 1e9
    D = float(aQ.mean() - aW.mean()); h = float(np.hypot(1.959964 * se_st, _tdist.ppf(0.975, df) * np.sqrt(vq + vw)))
    return dict(mean=D, lo=D - h, hi=D + h, diff=True, paired=True, kQ=int(q.sum()), kW=int((~q).sum()), n=n, se_states=float(se_st), se_seeds=float(np.sqrt(vq + vw)), df=float(df))


# wyniki eq (rozpoznawane po katalogu pary_sw/) wymagaja kompletu plikow sparowanych i kontroli; brak = stop,
# nie ciche przejscie na przedzial niesparowany
EQ_STRICT = os.path.isdir(os.path.join(os.environ.get("RES", "results"), "pary_sw")) and os.environ.get("ALLOW_UNPAIRED") != "1"
if EQ_STRICT:
    _req = ["pary", "pary_sino", "pary_sw", "energy_control.json"]
    _brak = [f for f in _req if not os.path.exists(os.path.join(os.environ.get("RES", "results"), f))]
    if _brak: raise SystemExit(f"BLAD: wyniki eq bez {_brak} w {os.environ.get('RES', 'results')}")


def put_diff(k, aq, aw, pf=None):
    """roznica Q-W w punktach procentowych z przedzialem; sparowana (pf = plik z qmi/x8/pary.py), jesli jest,
    w przeciwnym razie zachowawczo: wariancje Wilsona i ziaren Q i W sumowane jak dla niezaleznych zbiorow"""
    if pf and not os.path.exists(pf) and EQ_STRICT:
        raise SystemExit(f"BLAD: {k}: brak pliku sparowanego {pf} (wyniki eq wymagaja sparowania; przyblizenie niesparowane tylko z ALLOW_UNPAIRED=1)")
    if pf and os.path.exists(pf):
        c = paired_diff(pf)
        if abs(c["mean"] - (float(aq) - float(aw))) > 1e-9: raise SystemExit(f"BLAD: {k}: plik sparowany niezgodny z eval_shots")
        keys[k] = f"{100 * c['mean']:.1f}"; CI[k] = c; return
    cq, cw = ci_of(aq), ci_of(aw); d = cq["mean"] - cw["mean"]
    up = np.hypot(cq["hi"] - cq["mean"], cw["mean"] - cw["lo"]); dn = np.hypot(cq["mean"] - cq["lo"], cw["hi"] - cw["mean"])
    keys[k] = f"{100 * d:.1f}"; CI[k] = dict(mean=d, lo=d - dn, hi=d + up, diff=True, k=min(cq["k"], cw["k"]), n=min(cq["n"], cw["n"]))


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def pct(x):
    return f"{100 * x:.1f}\\%"


def seedmean(runs, ch, lvl):
    """runs: lista slownikow {chn:{level:{acc}}}; srednia acc po ziarnach dla (kanal, poziom)."""
    c = [r[ch][str(lvl)] for r in runs if ch in r and str(lvl) in r[ch]]
    return Acc([x["acc"] for x in c], [x["n"] for x in c]) if c else None


def thr_last_ge(runs, ch, level_list, cut=0.9):
    """ostatni poziom, do ktorego (wlacznie, od zera w gore, bez przerwy) srednia acc >= cut."""
    last = None
    for L in level_list:
        a = seedmean(runs, ch, L)
        if a is None or a < cut: break
        last = L
    return last


levels = list(range(0, 101, 5))

# --- sweepy na dokladnych obrazach: runs wytrenowane na dokladnych obrazach (kind=exact)
exact = {"W": [], "Q": []}
for f in sorted(glob.glob(os.path.join(R, "eval_sweep_clf_*_exact_s*.json"))):
    m = re.search(r"clf_([WQ])_exact", f); d = load(f)
    if m and d: exact[m.group(1)].append(d)
for rep in "WQ":
    if not exact[rep]: continue
    a = seedmean(exact[rep], "loss", 0)
    if a is not None: put(f"acc_clean_{rep}", a)
    t = thr_last_ge(exact[rep], "loss", levels)
    if t is not None: keys[f"loss_thr_{rep}"] = str(t)
    for ch, name in (("dephase", "deph"), ("gauss", "gauss")):
        a = seedmean(exact[rep], ch, 20)
        if a is not None: put(f"{name}_acc20_{rep}", a)
for rep in "WQ":
    if not exact[rep]: continue
    for L in (50, 60, 80, 90):
        a = seedmean(exact[rep], "loss", L)
        if a is not None: put(f"acc_loss{L}_{rep}", a)
    for ch, nm in (("dephase", "deph100"), ("gauss", "gauss100")):
        a = seedmean(exact[rep], ch, 100)
        if a is not None: put(f"acc_{nm}_{rep}", a)

# --- granica kwantowa
lim = load(os.path.join(R, "limit_loss_dim48.json"))
if lim:
    if "1.00" in lim: keys["Pstar_loss0"] = f"{lim['1.00']['P_opt']:.3f}"
    for L in (50, 80):
        k = f"{1 - L / 100:.2f}"
        if k in lim: keys[f"Pstar_loss{L}"] = f"{lim[k]['P_opt']:.3f}"

# --- szum zliczen: eval_shots.json  {model: {ch: {N: {lvl: {acc}}}}}
sh = {}
for _f in [os.path.join(R, "eval_shots.json")] + sorted(glob.glob(os.path.join(R, "eval_shots_s[0-9].json"))):
    _d = load(_f)
    if _d: sh.update(_d)
sh = sh or None
if sh:
    NS = (100, 1000, 10000, 100000, 1000000)

    def acc_at(kind, rep, ch, N, lvl):
        c = [m[ch][str(N)][str(lvl)] for name, m in sh.items()
             if f"clf_{rep}_{kind}_s" in name and ch in m and str(N) in m[ch] and str(lvl) in m[ch][str(N)]]
        return Acc([x["acc"] for x in c], [x["n"] for x in c]) if c else None

    def curve(kind, rep, ch, lvl):
        return np.array([acc_at(kind, rep, ch, N, lvl) for N in NS], float)

    def cross(y, level):
        """pierwsze N (interpolacja liniowa w log10 N), przy ktorym krzywa y(N) osiaga `level`; None gdy nigdy."""
        x = np.log10(NS)
        if y[0] >= level: return "lt"
        for i in range(1, len(NS)):
            if y[i] >= level:
                return x[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
        return None

    def fmt(c):
        if c is None: return "\\ensuremath{>10^{6}}"
        if c == "lt": return "\\ensuremath{<10^{2}}"
        e = int(np.floor(c)); m = 10 ** (c - e)
        return f"\\ensuremath{{{m:.1f}\\times10^{{{e}}}}}"

    for rep in "WQ":
        for ch, L, tag in (("loss", 0, "loss0"), ("loss", 50, "loss50"), ("loss", 80, "loss80"), ("dephase", 20, "deph20")):
            y = curve("shots", rep, ch, L); plat = acc_at("shots", rep, ch, 0, L)
            if not np.isnan(y).any() and plat is not None: keys[f"N95_{tag}_{rep}"] = fmt(cross(y, 0.95 * plat))
        # niedopasowanie: sieci trenowane na dokladnych obrazach vs na rekordach (te same rekordy testowe)
        for tag, ch, L, N in (("c1e2", "loss", 0, 100), ("l60", "loss", 60, 10000), ("l80", "loss", 80, 10000)):
            for kind in ("exact", "shots"):
                a = acc_at(kind, rep, ch, N, L)
                if a is not None: put(f"mm_{tag}_{kind}_{rep}", a)
        ge, gs = acc_at("exact", rep, "loss", 10000, 60), acc_at("shots", rep, "loss", 10000, 60)
        if ge is not None and gs is not None: keys[f"mm_gap_l60_{rep}"] = f"{100 * (gs - ge):.0f}"
        # plateau N->inf dla sieci trenowanych na rekordach
        for tag, ch, L in (("loss80", "loss", 80), ("deph20", "dephase", 20)):
            a = acc_at("shots", rep, ch, 0, L)
            if a is not None: put(f"plat_{tag}_{rep}", a)
        # dokladnosc przy N = 100 i 1000
        for tag, ch, L in (("loss0", "loss", 0), ("loss40", "loss", 40), ("loss60", "loss", 60), ("loss80", "loss", 80)):
            for N in (100, 1000):
                a = acc_at("shots", rep, ch, N, L)
                if a is not None: put(f"a{N}_{tag}_{rep}", a)
    # N*: najmniejsze N, od ktorego Wigner >= Q przy 80% strat (interpolacja)
    yW = curve("shots", "W", "loss", 80); yQ = curve("shots", "Q", "loss", 80)
    if not (np.isnan(yW).any() or np.isnan(yQ).any()):
        d = yW - yQ; x = np.log10(NS); cx = None
        for i in range(1, len(NS)):
            if d[i - 1] < 0 <= d[i]:
                cx = x[i - 1] + (0 - d[i - 1]) / (d[i] - d[i - 1]) * (x[i] - x[i - 1]); break
        if cx is not None: keys["Nx_loss80"] = fmt(cx)

# --- trening na waskim zakresie szumu (kontrola): eval_sweep_clf_{W,Q}_narrow_s*.json
narrow = {"W": [], "Q": []}
for f in sorted(glob.glob(os.path.join(R, "eval_sweep_clf_*_narrow_s*.json"))):
    m = re.search(r"clf_([WQ])_narrow", f); d = load(f)
    if m and d: narrow[m.group(1)].append(d)
for rep in "WQ":
    if not narrow[rep]: continue
    keys["nar_nseeds"] = str(len(narrow[rep]))
    for L in (40, 60, 80):
        a = seedmean(narrow[rep], "loss", L)
        if a is not None: put(f"nar_loss{L}_{rep}", a)
    for ch, nm in (("gauss", "gauss"), ("dephase", "deph")):
        for L in (20, 100):
            a = seedmean(narrow[rep], ch, L)
            if a is not None: put(f"nar_{nm}{L}_{rep}", a)

# --- homodyna bez rekonstrukcji (sinogram): eval_shots_sino.json
sino = load(os.path.join(R, "eval_shots_sino_s0-4.json")) or load(os.path.join(R, "eval_shots_sino.json"))   # 5 ziaren, jesli sa
if sino:
    NSS = (100, 1000, 10000, 100000, 1000000)
    def sacc(ch, N, L):
        v = [m[ch][str(N)][str(L)]["acc"] for n, m in sino.items() if "_sino_s" in n and ch in m]
        return float(np.mean(v)) if v else None
    def s95(ch, L):
        plat = sacc(ch, 0, L); y = [sacc(ch, N, L) for N in NSS]
        if plat is None or any(v is None for v in y): return None
        x = np.log10(NSS); level = 0.95 * plat
        if y[0] >= level: return "\\ensuremath{<10^{2}}"
        for i in range(1, len(NSS)):
            if y[i] >= level:
                c = x[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1]); e = int(np.floor(c)); mm = 10 ** (c - e)
                return f"\\ensuremath{{{mm:.1f}\\times10^{{{e}}}}}"
        return "\\ensuremath{>10^{6}}"
    for tag, ch, L in (("loss0", "loss", 0), ("loss40", "loss", 40), ("loss60", "loss", 60), ("loss80", "loss", 80), ("deph20", "dephase", 20)):
        for N in (100, 1000):
            a = sacc(ch, N, L)
            if a is not None: keys[f"sino_a{N}_{tag}"] = pct(a)
        a = sacc(ch, 0, L)
        if a is not None: keys[f"sino_plat_{tag}"] = pct(a)
        v = s95(ch, L)
        if v: keys[f"sino_N95_{tag}"] = v
    keys["sino_nseeds"] = str(len([n for n in sino if "_sino_s" in n]))

# --- hold-out parametrow
# zestaw Ahmeda: hold-out ze stratyfikowanym doborem chunkow (hoexactS/hoshotsS/baseline_hoS); zestaw bazowy: wersja pierwotna
HS = "S" if PFX == "ahm_" else ""
def _mean_runs(pat, ch, L):
    v = []
    for f in sorted(glob.glob(os.path.join(R, pat))):
        d = load(f)
        if d and ch in d and str(L) in d[ch]: v.append(d[ch][str(L)]["acc"])
    return float(np.mean(v)) if v else None
for rep in "WQ":
    for split in ("in", "out"):
        for tag, ch, L in (("loss0", "loss", 0), ("loss60", "loss", 60), ("deph20", "dephase", 20), ("gauss100", "gauss", 100)):
            a = _mean_runs(f"eval_ho_{split}_clf_{rep}_hoexact{HS}_s*.json", ch, L)
            if a is not None: keys[f"ho_{rep}_{split}_{tag}"] = pct(a)
for split in ("in", "out"):
    b = load(os.path.join(R, f"baseline_ho{HS}_{split}.json"))
    if b:
        for name in ("logreg", "knn"):
            for rep in "WQ":
                d = b.get(f"{name}_{rep}", {}).get("loss", {}).get("0")
                if d: keys[f"base_{name}_ho_{split}_loss0_{rep}"] = pct(d["acc"])
    hs = load(os.path.join(R, f"eval_shots_ho{('S' if HS else '')}_{split}.json"))
    if hs:
        for rep in "WQ":
            names = [n for n in hs if f"clf_{rep}_hoshots{HS}_" in n]
            for tag, ch, L, N in (("loss0_N100", "loss", 0, 100), ("loss0_N1000", "loss", 0, 1000), ("deph20_N1000", "dephase", 20, 1000), ("loss60_N1000", "loss", 60, 1000)):
                v = [hs[n][ch][str(N)][str(L)]["acc"] for n in names]
                if v: keys[f"hos_{tag}_{rep}_{split}"] = pct(float(np.mean(v)))
            if split == "out" and names:   # czulosc na klase (srednia z ziaren), zbior poza zakresem, 0% strat
                for N in (100, 1000, 10000):
                    cm = sum(np.array(hs[n]["loss"][str(N)]["0"]["cm"]) for n in names)
                    for ci, cn in ((0, "fock"), (3, "num"), (4, "gkp"), (5, "binom"), (6, "cat")):
                        keys[f"hos_cls_{cn}_N{N}_{rep}"] = pct(float(cm[ci, ci] / cm[ci].sum()))

# --- kontrola K (liczba katow homodyny): eval_shots_K{K}.json, sieci W trenowane i testowane z K katami
for K in (12, 18):
    d = {}
    for _f in sorted(glob.glob(os.path.join(R, f"eval_shots_K{K}*.json"))):
        _d = load(_f)
        if _d: d.update(_d)
    if not d: continue
    names = [n for n in d if f"shotsK{K}_s" in n]
    def kacc(ch, N, L):
        v = [d[n][ch][str(N)][str(L)]["acc"] for n in names]; return float(np.mean(v)) if v else None
    for L in (0, 40, 60, 80):
        a = kacc("loss", 100, L)
        if a is not None: keys[f"K{K}_a100_loss{L}"] = pct(a)
    a = kacc("loss", 0, 80)
    if a is not None: keys[f"K{K}_plat_loss80"] = pct(a)
    keys[f"K{K}_nseeds"] = str(len(names))

# --- skan K z przedzialami: eval_shots_Kscan_K{K}{war}.json, sieci clf_W_shotsK{K}{war}_s*; K=36 = clf_W_shots_s*
_ks = {}
for _f in sorted(glob.glob(os.path.join(R, "eval_shots_Kscan_K*.json"))):
    mm_ = re.search(r"Kscan_K(\d+)([a-z]*)\.json$", _f); _d = load(_f)
    if mm_ and _d: _ks[mm_.group(1) + mm_.group(2)] = (_d, f"clf_W_shotsK{mm_.group(1) + mm_.group(2)}_s")
if _ks and sh:
    _ks["36"] = (sh, "clf_W_shots_s")
    for kk, (d, pre) in sorted(_ks.items()):
        names = [n for n in d if n.startswith(pre)]
        if not names: continue
        keys[f"ks_K{kk}_nseeds"] = str(len(names))
        for tag, ch, L in (("loss0", "loss", 0), ("loss60", "loss", 60), ("loss80", "loss", 80), ("deph20", "dephase", 20), ("deph60", "dephase", 60)):
            for N in (100, 1000, 10000):
                c = [d[n][ch][str(N)][str(L)] for n in names if ch in d[n] and str(N) in d[n][ch]]
                if c: put(f"ks_K{kk}_a{N}_{tag}", Acc([x["acc"] for x in c], [x["n"] for x in c]))

# --- bazy niesieciowe (dokladne obrazy)
bl = load(os.path.join(R, "baseline.json"))
if bl:
    for name in ("logreg", "knn"):
        for rep in "WQ":
            b = bl.get(f"{name}_{rep}", {})
            for tag, ch, L in (("", "loss", 0), ("_loss60", "loss", 60), ("_loss80", "loss", 80), ("_deph20", "dephase", 20), ("_deph80", "dephase", 80), ("_gauss20", "gauss", 20)):
                d = b.get(ch, {}).get(str(L))
                if d: keys[f"base_{name}{tag}_{rep}"] = pct(d["acc"])

# --- dryf fazy LO: eval_drift.json {model: {ch: {lvl: {cfg: {N: {acc}}}}}}; sieci trenowane na rekordach bez dryfu
dr = load(os.path.join(R, "eval_drift.json"))
if dr:
    keys["dr_nseeds"] = str(sum(1 for m in dr if m.startswith("clf_W_")))
    for rep in "WQ":
        for ch, L in (("loss", 0), ("loss", 20), ("loss", 50), ("dephase", 20)):
            for cfg in ("walk0", "walk3", "walk10", "walk30", "jitter10", "jitter30"):
                v = [dr[m][ch][str(L)][cfg]["10000"]["acc"] for m in dr if m.startswith(f"clf_{rep}_") and cfg in dr[m][ch][str(L)]]
                if v: keys[f"dr_{rep}_{cfg}_{ch}{L}"] = pct(float(np.mean(v)))

# --- sieci trenowane z dryfem (dwa ziarna na reprezentacje): results/eval_drift_trained.json
drt = load(os.path.join(R, "eval_drift_trained.json"))   # baza: results/, Ahmed: results_ahm/
if drt:
    keys["drt_nseeds"] = str(sum(1 for m in drt if m.startswith("clf_W_")))
    for rep in "WQ":
        for ch, L in (("loss", 0), ("loss", 20), ("loss", 50), ("dephase", 20)):
            for cfg in ("walk0", "walk3", "walk10", "walk30", "jitter10", "jitter30"):
                v = [drt[m][ch][str(L)][cfg]["10000"]["acc"] for m in drt if m.startswith(f"clf_{rep}_") and cfg in drt[m][ch][str(L)]]
                if v: keys[f"drt_{rep}_{cfg}_{ch}{L}"] = pct(float(np.mean(v)))

# --- zestaw Ahmeda: dryf, dokladnosc na rodzine (N=1e4), z macierzy pomylek
if PFX == "ahm_" and dr:
    AH = ("fock", "coherent", "thermal", "num", "gkp", "binomial", "cat")
    for rep in "WQ":
        for ch, L in (("loss", 0), ("loss", 50)):
            for cfg in ("walk0", "walk30", "jitter30"):
                cm = np.zeros((7, 7))
                for m in dr:
                    if m.startswith(f"clf_{rep}_"): cm += np.array(dr[m][ch][str(L)][cfg]["10000"]["cm"], float)
                for i, fam in enumerate(AH): keys[f"drc_{rep}_{fam}_{cfg}_{ch}{L}"] = f"{100 * cm[i, i] / cm[i].sum():.0f}\\%"

# --- dryf fazy: dokladnosc na rodzine (N=1e4), z results/drift_perclass_base.json (macierze pomylek z eval_drift)
pcd = load(os.path.join(R, "drift_perclass_base.json")) if PFX == "" else None
if pcd:
    for k, v in pcd["acc"].items():
        rep, chl, cfg, N = k.split("|")
        if N != "10000": continue
        for i, fam in enumerate(pcd["labels"]): keys[f"drc_{rep}_{fam}_{cfg}_{chl}"] = f"{100 * v[i]:.0f}\\%"

# --- zestaw Ahmeda: dokladnosc na rodzine, sieci na dokladnych obrazach (macierze pomylek zsumowane po ziarnach)
if PFX == "ahm_":
    AH = ("fock", "coherent", "thermal", "num", "gkp", "binomial", "cat")
    for rep in "WQ":
        for ch, L in (("loss", 60), ("loss", 80), ("dephase", 20), ("gauss", 100)):
            cm = None
            for f in sorted(glob.glob(os.path.join(R, f"eval_sweep_clf_{rep}_exact_s*.json"))):
                e = load(f)
                try: m = np.array(e[ch][str(L)]["cm"], float)
                except Exception: continue
                cm = m if cm is None else cm + m
            if cm is not None:
                for i, fam in enumerate(AH): keys[f"cls_{rep}_{fam}_{ch}{L}"] = f"{100 * cm[i, i] / cm[i].sum():.0f}\\%"

# --- liczebnosci
c = load(os.path.join(R, "counts.json"))
if c:
    for k in ("n_train", "n_val", "n_test"):
        if k in c: keys[k] = f"{c[k]:,}".replace(",", "\\,")
keys["M_sigma"] = "2000"; keys["sdp_dim"] = "48"

os.makedirs(OUTDIR, exist_ok=True)

# --- regula wyboru pomiaru: dokladnosc W i Q przy N=1e2,1e3,1e4 (sieci 'shots', srednia po ziarnach), klucze rule_N{N}_{tag}_{rep}
_rs = {}
for _f in [os.path.join(R, "eval_shots.json")] + sorted(glob.glob(os.path.join(R, "eval_shots_s[0-9].json"))):
    _d = load(_f)
    if _d: _rs.update(_d)
if _rs:
    for tag, ch, L in (("clean", "loss", 0), ("loss50", "loss", 50), ("loss80", "loss", 80), ("deph20", "dephase", 20)):
        for N in (100, 1000, 10000):
            aa = {}
            for rep in "WQ":
                c = [_rs[m][ch][str(N)][str(L)] for m in _rs if m.startswith(f"clf_{rep}_shots_s") and ch in _rs[m]]
                if c: aa[rep] = Acc([x["acc"] for x in c], [x["n"] for x in c]); put(f"rule_N{N}_{tag}_{rep}", aa[rep])
            if len(aa) == 2: put_diff(f"rule_N{N}_{tag}_dQW", aa["Q"], aa["W"], os.path.join(R, "pary", f"{ch}_{L}_N{N}.npz"))
            # homodyna bez rekonstrukcji (sino) wobec Q; sparowanie z qmi/x8/pary.py --sino (katalog pary_sino)
            cs = [sino[m][ch][str(N)][str(L)] for m in sino if "_sino_s" in m and ch in sino[m]] if sino else []
            if cs and "Q" in aa:
                a_s = Acc([x["acc"] for x in cs], [x["n"] for x in cs]); put(f"rule_N{N}_{tag}_S", a_s)
                put_diff(f"rule_N{N}_{tag}_dQS", aa["Q"], a_s, os.path.join(R, "pary_sino", f"{ch}_{L}_N{N}.npz"))

# --- wplyw samej rekonstrukcji, S - W (S: homodyna bez rekonstrukcji, W: homodyna + FBP), w punktach z przedzialem 95%
# dSW : sparowanie po stanach z pary/ (W) i pary_sino/ (S); te same stany testowe, zdarzenia losowane osobno
# dSWe: te same zdarzenia (qmi/eq/pary_sw.py, katalog pary_sw/), takze N=1e5 i N=0 (klucz Ninf); sw_N{N}_{tag}_{W,S} = dokladnosci
def _sw(M, y, names, k):
    c = paired_diff(None, grp=lambda n: "_sino_" in n, M=M, y=y, names=names); keys[k] = f"{100 * c['mean']:.1f}"; CI[k] = c
for tag, ch, L in (("clean", "loss", 0), ("loss50", "loss", 50), ("loss80", "loss", 80), ("deph20", "dephase", 20)):
    for N in (100, 1000, 10000):
        fw, fs = os.path.join(R, "pary", f"{ch}_{L}_N{N}.npz"), os.path.join(R, "pary_sino", f"{ch}_{L}_N{N}.npz")
        if os.path.exists(fw) and os.path.exists(fs):
            a, b = np.load(fw), np.load(fs)
            if not np.array_equal(a["y"], b["y"]): raise SystemExit(f"BLAD: rozne stany w pary/pary_sino {ch} {L} N={N}")
            iw = [i for i, n in enumerate(a["names"]) if str(n).startswith("clf_W_shots")]; js = [j for j, n in enumerate(b["names"]) if "_sino_" in str(n)]
            if len(iw) > 1 and len(js) > 1:
                _sw(np.vstack([a["correct"][iw], b["correct"][js]]).astype(float), a["y"], [str(a["names"][i]) for i in iw] + [str(b["names"][j]) for j in js], f"rule_N{N}_{tag}_dSW")
    for N in (100, 1000, 10000, 100000, 0):
        fe = os.path.join(R, "pary_sw", f"{ch}_{L}_N{N}.npz")
        if os.path.exists(fe):
            d = np.load(fe); nm = [str(n) for n in d["names"]]; Mc = d["correct"].astype(float); nk = "inf" if N == 0 else str(N)
            _sw(Mc, d["y"], nm, f"rule_N{nk}_{tag}_dSWe")
            isS = np.array(["_sino_" in n for n in nm])
            keys[f"sw_N{nk}_{tag}_S"] = pct(float(Mc[isS].mean())); keys[f"sw_N{nk}_{tag}_W"] = pct(float(Mc[~isS].mean()))

# --- decyzja binarna "GKP czy nie" z macierzy pomylek eval_shots (wiersze: prawdziwa klasa, kolumny: przewidziana).
# Czulosc = P(przewidziane GKP | GKP), swoistosc = P(przewidziane nie-GKP | nie GKP); srednia po ziarnach. Klucze
# bin_gkp_N{N}_{tag}_{rep}_{sens,spec} dla rep W, Q (sieci shots) i S (sieci sino). GKP: indeks 5 w zbiorze bazowym, 4 u Ahmeda.
_G = 4 if PFX.startswith("ahm") else 5
def _bin(cms):
    se, sp = [], []
    for cm in cms:
        cm = np.array(cm, float); tp = cm[_G, _G]; pos = cm[_G].sum(); fp = cm[:, _G].sum() - tp; neg = cm.sum() - pos
        se.append(tp / pos); sp.append(1 - fp / neg)
    return float(np.mean(se)), float(np.mean(sp))
# rama testu diagnostycznego. LR+ = czulosc / (1 - swoistosc); "alarm" = P(nie GKP naprawde | test mowi "nie GKP")
# przy czestosci awarii zrodla 10% (stan docelowy GKP w 90% przebiegow); oba z usrednionych czulosci i swoistosci.
_PFAIL = 0.10
def _diag(se, sp):
    lrp = se / (1 - sp) if sp < 1 else float("inf")
    alarm = sp * _PFAIL / (sp * _PFAIL + (1 - se) * (1 - _PFAIL))
    return lrp, alarm
for tag, ch, L in (("clean", "loss", 0), ("loss50", "loss", 50), ("loss80", "loss", 80), ("deph20", "dephase", 20)):
    for N in (100, 1000, 10000):
        for rep, src, pat in (("W", _rs, "clf_W_shots_s"), ("Q", _rs, "clf_Q_shots_s"), ("S", sino or {}, "_sino_s")):
            cms = [src[m][ch][str(N)][str(L)]["cm"] for m in src if pat in m and ch in src[m] and "cm" in src[m][ch].get(str(N), {}).get(str(L), {})]
            if cms:
                se, sp = _bin(cms); keys[f"bin_gkp_N{N}_{tag}_{rep}_sens"] = pct(se); keys[f"bin_gkp_N{N}_{tag}_{rep}_spec"] = pct(sp)
                lrp, alarm = _diag(se, sp)
                if lrp == float("inf"):   # brak falszywych "GKP": dolna granica przy jednym falszywym alarmie wsrod wszystkich nie-GKP
                    nneg = float(np.mean([np.array(c).sum() - np.array(c)[_G].sum() for c in cms])); keys[f"bin_gkp_N{N}_{tag}_{rep}_lrp"] = f"{{>}}\\,{se * nneg:.0f}"
                else: keys[f"bin_gkp_N{N}_{tag}_{rep}_lrp"] = f"{lrp:.0f}" if lrp >= 10 else f"{lrp:.1f}"
                keys[f"bin_gkp_N{N}_{tag}_{rep}_alarm"] = pct(alarm)

# --- 5-krotna walidacja krzyzowa: eval_shots_cv5.json (sieci cvshots_f0-4) i eval_sweep_clf_*_cvexact_f*.json.
# Klucze cv_N{N}_{tag}_{rep} (srednia po foldach), cvsd_... (odchylenie standardowe, pkt), cvd_N{N}_{tag} (roznica Q-W, srednia), cvdsd_... (jej sd)
def pts(x):
    return f"{100 * x:.1f}"
_cv = load(os.path.join(R, "eval_shots_cv5.json"))
if _cv:
    for tag, ch, L in (("clean", "loss", 0), ("loss50", "loss", 50), ("loss80", "loss", 80), ("deph20", "dephase", 20)):
        for N in (100, 1000, 10000):
            per = {}
            for rep in "WQ":
                v = [_cv[f"clf_{rep}_cvshots_f{k}"][ch][str(N)][str(L)]["acc"] for k in range(5) if f"clf_{rep}_cvshots_f{k}" in _cv]
                if v:
                    per[rep] = np.array(v); keys[f"cv_N{N}_{tag}_{rep}"] = pct(float(np.mean(v)))
                    if len(v) > 1: keys[f"cvsd_N{N}_{tag}_{rep}"] = pts(float(np.std(v, ddof=1)))
            if len(per) == 2 and len(per["W"]) == len(per["Q"]) > 1:
                d = per["Q"] - per["W"]; keys[f"cvd_N{N}_{tag}"] = pts(float(np.mean(d))); keys[f"cvdsd_N{N}_{tag}"] = pts(float(np.std(d, ddof=1))); keys[f"cvneg_N{N}_{tag}"] = str(int((d < 0).sum()))
    keys["cv_nfolds"] = str(len([m for m in _cv if m.startswith("clf_W_cvshots_f")]))
for rep in "WQ":
    _ex = [load(f) for f in sorted(glob.glob(os.path.join(R, f"eval_sweep_clf_{rep}_cvexact_f*.json")))]
    _ex = [e for e in _ex if e]
    for tag, ch, L in (("clean", "loss", 0), ("loss60", "loss", 60), ("loss80", "loss", 80), ("deph20", "dephase", 20), ("gauss100", "gauss", 100)):
        v = [e[ch][str(L)]["acc"] for e in _ex if ch in e and str(L) in e[ch]]
        if len(v) > 1:
            keys[f"cvex_{tag}_{rep}"] = pct(float(np.mean(v))); keys[f"cvexsd_{tag}_{rep}"] = pts(float(np.std(v, ddof=1)))
# --- bazy niesieciowe z hiperparametrami z walidacji krzyzowej: baseline_cv.json
_bcv = load(os.path.join(R, "baseline_cv.json"))
if _bcv:
    for name, r in _bcv.items():
        keys[f"bcv_{name}_clean"] = pct(r["loss"]["0"]["acc"]); keys[f"bcv_{name}_loss80"] = pct(r["loss"]["80"]["acc"])
        keys[f"bcv_{name}_cv"] = pct(r["cv"]["best_mean"]); keys[f"bcv_{name}_cvsd"] = pts(r["cv"]["best_std"])

# --- ROC/AUC i kalibracja: scores_summary.json (skrot scores.json z qmi.eval_scores; N=-1 = dokladne obrazy)
# Klucze auc_N{N}_{tag}_{rep} (makro-AUC, 3 miejsca), ece_N{N}_{tag}_{rep}; N=ex dla sieci na dokladnych obrazach
_sc = load(os.path.join(R, "scores_summary.json"))
if _sc:
    for tag, ch, L in (("clean", "loss", 0), ("loss50", "loss", 50), ("loss80", "loss", 80), ("deph20", "dephase", 20)):
        for N, kind, nk in ((-1, "exact", "ex"), (100, "shots", "100"), (1000, "shots", "1000"), (10000, "shots", "10000")):
            for rep in "WQ":
                v = [m[ch][str(N)][str(L)] for n, m in _sc.items() if n.startswith(f"clf_{rep}_{kind}_s") and ch in m and str(N) in m[ch] and str(L) in m[ch][str(N)]]
                if v and all(x["auc_macro"] is not None for x in v):
                    keys[f"auc_N{nk}_{tag}_{rep}"] = f"{np.mean([x['auc_macro'] for x in v]):.3f}"
                    keys[f"ece_N{nk}_{tag}_{rep}"] = f"{np.mean([x['ece'] for x in v]):.2f}"

# --- (zbior eq): kontrola energii (qmi.energy_control -> energy_control.json, klasyfikator na samym oszacowaniu <n>,
# czyste stany = kanal loss poziom 0) oraz kontrola okna i bledu FBP (qmi/eq/okno_fbp.py -> okno_fbp.json)
_ec = load(os.path.join(R, "energy_control.json"))
if _ec:
    for det in ("het", "hom"):
        for N in (0, 100, 1000, 10000):
            r = _ec.get(f"{det}_N{N}", {}).get("loss", {}).get("0")
            if r: keys[f"ec_{det}_N{N}"] = pct(r["acc"]); keys[f"ec_{det}_N{N}_hi"] = pct(r["hi"])
    fin = [_ec[f"{d}_N{N}"]["loss"]["0"]["acc"] for d in ("het", "hom") for N in (100, 1000, 10000) if f"{d}_N{N}" in _ec]
    if fin: keys["ec_max"] = pct(max(fin))
_ok = load(os.path.join(R, "okno_fbp.json"))
if _ok:
    keys["win_qmass_min"] = f"{_ok['qmass_min']:.3f}"; keys["fbp_err_med"] = f"{_ok['fbp_err_med']:.2f}"; keys["fbp_err_p90"] = f"{_ok['fbp_err_p90']:.2f}"

for k, c in CI.items():
    if c.get("diff"):
        keys[f"{k}_lo"] = f"{100 * c['lo']:.1f}"; keys[f"{k}_hi"] = f"{100 * c['hi']:.1f}"
        keys[f"{k}_ci"] = f"\\ensuremath{{[{100 * c['lo']:.1f},\\,{100 * c['hi']:.1f}]}}"
    else:
        keys[f"{k}_lo"] = pct(c["lo"]); keys[f"{k}_hi"] = pct(c["hi"])
        keys[f"{k}_ci"] = f"\\ensuremath{{[{100 * c['lo']:.1f},\\,{100 * c['hi']:.1f}]}}\\%"
with open(os.path.join(OUTDIR, "ci%s.json" % ("_" + PFX.rstrip("_") if PFX else "")), "w") as f:
    json.dump(CI, f, indent=1, sort_keys=True)

with open(OUT, "w") as f:
    f.write("% generated by scripts/make_numbers.py -- do not edit by hand\n")
    for k, v in sorted(keys.items()):
        v = re.sub(r"^-(?=\d)", r"\\ensuremath{-}", re.sub(r"^-(?=0\.0+$)", "", str(v)))   # minus zamiast lacznika w tekscie (figury normalizuja z powrotem)
        f.write(f"\\expandafter\\def\\csname res@{PFX}{k}\\endcsname{{{v}}}\n")
print(f"{len(keys)} kluczy -> {OUT}")
for k, v in sorted(keys.items()): print(f"  {k:18s} {v}")
