"""Generator zbiorow QMI: Wigner + Husimi Q z TEJ SAMEJ macierzy gestosci, sweepy szumu w ukladzie
zrownowazonym, wznawialny (pliki czesciowe), ziarna rozlaczne miedzy zbiorami z konstrukcji.

Kazde zadanie = (split, kanal, poziom, klasa, chunk) -> jeden plik  part_*.npz. Istniejace pomijane.

  python -m qmi.gen sweep --split test  --out data/qmi --n 100 --workers 24
  python -m qmi.gen mixed --split train --out data/qmi --n-total 20000 --workers 24
"""
import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from qsdl.config import CUTOFF, GRID, XMAX
from qsdl.labels import LABELS, LABEL_TO_ID
from qsdl.states import sample_state
from qmi.holdout import make_sampler
from qmi.ahmed_states import AHM_LABELS, sample_ahmed, make_sampler_ahm
from qmi.eqn_states import EQN_LABELS, make_sampler_eqn
from qsdl.wigner import apply_wigner
from qsdl.husimi import apply_husimi
from qsdl.noise import (fixed_photon_loss_params, fixed_dephase_params, fixed_gauss_params,
                        apply_channel_noise, gaussian_noise, generate_params)

SPLITS = {"train": 0, "val": 1, "test": 2}
CHANNELS = ("loss", "dephase", "gauss")
CH_ID = {"clean": 0, "loss": 1, "dephase": 2, "gauss": 3, "combo": 4}
LEVELS = tuple(range(0, 101, 5))


def level_params(channel, pct):
    if channel == "loss":    return fixed_photon_loss_params(pct / 100.0)
    if channel == "dephase": return fixed_dephase_params(float(pct))
    if channel == "gauss":   return fixed_gauss_params(float(pct))
    return {"noise_level_pct": 0}


# Rodzina eqn: defazowanie gamma = GAMMA_MAX_EQN * p/100 (przy 15 skala nasycala sie od ok. 20%),
# a "szum gaussowski" to splot W i Q z tym samym gaussem o wariancji s2 (jednostki kwadratur, proznia = 1/2) zamiast niezaleznego
# szumu na pikselach: fizyczny kanal losowych przesuniec, W*G_s2 jest funkcja Wignera stanu, Q*G_s2 = W*G_(s2+1/2).
GAMMA_MAX_EQN = 3.0
CONV_S2_MAX = 1.0          # przemiatanie: s2 = CONV_S2_MAX * p/100 (przy s2 >= 1/2 W traci ujemnosc)
CONV_S2_COMBO = (0.0, 0.5) # combo: s2 ~ U(...) zamiast szumu detektora 10-30 dB wzgledem rms obrazu


def level_params_eqn(channel, pct):
    if channel == "dephase": return {"dephase": GAMMA_MAX_EQN * pct / 100.0, "noise_level_pct": int(round(pct))}
    if channel == "gauss":   return {"conv_s2": CONV_S2_MAX * pct / 100.0, "noise_level_pct": int(round(pct))}
    return level_params(channel, pct)


def convolve(img, s2):
    from scipy.ndimage import gaussian_filter
    if s2 <= 0: return img
    return gaussian_filter(np.asarray(img, np.float64), np.sqrt(s2) / (2 * XMAX / (GRID - 1)), mode="constant", truncate=6.0).astype(img.dtype)


def images(rho):
    return apply_wigner(rho, GRID, XMAX), apply_husimi(rho, GRID, XMAX)


def noisy_sample(label, channel, params, rng, sampler=sample_state):
    """Zwraca (W, Q, rho_noisy, rho_clean, meta). Szum gaussowski: sigma = frac * max|obraz| osobno
    dla W i Q (ten sam ulamek szczytu -> porownywalna sila), szum kanalowy na rho -> obie reprezentacje."""
    rho, st = sampler(label, CUTOFF, rng)
    rho_clean = np.asarray(rho.full(), dtype=np.complex64)
    if channel == "gauss":
        W, Q = images(rho)
        frac = params.get("gauss_peak_frac", 0.0)
        if frac > 0:
            W = gaussian_noise(W, frac * np.abs(W).max(), rng)
            Q = gaussian_noise(Q, frac * np.abs(Q).max(), rng)
        rho_n = rho
    else:
        rho_n = apply_channel_noise(rho, params) if params else rho
        W, Q = images(rho_n)
    if params.get("conv_s2", 0) > 0:                     # eqn: splot (rho_n zostaje stanem przed splotem)
        W, Q = convolve(W, params["conv_s2"]), convolve(Q, params["conv_s2"])
    return W, Q, np.asarray(rho_n.full(), dtype=np.complex64), rho_clean, {**st, **params}


def run_task(t):
    path = Path(t["path"])
    if path.exists():
        return str(path), "skip"
    fam = t.get("family", "base")
    noise = t.get("noise") or ("eq" if fam == "eqn" else "x8")   # definicje szumu eq (gamma_max 3, splot) lub x8 (gamma_max 15, szum pikselowy)
    key = [t["seed"], SPLITS[t["split"]], CH_ID[t["channel"]], t["level"] + 1, t["cls"], t["chunk"]]
    ss = np.random.SeedSequence(key + {"ahmed": [7], "eqn": [11]}.get(fam, []))
    rng = np.random.default_rng(ss)
    if fam == "ahmed": label = AHM_LABELS[t["cls"]]; sampler = make_sampler_ahm(t.get("holdout", "none"))
    elif fam == "eqn": label = EQN_LABELS[t["cls"]]; sampler = make_sampler_eqn(t.get("holdout", "none"), bool(t.get("gkpx")))
    else: label = LABELS[t["cls"]]; sampler = make_sampler(t.get("holdout", "none"))
    if t.get("gkpx") and fam != "eqn":            # eqn: gkpx juz w samplerze, z delta dobranym do <n>
        from qmi.gkpx import make_gkpx; sampler = make_gkpx(sampler)
    Ws, Qs, Rn, Rc, ch, lv, metas = [], [], [], [], [], [], []
    for _ in range(t["n"]):
        if t["channel"] == "combo":                      # mieszanka jak w oryginalnym generate_params
            params, chan = generate_params(rng), "combo"
            lev = -1
            if noise == "eq": params.pop("gauss", None); params["conv_s2"] = float(rng.uniform(*CONV_S2_COMBO))
        else:
            chan, lev = t["channel"], t["level"]
            params = (level_params_eqn if noise == "eq" else level_params)(chan, lev) if chan != "clean" else {}
        W, Q, rn, rc, meta = noisy_sample(label, "gauss" if "gauss_peak_frac" in params else chan, params, rng, sampler)
        if chan == "combo" and "gauss" in params:         # szum detektora (SNR w dB wzgledem rms obrazu), osobno W i Q
            f = 10 ** (params["gauss"] / 20)
            W = gaussian_noise(W, float(np.sqrt((W ** 2).mean())) / f, rng)
            Q = gaussian_noise(Q, float(np.sqrt((Q ** 2).mean())) / f, rng)
        Ws.append(W); Qs.append(Q); Rn.append(rn); Rc.append(rc)
        ch.append(CH_ID[chan]); lv.append(lev); metas.append(str({k: (float(v) if not isinstance(v, str) else v) for k, v in meta.items()}))
    tmp = path.with_name(path.name + ".tmp.npz")
    extra = {"xmax": np.float64(XMAX)} if "QMI_XMAX" in os.environ else {}   # znacznik okna tylko w nowych przebiegach
    _save = np.savez_compressed if os.environ.get("QMI_NPZ_COMPRESS") == "1" else np.savez
    _save(tmp, W=np.stack(Ws), Q=np.stack(Qs), rho=np.stack(Rn), rho_clean=np.stack(Rc),
             label=np.full(t["n"], t["cls"], np.int16), channel=np.array(ch, np.int8), level=np.array(lv, np.int16),
             meta=np.array(metas, dtype=object), **extra)
    os.replace(tmp, path)
    return str(path), "done"


def tasks_sweep(a):
    out = Path(a.out) / a.split / "sweep"; out.mkdir(parents=True, exist_ok=True)
    ts = []; labs = {"ahmed": AHM_LABELS, "eqn": EQN_LABELS}.get(a.family, LABELS)
    for chn in a.channels.split(","):
        for lev in a.levels:
            for c in range(7):
                ts.append(dict(path=str(out / f"part_{chn}_{lev:03d}_{labs[c]}.npz"), split=a.split, channel=chn,
                               level=lev, cls=c, chunk=0, n=a.n, seed=a.seed, holdout=a.holdout, family=a.family, noise=a.noise))
    return ts


def tasks_mixed(a):
    """Zbior treningowy: klasy po rowno; na kazda klase chunki po 200 z losowym kanalem/poziomem na chunk-niezaleznie:
    kazdy chunk = jeden (kanal,poziom) wylosowany z ziarna -> rownomiernie pokrywa siatke."""
    out = Path(a.out) / a.split / "mixed"; out.mkdir(parents=True, exist_ok=True)
    shared = (a.noise or ("eq" if a.family == "eqn" else "x8")) == "eq"
    per_class = a.n_total // 7; size = 25 if shared else 100; nch = per_class // size   # eq -> pliki po 25 (pelne pokrycie komorek)
    if getattr(a, "grid", "full") == "narrow":   # waski zakres jak w generate_params: straty <=40%, dephasing/gauss <=5%
        grid = [("clean", 0)] + [("loss", l) for l in LEVELS if 0 < l <= 40] + [("dephase", 5), ("gauss", 5)] + [("combo", -1)] * 6
    else:
        grid = [("clean", 0)] + [(c, l) for c in CHANNELS for l in LEVELS if l > 0] + [("combo", -1)] * 6
    rng = np.random.default_rng([a.seed, SPLITS[a.split], 99])
    strat = getattr(a, "stratify", False)
    if strat or shared:   # gwarantowane chunki czyste i combo; reszta z pozostalych komorek
        other = [g for g in grid if g[0] not in ("clean", "combo")]
        n_clean, n_combo = max(1, nch // 14), max(1, nch // 7)
    if shared:   # jedna ZBALANSOWANA sekwencja komorek, wspolna dla wszystkich klas:
        rest = nch - n_clean - n_combo   # kazda komorka szumu co najmniej raz, gdy rest >= len(other); reszta losowo bez powtorzen
        reps, extra = divmod(rest, len(other))
        noise = other * reps + [other[i] for i in rng.choice(len(other), extra, replace=False)]
        cells0 = [("clean", 0)] * n_clean + [("combo", -1)] * n_combo + noise
        rng.shuffle(cells0)
    ts = []
    for c in range(7):
        if shared: cells = cells0
        elif strat:
            cells = [("clean", 0)] * n_clean + [("combo", -1)] * n_combo + [other[int(rng.integers(len(other)))] for _ in range(nch - n_clean - n_combo)]
            rng.shuffle(cells)
        for k in range(nch):
            chn, lev = cells[k] if (strat or shared) else grid[int(rng.integers(len(grid)))]
            ts.append(dict(path=str(out / f"part_{c}_{k:04d}.npz"), split=a.split, channel=chn, level=lev,
                           cls=c, chunk=k, n=size, seed=a.seed, holdout=a.holdout, family=a.family, gkpx=getattr(a, "gkpx", False), noise=a.noise))
    return ts


def main():
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("sweep", "mixed"))
    p.add_argument("--split", required=True, choices=tuple(SPLITS)); p.add_argument("--out", default="data/qmi")
    p.add_argument("--n", type=int, default=100, help="probek na klase na poziom (sweep)")
    p.add_argument("--n-total", type=int, default=20000, help="razem (mixed)")
    p.add_argument("--channels", default=",".join(CHANNELS)); p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--family", default="base", choices=("base", "ahmed", "eqn"),
                   help="eqn: zestaw bazowy z rowna <n> we wszystkich klasach (qmi.eqn_states), proznia -> sciśnieta proznia")
    p.add_argument("--noise", default=None, choices=("x8", "eq"),
                   help="definicje szumu: x8 (gamma_max 15, szum pikselowy) albo eq (gamma_max 3, splot); domyslnie eq dla --family eqn, x8 dla reszty")
    p.add_argument("--grid", default="full", choices=("full", "narrow"))
    p.add_argument("--holdout", default="none", choices=("none", "train", "test"))
    p.add_argument("--gkpx", action="store_true", help="mixed: 50%% klasy gkp to scisniety GKP o szerokiej obwiedni (rodzina Konno)")
    p.add_argument("--stratify", action="store_true", help="mixed: gwarantowane chunki czyste i combo w kazdej klasie")
    p.add_argument("--levels", default=",".join(map(str, LEVELS)))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args(); a.levels = [int(x) for x in a.levels.split(",")]
    ts = tasks_sweep(a) if a.mode == "sweep" else tasks_mixed(a)
    todo = [t for t in ts if not Path(t["path"]).exists()]
    print(f"{len(ts)} zadan, do zrobienia {len(todo)}, XMAX={XMAX}, GRID={GRID}", flush=True)
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(run_task, t) for t in todo]):
            f.result(); done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
