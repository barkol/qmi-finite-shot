"""Data set generator: Wigner function W and Husimi Q function computed from the SAME density matrix, balanced noise sweeps,
resumable (partial files); the seeds of different splits are disjoint by construction.

Each task = (split, channel, level, class, chunk) -> one file part_*.npz. Existing files are skipped.

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


# Family eqn: dephasing gamma = GAMMA_MAX_EQN * p/100 (with 15 the scale saturates from about 20%), and "Gaussian noise" is the
# convolution of W and Q with the same Gaussian of variance s2 (quadrature units, vacuum = 1/2) instead of independent pixel noise:
# the physical channel of random displacements, W*G_s2 is the Wigner function of a state, Q*G_s2 = W*G_(s2+1/2).
GAMMA_MAX_EQN = 3.0
CONV_S2_MAX = 1.0          # sweep: s2 = CONV_S2_MAX * p/100 (for s2 >= 1/2 W loses its negativity)
CONV_S2_COMBO = (0.0, 0.5) # combo: s2 ~ U(...) instead of detector noise 10-30 dB relative to the image rms


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
    """Return (W, Q, rho_noisy, rho_clean, meta). Pixel Gaussian noise: sigma = frac * max|image|, separately for W and Q
    (the same fraction of the peak -> comparable strength); a channel acts on rho -> both representations."""
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
    if params.get("conv_s2", 0) > 0:                     # eqn: convolution (rho_n stays the state before the convolution)
        W, Q = convolve(W, params["conv_s2"]), convolve(Q, params["conv_s2"])
    return W, Q, np.asarray(rho_n.full(), dtype=np.complex64), rho_clean, {**st, **params}


def run_task(t):
    path = Path(t["path"])
    if path.exists():
        return str(path), "skip"
    fam = t.get("family", "base")
    noise = t.get("noise") or ("eq" if fam == "eqn" else "x8")   # noise definitions eq (gamma_max 3, convolution) or x8 (gamma_max 15, pixel noise)
    key = [t["seed"], SPLITS[t["split"]], CH_ID[t["channel"]], t["level"] + 1, t["cls"], t["chunk"]]
    ss = np.random.SeedSequence(key + {"ahmed": [7], "eqn": [11]}.get(fam, []))
    rng = np.random.default_rng(ss)
    if fam == "ahmed": label = AHM_LABELS[t["cls"]]; sampler = make_sampler_ahm(t.get("holdout", "none"))
    elif fam == "eqn": label = EQN_LABELS[t["cls"]]; sampler = make_sampler_eqn(t.get("holdout", "none"))
    else: label = LABELS[t["cls"]]; sampler = make_sampler(t.get("holdout", "none"))
    Ws, Qs, Rn, Rc, ch, lv, metas = [], [], [], [], [], [], []
    for _ in range(t["n"]):
        if t["channel"] == "combo":                      # mixture as in qsdl generate_params
            params, chan = generate_params(rng), "combo"
            lev = -1
            if noise == "eq": params.pop("gauss", None); params["conv_s2"] = float(rng.uniform(*CONV_S2_COMBO))
        else:
            chan, lev = t["channel"], t["level"]
            params = (level_params_eqn if noise == "eq" else level_params)(chan, lev) if chan != "clean" else {}
        W, Q, rn, rc, meta = noisy_sample(label, "gauss" if "gauss_peak_frac" in params else chan, params, rng, sampler)
        if chan == "combo" and "gauss" in params:         # detector noise (SNR in dB relative to the image rms), separately for W and Q
            f = 10 ** (params["gauss"] / 20)
            W = gaussian_noise(W, float(np.sqrt((W ** 2).mean())) / f, rng)
            Q = gaussian_noise(Q, float(np.sqrt((Q ** 2).mean())) / f, rng)
        Ws.append(W); Qs.append(Q); Rn.append(rn); Rc.append(rc)
        ch.append(CH_ID[chan]); lv.append(lev); metas.append(str({k: (float(v) if not isinstance(v, str) else v) for k, v in meta.items()}))
    tmp = path.with_name(path.name + ".tmp.npz")
    extra = {"xmax": np.float64(XMAX)} if "QMI_XMAX" in os.environ else {}   # window key, written when QMI_XMAX is set
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
    """Training set: equal class sizes; per class, chunks with one (channel, level) per chunk drawn from the seed,
    so that the chunks cover the grid of noise cells uniformly."""
    out = Path(a.out) / a.split / "mixed"; out.mkdir(parents=True, exist_ok=True)
    shared = (a.noise or ("eq" if a.family == "eqn" else "x8")) == "eq"
    per_class = a.n_total // 7; size = 25 if shared else 100; nch = per_class // size   # eq -> files of 25 (full coverage of the cells)
    if getattr(a, "grid", "full") == "narrow":   # narrow range as in generate_params: loss <= 40%, dephasing/gauss <= 5%
        grid = [("clean", 0)] + [("loss", l) for l in LEVELS if 0 < l <= 40] + [("dephase", 5), ("gauss", 5)] + [("combo", -1)] * 6
    else:
        grid = [("clean", 0)] + [(c, l) for c in CHANNELS for l in LEVELS if l > 0] + [("combo", -1)] * 6
    rng = np.random.default_rng([a.seed, SPLITS[a.split], 99])
    strat = getattr(a, "stratify", False)
    if strat or shared:   # guaranteed clean and combo chunks; the rest from the other cells
        other = [g for g in grid if g[0] not in ("clean", "combo")]
        n_clean, n_combo = max(1, nch // 14), max(1, nch // 7)
    if shared:   # one BALANCED sequence of cells, shared by all classes:
        rest = nch - n_clean - n_combo   # every noise cell at least once if rest >= len(other); the rest drawn without repetition
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
                           cls=c, chunk=k, n=size, seed=a.seed, holdout=a.holdout, family=a.family, noise=a.noise))
    return ts


def main():
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("sweep", "mixed"))
    p.add_argument("--split", required=True, choices=tuple(SPLITS)); p.add_argument("--out", default="data/qmi")
    p.add_argument("--n", type=int, default=100, help="samples per class per level (sweep)")
    p.add_argument("--n-total", type=int, default=20000, help="total (mixed)")
    p.add_argument("--channels", default=",".join(CHANNELS)); p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--family", default="base", choices=("base", "ahmed", "eqn"),
                   help="eqn: base set with equal <n> in all classes (qmi.eqn_states), vacuum -> squeezed vacuum")
    p.add_argument("--noise", default=None, choices=("x8", "eq"),
                   help="noise definitions: x8 (gamma_max 15, pixel noise) or eq (gamma_max 3, convolution); default eq for --family eqn, x8 otherwise")
    p.add_argument("--grid", default="full", choices=("full", "narrow"))
    p.add_argument("--holdout", default="none", choices=("none", "train", "test"))
    p.add_argument("--stratify", action="store_true", help="mixed: guaranteed clean and combo chunks in every class")
    p.add_argument("--levels", default=",".join(map(str, LEVELS)))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args(); a.levels = [int(x) for x in a.levels.split(",")]
    ts = tasks_sweep(a) if a.mode == "sweep" else tasks_mixed(a)
    todo = [t for t in ts if not Path(t["path"]).exists()]
    print(f"{len(ts)} tasks, {len(todo)} to do, XMAX={XMAX}, GRID={GRID}", flush=True)
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(run_task, t) for t in todo]):
            f.result(); done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
