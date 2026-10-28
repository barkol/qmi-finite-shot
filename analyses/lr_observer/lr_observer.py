#!/usr/bin/env python
"""Bayes-optimal (likelihood-ratio) observer for the BASE set (qmi.eqn_states), Level 2a.

The observer knows the library of prototype states of every family (finite lists for Fock, thermal, GKP, binomial,
a fine grid of phases/axes for coherent, squeezed vacuum, cat), knows the noise cell, and decides the family from the
exact likelihood of N independent detection events, summed over the prototypes of a family (equal prior on families,
and the sampling distribution of the base set inside a family). Records are drawn from the stored TEST images of
data_eq with the same procedures as qmi/shots.py (multinomial counts on the same 128x128 grid for Q, K-phase
multinomial marginals of the discrete Radon transform of W with split_events for homodyne).

UPPER BOUND on any classifier for KNOWN prototypes. Not a performance claim for unknown states.

  QMI_XMAX=12 PYTHONPATH=~/work/qsdl-qmi python lr_observer.py build|run|agg|all [--smoke] [--workers 12]
Checkpoints: cache/proto_<cell>.npz (library), parts/part_<cell>_<family>.npz (predictions); existing files are skipped.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("QMI_XMAX", "12")
import argparse, glob, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
DATA = Path(os.path.expanduser("~/work/qsdl-qmi/data_eq/qmi/test/sweep"))
RES = Path(os.path.expanduser("~/work/qmi-finite-shot/paper/results"))
NS = (100, 1000, 10000)
KS = (36, 12, 6)
CELLS = {"clean": ("loss", 0), "deph20": ("dephase", 20), "loss50": ("loss", 50), "loss80": ("loss", 80),
         "gauss20": ("gauss", 20), "gauss50": ("gauss", 50)}
FAMS = ("fock", "coherent", "sqvac", "thermal", "cat", "gkp", "binomial")
SEED = 20261028


def lib_units(smoke):
    """List of (family index, kind, nbar, params) base states; rotations are applied to units flagged rot."""
    from qmi.eqn_states import NBAR, _BIN, _gkp_delta, _cat_amp, _trunc_param
    units = []
    for nb in NBAR:
        units.append((0, "fock", nb, {}))
        units.append((1, "coherent", nb, {}))
        units.append((2, "sqvac", nb, {"r": _trunc_param("sqvac", nb)}))
        units.append((3, "thermal", nb, {"nt": _trunc_param("thermal", nb)}))
        for phi in (0.0, np.pi): units.append((4, "cat", nb, {"phi": phi, "amp": _cat_amp(nb, phi)}))
        for mu in (0, 1): units.append((5, "gkp", nb, {"mu": mu, "delta": _gkp_delta(nb, mu)}))
        for (S, K, mu) in _BIN[nb]: units.append((6, "binomial", nb, {"S": S, "K": K, "mu": mu}))
    return units


def n_rot(kind, smoke):
    full = {"coherent": 144, "sqvac": 72, "cat": 72}   # 2.5 degree steps (period 2pi coherent, pi squeezed/cat)
    sm = {"coherent": 12, "sqvac": 6, "cat": 6}
    return (sm if smoke else full).get(kind, 1)


def build_unit(args):
    from qutip import Qobj
    from qsdl.config import CUTOFF, GRID, XMAX
    from qsdl.states import fock_state, coherent_state, thermal_state, cat_state, binomial_state, gkp_state
    from qsdl.wigner import apply_wigner
    from qsdl.husimi import apply_husimi
    from qsdl.noise import apply_channel_noise
    from qmi.eqn_states import sqvac_state
    from qmi.gen import convolve, level_params_eqn
    from qmi.shots import radon
    (fi, kind, nb, p), ch, lev, smoke = args
    if kind == "fock": rho = fock_state(CUTOFF, nb)
    elif kind == "coherent": rho = coherent_state(CUTOFF, np.sqrt(nb))
    elif kind == "sqvac": rho = sqvac_state(CUTOFF, p["r"], 0.0)
    elif kind == "thermal": rho = thermal_state(CUTOFF, p["nt"])
    elif kind == "cat": rho = cat_state(CUTOFF, p["amp"], p["phi"])
    elif kind == "gkp": rho = gkp_state(CUTOFF, p["delta"], p["mu"])
    else: rho = binomial_state(CUTOFF, p["S"], p["K"], p["mu"])
    par = level_params_eqn(ch, lev) if not (ch == "loss" and lev == 0) else {}
    rho = apply_channel_noise(rho, par) if (par and ch != "gauss") else rho
    a = rho.full(); m = np.arange(a.shape[0]); dm = m[:, None] - m[None, :]
    R = n_rot(kind, smoke); span = {"coherent": 2 * np.pi}.get(kind, np.pi)
    th = np.linspace(0, 180, 36, endpoint=False)
    LQ, LP = [], []
    for j in range(R):
        phi = span * j / R if R > 1 else 0.0
        r = Qobj(a * np.exp(1j * phi * dm))
        W, Q = apply_wigner(r, GRID, XMAX), apply_husimi(r, GRID, XMAX)
        if par.get("conv_s2", 0) > 0: W, Q = convolve(W, par["conv_s2"]), convolve(Q, par["conv_s2"])
        q = np.clip(np.asarray(Q, np.float64), 0, None).ravel(); q /= q.sum()
        sino = radon(np.asarray(W, np.float64), th); pk = np.clip(sino, 0, None); pk /= pk.sum(1, keepdims=True)
        LQ.append(np.log(np.maximum(q, 1e-30))); LP.append(np.log(np.maximum(pk, 1e-12)))
    return fi, np.array(LQ), np.array(LP), R


def build(cell, workers, smoke, outdir):
    f = outdir / "cache" / f"proto_{cell}.npz"
    if f.exists(): return
    f.parent.mkdir(parents=True, exist_ok=True); ch, lev = CELLS[cell]
    units = lib_units(smoke); t0 = time.time()
    order = sorted(range(len(units)), key=lambda i: -n_rot(units[i][1], smoke))
    jobs = [(units[i], ch, lev, smoke) for i in order]
    with Pool(workers) as pool: res = pool.map(build_unit, jobs, chunksize=1)
    # restore library order; weights: 1/4 per nbar, then uniform over sub-choices of that nbar, then uniform over rotations
    byunit = {order[k]: res[k] for k in range(len(units))}
    fam, LQ, LP, lw = [], [], [], []
    nsub = {}
    for (fi, kind, nb, _p) in units: nsub[(fi, nb)] = nsub.get((fi, nb), 0) + 1
    for i, (fi, kind, nb, _p) in enumerate(units):
        _, q, p, R = byunit[i]
        fam += [fi] * R; LQ.append(q); LP.append(p); lw += [np.log(0.25 / nsub[(fi, nb)] / R)] * R
    np.savez(str(f) + ".tmp.npz", fam=np.array(fam), logq=np.concatenate(LQ), logp=np.concatenate(LP), logw=np.array(lw))
    os.replace(str(f) + ".tmp.npz", f)
    print(f"[build] {cell}: {len(fam)} prototypes, {time.time()-t0:.0f}s", flush=True)


def run_task(args):
    cell, fi, R, smoke, nstate, outdir = args
    from qmi.shots import radon, split_events
    out = outdir / "parts" / f"part_{cell}_{FAMS[fi]}.npz"
    if out.exists(): return str(out), "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    ch, lev = CELLS[cell]; d = np.load(DATA / f"part_{ch}_{lev:03d}_{FAMS[fi]}.npz", allow_pickle=True)
    Wt, Qt = d["W"][:nstate], d["Q"][:nstate]; ns = len(Wt)
    P = np.load(outdir / "cache" / f"proto_{cell}.npz"); fam = P["fam"]; logw = P["logw"]
    logq = P["logq"]; logp = P["logp"]; npro = len(fam)
    sl = [np.where(fam == f)[0] for f in range(7)]
    th = np.linspace(0, 180, 36, endpoint=False)
    pred = {}; t0 = time.time()

    def decide(C, LG):
        pr = np.empty(len(C), np.int8)
        for a in range(0, len(C), 500):
            L = C[a:a + 500].astype(np.float64) @ LG.T + logw[None, :]
            S = np.stack([logsumexp(L[:, s], axis=1) for s in sl], 1); pr[a:a + 500] = S.argmax(1)
        return pr
    for N in NS:
        rng = np.random.default_rng(np.random.SeedSequence([SEED, list(CELLS).index(cell), fi, N]))
        Cq = np.empty((ns * R, logq.shape[1]), np.float32)
        Cw = {K: np.empty((ns * R, K * 128), np.float32) for K in KS}
        for i in range(ns):
            q = np.clip(np.asarray(Qt[i], np.float64), 0, None).ravel(); q /= q.sum()
            Cq[i * R:(i + 1) * R] = rng.multinomial(N, q, size=R)
            sino = radon(np.asarray(Wt[i], np.float64), th); pk = np.clip(sino, 0, None)
            for K in KS:
                per = split_events(N, K); blk = np.zeros((R, K, 128), np.float32)
                for k in range(K):
                    if per[k] == 0: continue
                    ang = k * (36 // K); p = pk[ang]; s = p.sum()
                    if s > 0: blk[:, k] = rng.multinomial(per[k], p / s, size=R)
                Cw[K][i * R:(i + 1) * R] = blk.reshape(R, K * 128)
        pred[f"Q_{N}"] = decide(Cq, logq)
        for K in KS:
            sel = np.arange(K) * (36 // K)
            pred[f"W{K}_{N}"] = decide(Cw[K], logp[:, sel, :].reshape(npro, K * 128))
        print(f"[run] {cell} {FAMS[fi]} N={N} {time.time()-t0:.0f}s  errQ={np.mean(pred[f'Q_{N}']!=fi):.3f} errW36={np.mean(pred[f'W36_{N}']!=fi):.3f}", flush=True)
    np.savez(str(out) + ".tmp.npz", R=R, nstate=ns, **pred); os.replace(str(out) + ".tmp.npz", out)
    return str(out), "done"


def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


def net_errors():
    """Seed-mean pooled error of the stored networks, as in make_numbers.py (loss/dephase cells only)."""
    sh = {}
    for f in [RES / "eval_shots.json"] + sorted(RES.glob("eval_shots_s[0-9].json")): sh.update(json.load(open(f)))
    k12 = {}
    for f in sorted(RES.glob("eval_shots_K12*.json")): k12.update(json.load(open(f)))
    k6 = json.load(open(RES / "eval_shots_Kscan_K6.json"))
    src = {"Q": (sh, "clf_Q_shots_s"), "W36": (sh, "clf_W_shots_s"), "W12": (k12, "shotsK12_s"), "W6": (k6, "shotsK6_s")}
    cellkey = {"clean": ("loss", "0"), "deph20": ("dephase", "20"), "loss50": ("loss", "50"), "loss80": ("loss", "80")}
    out = {}
    for cell, (ch, lv) in cellkey.items():
        for rep, (d, pat) in src.items():
            for N in NS:
                v = [(1 - m[ch][str(N)][lv]["acc"]) for nm, m in d.items() if pat in nm]
                nn = [m[ch][str(N)][lv]["n"] for nm, m in d.items() if pat in nm][0]
                err = float(np.mean(v)); k = err * nn
                out[f"{cell}|{rep}|{N}"] = dict(err=err, lo=wilson(k, nn)[0], hi=wilson(k, nn)[1], nruns=len(v), min=float(min(v)), max=float(max(v)), n=nn)
    return out


def agg(outdir, B=4000):
    parts = sorted((outdir / "parts").glob("part_*.npz")); net = net_errors(); rng = np.random.default_rng(7)
    res = {}
    for cell in CELLS:
        fs = [outdir / "parts" / f"part_{cell}_{f}.npz" for f in FAMS]
        if not all(f.exists() for f in fs): continue
        D = [np.load(f) for f in fs]; R = int(D[0]["R"]); ns = int(D[0]["nstate"])
        E = {}  # per-state error arrays (7, ns) per rep, N
        for rep in ("Q", "W36", "W12", "W6"):
            for N in NS:
                E[(rep, N)] = np.stack([(D[f][f"{rep}_{N}"].reshape(ns, R) != f).mean(1) for f in range(7)])
        idx = rng.integers(0, ns, size=(B, 7, ns))
        def boot(e): return np.take_along_axis(np.broadcast_to(e, (B, 7, ns)), idx, 2).mean((1, 2))
        for N in NS:
            for rep in ("Q", "W36", "W12", "W6"):
                e = E[(rep, N)]; m = e.mean(); bs = boot(e); k = e.mean() * 7 * ns * R
                key = f"{cell}|{rep}|{N}"
                res[key] = dict(obs_err=float(m), boot_lo=float(np.percentile(bs, 2.5)), boot_hi=float(np.percentile(bs, 97.5)),
                                wilson_lo=wilson(k, 7 * ns * R)[0], wilson_hi=wilson(k, 7 * ns * R)[1], n_records=7 * ns * R,
                                per_family={FAMS[f]: float(e[f].mean()) for f in range(7)}, net=net.get(key))
            for rep in ("W36", "W12", "W6"):   # gap Q-W in accuracy points = errW - errQ
                g = E[(rep, N)] - E[("Q", N)]; gb = boot(g)
                nq, nw = net.get(f"{cell}|Q|{N}"), net.get(f"{cell}|{rep}|{N}")
                gk = f"{cell}|gap_{rep}|{N}"
                res[gk] = dict(obs_gap_pts=100 * float(g.mean()), lo=100 * float(np.percentile(gb, 2.5)), hi=100 * float(np.percentile(gb, 97.5)),
                               net_gap_pts=(100 * (nw["err"] - nq["err"]) if nq and nw else None))
                if res[gk]["net_gap_pts"] is not None and abs(res[gk]["net_gap_pts"]) > 1e-9: res[gk]["ratio_obs_over_net"] = res[gk]["obs_gap_pts"] / res[gk]["net_gap_pts"]
    json.dump(dict(meta=dict(R_records_per_state=R, n_states_per_family=ns, bootstrap=B, note="obs_err: likelihood-ratio observer for known prototypes (upper bound for any classifier of known prototypes); CI: percentile bootstrap over test states, stratified by family; wilson over records ignores state clustering"), cells=res),
              open(outdir / "lr_results.json", "w"), indent=1)
    print("wrote", outdir / "lr_results.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("build", "run", "agg", "all"))
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--R", type=int, default=50, help="records per test state and N"); ap.add_argument("--cells", default=",".join(CELLS))
    a = ap.parse_args()
    outdir = HERE / ("smoke_out" if a.smoke else "."); cells = a.cells.split(",")
    if a.smoke: cells = ["clean"]
    R = 3 if a.smoke else a.R; nstate = 10 if a.smoke else 100
    if a.mode in ("build", "all"):
        for c in cells: build(c, a.workers, a.smoke, outdir)
    if a.mode in ("run", "all"):
        tasks = [(c, f, R, a.smoke, nstate, outdir) for c in cells for f in range(7)]; t0 = time.time()
        with Pool(a.workers) as pool:
            for r in pool.imap_unordered(run_task, tasks): print(r, f"{time.time()-t0:.0f}s", flush=True)
    if a.mode in ("agg", "all"): agg(outdir)


if __name__ == "__main__":
    main()
