"""Krok 4: klasyfikacja rekordow jednego modu EPR (hd1) sieciami W projektu, protokol jak qmi.x8.konno.klasyf (losowe podzbiory,
kolejne bloki, rekordy symulowane), uogolniony na K katow.
Rekord ma tylko dwie fazy (x -> 0 st., p -> 90 st.). Przypisanie zdarzen do K katow potoku (th_k = 180 k / K):
  A ('xp+pula'): kat 0 <- zdarzenia x, kat 90 <- zdarzenia p, pozostale katy <- losowo z puli pozostalych zdarzen x i p
                 (bez powtorzen w rekordzie). Zaklada niezmienniczosc fazowa stanu poza 0 i 90 st.
  B ('pula')   : wszystkie katy z puli x u p (pelna symetryzacja fazy).
Zrodla: exp_los (losowe podzbiory rekordu), exp_blok (kolejne bloki czasu: blok x i blok p, w bloku to samo przypisanie),
  sym_gauss (to samo przypisanie, zdarzenia z N(0,Vx) i N(0,Vp) o zmierzonych wariancjach: model szumu zliczen + gaussowskosc + iid),
  sym_term (stan termiczny o zmierzonym n = (Vx+Vp)/2 - 1/2: wszystkie katy N(0, n+1/2)).
Obraz: histogram na pikselach siatki potoku -> skala masy rzutu M (proznia) -> FBP (qmi.shots.fbp), jak konno.img_counts.
Wznowienie: klasyf/<grupa>_<wariant>/<zestaw>.json pomijany, gdy istnieje. Inferencja na CPU (CUDA_VISIBLE_DEVICES='').
  CUDA_VISIBLE_DEVICES= QMI_XMAX=12 python -I krok4_klasyf.py --K 6 --wariant A"""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
if os.environ.get("QMI_XMAX") != "12": sys.exit("BLAD: QMI_XMAX musi byc 12 (siatka eq)")
if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "": sys.exit("BLAD: ustaw CUDA_VISIBLE_DEVICES= (GPU zajete)")
Q = os.environ.get("QMI_ROOT", str(Path(__file__).resolve().parents[2])); sys.path.insert(0, Q)
import qmi  # noqa: F401  sets qsdl.config.XMAX from QMI_XMAX (pinned qsdl has XMAX=5.0); must precede qsdl imports
from qsdl.config import GRID, XMAX, CUTOFF
from qmi.shots import radon, fbp, split_events

OUT = Path(__file__).resolve().parent; R = OUT / "rekordy"; LOG = OUT / "krok4.log"
NS = (100, 1000, 10000)
DX = 2 * XMAX / (GRID - 1); EDGES = np.linspace(-XMAX - DX / 2, XMAX + DX / 2, GRID + 1)
GRUPY = {6: {"baza_K6": (f"{Q}/runs_eq", r"re:^clf_W_shotsK6_s\d+$", 3, "eqn"),
             "ahm_K6": (f"{Q}/analyses/kawasaki/runs_ahmK6", r"re:^clf_W_shotsK6_s\d+$", 3, "ahm")},
         36: {"baza_K36": (f"{Q}/runs_eq", r"re:^clf_W_shots_s\d+$", 5, "eqn"),
              "ahm_K36": (f"{Q}/runs_ahm_eq", r"re:^clf_W_shots_s\d+$", 3, "ahm")}}
LABELS = {"eqn": ("fock", "coherent", "sqvac", "thermal", "cat", "gkp", "binomial"),
          "ahm": ("fock", "coherent", "thermal", "num", "gkp", "binomial", "cat")}
G = {}


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); open(LOG, "a").write(s + "\n")


def M_mass():
    import qutip as qt
    y = np.linspace(-XMAX, XMAX, GRID); return float(radon(qt.wigner(qt.fock_dm(CUTOFF, 0), y, y), [0])[0].sum())


def przydziel(x, p, per, th, war, rng):
    """Zdarzenia (juz w losowej kolejnosci lub blok) -> lista tablic na katy. x, p: zrodla; per: liczby na kat."""
    out = [None] * len(per)
    if war == "A":
        i0 = int(np.where(th == 0)[0][0]); i90 = int(np.where(th == 90)[0][0])
        out[i0] = x[:per[i0]]; out[i90] = p[:per[i90]]
        pool = rng.permutation(np.concatenate([x[per[i0]:], p[per[i90]:]])); k = 0
        for i in range(len(per)):
            if out[i] is None: out[i] = pool[k:k + per[i]]; k += per[i]
        if k > len(pool): raise RuntimeError("za malo zdarzen w puli")
    else:
        pool = rng.permutation(np.concatenate([x, p])); c = np.concatenate([[0], np.cumsum(per)])
        if c[-1] > len(pool): raise RuntimeError("za malo zdarzen")
        out = [pool[c[i]:c[i + 1]] for i in range(len(per))]
    return out


def potrzeba(per, th, war):
    """Ile zdarzen x i p zuzywa jeden rekord (do blokow)."""
    n = int(per.sum())
    if war == "A":
        i0 = int(np.where(th == 0)[0][0]); i90 = int(np.where(th == 90)[0][0]); rest = n - per[i0] - per[i90]
        return int(per[i0] + (rest + 1) // 2), int(per[i90] + (rest + 1) // 2)
    return (n + 1) // 2, (n + 1) // 2


def obraz(ev, per):
    C = [np.histogram(e, EDGES)[0] for e in ev]
    return fbp(np.stack([c / max(1, q) * G["M"] for c, q in zip(C, per)]), G["th"], GRID)


def zadanie(args):
    """Jeden obraz: (src, N, j) -> obraz float32. Generator z SeedSequence (zestaw, src, N, j) - powtarzalnie."""
    src, N, j = args; x, p = G["x"], G["p"]; th = G["th"]; war = G["war"]; per = split_events(N, len(th))
    rng = np.random.default_rng(np.random.SeedSequence([G["seed"], ["exp_los", "exp_blok", "sym_gauss", "sym_term"].index(src), N, j]))
    if src == "exp_los": ev = przydziel(rng.permutation(x), rng.permutation(p), per, th, war, rng)
    elif src == "exp_blok":
        nx, np_ = potrzeba(per, th, war); ev = przydziel(x[j * nx:(j + 1) * nx], p[j * np_:(j + 1) * np_], per, th, war, rng)
    elif src == "sym_gauss":
        nx, np_ = potrzeba(per, th, war) if war == "A" else (int(per.sum()), int(per.sum()))
        ev = przydziel(rng.normal(0, np.sqrt(G["vx"]), nx + 2), rng.normal(0, np.sqrt(G["vp"]), np_ + 2), per, th, war, rng)
    else: ev = [rng.normal(0, np.sqrt(G["nbar"] + 0.5), q) for q in per]
    return src, N, j, obraz(ev, per).astype(np.float32)


def init(g): G.update(g)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, required=True); ap.add_argument("--wariant", choices=("A", "B"), required=True)
    ap.add_argument("--draws", type=int, default=200); ap.add_argument("--zestawy", default=None); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grupy", default=None); a = ap.parse_args()
    import torch; torch.set_num_threads(2)
    from qmi.eval_shots import load_models, pred
    th = np.linspace(0, 180, a.K, endpoint=False)
    if a.wariant == "A" and not (0 in th and 90 in th): raise RuntimeError("wariant A wymaga katow 0 i 90")
    grupy = {k: v for k, v in GRUPY[a.K].items() if a.grupy is None or k in a.grupy.split(",")}
    models = {}
    for g, (runs, rx, nexp, lab) in grupy.items():
        m = load_models(runs, rx)
        if len(m) != nexp: raise RuntimeError(f"{g}: {len(m)} modeli zamiast {nexp}: {sorted(m)}")
        for nm, v in m.items():
            if v[1] != "W": raise RuntimeError(f"{g}/{nm}: rep {v[1]}")
        models[g] = m
    log(f"start {time.ctime()} K={a.K} wariant={a.wariant} draws={a.draws} modele " + "; ".join(f"{g}: {sorted(m)}" for g, m in models.items()))
    M = M_mass(); files = sorted(R.glob("*.npz"))
    if a.zestawy: files = [f for f in files if any(z in f.stem for z in a.zestawy.split(","))]
    if not files: raise RuntimeError("brak rekordow")
    for f in files:
        outs = {g: OUT / "klasyf" / f"{g}_{a.wariant}" / f"{f.stem}.json" for g in models}
        if all(o.exists() for o in outs.values()): log(f"pomijam (gotowe) {f.stem}"); continue
        t0 = time.time(); d = np.load(f); w = json.loads(str(d["wyniki"]))["1"]
        x = d["x1"].astype(np.float64); p = d["p1"].astype(np.float64)
        g0 = dict(x=x, p=p, th=th, war=a.wariant, M=M, vx=float(x.var()), vp=float(p.var()), nbar=float((x.var() + p.var()) / 2 - 0.5),
                  seed=int.from_bytes(f.stem.encode()[-8:], "little") % (2 ** 31))
        tasks = []
        for N in NS:
            per = split_events(N, a.K); nx, np_ = potrzeba(per, th, a.wariant)
            nb = min(a.draws, len(x) // nx, len(p) // np_)
            tasks += [(s, N, j) for s in ("exp_los", "sym_gauss", "sym_term") for j in range(a.draws)] + [("exp_blok", N, j) for j in range(nb)]
        imgs = {}
        with Pool(a.workers, initializer=init, initargs=(g0,)) as pool:
            for src, N, j, im in pool.imap_unordered(zadanie, tasks, chunksize=8): imgs.setdefault((src, N), []).append((j, im))
        for g, mm in models.items():
            lab = LABELS[grupy[g][3]]; res = {}
            for (src, N), L in imgs.items():
                X = np.stack([im for _, im in sorted(L, key=lambda t: t[0])])
                for nm, (m, rep, sc) in mm.items(): res[f"{nm}|{src}|N{N}"] = np.bincount(pred(m, X, sc), minlength=7).tolist()
            outs[g].parent.mkdir(parents=True, exist_ok=True)
            json.dump(dict(labels=lab, K=a.K, wariant=a.wariant, nbar=g0["nbar"], vx=g0["vx"], vp=g0["vp"], counts=res), open(str(outs[g]) + ".tmp", "w"))
            os.replace(str(outs[g]) + ".tmp", outs[g])
            ti = lab.index("thermal")
            log(f"  {f.stem} {g}: n={g0['nbar']:.2f} frakcja 'thermal' " + "; ".join(
                f"N={N}: " + " ".join(f"{s[:8]} {np.mean([res[f'{nm}|{s}|N{N}'][ti] / max(1, sum(res[f'{nm}|{s}|N{N}'])) for nm in mm]):.2f}" for s in ("exp_los", "exp_blok", "sym_gauss", "sym_term")) for N in NS))
        log(f"{f.stem} gotowe {time.time()-t0:.0f}s")
    log(f"KROK4_DONE K={a.K} wariant={a.wariant} {time.ctime()}")
