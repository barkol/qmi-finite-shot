"""Wykladniki Chernoffa dla heterodyny (Q), homodyny (K katow) i granica kwantowa (QCB).

Dla stanu testowego rho z klasy i i stanu sigma z innej klasy liczymy klasyczna informacje Chernoffa rozkladow wynikow
JEDNEGO zdarzenia, na tej samej siatce pikseli co dane (128 x 128, okno +-12, x = (a + a^+)/sqrt2, proznia 1/2):
  heterodyna: p(x,p) ~ <alpha|rho|alpha>,  C_Q = -min_s log sum p^s q^(1-s)
  homodyna:   K = 36 katow w [0, pi), N/K zdarzen na kat (jak qmi.shots.split_events),
              C_W = -min_s (1/K) sum_k log sum_x p_k^s q_k^(1-s)
  kwantowa:   C_QCB = -min_s log Tr rho^s sigma^(1-s)   (granica dla dowolnego pomiaru na pojedynczych kopiach... i wiecej)
xi_X(rho) = min po sigma z innych klas C_X(rho, sigma); P_e(N) ~ exp(-N xi). Stany: te same samplery i kanaly co test sweep
(qmi.eqn_states / qmi.ahmed_states, holdout "none"; qmi.gen.level_params_eqn + qsdl.noise.apply_channel_noise).
Wznawialne: jeden plik npz na (zbior, komorke) w out/.   python analyses/chernoff/chernoff.py [--test] [--M 40]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
from scipy.special import gammaln, eval_hermite
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import qmi  # noqa: F401  sets qsdl.config.XMAX from QMI_XMAX (pinned qsdl has XMAX=5.0); must precede qsdl imports
from qsdl.config import CUTOFF
from qsdl.noise import apply_channel_noise
from qmi.gen import level_params_eqn

G, XM, K = 128, 12.0, 36
OUT = Path(__file__).resolve().parent / "out"
CELLS = [("clean", 0), ("loss", 50), ("loss", 80), ("dephase", 20)]
S = np.linspace(0.05, 0.95, 19)
x = np.linspace(-XM, XM, G); dx = x[1] - x[0]
nn = np.arange(CUTOFF)

def hermite_funcs():
    """psi_n(x) dla n < CUTOFF stabilnie rekurencyjnie."""
    P = np.zeros((G, CUTOFF)); P[:, 0] = np.pi ** -0.25 * np.exp(-x * x / 2)
    if CUTOFF > 1: P[:, 1] = np.sqrt(2) * x * P[:, 0]
    for n in range(2, CUTOFF): P[:, n] = np.sqrt(2 / n) * x * P[:, n - 1] - np.sqrt((n - 1) / n) * P[:, n - 2]
    return P
PSI = hermite_funcs()
TH = np.linspace(0, np.pi, K, endpoint=False)
XX, PP = np.meshgrid(x, x, indexing="ij"); AL = ((XX + 1j * PP) / np.sqrt(2)).ravel()
_r = np.abs(AL); _lr = np.log(np.where(_r > 0, _r, 1e-300))
V = np.exp(-_r[:, None] ** 2 / 2 + nn[None, :] * _lr[:, None] - 0.5 * gammaln(nn + 1)[None, :]) * np.exp(1j * nn[None, :] * np.angle(AL)[:, None])
V[_r == 0, 1:] = 0

def dm(rho): return rho.full() if hasattr(rho, "full") else np.asarray(rho)

def q_probs(r):
    q = np.real(np.einsum("am,mn,an->a", V.conj(), r, V, optimize=True)); q = np.clip(q, 0, None); return q / q.sum()

def hom_probs(r):
    out = np.empty((K, G))
    for k, th in enumerate(TH):
        d = np.exp(-1j * nn * th); rt = d[:, None] * r * d.conj()[None, :]
        p = np.real(np.einsum("xm,mn,xn->x", PSI, rt, PSI, optimize=True)); p = np.clip(p, 0, None); out[k] = p / p.sum()
    return out

def chern_rows(Lp, Lq, axis_avg=None):
    """Lp: (d,) log p; Lq: (m, d) log q -> (m,) C = -min_s log sum exp(s Lp + (1-s) Lq). axis_avg: K dla homodyny (d = K*G)."""
    best = np.full(Lq.shape[0], np.inf)
    for s in S:
        z = s * Lp[None, :] + (1 - s) * Lq
        if axis_avg:
            z = z.reshape(Lq.shape[0], axis_avg, -1); mx = z.max(axis=2, keepdims=True)
            v = (np.log(np.exp(z - mx).sum(axis=2)) + mx[..., 0]).mean(axis=1)
        else:
            mx = z.max(axis=1, keepdims=True); v = np.log(np.exp(z - mx).sum(axis=1)) + mx[:, 0]
        best = np.minimum(best, v)
    return -best

def qcb_matrix(rhos):
    ev = [np.linalg.eigh(r) for r in rhos]; n = len(rhos); C = np.zeros((n, n))
    for i in range(n):
        li, ui = ev[i]; li = np.clip(li, 0, None)
        for j in range(i + 1, n):
            lj, uj = ev[j]; lj = np.clip(lj, 0, None); O = np.abs(ui.conj().T @ uj) ** 2
            with np.errstate(divide="ignore"):
                vals = [np.sum((li[:, None] ** s) * (lj[None, :] ** (1 - s)) * O) for s in S]
            C[i, j] = C[j, i] = -np.log(max(min(vals), 1e-300))
    return C

def samples(fam, M, seed):
    if fam == "eqn":
        from qmi.eqn_states import EQN_LABELS as L, make_sampler_eqn as mk
    else:
        from qmi.ahmed_states import AHM_LABELS as L, make_sampler_ahm as mk
    rhos, lab, meta = [], [], []
    for c, name in enumerate(L):
        smp = mk("none"); rng = np.random.default_rng([seed, c])
        for _ in range(M):
            r, st = smp(name, CUTOFF, rng); rhos.append(r); lab.append(c); meta.append(st)
    return rhos, np.array(lab), list(L), meta

def run_cell(fam, cell, rhos0, lab, names, tag):
    f = OUT / f"{tag}_{fam}_{cell[0]}{cell[1]}.npz"
    if f.exists(): print("jest", f.name); return
    t0 = time.time()
    par = level_params_eqn(*cell) if cell[0] != "clean" else {}
    rhos = [dm(apply_channel_noise(r, par)) if par else dm(r) for r in rhos0]
    Qp = np.array([q_probs(r) for r in rhos]); Hp = np.array([hom_probs(r).ravel() for r in rhos])
    LQ = np.log(np.maximum(Qp, 1e-300)); LH = np.log(np.maximum(Hp, 1e-300)); n = len(rhos)
    CQ = np.zeros((n, n)); CW = np.zeros((n, n))
    for i in range(n):
        j = np.arange(i + 1, n)
        if len(j): CQ[i, j] = CQ[j, i] = chern_rows(LQ[i], LQ[j]); CW[i, j] = CW[j, i] = chern_rows(LH[i], LH[j], K)
    CQCB = qcb_matrix(rhos)
    np.savez_compressed(f, CQ=CQ, CW=CW, CQCB=CQCB, lab=lab, names=np.array(names), cell=np.array(cell, dtype=object), K=K)
    print(f"{f.name}: {time.time() - t0:.0f} s", flush=True)

def selftest():
    """Dwa stany koherentne: C_Q = C_W = |da|^2/4, QCB = |da|^2 (rachunek analityczny dla gaussow o rownej kowariancji)."""
    from qsdl.states import coherent_state
    a, b = 1.0 + 0.5j, -0.3 + 1.2j; ra, rb = dm(coherent_state(CUTOFF, a)), dm(coherent_state(CUTOFF, b))
    LQa, LQb = np.log(np.maximum(q_probs(ra), 1e-300)), np.log(np.maximum(q_probs(rb), 1e-300))
    LHa, LHb = np.log(np.maximum(hom_probs(ra).ravel(), 1e-300)), np.log(np.maximum(hom_probs(rb).ravel(), 1e-300))
    d2 = abs(a - b) ** 2
    print("C_Q", chern_rows(LQa, LQb[None])[0], "C_W", chern_rows(LHa, LHb[None], K)[0], "oczek.", d2 / 4,
          "QCB", qcb_matrix([ra, rb])[0, 1], "oczek.", d2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--M", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2210); ap.add_argument("--fams", default="eqn,ahm"); a = ap.parse_args()
    if a.test: selftest(); sys.exit()
    OUT.mkdir(exist_ok=True)
    for fam in a.fams.split(","):
        rhos, lab, names, _ = samples(fam, a.M, a.seed)
        for cell in CELLS: run_cell(fam, cell, rhos, lab, names, f"M{a.M}")
    print("CHERNOFF_DONE")
