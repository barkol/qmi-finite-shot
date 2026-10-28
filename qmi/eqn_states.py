"""Zestaw bazowy z wyrownana srednia liczba fotonow: wszystkie klasy losuja <n> z tej samej siatki
NBAR, a parametr rodziny wynika z <n>, wiec rozkład <n> jest identyczny w kazdej klasie i sama energia nie niesie informacji
o klasie. Prozni (<n> = 0) nie da sie wyrownac; jej miejsce (indeks 2) zajmuje sciśnieta proznia, sinh^2 r = <n>.

Rodziny i parametry jak w qsdl.states (fock, coherent, thermal, cat, gkp, binomial), poza tym, ze:
  fock      n = <n>
  coherent  |alpha| = sqrt(<n>), faza U(0, 2pi)
  sqvac     r ~ asinh(sqrt(<n>)), os U(0, pi)   (r i nbar dobrane tak, by stan po obcieciu do CUTOFF mial dokladnie <n>)
  thermal   nbar ~ <n>
  cat       |alpha| z <n> = |a|^2 tanh|a|^2 (parzysty) lub |a|^2 coth|a|^2 (nieparzysty), faza U(0, 2pi)
  gkp       Delta z <n>(Delta) (tablica monotoniczna, osobno dla mu = 0, 1)
  binomial  (S, K, mu) losowane sposrod trojek z (S+1)(K+1)/2 = <n>, S, K >= 1, bez (K=1, mu=1) (to stan Focka |<n>>)
  gkpx      (qmi.gkpx, 50% klasy gkp przy --gkpx) r, psi losowane jak tam, delta dobierane do <n>
Kanal strat skaluje <n> tym samym czynnikiem eta we wszystkich klasach, dephasing go nie zmienia, wiec rownosc zostaje.
"""
from functools import lru_cache

import numpy as np
import qutip as qt
from scipy.optimize import brentq

from qsdl.config import CUTOFF
from qsdl.states import fock_state, coherent_state, thermal_state, cat_state, binomial_state, gkp_state

EQN_LABELS = ("fock", "coherent", "sqvac", "thermal", "cat", "gkp", "binomial")
NBAR = (2, 3, 4, 5)
TOL = 1e-6                      # dopuszczalna odchylka <n> stanu od celu
_N = np.arange(CUTOFF)


def nbar(rho):
    return float(np.real(np.diag(rho.full() if hasattr(rho, "full") else rho) @ _N[:rho.shape[0]]))


def _cat_amp(nb, phi):
    f = (lambda a: a * a * np.tanh(a * a) - nb) if phi == 0.0 else (lambda a: a * a / np.tanh(a * a) - nb)
    return brentq(f, 1e-3, 10.0, xtol=1e-12)


# slowo kodowe K=1, mu=1 to |S+1> = |nbar>, czyli stan Focka tej samej energii; wylaczone.
_BIN = {nb: [(S, K, mu) for S in range(1, 8) for K in range(1, 12) for mu in (0, 1)
             if (S + 1) * (K + 1) == 2 * nb and (K + 1) * (S + 1) < CUTOFF and not (K == 1 and mu == 1)] for nb in NBAR}

_GKP = {}
@lru_cache(maxsize=None)
def _gkp_delta(nb, mu):
    """Delta dajace <n> = nb dla qsdl.states.gkp_state (smax = 4); tablica na siatce, potem brentq na przedziale."""
    if mu not in _GKP:
        d = np.linspace(0.15, 0.7, 56); _GKP[mu] = (d, np.array([nbar(gkp_state(CUTOFF, x, mu)) for x in d]))
    d, n = _GKP[mu]; i = np.where((n[:-1] - nb) * (n[1:] - nb) <= 0)[0]
    if len(i) == 0: raise ValueError(f"gkp: <n>={nb} poza zakresem tablicy (mu={mu})")
    return brentq(lambda x: nbar(gkp_state(CUTOFF, x, mu)) - nb, d[i[0]], d[i[0] + 1], xtol=1e-9)


@lru_cache(maxsize=None)
def _trunc_param(kind, nb):
    """Parametr, przy ktorym stan OBCIETY do CUTOFF ma dokladnie <n> = nb (bez tego sqvac przy <n> = 5 ma 4,989, thermal 4,9995,
    a takie stale przesuniecia odroznialy klasy w kontroli energetycznej na dokladnych obrazach)."""
    if kind == "sqvac":
        return brentq(lambda r: nbar(sqvac_state(CUTOFF, r, 0.0)) - nb, 0.5 * np.arcsinh(np.sqrt(nb)), 1.5 * np.arcsinh(np.sqrt(nb)), xtol=1e-12)
    return brentq(lambda x: nbar(thermal_state(CUTOFF, x)) - nb, 0.5 * nb, 2.0 * nb, xtol=1e-12)


def sqvac_state(N, r, psi):
    return qt.ket2dm(qt.squeeze(N, r * np.exp(2j * psi)) * qt.basis(N, 0))


def sample_eqn(label, N, rng, nb=None):
    nb = int(NBAR[int(rng.integers(len(NBAR)))]) if nb is None else nb
    if label == "fock":
        rho, st = fock_state(CUTOFF, nb), {"n": nb}
    elif label == "coherent":
        ph = float(rng.uniform(0, 2 * np.pi)); a = np.sqrt(nb) * np.exp(1j * ph)
        rho, st = coherent_state(CUTOFF, a), {"amp": float(np.sqrt(nb)), "phase": ph}
    elif label == "sqvac":
        r = _trunc_param("sqvac", nb); psi = float(rng.uniform(0, np.pi))
        rho, st = sqvac_state(CUTOFF, r, psi), {"r": r, "psi": psi}
    elif label == "thermal":
        nt = _trunc_param("thermal", nb); rho, st = thermal_state(CUTOFF, nt), {"nbar_param": nt}
    elif label == "cat":
        phi = float(rng.choice([0.0, np.pi])); amp = _cat_amp(nb, phi); ph = float(rng.uniform(0, 2 * np.pi))
        rho, st = cat_state(CUTOFF, amp * np.exp(1j * ph), phi), {"amp": amp, "phase": ph, "phi": phi}
    elif label == "gkp":
        mu = int(rng.integers(0, 2)); delta = _gkp_delta(nb, mu)
        rho, st = gkp_state(CUTOFF, delta, mu), {"delta": delta, "mu": mu, "smax": 4}
    elif label == "binomial":
        S, K, mu = _BIN[nb][int(rng.integers(len(_BIN[nb])))]
        rho, st = binomial_state(CUTOFF, S, K, mu), {"S": S, "K": K, "mu": mu}
    else:
        raise ValueError(label)
    got = nbar(rho)
    if abs(got - nb) > TOL: raise RuntimeError(f"{label}: <n>={got:.4f}, cel {nb} (TOL {TOL})")
    return rho, {**st, "nbar_target": nb, "nbar": got}


def sample_gkpx_eqn(rng, nb):
    """Sciśniety GKP (qmi.gkpx) z delta dobranym do <n> przy wylosowanych r, psi; r losowane ponownie, gdy <n> nieosiagalne."""
    from qmi.gkpx import _ket, HB
    def state(mu, delta, r, psi):
        v = (qt.squeeze(HB, r * np.exp(2j * psi)) * _ket(mu, delta)).full()[:CUTOFF]; v = v / np.linalg.norm(v)
        return qt.ket2dm(qt.Qobj(v))
    for _ in range(50):
        mu = int(rng.integers(2)); r = float(rng.uniform(0.1, 0.6)); psi = float(rng.uniform(0, np.pi))
        f = lambda d: nbar(state(mu, d, r, psi)) - nb
        lo, hi = 0.2, 0.95
        if f(lo) * f(hi) > 0: continue
        delta = brentq(f, lo, hi, xtol=1e-12); rho = state(mu, delta, r, psi); got = nbar(rho)
        if abs(got - nb) <= TOL:
            return rho, {"delta": delta, "mu": mu, "r": r, "psi": psi, "gkpx": 1, "nbar_target": nb, "nbar": got}
    raise RuntimeError(f"gkpx: <n>={nb} nieosiagalne w 50 probach")


def in_holdout_eqn(label, st):
    """Hold-out: najwyzsza wartosc siatki <n> we wszystkich klasach naraz (tak samo dla kazdej klasy)."""
    return st["nbar_target"] == NBAR[-1]


def make_sampler_eqn(mode, gkpx=False):
    """mode: none | train (bez hold-outu) | test (tylko hold-out). gkpx: 50% klasy gkp to sciśniety GKP o tym samym <n>.
    <n> nie jest losowane, tylko przechodzi cyklicznie po dozwolonej czesci siatki od poczatku kazdego pliku (sampler powstaje
    na nowo w kazdym zadaniu qmi.gen), wiec kazda klasa ma w kazdym pliku dokladnie te same liczebnosci <n>."""
    grid = {"none": NBAR, "train": NBAR[:-1], "test": NBAR[-1:]}[mode]; i = [0]
    def sampler(label, N, rng):
        nb = int(grid[i[0] % len(grid)]); i[0] += 1
        if gkpx and label == "gkp" and rng.random() < 0.5: return sample_gkpx_eqn(rng, nb)
        return sample_eqn(label, N, rng, nb)
    return sampler
