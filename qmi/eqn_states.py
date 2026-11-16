"""Base set with equal mean photon number: all families draw <n> from the same grid NBAR and the parameter of the family
follows from <n>, so the distribution of <n> is the same in every family and the energy alone carries no information on the
family. The vacuum (<n> = 0) cannot be matched; its place (index 2) is taken by the squeezed vacuum, sinh^2 r = <n>.

Families and parameters as in qsdl.states (fock, coherent, thermal, cat, gkp, binomial), except:
  fock      n = <n>
  coherent  |alpha| = sqrt(<n>), phase U(0, 2pi)
  sqvac     r ~ asinh(sqrt(<n>)), axis U(0, pi)   (r and nbar chosen so that the state truncated to CUTOFF has exactly <n>)
  thermal   nbar ~ <n>
  cat       |alpha| from <n> = |a|^2 tanh|a|^2 (even) or |a|^2 coth|a|^2 (odd), phase U(0, 2pi)
  gkp       Delta from <n>(Delta) (monotonic table, separately for mu = 0, 1)
  binomial  (S, K, mu) drawn among the triples with (S+1)(K+1)/2 = <n>, S, K >= 1, without (K=1, mu=1) (the Fock state |<n>>)
The loss channel scales <n> by the same factor eta in all families and dephasing does not change it, so the equality is kept.
"""
from functools import lru_cache

import numpy as np
import qutip as qt
from scipy.optimize import brentq

from qsdl.config import CUTOFF
from qsdl.states import fock_state, coherent_state, thermal_state, cat_state, binomial_state, gkp_state

EQN_LABELS = ("fock", "coherent", "sqvac", "thermal", "cat", "gkp", "binomial")
NBAR = (2, 3, 4, 5)
TOL = 1e-6                      # allowed deviation of <n> of the state from the target
_N = np.arange(CUTOFF)


def nbar(rho):
    return float(np.real(np.diag(rho.full() if hasattr(rho, "full") else rho) @ _N[:rho.shape[0]]))


def _cat_amp(nb, phi):
    f = (lambda a: a * a * np.tanh(a * a) - nb) if phi == 0.0 else (lambda a: a * a / np.tanh(a * a) - nb)
    return brentq(f, 1e-3, 10.0, xtol=1e-12)


# the code word K=1, mu=1 is |S+1> = |nbar>, i.e. a Fock state of the same energy; excluded.
_BIN = {nb: [(S, K, mu) for S in range(1, 8) for K in range(1, 12) for mu in (0, 1)
             if (S + 1) * (K + 1) == 2 * nb and (K + 1) * (S + 1) < CUTOFF and not (K == 1 and mu == 1)] for nb in NBAR}

_GKP = {}
@lru_cache(maxsize=None)
def _gkp_delta(nb, mu):
    """Delta giving <n> = nb for qsdl.states.gkp_state (smax = 4); table on a grid, then brentq on the bracketing interval."""
    if mu not in _GKP:
        d = np.linspace(0.15, 0.7, 56); _GKP[mu] = (d, np.array([nbar(gkp_state(CUTOFF, x, mu)) for x in d]))
    d, n = _GKP[mu]; i = np.where((n[:-1] - nb) * (n[1:] - nb) <= 0)[0]
    if len(i) == 0: raise ValueError(f"gkp: <n>={nb} outside the table (mu={mu})")
    return brentq(lambda x: nbar(gkp_state(CUTOFF, x, mu)) - nb, d[i[0]], d[i[0] + 1], xtol=1e-9)


@lru_cache(maxsize=None)
def _trunc_param(kind, nb):
    """Parameter for which the state TRUNCATED to CUTOFF has exactly <n> = nb (otherwise sqvac at <n> = 5 has 4.989 and thermal
    4.9995, and such constant offsets would separate the families in the energy control on exact images)."""
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
    if abs(got - nb) > TOL: raise RuntimeError(f"{label}: <n>={got:.4f}, target {nb} (TOL {TOL})")
    return rho, {**st, "nbar_target": nb, "nbar": got}


def in_holdout_eqn(label, st):
    """Hold-out: the largest value of the <n> grid, in all classes at once (the same for every class)."""
    return st["nbar_target"] == NBAR[-1]


def make_sampler_eqn(mode):
    """mode: none | train (without the hold-out) | test (hold-out only).
    <n> is not drawn but cycles over the allowed part of the grid from the start of every file (the sampler is
    created anew in every task of qmi.gen), so every class has exactly the same counts of <n> in every file."""
    grid = {"none": NBAR, "train": NBAR[:-1], "test": NBAR[-1:]}[mode]; i = [0]
    def sampler(label, N, rng):
        nb = int(grid[i[0] % len(grid)]); i[0] += 1
        return sample_eqn(label, N, rng, nb)
    return sampler
