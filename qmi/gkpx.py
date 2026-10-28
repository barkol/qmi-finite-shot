"""Rozszerzona klasa GKP: stan |mu_L> (suma koherentnych jak w ahmed_states) z szerszą obwiednią delta, sciśnięty gaussowsko
(r, os psi). Odpowiada rodzinie dopasowanej do rekordów Konno+ 2024 (|1_L>, ok. 3 dB ściśnięcia, obwiednia 0,5-0,85); strat nie dodajemy,
bo robi to kanał szumu w generatorze. W 50% próbek klasy gkp zamiast stanu bazowego."""
import numpy as np, qutip as qt
from qsdl.config import CUTOFF
HB = 100

_LF = np.cumsum(np.log(np.maximum(np.arange(HB), 1))) / 2   # log sqrt(n!)


def _ket(mu, delta):
    """|mu_L> jako suma stanow koherentnych |a>, a = c (2 n1 + mu + i n2), c = sqrt(pi/2), z faza D(x)D(p) exp(-i c^2 (2 n1 + mu) n2).
    Faza zawiera mu; dla mu=1 daje to |1_L>: <Zbar> = -0,85, wiernosc 0,97 do qsdl.states.gkp_state.
    Amplitudy <n|a> liczone analitycznie (zgodne z qt.coherent do 1e-13), bez 1600 eksponent macierzy."""
    c = np.sqrt(np.pi / 2); n1, n2 = np.meshgrid(np.arange(-20, 20), np.arange(-20, 20), indexing="ij")
    a = (c * (2 * n1 + mu + 1j * n2)).ravel(); w = np.exp(-delta ** 2 * np.abs(a) ** 2); keep = w >= 1e-12
    ph = np.exp(-1j * c ** 2 * (2 * n1 + mu) * n2).ravel()
    a, w = a[keep], (w * ph)[keep]; n = np.arange(HB)
    la = np.log(np.where(a == 0, 1, np.abs(a)))
    amp = np.exp(-np.abs(a)[:, None] ** 2 / 2 + n[None, :] * la[:, None] - _LF[None, :] + 1j * n[None, :] * np.angle(a)[:, None])
    amp[a == 0] = (n == 0)
    v = (w[:, None] * amp).sum(0); return qt.Qobj(v[:, None] / np.linalg.norm(v))

def sample_gkpx(rng):
    mu = int(rng.integers(2)); delta = float(rng.uniform(0.4, 0.9)); r = float(rng.uniform(0.1, 0.6)); psi = float(rng.uniform(0, np.pi))
    ket = _ket(mu, delta)
    ket = qt.squeeze(HB, r * np.exp(2j * psi)) * ket                # os scisniecia obrocona o psi
    v = ket.full()[:CUTOFF]; v = v / np.linalg.norm(v)
    return qt.ket2dm(qt.Qobj(v)), {"delta": delta, "mu": mu, "r": r, "psi": psi, "gkpx": 1}

def make_gkpx(base):
    def sampler(label, N, rng):
        if label == "gkp" and rng.random() < 0.5: return sample_gkpx(rng)
        return base(label, N, rng)
    return sampler
