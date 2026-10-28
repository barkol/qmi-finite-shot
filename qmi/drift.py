"""Dryf fazy w skonczonych rekordach (modele wspolne dla W i Q).
walk:   faza LO = faza nominalna + delta_k, delta_k = bladzenie losowe w czasie zapisu o odchyleniu koncowym sigma (stopnie).
        Homodyna: rzut k mierzony pod katem theta_k + delta_k, rekonstrukcja zaklada theta_k.
        Heterodyna: N/K zdarzen z segmentu k pochodzi z Q obroconego o delta_k (chmura punktow w ukladzie LO).
jitter: szybki szum fazy, delta_i ~ N(0, sigma^2) na zdarzenie = rozmycie azymutalne dokladnego obrazu (usrednienie po obrotach).
Niezalezne od konwencji jednostek: obrot obrazu wokol srodka siatki."""
import numpy as np
from scipy.ndimage import rotate
from qmi.shots import radon, fbp, heterodyne_image, homodyne_wigner, split_events
from qsdl.config import XMAX as _XMAX   # domyslne okno z QMI_XMAX

GH_N = 15
_x, _w = np.polynomial.hermite_e.hermegauss(GH_N)   # waga exp(-x^2/2)
_w = _w / _w.sum()


def path(K, sigma, rng):
    """delta_k (stopnie), k=1..K; bladzenie losowe, std(delta_K)=sigma."""
    return sigma * np.cumsum(rng.standard_normal(K)) / np.sqrt(K)


def blur(img, sigma):
    if sigma == 0: return np.asarray(img, np.float64)
    return sum(w * rotate(np.asarray(img, np.float64), sigma * x, reshape=False, order=1, mode="constant") for x, w in zip(_x, _w))


def hetero_walk(Q, N, rng, K=36, sigma=0.0, xmax=None):
    if xmax is None: xmax = _XMAX
    if sigma == 0: return heterodyne_image(Q, N, rng, xmax)
    d = path(K, sigma, rng); per = split_events(N, K)
    Q64 = np.clip(np.asarray(Q, np.float64), 0, None); mass0 = Q64.sum()
    cnt = np.zeros(Q64.size)
    for dk, nk in zip(d, per):
        if nk == 0: continue
        p = np.clip(rotate(Q64, dk, reshape=False, order=1, mode="constant"), 0, None).ravel(); s = p.sum()
        cnt += rng.multinomial(nk, p / s)
    dx = 2 * xmax / (Q.shape[0] - 1); mass = float(mass0 * dx * dx)
    return (cnt.reshape(Q.shape) / (per.sum() * dx * dx) * mass).astype(np.float32)


def homo_walk(W, N, rng, K=36, sigma=0.0):
    if sigma == 0: return homodyne_wigner(W, N, rng, K)
    th = np.linspace(0, 180, K, endpoint=False); d = path(K, sigma, rng)
    W64 = np.asarray(W, np.float64); out = np.zeros((K, W.shape[0])); per = split_events(N, K)
    for k in range(K):
        row = radon(W64, [th[k] + d[k]])[0]; p = np.clip(row, 0, None); s = p.sum()
        if per[k] == 0 or s <= 0: continue
        out[k] = rng.multinomial(per[k], p / s) / per[k] * row.sum()
    return fbp(out, th, W.shape[0])


def make(W, Q, N, rng, mode, sigma, K=36):
    if mode == "walk": return homo_walk(W, N, rng, K, sigma), hetero_walk(Q, N, rng, K, sigma)
    if mode == "jitter": return homodyne_wigner(blur(W, sigma), N, rng, K), heterodyne_image(blur(Q, sigma), N, rng)
    raise ValueError(mode)
