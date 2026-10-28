"""Level 2b (28.10.2026): lepszy wariant FBP (interpolacja rzedu 2 + filtr Kaka-Slaneya) jako jawne parametry (nie zmienne srodowiskowe),
oraz rekordy homodynowe, w ktorych strumien rng (N, Q, rng glowny) zuzywa sie IDENTYCZNIE jak w pipeline artykulu (rzut rzedu 1),
wiec rekordy Q i ciag N sa bitowo takie same jak w pracy. Zliczenia nowego W losuja sie z osobnego strumienia rng2."""
import numpy as np
from scipy.ndimage import rotate
from qmi.shots import split_events

def radon_o(img, th, order):
    return np.stack([rotate(img, t, reshape=False, order=order, mode="constant").sum(axis=0) for t in th])

VARIANTS = {"kak2": (2, "kak", None), "hann2": (2, "kak", "hann"), "old": (1, "freq", None)}   # (rzad interpolacji, filtr, okno apodyzacji)

def ramp_f(n, filt, win=None):
    r = _ramp0(n, filt)
    if win == "hann": r = r * 0.5 * (1 + np.cos(2 * np.pi * np.abs(np.fft.fftfreq(n))))   # okno Hanna do Nyquista
    return r

def _ramp0(n, filt):
    if filt == "kak":
        m = np.concatenate((np.arange(1, n // 2 + 1, 2), np.arange(n // 2 - 1, 0, -2)))
        h = np.zeros(n); h[0] = 0.25; h[1::2] = -1 / (np.pi * m) ** 2
        return 2 * np.real(np.fft.fft(h))
    return 2 * np.abs(np.fft.fftfreq(n))

def fbp_o(sino, th, out_n, order, filt, win=None):
    n = sino.shape[1]; pad = int(2 ** np.ceil(np.log2(2 * n)))
    P = np.fft.fft(sino, n=pad, axis=1) * ramp_f(pad, filt, win)[None, :]
    f = np.real(np.fft.ifft(P, axis=1))[:, :n]; img = np.zeros((n, n))
    for row, t in zip(f, th):
        img += rotate(np.tile(row, (n, 1)), -t, reshape=False, order=order, mode="constant")
    return (img * np.pi / (2 * len(th))).astype(np.float32)

def exact_o(W, K, order, filt, win=None):
    th = np.linspace(0, 180, K, endpoint=False)
    return fbp_o(radon_o(np.asarray(W, np.float64), th, order), th, W.shape[0], order, filt, win)

def _draw(sino, per, rng):
    out = np.zeros_like(sino)
    for k in range(len(per)):
        p = np.clip(sino[k], 0, None); s = p.sum()
        if per[k] == 0 or s <= 0: continue
        out[k] = rng.multinomial(per[k], p / s) / per[k] * sino[k].sum()
    return out

def homodyne_new(W, N, rng, rng2, K=36, variant="kak2"):
    """rng: glowny strumien rekordu (zuzywany jak w pracy, wynik odrzucony); rng2: strumien zliczen nowego W."""
    order, filt, win = VARIANTS[variant]
    th = np.linspace(0, 180, K, endpoint=False); W = np.asarray(W, np.float64); per = split_events(N, K)
    _draw(radon_o(W, th, 1), per, rng)                       # zuzycie rng jak w homodyne_wigner artykulu
    return fbp_o(_draw(radon_o(W, th, order), per, rng2), th, W.shape[0], order, filt, win)
