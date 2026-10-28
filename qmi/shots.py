"""Skonczona liczba zdarzen pomiarowych (rama 1-3, element 1).

Heterodyna:  Q(beta) = <beta|rho|beta>/pi jest gestoscia prawdopodobienstwa wynikow detekcji heterodynowej
             -> N zdarzen = histogram (multinomial) na siatce obrazu. Bezposrednio, bez inwersji.
Homodyna:    Wigner nie jest mierzony wprost. N zdarzen rozdzielone na K katow; z kazdego kata histogram
             rozkladu brzegowego (projekcja Radona W); obraz z filtrowanej projekcji wstecznej (FBP, filtr ramp).
Obie funkcje pracuja na obrazach juz policzonych (apply_wigner / apply_husimi), wiec sa niezalezne od konwencji
jednostek kwadratur: projekcje to calki liniowe obrazu na siatce pikseli.
"""
import os
import numpy as np
from scipy.ndimage import rotate
from qsdl.config import XMAX as _XMAX   # domyslne okno z QMI_XMAX (5.0 jak dotad)

DX = lambda grid, xmax: 2 * xmax / (grid - 1)


def heterodyne_image(Q, N, rng, xmax=None):
    """Q: dokladny obraz Husimi (grid x grid). Zwraca estymate gestosci z N zdarzen (ta sama skala co Q)."""
    p = np.clip(np.asarray(Q, np.float64), 0, None).ravel()
    p /= p.sum()
    counts = rng.multinomial(int(N), p).reshape(Q.shape)
    dx = DX(Q.shape[0], _XMAX if xmax is None else xmax)
    mass = float(np.clip(Q, 0, None).sum() * dx * dx)   # ~1 (calka Q po siatce) -> zachowuje skale Q
    return (counts / (N * dx * dx) * mass).astype(np.float32)


# wariant FBP przez zmienne srodowiskowe; domyslnie bez zmian (rzad 1, ramp 2|f|).
#   QMI_FBP_ORDER=2  interpolacja bikwadratowa w obrotach (rzuty i projekcja wsteczna)
#   QMI_FBP_FILTER=kak  filtr Ram-Lak z dziedziny przestrzennej (Kak-Slaney), poprawna skladowa stala
_ORDER = int(os.environ.get("QMI_FBP_ORDER", "1"))
_FILTER = os.environ.get("QMI_FBP_FILTER", "freq")
assert _ORDER in (1, 2, 3) and _FILTER in ("freq", "kak"), (_ORDER, _FILTER)


def radon(img, thetas_deg):
    """Projekcje: dla kazdego kata suma po osi 0 obrazu obroconego o kat. Zwraca (n_angles, n)."""
    return np.stack([rotate(img, th, reshape=False, order=_ORDER, mode="constant").sum(axis=0) for th in thetas_deg])


def _ramp(n):
    if _FILTER == "kak":
        m = np.concatenate((np.arange(1, n // 2 + 1, 2), np.arange(n // 2 - 1, 0, -2)))
        h = np.zeros(n); h[0] = 0.25; h[1::2] = -1 / (np.pi * m) ** 2
        return 2 * np.real(np.fft.fft(h))
    f = np.fft.fftfreq(n)
    return 2 * np.abs(f)        # ramp do 1 (jak w skimage), inaczej wynik o czynnik 2 za maly


def fbp(sino, thetas_deg, out_n):
    """Filtrowana projekcja wsteczna (ramp). sino: (n_angles, n)."""
    n = sino.shape[1]; pad = int(2 ** np.ceil(np.log2(2 * n)))
    P = np.fft.fft(sino, n=pad, axis=1) * _ramp(pad)[None, :]
    filt = np.real(np.fft.ifft(P, axis=1))[:, :n]
    img = np.zeros((n, n))
    for row, th in zip(filt, thetas_deg):
        img += rotate(np.tile(row, (n, 1)), -th, reshape=False, order=_ORDER, mode="constant")
    return (img * np.pi / (2 * len(thetas_deg))).astype(np.float32)


def split_events(N, K):
    """Rozdziela dokladnie N zdarzen na K katow: N//K kazdemu, reszte N%K po jednym na katy rozlozone rowno
    (deterministycznie, bez losowania). (Suma po katach daje dokladnie N.)"""
    N, K = int(N), int(K); per = np.full(K, N // K, dtype=np.int64); r = N % K
    if r: per[(np.arange(r) * K) // r] += 1
    return per


def homodyne_wigner(W, N, rng, n_angles=24):
    """N zdarzen homodynowych rozlozonych rowno na n_angles katow w [0, 180). Zwraca obraz Wignera z FBP."""
    th = np.linspace(0, 180, n_angles, endpoint=False)
    sino = radon(np.asarray(W, np.float64), th)            # (K, n): calki liniowe W
    out = np.zeros_like(sino); per = split_events(N, n_angles)
    for k in range(n_angles):
        p = np.clip(sino[k], 0, None); s = p.sum()
        if per[k] == 0 or s <= 0: continue
        out[k] = rng.multinomial(per[k], p / s) / per[k] * sino[k].sum()   # zachowanie masy rzutu
    return fbp(out, th, W.shape[0])


def homodyne_wigner_exact(W, n_angles=24):
    th = np.linspace(0, 180, n_angles, endpoint=False)
    return fbp(radon(np.asarray(W, np.float64), th), th, W.shape[0])


def sinogram_counts(W, N, rng, K=36):
    """Wejscie dla sieci homodynowej BEZ rekonstrukcji: histogramy N/K zdarzen z K projekcji (K x n); N=0 -> dokladne projekcje."""
    th = np.linspace(0, 180, K, endpoint=False)
    sino = radon(np.asarray(W, np.float64), th)
    if N == 0: return sino.astype(np.float32)
    out = np.zeros_like(sino); per = split_events(N, K)
    for k in range(K):
        p = np.clip(sino[k], 0, None); s = p.sum()
        if per[k] == 0 or s <= 0: continue
        out[k] = rng.multinomial(per[k], p / s) / per[k] * sino[k].sum()
    return out.astype(np.float32)
