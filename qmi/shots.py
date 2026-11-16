"""Finite records of N measurement events.

Heterodyne detection: Q(beta) = <beta|rho|beta>/pi is the probability density of the outcomes
             -> N events = histogram (multinomial) on the image grid, directly, without an inversion.
Homodyne detection: the Wigner function is not measured directly. The N events are split among K phases; each phase gives a
             histogram of the quadrature marginal (Radon projection of W); the image is reconstructed by filtered
             back-projection (FBP, ramp filter).
Both functions act on precomputed images (apply_wigner / apply_husimi), so they do not depend on the quadrature units:
the projections are line integrals of the image on the pixel grid.
"""
import os
import numpy as np
from scipy.ndimage import rotate
from qsdl.config import XMAX as _XMAX   # default window from QMI_XMAX

DX = lambda grid, xmax: 2 * xmax / (grid - 1)


def heterodyne_image(Q, N, rng, xmax=None):
    """Q: exact Husimi image (grid x grid). Returns the density estimate from N events (same scale as Q)."""
    p = np.clip(np.asarray(Q, np.float64), 0, None).ravel()
    p /= p.sum()
    counts = rng.multinomial(int(N), p).reshape(Q.shape)
    dx = DX(Q.shape[0], _XMAX if xmax is None else xmax)
    mass = float(np.clip(Q, 0, None).sum() * dx * dx)   # ~1 (integral of Q over the grid) -> keeps the scale of Q
    return (counts / (N * dx * dx) * mass).astype(np.float32)


# FBP variant selected by environment variables; default: order 1, ramp 2|f| (used for all results).
#   QMI_FBP_ORDER=2  biquadratic interpolation in the rotations (projections and back-projection)
#   QMI_FBP_FILTER=kak  Ram-Lak filter from the spatial domain (Kak-Slaney), exact zero-frequency component
_ORDER = int(os.environ.get("QMI_FBP_ORDER", "1"))
_FILTER = os.environ.get("QMI_FBP_FILTER", "freq")
assert _ORDER in (1, 2, 3) and _FILTER in ("freq", "kak"), (_ORDER, _FILTER)


def radon(img, thetas_deg):
    """Projections: for every angle, the sum over axis 0 of the image rotated by that angle. Returns (n_angles, n)."""
    return np.stack([rotate(img, th, reshape=False, order=_ORDER, mode="constant").sum(axis=0) for th in thetas_deg])


def _ramp(n):
    if _FILTER == "kak":
        m = np.concatenate((np.arange(1, n // 2 + 1, 2), np.arange(n // 2 - 1, 0, -2)))
        h = np.zeros(n); h[0] = 0.25; h[1::2] = -1 / (np.pi * m) ** 2
        return 2 * np.real(np.fft.fft(h))
    f = np.fft.fftfreq(n)
    return 2 * np.abs(f)        # ramp up to 1 (as in skimage); otherwise the result is a factor 2 too small


def fbp(sino, thetas_deg, out_n):
    """Filtered back-projection (ramp filter). sino: (n_angles, n)."""
    n = sino.shape[1]; pad = int(2 ** np.ceil(np.log2(2 * n)))
    P = np.fft.fft(sino, n=pad, axis=1) * _ramp(pad)[None, :]
    filt = np.real(np.fft.ifft(P, axis=1))[:, :n]
    img = np.zeros((n, n))
    for row, th in zip(filt, thetas_deg):
        img += rotate(np.tile(row, (n, 1)), -th, reshape=False, order=_ORDER, mode="constant")
    return (img * np.pi / (2 * len(thetas_deg))).astype(np.float32)


def split_events(N, K):
    """Split exactly N events among K phases: N//K to each, the remaining N%K one each to evenly spaced phases
    (deterministic, no random draw). The sum over the phases is exactly N."""
    N, K = int(N), int(K); per = np.full(K, N // K, dtype=np.int64); r = N % K
    if r: per[(np.arange(r) * K) // r] += 1
    return per


def homodyne_wigner(W, N, rng, n_angles=24):
    """N homodyne events split evenly among n_angles phases in [0, 180). Returns the Wigner image reconstructed by FBP."""
    th = np.linspace(0, 180, n_angles, endpoint=False)
    sino = radon(np.asarray(W, np.float64), th)            # (K, n): line integrals of W
    out = np.zeros_like(sino); per = split_events(N, n_angles)
    for k in range(n_angles):
        p = np.clip(sino[k], 0, None); s = p.sum()
        if per[k] == 0 or s <= 0: continue
        out[k] = rng.multinomial(per[k], p / s) / per[k] * sino[k].sum()   # keeps the mass of the projection
    return fbp(out, th, W.shape[0])


def homodyne_wigner_exact(W, n_angles=24):
    th = np.linspace(0, 180, n_angles, endpoint=False)
    return fbp(radon(np.asarray(W, np.float64), th), th, W.shape[0])


def sinogram_counts(W, N, rng, K=36):
    """Input of the homodyne network WITHOUT reconstruction: histograms of N/K events at each of K phases (K x n); N=0 -> exact projections."""
    th = np.linspace(0, 180, K, endpoint=False)
    sino = radon(np.asarray(W, np.float64), th)
    if N == 0: return sino.astype(np.float32)
    out = np.zeros_like(sino); per = split_events(N, K)
    for k in range(K):
        p = np.clip(sino[k], 0, None); s = p.sum()
        if per[k] == 0 or s <= 0: continue
        out[k] = rng.multinomial(per[k], p / s) / per[k] * sino[k].sum()
    return out.astype(np.float32)
