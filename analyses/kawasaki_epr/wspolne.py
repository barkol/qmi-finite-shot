"""Wspolne funkcje: wczytanie przebiegow Keysight UXR (HDF5) i kwadratury modow czasowych (Kawasaki+ 2024 / Asavanant+ 2019)."""
import os, re, numpy as np, h5py
from pathlib import Path
RAW = os.path.join(os.environ.get("QMI_ROOT", str(Path(__file__).resolve().parents[2])), "dane_eksp/kawasaki2024_epr/raw/bEPR_for_dryad")
DT = 3.90625e-12          # XInc (256 GSa/s), sprawdzane przy wczytaniu
NPTS = 5121; NSEG = 5000


def load(path, tags=False):
    """Zwraca (V[nseg, npts] float32 w woltach, time_tags[nseg], xorg[nseg]) w kolejnosci numerow segmentow. Strażnik formatu."""
    with h5py.File(path, "r") as h:
        W = h["Waveforms"]; chs = list(W.keys())
        if len(chs) != 1: raise RuntimeError(f"{path}: kanaly {chs}")
        g = W[chs[0]]; a = g.attrs
        if abs(a["XInc"] - DT) > 1e-18 or a["NumPoints"] != NPTS: raise RuntimeError(f"{path}: XInc {a['XInc']} NumPoints {a['NumPoints']}")
        keys = sorted(g.keys(), key=lambda k: int(re.search(r"Seg(\d+)Data", k).group(1)))
        if len(keys) != a["NumSegments"]: raise RuntimeError(f"{path}: {len(keys)} segmentow != {a['NumSegments']}")
        X = np.empty((len(keys), NPTS), np.float32); T = np.empty(len(keys)); O = np.empty(len(keys))
        for i, k in enumerate(keys):
            d = g[k]; X[i] = d[()]
            if tags: T[i] = d.attrs["SegmentedTimeTag"]; O[i] = d.attrs["SegmentedXOrg"]
        X *= np.float32(a["YInc"]); X += np.float32(a["YOrg"])
    return X, T, O


def mode_fn(nsamp=10, gamma=2 * np.pi * 10.5e9, dt=DT, tc=0.0):
    """Funkcja modu Asavanant+ 2019 (supl. S1), skrocona 1e3 razy jak u Kawasaki+ (40 ns -> 40 ps):
    f(t) ~ (t - t0 + 0.5 + tc) exp[-g^2 (t - t0 + 0.5)^2] w oknie |t - t0| <= Dt/2. W S1 czas w ns, wiec '0.5' = 0.5 ns = pol probki
    przy 1 GSa/s; tutaj przesuniecie o pol probki odpowiada srodkowi okna z parzysta liczba probek. Normalizacja sum f^2 = 1."""
    t = (np.arange(nsamp) - (nsamp - 1) / 2) * dt
    f = (t + tc) * np.exp(-gamma ** 2 * t ** 2)
    return (f / np.sqrt(np.sum(f ** 2))).astype(np.float64)


def quads(V, f, step=None):
    """Kwadratury kolejnych, rozlacznych pakietow: q[seg, k] = sum_j f_j V[seg, k*step + j]."""
    n = len(f); step = step or n; m = (V.shape[1] - n) // step + 1
    idx = np.arange(m)[:, None] * step + np.arange(n)[None, :]
    return np.einsum("skj,j->sk", V[:, idx].astype(np.float64), f)
