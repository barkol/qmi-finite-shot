"""Wczytywanie plikow czesciowych qmi.gen -> tablice."""
from pathlib import Path
import numpy as np


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)

CH_NAMES = {0: "clean", 1: "loss", 2: "dephase", 3: "gauss", 4: "combo"}


def load_dir(root, rep="W", channels=None, with_rho=False):
    """rep: W (Wigner) lub Q (Husimi). channels: iterowalne nazwy kanalow (None = wszystkie)."""
    want = None if channels is None else {k for k, v in CH_NAMES.items() if v in set(channels)}
    X, y, ch, lv, rho = [], [], [], [], []
    for f in sorted(Path(root).glob("part_*.npz")):
        d = np.load(f, allow_pickle=True)
        if want is not None and int(d["channel"][0]) not in want:
            continue
        X.append(d[rep]); y.append(d["label"]); ch.append(d["channel"]); lv.append(d["level"])
        if with_rho: rho.append(d["rho_clean"])
    if not X:
        raise FileNotFoundError(f"brak plikow part_*.npz w {root}")
    out = dict(X=np.concatenate(X).astype(np.float32), y=np.concatenate(y).astype(np.int64),
               channel=np.concatenate(ch), level=np.concatenate(lv))
    if with_rho: out["rho_clean"] = np.concatenate(rho)
    return out
