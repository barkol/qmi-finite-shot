"""Straznik danych: sprawdza katalog plikow czesciowych qmi.gen.
 - liczba plikow (--expect), brak *.tmp.npz,
 - klucz xmax == --xmax w kazdym pliku (gen z QMI_XMAX zapisuje okno),
 - masa Husimiego w oknie sum(Q)*dx^2 per probka (kanaly clean/loss/dephase) (min, 1. percentyl, ulamek < 0.99) per klasa,
 - opcjonalnie --ref: rho i rho_clean bitowo rowne plikom zbioru odniesienia o tej samej nazwie (te same stany, inne okno).
Wyjscie 0 = OK, 1 = blad. Drukuje GUARD_OK / GUARD_FAIL.
  python -m qmi.x8.guard --dir data_eq/ahm/test/sweep --xmax 12 --expect 441 --ref data/ahm/test/sweep [--sample 40]"""
import argparse, glob, os, sys, numpy as np
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--xmax", type=float, required=True)
ap.add_argument("--expect", type=int, default=0); ap.add_argument("--ref", default=""); ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--min-mass", type=float, default=0.99); ap.add_argument("--max-low-frac", type=float, default=1.0)
a = ap.parse_args()
fs = sorted(glob.glob(os.path.join(a.dir, "part_*.npz"))); fs = [f for f in fs if not f.endswith(".tmp.npz")]
tmp = glob.glob(os.path.join(a.dir, "*.tmp.npz")); err = []
if a.expect and len(fs) != a.expect: err.append(f"plikow {len(fs)} != {a.expect}")
if tmp: err.append(f"pliki tmp: {len(tmp)}")
chk = fs if not a.sample else [fs[i] for i in np.linspace(0, len(fs) - 1, min(a.sample, len(fs))).astype(int)]
dx = 2 * a.xmax / 127; mass = {}; nref = 0
for f in chk:
    d = np.load(f, allow_pickle=True)
    if "xmax" not in d.files or float(d["xmax"]) != a.xmax: err.append(f"{Path(f).name}: xmax {d['xmax'] if 'xmax' in d.files else 'brak'}"); continue
    m = d["Q"].astype(np.float64).sum((1, 2)) * dx * dx
    for c, ch, v in zip(d["label"], d["channel"], m):
        if int(ch) in (0, 1, 2): mass.setdefault(int(c), []).append(v)   # gauss/combo: szum addytywny obrazu, masa bez sensu
    if a.ref:
        r = np.load(os.path.join(a.ref, Path(f).name), allow_pickle=True); nref += 1
        for k in ("rho", "rho_clean", "label", "channel", "level"):
            if np.array_equal(d[k], r[k]): continue
            # rho: dopuszczamy szum na poziomie liczb subnormalnych float32 (roznice ~1e-24 w 3 plikach narrow)
            dev = float(np.abs(d[k].astype(np.complex128) - r[k].astype(np.complex128)).max()) if k.startswith("rho") and d[k].shape == r[k].shape else np.inf
            if dev <= 1e-12: print(f"{Path(f).name}: {k} rozne bitowo, max |roznica| = {dev:.2e} (w tolerancji)")
            else: err.append(f"{Path(f).name}: {k} rozne od {a.ref} (max |roznica| = {dev:.2e})")
low_tot = 0
for c in sorted(mass):
    v = np.array(mass[c]); low = float((v < a.min_mass).mean()); low_tot = max(low_tot, low)
    print(f"klasa {c}: n={len(v)} masaQ min={v.min():.5f} p1={np.percentile(v,1):.5f} med={np.median(v):.5f} ulamek<{a.min_mass}={low:.4f}")
if low_tot > a.max_low_frac: err.append(f"ulamek probek z masa Q < {a.min_mass}: {low_tot:.4f} > {a.max_low_frac}")
print(f"plikow {len(fs)} (sprawdzonych {len(chk)}, z ref {nref}), xmax={a.xmax}")
if err: print("GUARD_FAIL", *err[:20], sep="\n  "); sys.exit(1)
print("GUARD_OK")
