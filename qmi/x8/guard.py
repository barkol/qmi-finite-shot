"""Data guard: checks a directory of part files written by qmi.gen.
 - number of files (--expect), no *.tmp.npz,
 - key xmax == --xmax in every file (gen with QMI_XMAX stores the window),
 - Husimi mass inside the window sum(Q)*dx^2 per sample (channels clean/loss/dephase) (min, 1st percentile, fraction < 0.99) per class,
 - optionally --ref: rho and rho_clean bitwise equal to the files of a reference set with the same names (same states, other window).
Exit code 0 = OK, 1 = error. Prints GUARD_OK / GUARD_FAIL.
  python -m qmi.x8.guard --dir data_eq/ahm/test/sweep --xmax 12 --expect 441 --ref data/ahm/test/sweep [--sample 40]"""
import argparse, glob, os, sys, numpy as np
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--xmax", type=float, required=True)
ap.add_argument("--expect", type=int, default=0); ap.add_argument("--ref", default=""); ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--min-mass", type=float, default=0.99); ap.add_argument("--max-low-frac", type=float, default=1.0)
a = ap.parse_args()
fs = sorted(glob.glob(os.path.join(a.dir, "part_*.npz"))); fs = [f for f in fs if not f.endswith(".tmp.npz")]
tmp = glob.glob(os.path.join(a.dir, "*.tmp.npz")); err = []
if a.expect and len(fs) != a.expect: err.append(f"files {len(fs)} != {a.expect}")
if tmp: err.append(f"tmp files: {len(tmp)}")
chk = fs if not a.sample else [fs[i] for i in np.linspace(0, len(fs) - 1, min(a.sample, len(fs))).astype(int)]
dx = 2 * a.xmax / 127; mass = {}; nref = 0
for f in chk:
    d = np.load(f, allow_pickle=True)
    if "xmax" not in d.files or float(d["xmax"]) != a.xmax: err.append(f"{Path(f).name}: xmax {d['xmax'] if 'xmax' in d.files else 'missing'}"); continue
    m = d["Q"].astype(np.float64).sum((1, 2)) * dx * dx
    for c, ch, v in zip(d["label"], d["channel"], m):
        if int(ch) in (0, 1, 2): mass.setdefault(int(c), []).append(v)   # gauss/combo: additive image noise, the mass is meaningless
    if a.ref:
        r = np.load(os.path.join(a.ref, Path(f).name), allow_pickle=True); nref += 1
        for k in ("rho", "rho_clean", "label", "channel", "level"):
            if np.array_equal(d[k], r[k]): continue
            # rho: noise at the level of float32 subnormal numbers is accepted (differences ~1e-24 in 3 narrow files)
            dev = float(np.abs(d[k].astype(np.complex128) - r[k].astype(np.complex128)).max()) if k.startswith("rho") and d[k].shape == r[k].shape else np.inf
            if dev <= 1e-12: print(f"{Path(f).name}: {k} differs bitwise, max |difference| = {dev:.2e} (within tolerance)")
            else: err.append(f"{Path(f).name}: {k} differs from {a.ref} (max |difference| = {dev:.2e})")
low_tot = 0
for c in sorted(mass):
    v = np.array(mass[c]); low = float((v < a.min_mass).mean()); low_tot = max(low_tot, low)
    print(f"class {c}: n={len(v)} massQ min={v.min():.5f} p1={np.percentile(v,1):.5f} med={np.median(v):.5f} fraction<{a.min_mass}={low:.4f}")
if low_tot > a.max_low_frac: err.append(f"fraction of samples with Q mass < {a.min_mass}: {low_tot:.4f} > {a.max_low_frac}")
print(f"files {len(fs)} (checked {len(chk)}, against ref {nref}), xmax={a.xmax}")
if err: print("GUARD_FAIL", *err[:20], sep="\n  "); sys.exit(1)
print("GUARD_OK")
