"""Compare two JSON files with a relative tolerance: python json_close.py A.json B.json [rtol]
Exit 0 if the structure and strings are equal and all numbers agree to rtol (default 1e-9), else 1 (first differences printed)."""
import json, math, sys


def walk(a, b, path, rtol, bad):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): bad.append(f"{path}: keys differ {sorted(set(a) ^ set(b))[:5]}"); return
        for k in a: walk(a[k], b[k], f"{path}/{k}", rtol, bad)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): bad.append(f"{path}: length {len(a)} vs {len(b)}"); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]", rtol, bad)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not (a == b or math.isclose(a, b, rel_tol=rtol, abs_tol=1e-300) or (a != a and b != b)): bad.append(f"{path}: {a!r} vs {b!r}")
    elif a != b: bad.append(f"{path}: {a!r} vs {b!r}")


if __name__ == "__main__":
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
    bad = []; walk(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), "", rtol, bad)
    for m in bad[:10]: print("   ", m)
    sys.exit(1 if bad else 0)
