"""Markdown tables for REPORT.md from lr_results.json (percent error)."""
import json, sys
from pathlib import Path
d = json.load(open(Path(__file__).resolve().parent / "lr_results.json"))["cells"]
names = {"clean": "noiseless", "deph20": "20% dephasing", "loss50": "50% loss", "loss80": "80% loss", "gauss20": "Gaussian 20% (s2=0.2)", "gauss50": "Gaussian 50% (s2=0.5)"}
def f(r):
    s = f"{100*r['obs_err']:.2f} [{100*r['boot_lo']:.2f}, {100*r['boot_hi']:.2f}]"
    if r['boot_hi'] == 0: s = f"0.00 [0, {100*r['wilson_hi']:.2f}]"
    return s
def n(r): return "n/a" if not r["net"] else f"{100*r['net']['err']:.1f}"
print("| cell | N | Q obs | Q net | W36 obs | W36 net | W12 obs | W12 net | W6 obs | W6 net |\n|---|---|---|---|---|---|---|---|---|---|")
for c in names:
    for N in (100, 1000, 10000):
        rs = [d[f"{c}|{rep}|{N}"] for rep in ("Q", "W36", "W12", "W6")]
        print(f"| {names[c]} | {N} | " + " | ".join(f"{f(r)} | {n(r)}" for r in rs) + " |")
print("\n| cell | N | gap obs K=36 (pts) | gap net K=36 | ratio | gap obs K=12 | gap net K=12 | ratio | gap obs K=6 | gap net K=6 | ratio |\n|---|---|---|---|---|---|---|---|---|---|---|")
for c in list(names)[:4]:
    for N in (100, 1000, 10000):
        row = f"| {names[c]} | {N} |"
        for K in (36, 12, 6):
            g = d[f"{c}|gap_W{K}|{N}"]; ng = g["net_gap_pts"]; ra = g.get("ratio_obs_over_net")
            row += f" {g['obs_gap_pts']:.2f} [{g['lo']:.2f}, {g['hi']:.2f}] | {ng:.2f} | " + (f"{ra:.2f}" if ra is not None and abs(ng) >= 1 else "-") + " |"
        print(row)
