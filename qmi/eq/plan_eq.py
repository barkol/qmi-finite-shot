"""Plan of the eq run of the base set with equal <n> in all families: family eqn (qmi.eqn_states,
vacuum -> squeezed vacuum), window [-12,12]^2, 128x128 (QMI_XMAX=12).
Stages and networks come from sm/plan.py (seeds, sizes, arguments) with --family eqn added; in addition the sino networks
s3, s4 (sino has 5 seeds). Dataset id = path relative to data_eq/; networks in runs_eq/.
Hold-out in eqn: training <n> in {2,3,4}, ho_out = <n> = 5 in all families (qmi.eqn_states.in_holdout_eqn).
QEQ_SET=ahm: Ahmed set with the eq noise definitions (--noise eq) and window 12;
dataset ids as in sm/plan.py (ahm/..., ahm_ho..., ahm_narrow), data in data_eq_ahm/, networks in runs_ahm_eq/ (52 from the plan + sino s3, s4)."""
import os
import copy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sm"))
import plan as _p

SET = os.environ.get("QEQ_SET", "base"); assert SET in ("base", "ahm"), SET
AHM = SET == "ahm"
R = "runs_ahm_eq" if AHM else "runs_eq"
STAGES = []
for s in _p.STAGES:
    if s["id"].startswith("ahm") != AHM: continue
    s = copy.deepcopy(s)
    if s["kind"] == "gen":
        if AHM: assert "--family" in s["args"] and "--noise" not in s["args"]; s["args"] += ["--noise", "eq"]
        else: assert "--family" not in s["args"]; s["args"] += ["--family", "eqn"]
    if s["kind"] == "gen" and s["args"][0] == "mixed":   # files of 25 samples (qmi/gen.py, eq noise) -> more files
        nt = int(s["args"][s["args"].index("--n-total") + 1]); s["expect"] = 7 * ((nt // 7) // 25)
    STAGES.append(s)
STAGE_BY_ID = {s["id"]: s for s in STAGES}
TESTS = [t for t in _p.TESTS if t.startswith("ahm") == AHM]

NETS = []
src = _p.RA if AHM else _p.R
for n in _p.NETS:
    if not n["run"].startswith(src + "/"): continue
    n = copy.deepcopy(n); n["run"] = R + n["run"][len(src):]; NETS.append(n)
for s in (3, 4):
    NETS.append(_p._net(f"{R}/clf_W_sino_s{s}", "W", s, *_p._tv("ahm" if AHM else "qmi", "sino"), 3, "workstation"))
NETS.sort(key=lambda n: n["prio"])
deps = _p.deps

if __name__ == "__main__":
    print(len(STAGES), "stages,", len(NETS), "networks,", len(TESTS), "tests")
    assert len({n["run"] for n in NETS}) == len(NETS)
    for n in NETS:
        for p in (n["train"], n["val"]):
            for d in deps(p): assert d in STAGE_BY_ID, (n["run"], d)
        for t, _ in n["evals"]: assert t in STAGE_BY_ID, t
    for s in STAGES:
        if s["kind"] == "gen": print(" ", s["id"], " ".join(s["args"]), s["expect"])
