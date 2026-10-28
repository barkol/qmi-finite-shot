"""Plan przeliczenia zbioru bazowego z rowna <n> we wszystkich klasach: rodzina eqn (qmi.eqn_states,
proznia -> sciśnieta proznia), okno [-12,12]^2, 128x128 (QMI_XMAX=12). 
Etapy i sieci pochodza z sm/plan.py (ziarna, liczebnosci, argumenty), z dopisanym --family eqn; dodatkowo sieci sino s3, s4 (sino ma 5 ziaren). Id zbioru = sciezka wzgledem data_eq/; sieci w runs_eq/.
Hold-out w eqn: trening <n> w {2,3,4}, ho_out = <n> = 5 we wszystkich klasach (qmi.eqn_states.in_holdout_eqn).
QEQ_SET=ahm: zbior Ahmeda z definicjami szumu eq (--noise eq) i oknem 12;
id zbiorow jak w sm/plan.py (ahm/..., ahm_ho..., ahm_narrow), dane w data_eq_ahm/, sieci w runs_ahm_eq/ (52 z planu + sino s3, s4)."""
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
    if s["kind"] == "gen" and s["args"][0] == "mixed":   # pliki po 25 probek (qmi/gen.py, noise eq) -> wiecej plikow
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
    print(len(STAGES), "etapow,", len(NETS), "sieci,", len(TESTS), "testow")
    assert len({n["run"] for n in NETS}) == len(NETS)
    for n in NETS:
        for p in (n["train"], n["val"]):
            for d in deps(p): assert d in STAGE_BY_ID, (n["run"], d)
        for t, _ in n["evals"]: assert t in STAGE_BY_ID, t
    for s in STAGES:
        if s["kind"] == "gen": print(" ", s["id"], " ".join(s["args"]), s["expect"])
