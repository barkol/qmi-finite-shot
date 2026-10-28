"""Plan etapow i sieci (okno [-8,8]^2, 128x128): jedno zrodlo prawdy dla generowania danych, treningu i ewaluacji.
Ziarna, liczebnosci i argumenty etapow; qmi/eq/plan_eq.py wyprowadza z niego plan dla okna 12 (rodzina eqn, szum eq).
Id zbioru = sciezka wzgledem korzenia danych."""
LV_HO = "0,20,40,60,80,100"

# ---- dane dokladne: (id_wyjscia_bazowe, tryb, split, argumenty qmi.gen, oczekiwana liczba plikow) ----
def _mx(out, split, extra=""):
    n = "--n-total 4200" if split == "val" else "--n-total 20000"
    return dict(kind="gen", id=f"{out}/{split}/mixed", out=out, args=f"mixed --split {split} {n} {extra}".split(), expect=42 if split == "val" else 196)
def _sw(out, extra="", expect=441):
    return dict(kind="gen", id=f"{out}/test/sweep", out=out, args=f"sweep --split test --n 100 {extra}".split(), expect=expect)
def _rec(mod, base, src, dst, extra=""):
    return [dict(kind="rec", mod=mod, id=f"{base}/{sp}/{dst}", src=f"{base}/{sp}/{src}", args=extra.split()) for sp in ("val", "train")]

AH = "--family ahmed"
STAGES = (
    [_mx("qmi", "val"), _mx("qmi", "train"), _mx("ahm", "val", AH), _mx("ahm", "train", AH)]
    + _rec("shots_data", "qmi", "mixed", "shots", "--copies 2") + _rec("shots_data", "ahm", "mixed", "shots", "--copies 2")
    + _rec("shots_data", "qmi", "mixed", "shotsK12", "--copies 2 --K 12") + _rec("shots_data", "qmi", "mixed", "shotsK6", "--copies 2 --K 6")
    + _rec("shots_data", "qmi", "mixed", "shotsK18", "--copies 2 --K 18") + _rec("sino_data", "qmi", "mixed", "sino")
    + _rec("sino_data", "ahm", "mixed", "sino") + _rec("shots_data", "ahm", "mixed", "shotsK12", "--copies 2 --K 12")
    + [_sw("qmi"), _sw("ahm", AH)]
    + _rec("shots_data_drift", "qmi", "mixed", "shots_drift", "--copies 2") + _rec("shots_data_drift", "ahm", "mixed", "shots_drift", "--copies 2")
    + _rec("shots_data", "qmi", "mixed", "shotsK24", "--copies 2 --K 24") + _rec("shots_data", "qmi", "mixed", "shotsK72", "--copies 2 --K 72")
    + [_mx("ho", "val", "--holdout train --seed 2027"), _mx("ho", "train", "--holdout train --seed 2027")] + _rec("shots_data", "ho", "mixed", "shots", "--copies 2")
    + [_mx("ahm_hoS", "val", AH + " --holdout train --stratify --seed 5051"), _mx("ahm_hoS", "train", AH + " --holdout train --stratify --seed 5051")]
    + _rec("shots_data", "ahm_hoS", "mixed", "shots", "--copies 2")
    + [_sw("ho_in", f"--holdout train --levels {LV_HO} --seed 2028", 126), _sw("ho_out", f"--holdout test --levels {LV_HO} --seed 2029", 126),
       _sw("ahm_ho_in", f"{AH} --holdout train --levels {LV_HO} --seed 4048", 126), _sw("ahm_ho_out", f"{AH} --holdout test --levels {LV_HO} --seed 4049", 126)]
    + [_mx("qmi_gx", "val", "--gkpx --stratify --seed 6061"), _mx("qmi_gx", "train", "--gkpx --stratify --seed 6061")]
    + _rec("shots_data", "qmi_gx", "mixed", "shotsK6", "--copies 2 --K 6")
    + [_mx("qmi_narrow", "val", "--grid narrow --seed 3031"), _mx("qmi_narrow", "train", "--grid narrow --seed 3031"),
       _mx("ahm_narrow", "val", AH + " --grid narrow --seed 3031"), _mx("ahm_narrow", "train", AH + " --grid narrow --seed 3031")]
    + [_mx("ahm_ho", "val", AH + " --holdout train --seed 4041"), _mx("ahm_ho", "train", AH + " --holdout train --seed 4041")]
    + _rec("shots_data", "ahm_ho", "mixed", "shots", "--copies 2")
)
STAGE_BY_ID = {s["id"]: s for s in STAGES}
TESTS = ["qmi/test/sweep", "ahm/test/sweep", "ho_in/test/sweep", "ho_out/test/sweep", "ahm_ho_in/test/sweep", "ahm_ho_out/test/sweep"]

# ---- sieci: 131 (baza 79, Ahmed 52) ----
R, RA = "runs_x8", "runs_ahm_x8"
def _net(run, rep, seed, tr, va, prio, where="cloud", evals=()):
    return dict(run=run, rep=rep, seed=seed, train=tr, val=va, prio=prio, where=where, evals=list(evals))
def _tv(base, ds): return f"{base}/train/{ds}", f"{base}/val/{ds}"
SW = lambda base: [(f"{base}/test/sweep", "eval_sweep.json")]
HO = lambda a, b: [(f"{a}/test/sweep", "eval_ho_in.json"), (f"{b}/test/sweep", "eval_ho_out.json")]
NETS = []
for s in range(3):
    for rep in "WQ":
        NETS.append(_net(f"{R}/clf_{rep}_exact_s{s}", rep, s, *_tv("qmi", "mixed"), 1, evals=SW("qmi")))
        NETS.append(_net(f"{RA}/clf_{rep}_exact_s{s}", rep, s, *_tv("ahm", "mixed"), 1, evals=SW("ahm")))
        NETS.append(_net(f"{RA}/clf_{rep}_shots_s{s}", rep, s, *_tv("ahm", "shots"), 1, evals=SW("ahm")))
for s in range(5):
    for rep in "WQ": NETS.append(_net(f"{R}/clf_{rep}_shots_s{s}", rep, s, *_tv("qmi", "shots"), 1, evals=SW("qmi")))
for f in range(5):
    for rep in "WQ":
        NETS.append(_net(f"{R}/clf_{rep}_cvexact_f{f}", rep, 0, f"cv5/mixed/fold{f}/train", f"cv5/mixed/fold{f}/val", 2, evals=SW("qmi")))
        NETS.append(_net(f"{R}/clf_{rep}_cvshots_f{f}", rep, 0, f"cv5/shots/fold{f}/train", f"cv5/shots/fold{f}/val", 2))
for s in range(2):
    for rep in "WQ":
        NETS.append(_net(f"{R}/clf_{rep}_drift_s{s}", rep, s, *_tv("qmi", "shots_drift"), 2))
        NETS.append(_net(f"{RA}/clf_{rep}_drift_s{s}", rep, s, *_tv("ahm", "shots_drift"), 2))
for s in range(3):
    for rep in "WQ":
        NETS.append(_net(f"{R}/clf_{rep}_hoexact_s{s}", rep, s, *_tv("ho", "mixed"), 2, evals=HO("ho_in", "ho_out")))
        NETS.append(_net(f"{R}/clf_{rep}_hoshots_s{s}", rep, s, *_tv("ho", "shots"), 2))
        NETS.append(_net(f"{RA}/clf_{rep}_hoexactS_s{s}", rep, s, *_tv("ahm_hoS", "mixed"), 2, evals=HO("ahm_ho_in", "ahm_ho_out")))
        NETS.append(_net(f"{RA}/clf_{rep}_hoshotsS_s{s}", rep, s, *_tv("ahm_hoS", "shots"), 2))
        NETS.append(_net(f"{R}/clf_{rep}_narrow_s{s}", rep, s, *_tv("qmi_narrow", "mixed"), 3, evals=SW("qmi")))
        NETS.append(_net(f"{RA}/clf_{rep}_narrow_s{s}", rep, s, *_tv("ahm_narrow", "mixed"), 3, evals=SW("ahm")))
        ev = HO("ahm_ho_in", "ahm_ho_out") + ([("ahm/test/sweep", "eval_check_mainsweep.json")] if (rep, s) == ("W", 0) else [])
        NETS.append(_net(f"{RA}/clf_{rep}_hoexact_s{s}", rep, s, *_tv("ahm_ho", "mixed"), 4, evals=ev))
        NETS.append(_net(f"{RA}/clf_{rep}_hoshots_s{s}", rep, s, *_tv("ahm_ho", "shots"), 4))
for s in range(3):
    for K in (12, 6, 18): NETS.append(_net(f"{R}/clf_W_shotsK{K}_s{s}", "W", s, *_tv("qmi", f"shotsK{K}"), 3, "workstation"))
    NETS.append(_net(f"{R}/clf_W_sino_s{s}", "W", s, *_tv("qmi", "sino"), 3, "workstation"))
    NETS.append(_net(f"{RA}/clf_W_sino_s{s}", "W", s, *_tv("ahm", "sino"), 3, "workstation"))
    NETS.append(_net(f"{RA}/clf_W_shotsK12_s{s}", "W", s, *_tv("ahm", "shotsK12"), 3, "workstation"))
    for K in (24, 72): NETS.append(_net(f"{R}/clf_W_shotsK{K}_s{s}", "W", s, *_tv("qmi", f"shotsK{K}"), 3))
    NETS.append(_net(f"{R}/clf_W_gkpxK6_s{s}", "W", s, *_tv("qmi_gx", "shotsK6"), 3))
NETS.sort(key=lambda n: n["prio"])   # stabilne: w obrebie priorytetu kolejnosc jak wyzej

def deps(path):
    """Zbiory danych potrzebne dla sciezki treningowej (cv5 = suma train+val zrodla)."""
    if path.startswith("cv5/"):
        kind = path.split("/")[1]; return [f"qmi/train/{kind}", f"qmi/val/{kind}"]
    return [path]

if __name__ == "__main__":
    import collections
    print(len(STAGES), "etapow danych;", len(NETS), "sieci:", dict(collections.Counter(n["where"] for n in NETS)),
          "baza", sum(n["run"].startswith(R + "/") for n in NETS), "Ahmed", sum(n["run"].startswith(RA + "/") for n in NETS))
    assert len({n["run"] for n in NETS}) == len(NETS)
    for n in NETS:
        for p in (n["train"], n["val"]):
            for d in deps(p): assert d in STAGE_BY_ID, (n["run"], d)
        for t, _ in n["evals"]: assert t in STAGE_BY_ID, t
