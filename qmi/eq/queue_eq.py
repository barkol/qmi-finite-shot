"""Job queue of the eq run for the base set with equal <n> (family eqn, window [-12,12]^2, 128x128).
Runs on one machine with a GPU: exact datasets generated locally (qmi.gen --family eqn + mass guard, xmax=12), records,
78 networks (qmi/eq/plan_eq.py), evaluations (sino with 5 seeds, paired Q-W and Q-sino comparisons, energy control) and the
final step qmi/eq/finish_eq.py. Parallel lanes (data, training, evaluations, tests, baselines); stops on the first error
(data_eq/_markers/FAILED_queue).
QEQ_SET=ahm: Ahmed set (eq noise, window 12): data_eq_ahm/, runs_ahm_eq/, logs in logs/eq_ahm/; the training lanes wait
until the base queue has trained its 78 networks (RAM and GPU); data and non-network baselines start at once.
Distributed work: QEQ_TRAIN_RX = regex of the networks trained on THIS machine (the others are only awaited as history.json
delivered from elsewhere; datasets needed only by them are not prepared); QEQ_LANES = lanes to run
(data,train,eval,tests,baselines; default all); QEQ_GEN_WAIT_RX = regex of the ids of exact datasets delivered from elsewhere
(wait for the marker and the number of files instead of generating).
The data epoch marker _markers/EPOCH (an empty file created before the run) invalidates step markers in older logs.
Start/resume: [QEQ_SET=ahm] QMI_ROOT=<work dir> setsid nohup python -m qmi.eq.queue_eq >> logs/eq/queue.log 2>&1 < /dev/null & disown"""
import os, sys, time, json, glob, shutil, subprocess, threading, fcntl, re, traceback
from pathlib import Path

ROOT = Path(os.environ.get("QMI_ROOT", Path(__file__).resolve().parents[2])); os.chdir(ROOT); sys.path.insert(0, str(ROOT))   # QMI_ROOT: working dir with data/runs
from qmi.eq import plan_eq as plan
PY = os.environ.get("QMI_PY", sys.executable)
AHM = plan.AHM
PERM = ROOT / ("data_eq_ahm" if AHM else "data_eq"); CACHE = Path(os.environ.get("QX_CACHE", "/dev/shm/qmi-eqa-cache" if AHM else "/dev/shm/qmi-eq-cache")); M = PERM / "_markers"
L = ROOT / ("logs/eq_ahm" if AHM else "logs/eq"); XM = 12
CACHE_GB = float(os.environ.get("QX_CACHE_GB", "11")); WK = int(os.environ.get("QX_WK", "20")); EPOCHS = 25
DRY = bool(os.environ.get("QX_DRY"))   # dry run: only prints the steps, computes nothing
TRAIN_RX = os.environ.get("QEQ_TRAIN_RX", ""); GEN_WAIT_RX = os.environ.get("QEQ_GEN_WAIT_RX", "")
LANES = set(os.environ.get("QEQ_LANES", "data,train,eval,tests,baselines").split(","))
for d in (PERM, CACHE, M, L): d.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, QMI_XMAX=str(XM), QMI_NPZ_COMPRESS="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONPATH=str(ROOT))
LOCK = threading.Lock(); EVLOCK = threading.Lock(); STOP = threading.Event()
def ts(): return time.strftime("%H:%M:%S")
def say(*a):
    with LOCK: print(ts(), *a, flush=True)
def fail(msg):
    say("ERROR:", msg); (M / "FAILED_queue").write_text(ts() + " " + msg + "\n"); STOP.set()

# ---------------- datasets ----------------
TESTS = plan.TESTS
B = "ahm" if AHM else "qmi"
PERM_IDS = set(TESTS) | {f"{B}/train/mixed", f"{B}/val/mixed", f"{B}_narrow/train/mixed", f"{B}_narrow/val/mixed"}
def disk_ok():
    st = os.statvfs(ROOT); free = st.f_bavail * st.f_frsize / 1e9
    if free < 10: fail(f"not enough disk space: {free:.1f} GB < 10 GB"); return False
    return True
def tag(i): return i.replace("/", "_")
def path(i):
    if i.startswith("cv5/"):
        kind = i.split("/")[1]; return (PERM if kind == "mixed" else CACHE) / i
    return (PERM if i in PERM_IDS else CACHE) / i
def nfiles(d): return len([f for f in glob.glob(str(d) + "/part_*.npz") if not f.endswith(".tmp.npz")])
def expect(i):
    st = plan.STAGE_BY_ID[i]
    if st["kind"] == "gen": return st["expect"]
    src = plan.STAGE_BY_ID[st["src"]]; return None   # records: number of source files without the gauss channel (counted afterwards)
def run(cmd, log, env=ENV):
    if DRY: say("[DRY]", " ".join(map(str, cmd))); return 0
    with open(log, "a") as f:
        f.write(f"\n### {ts()} {' '.join(map(str, cmd))}\n"); f.flush()
        return subprocess.run(list(map(str, cmd)), env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
def gb(p): return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file() and not f.is_symlink()) / 1e9 if Path(p).exists() else 0.0

READY = {}   # id -> True (verified in this process)
def ready(i):
    if READY.get(i): return True
    if i.startswith("cv5/"): return False
    st = plan.STAGE_BY_ID[i]; d = path(i)
    ok_mark = (M / f"{tag(i)}.ok").exists() if i in PERM_IDS else (M / f"cache_{tag(i)}.ok").exists()
    if not ok_mark: return False
    n = nfiles(d)
    if st["kind"] == "gen" and n != st["expect"]: return False
    if st["kind"] == "rec":
        exp = json.loads((M / f"cache_{tag(i)}.ok").read_text() or "{}").get("n")
        if n != exp or not n: return False
    READY[i] = True; return True

def prepare(i):
    """Prepares dataset i (not cv5). Returns when it is ready or raises an exception."""
    if ready(i): return
    if DRY: say("[DRY] prepare", i, "->", path(i)); READY[i] = True; return
    st = plan.STAGE_BY_ID[i]; d = path(i); lg = L / f"data_{tag(i)}.log"
    if st["kind"] == "gen" and GEN_WAIT_RX and re.search(GEN_WAIT_RX, i):   # dataset delivered from elsewhere
        mk = M / (f"{tag(i)}.ok" if i in PERM_IDS else f"cache_{tag(i)}.ok"); say("waiting for delivery:", i)
        while not (mk.exists() and nfiles(d) == st["expect"]):
            if STOP.is_set(): raise RuntimeError("stop")
            time.sleep(30)
        say("delivered", i, nfiles(d), "files")
    elif st["kind"] == "gen":                       # exact dataset generated locally
        base = (PERM if i in PERM_IDS else CACHE) / st["out"]
        if d.exists(): [f.unlink() for f in d.glob("*.tmp.npz")]
        say("generating", i, "->", d)
        if run([PY, "-m", "qmi.gen", *st["args"], "--out", base, "--workers", WK], lg): raise RuntimeError(f"gen {i} (log {lg})")
        if nfiles(d) != st["expect"]: raise RuntimeError(f"gen {i}: files {nfiles(d)} != {st['expect']}")
        if run([PY, "-m", "qmi.x8.guard", "--dir", d, "--xmax", XM, "--expect", st["expect"], "--sample", 40, "--max-low-frac", 0.0], lg):
            raise RuntimeError(f"guard {i} (log {lg})")
        (M / (f"{tag(i)}.ok" if i in PERM_IDS else f"cache_{tag(i)}.ok")).write_text(json.dumps({"n": st["expect"]}))
        say("ready", i, nfiles(d), "files", f"{gb(d):.2f} GB")
    else:                                         # records computed locally from the source
        prepare(st["src"]); src = path(st["src"])
        if d.exists(): [f.unlink() for f in d.glob("*.tmp.npz")]
        say("records", i, "<-", st["src"])
        if run([PY, "-m", f"qmi.{st['mod']}", "--src", src, "--dst", d, "--workers", WK, *st["args"]], lg): raise RuntimeError(f"records {i} (log {lg})")
        mark = "SINO_DATA_DONE" if st["mod"] == "sino_data" else "SHOTS_DATA_DONE"
        if not DRY and mark not in lg.read_text(): raise RuntimeError(f"records {i}: no {mark}")
        import numpy as np
        exp = sum(int(np.load(f)["channel"][0]) != 3 for f in sorted(glob.glob(str(src) + "/part_*.npz")))
        if nfiles(d) != exp or not exp: raise RuntimeError(f"records {i}: files {nfiles(d)} != {exp}")
        (M / f"cache_{tag(i)}.ok").write_text(json.dumps({"n": exp}))
        say("ready", i, nfiles(d), "files", f"{gb(d):.2f} GB")
    READY[i] = True

def cv5(kind):
    """Five-fold split (symbolic links) as in qmi.cv_folds."""
    if DRY: say("[DRY] cv5", kind); (M / f"dry_cv5_{kind}").write_text(""); return
    dst = path(f"cv5/{kind}/fold0/train").parents[1]
    okf = M / (f"cv5_{kind}.ok" if kind == "mixed" else f"cache_cv5_{kind}.ok")
    if okf.exists() and (dst / "folds.json").exists() and all((dst / f"fold{k}/val").exists() for k in range(5)): return
    for p in ("train", "val"): prepare(f"qmi/{p}/{kind}")
    lg = L / f"data_cv5_{kind}.log"
    if run([PY, "-m", "qmi.cv_folds", "--train", path(f"qmi/train/{kind}"), "--val", path(f"qmi/val/{kind}"), "--dst", dst], lg): raise RuntimeError(f"cv5 {kind}")
    okf.write_text("{}"); say("cv5", kind, "ready")

def deps_ready(n):
    for p in (n["train"], n["val"]):
        if p.startswith("cv5/"):
            kind = p.split("/")[1]
            okf = M / ((f"cv5_{kind}.ok" if kind == "mixed" else f"cache_cv5_{kind}.ok") if not DRY else f"dry_cv5_{kind}")
            if not okf.exists() or not all(ready(x) for x in plan.deps(p)): return False
        elif not ready(p): return False
    return True

# ---------------- training order (groups by data) ----------------
GROUPS = [r"runs_eq/clf_._(exact|cvexact)_", r"runs_ahm_eq/clf_._exact_", r"runs_eq/clf_._(shots|cvshots)_[sf]", r"runs_ahm_eq/clf_._shots_s",
          r"_drift_s", r"runs_eq/clf_W_shotsK12_", r"runs_eq/clf_W_shotsK6_", r"runs_eq/clf_W_shotsK18_", r"runs_eq/clf_W_shotsK24_",
          r"runs_eq/clf_W_shotsK72_", r"runs_ahm_eq/clf_W_shotsK12_", r"_sino_", r"runs_eq/clf_._ho(exact|shots)_",
          r"runs_ahm_eq/clf_._ho(exact|shots)S_", r"runs_eq/clf_._narrow_", r"runs_ahm_eq/clf_._narrow_", r"runs_ahm_eq/clf_._ho(exact|shots)_s"]
GROUPS = [g for g in os.environ.get("QEQ_GROUPS_FIRST", "").split(";") if g] + GROUPS   # priority groups (regexes, in order)
def gidx(n): return next(k for k, g in enumerate(GROUPS) if re.search(g, n["run"]))
NETS = sorted(plan.NETS, key=lambda n: (gidx(n), plan.NETS.index(n)))
assert len(NETS) == len(plan.NETS) == (54 if AHM else 78) and len({n["run"] for n in NETS}) == len(NETS)
def done(n): return (ROOT / n["run"] / "history.json").exists()
def mine(n): return not TRAIN_RX or bool(re.search(TRAIN_RX, n["run"]))

# datasets in the order needed by training, with their users (for evicting the cache)
ORDER, USERS = [], {}
def _add(i, user):
    if i not in ORDER: ORDER.append(i)
    USERS.setdefault(i, set()).add(user)
for n in NETS:
    if not mine(n): continue
    for p in (n["train"], n["val"]):
        for i in plan.deps(p):
            st = plan.STAGE_BY_ID[i]
            if st["kind"] == "rec": _add(st["src"], "rec:" + i)
            _add(i, n["run"])
        if p.startswith("cv5/shots"):
            for i in plan.deps(p): USERS[i].add(n["run"])
BL = {"runs_ahm_eq/_bl_ho": ["ahm_hoS/train/mixed", "ahm_ho/train/mixed"]} if AHM else {"runs_eq/_bl_ho": ["ho/train/mixed"]}
for u, ids in BL.items():
    for i in ids: USERS.setdefault(i, set()).add(u)
CV = {"mixed": [n["run"] for n in NETS if "cvexact" in n["run"]], "shots": [n["run"] for n in NETS if "cvshots" in n["run"]]}

def evictable(i):
    if path(i).is_relative_to(PERM) or i in PERM_IDS: return False
    for u in USERS.get(i, ()):
        if u.startswith("rec:"):
            if not ready(u[4:]): return False
        elif u != "test" and not (ROOT / u / "history.json").exists(): return False
    return True
def evict():
  with EVLOCK:
    for i in list(ORDER):
        d = path(i)
        if d.exists() and not i in PERM_IDS and evictable(i) and READY.get(i):
            if i in ("qmi/train/shots", "qmi/val/shots") and not all((ROOT / r / "history.json").exists() for r in CV["shots"]): continue
            say("evicting from cache:", i, f"{gb(d):.2f} GB"); shutil.rmtree(d); READY[i] = False
            (M / f"cache_{tag(i)}.ok").unlink(missing_ok=True)
            if i.startswith("qmi/") and "shots" == i.split("/")[-1] and (CACHE / "cv5/shots").exists():
                shutil.rmtree(CACHE / "cv5/shots"); (M / "cache_cv5_shots.ok").unlink(missing_ok=True)

EST = {"mixed": 1.6, "sweep": 6.0}      # estimated GB per dataset (train ~ 2.5, val ~ 0.5 for mixed; records ~1.8/0.4)
def est(i):
    if i in PERM_IDS: return 0.0
    k = i.split("/")[-1]; sp = i.split("/")[1] if "/" in i else ""
    if k == "sino": return 0.12
    base = 2.6 if k == "mixed" else 1.9
    return base if sp == "train" else base / 4.5

def lane_data():
    try:
        for i in ORDER:
            if STOP.is_set() or not disk_ok(): return
            # cache limit (RAM): wait until training releases datasets
            while not i in PERM_IDS and not ready(i) and gb(CACHE) + est(i) > CACHE_GB:
                evict()
                if STOP.is_set(): return
                time.sleep(30)
            prepare(i)
            if i == "qmi/val/mixed" or i == "qmi/train/mixed":
                if ready("qmi/train/mixed") and ready("qmi/val/mixed"): cv5("mixed")
            if i in ("qmi/train/shots", "qmi/val/shots") and ready("qmi/train/shots") and ready("qmi/val/shots"): cv5("shots")
        say("DATA_EQ_READY")
    except Exception as e:
        fail(f"data lane: {e!r}\n{traceback.format_exc()[-1500:]}")

# ---------------- training ----------------
def base_queue_done():
    """Ahmed set: training waits until the base queue has trained all its networks (counted in runs_eq/)."""
    if not AHM or os.environ.get("QEQ_NO_WAIT_BASE"): return True   # e.g. on a cloud instance that does not train the base set
    return len(list((ROOT / "runs_eq").glob("clf_*/history.json"))) >= 78
TAKEN = set()
def next_net():
    with LOCK:
        for n in NETS:
            if n["run"] in TAKEN or done(n) or not mine(n): continue
            if deps_ready(n): TAKEN.add(n["run"]); return n
        return None
def gpu_free():
    try: return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
    except Exception: return 0
def train_one(n):
    run_d = ROOT / n["run"]; tr = path(n["train"]); va = path(n["val"]); lg = L / f"train_{n['run'].replace('/', '_')}.log"
    for t in (1, 2, 3):
        while gpu_free() < 6000 and not DRY:
            if STOP.is_set(): return False
            time.sleep(20)
        say(f"training {n['run']} (attempt {t})")
        run([PY, "-m", "qmi.clf", "train", "--rep", n["rep"], "--seed", n["seed"], "--train", tr, "--val", va, "--run", run_d, "--epochs", EPOCHS], lg)
        if DRY or (run_d / "history.json").exists(): say("TRAIN_DONE", n["run"]); return True
        say("FAILED", n["run"], "waiting 90 s"); time.sleep(90)
    return False
def lane_tests():
    try:
        for t in TESTS: prepare(t)
        say("TESTS_EQ_READY")
    except Exception as e:
        fail(f"test lane: {e!r}\n{traceback.format_exc()[-1500:]}")

def lane_train(k):
    try:
        while not DRY and not base_queue_done():
            if STOP.is_set(): return
            time.sleep(60)
        while not STOP.is_set():
            if all(done(n) for n in NETS if mine(n)) or (DRY and len(TAKEN) == sum(map(mine, NETS))): break
            if not disk_ok(): return
            n = next_net()
            if n is None: time.sleep(30); continue
            if not train_one(n): fail(f"{n['run']} not trained after 3 attempts"); return
            evict()
        say(f"training lane {k}: finished")
    except Exception as e:
        fail(f"training lane {k}: {e!r}\n{traceback.format_exc()[-1500:]}")

# ---------------- evaluations ----------------
R = plan.R
def nets(rx): return [n["run"] for n in NETS if re.search(rx, n["run"])]
def wait_for(runs, tests=()):
    while True:
        if STOP.is_set(): raise RuntimeError("stop")
        if all((ROOT / r / "history.json").exists() for r in runs) and all(ready(t) for t in tests): return
        if DRY: return
        time.sleep(60)
def ev(out, mark, cmd, runs, tests):
    out = ROOT / out
    if out.exists(): return
    wait_for(runs, tests); lg = L / f"eval_{out.parent.name}_{out.stem}.log"; say("evaluation", out)
    if run(cmd + ["--out", f"{out}.tmp.json"], lg) or (not DRY and mark not in lg.read_text()): raise RuntimeError(f"evaluation {out} (log {lg})")
    if not DRY: os.replace(f"{out}.tmp.json", out)
    say("ready", out)
def jsplit(src, dst, rx):
    src, dst = ROOT / src, ROOT / dst
    if dst.exists() or DRY: return
    d = json.load(open(src)); o = {k: v for k, v in d.items() if re.search(rx, k)}
    if not o: raise RuntimeError(f"empty subset {dst}")
    json.dump(o, open(str(dst) + ".tmp", "w")); os.replace(str(dst) + ".tmp", dst); say("split", dst.name, sorted(o))
def clf_eval(run_name, test, outname):
    o = ROOT / run_name / outname
    if o.exists(): return
    wait_for([run_name], [test]); lg = ROOT / run_name / f"eval_{outname.replace('.json', '')}.log"
    if run([PY, "-m", "qmi.clf", "eval", "--run", run_name, "--test", path(test), "--out", outname], lg) or (not DRY and not o.exists()):
        raise RuntimeError(f"clf eval {run_name} {outname}")
def clf_evals(rx):
    for n in NETS:
        if re.search(rx, n["run"]):
            for t, o in n["evals"]: clf_eval(n["run"], t, o)
SH = lambda runs, test, *a: [PY, "-m", "qmi.eval_shots", "--workers", WK, "--runs", runs, "--test", path(test), *a]
DR = lambda runs, test, *a: [PY, "-m", "qmi.eval_drift", "--workers", WK, "--runs", runs, "--test", path(test), *a]
def scores(runs, test):
    if (ROOT / runs / "scores.json").exists(): return
    wait_for(nets(rf"^{runs}/clf_._(exact|shots)_s"), [test]); lg = L / f"eval_{runs}_scores.log"; say("evaluation", runs, "scores")
    if run([PY, "-m", "qmi.eval_scores", "--runs", runs, "--match", "_(exact|shots)_s", "--test", path(test), "--out", f"{runs}/scores", "--workers", WK], lg) \
       or (not DRY and "EVAL_SCORES_DONE" not in lg.read_text()): raise RuntimeError(f"scores {runs}")
def post(script, *a):
    lg = L / f"post_{Path(script).stem}.log"; say("final step", script, *a)
    if DRY: return
    if run([PY, script, *a], lg): raise RuntimeError(f"{script} (log {lg})")

def step(mark, cmd, lgname):
    """Step with its own marker in the log (skipped when the marker is already there)."""
    lg = L / lgname
    # a marker counts only in a log newer than _markers/EPOCH (logs of an older data epoch cannot
    # skip a step); without the EPOCH file no marker is accepted
    ep = M / "EPOCH"
    if mark and lg.exists() and mark in lg.read_text() and ep.exists() and lg.stat().st_mtime > ep.stat().st_mtime: return
    say("step", lgname)
    if run(cmd, lg) or (not DRY and mark and mark not in lg.read_text()): raise RuntimeError(f"{lgname}: no {mark}")

def lane_bl():
    """Non-network baselines (local); _bl_ho/history.json releases the ho dataset from the cache."""
    try:
        def bl(out, tr, te, mod="baseline"):
            o = ROOT / R / out
            if o.exists(): return
            wait_for([], [tr, te]); lg = L / f"{Path(out).stem}.log"; say("baseline", out)
            extra = ["--jobs", 12] if mod == "baseline_cv" else []
            if run([PY, "-m", f"qmi.{mod}", "--train", path(tr), "--test", path(te), "--out", f"{o}.tmp.json", *extra], lg) or (not DRY and not Path(f"{o}.tmp.json").exists()):
                raise RuntimeError(f"baseline {out} (log {lg})")
            if not DRY: os.replace(f"{o}.tmp.json", o)
        if AHM:
            for nm, tr in (("hoS", "ahm_hoS"), ("ho", "ahm_ho")):
                for sp in ("in", "out"): bl(f"baseline_{nm}_{sp}.json", f"{tr}/train/mixed", f"ahm_ho_{sp}/test/sweep")
        else:
            bl("baseline_ho_in.json", "ho/train/mixed", "ho_in/test/sweep"); bl("baseline_ho_out.json", "ho/train/mixed", "ho_out/test/sweep")
        (ROOT / R / "_bl_ho").mkdir(parents=True, exist_ok=True); (ROOT / R / "_bl_ho/history.json").write_text("[]")
        bl("baseline.json", f"{B}/train/mixed", f"{B}/test/sweep"); bl("baseline_cv.json", f"{B}/train/mixed", f"{B}/test/sweep", "baseline_cv")
        say("BASELINES_READY")
    except Exception as e:
        fail(f"baseline lane: {e!r}\n{traceback.format_exc()[-1500:]}")

def lane_eval_ahm():
    try:
        A = "ahm/test/sweep"
        clf_evals(rf"^{R}/clf_._exact_")
        ev(f"{R}/eval_shots.json", "EVAL_SHOTS_DONE", SH(R, A, "--match", "re:_(exact|shots)_s[0-9]$"), nets(rf"^{R}/clf_._(exact|shots)_s"), [A])
        clf_evals(rf"^{R}/clf_._shots_s")
        ev(f"{R}/eval_drift.json", "EVAL_DRIFT_DONE", DR(R, A, "--match", "_shots_s"), nets(rf"^{R}/clf_._shots_s"), [A])
        scores(R, A)
        wait_for(nets(rf"^{R}/clf_._(exact|shots)_s"), [A])
        step("PAIRS_DONE", [PY, "-m", "qmi.x8.pairs", "--runs", R, "--test", path(A), "--ref", f"{R}/eval_shots.json"], "pairs.log")
        ev(f"{R}/eval_drift_trained.json", "EVAL_DRIFT_DONE", DR(R, A, "--match", "_drift_s"), nets(rf"^{R}/clf_._drift_s"), [A])
        ev(f"{R}/eval_shots_K12_all.json", "EVAL_SHOTS_DONE", SH(R, A, "--K", 12, "--match", "shotsK12_s"), nets(rf"^{R}/clf_W_shotsK12_"), [A])
        for s_ in range(3): jsplit(f"{R}/eval_shots_K12_all.json", f"{R}/eval_shots_K12_s{s_}.json", rf"shotsK12_s{s_}$")
        ev(f"{R}/eval_shots_sino_s0-4.json", "EVAL_SHOTS_DONE", SH(R, A, "--mode", "sino", "--match", "_sino_"), nets(rf"^{R}/clf_W_sino_"), [A])
        jsplit(f"{R}/eval_shots_sino_s0-4.json", f"{R}/eval_shots_sino.json", r"_sino_s[012]$")
        step("PAIRS_DONE", [PY, "-m", "qmi.x8.pairs", "--runs", R, "--test", path(A), "--match", r"re:^clf_Q_shots_s\d+$", "--sino", r"re:^clf_W_sino_s\d+$",
                           "--out", "pairs_sino", "--ref", f"{R}/eval_shots.json", f"{R}/eval_shots_sino_s0-4.json"], "pairs_sino.log")
        clf_evals(rf"^{R}/clf_._hoexactS_")
        for sp in ("in", "out"):
            ev(f"{R}/eval_shots_hoS_{sp}.json", "EVAL_SHOTS_DONE", SH(R, f"ahm_ho_{sp}/test/sweep", "--match", "re:_hoshotsS_s"), nets(rf"^{R}/clf_._hoshotsS_"), [f"ahm_ho_{sp}/test/sweep"])
        clf_evals(rf"^{R}/clf_._narrow_"); clf_evals(rf"^{R}/clf_._hoexact_s")
        for sp in ("in", "out"):
            ev(f"{R}/eval_shots_ho_{sp}.json", "EVAL_SHOTS_DONE", SH(R, f"ahm_ho_{sp}/test/sweep", "--match", "re:_hoshots_s"), nets(rf"^{R}/clf_._hoshots_s"), [f"ahm_ho_{sp}/test/sweep"])
        wait_for([], ["ahm/train/mixed", A])
        step("ENERGY_CONTROL_DONE", [PY, "-m", "qmi.energy_control", "--train", path("ahm/train/mixed"), "--test", path(A), "--out", f"{R}/energy_control",
                                     "--workers", WK], "energy_control.log")
        while not (ROOT / R / "baseline_cv.json").exists():
            if STOP.is_set(): raise RuntimeError("stop")
            if DRY: break
            time.sleep(60)
        say("NETWORK_EVALUATIONS_READY")
        post("qmi/eq/finish_eq.py")
        say("EQ_AHM_DONE")
    except Exception as e:
        fail(f"evaluation lane: {e!r}\n{traceback.format_exc()[-1500:]}")

def lane_eval():
    try:
        Q = "qmi/test/sweep"
        clf_evals(rf"^{R}/clf_._exact_"); clf_evals(rf"^{R}/clf_._cvexact_")
        ev(f"{R}/eval_shots_all_s0-4.json", "EVAL_SHOTS_DONE", SH(R, Q, "--match", "re:_(exact|shots)_s[0-9]$"), nets(rf"^{R}/clf_._(exact|shots)_s"), [Q])
        jsplit(f"{R}/eval_shots_all_s0-4.json", f"{R}/eval_shots.json", r"_(exact|shots)_s[012]$")
        jsplit(f"{R}/eval_shots_all_s0-4.json", f"{R}/eval_shots_s3.json", r"_shots_s3$")
        jsplit(f"{R}/eval_shots_all_s0-4.json", f"{R}/eval_shots_s4.json", r"_shots_s4$")
        wait_for([], ["qmi/train/mixed", Q])
        step("ENERGY_CONTROL_DONE", [PY, "-m", "qmi.energy_control", "--train", path("qmi/train/mixed"), "--test", path(Q), "--out", f"{R}/energy_control",
                                     "--workers", WK], "energy_control.log")
        clf_evals(rf"^{R}/clf_._shots_s")
        ev(f"{R}/eval_drift.json", "EVAL_DRIFT_DONE", DR(R, Q, "--match", "_shots_s"), nets(rf"^{R}/clf_._shots_s"), [Q])
        scores(R, Q)
        wait_for(nets(rf"^{R}/clf_._shots_s"), [Q])
        step("PAIRS_DONE", [PY, "-m", "qmi.x8.pairs", "--runs", R, "--test", path(Q), "--ref", f"{R}/eval_shots_all_s0-4.json"], "pairs.log")
        ev(f"{R}/eval_drift_trained.json", "EVAL_DRIFT_DONE", DR(R, Q, "--match", "_drift_s"), nets(rf"^{R}/clf_._drift_s"), [Q])
        ev(f"{R}/eval_shots_K12_all.json", "EVAL_SHOTS_DONE", SH(R, Q, "--K", 12, "--match", "shotsK12_s"), nets(rf"^{R}/clf_W_shotsK12_"), [Q])
        jsplit(f"{R}/eval_shots_K12_all.json", f"{R}/eval_shots_K12.json", r"shotsK12_s0$")
        jsplit(f"{R}/eval_shots_K12_all.json", f"{R}/eval_shots_K12_s1.json", r"shotsK12_s1$")
        jsplit(f"{R}/eval_shots_K12_all.json", f"{R}/eval_shots_K12_s2.json", r"shotsK12_s2$")
        ev(f"{R}/eval_shots_sino_s0-4.json", "EVAL_SHOTS_DONE", SH(R, Q, "--mode", "sino", "--match", "_sino_"), nets(rf"^{R}/clf_W_sino_"), [Q])
        jsplit(f"{R}/eval_shots_sino_s0-4.json", f"{R}/eval_shots_sino.json", r"_sino_s[012]$")
        step("PAIRS_DONE", [PY, "-m", "qmi.x8.pairs", "--runs", R, "--test", path(Q), "--match", r"re:^clf_Q_shots_s\d+$", "--sino", r"re:^clf_W_sino_s\d+$",
                           "--out", "pairs_sino", "--ref", f"{R}/eval_shots_all_s0-4.json", f"{R}/eval_shots_sino_s0-4.json"], "pairs_sino.log")
        clf_evals(rf"^{R}/clf_._hoexact_")
        for sp in ("in", "out"):
            ev(f"{R}/eval_shots_ho_{sp}.json", "EVAL_SHOTS_DONE", SH(R, f"ho_{sp}/test/sweep", "--match", "re:_hoshots_s"), nets(rf"^{R}/clf_._hoshots_"), [f"ho_{sp}/test/sweep"])
        clf_evals(rf"^{R}/clf_._narrow_")
        # cross-validation on records and the scan of K last (these networks are trained last)
        ev(f"{R}/eval_shots_cv5.json", "EVAL_SHOTS_DONE", SH(R, Q, "--match", "cvshots_f"), nets(rf"^{R}/clf_._cvshots_"), [Q])
        for K in (6, 18, 24, 72):
            ev(f"{R}/eval_shots_Kscan_K{K}.json", "EVAL_SHOTS_DONE", SH(R, Q, "--K", K, "--match", f"shotsK{K}_s"), nets(rf"^{R}/clf_W_shotsK{K}_"), [Q])
        jsplit(f"{R}/eval_shots_Kscan_K18.json", f"{R}/eval_shots_K18.json", r"shotsK18_s0$")
        while not (ROOT / R / "baseline_cv.json").exists() or not (ROOT / R / "limit_loss_dim48.json").exists():
            if STOP.is_set(): raise RuntimeError("stop")
            if DRY: break
            time.sleep(60)
        say("NETWORK_EVALUATIONS_READY")
        post("qmi/eq/finish_eq.py")
        say("EQ_DONE")
    except Exception as e:
        fail(f"evaluation lane: {e!r}\n{traceback.format_exc()[-1500:]}")

if __name__ == "__main__":
    lf = open(M / "queue.lock", "w")
    try: fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: print(ts(), "queue already running (lock)"); sys.exit(1)
    if (M / "FAILED_queue").exists(): os.replace(M / "FAILED_queue", M / f"FAILED_queue.prev_{time.strftime('%Y%m%d-%H%M%S')}"); say("FAILED marker of the interrupted run moved aside, resuming")
    say("start of the eq queue, PID", os.getpid(), "networks", len(NETS), "done", sum(done(n) for n in NETS), "datasets", len(ORDER), "CACHE", CACHE, f"{CACHE_GB} GB")
    th = ([threading.Thread(target=lane_data, name="data")] if "data" in LANES else []) \
       + ([threading.Thread(target=lane_train, args=(1,)), threading.Thread(target=lane_train, args=(2,))] if "train" in LANES else []) \
       + ([threading.Thread(target=lane_eval_ahm if AHM else lane_eval, name="eval")] if "eval" in LANES else []) \
       + ([threading.Thread(target=lane_tests, name="tests")] if "tests" in LANES else []) + ([threading.Thread(target=lane_bl, name="baselines")] if "baselines" in LANES else [])
    say("lanes:", sorted(LANES), "TRAIN_RX:", TRAIN_RX or "(all)", "GEN_WAIT_RX:", GEN_WAIT_RX or "(none)", "networks of this machine:", sum(map(mine, NETS)))
    for t in th: t.start()
    for t in th: t.join()
    if STOP.is_set(): say("STOPPED WITH AN ERROR (FAILED_queue)"); sys.exit(1)
    say("QUEUE_EQ_FINISHED")
