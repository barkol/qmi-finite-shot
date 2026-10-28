"""Kolejka stacji roboczej dla zbioru bazowego z rowna <n> (rodzina eqn, okno [-12,12]^2, 128x128).
Wszystko na stacji roboczej (bez serwera obliczeniowego): zbiory dokladne generowane lokalnie (qmi.gen --family eqn + straznik masy, xmax=12), rekordy,
81 sieci (qmi/eq/plan_eq.py), ewaluacje (tylko czesc bazowa, sino 5 ziaren, pary Q-W i Q-sino, kontrola energetyczna), krok koncowy
qmi/eq/koniec_eq.py. Trzy tory (dane, trening, ewaluacje); stop na pierwszym bledzie (data_eq/_markers/FAILED_kolejka).
QEQ_SET=ahm: zbior Ahmeda (szum eq, okno 12): data_eq_ahm/, runs_ahm_eq/, log logs/eq_ahm/; tory treningu
czekaja, az kolejka bazowa wytrenuje swoje 81 sieci (RAM i GPU), dane i bazy niesieciowe licza sie od razu.
Praca rozproszona: QEQ_TRAIN_RX = regex sieci trenowanych NA TEJ maszynie (pozostale tylko czekaja na
history.json, dostarczane z zewnatrz; zbiory potrzebne tylko im nie sa przygotowywane); QEQ_LANES = tory do uruchomienia
(dane,trening,ewaluacje,testy,bazy; domyslnie wszystkie); QEQ_GEN_WAIT_RX = regex id zbiorow dokladnych, ktore przychodza z zewnatrz
(czekamy na znacznik i liczbe plikow zamiast generowac).
Start/wznowienie: [QEQ_SET=ahm] setsid nohup ~/anaconda3/envs/ml/bin/python -m qmi.eq.kolejka_eq >> logs/eq/kolejka.log 2>&1 < /dev/null & disown"""
import os, sys, time, json, glob, shutil, subprocess, threading, fcntl, re, traceback
from pathlib import Path

ROOT = Path(os.environ.get("QMI_ROOT", Path.home() / "work/qsdl-qmi")); os.chdir(ROOT); sys.path.insert(0, str(ROOT))   # QMI_ROOT: working dir with data/runs
from qmi.eq import plan_eq as plan
PY = os.environ.get("QMI_PY", sys.executable)
AHM = plan.AHM
PERM = ROOT / ("data_eq_ahm" if AHM else "data_eq"); CACHE = Path(os.environ.get("QX_CACHE", "/dev/shm/qmi-eqa-cache" if AHM else "/dev/shm/qmi-eq-cache")); M = PERM / "_markers"
L = ROOT / ("logs/eq_ahm" if AHM else "logs/eq"); XM = 12
CACHE_GB = float(os.environ.get("QX_CACHE_GB", "11")); WK = int(os.environ.get("QX_WK", "20")); EPOCHS = 25
DRY = bool(os.environ.get("QX_DRY"))
TRAIN_RX = os.environ.get("QEQ_TRAIN_RX", ""); GEN_WAIT_RX = os.environ.get("QEQ_GEN_WAIT_RX", "")
LANES = set(os.environ.get("QEQ_LANES", "dane,trening,ewaluacje,testy,bazy").split(","))       # test: tylko wypisuje kroki, nic nie liczy
for d in (PERM, CACHE, M, L): d.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, QMI_XMAX=str(XM), QMI_NPZ_COMPRESS="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONPATH=str(ROOT))
LOCK = threading.Lock(); EVLOCK = threading.Lock(); STOP = threading.Event()
def ts(): return time.strftime("%H:%M:%S")
def pisz(*a):
    with LOCK: print(ts(), *a, flush=True)
def blad(msg):
    pisz("BLAD:", msg); (M / "FAILED_kolejka").write_text(ts() + " " + msg + "\n"); STOP.set()

# ---------------- zbiory ----------------
TESTS = plan.TESTS
B = "ahm" if AHM else "qmi"
PERM_IDS = set(TESTS) | {f"{B}/train/mixed", f"{B}/val/mixed", f"{B}_narrow/train/mixed", f"{B}_narrow/val/mixed"}
def disk_ok():
    st = os.statvfs(ROOT); free = st.f_bavail * st.f_frsize / 1e9
    if free < 10: blad(f"za malo miejsca na /home: {free:.1f} GB < 10 GB"); return False
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
    src = plan.STAGE_BY_ID[st["src"]]; return None   # rekordy: liczba plikow zrodla bez kanalu gauss (liczona po fakcie)
def run(cmd, log, env=ENV):
    if DRY: pisz("[DRY]", " ".join(map(str, cmd))); return 0
    with open(log, "a") as f:
        f.write(f"\n### {ts()} {' '.join(map(str, cmd))}\n"); f.flush()
        return subprocess.run(list(map(str, cmd)), env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
def gb(p): return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file() and not f.is_symlink()) / 1e9 if Path(p).exists() else 0.0

READY = {}   # id -> True (zweryfikowany w tym procesie)
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
    """Przygotowuje zbior i (bez cv5). Zwraca po gotowosci albo podnosi wyjatek."""
    if ready(i): return
    if DRY: pisz("[DRY] przygotowanie", i, "->", path(i)); READY[i] = True; return
    st = plan.STAGE_BY_ID[i]; d = path(i); lg = L / f"dane_{tag(i)}.log"
    if st["kind"] == "gen" and GEN_WAIT_RX and re.search(GEN_WAIT_RX, i):   # zbior dostarczany z zewnatrz
        mk = M / (f"{tag(i)}.ok" if i in PERM_IDS else f"cache_{tag(i)}.ok"); pisz("czekam na dostawe:", i)
        while not (mk.exists() and nfiles(d) == st["expect"]):
            if STOP.is_set(): raise RuntimeError("stop")
            time.sleep(30)
        pisz("dostarczono", i, nfiles(d), "plikow")
    elif st["kind"] == "gen":                       # zbior dokladny generowany lokalnie
        base = (PERM if i in PERM_IDS else CACHE) / st["out"]
        if d.exists(): [f.unlink() for f in d.glob("*.tmp.npz")]
        pisz("generuje", i, "->", d)
        if run([PY, "-m", "qmi.gen", *st["args"], "--out", base, "--workers", WK], lg): raise RuntimeError(f"gen {i} (log {lg})")
        if nfiles(d) != st["expect"]: raise RuntimeError(f"gen {i}: plikow {nfiles(d)} != {st['expect']}")
        if run([PY, "-m", "qmi.x8.guard", "--dir", d, "--xmax", XM, "--expect", st["expect"], "--sample", 40, "--max-low-frac", 0.0], lg):
            raise RuntimeError(f"straznik {i} (log {lg})")
        (M / (f"{tag(i)}.ok" if i in PERM_IDS else f"cache_{tag(i)}.ok")).write_text(json.dumps({"n": st["expect"]}))
        pisz("gotowe", i, nfiles(d), "plikow", f"{gb(d):.2f} GB")
    else:                                         # rekordy liczone lokalnie ze zrodla
        prepare(st["src"]); src = path(st["src"])
        if d.exists(): [f.unlink() for f in d.glob("*.tmp.npz")]
        pisz("rekordy", i, "<-", st["src"])
        if run([PY, "-m", f"qmi.{st['mod']}", "--src", src, "--dst", d, "--workers", WK, *st["args"]], lg): raise RuntimeError(f"rekordy {i} (log {lg})")
        mark = "SINO_DATA_DONE" if st["mod"] == "sino_data" else "SHOTS_DATA_DONE"
        if not DRY and mark not in lg.read_text(): raise RuntimeError(f"rekordy {i}: brak {mark}")
        import numpy as np
        exp = sum(int(np.load(f)["channel"][0]) != 3 for f in sorted(glob.glob(str(src) + "/part_*.npz")))
        if nfiles(d) != exp or not exp: raise RuntimeError(f"rekordy {i}: plikow {nfiles(d)} != {exp}")
        (M / f"cache_{tag(i)}.ok").write_text(json.dumps({"n": exp}))
        pisz("gotowe", i, nfiles(d), "plikow", f"{gb(d):.2f} GB")
    READY[i] = True

def cv5(kind):
    """Podzial 5-krotny (dowiazania) jak w qmi.cv_folds; porownanie przydzialu plikow z data/cv5/<kind>."""
    if DRY: pisz("[DRY] cv5", kind); (M / f"dry_cv5_{kind}").write_text(""); return
    dst = path(f"cv5/{kind}/fold0/train").parents[1]
    okf = M / (f"cv5_{kind}.ok" if kind == "mixed" else f"cache_cv5_{kind}.ok")
    if okf.exists() and (dst / "folds.json").exists() and all((dst / f"fold{k}/val").exists() for k in range(5)): return
    for p in ("train", "val"): prepare(f"qmi/{p}/{kind}")
    lg = L / f"dane_cv5_{kind}.log"
    if run([PY, "-m", "qmi.cv_folds", "--train", path(f"qmi/train/{kind}"), "--val", path(f"qmi/val/{kind}"), "--dst", dst], lg): raise RuntimeError(f"cv5 {kind}")
    okf.write_text("{}"); pisz("cv5", kind, "gotowe")

def deps_ready(n):
    for p in (n["train"], n["val"]):
        if p.startswith("cv5/"):
            kind = p.split("/")[1]
            okf = M / ((f"cv5_{kind}.ok" if kind == "mixed" else f"cache_cv5_{kind}.ok") if not DRY else f"dry_cv5_{kind}")
            if not okf.exists() or not all(ready(x) for x in plan.deps(p)): return False
        elif not ready(p): return False
    return True

# ---------------- kolejnosc treningu (grupy wg danych) ----------------
GROUPS = [r"runs_eq/clf_._(exact|cvexact)_", r"runs_ahm_eq/clf_._exact_", r"runs_eq/clf_._(shots|cvshots)_[sf]", r"runs_ahm_eq/clf_._shots_s",
          r"_drift_s", r"runs_eq/clf_W_shotsK12_", r"runs_eq/clf_W_shotsK6_", r"runs_eq/clf_W_shotsK18_", r"runs_eq/clf_W_shotsK24_",
          r"runs_eq/clf_W_shotsK72_", r"runs_ahm_eq/clf_W_shotsK12_", r"_sino_", r"runs_eq/clf_._ho(exact|shots)_",
          r"runs_ahm_eq/clf_._ho(exact|shots)S_", r"gkpxK6", r"runs_eq/clf_._narrow_", r"runs_ahm_eq/clf_._narrow_", r"runs_ahm_eq/clf_._ho(exact|shots)_s"]
GROUPS = [g for g in os.environ.get("QEQ_GROUPS_FIRST", "").split(";") if g] + GROUPS   # grupy z priorytetem (regexy, w kolejnosci)
def gidx(n): return next(k for k, g in enumerate(GROUPS) if re.search(g, n["run"]))
NETS = sorted(plan.NETS, key=lambda n: (gidx(n), plan.NETS.index(n)))
assert len(NETS) == len(plan.NETS) == (54 if AHM else 81) and len({n["run"] for n in NETS}) == len(NETS)
def done(n): return (ROOT / n["run"] / "history.json").exists()
def mine(n): return not TRAIN_RX or bool(re.search(TRAIN_RX, n["run"]))

# zbiory w kolejnosci potrzeb (trening), z uzytkownikami (dla zwalniania pamieci podrecznej)
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
            pisz("zwalniam pamiec podreczna:", i, f"{gb(d):.2f} GB"); shutil.rmtree(d); READY[i] = False
            (M / f"cache_{tag(i)}.ok").unlink(missing_ok=True)
            if i.startswith("qmi/") and "shots" == i.split("/")[-1] and (CACHE / "cv5/shots").exists():
                shutil.rmtree(CACHE / "cv5/shots"); (M / "cache_cv5_shots.ok").unlink(missing_ok=True)

EST = {"mixed": 1.6, "sweep": 6.0}      # GB na polowe pary (train ~ 2.5, val ~ 0.5 dla mixed; rekordy ~1.8/0.4)
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
            # limit pamieci podrecznej (RAM): czekaj, az trening zwolni zbiory
            while not i in PERM_IDS and not ready(i) and gb(CACHE) + est(i) > CACHE_GB:
                evict()
                if STOP.is_set(): return
                time.sleep(30)
            prepare(i)
            if i == "qmi/val/mixed" or i == "qmi/train/mixed":
                if ready("qmi/train/mixed") and ready("qmi/val/mixed"): cv5("mixed")
            if i in ("qmi/train/shots", "qmi/val/shots") and ready("qmi/train/shots") and ready("qmi/val/shots"): cv5("shots")
        pisz("DANE_EQ_GOTOWE")
    except Exception as e:
        blad(f"tor danych: {e!r}\n{traceback.format_exc()[-1500:]}")

# ---------------- trening ----------------
def fix_split_over():
    if not AHM or os.environ.get("QEQ_NO_WAIT_BASE"): return True   # na instancji w chmurze nie czekamy na siec bazowe stacji roboczej
    from qmi.eq import plan_eq as _pb                          # kolejka bazowa: wszystkie jej sieci gotowe?
    base = [n["run"].replace("runs_ahm_eq", "runs_eq") for n in NETS]   # nazwy nieistotne; liczymy po katalogu runs_eq
    return len(list((ROOT / "runs_eq").glob("clf_*/history.json"))) >= 81
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
        pisz(f"trening {n['run']} (proba {t})")
        run([PY, "-m", "qmi.clf", "train", "--rep", n["rep"], "--seed", n["seed"], "--train", tr, "--val", va, "--run", run_d, "--epochs", EPOCHS], lg)
        if DRY or (run_d / "history.json").exists(): pisz("TRAIN_DONE", n["run"]); return True
        pisz("NIEPOWODZENIE", n["run"], "czekam 90 s"); time.sleep(90)
    return False
def lane_tests():
    try:
        for t in TESTS: prepare(t)
        pisz("TESTY_EQ_GOTOWE")
    except Exception as e:
        blad(f"tor testow: {e!r}\n{traceback.format_exc()[-1500:]}")

def lane_train(k):
    try:
        while not DRY and not fix_split_over():
            if STOP.is_set(): return
            time.sleep(60)
        while not STOP.is_set():
            if all(done(n) for n in NETS if mine(n)) or (DRY and len(TAKEN) == sum(map(mine, NETS))): break
            if not disk_ok(): return
            n = next_net()
            if n is None: time.sleep(30); continue
            if not train_one(n): blad(f"{n['run']} nie wytrenowany po 3 probach"); return
            evict()
        pisz(f"tor treningu {k}: koniec")
    except Exception as e:
        blad(f"tor treningu {k}: {e!r}\n{traceback.format_exc()[-1500:]}")

# ---------------- ewaluacje ----------------
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
    wait_for(runs, tests); lg = L / f"eval_{out.parent.name}_{out.stem}.log"; pisz("ewaluacja", out)
    if run(cmd + ["--out", f"{out}.tmp.json"], lg) or (not DRY and mark not in lg.read_text()): raise RuntimeError(f"ewaluacja {out} (log {lg})")
    if not DRY: os.replace(f"{out}.tmp.json", out)
    pisz("gotowe", out)
def jsplit(src, dst, rx):
    src, dst = ROOT / src, ROOT / dst
    if dst.exists() or DRY: return
    d = json.load(open(src)); o = {k: v for k, v in d.items() if re.search(rx, k)}
    if not o: raise RuntimeError(f"pusty podzbior {dst}")
    json.dump(o, open(str(dst) + ".tmp", "w")); os.replace(str(dst) + ".tmp", dst); pisz("podzial", dst.name, sorted(o))
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
    wait_for(nets(rf"^{runs}/clf_._(exact|shots)_s"), [test]); lg = L / f"eval_{runs}_scores.log"; pisz("ewaluacja", runs, "scores")
    if run([PY, "-m", "qmi.eval_scores", "--runs", runs, "--match", "_(exact|shots)_s", "--test", path(test), "--out", f"{runs}/scores", "--workers", WK], lg) \
       or (not DRY and "EVAL_SCORES_DONE" not in lg.read_text()): raise RuntimeError(f"scores {runs}")
def post(script, *a):
    lg = L / f"post_{Path(script).stem}.log"; pisz("krok koncowy", script, *a)
    if DRY: return
    if run([PY, script, *a], lg): raise RuntimeError(f"{script} (log {lg})")

def step(mark, cmd, lgname):
    """Krok z wlasnym znacznikiem w logu (pomijany, gdy znacznik juz jest)."""
    lg = L / lgname
    # znacznik wazny tylko w logu nowszym niz _markers/EPOKA (logi z poprzedniej epoki danych nie
    # moga kazac pominac kroku); bez pliku EPOKA zaden znacznik nie wystarcza
    ep = M / "EPOKA"
    if mark and lg.exists() and mark in lg.read_text() and ep.exists() and lg.stat().st_mtime > ep.stat().st_mtime: return
    pisz("krok", lgname)
    if run(cmd, lg) or (not DRY and mark and mark not in lg.read_text()): raise RuntimeError(f"{lgname}: brak {mark}")

def lane_bl():
    """Bazy niesieciowe (lokalnie); _bl_ho/history.json zwalnia zbior ho z pamieci podrecznej."""
    try:
        def bl(out, tr, te, mod="baseline"):
            o = ROOT / R / out
            if o.exists(): return
            wait_for([], [tr, te]); lg = L / f"{Path(out).stem}.log"; pisz("baza", out)
            extra = ["--jobs", 12] if mod == "baseline_cv" else []
            if run([PY, "-m", f"qmi.{mod}", "--train", path(tr), "--test", path(te), "--out", f"{o}.tmp.json", *extra], lg) or (not DRY and not Path(f"{o}.tmp.json").exists()):
                raise RuntimeError(f"baza {out} (log {lg})")
            if not DRY: os.replace(f"{o}.tmp.json", o)
        if AHM:
            for nm, tr in (("hoS", "ahm_hoS"), ("ho", "ahm_ho")):
                for sp in ("in", "out"): bl(f"baseline_{nm}_{sp}.json", f"{tr}/train/mixed", f"ahm_ho_{sp}/test/sweep")
        else:
            bl("baseline_ho_in.json", "ho/train/mixed", "ho_in/test/sweep"); bl("baseline_ho_out.json", "ho/train/mixed", "ho_out/test/sweep")
        (ROOT / R / "_bl_ho").mkdir(parents=True, exist_ok=True); (ROOT / R / "_bl_ho/history.json").write_text("[]")
        bl("baseline.json", f"{B}/train/mixed", f"{B}/test/sweep"); bl("baseline_cv.json", f"{B}/train/mixed", f"{B}/test/sweep", "baseline_cv")
        pisz("BAZY_GOTOWE")
    except Exception as e:
        blad(f"tor baz: {e!r}\n{traceback.format_exc()[-1500:]}")

def lane_eval_ahm():
    try:
        A = "ahm/test/sweep"
        clf_evals(rf"^{R}/clf_._exact_")
        ev(f"{R}/eval_shots.json", "EVAL_SHOTS_DONE", SH(R, A, "--match", "re:_(exact|shots)_s[0-9]$"), nets(rf"^{R}/clf_._(exact|shots)_s"), [A])
        clf_evals(rf"^{R}/clf_._shots_s")
        ev(f"{R}/eval_drift.json", "EVAL_DRIFT_DONE", DR(R, A, "--match", "_shots_s"), nets(rf"^{R}/clf_._shots_s"), [A])
        scores(R, A)
        wait_for(nets(rf"^{R}/clf_._(exact|shots)_s"), [A])
        step("PARY_DONE", [PY, "-m", "qmi.x8.pary", "--runs", R, "--test", path(A), "--ref", f"{R}/eval_shots.json"], "pary.log")
        ev(f"{R}/eval_drift_trained.json", "EVAL_DRIFT_DONE", DR(R, A, "--match", "_drift_s"), nets(rf"^{R}/clf_._drift_s"), [A])
        ev(f"{R}/eval_shots_K12_all.json", "EVAL_SHOTS_DONE", SH(R, A, "--K", 12, "--match", "shotsK12_s"), nets(rf"^{R}/clf_W_shotsK12_"), [A])
        for s_ in range(3): jsplit(f"{R}/eval_shots_K12_all.json", f"{R}/eval_shots_K12_s{s_}.json", rf"shotsK12_s{s_}$")
        ev(f"{R}/eval_shots_sino_s0-4.json", "EVAL_SHOTS_DONE", SH(R, A, "--mode", "sino", "--match", "_sino_"), nets(rf"^{R}/clf_W_sino_"), [A])
        jsplit(f"{R}/eval_shots_sino_s0-4.json", f"{R}/eval_shots_sino.json", r"_sino_s[012]$")
        step("PARY_DONE", [PY, "-m", "qmi.x8.pary", "--runs", R, "--test", path(A), "--match", r"re:^clf_Q_shots_s\d+$", "--sino", r"re:^clf_W_sino_s\d+$",
                           "--out", "pary_sino", "--ref", f"{R}/eval_shots.json", f"{R}/eval_shots_sino_s0-4.json"], "pary_sino.log")
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
        pisz("EWALUACJE_SIECI_GOTOWE")
        post("qmi/eq/koniec_eq.py")
        pisz("EQ_AHM_DONE")
    except Exception as e:
        blad(f"tor ewaluacji: {e!r}\n{traceback.format_exc()[-1500:]}")

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
        step("PARY_DONE", [PY, "-m", "qmi.x8.pary", "--runs", R, "--test", path(Q), "--ref", f"{R}/eval_shots_all_s0-4.json"], "pary.log")
        ev(f"{R}/eval_drift_trained.json", "EVAL_DRIFT_DONE", DR(R, Q, "--match", "_drift_s"), nets(rf"^{R}/clf_._drift_s"), [Q])
        ev(f"{R}/eval_shots_K12_all.json", "EVAL_SHOTS_DONE", SH(R, Q, "--K", 12, "--match", "shotsK12_s"), nets(rf"^{R}/clf_W_shotsK12_"), [Q])
        jsplit(f"{R}/eval_shots_K12_all.json", f"{R}/eval_shots_K12.json", r"shotsK12_s0$")
        jsplit(f"{R}/eval_shots_K12_all.json", f"{R}/eval_shots_K12_s1.json", r"shotsK12_s1$")
        jsplit(f"{R}/eval_shots_K12_all.json", f"{R}/eval_shots_K12_s2.json", r"shotsK12_s2$")
        ev(f"{R}/eval_shots_sino_s0-4.json", "EVAL_SHOTS_DONE", SH(R, Q, "--mode", "sino", "--match", "_sino_"), nets(rf"^{R}/clf_W_sino_"), [Q])
        jsplit(f"{R}/eval_shots_sino_s0-4.json", f"{R}/eval_shots_sino.json", r"_sino_s[012]$")
        step("PARY_DONE", [PY, "-m", "qmi.x8.pary", "--runs", R, "--test", path(Q), "--match", r"re:^clf_Q_shots_s\d+$", "--sino", r"re:^clf_W_sino_s\d+$",
                           "--out", "pary_sino", "--ref", f"{R}/eval_shots_all_s0-4.json", f"{R}/eval_shots_sino_s0-4.json"], "pary_sino.log")
        clf_evals(rf"^{R}/clf_._hoexact_")
        for sp in ("in", "out"):
            ev(f"{R}/eval_shots_ho_{sp}.json", "EVAL_SHOTS_DONE", SH(R, f"ho_{sp}/test/sweep", "--match", "re:_hoshots_s"), nets(rf"^{R}/clf_._hoshots_"), [f"ho_{sp}/test/sweep"])
        ev(f"{R}/eval_shots_gkpxK6.json", "EVAL_SHOTS_DONE", SH(R, Q, "--K", 6, "--match", "gkpxK6_s"), nets(rf"^{R}/clf_W_gkpxK6_"), [Q])
        for nm, rx in (("gkpxK6", "gkpxK6_s"), ("K6", "shotsK6_s")):
            o = ROOT / R / f"exp_gkp_{nm}.json"
            if not o.exists():
                wait_for(nets(rf"^{R}/clf_W_{'gkpxK6' if nm == 'gkpxK6' else 'shotsK6'}_")); lg = L / f"exp_gkp_{nm}.log"; pisz("exp_gkp", nm)
                if run([PY, "-m", "qmi.exp_gkp", "--runs", R, "--match", rx, "--out", o], lg): raise RuntimeError(f"exp_gkp {nm}")
        for nm, rx in (("baza", r"re:^clf_W_shotsK6_s\d+$"), ("gkpx", r"re:^clf_W_gkpxK6_s\d+$")):
            wait_for(nets(rf"^{R}/clf_W_{'shotsK6' if nm == 'baza' else 'gkpxK6'}_"))
            if not (ROOT / R / "konno" / f"klasyf_{nm}.json").exists():
                step("", [PY, "-m", "qmi.x8.konno", "--runs", R, "--out", f"{R}/konno", "--siec", rx, "--nsieci", 3, "--klasyf-out", f"klasyf_{nm}.json", "--etapy", "kalib,iid,stac,dopas,ml,klasyf"], f"konno_{nm}.log")
                if not DRY and not (ROOT / R / "konno" / f"klasyf_{nm}.json").exists(): raise RuntimeError(f"konno {nm}: brak klasyf_{nm}.json")
        clf_evals(rf"^{R}/clf_._narrow_")
        # walidacja krzyzowa shots i skan K na koncu (te sieci trenuja sie ostatnie)
        ev(f"{R}/eval_shots_cv5.json", "EVAL_SHOTS_DONE", SH(R, Q, "--match", "cvshots_f"), nets(rf"^{R}/clf_._cvshots_"), [Q])
        for K in (6, 18, 24, 72):
            ev(f"{R}/eval_shots_Kscan_K{K}.json", "EVAL_SHOTS_DONE", SH(R, Q, "--K", K, "--match", f"shotsK{K}_s"), nets(rf"^{R}/clf_W_shotsK{K}_"), [Q])
        jsplit(f"{R}/eval_shots_Kscan_K18.json", f"{R}/eval_shots_K18.json", r"shotsK18_s0$")
        while not (ROOT / R / "baseline_cv.json").exists() or not (ROOT / R / "limit_loss_dim48.json").exists():
            if STOP.is_set(): raise RuntimeError("stop")
            if DRY: break
            time.sleep(60)
        pisz("EWALUACJE_SIECI_GOTOWE")
        post("qmi/eq/koniec_eq.py")
        pisz("EQ_DONE")
    except Exception as e:
        blad(f"tor ewaluacji: {e!r}\n{traceback.format_exc()[-1500:]}")

if __name__ == "__main__":
    lf = open(M / "kolejka.lock", "w")
    try: fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: print(ts(), "kolejka juz dziala (blokada)"); sys.exit(1)
    if (M / "FAILED_kolejka").exists(): os.replace(M / "FAILED_kolejka", M / f"FAILED_kolejka.prev_{time.strftime('%Y%m%d-%H%M%S')}"); pisz("poprzedni FAILED przeniesiony, wznawiam")
    pisz("start kolejki eq PID", os.getpid(), "sieci", len(NETS), "gotowe", sum(done(n) for n in NETS), "zbiory", len(ORDER), "CACHE", CACHE, f"{CACHE_GB} GB")
    th = ([threading.Thread(target=lane_data, name="dane")] if "dane" in LANES else []) \
       + ([threading.Thread(target=lane_train, args=(1,)), threading.Thread(target=lane_train, args=(2,))] if "trening" in LANES else []) \
       + ([threading.Thread(target=lane_eval_ahm if AHM else lane_eval, name="ewaluacje")] if "ewaluacje" in LANES else []) \
       + ([threading.Thread(target=lane_tests, name="testy")] if "testy" in LANES else []) + ([threading.Thread(target=lane_bl, name="bazy")] if "bazy" in LANES else [])
    pisz("tory:", sorted(LANES), "TRAIN_RX:", TRAIN_RX or "(wszystkie)", "GEN_WAIT_RX:", GEN_WAIT_RX or "(brak)", "moje sieci:", sum(map(mine, NETS)))
    for t in th: t.start()
    for t in th: t.join()
    if STOP.is_set(): pisz("KONIEC Z BLEDEM (FAILED_kolejka)"); sys.exit(1)
    pisz("KOLEJKA_EQ_KONIEC")
