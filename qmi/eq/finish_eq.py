"""Final step of the eq run (base set with equal <n>, window 12).
Derived files, sample counts and assembly of results_eq/ in the layout of paper/results (Ahmed set:
QEQ_SET=ahm). It does NOT run make_numbers and does not touch paper/. Stops on the first error. Checks: completeness with
respect to paper/results (file names), all results newer than the start of the run (the data epoch marker _markers/EPOCH).
   QMI_XMAX=12 python qmi/eq/finish_eq.py      (QEQ_SET=ahm: runs_ahm_eq -> results_ahm_eq, compared with paper/results_ahm)"""
import os, sys, json, glob, shutil, subprocess, time
from pathlib import Path
ROOT = Path(os.environ.get("QMI_ROOT", Path(__file__).resolve().parents[2])); os.chdir(ROOT); sys.path.insert(0, str(ROOT))   # QMI_ROOT: working dir with data/runs
PAPER = Path(os.environ.get("QMI_PAPER", Path(__file__).resolve().parents[2] / "paper")); PY = sys.executable
_EP = ROOT / ("data_eq_ahm" if os.environ.get("QEQ_SET", "base") == "ahm" else "data_eq") / "_markers/EPOCH"   # data epoch marker
if not _EP.exists(): raise SystemExit(f"missing {_EP}")
START = _EP.stat().st_mtime
AHM = os.environ.get("QEQ_SET", "base") == "ahm"
R, OR = (ROOT / "runs_ahm_eq", ROOT / "results_ahm_eq") if AHM else (ROOT / "runs_eq", ROOT / "results_eq")
def say(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
if os.environ.get("QMI_XMAX") != "12": raise SystemExit("ERROR: QMI_XMAX=12 not set")

# 1. derived files
from qmi.x8 import derived
derived.scores(str(R / "scores.json"), str(R / "scores_summary.json"))
if not AHM: derived.drift(str(R / "eval_drift.json"), str(R / "drift_perclass_base.json"))

# 2. sample counts: samples in the files (training files hold 25 samples each, so the contents are counted)
import numpy as np
c = lambda p: int(sum(len(np.load(f, allow_pickle=True)["label"]) for f in glob.glob(str(ROOT / "data_eq" / p / "part_*.npz"))))
if not AHM: json.dump({"n_train": c("qmi/train/mixed"), "n_val": c("qmi/val/mixed"), "n_test": c("qmi/test/sweep")}, open(R / "counts.json", "w"))

# 2b. check of the window and of the FBP error
if not (R / "fbp_window.json").exists():
    tst = ROOT / ("data_eq_ahm/ahm/test/sweep" if AHM else "data_eq/qmi/test/sweep")
    if subprocess.run([PY, "-m", "qmi.eq.fbp_window", "--test", str(tst), "--out", str(R / "fbp_window.json")], env=dict(os.environ, PYTHONPATH=str(ROOT))).returncode:
        raise SystemExit("fbp_window")

# 2c. effect of the reconstruction alone: W and S on the same events (qmi/eq/pairs_sw.py, checkpoint = one file per cell)
tst = ROOT / ("data_eq_ahm/ahm/test/sweep" if AHM else "data_eq/qmi/test/sweep")
if subprocess.run([PY, "-m", "qmi.eq.pairs_sw", "--runs", str(R), "--test", str(tst)], env=dict(os.environ, PYTHONPATH=str(ROOT))).returncode:
    raise SystemExit("pairs_sw")

# 3. assembly of results_eq/
OR.mkdir(exist_ok=True)
for f in R.glob("*.json"):
    if ".partial" in f.name or ".tmp" in f.name: continue
    shutil.copy2(f, OR / f.name)
for run in sorted(R.glob("clf_*")):
    for f in run.glob("eval*.json"): shutil.copy2(f, OR / f"{f.stem}_{run.name}.json")
for sub in ("pairs", "pairs_sino", "pairs_sw"):                       # paired correctness matrices (make_numbers: intervals of the differences)
    if (R / sub).exists(): shutil.copytree(R / sub, OR / sub, dirs_exist_ok=True)
if (R / "energy_control/energy_control.json").exists(): shutil.copy2(R / "energy_control/energy_control.json", OR / "energy_control.json")
say("assembled", OR, len(list(OR.glob("*.json"))), "files")

# 4. checks: completeness and times
SKIP = {"numbers.tex", "numbers_chernoff.tex", "numbers_fisher.tex", "numbers_holm.tex", "eval_drift_partial.json", "ci.json", "ci_ahm.json"}
PR = PAPER / ("results_ahm" if AHM else "results")
missing = [f.name for f in PR.iterdir() if f.is_file() and f.name not in SKIP and not (OR / f.name).exists()]
older = [str(f) for f in OR.rglob("*") if f.is_file() and f.stat().st_mtime < START]
json.dump({"missing": missing, "older_than_start": older, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(OR / "CHECK_EQ.json", "w"), indent=1)
# results not covered by the check of the names against paper/results
missing += [f for f in ["pairs", "pairs_sino", "pairs_sw", "energy_control.json"] + ([] if AHM else ["fbp_window.json", "limit_loss_dim48.json"])
         if not (OR / f).exists() or ((OR / f).is_dir() and not any((OR / f).iterdir()))]
if missing: raise SystemExit(f"files missing with respect to paper/results: {missing}")
if older: raise SystemExit(f"files older than the start of the run: {older[:10]}")
say("FINISH_EQ_OK: results_eq/ ready (make_numbers NOT run)")
