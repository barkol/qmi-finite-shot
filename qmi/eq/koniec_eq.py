"""Krok koncowy przeliczenia eq (zbior bazowy z rowna <n>, okno 12).
Pliki pochodne, liczebnosci, sekcja detekcji i zlozenie results_eq/ w ukladzie paper/results (Ahmed: QEQ_SET=ahm).
NIE uruchamia make_numbers i nie dotyka paper/. Stop na pierwszym bledzie. Kontrole: komplet wzgledem paper/results
(nazwy plikow), wszystkie wyniki nowsze niz start przeliczenia.
   QMI_XMAX=12 python qmi/eq/koniec_eq.py      (QEQ_SET=ahm: runs_ahm_eq -> results_ahm_eq, wzgledem paper/results_ahm)"""
import os, sys, json, glob, shutil, subprocess, time
from pathlib import Path
ROOT = Path(os.environ.get("QMI_ROOT", Path.home() / "work/qsdl-qmi")); os.chdir(ROOT); sys.path.insert(0, str(ROOT))   # QMI_ROOT: working dir with data/runs
PAPER = Path(os.environ.get("QMI_PAPER", Path(__file__).resolve().parents[2] / "paper")); PY = sys.executable
_EP = ROOT / ("data_eq_ahm" if os.environ.get("QEQ_SET", "base") == "ahm" else "data_eq") / "_markers/EPOKA"   # epoka danych (K2)
if not _EP.exists(): raise SystemExit(f"brak {_EP}")
START = _EP.stat().st_mtime
AHM = os.environ.get("QEQ_SET", "base") == "ahm"
R, OR = (ROOT / "runs_ahm_eq", ROOT / "results_ahm_eq") if AHM else (ROOT / "runs_eq", ROOT / "results_eq")
def pisz(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
if os.environ.get("QMI_XMAX") != "12": raise SystemExit("BLAD: brak QMI_XMAX=12")

# 1. pliki pochodne
from qmi.x8 import derived
derived.scores(str(R / "scores.json"), str(R / "scores_summary.json"))
if not AHM: derived.drift(str(R / "eval_drift.json"), str(R / "drift_perclass_base.json"))

# 2. liczebnosci: suma probek w plikach (pliki treningowe maja po 25 probek, wiec liczymy zawartosc plikow)
import numpy as np
c = lambda p: int(sum(len(np.load(f, allow_pickle=True)["label"]) for f in glob.glob(str(ROOT / "data_eq" / p / "part_*.npz"))))
if not AHM: json.dump({"n_train": c("qmi/train/mixed"), "n_val": c("qmi/val/mixed"), "n_test": c("qmi/test/sweep")}, open(R / "counts.json", "w"))

# 2b. kontrola okna i bledu FBP (tekst metod)
if not (R / "okno_fbp.json").exists():
    tst = ROOT / ("data_eq_ahm/ahm/test/sweep" if AHM else "data_eq/qmi/test/sweep")
    if subprocess.run([PY, "-m", "qmi.eq.okno_fbp", "--test", str(tst), "--out", str(R / "okno_fbp.json")], env=dict(os.environ, PYTHONPATH=str(ROOT))).returncode:
        raise SystemExit("okno_fbp")

# 2c. wplyw samej rekonstrukcji: W i S na tych samych zdarzeniach (qmi/eq/pary_sw.py, punkt kontrolny = plik na komorke)
tst = ROOT / ("data_eq_ahm/ahm/test/sweep" if AHM else "data_eq/qmi/test/sweep")
if subprocess.run([PY, "-m", "qmi.eq.pary_sw", "--runs", str(R), "--test", str(tst)], env=dict(os.environ, PYTHONPATH=str(ROOT))).returncode:
    raise SystemExit("pary_sw")

# 3. zlozenie results_eq/
OR.mkdir(exist_ok=True)
for f in R.glob("*.json"):
    if ".partial" in f.name or ".tmp" in f.name: continue
    shutil.copy2(f, OR / f.name)
for run in sorted(R.glob("clf_*")):
    for f in run.glob("eval*.json"): shutil.copy2(f, OR / f"{f.stem}_{run.name}.json")
for sub in ("pary", "pary_sino", "pary_sw"):                       # sparowane macierze poprawnosci (make_numbers: przedzialy roznic)
    if (R / sub).exists(): shutil.copytree(R / sub, OR / sub, dirs_exist_ok=True)
if (R / "konno").exists():
    shutil.copytree(R / "konno", OR / "konno", dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.log"))
    for f in ("eval_shots_gkpxK6.json", "eval_shots_Kscan_K6.json"): shutil.copy2(R / f, OR / "konno" / f)
if (R / "energy_control/energy_control.json").exists(): shutil.copy2(R / "energy_control/energy_control.json", OR / "energy_control.json")
pisz("zlozono", OR, len(list(OR.glob("*.json"))), "plikow")

# 4. sekcja detekcji (okno 12)
r = subprocess.CompletedProcess([], 0) if AHM else subprocess.run([PY, "-m", "qmi.x8.detekcja", str(OR / "detekcja")], env=dict(os.environ, PYTHONPATH=str(ROOT)))
if r.returncode: raise SystemExit("detekcja")

# 5. kontrole: komplet i czasy
SKIP = {"numbers.tex", "numbers_ahm.tex", "numbers_konno.tex", "numbers_chernoff.tex", "eval_drift_partial.json", "ci.json", "ci_ahm.json"}
PR = PAPER / ("results_ahm" if AHM else "results")
brak = [f.name for f in PR.iterdir() if f.is_file() and f.name not in SKIP and not (OR / f.name).exists()]
if not AHM: brak += [f"konno/{f.name}" for f in (PAPER / "results/konno").iterdir() if f.is_file() and not (OR / "konno" / f.name).exists()]
stare = [str(f) for f in OR.rglob("*") if f.is_file() and f.stat().st_mtime < START]
json.dump({"brak": brak, "starsze_niz_start": stare, "utworzono": time.strftime("%Y-%m-%d %H:%M:%S")}, open(OR / "KONTROLA_EQ.json", "w"), indent=1)
# wyniki, ktorych nie obejmuje kontrola nazw wzgledem paper/results
brak += [f for f in ["pary", "pary_sino", "pary_sw", "energy_control.json"] + ([] if AHM else ["okno_fbp.json", "konno", "detekcja", "limit_loss_dim48.json"])
         if not (OR / f).exists() or ((OR / f).is_dir() and not any((OR / f).iterdir()))]
if brak: raise SystemExit(f"brak plikow wzgledem paper/results: {brak}")
if stare: raise SystemExit(f"pliki starsze niz start przeliczenia: {stare[:10]}")
pisz("KONIEC_EQ_OK: results_eq/ gotowe (make_numbers NIE uruchamiany)")
