#!/bin/bash
# Siec W K=6 dla zbioru Ahmeda (eq) - brak w runs_ahm_eq; dane i sieci tylko w analyses/kawasaki.
# Ten sam przepis co runs_eq/clf_W_shotsK6 (qmi.shots_data --copies 2 --K 6, qmi.clf train, 25 epok). Wznawialne: shots_data pomija gotowe pliki,
# trening pomijany, gdy jest history.json. Stop na pierwszym bledzie.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."   # repository root (QMI_ROOT layout: data_eq/, runs_eq/ here)
PY=${PY:-python}
export QMI_XMAX=12 QMI_NPZ_COMPRESS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD:${PYTHONPATH:-}
A=analyses/kawasaki
for sp in val train; do
  $PY -m qmi.shots_data --src data_eq_ahm/ahm/$sp/mixed --dst $A/cache_ahmK6/$sp --copies 2 --K 6 --workers ${WK:-8} > $A/logs/ahmK6_data_$sp.log 2>&1
  grep -q SHOTS_DATA_DONE $A/logs/ahmK6_data_$sp.log || { echo "BLAD dane $sp"; exit 1; }
done
for s in 0 1 2; do
  R=$A/runs_ahmK6/clf_W_shotsK6_s$s
  [ -f $R/history.json ] && continue
  OMP_NUM_THREADS=2 $PY -m qmi.clf train --rep W --seed $s --train $A/cache_ahmK6/train --val $A/cache_ahmK6/val --run $R --epochs 25 > $A/logs/ahmK6_train_s$s.log 2>&1
  [ -f $R/history.json ] || { echo "BLAD trening s$s"; exit 1; }
done
echo AHM_K6_DONE
