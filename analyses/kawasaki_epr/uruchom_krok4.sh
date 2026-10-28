#!/bin/bash
# Krok 4 dla wszystkich zestawow: K=6 (A, B; 200 losowan) i K=36 (A, B; 100 losowan). Wznawialne, stop na pierwszym bledzie.
set -euo pipefail
cd /tmp
export CUDA_VISIBLE_DEVICES= QMI_XMAX=12 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PY:-python}; S="$(dirname "$(readlink -f "$0")")"/krok4_klasyf.py
$PY -I $S --K 6 --wariant A --draws 200 --workers 8
$PY -I $S --K 6 --wariant B --draws 200 --workers 8
$PY -I $S --K 36 --wariant A --draws 100 --workers 8
$PY -I $S --K 36 --wariant B --draws 100 --workers 8
echo URUCHOM_KROK4_DONE
