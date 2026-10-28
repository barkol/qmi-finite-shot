#!/bin/bash
# Druga czesc kroku 4 (po skroceniu planu z powodu czasu): K=6 wariant B (wszystkie zestawy, 200 losowan),
# K=36 wariant B (5 zestawow, 100 losowan). Czeka na zakonczenie biegu K=6 A (PID z argumentu). Wznawialne, stop na bledzie.
set -euo pipefail
while kill -0 "$1" 2>/dev/null; do sleep 5; done
grep -q "KROK4_DONE K=6 wariant=A" "$(dirname "$(readlink -f "$0")")"/krok4.log || { echo "BLAD: K=6 A niedokonczone"; exit 1; }
cd /tmp
export CUDA_VISIBLE_DEVICES= QMI_XMAX=12 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PY:-python}; S="$(dirname "$(readlink -f "$0")")"/krok4_klasyf.py
$PY -I $S --K 6 --wariant B --draws 200 --workers 8
$PY -I $S --K 36 --wariant B --draws 100 --workers 8 --zestawy EPR1_pump1__050mW,EPR1_pump1__100mW,EPR1_pump1__150mW,EPR1_pump1__200mW,EPR3_loss__200mW_800mW_10dB
echo URUCHOM_KROK4B_DONE
