#!/bin/bash
# czeka na koniec treningu 5 ziaren obu wariantow, potem pary2 (wznawialne) i analiza
cd "$(dirname "$(readlink -f "$0")")"; QMI_ROOT="${QMI_ROOT:-$HOME/qmi-work}"; export QMI_XMAX=12 PYTHONPATH=. OMP_NUM_THREADS=2; P=${PY:-python}; T=$QMI_ROOT/data_eq/qmi/test/sweep
for v in kak2 hann2; do for s in 0 1 2 3 4; do while [ ! -f ../runs/clf_W_${v}_s$s/history.json ]; do sleep 30; done; done; done
for v in kak2 hann2; do $P -m qmi.pary2 --variant $v --test $T --runsnew ../runs --runsold $QMI_ROOT/runs_eq --out ../results/pary2 || exit 1; done
$P -m qmi.analiza2 --old $QMI_ROOT/runs_eq/pary --new ../results/pary2 --variants kak2,hann2 --out ../results/results_full.json > ../results/results_full.txt && echo EVAL_ALL_DONE
