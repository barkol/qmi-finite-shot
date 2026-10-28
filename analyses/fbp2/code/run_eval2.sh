#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")"; QMI_ROOT="${QMI_ROOT:-$HOME/qmi-work}"; export QMI_XMAX=12 PYTHONPATH=. OMP_NUM_THREADS=1
P=${PY:-python}; T=$QMI_ROOT/data_eq/qmi/test/sweep
while [ $(ls ../runs/*/history.json | wc -l) -lt 10 ]; do sleep 20; done
for v in kak2 hann2; do for c in loss:0 loss:50 loss:80 dephase:20; do echo "$v $c"; done; done | xargs -P 8 -L1 bash -c '$0 -m qmi.pary2 --variant $1 --test '"$T"' --runsnew ../runs --runsold $QMI_ROOT/runs_eq --out ../results/pary2 --cells $2' $P || exit 1
$P -m qmi.analiza2 --old $QMI_ROOT/runs_eq/pary --new ../results/pary2 --variants kak2,hann2 --out ../results/results_full.json > ../results/results_full.txt && echo EVAL_ALL_DONE
