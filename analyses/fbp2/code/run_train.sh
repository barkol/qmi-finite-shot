#!/bin/bash
# usage: run_train.sh VARIANT SEED [SEED...]   (wznawialne: ckpt.pt co epoke, history.json = gotowe)
cd "$(dirname "$(readlink -f "$0")")"; QMI_ROOT="${QMI_ROOT:-$HOME/qmi-work}"; export QMI_XMAX=12 PYTHONPATH=. OMP_NUM_THREADS=2
v=$1; shift
for s in "$@"; do
  r=../runs/clf_W_${v}_s$s
  [ -f $r/history.json ] && continue
  need=2; [ $v = hann2 ] && need=4
  while [ $(grep -c SHOTS_DATA2_DONE ../logs/data.log) -lt $need ]; do sleep 30; done
  ${PY:-python} -m qmi.clf train --rep W --seed $s --train ../data/$v/train --val ../data/$v/val --run $r --epochs 25 || exit 1
done; echo TRAIN_DONE_$v
