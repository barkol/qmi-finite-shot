#!/bin/bash
# Test drugiej architektury na wynajetej maszynie z GPU: dane rekordow -> tani test (z wznowieniem) -> 9 sieci r18 -> pary.
# Wznawialny: kazdy etap pomija gotowe (znaczniki / history.json / pliki pary). Stop na pierwszym bledzie.
#   setsid nohup bash analyses/arch2/run_cloud.sh >> analyses/arch2/logs/run_cloud.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."   # repository root (QMI_ROOT layout: data_eq/, runs_eq/ here)
export LC_ALL=C.UTF-8 QMI_XMAX=12 PYTHONPATH=$PWD:${PYTHONPATH:-} QMI_NPZ_COMPRESS=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}; A=analyses/arch2; D=$A/data/qmi; L=$A/logs; mkdir -p $D $L $A/runs
ts() { date +%H:%M:%S; }
fail() { echo "$(ts) BLAD: $*"; touch $A/FAILED; exit 1; }
trap 'fail "linia $LINENO"' ERR
rm -f $A/FAILED

# --- 1. rekordy (te same moduly, ziarna i argumenty co kolejka eq: shots --copies 2 [seed 31337, K 36], sino [seed 5150, copies 2])
nsrc() { $PY -c "import numpy as np,glob,sys;print(sum(int(np.load(f)['channel'][0])!=3 for f in glob.glob(sys.argv[1]+'/part_*.npz')))" "$1"; }
for sp in val train; do
  for kind in shots sino; do
    mk=$A/data/${sp}_${kind}.ok; [ -f $mk ] && continue
    src=data_eq/qmi/$sp/mixed; dst=$D/$sp/$kind; lg=$L/data_${sp}_${kind}.log
    echo "$(ts) rekordy $sp $kind"
    if [ $kind = shots ]; then $PY -m qmi.shots_data --src $src --dst $dst --copies 2 --workers 12 > $lg 2>&1; grep -q SHOTS_DATA_DONE $lg
    else $PY -m qmi.sino_data --src $src --dst $dst --workers 12 > $lg 2>&1; grep -q SINO_DATA_DONE $lg; fi
    n=$(ls $dst/part_*.npz | grep -vc tmp); e=$(nsrc $src); [ "$n" = "$e" ] || fail "$sp $kind: $n plikow != $e"
    echo "$n" > $mk; echo "$(ts) gotowe $sp $kind $n plikow"
  done
done

# --- 2. tani test: 2 epoki na podzbiorze, wznowienie do 3 epok, pary na 1 komorce (straznik zgodnosci v2)
if [ ! -f $A/test.ok ]; then
  T=$A/runs_test/r18_Q_shots_s0; rm -rf $A/runs_test $A/pary_test
  $PY $A/clf_arch2.py train --rep Q --seed 0 --train $D/train/shots --val $D/val/shots --run $T --epochs 2 --limit 2000 > $L/test_train.log 2>&1
  $PY $A/clf_arch2.py train --rep Q --seed 0 --train $D/train/shots --val $D/val/shots --run $T --epochs 3 --limit 2000 > $L/test_resume.log 2>&1
  grep -q "wznowiono od epoki 2" $L/test_resume.log || fail "wznowienie nie zadzialalo"
  [ "$($PY -c "import json;print(len(json.load(open('$T/history.json'))))")" = 3 ] || fail "history po wznowieniu != 3 epoki"
  $PY $A/pary_arch2.py --runs $A/runs_test --out $A/pary_test --max-cells 1 > $L/test_pary.log 2>&1; grep -q PARY_ARCH2_DONE $L/test_pary.log
  touch $A/test.ok; echo "$(ts) tani test OK"
fi
[ "${STOP_AFTER_TEST:-0}" = 1 ] && { echo "$(ts) STOP_AFTER_TEST"; exit 0; }

# --- 3. trening: trzy tory rownolegle na GPU0 (Q, W, S), po 3 ziarna
lane() {   # rep kind name
  for s in 0 1 2; do
    R=$A/runs/r18_$3_s$s; [ -f $R/history.json ] && continue
    echo "$(ts) trening $R"
    $PY $A/clf_arch2.py train --rep $1 --seed $s --train $D/train/$2 --val $D/val/$2 --run $R --epochs 25 >> $L/train_r18_$3_s$s.log 2>&1 || { echo "$(ts) BLAD treningu $R"; touch $A/FAILED; return 1; }
    echo "$(ts) gotowe $R"
  done
}
lane Q shots Q_shots & p1=$!; lane W shots W_shots & p2=$!; lane W sino W_sino & p3=$!
ok=1; for p in $p1 $p2 $p3; do wait $p || ok=0; done; [ $ok = 1 ] || fail "trening"

# --- 4. pary
$PY $A/pary_arch2.py --runs $A/runs --out $A/pary > $L/pary.log 2>&1; grep -q PARY_ARCH2_DONE $L/pary.log
echo "$(ts) WSZYSTKO_GOTOWE"
