#!/bin/bash
# generacja rekordow W (resumable: gotowe pliki pomijane)
cd "$(dirname "$(readlink -f "$0")")"; QMI_ROOT="${QMI_ROOT:-$HOME/qmi-work}"; export QMI_XMAX=12 PYTHONPATH=. OMP_NUM_THREADS=1
S=$QMI_ROOT/data_eq/qmi
for v in "$@"; do for sp in val train; do
  ${PY:-python} -m qmi.shots_data2 --src $S/$sp/mixed --dst ../data/$v/$sp --variant $v --workers 22 || exit 1
done; done; echo ALL_DATA_DONE
