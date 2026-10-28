#!/usr/bin/env bash
# Regenerate every numbers*.tex from the result files in the repository and compare them with the committed
# versions. Takes about a minute; needs no GPU, no qsdl and no large data.
#
#   ./reproduce_numbers.sh                 (PY=<python with numpy+scipy> to override the interpreter)
#
# Steps
#   1. copies the committed numbers*.tex and ci*.json aside,
#   2. paper/scripts/make_numbers.py (base set, Ahmed set), make_numbers_konno.py, and the generators of analyses/
#      (chernoff, maxlik, arch2, kawasaki) write paper/results/numbers*.tex and ci*.json again,
#   3. compare with the copies: numbers*.tex byte for byte, ci*.json with relative tolerance 1e-9 (paper/scripts/json_close.py;
#      last-digit float differences between machines are not a failure); any difference = exit code 1.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/anaconda3/bin/python}"
command -v "$PY" >/dev/null || { echo "BLAD: no python at $PY (set PY=...)"; exit 2; }
"$PY" -c "import numpy, scipy" || { echo "BLAD: $PY lacks numpy/scipy"; exit 2; }

RES=paper/results
REF="$(mktemp -d)"; trap 'rm -rf "$REF"' EXIT
FILES="numbers.tex numbers_ahm.tex numbers_konno.tex numbers_chernoff.tex numbers_maxlik.tex numbers_arch2.tex numbers_kawasaki.tex numbers_shap.tex numbers_holm.tex numbers_kscan.tex numbers_robust.tex numbers_epr.tex numbers_fbp2.tex ci.json ci_ahm.json kscan.json robustness.json ci_coverage.json"
for f in $FILES; do
  [ -f "$RES/$f" ] || { echo "BLAD: committed $RES/$f missing"; exit 2; }
  cp "$RES/$f" "$REF/$f"
done

echo "== 1/2 numbers from the result files"
( cd paper
  "$PY" scripts/make_numbers.py                                       > "$REF/log_base.txt"
  RES=results_ahm PFX=ahm_ OUTDIR=results "$PY" scripts/make_numbers.py > "$REF/log_ahm.txt"
  RES=results "$PY" scripts/make_numbers_konno.py                     > "$REF/log_konno.txt" )
export QMI_PAPER="$ROOT/paper"
"$PY" analyses/chernoff/numbers_chernoff.py   > "$REF/log_chernoff.txt"
"$PY" analyses/maxlik/numbers_maxlik.py       > "$REF/log_maxlik.txt"
"$PY" analyses/arch2/numbers_arch2.py         > "$REF/log_arch2.txt"
"$PY" -I analyses/kawasaki/numbers_kawasaki.py > "$REF/log_kawasaki.txt"
"$PY" analyses/shap/numbers_shap.py > "$REF/log_shap.txt"
"$PY" paper/scripts/holm_rule.py paper/results > "$REF/log_holm.txt"
"$PY" paper/scripts/kscan_table.py > "$REF/log_kscan.txt"
"$PY" paper/scripts/robustness_headline.py > "$REF/log_robust.txt"
"$PY" paper/scripts/numbers_epr.py > "$REF/log_epr.txt"
"$PY" paper/scripts/fbp2_table.py analyses/fbp2/results_full.json analyses/fbp2/fbp_err.json > "$REF/log_fbp2.txt"
"$PY" paper/scripts/ci_coverage.py 20000 > "$REF/log_coverage.txt"     # about 80 s

echo "== 2/2 diff with the committed versions"
ndiff=0
for f in $FILES; do
  case "$f" in
    *.json) same() { "$PY" paper/scripts/json_close.py "$REF/$f" "$RES/$f" 1e-9 >/dev/null; } ;;   # floats: relative tolerance 1e-9
    *)      same() { cmp -s "$REF/$f" "$RES/$f"; } ;;                                            # numbers*.tex: byte for byte
  esac
  if same; then
    if cmp -s "$REF/$f" "$RES/$f"; then how=identical; else how="equal to rel. tol. 1e-9"; fi
    printf '  %-22s %s (%s lines)\n' "$f" "$how" "$(wc -l < "$RES/$f")"
  else
    n=$(diff "$REF/$f" "$RES/$f" | grep -c '^[<>]' || true)
    printf '  %-22s DIFFERS (%s changed lines)\n' "$f" "$n"; diff "$REF/$f" "$RES/$f" | head -20 || true
    ndiff=$((ndiff + 1))
  fi
done

if [ "$ndiff" -ne 0 ]; then
  echo "WYNIK: FAIL ($ndiff files differ)"; exit 1
fi
echo "WYNIK: OK - numbers*.tex identical and ci*.json equal to rel. tol. 1e-9 with the committed versions"
