#!/usr/bin/env bash
# Regenerate every numbers*.tex from the result files in the repository and compare them with the committed
# versions. Takes well under a minute; needs no GPU, no qsdl and no large data.
#
#   ./reproduce_numbers.sh                 (PY=<python with numpy+scipy> to override the interpreter)
#
# Steps
#   1. copies the committed numbers*.tex and ci*.json aside,
#   2. paper/scripts/make_numbers.py (base set, Ahmed set), paper/scripts/numbers_fisher.py,
#      analyses/chernoff/numbers_chernoff.py and paper/scripts/holm_rule.py write paper/results/numbers*.tex and ci*.json again,
#   3. compare with the copies: numbers*.tex byte for byte, ci*.json with relative tolerance 1e-9 (paper/scripts/json_close.py;
#      last-digit float differences between machines are not a failure); any difference = exit code 1.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
command -v "$PY" >/dev/null || { echo "ERROR: no python at $PY (set PY=...)"; exit 2; }
"$PY" -c "import numpy, scipy" || { echo "ERROR: $PY lacks numpy/scipy"; exit 2; }

RES=paper/results
REF="$(mktemp -d)"; trap 'rm -rf "$REF"' EXIT
FILES="numbers.tex numbers_chernoff.tex numbers_holm.tex numbers_fisher.tex ci.json ci_ahm.json"
for f in $FILES; do
  [ -f "$RES/$f" ] || { echo "ERROR: committed $RES/$f missing"; exit 2; }
  cp "$RES/$f" "$REF/$f"
done

echo "== 1/2 numbers from the result files"
( cd paper
  "$PY" scripts/make_numbers.py                                       > "$REF/log_base.txt"
  RES=results_ahm PFX=ahm_ OUTDIR=results "$PY" scripts/make_numbers.py > "$REF/log_ahm.txt"   # ci_ahm.json only
  "$PY" scripts/numbers_fisher.py                                     > "$REF/log_fisher.txt" )
export QMI_PAPER="$ROOT/paper"
"$PY" analyses/chernoff/numbers_chernoff.py   > "$REF/log_chernoff.txt"
"$PY" paper/scripts/holm_rule.py paper/results > "$REF/log_holm.txt"

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
  echo "RESULT: FAIL ($ndiff files differ)"; exit 1
fi
echo "RESULT: OK - numbers*.tex identical and ci*.json equal to rel. tol. 1e-9 with the committed versions"
