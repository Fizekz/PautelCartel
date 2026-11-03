#!/usr/bin/env bash
# Reproduce every kavi-e output (v3) from the data files. Run from the worktree root:
#   bash src/models/kavi-e/run_all.sh            (uses the committed special-election CSVs)
#   FETCH=1 bash src/models/kavi-e/run_all.sh    (re-downloads The Downballot sheets first)
set -euo pipefail
PY=${PY:-python3}
M=src/models/kavi-e
if [ "${FETCH:-0}" = "1" ]; then "$PY" $M/fetch_specials.py; fi
"$PY" $M/specials.py
"$PY" $M/national.py
"$PY" $M/house_data.py > /dev/null
"$PY" $M/senate_data.py > /dev/null
"$PY" $M/independents.py > out/work/independents.txt
"$PY" $M/within_party.py > out/work/within_party.txt
KAVI_E_V3="" "$PY" $M/eval_loyo.py > out/work/selection_loyo_pre2018.txt 2>/dev/null   # v2 selection (pool off)
"$PY" $M/eval_v3.py > out/work/selection_v3.txt 2>/dev/null   # v3 selection (non-test LOYO)
"$PY" $M/backtest.py
"$PY" src/scorecard.py --pred 2018=out/backtest_2018.csv --pred 2022=out/backtest_2022.csv --name kavi-e --json out/scorecard.json
for v in no_rating with_prev; do
  "$PY" $M/backtest.py --variant $v > /dev/null
  "$PY" src/scorecard.py --pred 2018=out/backtest_2018_$v.csv --pred 2022=out/backtest_2022_$v.csv --name kavi-e-$v --json out/scorecard_$v.json
done
"$PY" $M/backtest.py > /dev/null   # leave the main variant's race files in out/work
"$PY" $M/predict_2026.py > /dev/null
"$PY" $M/simulate.py
