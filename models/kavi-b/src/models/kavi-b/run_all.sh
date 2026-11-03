#!/usr/bin/env bash
# Reproduce every kavi-b output from the data files (run from anywhere).
# Network steps (optional, results are committed): fetch_fec_2018_snapshot.py --refresh, fetch_primaries.py --refresh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-python3}"
cd "$ROOT"
"$PY" src/models/kavi-b/fetch_fec_2018_snapshot.py          # uses cached data/kavi-b/raw/weball18_20181107.zip
"$PY" src/models/kavi-b/fetch_primaries.py                  # uses cached data/kavi-b/raw/wiki/*.json
"$PY" src/models/kavi-b/fetch_ca_polls_2026.py
"$PY" src/models/kavi-b/select_v2.py > /dev/null                # v2 design selection table (out/report/model_selection_v2.csv)
"$PY" src/models/kavi-b/select_v3.py > /dev/null                # v3 cycle-design table (out/report/model_selection_v3.csv)
"$PY" src/models/kavi-b/backtest.py --variants
"$PY" src/scorecard.py --pred 2018=out/backtest_2018.csv --pred 2022=out/backtest_2022.csv --name kavi-b --json out/scorecard.json | tee out/scorecard.txt
for v in v2_mid_mo bma_nofloor mid_gb_nofloor v1_like_const_floor const_nofloor mo_nofloor mid_mo_poll_sh ratings_only_mid_mo; do
  "$PY" src/scorecard.py --pred 2018=out/variants/${v}_2018.csv --pred 2022=out/variants/${v}_2022.csv --name $v
done > out/variants/scorecard_variants.txt
"$PY" src/models/kavi-b/predict_2026.py                     # reads data/dossiers/summary.csv (late-info pool)
"$PY" src/models/kavi-b/simulate.py --n 20000 --seed 2026
"$PY" src/models/kavi-b/report.py > /dev/null
