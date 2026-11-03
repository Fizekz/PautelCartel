#!/usr/bin/env bash
# Reproduce every kavi-d (v3) output from the data files. Run from anywhere.
set -euo pipefail
PY=${PY:-python3}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
mkdir -p "$ROOT/out/cache" "$ROOT/out/variants"
rm -f "$ROOT/out/cache/train_features_w60.csv" "$ROOT/out/cache/sameparty_train.csv"
cd "$HERE"
$PY "$ROOT/src/models/kavi-d/fetch_ca_primary.py"        # uses data/extra/raw cache if present, else Wikipedia
$PY history.py                                            # training table -> out/cache/train_features_w60.csv
$PY sameparty.py                                          # same-party model fits (with / without money, primary)
$PY backtest.py                                           # out/backtest_2018.csv, out/backtest_2022.csv (final_v3)
for c in final logit_v1 op final_v2 logit_gbmid env_e env_e_op env_e_gbmid env_e_gbmid_polls; do                   # decomposition variants -> out/variants/<config>/
  $PY backtest.py --config "$c" --outdir "$ROOT/out/variants/$c"
done
cd "$ROOT"
$PY src/scorecard.py --pred 2018=out/backtest_2018.csv --pred 2022=out/backtest_2022.csv --name kavi-d \
    --json out/scorecard.json | tee out/scorecard.txt
for c in final logit_v1 op final_v2 logit_gbmid env_e env_e_op env_e_gbmid env_e_gbmid_polls; do
  $PY src/scorecard.py --pred 2018=out/variants/$c/backtest_2018.csv --pred 2022=out/variants/$c/backtest_2022.csv \
      --name "$c" --json "out/variants/$c/scorecard.json" > "out/variants/$c/scorecard.txt"
done
cd "$HERE"
$PY predict_2026.py --no-late-info && cp "$ROOT/out/pred_2026.csv" "$ROOT/out/variants/pred_2026_model_only.csv"
$PY predict_2026.py                                       # out/pred_2026.csv, disagreements_2026.csv, late_info_2026.csv
$PY simulate.py --n 20000                                 # out/sims_summary.json
$PY analysis.py                                           # out/analysis_*.csv (incl. analysis_midterm.csv)
$PY experiments_v2.py v1_final logit_v1 op op_close op_both oth rcp op_oth op_rcp logit_op logit_op_rcp logit_oth \
    logit_op_oth                                          # out/experiments_v2.csv (LOYO pre-2018 + rolling)
$PY environment.py                                        # N_hat table (own GB-bias estimate and kavi-e's)
$PY experiments_v3.py > "$ROOT/out/experiments_v3.log"   # out/experiments_v3.csv (LOYO pre-2018, rolling, LOYO all 8)
KAVID_POLL_SPARSE_DAYS=150 $PY history.py > /dev/null     # sparse-poll window variant (not used by final_v2)
KAVID_POLL_SPARSE_DAYS=150 $PY experiments_v2.py v1_final logit_op op
$PY check_outputs.py                                       # invariants on the outputs
