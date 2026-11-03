#!/usr/bin/env bash
# Reproduce every kavi-c output from the data files. Run from the worktree root.
set -euo pipefail
PY=${PY:-python3}
$PY src/models/kavi-c/backtest.py
$PY src/scorecard.py --pred 2018=out/backtest_2018.csv --pred 2022=out/backtest_2022.csv --name kavi-c --json out/scorecard.json
$PY src/models/kavi-c/predict_2026.py
$PY src/models/kavi-c/simulate.py --n 20000
$PY src/models/kavi-c/baselines.py
# v2 diagnostics (not needed for the outputs above):
#   $PY src/models/kavi-c/experiments.py --forward   -> out/experiments_v2.csv (design selection, pre-2018 LOYO)
#   $PY src/models/kavi-c/ablation_v2.py             -> out/ablation_v2.csv (v1 -> v2 step effects; reruns with v3 on)
# v3 diagnostics:
#   $PY src/models/kavi-c/pool_experiment.py rating  -> out/pool_experiment_v3_rating.csv (pool weight, pre-2018 LOYO)
#   $PY src/models/kavi-c/experiments.py --forward --out experiments_v3.csv --only "..."  (env x mid x comp test)
#   $PY src/models/kavi-c/ablation_v3.py             -> out/ablation_v3.csv (v2 -> v3 step effects on 2018/2022)
