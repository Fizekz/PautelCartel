#!/usr/bin/env bash
# Rebuild the five Kavi-side model forecasts (models/kavi-a..e/out/pred_2026.csv and the 2018/2022 backtests)
# from the shared data in data/ and each model's own inputs in models/kavi-X/data/. No network access.
# Usage (from the repository root):  PY=python3 bash scripts/run_kavi_models.sh
set -euo pipefail
PY="${PY:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$ROOT/models"

# Each model reads the shared data through its own data/ folder (the layout of its original worktree):
# link models/kavi-X/data/{clean,benchmark,kalshi,templates,dossiers,raw} and models/kavi-X/src/data to the shared copies.
for m in a b c d e; do
  for x in clean benchmark kalshi templates dossiers raw; do ln -sfn "../../../data/$x" "$M/kavi-$m/data/$x"; done
  ln -sfn "../../../src/data" "$M/kavi-$m/src/data"
done

echo "== kavi-a"
mkdir -p "$M/kavi-a/out"
( cd "$M/kavi-a/src/models/kavi-a"
  "$PY" features.py > /dev/null
  "$PY" fetch_ca_primaries.py --offline > /dev/null
  "$PY" fetch_fec_preelection.py --offline > /dev/null
  "$PY" backtest.py > /dev/null
  "$PY" predict_2026.py --sims 20000 > /dev/null )

echo "== kavi-b"
mkdir -p "$M/kavi-b/out/report"
( cd "$M/kavi-b"
  "$PY" src/models/kavi-b/fetch_fec_2018_snapshot.py > /dev/null
  "$PY" src/models/kavi-b/fetch_primaries.py > /dev/null
  "$PY" src/models/kavi-b/fetch_ca_polls_2026.py > /dev/null
  "$PY" src/models/kavi-b/backtest.py > /dev/null
  "$PY" src/models/kavi-b/predict_2026.py > /dev/null )

echo "== kavi-c"
mkdir -p "$M/kavi-c/out"
( cd "$M/kavi-c"
  "$PY" src/models/kavi-c/backtest.py > /dev/null
  "$PY" src/models/kavi-c/predict_2026.py > /dev/null )

echo "== kavi-d"
mkdir -p "$M/kavi-d/out/cache"
rm -f "$M/kavi-d/out/cache/train_features_w60.csv" "$M/kavi-d/out/cache/sameparty_train.csv"
( cd "$M/kavi-d/src/models/kavi-d"
  "$PY" history.py > /dev/null
  "$PY" sameparty.py > /dev/null
  "$PY" backtest.py > /dev/null
  "$PY" predict_2026.py > /dev/null )

echo "== kavi-e"
mkdir -p "$M/kavi-e/out/work"
( cd "$M/kavi-e"
  for s in specials national house_data senate_data backtest predict_2026; do "$PY" "src/models/kavi-e/$s.py" > /dev/null; done )

echo "done: models/kavi-{a,b,c,d,e}/out/pred_2026.csv"
