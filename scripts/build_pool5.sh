#!/usr/bin/env bash
# Equal log-linear pool of the five Kavi-side models -> out/joint/pool5_2026.csv.
# Model order matters only for row order and which extra columns are kept (the first file's): keep a, b, c, d, e.
set -euo pipefail
PY="${PY:-python3}"
cd "$(dirname "$0")/.."
mkdir -p out/joint
"$PY" src/final/ensemble.py --out out/joint/pool5_2026.csv \
  models/kavi-a/out/pred_2026.csv models/kavi-b/out/pred_2026.csv models/kavi-c/out/pred_2026.csv \
  models/kavi-d/out/pred_2026.csv models/kavi-e/out/pred_2026.csv
