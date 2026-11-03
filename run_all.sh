#!/usr/bin/env bash
# Rebuild the submission from the committed model outputs, and check it against ./predictions_full_congress.csv.
#
#   bash run_all.sh            # joint step only (seconds)
#   bash run_all.sh --models   # first re-run the five Kavi models from the frozen data (about 2 minutes, offline)
#
# Environment: PY=<python with requirements.txt installed> (default python3).
#              FRAME=pool5 uses the five-model pool as the candidate frame instead of the recorded one (see README).
set -euo pipefail
PY="${PY:-python3}"
# Later steps change directory, so make a relative interpreter path absolute.
if [[ "$PY" == */* ]]; then PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"; fi
cd "$(dirname "$0")"
FRAME="${FRAME:-recorded}"
PAU_STACK=models/pau-stack/results/joint2026/stack_noprim_2026_refreshed.csv

# Checksums of the committed model outputs; works in a git checkout and in a downloaded ZIP alike.
model_outputs_sha() {
  local f
  for f in models/kavi-{a,b,c,d,e}/out/{pred_2026,backtest_2018,backtest_2022}.csv; do
    "$PY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(), sys.argv[1])' "$f"
  done
}

if [[ "${1:-}" == "--models" ]]; then
  echo "== 0. re-run the five Kavi models (models/kavi-a..e/out/)"
  before="$(model_outputs_sha)"
  PY="$PY" bash scripts/run_kavi_models.sh
  after="$(model_outputs_sha)"
  if [[ "$before" == "$after" ]]; then
    echo "   model outputs identical to the ones shipped (15 files, SHA-256)"
  else
    echo "   WARNING: model outputs differ from the ones shipped:"
    diff <(echo "$before") <(echo "$after") || true
  fi
fi

echo "== 1. Pau stack: standard per-race file from the model's raw 2026 output (check only)"
( cd models/pau-stack && PYTHON="$PY" bash run.sh )

echo "== 2. pool5: equal log-linear pool of kavi-a..e"
PY="$PY" bash scripts/build_pool5.sh

echo "== 3. map the Pau stack's party-level probabilities onto candidate rows"
case "$FRAME" in
  recorded) FRAME_FILE=out/joint/candidate_frame_2026.csv ;;
  pool5)    FRAME_FILE=out/joint/pool5_2026.csv ;;
  *) echo "FRAME must be 'recorded' or 'pool5'" >&2; exit 2 ;;
esac
"$PY" src/final/pau_to_candidates.py "$FRAME_FILE" "$PAU_STACK" --out out/joint/pau_cand.csv

echo "== 4. joint pool: equal log-linear pool of pool5 and the Pau stack"
"$PY" src/final/ensemble.py --out out/joint/joint_2026.csv out/joint/pool5_2026.csv out/joint/pau_cand.csv

echo "== 5. picks: rule C', CA-40 and CA-07 forced to the market favourite"
"$PY" src/final/make_submission.py out/joint/joint_2026.csv --rule Cprime --favourite CA-40,CA-07 \
  --out out/predictions_full_congress.csv --detail out/picks_detail.csv

echo "== 6. compare with the submitted file"
if cmp -s out/predictions_full_congress.csv predictions_full_congress.csv; then
  echo "   OK: out/predictions_full_congress.csv is byte-identical to predictions_full_congress.csv"
else
  echo "   DIFFERENT: out/predictions_full_congress.csv vs predictions_full_congress.csv" >&2
  [[ "$FRAME" == "pool5" ]] && echo "   (expected with FRAME=pool5: ID-SEN win_probability 0.998 -> 0.9981)" >&2
  exit 1
fi
