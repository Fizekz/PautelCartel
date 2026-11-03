#!/usr/bin/env bash
# Rebuild the submission from the committed model outputs, and check it against ./predictions_full_congress.csv.
#
#   bash run_all.sh            # joint step only (seconds)
#   bash run_all.sh --models   # first re-run the five Kavi models from the frozen data (about 6 minutes single-threaded, offline)
#
# Environment: PY=<python with requirements.txt installed> (default python3).
set -euo pipefail
PY="${PY:-python3}"
# Later steps change directory, so make a relative interpreter path absolute.
if [[ "$PY" == */* ]]; then PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"; fi
cd "$(dirname "$0")"
# Pau's committed 2026 output (pau/sub-1103 5bbff94; poll pull of Nov 3 01:25Z). Step 1 re-derives it and checks it.
PAU_STACK=models/pau-stack/reference/2026-11-03T0125Z/stack_noprim_2026.csv

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

echo "== 1. Pau stack: re-fit on 2010-2024 and re-forecast 2026 (about 6 s, offline), checked against the committed output"
PAU_CHECK="$(mktemp -d)"
( cd models/pau-stack && OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 "$PY" run_2026.py --check-ref --tol 1e-6 --out "$PAU_CHECK" | tail -1 )
rm -rf "$PAU_CHECK"

echo "== 2. pool5: equal log-linear pool of kavi-a..e"
PY="$PY" bash scripts/build_pool5.sh

echo "== 3. map the Pau stack's party-level probabilities onto candidate rows"
"$PY" src/final/pau_to_candidates.py out/joint/pool5_2026.csv "$PAU_STACK" --out out/joint/pau_cand.csv

echo "== 4. joint pool: equal log-linear pool of pool5 and the Pau stack"
"$PY" src/final/ensemble.py --out out/joint/joint_2026.csv out/joint/pool5_2026.csv out/joint/pau_cand.csv

echo "== 5. picks: rule C', the four same-party California races CA-07, CA-14, CA-34, CA-40 forced to the market favourite"
"$PY" src/final/make_submission.py out/joint/joint_2026.csv --rule Cprime --favourite CA-07,CA-14,CA-34,CA-40 \
  --out out/predictions_full_congress.csv --detail out/picks_detail.csv

echo "== 6. compare with the submitted file"
if cmp -s out/predictions_full_congress.csv predictions_full_congress.csv; then
  echo "   OK: out/predictions_full_congress.csv is byte-identical to predictions_full_congress.csv"
else
  echo "   DIFFERENT: out/predictions_full_congress.csv vs predictions_full_congress.csv" >&2
  exit 1
fi
