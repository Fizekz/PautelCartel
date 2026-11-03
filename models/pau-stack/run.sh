#!/usr/bin/env bash
# No-primaries stack: how its outputs are produced. Run from models/pau-stack/.
# Step A runs anywhere (pandas + numpy). Step B needs the intermediate tables in pbase/work/, rebuilt by step C
# from raw downloads that are not redistributed in this repo.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# --- A. standard per-race file from the model's 2026 output (offline), checked against the committed file ---
CHECK=results/joint2026/stack_noprim_2026_refreshed.check.csv
"$PY" src/to_standard.py --model noprim --tag refreshed --out "$CHECK"
"$PY" - "$CHECK" results/joint2026/stack_noprim_2026_refreshed.csv <<'PYEOF'
import sys, pandas as pd
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
assert list(a.columns) == list(b.columns) and a.race_id.equals(b.race_id), "row/column mismatch"
d = max(float((a[c] - b[c]).abs().max()) for c in ("q_dem", "q_rep", "q_other", "k_pD_nov1"))
assert d < 1e-12, f"max abs difference {d}"
print(f"OK: {len(a)} races, max abs difference {d:.1e}")
PYEOF

if [[ "${1:-}" != "--full" ]]; then
  echo "Done (step A only). Pass --full to re-run the model itself; see README.md for what that needs."
  exit 0
fi

# --- B. the model (needs pbase/work/: features_hist.csv, work/stack/features_2026_refresh1102.csv,
#        pg_features_rrrp.csv, pt_edges.csv, pe_rchg_features.csv, px/primaries.csv, stack/wiki/*.wikitext,
#        national_ext.csv) ---
( cd pbase/src
  "$PY" pk_2026.py ../work/stack/features_2026_refresh1102.csv refreshed   # -> results/pk/pk_2026_refreshed.*
  "$PY" pk_proto.py node+geo+field+rchg:clip2 pk_stack_noprim )             # -> proto/pk_stack_noprim/out/
"$PY" src/to_standard.py --model noprim --tag refreshed   # overwrites the committed file

# --- C. rebuilding pbase/work/ (order; each script's docstring names its raw sources) ---
#  1. pbase/src/run_all.py (features.py, national.py, national_ext.py): features_hist.csv, features_2026.csv
#  2. pbase/src/refresh_polls_2026.py + refresh_v6nl_2026.py: the Nov 2 poll refresh -> features_2026_new.csv,
#     copied to pbase/work/stack/features_2026_refresh1102.csv
#  3. pbase/src/pg_fetch.py, pg_geom.py, pg_data.py, pg_features.py, pg_demo.py: pg_features_rrrp.csv (history-chosen directions)
#  4. pbase/src/pt_graphs.py: pt_edges.csv (state graph)
#  5. pbase/src/pe_rchg_fetch.py, pe_rchg.py, pe_rchg_2026.py: pe_rchg_features.csv and the 2026 rating pages
#  6. pbase/src/px_primaries.py: px/primaries.csv (read at import; not used by the no-primaries variant). It also
#     imports a vendored spreadsheet helper from pbase/work/pylib/, which is not redistributed here.
#  Beyond requirements.txt, these rebuild scripts need beautifulsoup4, pyshp, shapely and openpyxl (steps 4 and 6 read Excel files).
