"""v2 -> v3 step-by-step effect on the 2018/2022 backtests (scored with src/scorecard.py, all and contested).

Steps are cumulative; each also shows a leave-one-out row (v3 minus that step). These use the test years, so they
are reported, not used to choose (the pool weight comes from pool_experiment.py, pre-2018 LOYO).

Usage: python src/models/kavi-c/ablation_v3.py -> out/ablation_v3.csv
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest  # noqa: E402
import pool  # noqa: E402
from common import OUT, ROOT  # noqa: E402

_spec = importlib.util.spec_from_file_location("scorecard", ROOT / "src" / "scorecard.py")
scorecard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorecard)

V3_W = pool.POOL_W
# (name, 2022 June FEC cut, pool weight, primary split)
STEPS = [
    ("v2 (2022 without money)", False, 0.0, False),
    ("+ 2022 June pre-election FEC cut", True, 0.0, False),
    ("+ ratings-logit pool w=0.25", True, V3_W, False),
    ("+ primary same-party split = v3", True, V3_W, True),
    ("v3 without the pool", True, 0.0, True),
    ("v3 without the 2022 cut", False, V3_W, True),
]


def main() -> int:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, cut, w, prim in STEPS:
            backtest.PRE_CUT_2022 = cut
            backtest.NO_MONEY_BACKTEST = set() if cut else {2022}
            pool.POOL_W = w
            backtest.USE_PRIMARY_SPLIT = prim
            row = {"step": name}
            for year in (2018, 2022):
                out, _ = backtest.run_year(year)
                path = Path(tmp) / f"bt_{year}.csv"
                out.to_csv(path, index=False)
                rep = scorecard.score_year(year, path)
                row[f"ll_{year}"] = rep["all"]["log_loss"]
                row[f"ll_contested_{year}"] = rep["contested"]["log_loss"]
                row[f"brier_{year}"] = rep["all"]["brier"]
            rows.append(row)
            print(row, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ablation_v3.csv", index=False)
    print(res.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
