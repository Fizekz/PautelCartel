"""v1 -> v2 step-by-step effect on the 2018/2022 backtests (scored with src/scorecard.py, all and contested).

Steps are cumulative. These numbers use the test years, so they are reported, never used to choose a design
(design choices come from experiments.py, pre-2018 leave-one-year-out).

Usage: python src/models/kavi-c/ablation_v2.py -> out/ablation_v2.csv
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest  # noqa: E402
import dataset  # noqa: E402
import hist_features  # noqa: E402
import model  # noqa: E402
import polls  # noqa: E402
from common import OUT, ROOT  # noqa: E402

_spec = importlib.util.spec_from_file_location("scorecard", ROOT / "src" / "scorecard.py")
scorecard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorecard)

V1_MODEL = dict(USE_ENV_MIDTERM=True, USE_MID_OUT=False, USE_MID_OUT_COMP=False, USE_OTHER_DEV=False)
STEPS = [
    ("bug fixes (surname, RCV rounds, L2)", V1_MODEL, dict(USE_SPONSOR_OFFSET=True), False, set()),
    ("+ FEC: 2018 pre-election snapshot", V1_MODEL, dict(USE_SPONSOR_OFFSET=True), True, set()),
    ("+ FEC: 2022 money dropped", V1_MODEL, dict(USE_SPONSOR_OFFSET=True), True, {2022}),
    ("+ no sponsor offset", V1_MODEL, dict(USE_SPONSOR_OFFSET=False), True, {2022}),
    ("+ mid_out x comp (v2 backtest)", dict(V1_MODEL, USE_MID_OUT_COMP=True), dict(USE_SPONSOR_OFFSET=False),
     True, {2022}),
    ("v2 with 2022 full-cycle money (leaky)", dict(V1_MODEL, USE_MID_OUT_COMP=True),
     dict(USE_SPONSOR_OFFSET=False), True, set()),
]


def configure(mflags: dict, pflags: dict, fec_pre: bool, no_money: set) -> None:
    for k, v in mflags.items():
        setattr(model, k, v)
    for k, v in pflags.items():
        setattr(polls, k, v)
    hist_features.USE_FEC_2018_PRE = fec_pre
    backtest.NO_MONEY_BACKTEST = no_money
    hist_features.load_fec_hist.cache_clear()
    dataset.features.cache_clear()
    dataset.poll_table.cache_clear()


def main() -> int:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, mflags, pflags, fec_pre, no_money in STEPS:
            configure(mflags, pflags, fec_pre, no_money)
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
    res.to_csv(OUT / "ablation_v2.csv", index=False)
    print(res.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
