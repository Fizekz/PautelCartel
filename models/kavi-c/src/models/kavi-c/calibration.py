"""Calibration table of the favourite's q, contested races 2014-2024, each year fitted on earlier years only.

Usage: python src/models/kavi-c/calibration.py -> out/calibration_v2.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import model  # noqa: E402
from common import HIST_YEARS, OUT  # noqa: E402

BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0001]


def main() -> int:
    rows = []
    for y in HIST_YEARS[2:]:
        m = model.fit(dataset.training([t for t in HIST_YEARS if t < y]))
        te = dataset.year_table(y)
        te = te[te["trainable"]].reset_index(drop=True)
        q = m.predict(te)["qA"].to_numpy()
        won = (te["share"] > 0.5).to_numpy()
        fav_q = np.maximum(q, 1 - q)
        fav_won = np.where(q >= 0.5, won, ~won)
        rows.append(pd.DataFrame({"year": y, "fav_q": fav_q, "fav_won": fav_won}))
    d = pd.concat(rows)
    d["bin"] = pd.cut(d["fav_q"], BINS, right=False)
    out = d.groupby("bin", observed=True).agg(n=("fav_q", "size"), mean_q=("fav_q", "mean"),
                                              hit_rate=("fav_won", "mean")).reset_index()
    out["bin"] = out["bin"].astype(str)
    out.to_csv(OUT / "calibration_v2.csv", index=False)
    print(out.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
