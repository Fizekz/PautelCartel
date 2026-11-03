"""v2 design selection: side-model log loss (non-Safe contested races) under
  LOYO 2010-2016 held-out midterms 2010 + 2014 (the v2 selection criterion: 2026 is a midterm; never sees 2018/2022),
  LOYO 2010-2016  (all four held-out years; the v1 criterion),
  LOYO 2010-2024  (check; includes the test years),
  forward 2018 (fit 2010-16) and forward 2022 (fit 2010-20) (reported, not used to choose).

Usage: python src/models/kavi-b/select_v2.py   -> out/report/model_selection_v2.csv
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402
from features import hist_table  # noqa: E402
from model import Config, SideModel, log_loss  # noqa: E402

warnings.filterwarnings("ignore")

FEATS = {
    "v1: r+poll+lean": ("r", "poll", "lean"),
    "r+poll_sh+lean": ("r", "poll_sh", "lean"),
    "r+poll_sh+lean+oth": ("r", "poll_sh", "lean", "oth"),
    "r+poll+lean+oth": ("r", "poll", "lean", "oth"),
    "r+poll_sh+lean+r_mid": ("r", "poll_sh", "lean", "r_mid"),
    "r+poll_sh+lean+fec": ("r", "poll_sh", "lean", "fec"),
}
CYCLES = ("const", "mo", "mid_mo")
FLOORS = (True, False)


def scored(te: pd.DataFrame) -> pd.DataFrame:
    return te[(te["same_party"] == 0) & (te["no_contest"] == 0) & (te["r_mean"].abs() < 3)]


def side_ll(hist: pd.DataFrame, cfg: Config, train_years: list[int], test_year: int) -> tuple[float, int]:
    m = SideModel(cfg).fit(hist[hist["year"].isin(train_years)])
    te = scored(hist[hist["year"] == test_year])
    return log_loss(te["y"].to_numpy(), m.predict(te)), len(te)


def loyo_mean(hist: pd.DataFrame, cfg: Config, years: list[int]) -> float:
    """Race-weighted mean log loss over held-out years."""
    tot, n = 0.0, 0
    for y in years:
        ll, k = side_ll(hist, cfg, [v for v in years if v != y], y)
        tot, n = tot + ll * k, n + k
    return tot / n


def loyo_midterms(hist: pd.DataFrame, cfg: Config, years: list[int]) -> float:
    """Race-weighted LOYO log loss over the held-out MIDTERM years only (training on all other years in `years`).
    The 2026 target is a midterm, so this is the v2 selection criterion (with years = 2010-2016)."""
    tot, n = 0.0, 0
    for y in years:
        if y % 4 != 2:
            continue
        ll, k = side_ll(hist, cfg, [v for v in years if v != y], y)
        tot, n = tot + ll * k, n + k
    return tot / n


def evaluate(hist: pd.DataFrame, cfg: Config) -> dict:
    return {
        "loyo_mid_10_14": loyo_midterms(hist, cfg, [2010, 2012, 2014, 2016]),
        "loyo_10_16": loyo_mean(hist, cfg, [2010, 2012, 2014, 2016]),
        "loyo_mid_10_22": loyo_midterms(hist, cfg, [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]),
        "loyo_10_24": loyo_mean(hist, cfg, [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]),
        "fwd_2018": side_ll(hist, cfg, [2010, 2012, 2014, 2016], 2018)[0],
        "fwd_2022": side_ll(hist, cfg, [2010, 2012, 2014, 2016, 2018, 2020], 2022)[0],
    }


def main() -> None:
    hist = hist_table(pre2018_fec=True)
    rows = []
    for (fname, feats), cyc, fl in itertools.product(FEATS.items(), CYCLES, FLOORS):
        cfg = Config(f"{fname}|{cyc}|floor={fl}", feats, ridge=0.005, cycle=cyc, floor=fl)
        r = {"features": fname, "cycle": cyc, "floor": fl, **evaluate(hist, cfg)}
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    for k in (0.5, 2.0, 4.0):
        cfg = Config(f"poll_k={k}", ("r", "poll_sh", "lean"), ridge=0.005, cycle="mid_mo", poll_k=k)
        r = {"features": f"r+poll_sh(k={k})+lean", "cycle": "mid_mo", "floor": True, **evaluate(hist, cfg)}
        rows.append(r)
        print(r, flush=True)
    out = pd.DataFrame(rows).sort_values("loyo_mid_10_14")
    (OUT / "report").mkdir(exist_ok=True, parents=True)
    out.round(5).to_csv(OUT / "report" / "model_selection_v2.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
