"""Diagnostics for METHODS.md: rating-bin calibration (per forecaster, combined, by side and year), forecaster
disagreement, cycle effects, and the leave-one-year-out model selection table.

Usage: python src/models/kavi-b/report.py   -> out/report/*.csv (all in-sample descriptive unless marked LOYO)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402
from features import hist_table  # noqa: E402
from model import Config, SideModel, fit_set, loyo  # noqa: E402

warnings.filterwarnings("ignore")
BINS = [-0.01, 0.0001, 0.5, 1.0, 1.5, 2.0, 2.5, 2.99, 3.0]
LABELS = ["exact toss-up", "tilt (0,0.5]", "lean (0.5,1]", "(1,1.5]", "likely (1.5,2]", "(2,2.5]", "(2.5,3)", "safe 3"]
SELECTION = [
    Config("ratings_only", ("r",), ridge=0.005),
    Config("ratings_polls", ("r", "poll"), ridge=0.005),
    Config("ratings_polls_lean (v1 FINAL; cycle const)", ("r", "poll", "lean"), ridge=0.005),
    Config("+hinge", ("r", "r_hinge", "poll", "lean"), ridge=0.005),
    Config("+spread", ("r", "r_x_spread", "poll", "lean"), ridge=0.005),
    Config("per-forecaster", ("r_cook", "r_sabato", "r_inside", "poll", "lean"), ridge=0.005),
    Config("+incumbency", ("r", "poll", "lean", "inc"), ridge=0.005),
    Config("+fec", ("r", "poll", "lean", "fec"), ridge=0.005),
    Config("full_fundamentals", ("r", "r_hinge", "lean", "inc", "fec", "poll_s", "poll_h"), ridge=0.005),
    Config("v2 FINAL: r+poll+lean, cycle mid_mo, no floor", ("r", "poll", "lean"), ridge=0.005, cycle="mid_mo",
           floor=False),
    Config("v3 FINAL: r+poll+lean, cycle gb_mid, no floor", ("r", "poll", "lean"), ridge=0.005, cycle="gb_mid",
           floor=False),
]


def bin_table(d: pd.DataFrame, col: str) -> pd.DataFrame:
    x = d[d[col].notna()].copy()
    x["bin"] = pd.cut(x[col].abs(), BINS, labels=LABELS)
    x["fav_won"] = np.where(x[col] > 0, x["y"], np.where(x[col] < 0, 1 - x["y"], np.nan))
    return x.groupby("bin", observed=True).agg(n=("y", "size"), fav_win_rate=("fav_won", "mean"),
                                               d_win_rate=("y", "mean")).round(3)


def main() -> None:
    out = OUT / "report"
    out.mkdir(parents=True, exist_ok=True)
    h = hist_table()
    d = h[(h["same_party"] == 0) & (h["no_contest"] == 0)]
    tabs = {c: bin_table(d, c) for c in ("r_mean", "r_cook", "r_sabato", "r_inside")}
    pd.concat(tabs, axis=1).to_csv(out / "calibration_by_bin.csv")
    x = fit_set(d).copy()
    x = x[x["r_mean"] != 0]
    x["side"] = np.where(x["r_mean"] > 0, "D favoured", "R favoured")
    x["fav_won"] = np.where(x["r_mean"] > 0, x["y"], 1 - x["y"])
    x["bin"] = pd.cut(x["r_mean"].abs(), [0, 0.5, 1.5, 2.99], labels=["toss-up/tilt", "lean", "likely"])
    piv = x.pivot_table(index=["bin", "side"], columns="year", values="fav_won", aggfunc="mean", observed=True).round(2)
    cnt = x.pivot_table(index=["bin", "side"], columns="year", values="fav_won", aggfunc="size", observed=True)
    piv.to_csv(out / "fav_win_by_side_year.csv")
    # v2: midterms only, favourite = out-party vs in-party (in-sample, descriptive)
    mt = x[x["midterm"] == 1].copy()
    mt["fav_is_out"] = np.where((mt["r_mean"] > 0) == (mt["out_d"] > 0), "out-party favoured", "in-party favoured")
    mt.pivot_table(index=["bin", "fav_is_out"], columns="year", values="fav_won", aggfunc=["mean", "size"],
                   observed=True).round(2).to_csv(out / "midterm_fav_win_out_in.csv")
    cnt.to_csv(out / "fav_win_by_side_year_n.csv")
    # forecaster disagreement: spread of the three ratings among non-safe races
    x["spread_bin"] = pd.cut(x["r_spread"], [-0.01, 0.0, 1.0, 6.0], labels=["agree", "differ by <=1", "differ by >1"])
    x.groupby(["bin", "spread_bin"], observed=True).agg(n=("y", "size"), fav_win=("fav_won", "mean")).round(3) \
        .to_csv(out / "disagreement.csv")
    m = SideModel(SELECTION[-1]).fit(h)
    nat = h.groupby("year")[["generic_margin_dr", "midterm", "pres_d"]].first()
    fx = m.year_fx.rename(lambda s: int(s[2:]))
    nat.assign(year_effect=fx.round(3)).to_csv(out / "year_effects.csv")
    rows = []
    for cfg in SELECTION:
        a = loyo(h, cfg, [2010, 2012, 2014, 2016])
        b = loyo(h, cfg, list(range(2010, 2025, 2)))
        rows.append({"config": cfg.name, "loyo_2010_16_nonsafe_ll": round(a["ll_nonsafe"].mean(), 4),
                     "loyo_2010_24_nonsafe_ll": round(b["ll_nonsafe"].mean(), 4),
                     **{f"loyo_{int(r.year)}": round(r.ll_nonsafe, 4) for r in b.itertuples()}})
    pd.DataFrame(rows).to_csv(out / "model_selection_loyo.csv", index=False)
    print(pd.read_csv(out / "calibration_by_bin.csv").to_string())
    print(piv.to_string())
    print(pd.read_csv(out / "disagreement.csv").to_string())
    print(pd.read_csv(out / "year_effects.csv").to_string())
    print(pd.DataFrame(rows).iloc[:, :3].to_string())


if __name__ == "__main__":
    main()
