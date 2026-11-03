"""Out-of-sample backtests: predict 2018 from 2010-2016 and 2022 from 2010-2020.

Usage (from the worktree root):
    python src/models/kavi-b/backtest.py            # writes out/backtest_2018.csv, out/backtest_2022.csv
    python src/models/kavi-b/backtest.py --variants # also scores the alternative configs (out/variants/)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allocate import allocate_race, fec_lookup, fit_within_party, primary_pct, within_party_data  # noqa: E402
from common import BENCH, OUT  # noqa: E402
from features import _fec_hist, hist_table  # noqa: E402
from model import Config, SideModel  # noqa: E402

warnings.filterwarnings("ignore")

# v3 FINAL: cycle design gb_mid (year effect = c + b x midterm x GB/10), chosen on LOYO 2010-2024 (select_v3.py,
# out/report/model_selection_v3.csv); this criterion includes the test years 2018/2022 (flagged in LOG/METHODS).
# v2 FINAL was cycle mid_mo, chosen by held-out-midterm LOYO within 2010-2016 (select_v2.py).
FINAL = Config("final_v3", ("r", "poll", "lean"), ridge=0.005, cycle="gb_mid", floor=False)
VARIANTS = [
    Config("v2_mid_mo", ("r", "poll", "lean"), ridge=0.005, cycle="mid_mo", floor=False),
    Config("bma_nofloor", ("r", "poll", "lean"), ridge=0.005, cycle="bma", floor=False),
    Config("mid_gb_nofloor", ("r", "poll", "lean"), ridge=0.005, cycle="mid_gb", floor=False),
    Config("v1_like_const_floor", ("r", "poll", "lean"), ridge=0.005, cycle="const", floor=True),
    Config("const_nofloor", ("r", "poll", "lean"), ridge=0.005, cycle="const", floor=False),
    Config("mo_nofloor", ("r", "poll", "lean"), ridge=0.005, cycle="mo", floor=False),
    Config("mid_mo_poll_sh", ("r", "poll_sh", "lean"), ridge=0.005, cycle="mid_mo", floor=False),
    Config("ratings_only_mid_mo", ("r",), ridge=0.005, cycle="mid_mo", floor=False),
]
# Years whose FEC totals exist only as full-cycle (post-election) files: their own money is not used (v2).
NO_PRE_ELECTION_FEC: set = set()   # v3: 2022 now has a pre-election (June) cut, see features.PRE_ELECTION_FEC


def predict_year(hist: pd.DataFrame, fec: pd.DataFrame, year: int, cfg: Config) -> pd.DataFrame:
    train = hist[hist["year"] < year]
    side = SideModel(cfg).fit(train)
    use_money = year not in NO_PRE_ELECTION_FEC
    wp = fit_within_party(within_party_data(train, fec, sorted(train["year"].unique())), use_money=use_money)
    bench = pd.read_csv(BENCH / f"fte_final_{year}.csv")
    feats = hist[hist["year"] == year].set_index("race_id")
    out = []
    for rid, cands in bench.groupby("race_id", sort=False):
        cands = cands.copy()
        rec = [fec_lookup(fec, year, rid, n, p) for n, p in zip(cands["candidate"], cands["party"])]
        cands["receipts"] = [r[0] if use_money else 0.0 for r in rec]
        cands["is_inc"] = [r[1] for r in rec]
        cands["is_minor"] = cands["candidate"].str.lower().eq("others") | ~cands["party"].isin(["DEM", "REP", "IND"])
        cands["side"] = np.where(cands["party"] == "REP", "R", "D")
        cands["prim_pct"] = [primary_pct(year, rid, n) for n in cands["candidate"]]
        if rid in feats.index:
            f = feats.loc[[rid]]
            row = f.iloc[0]
            same = None
            if row["same_party"] == 1:
                majors = cands.loc[cands["party"].isin(["DEM", "REP"]), "party"]
                same = majors.value_counts().idxmax() if len(majors) else None
            q_d = float(side.predict(f)[0]) if row["no_contest"] == 0 else (0.999 if pd.notna(row["dside_name"]) else 0.001)
            d_party = "IND" if row["dside_is_ind"] == 1 else "DEM"
        else:   # not in the results/ratings join: fall back to a coin flip between the sides
            same, q_d, d_party = None, 0.5, "DEM"
        cands["q"] = allocate_race(cands, q_d, d_party, same, wp)
        cands["q_dside"] = q_d
        out.append(cands)
    res = pd.concat(out, ignore_index=True)
    res.attrs["side"], res.attrs["wp"] = side, wp
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", action="store_true")
    args = ap.parse_args()
    hist = hist_table(pre2018_fec=True)
    fec = _fec_hist(pre2018=True)
    OUT.mkdir(exist_ok=True)
    cfgs = [FINAL] + (VARIANTS if args.variants else [])
    for cfg in cfgs:
        for year in (2018, 2022):
            res = predict_year(hist, fec, year, cfg)
            side, wp = res.attrs["side"], res.attrs["wp"]
            path = OUT / f"backtest_{year}.csv" if cfg is FINAL else OUT / "variants" / f"{cfg.name}_{year}.csv"
            path.parent.mkdir(exist_ok=True)
            res[["year", "race_id", "candidate", "party", "q"]].to_csv(path, index=False)
            cs = side.cycle_summary(hist[hist["year"] == year])
            print(f"{cfg.name} {year}: beta {side.beta.round(3).to_dict()} cycle mean {cs['pred_mean']:+.3f} "
                  f"sd {cs['pred_sd']:.3f} floor {side.tail_floor:.4f} within-party {wp} -> {path.relative_to(OUT.parent)}")


if __name__ == "__main__":
    main()
