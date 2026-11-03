"""Out-of-sample backtests: predict 2018 from 2010-2016 and 2022 from 2010-2020.

Candidate lists (names, parties) come from data/benchmark/fte_final_<year>.csv; p_fte is never read.
Incumbency comes from fec_hist; receipts from the pre-election FEC snapshot (fetch_fec_preelection.py), race features from hist_features
(lean, prior result, polls, ratings: all pre-election). The year's own results are used only by trainable() to
pick *training* rows, which exclude the test year.

Usage: python backtest.py [--variant F+P+R]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import APPOINTED_INCUMBENTS, BENCH, OUT
from features import fec_lookup, match_fec_full
from model import fit_all, predict_chamber, win_prob
from races import assign_roles, candidate_q, prev_same, race_frame

VARIANTS = {"F": (False, False, False), "F+P": (True, False, False), "F+P+R": (True, True, False),
            "F+C": (False, False, True), "F+P+R+C": (True, True, True)}
RACE_COLS = ["lean", "lean2", "prev_dev", "prev_winner", "poll_share", "poll_weight", "n_polls", "rating", "g",
             "state", "chamber", "midterm", "pres_d"]


def bench_candidates(year: int) -> pd.DataFrame:
    b = pd.read_csv(BENCH / f"fte_final_{year}.csv")[["year", "race_id", "chamber", "candidate", "party"]]
    fec = fec_lookup(pre_year=year)  # pre-election receipts: no post-election look-ahead
    vals = [match_fec_full(fec, year, r.race_id, r.candidate, r.party if r.party in ("DEM", "REP") else "IND")
            for r in b.itertuples()]
    # v3: two candidates of one race matched to the same FEC row (same surname, WA-09 2018 Adam / Sarah Smith):
    # re-match those by first name
    key = pd.Series([f"{r}|{v[1]}|{v[0]}" for r, v in zip(b["race_id"], vals)], index=b.index)
    clash = key.duplicated(keep=False) & (pd.Series([v[1] for v in vals], index=b.index) > 0)
    for i in b.index[clash]:
        r = b.loc[i]
        vals[b.index.get_loc(i)] = match_fec_full(fec, year, r.race_id, r.candidate,
                                                  r.party if r.party in ("DEM", "REP") else "IND", by_first_name=True)
    b["is_inc"] = [int(v[0] == "I") for v in vals]
    for (yr, rid), surname in APPOINTED_INCUMBENTS.items():
        hit = (b["year"] == yr) & (b["race_id"] == rid) & b["candidate"].str.lower().str.contains(surname)
        b.loc[hit, "is_inc"] = 1
    b["receipts"] = [v[1] for v in vals]
    b["fec_bucket"] = [v[2] for v in vals]
    return b


def predict_races(c: pd.DataFrame, feats: pd.DataFrame, fits: dict, ind_bonus: float = 0.0) -> pd.DataFrame:
    """Race table with mu / sd / p_d for every race in the candidate list."""
    c = assign_roles(c)
    races = race_frame(c).join(feats[RACE_COLS], how="left")
    races["prev_same_inc"] = prev_same(races)
    has_dem = c.groupby("race_id")["party"].apply(lambda s: (s == "DEM").any())
    ind_with_dem = (races["d_is_ind"] == 1) & has_dem.reindex(races.index).fillna(False)
    if "force_side" not in c:  # backtests: the poll's D column is the minor Democrat, not the IND (ME 2018)
        races.loc[ind_with_dem, ["poll_share", "poll_weight"]] = np.nan
    races["dem_share_2p"] = np.nan
    races["p_d"] = np.nan
    for ch, fit in fits.items():
        idx = races.index[(races["chamber"] == ch) & (races["kind"] == "two_party")]
        if len(idx) == 0:
            continue
        sub = races.loc[idx]
        pr = predict_chamber(fit, sub, float(sub["g"].iloc[0]))
        # red-state independents beat a Democrat's fundamentals (KS 2014, UT 2022, NE 2024: +4 to +6 pts); with
        # polls the stage-2 residual is ~0, so the bonus applies only to unpolled independent-led races
        bonus_rows = (sub["d_is_ind"] == 1) & (sub["lean"] < 0) & sub["poll_share"].isna()
        pr["mu"] = pr["mu"] + ind_bonus * bonus_rows.to_numpy()
        pr["mu_raw"] = pr["mu_raw"] + ind_bonus * bonus_rows.to_numpy()
        cols = ["mu", "mu_raw", "mu_f", "sd_nat", "sd_state", "sd_race", "sd_total"]
        races.loc[idx, cols] = pr[cols].to_numpy()
        races.loc[idx, "p_d"] = win_prob(pr["mu"], pr["sd_nat"], pr["sd_state"], pr["sd_race"])
        races.loc[idx, "p_d_raw"] = win_prob(pr["mu_raw"], pr["sd_nat"], pr["sd_state"], pr["sd_race"])
    c["q"] = candidate_q(c, races)
    return c, races


def run(year: int, variant: str, hist: pd.DataFrame) -> pd.DataFrame:
    polls, ratings, cal = VARIANTS[variant]
    train_years = [y for y in range(2010, year, 2)]
    fits = fit_all(hist, train_years, polls, ratings, calibrate=cal)
    feats = hist[hist["year"] == year].drop_duplicates("race_id").set_index("race_id")
    c = bench_candidates(year)
    c, races = predict_races(c, feats, fits)
    return c, races


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="F+P+R+C", choices=list(VARIANTS))
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    hist = pd.read_csv(OUT / "hist_features.csv")
    for year in (2018, 2022):
        c, races = run(year, args.variant, hist)
        c[["year", "race_id", "candidate", "party", "q"]].to_csv(OUT / f"backtest_{year}{args.suffix}.csv",
                                                                   index=False)
        races.to_csv(OUT / f"backtest_races_{year}{args.suffix}.csv")
        print(year, args.variant, races["kind"].value_counts().to_dict())


if __name__ == "__main__":
    main()
