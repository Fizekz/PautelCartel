"""Assemble per-year modelling tables (features + poll averages) for the historical years."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

import hist_features
import polls
from common import HIST_YEARS


@lru_cache(maxsize=1)
def features() -> pd.DataFrame:
    df = hist_features.build()
    df["env"] = df["env_generic"]
    df["share"] = df["dem_share_2p"]
    return df


@lru_cache(maxsize=None)
def poll_table(year: int) -> pd.DataFrame:
    """Poll averages for `year`, using a pollster track record from years before `year` only."""
    res = hist_features.load_results()
    track = polls.pollster_track([y for y in HIST_YEARS if y < year], res)
    ag, _ = polls.aggregate(polls.hist_polls(year, res), track)
    return ag.assign(year=year)


def year_table(year: int) -> pd.DataFrame:
    df = features()
    d = df[df["year"] == year].merge(poll_table(year), on=["year", "race_id"], how="left")
    return d


def training(years: list[int]) -> pd.DataFrame:
    d = pd.concat([year_table(y) for y in years], ignore_index=True)
    return d[d["trainable"]].reset_index(drop=True)
