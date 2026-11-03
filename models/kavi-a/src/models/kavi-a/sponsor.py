"""v3: partisan-sponsor poll shift (idea from pau-a v3, which shifts D/R-sponsored polls by the sponsor's history;
kavi-e uses a fixed 2-point margin shift). Adapted here to my share scale and estimated per fit cutoff.

House effect of a sponsored poll = its two-party D share minus the mean share of the nonpartisan polls of the same
race and year (final 60 days, general stage). This needs no results, only polls; it is estimated on the training
years of each fit (years < the backtest year), and the poll average is corrected by
  poll_share' = poll_share - SPONSOR_SCALE * (frac_D * bias_D + frac_R * bias_R)
where frac_* are the weight fractions of D- and R-sponsored polls in the race average (features.summarise_polls).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from common import CLEAN, ELECTION_DAY
from features import POLL_WINDOW_DAYS, sponsor_side

SPONSOR_SCALE = 0.0    # 0 = v2 (no shift); select_v3.py tests 0.5 and 1


@lru_cache(maxsize=None)
def _house_effects() -> pd.DataFrame:
    p = pd.read_csv(CLEAN / "polls_hist.csv", low_memory=False)
    p = p[(p["stage"] == "general") & (p["is_hypothetical"] != True) & p["year"].isin(ELECTION_DAY)].copy()
    p["special"] = p["special"].astype(str).str.lower().eq("true").astype(int)
    days = (pd.to_datetime(p["year"].map(ELECTION_DAY)) - pd.to_datetime(p["end_date"], errors="coerce")).dt.days
    p = p[days.between(0, POLL_WINDOW_DAYS)].copy()
    d = pd.to_numeric(p["dem_pct"], errors="coerce")
    r = pd.to_numeric(p["rep_pct"], errors="coerce")
    p["share"] = d / (d + r)
    p = p[p["share"].between(0.02, 0.98)]
    p["side"] = sponsor_side(p["partisan_sponsor"])
    keys = ["year", "race_id", "special"]
    neutral = p[p["side"] == ""].groupby(keys)["share"].mean().rename("neutral")
    p = p.join(neutral, on=keys)
    p = p[(p["side"] != "") & p["neutral"].notna()]
    return p.assign(rel=p["share"] - p["neutral"])[["year", "side", "rel"]]


def sponsor_bias(train_years: tuple) -> tuple[float, float]:
    """(bias of D-sponsored, bias of R-sponsored polls) in two-party D share, from the training years' polls."""
    e = _house_effects()
    e = e[e["year"].isin(train_years)]
    by = e.groupby("side")["rel"].mean()
    return float(by.get("D", 0.0)), float(by.get("R", 0.0))


def shift_polls(rows: pd.DataFrame, bias: tuple[float, float] | None) -> pd.DataFrame:
    """Copy of rows with the sponsor-corrected poll_share (no-op when the scale is 0 or no bias given)."""
    if bias is None or SPONSOR_SCALE == 0 or "poll_frac_d" not in rows:
        return rows
    fd = rows["poll_frac_d"].fillna(0.0)
    fr = rows["poll_frac_r"].fillna(0.0)
    shift = SPONSOR_SCALE * (fd * bias[0] + fr * bias[1])
    return rows.assign(poll_share=np.clip(rows["poll_share"] - shift, 0.02, 0.98))
