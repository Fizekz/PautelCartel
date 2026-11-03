"""v2: the other raters' disagreement with the core three (Cook, Sabato, Inside).

rgap = mean of the OTHER outlets' numeric ratings that rated the race (same -3 Safe D .. +3 Safe R scale) minus the
core-3 mean rating; 0 where no other outlet rated it (mostly safe seats). 538's own rating (other_fte) is never
used: it is the benchmark. History (2010-2024) has RCP every year plus a changing mix (CQ, NYT, Daily Kos,
DDHQ 2020-24, Economist 2020-22, Elections Daily, Split Ticket 2024, Fox, Politico...). 2026 has DDHQ, Economist,
FiftyPlusOne, Fox, NBC, RCP, Silver Bulletin and Split Ticket, the faster-moving quantitative outlets the cross-model
investigation flagged (they sit ~0.38 rating points to the D side of the core three in 2026 House races).
"""
from __future__ import annotations

import pandas as pd

from common import read

EXCLUDE = {"other_fte"}


def _gap(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.startswith("other_") and c not in EXCLUDE]
    others = df[cols].apply(pd.to_numeric, errors="coerce")
    return (others.mean(axis=1) - df["mean_rating"]).fillna(0.0)


def rgap_hist() -> pd.DataFrame:
    r = read("ratings_hist.csv")
    return r.assign(rgap=_gap(r))[["year", "race_id", "rgap"]]


def rgap_2026() -> pd.DataFrame:
    r = read("ratings_2026.csv")
    return r.assign(rgap=_gap(r))[["race_id", "rgap"]]
