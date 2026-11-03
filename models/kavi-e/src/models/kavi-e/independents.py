"""v2: poll treatment of major independent candidates on the D side, estimated from the history it can be.

Challenger independents (Orman KS-SEN 2014, McMullin UT-SEN 2022, Osborn NE-SEN 2024) ran behind their final
polls. The part of that miss that is specific to the independent is the miss minus the year's mean Senate poll
miss (all contested D-v-R Senate races of that year). For a target year only earlier cases are used, and the mean is
shrunk toward 0 with the same K = 2 pseudo-years as national.py: shift = mean * n / (n + 2).
Incumbent independents (King, Sanders) are not shifted.
Actual two-party IND-R margins (official results): KS 2014 Orman 42.5 / Roberts 53.1; UT 2022 McMullin 42.7 /
Lee 53.2; NE 2024 Osborn 46.5 / Fischer 53.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from polls import history_polls

CASES = {(2014, "KS-SEN"): ("Orman", 42.5, 53.1), (2022, "UT-SEN"): ("McMullin", 42.7, 53.2),
         (2024, "NE-SEN"): ("Osborn", 46.5, 53.2)}
INCUMBENT_IND = {(2012, "VT-SEN"): "Sanders", (2018, "VT-SEN"): "Sanders", (2024, "VT-SEN"): "Sanders",
                 (2018, "ME-SEN"): "King", (2024, "ME-SEN"): "King"}
SHRINK_K = 2.0
SIGMA_MULT = 1.25  # extra idiosyncratic sd for a challenger independent: the shift itself is uncertain (se ~2.3)


def hist_ind_map() -> dict[tuple[int, str], str]:
    return {**{k: v[0] for k, v in CASES.items()}, **INCUMBENT_IND}


def case_misses(senate_hist: pd.DataFrame) -> pd.DataFrame:
    hp = history_polls(ind=hist_ind_map())
    hp = hp[(hp["chamber"] == "Senate") & ~hp["special"]]
    base = history_polls()
    base = base[(base["chamber"] == "Senate") & ~base["special"]]
    c = senate_hist.loc[senate_hist["contested"], ["year", "race_id", "margin"]].merge(base[["year", "race_id", "poll_margin"]], on=["year", "race_id"])
    year_miss = (c["margin"] - c["poll_margin"]).groupby(c["year"]).mean()
    rows = []
    for (yr, rid), (name, ind_pct, rep_pct) in CASES.items():
        pm = hp[(hp["year"] == yr) & (hp["race_id"] == rid)]["poll_margin"]
        actual = 100 * (ind_pct - rep_pct) / (ind_pct + rep_pct)
        miss = actual - float(pm.iloc[0])
        rows.append({"year": yr, "race_id": rid, "name": name, "actual": actual, "poll": float(pm.iloc[0]),
                     "miss": miss, "year_mean_miss": float(year_miss.get(yr, 0.0)),
                     "net_miss": miss - float(year_miss.get(yr, 0.0))})
    return pd.DataFrame(rows)


def poll_shift(target_year: int, senate_hist: pd.DataFrame) -> float:
    m = case_misses(senate_hist)
    m = m[m["year"] < target_year]
    if m.empty:
        return 0.0
    return float(m["net_miss"].mean() * len(m) / (len(m) + SHRINK_K))


if __name__ == "__main__":
    from common import WORK
    sh = pd.read_csv(WORK / "senate_hist.csv")
    print(case_misses(sh).round(2).to_string())
    for y in (2018, 2022, 2026):
        print(y, round(poll_shift(y, sh), 2))
