"""RESEARCH Q1(c), v5 test: national generic ballot and House vote 1998-2024, for a year-level model that can use
cycles before 2010 (which have no ratings, so they cannot enter the race-level fit).

gb : D minus R, mean of each pollster's latest non-partisan House-G-US poll ending in the 21 days before election day
     (538 pollster-ratings raw_polls.csv; the shared history_national.csv rule) for 1998-2008; history_national.csv
     for 2010-2024 (the same rule, so the two parts are comparable).
N  : national House two-party margin, D minus R (MIT 1976-2024 House returns, sum over November generals).
"""
import numpy as np
import pandas as pd

from common import RAW, FEAT, WORK, ELECTION_DATES

DATES = {**ELECTION_DATES, 1998: "1998-11-03", 2000: "2000-11-07", 2002: "2002-11-05", 2004: "2004-11-02",
         2006: "2006-11-07"}


def build():
    rp = pd.read_csv(RAW / "history/national/538_raw_polls.csv", low_memory=False)
    rp = rp[rp.type_simple == "House-G-US"].copy()
    rp["end"] = pd.to_datetime(rp.polldate)
    rp["m"] = np.where(rp.cand1_party == "DEM", rp.cand1_pct - rp.cand2_pct, rp.cand2_pct - rp.cand1_pct)
    rows = []
    for y in range(1998, 2010, 2):
        ed = pd.Timestamp(DATES[y])
        g = rp[(rp.cycle == y) & (rp.end <= ed) & (rp.end >= ed - pd.Timedelta(days=21))]
        g = g[g.partisan.isna() | (g.partisan == "")]
        g = g.sort_values("end").groupby("pollster").tail(1)
        rows.append(dict(year=y, gb=float(g.m.mean()), gb_pollsters=len(g)))
    hn = pd.read_csv(FEAT / "history_national.csv")
    for r in hn.itertuples():
        rows.append(dict(year=int(r.year), gb=-float(r.generic_ballot_margin_r_minus_d),
                         gb_pollsters=int(r.generic_ballot_n_pollsters)))
    out = pd.DataFrame(rows)
    h = pd.read_csv(RAW / "history/mit_1976-2024-house.tab", low_memory=False)
    h = h[(h.stage == "GEN") & (h.year >= 1998) & (h.special.astype(str).str.upper() != "TRUE")]
    p = h.party.fillna("").str.upper()
    d = h[p.str.startswith("DEMOCRAT")].groupby("year").candidatevotes.sum()
    r = h[p.str.startswith("REPUBLICAN")].groupby("year").candidatevotes.sum()
    N = (100 * (d - r) / (d + r)).rename("N").reset_index()
    out = out.merge(N, on="year", how="left")
    out["midterm"] = (out.year % 4 == 2).astype(int)
    out.to_csv(WORK / "national_ext.csv", index=False)
    return out


if __name__ == "__main__":
    print(build().round(2).to_string(index=False))
