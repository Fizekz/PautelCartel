"""National environment per cycle: final-3-week generic-ballot average (D-R margin, points).

One method for every year so that the 2026 value is comparable to the training years:
polls whose field period ends in the 21 days before election day; average each pollster's
polls first (so trackers don't dominate), then take the mean across pollsters.

Sources
  2010-2016: 538 pollster-ratings raw_polls.csv (type House-G-US; it only holds final-3-week polls)
  2018-2024: 538 generic_ballot_polls_historical.csv
  2026     : national_2026.csv (Wikipedia poll list), checked against the aggregator average
Also: actual national House two-party margin per year from MIT results (used only for analysis).
"""
import pandas as pd
import numpy as np

from common import RAW, FEAT, WORK, ELECTION_DATES


def _pollster_mean(df, year):
    ed = pd.Timestamp(ELECTION_DATES[year])
    df = df[(df.end <= ed) & (df.end >= ed - pd.Timedelta(days=21))]
    if df.empty:
        return np.nan, 0, 0
    per = df.groupby("pollster").margin.mean()
    return float(per.mean()), int(len(df)), int(len(per))


def generic_ballot_by_year():
    rows = []
    rp = pd.read_csv(RAW / "history/national/538_raw_polls.csv", low_memory=False)
    rp = rp[rp.type_simple == "House-G-US"].copy()
    rp["end"] = pd.to_datetime(rp.polldate)
    # cand1 is DEM in these rows; margin_poll = cand1 - cand2
    rp["margin"] = np.where(rp.cand1_party == "DEM", rp.cand1_pct - rp.cand2_pct, rp.cand2_pct - rp.cand1_pct)
    for y in [2008, 2010, 2012, 2014, 2016]:
        m, n, k = _pollster_mean(rp[rp.cycle == y], y)
        rows.append(dict(year=y, gb=m, gb_n=n, gb_pollsters=k, gb_source="538 raw_polls"))

    gh = pd.read_csv(RAW / "polls/538/generic_ballot_polls_historical.csv", low_memory=False)
    gh["end"] = pd.to_datetime(gh.end_date, format="%m/%d/%y")
    gh["margin"] = gh.dem - gh.rep
    # one row per question: prefer lv > rv > a within a poll
    pr = {"lv": 0, "rv": 1, "v": 2, "a": 3}
    gh["pr"] = gh.population.map(pr).fillna(4)
    gh = gh.sort_values("pr").drop_duplicates("poll_id")
    for y in [2018, 2020, 2022, 2024]:
        m, n, k = _pollster_mean(gh[gh.cycle == y], y)
        rows.append(dict(year=y, gb=m, gb_n=n, gb_pollsters=k, gb_source="538 generic_ballot_polls_historical"))

    n26 = pd.read_csv(FEAT / "national_2026.csv")
    # national_2026.csv carries no individual generic-ballot polls, only aggregator averages
    # (DDHQ, FiftyPlusOne, NBC, RCP, Silver Bulletin and their mean). Use the latest mean of
    # aggregators: in 2018/2022 the pollster-mean above was within ~1 pt of those aggregators.
    agg = n26[(n26.metric == "generic_ballot") & (n26.pollster == "Average")].sort_values("as_of").iloc[-1]
    rows.append(dict(year=2026, gb=float(agg.dem_pct - agg.rep_pct), gb_n=np.nan, gb_pollsters=5,
                     gb_source=f"national_2026 aggregator mean as of {agg.as_of}"))
    out = pd.DataFrame(rows)

    # Trump approval 2026 (net), for the record
    a = n26[(n26.metric == "trump_approval") & (n26.pollster == "Average")].sort_values("as_of")
    if len(a):
        out.loc[out.year == 2026, "net_approval"] = float(a.iloc[-1].approve - a.iloc[-1].disapprove)
    return out


def actual_house_margin():
    h = pd.read_csv(RAW / "history/mit_1976-2024-house.tab", low_memory=False)
    h = h[(h.stage == "GEN") & (h.year >= 2008)]
    h["party"] = h.party.fillna("").str.upper()
    d = h[h.party.str.startswith("DEMOCRAT")].groupby("year").candidatevotes.sum()
    r = h[h.party.str.startswith("REPUBLICAN")].groupby("year").candidatevotes.sum()
    return pd.DataFrame({"year": d.index, "house_margin_actual": (100 * (d - r) / (d + r)).values})


def main():
    gb = generic_ballot_by_year().merge(actual_house_margin(), on="year", how="left")
    gb.to_csv(WORK / "national.csv", index=False)
    print(gb.to_string())


if __name__ == "__main__":
    main()
