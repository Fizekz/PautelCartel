"""Race-level poll averages, one method for history and 2026.

Window: polls whose field period ends in the 21 days before election day (the 538 raw_polls file
only keeps that window, so the 2026 average uses the same window to stay comparable).
Margin: two-party, (Dside - R) / (Dside + R) * 100, where Dside is the Democrat or, in a race
where the Democrat is not the main opponent, the independent who is (e.g. Orman 2014, Osborn).
Weights: sqrt(sample size, capped at 1500) * exp(-days_before_election / 10); partisan polls get
half weight and are shifted 3 pts against their sponsor. Each pollster's polls are averaged
first, then pollsters are averaged with the sum of their weights (capped, so one tracker can't
dominate). n_pollsters is the effective count used for the poll's weight in the model.

History source: 538 pollster-ratings raw_polls.csv (House-G / Sen-G, 2010-2022).
2026 source   : polls_2026.csv (Wikipedia poll tables, compiled by the shared data job).
"""
import re

import numpy as np
import pandas as pd

from common import RAW, FEAT, WORK, ELECTION_DATES, norm_name, surname

PARTISAN_SHIFT = 3.0
DECAY_DAYS = 10.0


def _aggregate(df):
    """df: rows with race_id, year, pollster, days, n, margin, partisan(+1 D/-1 R/0)."""
    df = df.copy()
    df["n"] = df.n.fillna(500).clip(100, 1500)
    df["w"] = np.sqrt(df.n) * np.exp(-df.days / DECAY_DAYS) * np.where(df.partisan != 0, 0.5, 1.0)
    df["m_adj"] = df.margin - PARTISAN_SHIFT * df.partisan
    out = []
    if "dside_party" not in df:
        df["dside_party"] = "DEM"
    for (y, rid, dsp), g in df.groupby(["year", "race_id", "dside_party"]):
        per = g.groupby("pollster").apply(
            lambda x: pd.Series({"m": np.average(x.m_adj, weights=x.w), "w": min(x.w.sum(), 2.5 * x.w.max())}),
            include_groups=False)
        m = np.average(per.m, weights=per.w)
        out.append(dict(year=y, race_id=rid, dside_party=dsp, poll_margin=m, poll_n=len(g), poll_pollsters=len(per),
                        poll_sd_between=float(per.m.std()) if len(per) > 1 else np.nan,
                        poll_days_min=float(g.days.min())))
    return pd.DataFrame(out)


def history_polls(hist_races):
    rp = pd.read_csv(RAW / "history/national/538_raw_polls.csv", low_memory=False)
    rp = rp[rp.type_simple.isin(["House-G", "Sen-G"]) & (rp.cycle >= 2010) & (rp.cycle <= 2024)].copy()
    rp = rp[rp.cycle % 2 == 0]
    at_large = set(hist_races.loc[hist_races.district == "AL", "state"])

    def rid(row):
        race = row.race
        st = str(row.location).split("-")[0]
        if "_Sen-GS_" in race:
            return f"{st}-SEN-SPECIAL"
        if "_Sen-G_" in race:
            return f"{st}-SEN"
        if "_House-G_" in race:
            d = race.split("_")[-1].split("-")[-1]
            try:
                d = int(d)
            except ValueError:
                return None
            if st in at_large and d <= 1:
                return f"{st}-AL"
            return f"{st}-{d:02d}"
        return None

    rp["race_id"] = rp.apply(rid, axis=1)
    rp = rp[rp.race_id.notna()]
    rows = []
    for r in rp.itertuples():
        parties = {r.cand1_party: r.cand1_pct, r.cand2_party: r.cand2_pct}
        if "REP" not in parties or len(parties) < 2:
            continue  # same-party or no Republican
        other = [p for p in parties if p != "REP"][0]
        if other not in ("DEM", "IND"):
            continue
        d, rr = parties[other], parties["REP"]
        if d + rr <= 0:
            continue
        ed = pd.Timestamp(ELECTION_DATES[int(r.cycle)])
        days = (ed - pd.Timestamp(r.polldate)).days
        part = {"DEM": 1, "REP": -1}.get(r.partisan, 0) if isinstance(r.partisan, str) else 0
        rows.append(dict(year=int(r.cycle), race_id=r.race_id, pollster=r.pollster, days=days, n=r.samplesize,
                         margin=100 * (d - rr) / (d + rr), partisan=part, dside_party=other))
    df = pd.DataFrame(rows)
    df = df[(df.days >= 0) & (df.days <= 21)]
    return _aggregate(df)


def _parse_cand_pcts(s):
    out = {}
    if not isinstance(s, str):
        return out
    for part in s.split(";"):
        m = re.match(r"\s*(.+?)\s*\(([^)]*)\)\s*=\s*([\d.]+)", part)
        if m:
            out[m.group(1).strip()] = (m.group(2).strip(), float(m.group(3)))
    return out


def polls_2026(dside_names, rside_names):
    """dside_names / rside_names: dict race_id -> candidate name for the two main sides."""
    p = pd.read_csv(FEAT / "polls_2026.csv")
    ed = pd.Timestamp(ELECTION_DATES[2026])
    p["days"] = (ed - pd.to_datetime(p.end_date)).dt.days
    p = p[(p.days >= 0) & (p.days <= 21)]
    p = p[p.population.isin(["lv", "rv", "v"])]
    rows = []
    for r in p.itertuples():
        if r.race_id not in dside_names or r.race_id not in rside_names:
            continue
        cp = _parse_cand_pcts(r.cand_pcts)
        dn, rn = dside_names[r.race_id], rside_names[r.race_id]

        def find(name):
            key = surname(name)
            for k, (_, v) in cp.items():
                if surname(k) == key:
                    return v
            return None

        d, rr = find(dn), find(rn)
        if d is None or rr is None:
            # fall back to the dem/rep columns when the D-side is the Democrat
            if isinstance(r.dem_candidate, str) and surname(r.dem_candidate) == surname(dn):
                d = r.dem_pct
            if isinstance(r.rep_candidate, str) and surname(r.rep_candidate) == surname(rn):
                rr = r.rep_pct
        if d is None or rr is None or pd.isna(d) or pd.isna(rr) or d + rr <= 0:
            continue
        part = {"D": 1, "R": -1}.get(r.partisan_sponsor, 0) if isinstance(r.partisan_sponsor, str) else 0
        rows.append(dict(year=2026, race_id=r.race_id, pollster=r.pollster, days=r.days, n=r.sample_size,
                         margin=100 * (d - rr) / (d + rr), partisan=part))
    df = pd.DataFrame(rows)
    return _aggregate(df)


if __name__ == "__main__":
    h = pd.read_csv(FEAT / "history_races.csv")
    hp = history_polls(h)
    hp.to_csv(WORK / "polls_hist.csv", index=False)
    print(hp.groupby("year").size())
