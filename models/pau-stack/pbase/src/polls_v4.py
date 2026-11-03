"""v4 race-poll averages (method borrowed from kavi-e, METHODS.md section 2, re-implemented here).

History: kavi-e's cleaned 538 archive (data/borrowed/kavi-e/polls_hist.csv): raw_polls for 2010-2016 (final ~60
days only), the full-cycle house/senate_polls_historical files for 2018-2024.
2026: the shared Wikipedia poll file (features/polls_2026.csv). kavi-e's 2026 file also holds VoteHub rows, which
our rules put off limits, so it is not used.

Per poll (one row per poll_id: the ranked-choice final round first, then population lv > v > rv > a, then the mean
of what is left): two-way D-side margin 100 (Dside - R) / (Dside + R); the D-side is the independent in races where
he is the main anti-Republican candidate (other_pct). Partisan polls are shifted SHIFT points against the sponsor
(2026: SHIFT_2026, Wikipedia tags many more polls than 538 did).
Weight = sqrt(min(n, 1500)) * 0.5 ** (days / HALF_LIFE) / sqrt(number of that pollster's polls in the race).
Window: polls ending 0..WINDOW days before election day. Output per race: weighted mean margin `pm4`,
Kish effective size `pn4` = (sum w)^2 / sum w^2, number of pollsters `pk4`, days since the latest poll `pd4`.
"""
import re

import numpy as np
import pandas as pd

from common import ROOT, FEAT, ELECTION_DATES, surname
import polls as P

BORROWED = ROOT / "data" / "borrowed" / "kavi-e"
HALF_LIFE = 14.0
WINDOW = 120
SHIFT = 2.0
SHIFT_2026 = 1.0
POP_RANK = {"lv": 0, "v": 1, "rv": 2, "a": 3}
ALT = {"_wz8": dict(winsor=8.0), "_wz12": dict(winsor=12.0), "_nopop": dict(drop_nopop=True),
       "_w60": dict(window=60), "_w30": dict(window=30), "_h7": dict(half_life=7.0), "_h28": dict(half_life=28.0),
       "_pw": dict(partisan_weight=0.5)}


def _two_way(d, r):
    d, r = np.asarray(d, float), np.asarray(r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d + r > 0, 100 * (d - r) / (d + r), np.nan)


def hist_rows(indep_races):
    """indep_races: set of (year, race_id) whose D-side is an independent (history race ids)."""
    p = pd.read_csv(BORROWED / "polls_hist.csv", low_memory=False)
    p = p[(p.stage == "general") & (p.is_hypothetical.astype(str) != "True")].copy()
    p["race_id"] = np.where((p.chamber == "Senate") & (p.special.astype(str) == "True"),
                            p.race_id + "-SPECIAL", p.race_id)
    p["rcv"] = (p.ranked_choice_reallocated.astype(str) == "True").astype(int)
    p["pr"] = p.population.map(POP_RANK).fillna(4)
    p = p.sort_values(["rcv", "pr"], ascending=[False, True])
    first = p.groupby(["year", "race_id", "poll_id"]).head(1)
    # mean of the rows of equal rank (several questions with the same population)
    key = first[["year", "race_id", "poll_id", "rcv", "pr"]]
    p = p.merge(key, on=["year", "race_id", "poll_id", "rcv", "pr"])
    ind = np.array([(y, r) in indep_races for y, r in zip(p.year, p.race_id)])
    p["dside"] = np.where(ind, p.other_pct, p.dem_pct)
    p["margin"] = _two_way(p.dside, p.rep_pct)
    g = p.groupby(["year", "race_id", "poll_id"]).agg(
        pollster=("pollster", "first"), end_date=("end_date", "first"), n=("sample_size", "mean"),
        sponsor=("partisan_sponsor", "first"), margin=("margin", "mean")).reset_index()
    g["days"] = [(pd.Timestamp(ELECTION_DATES[int(y)]) - pd.Timestamp(e)).days for y, e in zip(g.year, g.end_date)]
    g["part"] = g.sponsor.map({"DEM": 1, "REP": -1}).fillna(0)
    g["shift"] = SHIFT
    return g[g.margin.notna()]


def rows_2026(dside_names, rside_names):
    p = pd.read_csv(FEAT / "polls_2026.csv")
    ed = pd.Timestamp(ELECTION_DATES[2026])
    p["days"] = (ed - pd.to_datetime(p.end_date)).dt.days
    # RedBox Research polls are sponsored by a Democratic super PAC (A Fight Worth Having); the file leaves some
    # untagged (cross-model investigation, data trap)
    p.loc[p.pollster.fillna("").str.contains("RedBox", case=False), "partisan_sponsor"] = "D"
    p["pr"] = p.population.map(POP_RANK).fillna(4)
    p = p.sort_values("pr")
    rows = []
    for r in p.itertuples():
        if r.race_id not in dside_names or r.race_id not in rside_names:
            continue
        cp = P._parse_cand_pcts(r.cand_pcts)
        dn, rn = dside_names[r.race_id], rside_names[r.race_id]

        def find(name):
            key = surname(name)
            for k, (_, v) in cp.items():
                if surname(k) == key:
                    return v
            return None

        d, rr = find(dn), find(rn)
        if d is None or rr is None:
            if isinstance(r.dem_candidate, str) and surname(r.dem_candidate) == surname(dn):
                d = r.dem_pct
            if isinstance(r.rep_candidate, str) and surname(r.rep_candidate) == surname(rn):
                rr = r.rep_pct
        if d is None or rr is None or pd.isna(d) or pd.isna(rr) or d + rr <= 0:
            continue
        rows.append(dict(year=2026, race_id=r.race_id, pollster=r.pollster, end_date=r.end_date,
                         start_date=r.start_date, n=r.sample_size, days=r.days, rcv=int("Ranked" in str(r.wiki_section)),
                         pr=r.pr, margin=float(_two_way(d, rr)),
                         part={"D": 1, "R": -1}.get(r.partisan_sponsor, 0) if isinstance(r.partisan_sponsor, str) else 0))
    df = pd.DataFrame(rows)
    # one row per poll (pollster x field dates): ranked-choice final round first, then the best population
    df = df.sort_values(["rcv", "pr"], ascending=[False, True]).drop_duplicates(
        ["race_id", "pollster", "start_date", "end_date"])
    df["shift"] = SHIFT_2026
    return df


def average(rows, window=WINDOW, half_life=HALF_LIFE, partisan_weight=1.0, winsor=None, drop_nopop=False):
    df = rows[(rows.days >= 0) & (rows.days <= window)].copy()
    if drop_nopop and "pr" in df:       # 2026 only: drop polls with neither a population nor a sample size
        df = df[~((df.pr >= 4) & df.n.isna())]
    if df.empty:
        return pd.DataFrame(columns=["year", "race_id", "pm4", "pn4", "pk4", "pd4"])
    df["m_adj"] = df.margin - df["shift"] * df.part
    if winsor:                          # pull outliers to within +-winsor of the race's median poll
        med = df.groupby(["year", "race_id"]).m_adj.transform("median")
        df["m_adj"] = med + (df.m_adj - med).clip(-winsor, winsor)
    cnt = df.groupby(["year", "race_id", "pollster"]).margin.transform("size")
    df["w"] = np.sqrt(df.n.fillna(500).clip(100, 1500)) * 0.5 ** (df.days / half_life) / np.sqrt(cnt)
    df["w"] *= np.where(df.part != 0, partisan_weight, 1.0)
    out = []
    for (y, rid), g in df.groupby(["year", "race_id"]):
        w = g.w.values
        out.append(dict(year=y, race_id=rid, pm4=float(np.average(g.m_adj, weights=w)),
                        pn4=float(w.sum() ** 2 / (w ** 2).sum()), pk4=int(g.pollster.nunique()),
                        pd4=float(g.days.min())))
    return pd.DataFrame(out)
