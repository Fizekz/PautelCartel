"""v3 poll averages, method borrowed from pau-a (METHODS.md section 2, src/features.py), re-implemented.

Differences from v2's polls.py:
  * window 30 days (v2: 21); 2024 race polls added from the 538 polls page (v2 had none for 2024)
  * latest poll per pollster only; weight sqrt(clip(n, 200, 2000)) * 0.5 ** (age / 14), x 0.5 if partisan
  * partisan polls de-biased by the sponsor type's historical error relative to non-partisan polls:
        bias_type = mean over cycles of (mean(poll - result | type) - mean(poll - result | non-partisan))
    estimated on the cycles a regime may see: "fwd" = earlier cycles, "loyo" = all other cycles,
    "all" = 2010-2024 (for 2026). Default +-2 when no cycle is available.
  * effective n = sum(1 per non-partisan poll, 0.5 per partisan poll)
Margins stay two-way (100 (Dside - R) / (Dside + R)), as in the rest of this model.
"""
import numpy as np
import pandas as pd

from common import RAW, FEAT, ELECTION_DATES, STATE_ABBR, surname

WINDOW = 30
DEFAULT_BIAS = {"DEM": 2.0, "REP": -2.0}


def _two_way(d, r):
    return 100 * (d - r) / (d + r) if d + r > 0 else np.nan


def hist_poll_rows(at_large):
    rows = []
    rp = pd.read_csv(RAW / "history/national/538_raw_polls.csv", low_memory=False)
    rp = rp[rp.type_simple.isin(["House-G", "Sen-G"]) & rp.cycle.isin(range(2010, 2024, 2))]
    rp = rp[(rp.time_to_election >= 0) & (rp.time_to_election <= WINDOW)]
    rp = rp[~rp.race.str.contains("_Sen-R_|_Sen-SR_|_House-R_")]
    for x in rp.itertuples():
        loc = str(x.location)
        st = loc.split("-")[0]
        if x.type_simple == "Sen-G":
            rid = f"{st}-SEN" + ("-SPECIAL" if "_Sen-GS_" in x.race else "")
        else:
            try:
                d = int(loc.split("-")[1])
            except (IndexError, ValueError):
                continue
            rid = f"{st}-AL" if (st in at_large.get(int(x.cycle), set()) and d <= 1) else f"{st}-{d:02d}"
        parties = {x.cand1_party: x.cand1_pct, x.cand2_party: x.cand2_pct}
        if "REP" not in parties or len(parties) < 2:
            continue
        other = [p for p in parties if p != "REP"][0]
        if other not in ("DEM", "IND"):
            continue
        part = {"DEM": "DEM", "REP": "REP"}.get(x.partisan, "NONE") if isinstance(x.partisan, str) else "NONE"
        rows.append(dict(year=int(x.cycle), race_id=rid, dside=other, pollster=x.pollster, ptype=part,
                         n=x.samplesize, age=x.time_to_election, margin=_two_way(parties[other], parties["REP"])))
    # 2024: 538 polls page (long format, one row per answer)
    for kind, ch in (("senate", "Senate"), ("house", "House")):
        d = pd.read_csv(RAW / f"polls/538/{kind}_polls_historical.csv", low_memory=False)
        d = d[(d.cycle == 2024) & (d.stage == "general")].copy()
        d["end"] = pd.to_datetime(d.end_date, format="%m/%d/%y", errors="coerce")
        d["age"] = (pd.Timestamp(ELECTION_DATES[2024]) - d.end).dt.days
        d = d[(d.age >= 0) & (d.age <= WINDOW)]
        abbr = {k.title(): v for k, v in STATE_ABBR.items()}
        for qid, g in d.groupby("question_id"):
            byp = g.groupby("party").pct.max()
            if "REP" not in byp:
                continue
            x = g.iloc[0]
            st = abbr.get(str(x.state))
            if st is None:
                continue
            if ch == "Senate":
                rid = f"{st}-SEN" + ("-SPECIAL" if (st == "NE" and str(x.seat_name) == "Class II") else "")
            else:
                if pd.isna(x.seat_number):
                    continue
                k = int(x.seat_number)
                rid = f"{st}-AL" if (k == 0 or st in at_large.get(2024, set())) else f"{st}-{k:02d}"
            part = {"DEM": "DEM", "REP": "REP"}.get(x.partisan, "NONE") if isinstance(x.partisan, str) else "NONE"
            if "DEM" in byp:
                rows.append(dict(year=2024, race_id=rid, dside="DEM", pollster=x.pollster, ptype=part,
                                 n=x.sample_size, age=x.age, margin=_two_way(byp["DEM"], byp["REP"])))
            oth = byp.drop([p for p in ("REP", "DEM") if p in byp])
            if len(oth):
                rows.append(dict(year=2024, race_id=rid, dside="IND", pollster=x.pollster, ptype=part,
                                 n=x.sample_size, age=x.age, margin=_two_way(oth.max(), byp["REP"])))
    return pd.DataFrame(rows)


def poll_rows_2026(dside_names, rside_names):
    import polls as P
    p = pd.read_csv(FEAT / "polls_2026.csv")
    p["age"] = (pd.Timestamp(ELECTION_DATES[2026]) - pd.to_datetime(p.end_date)).dt.days
    p = p[(p.age >= 0) & (p.age <= WINDOW) & p.population.isin(["lv", "rv", "v"])]
    rows = []
    for r in p.itertuples():
        if r.race_id not in dside_names:
            continue
        cp = P._parse_cand_pcts(r.cand_pcts)
        find = lambda nm: next((v for k, (_, v) in cp.items() if surname(k) == surname(nm)), None)
        d, rr = find(dside_names[r.race_id]), find(rside_names[r.race_id])
        if d is None or rr is None:
            continue
        part = {"D": "DEM", "R": "REP"}.get(r.partisan_sponsor, "NONE") if isinstance(r.partisan_sponsor, str) else "NONE"
        rows.append(dict(year=2026, race_id=r.race_id, dside="X", pollster=r.pollster, ptype=part,
                         n=r.sample_size, age=r.age, margin=_two_way(d, rr)))
    return pd.DataFrame(rows)


def bias_table(rows, results):
    """Per cycle: mean(poll - result) by sponsor type (D-side margins of contested races)."""
    m = rows.merge(results, on=["year", "race_id"], how="inner")
    m["err"] = m.margin - m.y
    return m.groupby(["year", "ptype"]).err.mean().unstack()


def bias_from(bt, years):
    q = bt[bt.index.isin(years)]
    if not len(q) or "NONE" not in q:
        return dict(DEFAULT_BIAS)
    rel = q[["DEM", "REP"]].sub(q["NONE"], axis=0).mean()
    return {k: (float(v) if np.isfinite(v) else DEFAULT_BIAS[k]) for k, v in rel.items()}


def average(rows, bias):
    if rows.empty:
        return pd.DataFrame(columns=["year", "race_id", "pm", "pn"])
    p = rows.copy()
    p["adj"] = p.margin - p.ptype.map(bias).fillna(0.0)
    p = p.sort_values("age").groupby(["year", "race_id", "pollster"], as_index=False).first()
    p["w"] = np.sqrt(p.n.fillna(500).clip(200, 2000)) * 0.5 ** (p.age / 14.0) * np.where(p.ptype == "NONE", 1.0, 0.5)
    p["ne"] = np.where(p.ptype == "NONE", 1.0, 0.5)
    g = p.groupby(["year", "race_id"])
    out = pd.DataFrame({"pm": g.apply(lambda d: np.average(d.adj, weights=d.w), include_groups=False),
                        "pn": g.ne.sum()}).reset_index()
    return out
