"""Race-level poll averages: recency, sample size, population, partisan sponsor, house effects, track record.

Every poll is reduced to one number, `share` = side A's share of the two-way (A + B) vote, where side A is the
Democrat (or the main anti-Republican independent) and side B the Republican. The same code averages the 538
archive (backtests) and the 2026 Wikipedia/VoteHub polls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (CLEAN, ELECTION_DAY, PARTISAN_WEIGHT, POLL_HALFLIFE_DAYS, POLL_WINDOW_DAYS, POP_WEIGHT,
                    last_name, norm_pollster)

POP_RANK = {"lv": 0, "v": 1, "rv": 2, "a": 3}
HOUSE_EFFECT_SHRINK = 3.0      # pseudo-polls pulling a pollster's house effect to 0
SPONSOR_SHRINK = 10.0          # pseudo-polls pulling the sponsor offset to 0
TRACK_SHRINK = 8.0             # pseudo-polls pulling a pollster's error variance to the median
USE_HOUSE_EFFECTS = True       # subtract within-cycle pollster house effects
USE_SPONSOR_OFFSET = False     # v2: off (pre-2018 LOYO 0.0755 vs 0.0765 with it); partisan polls keep weight 0.6


def _standardise(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Common columns: race_id, pollster, pkey, end, days, n, pop, sponsor, share."""
    eday = pd.Timestamp(ELECTION_DAY[year])
    out = df.copy()
    out["end"] = pd.to_datetime(out["end_date"], errors="coerce")
    out["days"] = (eday - out["end"]).dt.days
    out["n"] = pd.to_numeric(out["sample_size"], errors="coerce").fillna(500.0).clip(100, 3000)
    out["pop"] = out["population"].fillna("lv").str.lower().where(lambda s: s.isin(POP_WEIGHT), "lv")
    sp = out["partisan_sponsor"].fillna("").astype(str).str.upper()
    out["sponsor"] = np.select([sp.str.startswith("D"), sp.str.startswith("R")], ["D", "R"], "")
    out.loc[sp.str.contains("/"), "sponsor"] = ""
    out["pkey"] = out["pollster"].map(norm_pollster)
    out = out[(out["days"] >= 0) & (out["days"] <= POLL_WINDOW_DAYS) & out["share"].between(0.05, 0.95)]
    # one row per (race, pollster, end date): best population, then mean of the remaining versions
    out["pop_rank"] = out["pop"].map(POP_RANK)
    best = out.groupby(["race_id", "pkey", "end"])["pop_rank"].transform("min")
    out = out[out["pop_rank"] == best]
    agg = out.groupby(["race_id", "pkey", "end"], as_index=False).agg(
        pollster=("pollster", "first"), days=("days", "first"), n=("n", "max"), pop=("pop", "first"),
        sponsor=("sponsor", "first"), share=("share", "mean"))
    return agg


def rcv_final_only(p: pd.DataFrame) -> pd.DataFrame:
    """In RCV races (ME 2018+, AK 2022+) keep only a poll's final two-candidate round when it reports one.

    The 538 archive stores first-preference, intermediate and final rounds as separate rows of one poll_id; v1
    averaged all of them as head-to-heads (AK-AL 2022: Peltola 48 / Palin 25 first-round rows)."""
    if p.empty:
        return p
    oc = p["other_cand"].fillna("")
    named = (p["dem_pct"].notna().astype(int) + p["rep_pct"].notna().astype(int)
             + (p["other_cand"].notna() & ~oc.str.contains(";")).astype(int))
    tot = p["dem_pct"].fillna(0) + p["rep_pct"].fillna(0) + p["other_pct"].fillna(0)
    final = (named == 2) & (tot >= 97) & ~oc.str.contains(";")
    st = p["race_id"].str[:2]
    rcv = (st.eq("ME") & p["year"].ge(2018)) | (st.eq("AK") & p["year"].ge(2022))
    has_final = final.groupby([p["year"], p["race_id"], p["poll_id"]]).transform("any")
    return p[~rcv | ~has_final | final]


def hist_polls(year: int, results: pd.DataFrame) -> pd.DataFrame:
    """538-archive polls for `year`, matched to MIT nominees (both surnames must match)."""
    p = pd.read_csv(CLEAN / "polls_hist.csv")
    p = p[(p["year"] == year) & (p["stage"] == "general") & ~p["is_hypothetical"].fillna(False).astype(bool)]
    p = rcv_final_only(p)
    p = p[p["dem_pct"].notna() & p["rep_pct"].notna()].copy()
    res = results[results["year"] == year]
    specials = set(res.loc[res["race_id"].str.endswith("-SEN-S"), "race_id"])
    sp = p["special"].fillna(False).astype(bool)
    cand = p["race_id"] + "-S"
    p["race_id"] = np.where(sp & cand.isin(specials), cand, p["race_id"])
    noms = res.set_index("race_id")[["dem_cand", "rep_cand"]]
    p = p[p["race_id"].isin(noms.index)]
    dl = p["race_id"].map(noms["dem_cand"]).map(lambda x: last_name(x) if isinstance(x, str) else "")
    rl = p["race_id"].map(noms["rep_cand"]).map(lambda x: last_name(x) if isinstance(x, str) else "")
    ok = (p["dem_cand"].map(last_name) == dl) & (p["rep_cand"].map(last_name) == rl)
    p = p[ok].copy()
    p["share"] = p["dem_pct"] / (p["dem_pct"] + p["rep_pct"])
    return _standardise(p, year)


def pollster_track(years: list[int], results: pd.DataFrame) -> dict[str, float]:
    """Weight multiplier per pollster key from its past squared error vs results (only `years`)."""
    rows = []
    for y in years:
        hp = hist_polls(y, results)
        act = results[results["year"] == y].set_index("race_id")["dem_share_2p"]
        hp = hp[hp["race_id"].isin(act.index)]
        err = hp["share"] - hp["race_id"].map(act)
        # remove the cycle-wide bias so the track record measures pollster noise, not the year
        err = err - err.mean()
        rows.append(pd.DataFrame({"pkey": hp["pkey"], "se": err ** 2}))
    if not rows:
        return {}
    allp = pd.concat(rows)
    med = allp["se"].median() if len(allp) else 0.0
    base = allp["se"].mean()
    g = allp.groupby("pkey")["se"].agg(["sum", "count"])
    shrunk = (g["sum"] + TRACK_SHRINK * base) / (g["count"] + TRACK_SHRINK)
    mult = (base / shrunk).clip(0.5, 1.6)
    del med
    return mult.to_dict()


def _house_effects(p: pd.DataFrame, w: np.ndarray) -> tuple[pd.Series, dict]:
    """Iteratively estimate per-pollster house effects and D/R sponsor offsets within the cycle."""
    adj = p["share"].copy()
    house = pd.Series(0.0, index=p["pkey"].unique())
    sponsor = {"D": 0.0, "R": 0.0}
    for _ in range(5):
        avg = (adj * w).groupby(p["race_id"]).sum() / pd.Series(w, index=p.index).groupby(p["race_id"]).sum()
        dev = p["share"] - p["race_id"].map(avg)
        multi = p.groupby("race_id")["pkey"].transform("nunique") > 1
        for s in ("D", "R"):
            m = multi & (p["sponsor"] == s)
            sponsor[s] = float(dev[m].sum() / (m.sum() + SPONSOR_SHRINK)) if m.any() else 0.0
        dev2 = dev - p["sponsor"].map(sponsor).fillna(0.0)
        g = dev2[multi].groupby(p.loc[multi, "pkey"]).agg(["sum", "count"])
        house = (g["sum"] / (g["count"] + HOUSE_EFFECT_SHRINK)).reindex(house.index).fillna(0.0)
        if not USE_SPONSOR_OFFSET:
            sponsor = {"D": 0.0, "R": 0.0}
        if not USE_HOUSE_EFFECTS:
            house = house * 0.0
        adj = p["share"] - p["sponsor"].map(sponsor).fillna(0.0) - p["pkey"].map(house).fillna(0.0)
    return adj, {"sponsor": sponsor, "house": house}


def aggregate(p: pd.DataFrame, track: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """Per race: weighted poll average of A's two-way share and effective number of polls."""
    if p.empty:
        return pd.DataFrame(columns=["race_id", "poll_share", "poll_neff", "poll_n", "poll_days"]), {}
    p = p.copy()
    rec = np.exp(-np.log(2) * p["days"] / POLL_HALFLIFE_DAYS)
    size = np.sqrt(p["n"] / 600.0)
    popw = p["pop"].map(POP_WEIGHT).fillna(0.8)
    part = np.where(p["sponsor"] != "", PARTISAN_WEIGHT, 1.0)
    qual = p["pkey"].map(track).fillna(1.0)
    # a pollster's k-th most recent poll in a race counts 1/k of a poll (tracking polls do not swamp the average)
    k = p.groupby(["race_id", "pkey"])["days"].rank(method="first")
    w = (rec * size * popw * part * qual / k).to_numpy()
    adj, effects = _house_effects(p, w)
    p["adj"] = adj
    p["w"] = w
    g = p.groupby("race_id")
    out = pd.DataFrame({
        "poll_share": g.apply(lambda d: np.average(d["adj"], weights=d["w"])),
        "poll_neff": g["w"].sum(),               # in units of "one fresh 600-person LV poll"
        "poll_n": g.size(),
        "poll_days": g["days"].min(),
    }).reset_index()
    return out, effects
