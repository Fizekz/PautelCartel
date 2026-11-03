"""Race poll averages (two-party D-R margin, points) for history and 2026.

Window: polls ending within WINDOW_DAYS before election day (2010-2016 archives only hold ~60 days, so 2018+ are
cut the same way). Weight = recency half-life HALF_LIFE days x sqrt(min(n, 1500)) / pollster-repeat penalty.
Partisan polls are shifted SPONSOR_SHIFT points against the sponsor. n_eff = sum(w)^2 / sum(w^2) (Kish).
For 2026, the "D side" can be an independent (NE/ID/SD/MT-SEN, AK-AL, CA-06): set via DSIDE_IND.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import ELECTION_DAY, last_name, read

WINDOW_DAYS = 60
# v2: where the archive allows (2018+; 2010-16 538 raw_polls keep only ~60 days) polls up to LONG_WINDOW days old
# are used: in 2020/2024 the sd of the poll miss is ~6.3 (House) / 4.7-5.3 (Senate) for polls 0-60 days old and
# ~6.2 / 5.5 at 60-120 days, so summer polls are nearly as informative (cross-model investigation, OH-07 etc.).
LONG_WINDOW = 120
LONG_FROM_YEAR = 2018
HALF_LIFE = 14.0
SPONSOR_SHIFT = 2.0
DSIDE_IND = {"NE-SEN": "Osborn", "ID-SEN": "Achilles", "SD-SEN": "Bengs", "MT-SEN": "Bodnar", "AK-AL": "Hill"}
RSIDE_IND = {"CA-06": "Kiley"}  # Kiley (I, ex-R incumbent) takes the R slot against Pan (D)


def weighted(df: pd.DataFrame, eday: pd.Timestamp) -> dict:
    age = (eday - pd.to_datetime(df["end_date"])).dt.days.clip(lower=0)
    n = df["sample_size"].fillna(500).clip(100, 1500)
    w = 0.5 ** (age / HALF_LIFE) * np.sqrt(n)
    reps = df.groupby("pollster")["pollster"].transform("size")
    w = w / np.sqrt(reps)
    m = np.average(df["m2p"], weights=w)
    return {"poll_margin": float(m), "poll_n": int(len(df)), "poll_neff": float(w.sum() ** 2 / (w ** 2).sum()),
            "poll_last_age": int(age.min())}


def two_party_margin(d: pd.Series, r: pd.Series) -> pd.Series:
    return 100 * (d - r) / (d + r)


def sponsor_shift(sponsor: pd.Series) -> pd.Series:
    s = sponsor.fillna("").astype(str).str.upper()
    return np.where(s.str.startswith("D") & ~s.str.contains("/"), -SPONSOR_SHIFT,
                    np.where(s.str.startswith("R"), SPONSOR_SHIFT, 0.0))


POP_RANK = {"lv": 0, "v": 1, "rv": 2, "a": 3}


def one_row_per_poll(p: pd.DataFrame) -> pd.DataFrame:
    """v2 (audit bug 2): 538 stores each QUESTION of a poll (LV/RV, with/without a third party, RCV rounds) as a row.
    Keep one row per poll_id: the RCV-reallocated (final-round) question if the poll has one, then the best
    population (lv > v > rv > a), then the mean margin of what is left."""
    p = p.copy()
    p["pid"] = p["poll_id"].fillna(-p.index.to_series()).astype(int)
    p["rcv"] = (p["ranked_choice_reallocated"] == True).astype(int)  # noqa: E712
    p["prank"] = p["population"].map(POP_RANK).fillna(4)
    keys = ["year", "race_id", "chamber", "special", "pid"]
    p = p[p["rcv"] == p.groupby(keys)["rcv"].transform("max")]
    p = p[p["prank"] == p.groupby(keys)["prank"].transform("min")]
    agg = p.groupby(keys).agg(m2p=("m2p", "mean"), end_date=("end_date", "first"), pollster=("pollster", "first"),
                              sample_size=("sample_size", "first"), eday=("eday", "first")).reset_index()
    return agg


def history_polls(dedupe: bool = True, ind: dict | None = None) -> pd.DataFrame:
    """ind: {(year, race_id): surname} of major D-side independents whose share (other_pct) replaces dem_pct."""
    p = read("polls_hist.csv")
    p = p[(p["stage"] == "general") & (p["is_hypothetical"] != True)].copy()  # noqa: E712
    for (yr, rid), surname in (ind or {}).items():
        sel = (p["year"] == yr) & (p["race_id"] == rid)
        hit = sel & p["other_cand"].astype(str).str.lower().str.contains(surname.lower())
        p.loc[sel & ~hit, "dem_pct"] = np.nan
        p.loc[hit, "dem_pct"] = p.loc[hit, "other_pct"]
    p = p[p["dem_pct"].notna() & p["rep_pct"].notna() & (p["dem_pct"] + p["rep_pct"] > 0)]
    p["eday"] = pd.to_datetime(p["year"].map(ELECTION_DAY))
    p["end"] = pd.to_datetime(p["end_date"], errors="coerce")
    window = np.where(p["year"] >= LONG_FROM_YEAR, LONG_WINDOW, WINDOW_DAYS)
    p = p[(p["end"] < p["eday"]) & ((p["eday"] - p["end"]).dt.days <= window)]
    p["m2p"] = two_party_margin(p["dem_pct"], p["rep_pct"]) + sponsor_shift(p["partisan_sponsor"])
    p["special"] = p["special"].fillna(False).astype(bool)
    if dedupe:
        p = one_row_per_poll(p)
    rows = []
    for (year, rid, chamber, special), g in p.groupby(["year", "race_id", "chamber", "special"]):
        rows.append({"year": year, "race_id": rid, "chamber": chamber, "special": bool(special),
                     **weighted(g, g["eday"].iloc[0])})
    return pd.DataFrame(rows)


def dside_pct(row: pd.Series) -> float:
    rid = row["race_id"]
    if rid in DSIDE_IND:
        if isinstance(row["top_other_cand"], str) and last_name(row["top_other_cand"]) == DSIDE_IND[rid].lower():
            return row["top_other_pct"]
        return np.nan
    return row["dem_pct"]


def _detail_share(detail: str, surname: str) -> float:
    for part in detail.split(";"):
        if surname.lower() in part.lower() and ":" in part:
            try:
                return float(part.rsplit(":", 1)[1])
            except ValueError:
                return np.nan
    return np.nan


def n_named(detail: str) -> int:
    return sum(1 for part in str(detail).split(";") if ":" in part)


def final_round_only(p: pd.DataFrame) -> pd.DataFrame:
    """v2 (audit, AK-AL): in RCV races keep each poll's final-round (two-candidate) question when it has one;
    otherwise drop intermediate rounds 2/3 as v1 did. A poll = (race, pollster, end_date)."""
    p = p.copy()
    p["n_named"] = p["cand_detail"].map(n_named)
    rcv_race = p["race_id"].isin(p.loc[p["rcv_round"].notna(), "race_id"].unique())
    keys = ["race_id", "pollster", "end_date"]
    final = rcv_race & (p["n_named"] == 2)
    has_final = final.groupby([p[k] for k in keys]).transform("max").astype(bool)
    keep = np.where(rcv_race & has_final, final, ~p["rcv_round"].astype(str).isin(["2", "3", "2.0", "3.0"]))
    return p[keep]


def retag_sponsor(p: pd.DataFrame) -> pd.Series:
    """v2: fill missing sponsor tags the file leaves blank but its own note (or the same pollster elsewhere) gives:
    RedBox Research / A Fight Worth Having is D-sponsored (tagged D in 16 of 18 rows, blank in MI-04 and MI-10);
    notes naming a progressive / Democratic-aligned sponsor -> D, conservative / Republican-aligned -> R."""
    tag = p["partisan_sponsor"].copy()
    note = p["sponsor_note"].fillna("").str.lower()
    blank = tag.isna()
    d = p["pollster"].fillna("").str.contains("RedBox", case=False) | note.str.contains(
        "fight worth having|progressive|democratic-aligned|democratic super pac")
    r = note.str.contains("conservative|republican-aligned|republican super pac")
    tag[blank & d] = "D"
    tag[blank & r & ~d] = "R"
    return tag


def polls_2026() -> pd.DataFrame:
    p = read("polls_2026.csv")
    usable = ((p["nominee_match"] == True) | (p["nominee_match"].isna() & (p["is_hypothetical"] == False)))  # noqa: E712
    p = p[usable].copy()
    p = final_round_only(p)
    p["partisan_sponsor"] = retag_sponsor(p)
    eday = pd.Timestamp(ELECTION_DAY[2026])
    p["end"] = pd.to_datetime(p["end_date"], errors="coerce")
    p = p[(eday - p["end"]).dt.days <= LONG_WINDOW]
    p["d_side"] = p.apply(dside_pct, axis=1)
    for rid, surname in RSIDE_IND.items():
        sel = p["race_id"] == rid
        p.loc[sel, "rep_pct"] = p.loc[sel, "cand_detail"].map(lambda d: _detail_share(str(d), surname))
    p = p[p["d_side"].notna() & p["rep_pct"].notna() & (p["d_side"] + p["rep_pct"] > 0)]
    p["m2p"] = two_party_margin(p["d_side"], p["rep_pct"]) + sponsor_shift(p["partisan_sponsor"]) * 0.5
    rows = []
    for (rid, chamber), g in p.groupby(["race_id", "chamber"]):
        g = g.drop_duplicates(["pollster", "end_date"])
        rows.append({"year": 2026, "race_id": rid, "chamber": chamber, "special": False, **weighted(g, eday)})
    return pd.DataFrame(rows)
