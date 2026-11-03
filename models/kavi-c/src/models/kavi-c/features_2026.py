"""2026 feature table (470 races) and 2026 poll averages, on the same definitions as the history.

Side A is the Democrat, except in the races where the real challenger to the Republican is an independent
(the Democrat withdrew or is a distant third): there side A is that independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import hist_features
import polls
from common import CLEAN, HIST_YEARS, last_name, money_feature

NATIONAL_PRES24 = 0.49250          # MIT national two-party D share 2024 (lean_hist.md)
HOUSE_2024_D_SHARE = 0.5 + (-2.55) / 200.0   # national House two-party D share 2024 (national_hist)

# Independent is the main anti-R candidate (KNOWN DATA TRAPS; candidates_2026.md "forced independents").
IND_SIDE_A = {"NE-SEN": "Dan Osborn", "MT-SEN": "Seth Bodnar", "ID-SEN": "Todd Achilles",
              "SD-SEN": "Brian Bengs", "AK-AL": "Bill Hill"}
# RCV races where the Democrat's first choices transfer to the independent in the final round.
RCV_ADD_DEM_TO_A = {"AK-AL"}
# Side B is an independent instead of a Republican: CA-06, ex-Republican incumbent Kiley (no R on the ballot).
IND_SIDE_B = {"CA-06": "Kevin Kiley"}
# Same-party top-two finals (D vs D, and CA-40 R vs R).
NON_PARTISAN_FINAL = {"CA-04", "CA-07", "CA-11", "CA-12", "CA-14", "CA-29", "CA-34", "CA-37", "CA-40"}
UNCONTESTED_D = {"FL-10", "MA-02", "MA-05", "MA-07", "WI-02", "AZ-03", "NJ-08", "PA-03", "MA-01"}


def generic_env() -> float:
    nat = pd.read_csv(CLEAN / "national_2026.csv")
    g = nat[nat["metric"] == "generic_ballot_avg"]
    return float(g["value_dem"].mean() / (g["value_dem"].mean() + g["value_rep"].mean()) - 0.5)


def usable_polls() -> pd.DataFrame:
    p = pd.read_csv(CLEAN / "polls_2026.csv")
    nm = p["nominee_match"]
    ok = (nm == True) | (nm.isna() & (p["is_hypothetical"] == False))  # noqa: E712
    ok &= ~p["rcv_round"].astype(str).isin(["2", "3", "2.0", "3.0"])
    return p[ok].copy()


def detail_pct(detail: str) -> dict[str, float]:
    """'Pete Ricketts (R): 49; Dan Osborn (I): 43' -> {surname: pct}."""
    out = {}
    for part in str(detail).split(";"):
        if ":" not in part:
            continue
        name, val = part.rsplit(":", 1)
        name = name.split("(")[0].strip()
        try:
            out[last_name(name)] = float(val)
        except ValueError:
            continue
    return out


def poll_shares_2026() -> pd.DataFrame:
    """One row per poll with side-A share, for the D-vs-R and independent-vs-R races."""
    p = usable_polls()
    p = p[~p["race_id"].isin(NON_PARTISAN_FINAL)].copy()
    a = p["dem_pct"].astype(float)
    ind = p["race_id"].isin(IND_SIDE_A)
    ind_a = p["top_other_pct"].astype(float)
    a = np.where(ind, ind_a, a)
    rcv_add = p["race_id"].isin(RCV_ADD_DEM_TO_A)
    a = np.where(rcv_add, ind_a + p["dem_pct"].astype(float).fillna(0.0), a)
    b = p["rep_pct"].astype(float)
    ind_b = p["race_id"].isin(IND_SIDE_B)
    b = np.where(ind_b, p["top_other_pct"].astype(float), b)
    p["share"] = a / (a + b)
    # independent rows must actually name that independent as top other
    good_ind = ~ind | (p["top_other_cand"].map(lambda x: last_name(x) if isinstance(x, str) else "")
                       == p["race_id"].map(lambda r: last_name(IND_SIDE_A.get(r, ""))))
    p = p[good_ind & p["share"].notna()]
    p["partisan_sponsor"] = p["partisan_sponsor"].where(p["partisan_sponsor"].isin(["D", "R"]), "")
    p["partisan_sponsor"] = consistent_sponsor(p["pollster"], p["partisan_sponsor"])
    return polls._standardise(p, 2026)


# Pollsters whose 2026 releases are all for one side's sponsor but are tagged inconsistently in polls_2026.csv:
# RedBox Research's Oct 20-23 batch was fielded for "A Fight Worth Having" (D-aligned super PAC); 16 of 18 rows
# carry partisan_sponsor=D, MI-04 and MI-10 are blank (cross-model investigation, section 3.2).
SPONSOR_BY_POLLSTER = {"redbox": "D"}


def consistent_sponsor(pollster: pd.Series, sponsor: pd.Series) -> pd.Series:
    key = pollster.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
    out = sponsor.copy()
    for k, side in SPONSOR_BY_POLLSTER.items():
        out = out.mask(key.str.contains(k), side)
    return out


def poll_table_2026(results_hist: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    track = polls.pollster_track(HIST_YEARS, results_hist)
    return polls.aggregate(poll_shares_2026(), track)


def build(results_hist: pd.DataFrame) -> pd.DataFrame:
    tmpl = pd.read_csv(CLEAN.parent / "templates" / "full_congress_template.csv")[["race_id", "chamber"]]
    lean = pd.read_csv(CLEAN / "lean_2026.csv")
    rat = pd.read_csv(CLEAN / "ratings_2026.csv")
    rat = rat.assign(other_dev=hist_features.other_dev(rat))[["race_id", "mean_rating", "other_dev"]]
    cand = pd.read_csv(CLEAN / "candidates_2026.csv")
    fec = pd.read_csv(CLEAN / "fec_race_2026.csv")
    df = tmpl.merge(lean[["race_id", "pres24_dem_2p", "district_changed_2026", "lines"]], on="race_id", how="left")
    df["lean"] = df["pres24_dem_2p"] - NATIONAL_PRES24
    df = df.merge(rat, on="race_id", how="left").rename(columns={"mean_rating": "rating"})
    df["senate"] = (df["chamber"] == "Senate").astype(int)
    df["year"] = 2026
    # incumbency of side A / side B
    def inc_of(rid: str) -> int:
        c = cand[(cand["race_id"] == rid) & (cand["is_incumbent"] == 1)]
        if c.empty:
            return 0
        party = c["party"].iloc[0]
        if rid in IND_SIDE_A:
            return -1 if party == "REP" else 1
        if rid in IND_SIDE_B:
            return -1 if party == "IND" else 1
        return 1 if party == "DEM" else (-1 if party == "REP" else 0)
    df["inc"] = df["race_id"].map(inc_of)
    f = fec.set_index("race_id")
    a_money = np.where(df["race_id"].isin(IND_SIDE_A), df["race_id"].map(f["ind_receipts"]),
                       df["race_id"].map(f["dem_receipts"]))
    b_money = np.where(df["race_id"].isin(IND_SIDE_B), df["race_id"].map(f["ind_receipts"]),
                       df["race_id"].map(f["rep_receipts"]))
    df["money"] = money_feature(pd.Series(a_money), pd.Series(b_money))
    # previous House result where the district is unchanged since 2024
    h24 = results_hist[(results_hist["year"] == 2024) & (results_hist["chamber"] == "House")].set_index("race_id")
    same = (df["chamber"] == "House") & (df["district_changed_2026"].fillna(1) == 0)
    prev_ok = same & df["race_id"].map(h24["uncontested"]).fillna(1).eq(0) & df["race_id"].map(h24["dem_share_2p"]).notna()
    df["prev_rel"] = np.where(prev_ok, df["race_id"].map(h24["dem_share_2p"]) - HOUSE_2024_D_SHARE, 0.0)
    df["has_prev"] = prev_ok.astype(int)
    df["env"] = generic_env()
    df["midterm"] = 1
    df["out_d"] = 1          # Republican president: the Democrats are the out-party
    df["other_dev"] = df["other_dev"].fillna(0.0)
    df["ind_side_a"] = df["race_id"].isin(IND_SIDE_A)
    df["ind_side_b"] = df["race_id"].isin(IND_SIDE_B)
    df["non_partisan_final"] = df["race_id"].isin(NON_PARTISAN_FINAL)
    df["uncontested_d"] = df["race_id"].isin(UNCONTESTED_D)
    df["state"] = df["race_id"].str[:2]
    return df
