"""Race-level Senate feature frames: history 2010-2024 and 2026.

Same features as the House where they exist: L (statewide lean vs the national pres share, margin points),
L_pres, N (national House margin), inc (+1 D incumbent / -1 R incumbent / 0 open), rating, poll_margin/poll_neff.
The 'D side' of a 2026 race is the independent where the independent is the main challenger (polls.DSIDE_IND).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import MIDTERMS, NAT_PRES_2P, WORK, ensure_dirs, override_incumbent, read, same_person
from polls import DSIDE_IND, history_polls, polls_2026

HIST_YEARS = (2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024)
MAJOR_THIRD_SHARE = 0.15  # races where a non-D/R candidate got >15% are not D-vs-R training rows


def senate_incumbent(name: str, party: str, state: str, year: int, res: pd.DataFrame, fec: pd.DataFrame) -> int:
    if not isinstance(name, str):
        return 0
    prior = res[(res["state"] == state) & (res["year"] < year) & (res["year"] >= year - 6)
                & (res["winner_party"] == party)]
    if any(same_person(name, w) for w in prior["winner_name"].dropna()):
        return 1
    f = fec[(fec["year"] == year) & (fec["race_id"] == f"{state}-SEN") & (fec["party_bucket"] == party)]
    hit = f[f["cand_name"].map(lambda n: same_person(n, name))]
    return int((hit["ici"] == "I").any())


def build_history() -> pd.DataFrame:
    res = read("results_senate.csv")
    lean = read("lean_hist.csv")
    lean = lean[lean["chamber"] == "Senate"].copy()
    lean["state"] = lean["race_id"].str[:2]
    lean["L"] = 200 * (lean["pres_prev_dem_2p"] - lean["national_pres_dem_2p"])
    nat = read("national_hist.csv").set_index("year")["house_margin_dr"]
    fec = read("fec_hist.csv")
    ratings = read("ratings_hist.csv")
    ratings = ratings[ratings["chamber"] == "Senate"][["year", "race_id", "mean_rating"]]
    from independents import hist_ind_map
    pol = history_polls(ind=hist_ind_map())  # v2: major-independent races get the independent-vs-R polls
    pol = pol[pol["chamber"] == "Senate"].drop(columns="chamber")
    rows = []
    for _, r in res[res["year"].isin(HIST_YEARS)].iterrows():
        third = str(r["notes"]) if pd.notna(r["notes"]) else ""
        major_third = "major non-D/R" in third or r["winner_party"] == "OTHER"
        d_inc = senate_incumbent(r["dem_cand"], "DEM", r["state"], r["year"], res, fec)
        r_inc = senate_incumbent(r["rep_cand"], "REP", r["state"], r["year"], res, fec)
        d_inc = max(d_inc, int(override_incumbent(r["year"], r["race_id"], "DEM", r["dem_cand"])))
        r_inc = max(r_inc, int(override_incumbent(r["year"], r["race_id"], "REP", r["rep_cand"])))
        rows.append({
            "year": r["year"], "race_id": r["race_id"], "state": r["state"], "special": int(r["special"]),
            "dem_cand": r["dem_cand"], "rep_cand": r["rep_cand"], "margin": 200 * (r["dem_share_2p"] - 0.5),
            "contested": bool(r["uncontested"] == 0 and 0.0001 < r["dem_share_2p"] < 0.9999 and not major_third),
            "winner_party": r["winner_party"], "inc": d_inc - r_inc, "pres_year": int(r["year"] not in MIDTERMS),
        })
    df = pd.DataFrame(rows)
    df = df.merge(lean[["year", "state", "L"]], on=["year", "state"], how="left")
    df["N"] = df["year"].map(nat)
    df = df.merge(ratings.rename(columns={"mean_rating": "rating"}), on=["year", "race_id"], how="left")
    df["poll_special"] = df["special"].astype(bool)
    pol = pol.rename(columns={"special": "poll_special"})
    pol["state"] = pol["race_id"].str[:2]
    df = df.merge(pol.drop(columns="race_id"), on=["year", "state", "poll_special"], how="left")
    return df.drop(columns="poll_special")


def build_2026() -> pd.DataFrame:
    lean = read("lean_2026.csv")
    lean = lean[lean["chamber"] == "Senate"].copy()
    lean["L"] = 200 * (lean["pres24_dem_2p"] - NAT_PRES_2P[2024])
    cand = read("candidates_2026.csv")
    cand = cand[cand["chamber"] == "Senate"]
    ratings = read("ratings_2026.csv")[["race_id", "mean_rating"]].rename(columns={"mean_rating": "rating"})
    pol = polls_2026()
    pol = pol[pol["chamber"] == "Senate"].drop(columns=["chamber", "special", "year"])
    rows = []
    for _, lr in lean.iterrows():
        rid = lr["race_id"]
        cs = cand[cand["race_id"] == rid]
        dside = "IND" if rid in DSIDE_IND else "DEM"
        d_inc = int(((cs["party"] == dside) & (cs["is_incumbent"] == 1)).any())
        r_inc = int(((cs["party"] == "REP") & (cs["is_incumbent"] == 1)).any())
        rows.append({"year": 2026, "race_id": rid, "state": rid[:2], "special": int(cs["special"].max()),
                     "L": lr["L"], "inc": d_inc - r_inc, "pres_year": 0, "dside_party": dside})
    df = pd.DataFrame(rows).merge(ratings, on="race_id", how="left").merge(pol, on="race_id", how="left")
    return df


def main() -> int:
    ensure_dirs()
    hist = build_history()
    hist.to_csv(WORK / "senate_hist.csv", index=False)
    f26 = build_2026()
    f26.to_csv(WORK / "senate_2026.csv", index=False)
    c = hist[hist["contested"]]
    print(c.groupby("year").agg(n=("margin", "size"), polled=("poll_margin", lambda s: s.notna().sum()),
                                inc=("inc", lambda s: (s != 0).mean())))
    print(f26.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
