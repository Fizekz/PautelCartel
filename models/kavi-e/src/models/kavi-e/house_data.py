"""Race-level House feature frames: history 2010-2024 (training / backtest) and 2026 (forecast).

Margins are D-R in points of the two-party vote. Lean L = 2 * 100 * (district pres D 2p share - national pres D 2p
share) for the most recent presidential election before the year (on that year's lines). Features:
  L, L_pres (L in presidential years), inc (+1 D incumbent running, -1 R incumbent, 0 open),
  prev_resid_inc / prev_resid_open (the seat's previous House margin minus L + N of that year, same lines only,
  split by whether the incumbent who won it runs again), fec (clipped ln D/R receipts),
  rating (mean Cook/Sabato/Inside, -3 Safe D .. +3 Safe R).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import (MIDTERMS, NAT_PRES_2P, WORK, clip_log_ratio, ensure_dirs, last_name, override_incumbent, read,
                    redrawn, same_person)

HIST_YEARS = (2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024)


def lean_frame() -> pd.DataFrame:
    lh = read("lean_hist.csv")
    lh = lh[lh["chamber"] == "House"].copy()
    lh["L"] = 200 * (lh["pres_prev_dem_2p"] - lh["national_pres_dem_2p"])
    return lh[["year", "race_id", "L"]]


def fec_lookup(fec: pd.DataFrame, year: int, race_id: str, bucket: str, name: str) -> tuple[float, str]:
    sub = fec[(fec["year"] == year) & (fec["race_id"] == race_id) & (fec["party_bucket"] == bucket)]
    if sub.empty or not isinstance(name, str):
        return np.nan, ""
    hit = sub[sub["cand_name"].map(lambda n: same_person(n, name))]
    if hit.empty:
        hit = sub[sub["cand_name"].map(last_name) == last_name(name)]
    row = (hit if not hit.empty else sub.sort_values("party_rank")).iloc[0]
    return float(row["receipts"]) if pd.notna(row["receipts"]) else 0.0, str(row["ici"])


def build_history() -> pd.DataFrame:
    res = read("results_house.csv")
    res = res[(res["special"] == 0) & (res["year"] >= 2008)].copy()
    nat = read("national_hist.csv").set_index("year")["house_margin_dr"]
    nat.loc[2008] = 10.6  # 2008 national House margin (Clerk); only used for 2010 prev-residuals
    lean = lean_frame()
    ratings = read("ratings_hist.csv")
    ratings = ratings[ratings["chamber"] == "House"][["year", "race_id", "mean_rating"]]
    fec = read("fec_hist.csv")
    res = res.merge(lean, on=["year", "race_id"], how="left")
    res["margin"] = 200 * (res["dem_share_2p"] - 0.5)
    res["contested"] = (res["uncontested"] == 0) & res["dem_share_2p"].between(0.0001, 0.9999)
    rows = []
    for year in HIST_YEARS:
        cur = res[res["year"] == year]
        prev = res[res["year"] == year - 2].set_index("race_id")
        prev_winners = res[res["year"] == year - 2]
        for _, r in cur.iterrows():
            rows.append(race_features(r, year, prev, prev_winners, nat, fec))
    df = pd.DataFrame(rows)
    df = df.merge(ratings.rename(columns={"mean_rating": "rating"}), on=["year", "race_id"], how="left")
    df["N"] = df["year"].map(nat)
    return df


def incumbent_flag(name: str, party: str, state: str, prev_winners: pd.DataFrame, ici: str) -> int:
    if not isinstance(name, str):
        return 0
    pw = prev_winners[(prev_winners["state"] == state) & (prev_winners["winner_party"] == party)]
    if any(same_person(name, w) for w in pw["winner_name"]):
        return 1
    return int(ici == "I")


def race_features(r: pd.Series, year: int, prev: pd.DataFrame, prev_winners: pd.DataFrame, nat: pd.Series,
                  fec: pd.DataFrame) -> dict:
    d_rec, d_ici = fec_lookup(fec, year, r["race_id"], "DEM", r["dem_cand"])
    r_rec, r_ici = fec_lookup(fec, year, r["race_id"], "REP", r["rep_cand"])
    d_inc = incumbent_flag(r["dem_cand"], "DEM", r["state"], prev_winners, d_ici)
    r_inc = incumbent_flag(r["rep_cand"], "REP", r["state"], prev_winners, r_ici)
    d_inc = max(d_inc, int(override_incumbent(year, r["race_id"], "DEM", r["dem_cand"])))
    r_inc = max(r_inc, int(override_incumbent(year, r["race_id"], "REP", r["rep_cand"])))
    inc = d_inc - r_inc
    prev_resid, same_inc = np.nan, 0
    if not redrawn(year, r["state"]) and r["race_id"] in prev.index:
        p = prev.loc[r["race_id"]]
        if bool(p["contested"]) and pd.notna(p["L"]):
            prev_resid = float(p["margin"] - (p["L"] + nat.loc[year - 2]))
            winner_cand = p["dem_cand"] if p["winner_party"] == "DEM" else p["rep_cand"]
            now_cand = r["dem_cand"] if p["winner_party"] == "DEM" else r["rep_cand"]
            same_inc = int(isinstance(now_cand, str) and isinstance(winner_cand, str)
                           and same_person(now_cand, winner_cand))
    return {
        "year": year, "race_id": r["race_id"], "state": r["state"], "dem_cand": r["dem_cand"],
        "rep_cand": r["rep_cand"], "margin": r["margin"], "contested": bool(r["contested"]),
        "winner_party": r["winner_party"], "L": r["L"], "inc": inc,
        "prev_resid_inc": prev_resid if same_inc else 0.0,
        "prev_resid_open": prev_resid if (not same_inc and pd.notna(prev_resid)) else 0.0,
        "has_prev": int(pd.notna(prev_resid)),
        "fec": clip_log_ratio(d_rec, r_rec) if (pd.notna(d_rec) and pd.notna(r_rec)) else 0.0,
        "pres_year": int(year not in MIDTERMS),
    }


def build_2026() -> pd.DataFrame:
    lean = read("lean_2026.csv")
    lean = lean[lean["chamber"] == "House"].copy()
    lean["L"] = 200 * (lean["pres24_dem_2p"] - NAT_PRES_2P[2024])
    cand = read("candidates_2026.csv")
    cand = cand[cand["chamber"] == "House"]
    fec = read("fec_race_2026.csv")
    ratings = read("ratings_2026.csv")[["race_id", "mean_rating"]].rename(columns={"mean_rating": "rating"})
    hist = build_history()
    h24 = hist[hist["year"] == 2024].set_index("race_id")
    nat24 = float(read("national_hist.csv").set_index("year").loc[2024, "house_margin_dr"])
    res = read("results_house.csv")
    r24 = res[res["year"] == 2024].set_index("race_id")
    lh24 = lean_frame().query("year == 2024").set_index("race_id")["L"]
    rows = []
    for _, lr in lean.iterrows():
        rid = lr["race_id"]
        cs = cand[cand["race_id"] == rid]
        d_inc = int(((cs["party"] == "DEM") & (cs["is_incumbent"] == 1)).any())
        r_inc = int(((cs["party"] == "REP") & (cs["is_incumbent"] == 1)).any())
        inc_same_seat = cs[(cs["is_incumbent"] == 1) & (cs["incumbent_running_in_different_district"] == 0)]
        prev_resid, same_inc = np.nan, 0
        unchanged = lr["district_changed_2026"] == 0
        if unchanged and rid in r24.index:
            p = r24.loc[rid]
            contested = p["uncontested"] == 0 and 0.0001 < p["dem_share_2p"] < 0.9999
            if contested:
                prev_resid = float(200 * (p["dem_share_2p"] - 0.5) - (lh24.loc[rid] + nat24))
                same_inc = int(any(same_person(n, p["winner_name"]) for n in inc_same_seat["candidate_name"]))
        f = fec[fec["race_id"] == rid].iloc[0]
        rows.append({
            "year": 2026, "race_id": rid, "state": rid[:2], "L": lr["L"], "inc": d_inc - r_inc,
            "prev_resid_inc": prev_resid if same_inc else 0.0,
            "prev_resid_open": prev_resid if (not same_inc and pd.notna(prev_resid)) else 0.0,
            "has_prev": int(pd.notna(prev_resid)),
            "fec": clip_log_ratio(f["dem_receipts"], f["rep_receipts"]), "pres_year": 0,
            "district_changed": int(not unchanged),
        })
    df = pd.DataFrame(rows).merge(ratings, on="race_id", how="left")
    del h24
    return df


def main() -> int:
    ensure_dirs()
    hist = build_history()
    hist.to_csv(WORK / "house_hist.csv", index=False)
    f26 = build_2026()
    f26.to_csv(WORK / "house_2026.csv", index=False)
    c = hist[hist["contested"]]
    print(c.groupby("year")[["L", "inc", "prev_resid_inc", "fec", "rating"]].agg(["mean", "std"]).round(2))
    print("2026 frame", f26.shape, f26[["L", "inc", "prev_resid_inc", "fec", "rating"]].describe().round(2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
