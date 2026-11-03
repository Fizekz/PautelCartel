"""2026 inputs: Kalshi candidate slots (names only, for output), candidate attributes, race features.

Kalshi prices are read here only to name the candidate slots and to fill the p_kalshi column; they never enter
the model. Model inputs: candidates_2026 (incumbency, is_major), fec_2026 / fec_race_2026 (receipts), lean_2026,
2024 House results on unchanged lines, polls_2026, ratings_2026, national_2026 generic-ballot average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CLEAN, ELECTION_DAY, KALSHI_RACES, NAT_PRES, last_name, lines_unchanged, national_table
from features import EXT_DAYS, QUANT_OUTLETS_2026, one_row_per_poll, summarise_polls

GENERIC_PLACEHOLDER = {"Democratic party": "DEM", "Republican party": "REP"}


def kalshi_slots() -> pd.DataFrame:
    k = pd.read_csv(KALSHI_RACES)
    rows = []
    for r in k.itertuples():
        for part in str(r.candidates).split(" | "):
            party, name, price = part.split(":")
            rows.append({"race_id": r.race_id, "candidate": name.strip(), "party_k": party, "mid": float(price)})
    s = pd.DataFrame(rows)
    s["p_kalshi"] = s["mid"] / s.groupby("race_id")["mid"].transform("sum")
    return s


def match_candidate(cands: pd.DataFrame, race_id: str, name: str, party_k: str) -> pd.Series | None:
    sub = cands[cands["race_id"] == race_id]
    if name in GENERIC_PLACEHOLDER:
        hit = sub[(sub["party"] == GENERIC_PLACEHOLDER[name]) & (sub["is_major"] == 1)]
        if hit.empty:
            hit = sub[(sub["party"] == GENERIC_PLACEHOLDER[name]) & (sub["is_incumbent"] == 1)]
        return hit.iloc[0] if len(hit) == 1 else None
    hit = sub[sub["candidate_name"] == name]
    if hit.empty:
        hit = sub[sub["candidate_name"].map(last_name) == last_name(name)]
        if party_k in ("DEM", "REP", "IND") and len(hit) > 1:
            hit = hit[hit["party"] == party_k]
    return hit.iloc[0] if len(hit) >= 1 else None


def candidate_table() -> pd.DataFrame:
    slots = kalshi_slots()
    cands = pd.read_csv(CLEAN / "candidates_2026.csv")
    fec = pd.read_csv(CLEAN / "fec_2026.csv")
    frace = pd.read_csv(CLEAN / "fec_race_2026.csv").set_index("race_id")
    out = []
    for s in slots.itertuples():
        m = match_candidate(cands, s.race_id, s.candidate, s.party_k)
        f = fec[(fec["race_id"] == s.race_id) & (fec["kalshi_name"] == s.candidate)]
        rec = float(pd.to_numeric(f["receipts"], errors="coerce").fillna(0).sum()) if len(f) else 0.0
        party = s.party_k if s.party_k != "UNK" else (m["party"] if m is not None else "UNK")
        placeholder = s.candidate in GENERIC_PLACEHOLDER
        if placeholder and frace.at[s.race_id, "receipts_basis"] == "all_active_party_filers":
            rec = float(frace.at[s.race_id, "dem_receipts" if party == "DEM" else "rep_receipts"])
        out.append({"race_id": s.race_id, "chamber": "Senate" if s.race_id.endswith("SEN") else "House",
                    "candidate": s.candidate, "party": party, "p_kalshi": s.p_kalshi,
                    "is_inc": int(m["is_incumbent"]) if m is not None else 0,
                    "is_major": int(m["is_major"]) if m is not None else 0,
                    "matched": m is not None, "placeholder": placeholder, "receipts": rec,
                    "system": m["system"] if m is not None else "",
                    "cand_note": m["notes"] if m is not None else ""})
    c = pd.DataFrame(out)
    c["force_side"] = force_sides(c)
    return c


def force_sides(c: pd.DataFrame) -> pd.Series:
    """D / R / '' per Kalshi slot.

    Majors by party; a major independent takes the D side where the race has no major Democrat (NE, MT, ID, SD
    Senate, AK-AL) and the R side where it has no major Republican (CA-06 Kiley, MA-01, PA-03). Party-only slots
    (LA House, AK-SEN) are majors of their party. Withdrawn or non-major Democrats are minor ('')."""
    side = pd.Series("", index=c.index)
    for rid, sub in c.groupby("race_id"):
        major = sub[(sub["is_major"] == 1) | (sub["placeholder"] & sub["race_id"].str.startswith("LA-")
                                             & sub["chamber"].eq("House")) |
                    (sub["placeholder"] & (sub["race_id"] == "AK-SEN"))]
        if rid.startswith("LA-") and rid != "LA-SEN":  # jungle: Kalshi prices the party, both are live
            major = sub[sub["placeholder"]]
        has_d = (major["party"] == "DEM").any()
        has_r = (major["party"] == "REP").any()
        for i, r in major.iterrows():
            if r["party"] == "DEM":
                side[i] = "D"
            elif r["party"] == "REP":
                side[i] = "R"
            elif not has_d:
                side[i] = "D"
            elif not has_r:
                side[i] = "R"
    return side


def prev_house_2024() -> pd.DataFrame:
    res = pd.read_csv(CLEAN / "results_house.csv")
    res = res[res["year"] == 2024]
    lean = pd.read_csv(CLEAN / "lean_hist.csv")
    lean = lean[(lean["year"] == 2024) & (lean["chamber"] == "House")]
    nat = national_table()
    r = res.merge(lean[["race_id", "pres_prev_dem_2p", "national_pres_dem_2p"]], on="race_id")
    r["prev_dev"] = r["dem_share_2p"] - nat.at[2024, "house_d2p"] - (r["pres_prev_dem_2p"] - r["national_pres_dem_2p"])
    r.loc[r["uncontested"] == 1, "prev_dev"] = np.nan
    r = r[[lines_unchanged(2026, s) for s in r["state"]]]
    return r.set_index("race_id")[["prev_dev", "winner_name"]].rename(columns={"winner_name": "prev_winner"})


def polls_2026(c: pd.DataFrame) -> pd.DataFrame:
    """Two-party share of the D side vs the R side, using the side's party column (IND -> top_other)."""
    p = pd.read_csv(CLEAN / "polls_2026.csv")
    nm = p["nominee_match"].astype(str).str.lower()
    hyp = p["is_hypothetical"].astype(str).str.lower()
    usable = (nm == "true") | ((p["nominee_match"].isna()) & (hyp == "false"))
    p = p[usable].copy()
    rcv = p["rcv_round"].astype(str)
    is_ak = p["race_id"].str.startswith("AK-")
    # AK (RCV): the final elimination round, as 538's historical files kept; elsewhere first-choice readings
    last_round = p[is_ak].assign(rr=pd.to_numeric(p["rcv_round"], errors="coerce").fillna(0)).groupby(
        ["race_id", "pollster", "end_date"])["rr"].transform("max")
    keep = ~is_ak & ~rcv.isin(["2", "3", "2.0", "3.0"])
    keep[is_ak] = pd.to_numeric(p.loc[is_ak, "rcv_round"], errors="coerce").fillna(0) == last_round
    p = p[keep].copy()
    p["end"] = pd.to_datetime(p["end_date"], errors="coerce")
    p["days"] = (pd.Timestamp(ELECTION_DAY[2026]) - p["end"]).dt.days
    p = p[(p["days"] >= 0) & (p["days"] <= EXT_DAYS)].copy()
    # RedBox Research polls for A Fight Worth Having (a D super PAC, per the dossiers): the file tags 18 of the
    # Oct 20-23 batch 'D' but leaves MI-04 / MI-10 blank. Tag them all D.
    p.loc[p["pollster"].fillna("").str.contains("RedBox", case=False), "partisan_sponsor"] = "D"
    d_party = c[(c["force_side"] == "D")].groupby("race_id")["party"].first()
    r_party = c[(c["force_side"] == "R")].groupby("race_id")["party"].first()
    # VoteHub answers carry no party; in independent-led races (MT-SEN) they swap the IND and the minor Democrat
    # relative to the same poll on Wikipedia (Quantus 10-18: Bodnar 31 vs 21). Use Wikipedia rows only there.
    ind_led = ~p["race_id"].map(d_party).eq("DEM")
    p = p[~(ind_led & p["source"].eq("votehub"))].copy()
    d = np.where(p["race_id"].map(d_party).eq("DEM"), p["dem_pct"], p["top_other_pct"])
    r = np.where(p["race_id"].map(r_party).eq("REP"), p["rep_pct"], p["top_other_pct"])
    d = pd.to_numeric(pd.Series(d, index=p.index), errors="coerce")
    r = pd.to_numeric(pd.Series(r, index=p.index), errors="coerce")
    p["share"] = d / (d + r)
    p = p[p["share"].between(0.02, 0.98)]
    # v3: one row per poll (same rule as the history): a poll = race x pollster x end date; Wikipedia and VoteHub
    # copies and alternative versions (with/without leaners, several ballot tests) are averaged into one row
    pkey = p["pollster"].fillna("?").str.lower().str.replace(r"[^a-z]", "", regex=True)
    p["poll_id"] = pd.factorize(p["race_id"] + "|" + pkey + "|" + p["end_date"].astype(str))[0]
    return summarise_polls(one_row_per_poll(p), ["race_id"]).set_index("race_id")


def generic_2026() -> float:
    n = pd.read_csv(CLEAN / "national_2026.csv")
    a = n[n["metric"] == "generic_ballot_avg"]
    return float((a["value_dem"] - a["value_rep"]).mean())


def race_features(c: pd.DataFrame, quant_lambda: float = 0.0) -> pd.DataFrame:
    """quant_lambda: rating = core + lambda (quant outlets' mean - core). v2 default 0: the blend did not help on
    the 2020/2024 forward tests (select_v2.py C); kept as a sensitivity switch."""
    lean = pd.read_csv(CLEAN / "lean_2026.csv").set_index("race_id")
    f = pd.DataFrame(index=lean.index)
    f["chamber"] = lean["chamber"]
    f["state"] = f.index.str[:2]
    f["lean"] = lean["pres24_dem_2p"] - NAT_PRES[2024]
    f["lean2"] = lean["pres20_dem_2p"] - NAT_PRES[2020]
    f = f.join(prev_house_2024(), how="left")
    f.loc[f["chamber"] == "Senate", ["prev_dev", "prev_winner"]] = np.nan
    f = f.join(polls_2026(c), how="left")
    rt = pd.read_csv(CLEAN / "ratings_2026.csv").set_index("race_id")
    f["rating_core"] = -pd.to_numeric(rt["mean_rating"], errors="coerce")
    f["rating_quant"] = -rt[QUANT_OUTLETS_2026].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    has_q = f["rating_quant"].notna() & f["rating_core"].notna()
    f["rating"] = f["rating_core"]
    f.loc[has_q, "rating"] = f.loc[has_q, "rating_core"] + quant_lambda * (f.loc[has_q, "rating_quant"]
                                                                          - f.loc[has_q, "rating_core"])
    f["generic_margin"] = generic_2026()
    f["g"] = f["generic_margin"] / 200.0
    f["midterm"] = 1      # 2026: midterm under a Republican president, so the Democrats are the out-party
    f["pres_d"] = -1
    return f
