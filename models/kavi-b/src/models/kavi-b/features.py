"""Race-level feature tables for the kavi-b model.

hist_table(): one row per (year, race_id) for 2010-2024 general elections with
    ratings (D-positive), lean, incumbency, FEC log ratio, poll margin, national context and the outcome.
table_2026(): the same features for the 470 races of 2026, plus the principal candidates on each side.

Sign convention everywhere: positive = good for the "D side" (the Democrat, or the main non-Republican when an
independent is the real challenger, e.g. King/Sanders/McMullin/Osborn).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import (OTHER_RATER_EXCLUDE, CLEAN, ELECTION_DAY, FEC_SMOOTH, KALSHI, KALSHI_STAMP, NAT_PRES, OWN, POLL_HALF_LIFE,
                    POLL_WINDOW_DAYS, POLL_WINDOW_EXT_DAYS, MIN_POLLS_IN_WINDOW,
                    clipped_log_ratio, fec_last_name, last_name)

MAJOR_IND_RE = re.compile(r"major non-D/R candidate (.+?) \(([\d.]+)%\)")


# ----------------------------------------------------------------------------------------------- shared
def other_raters(rat: pd.DataFrame) -> pd.DataFrame:
    """v2: mean D-positive rating of the non-core outlets (all other_* columns except 538) and their count.
    Ratings use the same -3 (Safe D) .. +3 (Safe R) scale as Cook/Sabato/Inside."""
    cols = [c for c in rat.columns if c.startswith("other_") and c not in OTHER_RATER_EXCLUDE]
    vals = rat[cols].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame({"r_oth": -vals.mean(axis=1), "n_oth": vals.notna().sum(axis=1)}, index=rat.index)


def harmonise_sponsor(p: pd.DataFrame) -> pd.DataFrame:
    """v2: one sponsored release often covers many races (RedBox Research for A Fight Worth Having, Oct 20-23, 18
    races) but Wikipedia tags the sponsor on some rows only. Within a (pollster, end date) batch of >= 3 rows, give
    every row the batch's majority non-blank sponsor tag; also tag rows whose sponsor note names a party-aligned
    sponsor."""
    p = p.copy()
    note = p.get("sponsor_note", pd.Series("", index=p.index)).fillna("").str.lower()
    d_note = note.str.contains(r"democratic|progressive|dccc|dscc|house majority pac")
    r_note = note.str.contains(r"republican|conservative|nrcc|nrsc|congressional leadership fund")
    p.loc[p["partisan_sponsor"].isna() & d_note & ~r_note, "partisan_sponsor"] = "D"
    p.loc[p["partisan_sponsor"].isna() & r_note & ~d_note, "partisan_sponsor"] = "R"
    for _, g in p.groupby(["pollster", "end_date"]):
        tags = g["partisan_sponsor"].dropna()
        if len(g) >= 3 and len(tags) > len(g) / 2 and tags.nunique() == 1:
            p.loc[g.index, "partisan_sponsor"] = tags.iloc[0]
    return p


# ----------------------------------------------------------------------------------------------- history
def _results() -> pd.DataFrame:
    h = pd.read_csv(CLEAN / "results_house.csv")
    s = pd.read_csv(CLEAN / "results_senate.csv")
    h["chamber"], s["chamber"] = "House", "Senate"
    r = pd.concat([h, s], ignore_index=True)
    r = r[(r["year"] >= 2010) & (r["special"].fillna(0).astype(int).isin([0, 1]))]
    return r.drop_duplicates(["year", "race_id"])


def _principals(res: pd.DataFrame) -> pd.DataFrame:
    """Provisional D-side principal: the independent named in MEDSL's 'major non-D/R candidate' note when there
    was no Democrat, otherwise the Democrat. hist_table() then applies the pre-election rule (_ind_rule)."""
    notes = res["notes"].fillna("")
    out = res.copy()
    ind_name, ind_pct = [], []
    for txt in notes:
        m = MAJOR_IND_RE.search(txt)
        ind_name.append(m.group(1) if m else None)
        ind_pct.append(float(m.group(2)) if m else np.nan)
    out["ind_name"], out["ind_pct"] = ind_name, ind_pct
    use_ind = out["ind_name"].notna() & out["dem_cand"].isna()
    out["dside_is_ind"] = use_ind.astype(int)
    out["dside_name"] = np.where(use_ind, out["ind_name"], out["dem_cand"])
    out["rside_name"] = out["rep_cand"]
    out["same_party"] = notes.str.contains("same-party top-two").astype(int)
    out["no_contest"] = (out["dside_name"].isna() | out["rside_name"].isna()).astype(int) * (1 - out["same_party"])
    out["y"] = (out["winner_party"] != "REP").astype(int)
    return out


# Pre-election FEC cuts for test years (file, year). 2018: Wayback weball18.zip of 2018-11-07 (kavi-b).
# v3: 2022: Wayback webl22.zip of 2022-08-03 (coverage mostly to 2022-06-30), extracted by kavi-a
# (origin/kavi/kavi-a-1101 data/kavi-a/fec_preelection.csv); 8 rows with coverage after the capture date dropped.
PRE_ELECTION_FEC = (("fec_2018_preelection.csv", 2018), ("fec_2022_preelection.csv", 2022))


def _apply_snapshot(f: pd.DataFrame, snap: pd.DataFrame, year: int) -> pd.DataFrame:
    """Replace `year`'s full-cycle receipts by the pre-election snapshot (by cand_id, else surname + seat)."""
    snap = snap.assign(last=snap["cand_name"].map(fec_last_name),
                       race_key=snap["race_id_raw"].str.replace(r"-0[01]$", "-AL", regex=True))
    by_id = snap.groupby("cand_id")["receipts_pre"].max()
    by_name = snap.groupby(["last", "race_key"])["receipts_pre"].max()
    isy = f["year"] == year
    pre = []
    for cid, last, rid in zip(f.loc[isy, "cand_id"], f.loc[isy, "last"], f.loc[isy, "race_id"]):
        v = by_id.get(cid, np.nan)
        if pd.isna(v) or v == 0:
            key = rid if rid.endswith("-SEN") else re.sub(r"-0[01]$", "-AL", rid)
            v = by_name.get((last, key), v)
        pre.append(0.0 if pd.isna(v) else float(v))
    out = f.copy()
    out.loc[isy, "receipts"] = pre
    return out


def _fec_hist(pre2018: bool) -> pd.DataFrame:
    """FEC two-year totals; with pre2018=True (name kept from v1) every test year with a pre-election
    snapshot (2018, v3: 2022) gets the snapshot's receipts instead of its post-election totals."""
    f = pd.read_csv(CLEAN / "fec_hist.csv")
    f["receipts"] = f["receipts"].fillna(0.0)
    f["last"] = f["cand_name"].map(fec_last_name)
    if pre2018:
        # fec_hist keeps one id per committee; a snapshot may file the same person under a sibling id, so
        # fall back to (surname, office, state, district); at-large seats are district 00 or 01 in FEC files
        for fname, year in PRE_ELECTION_FEC:
            f = _apply_snapshot(f, pd.read_csv(OWN / fname), year)
    return f


def _match_fec(fec: pd.DataFrame, year: int, race_id: str, name, bucket: str):
    """(receipts, is_incumbent) of the FEC row for a named nominee; fall back to the bucket's top fundraiser."""
    base = race_id.replace("-SEN-S", "-SEN")
    rows = fec[(fec["year"] == year) & (fec["race_id"] == base)]
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return 0.0, 0, False
    ln = last_name(name)
    hit = rows[rows["last"] == ln]
    if len(hit) > 1:
        hit = hit[hit["party_bucket"] == bucket] if (hit["party_bucket"] == bucket).any() else hit
    if hit.empty:
        cand = rows[rows["party_bucket"] == bucket].sort_values("receipts", ascending=False)
        if cand.empty:
            return 0.0, 0, False
        hit = cand.head(1)
        matched = False
    else:
        hit = hit.sort_values("receipts", ascending=False).head(1)
        matched = True
    row = hit.iloc[0]
    return float(row["receipts"]), int(row["ici"] == "I"), matched


def _poll_margin(polls: pd.DataFrame, eday: str) -> tuple[float, float, float]:
    """Recency-weighted D-minus-R margin; partisan/internal polls weigh half.

    Window: the final POLL_WINDOW_DAYS; when fewer than MIN_POLLS_IN_WINDOW distinct polls fall inside it, the window
    is widened to POLL_WINDOW_EXT_DAYS (v2: under-polled races keep their summer polls; 2010-2016 538 data only hold
    ~60 days, so the rule can only fire from 2018 on). Returns (margin, n distinct polls, W), where W is the summed
    weight (one fresh nonpartisan poll = 1) used by the model to shrink thin poll evidence."""
    if polls.empty:
        return np.nan, 0.0, 0.0
    end = pd.to_datetime(polls["end_date"])
    days = (pd.Timestamp(eday) - end).dt.days
    inwin = (days >= 1) & (days <= POLL_WINDOW_DAYS)
    n_in = polls[inwin].groupby(["pollster", "end_date"]).ngroups if inwin.any() else 0
    if n_in < MIN_POLLS_IN_WINDOW:
        inwin = (days >= 1) & (days <= POLL_WINDOW_EXT_DAYS)
    p = polls[inwin].copy()
    if p.empty:
        return np.nan, 0.0, 0.0
    p["days"] = days[p.index]
    p["w"] = 0.5 ** (p["days"] / POLL_HALF_LIFE) * np.where(p["partisan"], 0.5, 1.0)
    # one value per pollster + end date (several populations / versions of one survey)
    g = p.groupby(["pollster", "end_date"]).agg(margin=("margin", "mean"), w=("w", "max")).reset_index()
    # cap a single pollster's total weight so trackers do not dominate
    g["w"] = g["w"] / g.groupby("pollster")["w"].transform("count").clip(lower=1) ** 0.5
    return float(np.average(g["margin"], weights=g["w"])), float(len(g)), float(g["w"].sum())


def _hist_polls(base: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_csv(CLEAN / "polls_hist.csv", low_memory=False)
    p = p[(p["stage"] == "general") & (p["is_hypothetical"] != True)]  # noqa: E712
    has_s = set(zip(base["year"], base["race_id"]))
    sp = p["special"].astype(str).str.lower().eq("true")
    p["rid"] = [r + "-S" if s and (y, r + "-S") in has_s else r for y, r, s in zip(p["year"], p["race_id"], sp)]
    p["partisan"] = p["partisan_sponsor"].notna() | p["internal"].astype(str).str.lower().eq("true")
    out = []
    for row in base.itertuples():
        sub = p[(p["year"] == row.year) & (p["rid"] == row.race_id)]
        if sub.empty or pd.isna(row.dside_name) or pd.isna(row.rside_name):
            out.append((np.nan, 0.0, 0.0))
            continue
        dl, rl = last_name(row.dside_name), last_name(row.rside_name)
        rep_ok = sub["rep_cand"].map(last_name) == rl
        if row.dside_is_ind:
            ok = rep_ok & (sub["other_cand"].fillna("").map(last_name) == dl)
            margin = sub["other_pct"] - sub["rep_pct"]
        else:
            ok = rep_ok & (sub["dem_cand"].map(last_name) == dl)
            margin = sub["dem_pct"] - sub["rep_pct"]
        sub = sub.assign(margin=margin)[ok & margin.notna()]
        out.append(_poll_margin(sub, ELECTION_DAY[row.year]))
    base = base.copy()
    base["poll_margin"] = [o[0] for o in out]
    base["n_polls"] = [o[1] for o in out]
    base["poll_w"] = [o[2] for o in out]
    return base


def _ind_rule(base: pd.DataFrame, fec: pd.DataFrame) -> pd.DataFrame:
    """Pre-election choice of the D-side principal when both a Democrat and a notable independent ran:
    the independent if a forecaster rated the race for an independent ('Safe I', 'Likely I') or if the
    independent out-raised the Democrat. Uses no vote counts, so it is the same rule for training and test."""
    base = base.copy()
    raw = base[["cook_raw", "sabato_raw", "inside_raw"]].fillna("").agg(" ".join, axis=1)
    rated_i = raw.str.contains(r"\b(?:Safe|Solid|Likely|Lean|Tilt) I\b")
    for idx in base.index[base["ind_name"].notna() & base["dem_cand"].notna()]:
        row = base.loc[idx]
        ind = _match_fec(fec, row.year, row.race_id, row.ind_name, "IND")[0]
        dem = _match_fec(fec, row.year, row.race_id, row.dem_cand, "DEM")[0]
        # v2: 2022 FEC exists only as full-cycle (post-election) totals, so its money does not pick the principal
        money_says_ind = ind > dem and row.year != 2022
        if rated_i[idx] or money_says_ind:
            base.loc[idx, ["dside_name", "dside_is_ind"]] = [row.ind_name, 1]
    return base


def hist_table(pre2018_fec: bool = True) -> pd.DataFrame:
    rat = pd.read_csv(CLEAN / "ratings_hist.csv")
    res = _principals(_results())
    keep = ["year", "race_id", "winner_party", "winner_name", "dem_cand", "rep_cand", "dem_share_2p", "ind_name",
            "uncontested", "notes", "dside_is_ind", "dside_name", "rside_name", "same_party", "no_contest", "y"]
    base = rat.merge(res[keep], on=["year", "race_id"], how="inner")
    # ratings, D positive
    for c in ("cook", "sabato", "inside"):
        base[f"r_{c}"] = -base[c]
    base["r_mean"] = -base["mean_rating"]
    base["r_spread"] = base[["cook", "sabato", "inside"]].max(axis=1) - base[["cook", "sabato", "inside"]].min(axis=1)
    base = pd.concat([base, other_raters(base)], axis=1)
    # lean (D share minus national, points)
    lean = pd.read_csv(CLEAN / "lean_hist.csv")
    lean["lean"] = 100 * (lean["pres_prev_dem_2p"] - lean["national_pres_dem_2p"])
    base["lean_key"] = base["race_id"].str.replace("-SEN-S", "-SEN", regex=False)
    base = base.merge(lean[["year", "race_id", "lean"]].rename(columns={"race_id": "lean_key"}),
                      on=["year", "lean_key"], how="left")
    base["lean"] = base["lean"].fillna(-base["pvi_wiki"])
    # FEC + incumbency
    fec = _fec_hist(pre2018_fec)
    base = _ind_rule(base, fec)
    d_rec, r_rec, d_inc, r_inc, d_ok, r_ok = [], [], [], [], [], []
    for row in base.itertuples():
        bucket = "IND" if row.dside_is_ind else "DEM"
        a = _match_fec(fec, row.year, row.race_id, row.dside_name, bucket)
        b = _match_fec(fec, row.year, row.race_id, row.rside_name, "REP")
        d_rec.append(a[0]); d_inc.append(a[1]); d_ok.append(a[2])
        r_rec.append(b[0]); r_inc.append(b[1]); r_ok.append(b[2])
    base["d_receipts"], base["r_receipts"] = d_rec, r_rec
    base["fec_matched"] = np.array(d_ok) & np.array(r_ok)
    base["inc"] = np.array(d_inc) - np.array(r_inc)
    base["fec_lr"] = [clipped_log_ratio(d, r) for d, r in zip(d_rec, r_rec)]
    base = _hist_polls(base)
    nat = pd.read_csv(CLEAN / "national_hist.csv")
    base = base.merge(nat[["year", "generic_margin_dr", "president_party", "cycle_type"]], on="year", how="left")
    base["midterm"] = (base["cycle_type"] == "midterm").astype(int)
    base["pres_d"] = np.where(base["president_party"] == "D", 1, -1)
    base["out_d"] = -base["pres_d"]     # +1 when the D side is the out-party (Republican president)
    return base.drop(columns=["lean_key"])


# ----------------------------------------------------------------------------------------------- 2026
LA_JUNGLE = {"LA-01", "LA-02", "LA-03", "LA-04", "LA-05", "LA-06"}
IND_PRINCIPAL_2026 = {"NE-SEN": "Dan Osborn", "MT-SEN": "Seth Bodnar", "ID-SEN": "Todd Achilles",
                      "SD-SEN": "Brian Bengs", "AK-AL": "Bill Hill"}


def kalshi_candidates() -> pd.DataFrame:
    """One row per candidate in each race's main Kalshi event, with the normalised mid as p_kalshi."""
    races = pd.read_csv(KALSHI / f"races_{KALSHI_STAMP}.csv")
    rows = []
    for r in races.itertuples():
        for item in str(r.candidates).split(" | "):
            party, _, rest = item.partition(":")
            name, _, price = rest.rpartition(":")
            rows.append({"race_id": r.race_id, "party": party, "candidate": name, "mid": float(price)})
    k = pd.DataFrame(rows)
    k["p_kalshi"] = k["mid"] / k.groupby("race_id")["mid"].transform("sum")
    return k


def ca_subpage_polls_as_rows(shared: pd.DataFrame) -> pd.DataFrame:
    """D-vs-R polls from the 2026 California district sub-pages (data/kavi-b/ca_polls_2026.csv) in the
    polls_2026 schema, minus any poll already in the shared file (same race, end date within 2 days)."""
    c = pd.read_csv(OWN / "ca_polls_2026.csv")
    c = c[c["end_date"].notna() & (c["party_a"] != c["party_b"])]
    rows = []
    for r in c.itertuples():
        d_first = r.party_a == "D"
        rows.append({"race_id": r.race_id, "pollster": r.pollster, "end_date": r.end_date,
                     "dem_cand": r.cand_a if d_first else r.cand_b, "dem_pct": r.pct_a if d_first else r.pct_b,
                     "rep_cand": r.cand_b if d_first else r.cand_a, "rep_pct": r.pct_b if d_first else r.pct_a,
                     "partisan": bool(re.search(r"\((?:D|R)\)", r.pollster)), "source": "wiki_ca_subpage"})
    new = pd.DataFrame(rows)
    if new.empty:
        return new
    sh = shared[shared["race_id"].str.startswith("CA-")]
    keep = []
    for r in new.itertuples():
        same = sh[(sh["race_id"] == r.race_id)
                  & ((pd.to_datetime(sh["end_date"]) - pd.Timestamp(r.end_date)).abs().dt.days <= 2)]
        keep.append(same.empty)
    return new[keep]


def _polls_2026(tab: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_csv(CLEAN / "polls_2026.csv", low_memory=False)
    usable = (p["nominee_match"].astype(str).str.lower().eq("true")
              | (p["nominee_match"].isna() & p["is_hypothetical"].astype(str).str.lower().eq("false")))
    usable &= ~p["rcv_round"].astype(str).isin(["2", "3", "2.0", "3.0"])
    p = harmonise_sponsor(p[usable])
    p["partisan"] = p["partisan_sponsor"].notna()
    p = pd.concat([p, ca_subpage_polls_as_rows(p)], ignore_index=True)
    out = []
    for row in tab.itertuples():
        sub = p[p["race_id"] == row.race_id]
        if sub.empty or row.dside_name is None or row.rside_name is None or row.same_party:
            out.append((np.nan, 0.0, 0.0))
            continue
        dl, rl = last_name(row.dside_name), last_name(row.rside_name)
        if row.race_id == "CA-06":   # Pan (D) vs Kiley (I, the R-side principal)
            ok = (sub["dem_cand"].map(last_name) == dl) & (sub["top_other_cand"].fillna("").map(last_name) == rl)
            margin = sub["dem_pct"] - sub["top_other_pct"]
        elif row.dside_is_ind:
            ok = (sub["rep_cand"].map(last_name) == rl) & (sub["top_other_cand"].fillna("").map(last_name) == dl)
            margin = sub["top_other_pct"] - sub["rep_pct"]
        else:   # Kalshi 'Democratic party' / 'Republican party' slots (AK-SEN, LA) carry no name to match
            d_ok = True if row.dside_name.endswith(" party") else sub["dem_cand"].map(last_name) == dl
            r_ok = True if row.rside_name.endswith(" party") else sub["rep_cand"].map(last_name) == rl
            ok = (sub["dem_cand"].notna() & sub["rep_cand"].notna()) & d_ok & r_ok
            margin = sub["dem_pct"] - sub["rep_pct"]
        sub = sub.assign(margin=margin)[ok & margin.notna()]
        out.append(_poll_margin(sub, ELECTION_DAY[2026]))
    tab = tab.copy()
    tab["poll_margin"] = [o[0] for o in out]
    tab["n_polls"] = [o[1] for o in out]
    tab["poll_w"] = [o[2] for o in out]
    return tab


def table_2026() -> pd.DataFrame:
    rat = pd.read_csv(CLEAN / "ratings_2026.csv")
    cand = pd.read_csv(CLEAN / "candidates_2026.csv")
    lean = pd.read_csv(CLEAN / "lean_2026.csv")
    fec = pd.read_csv(CLEAN / "fec_2026.csv")
    fec_race = pd.read_csv(CLEAN / "fec_race_2026.csv")
    k = kalshi_candidates()
    tab = rat[["race_id", "chamber", "cook", "sabato", "inside", "mean_rating", "imputed_safe", "same_party_final",
               "cook_raw", "sabato_raw", "inside_raw"]].copy()
    for c in ("cook", "sabato", "inside"):
        tab[f"r_{c}"] = -tab[c]
    tab["r_mean"] = -tab["mean_rating"]
    tab["r_spread"] = tab[["cook", "sabato", "inside"]].max(axis=1) - tab[["cook", "sabato", "inside"]].min(axis=1)
    tab = pd.concat([tab, other_raters(rat)], axis=1)
    tab = tab.merge(lean[["race_id", "pres24_dem_2p"]], on="race_id", how="left")
    tab["lean"] = 100 * (tab["pres24_dem_2p"] - NAT_PRES[2024])
    info = cand.groupby("race_id").agg(system=("system", "first"), uncontested=("uncontested", "max"),
                                        open_seat=("open_seat", "max"))
    tab = tab.merge(info, on="race_id", how="left")
    recs = []
    for row in tab.itertuples():
        kc = k[k["race_id"] == row.race_id]
        cc = cand[cand["race_id"] == row.race_id]
        rec = {"race_id": row.race_id, "same_party": int(isinstance(row.same_party_final, str)),
               "la_jungle": int(row.race_id in LA_JUNGLE), "dside_is_ind": 0}
        dem = kc[kc["party"] == "DEM"]
        rep = kc[kc["party"] == "REP"]
        if row.race_id in IND_PRINCIPAL_2026:
            rec["dside_name"], rec["dside_is_ind"] = IND_PRINCIPAL_2026[row.race_id], 1
        else:
            rec["dside_name"] = dem["candidate"].iloc[0] if len(dem) else None
        rec["rside_name"] = rep["candidate"].iloc[0] if len(rep) else None
        if row.race_id == "CA-06":
            rec["rside_name"] = "Kevin Kiley"
        # incumbency of the principals (candidates_2026 is_incumbent)
        inc = 0
        for side, sign in (("dside_name", 1), ("rside_name", -1)):
            nm = rec[side]
            if nm and not nm.endswith(" party"):
                hit = cc[cc["candidate_name"].map(last_name) == last_name(nm)]
                if len(hit) and int(hit["is_incumbent"].max()) == 1:
                    inc += sign
        if row.race_id == "AK-SEN":      # parties only on Kalshi; incumbent Sullivan is R
            inc = -1
        rec["inc"] = inc
        # receipts of the principals
        fr = fec[fec["race_id"] == row.race_id]
        def receipts(nm):
            if not nm:
                return 0.0
            if nm.endswith(" party"):
                party = "DEM" if nm.startswith("Democratic") else "REP"
                return float(fr.loc[fr["party"] == party, "receipts"].fillna(0).sum())
            hit = fr[fr["kalshi_name"].fillna("").map(last_name) == last_name(nm)]
            return float(hit["receipts"].fillna(0).sum())
        d_r, r_r = receipts(rec["dside_name"]), receipts(rec["rside_name"])
        if row.race_id in LA_JUNGLE:   # all-party primary: use all active filers of each party
            fx = fec_race[fec_race["race_id"] == row.race_id].iloc[0]
            d_r, r_r = float(fx["dem_receipts"]), float(fx["rep_receipts"])
        rec["d_receipts"], rec["r_receipts"] = d_r, r_r
        rec["fec_lr"] = clipped_log_ratio(d_r, r_r)
        recs.append(rec)
    tab = tab.merge(pd.DataFrame(recs), on="race_id")
    tab["no_contest"] = ((tab["dside_name"].isna() | tab["rside_name"].isna()
                          | tab["rside_name"].fillna("").eq("Republican party") & ~tab["race_id"].isin(LA_JUNGLE)
                          & ~tab["race_id"].eq("AK-SEN")) & (tab["same_party"] == 0)).astype(int)
    tab = _polls_2026(tab)
    tab["year"] = 2026
    tab["midterm"], tab["pres_d"], tab["out_d"] = 1, -1, 1
    gb = pd.read_csv(CLEAN / "national_2026.csv")
    gba = gb[gb["metric"] == "generic_ballot_avg"]
    tab["generic_margin_dr"] = float((gba["value_dem"] - gba["value_rep"]).mean())
    return tab


if __name__ == "__main__":
    h = hist_table()
    print(h.groupby("year").agg(n=("race_id", "size"), fec_match=("fec_matched", "mean"),
                                polled=("n_polls", lambda s: (s > 0).sum()), inc_d=("inc", lambda s: (s == 1).sum()),
                                inc_r=("inc", lambda s: (s == -1).sum())))
    t = table_2026()
    print(t[["race_id", "r_mean", "lean", "inc", "fec_lr", "poll_margin", "n_polls", "no_contest", "same_party"]]
          .sort_values("r_mean").to_string(max_rows=40))
