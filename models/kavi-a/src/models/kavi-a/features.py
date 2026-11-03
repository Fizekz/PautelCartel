"""Historical race table (2010-2024) for fitting the vote-share model.

One row per general election with a D and an R (or a D-role independent) candidate. Every feature is something
known before that year's election day: presidential lean on that year's lines, incumbency (FEC I/C/O flag),
cycle receipts, the previous House result on unchanged lines, final pre-election polls and expert ratings.
"""
from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

from common import (APPOINTED_INCUMBENTS, CLEAN, ELECTION_DAY, fec_last_name, last_name, tokens, lines_unchanged, money_feature,
                    national_table)

YEARS = list(range(2010, 2025, 2))
POLL_WINDOW_DAYS = 60
FUZZY_CUTOFF = 0.75


def load_results() -> pd.DataFrame:
    h = pd.read_csv(CLEAN / "results_house.csv")
    h["chamber"] = "House"
    s = pd.read_csv(CLEAN / "results_senate.csv")
    s["chamber"] = "Senate"
    res = pd.concat([h, s], ignore_index=True)
    return res[res["year"].between(2008, 2024)].copy()


PRE_ELECTION_FEC = CLEAN.parent / "kavi-a" / "fec_preelection.csv"


def fec_lookup(pre_year: int | None = None) -> pd.DataFrame:
    """FEC candidate table. With pre_year, that year's receipts are replaced by the pre-election snapshot
    (fetch_fec_preelection.py): join on cand_id, else unique surname in the same race; a candidate absent from the
    snapshot (registered after it) gets 0. Training years keep the two-year totals (a consistent definition)."""
    f = pd.read_csv(CLEAN / "fec_hist.csv")
    f["last"] = f["cand_name"].map(fec_last_name)
    f["receipts"] = pd.to_numeric(f["receipts"], errors="coerce").fillna(0.0)
    if pre_year is not None:
        f = apply_preelection(f, pre_year)
    return f


def apply_preelection(f: pd.DataFrame, year: int) -> pd.DataFrame:
    pre = pd.read_csv(PRE_ELECTION_FEC)
    pre = pre[pre["year"] == year].copy()
    if pre.empty:
        raise SystemExit(f"no pre-election FEC snapshot for {year}")
    pre["last"] = pre["cand_name"].map(fec_last_name)
    by_id = pre.set_index("cand_id")["receipts_pre"]
    by_name = pre.groupby(["race_id", "last"])["receipts_pre"].agg(["sum", "size"])
    out = f.copy()
    mask = out["year"] == year
    vals = []
    for r in out[mask].itertuples():
        if r.cand_id in by_id.index:
            vals.append(float(by_id[r.cand_id]))
            continue
        key = (str(r.race_id).replace("-SEN-S", "-SEN"), r.last)
        hit = by_name.loc[key] if key in by_name.index else None
        vals.append(float(hit["sum"]) if hit is not None and hit["size"] == 1 else 0.0)
    out.loc[mask, "receipts"] = np.nan_to_num(np.array(vals, dtype=float))
    return out


def match_fec(fec: pd.DataFrame, year: int, race_id: str, name: str, bucket: str) -> tuple[str, float]:
    """(ici, receipts) for a general-election candidate; surname match within party bucket, then any bucket."""
    ici, rec, _ = match_fec_full(fec, year, race_id, name, bucket)
    return ici, rec


def match_fec_full(fec: pd.DataFrame, year: int, race_id: str, name: str, bucket: str,
                   by_first_name: bool = False) -> tuple[str, float, str]:
    """(ici, receipts, FEC party bucket) of the best surname match: in the race first, then (House) anywhere in
    the state, because redistricted incumbents often file under their old district number (Schweikert 2022)."""
    if not isinstance(name, str) or not name:
        return "", 0.0, ""
    fid = race_id.replace("-SEN-S", "-SEN")
    yr = fec[fec["year"] == year]
    hit = surname_hits(yr[yr["race_id"] == fid], name, bucket, allow_any_party=True)
    if hit.empty and not fid.endswith("SEN"):
        state = yr[yr["race_id"].str.startswith(fid[:2] + "-") & ~yr["race_id"].str.endswith("SEN")]
        hit = surname_hits(state, name, bucket, allow_any_party=False)
        hit = hit if len(hit) == 1 else hit.iloc[0:0]  # only an unambiguous statewide match
    if hit.empty:
        return "", 0.0, ""
    if by_first_name and hit["cand_name"].nunique() > 1:  # v3: WA-09 2018 Adam / Sarah Smith
        first = tokens(name)[0] if tokens(name) else ""
        by_first = hit[hit["cand_name"].map(lambda n: first in tokens(n))]
        hit = by_first if not by_first.empty else hit
    row = hit.sort_values("receipts", ascending=False).iloc[0]
    return (str(row["ici"]) if pd.notna(row["ici"]) else "", float(row["receipts"]), str(row["party_bucket"]))


def surname_hits(sub: pd.DataFrame, name: str, bucket: str, allow_any_party: bool) -> pd.DataFrame:
    if sub.empty:
        return sub
    ln = last_name(name)
    same = sub[sub["party_bucket"] == bucket]
    hit = same[same["last"] == ln]
    if hit.empty and allow_any_party:
        hit = sub[sub["last"] == ln]
    if hit.empty:  # compound surnames: MADELEINE DEAN CUNNANE = DEAN; ARENHOLZ, ASHLEY HINSON = Hinson
        hit = same[same["last"].isin(tokens(name)) | same["cand_name"].map(lambda n: ln in tokens(n)).astype(bool)]
    if hit.empty:  # MEDSL misspellings (CORTNEY, SCAKOWSKY): fuzzy surname within the party bucket
        close = difflib.get_close_matches(ln, same["last"].tolist(), n=1, cutoff=FUZZY_CUTOFF)
        hit = same[same["last"] == close[0]] if close else hit
    return hit


def lean_columns(lean: pd.DataFrame) -> pd.DataFrame:
    out = lean.copy()
    out["lean"] = out["pres_prev_dem_2p"] - out["national_pres_dem_2p"]
    out["lean2"] = out["pres_prev2_dem_2p"] - out["national_pres_prev2_dem_2p"]
    return out[["year", "race_id", "chamber", "lean", "lean2"]]


def add_prev_house(df: pd.DataFrame, res: pd.DataFrame, nat: pd.DataFrame, lean: pd.DataFrame) -> pd.DataFrame:
    """Previous-cycle House result relative to lean + national vote, only where the lines did not change."""
    prev = res[(res["chamber"] == "House")][["year", "race_id", "dem_share_2p", "uncontested", "winner_name",
                                              "state"]].copy()
    prev = prev.merge(lean[["year", "race_id", "lean"]], on=["year", "race_id"], how="left")
    prev["nat_d2p"] = prev["year"].map(nat["house_d2p"])
    prev["prev_dev"] = prev["dem_share_2p"] - prev["nat_d2p"] - prev["lean"]
    prev.loc[prev["uncontested"] == 1, "prev_dev"] = np.nan
    prev["year"] = prev["year"] + 2
    prev = prev.rename(columns={"winner_name": "prev_winner"})[["year", "race_id", "prev_dev", "prev_winner"]]
    out = df.merge(prev, on=["year", "race_id"], how="left")
    same = np.array([lines_unchanged(y, s) for y, s in zip(out["year"], out["state"])])
    out.loc[~same | (out["chamber"] != "House"), ["prev_dev", "prev_winner"]] = np.nan
    return out


# Poll windows. "v1": final 60 days, recency weight exp(-days/21). "thin": the same, plus, for races with fewer
# than THIN_MIN polls inside 60 days, polls ending 61-EXT_DAYS days out at weight exp(-60/21) exp(-(days-60)/TAU2)
# (continuous at 60 days). "wide": EXT_DAYS window for every race, decay TAU_WIDE. pre-2018 538 files only hold
# the final ~60 days, so the variants differ only from 2018 on.
POLL_MODES = ("v1", "thin", "wide")
POLL_MODE = "v1"       # chosen on the 2020 + 2024 forward tests (LOG.md, v2): thin/wide did not help
POLLSTER_DAMP = False  # v3 option, tested in select_v3.py --pollster
THIN_MIN, EXT_DAYS, TAU1, TAU2, TAU_WIDE = 2, 150, 21.0, 30.0, 30.0


def recency_weight(days: pd.Series, n_in_window: pd.Series, mode: str) -> pd.Series:
    """Recency factor of each poll (0 = dropped) under a poll-window mode."""
    if mode == "wide":
        return np.where(days <= EXT_DAYS, np.exp(-days / TAU_WIDE), 0.0)
    inside = np.where(days <= POLL_WINDOW_DAYS, np.exp(-days / TAU1), 0.0)
    if mode == "v1":
        return inside
    ext = (days > POLL_WINDOW_DAYS) & (days <= EXT_DAYS) & (n_in_window < THIN_MIN)
    tail = np.exp(-POLL_WINDOW_DAYS / TAU1) * np.exp(-(days - POLL_WINDOW_DAYS) / TAU2)
    return np.where(ext, tail, inside)


def poll_average(polls: pd.DataFrame, year: int, election_day: str, mode: str = POLL_MODE) -> pd.DataFrame:
    """Recency/sample weighted two-party D share of general-election race polls before election day."""
    p = polls[(polls["year"] == year) & (polls["stage"] == "general") & (polls["is_hypothetical"] != True)].copy()
    p["end"] = pd.to_datetime(p["end_date"], errors="coerce")
    days = (pd.Timestamp(election_day) - p["end"]).dt.days
    p = p[(days >= 0) & (days <= EXT_DAYS)].copy()
    p["days"] = days[p.index]
    d = pd.to_numeric(p["dem_pct"], errors="coerce")
    r = pd.to_numeric(p["rep_pct"], errors="coerce")
    o = pd.to_numeric(p["other_pct"], errors="coerce")
    d = d.where(d.notna(), o)  # D-role independent (McMullin, Orman, King when no D)
    p["share"] = d / (d + r)
    p = p[p["share"].between(0.02, 0.98)]
    return summarise_polls(one_row_per_poll(p), ["race_id", "special"], mode)


POP_RANK = {"lv": 0, "v": 1, "rv": 2, "a": 3}
POLL_DEDUP = "none"   # v3 switch: "none" (v2: every question row counts), "lv" (preferred population), "avg" (all)


def one_row_per_poll(p: pd.DataFrame) -> pd.DataFrame:
    """v3 bug fix: 538 lists a poll once per question (LV / RV / V versions, turnout models), and those rows were
    each counted as a separate poll (2018-2024 only; 1,147 duplicated rows). Keep the preferred population
    (LV > V > RV > A) and average its versions into one row."""
    if POLL_DEDUP == "none" or "poll_id" not in p or p["poll_id"].isna().all():
        return p
    if POLL_DEDUP == "avg":
        p = p.assign(population="lv")
    p = p.assign(pop_rank=p["population"].fillna("").str.lower().map(POP_RANK).fillna(4),
                 pid=p["poll_id"].fillna(-1 - pd.Series(range(len(p)), index=p.index)))
    keys = [k for k in ("race_id", "special") if k in p] + ["pid"]
    p = p[p["pop_rank"] == p.groupby(keys)["pop_rank"].transform("min")]
    share = p.groupby(keys)["share"].transform("mean")
    return p.assign(share=share).drop_duplicates(keys).drop(columns=["pop_rank", "pid"])


def summarise_polls(p: pd.DataFrame, keys: list[str], mode: str = POLL_MODE) -> pd.DataFrame:
    """p needs days, share, sample_size, partisan_sponsor; returns poll_share, n_polls, poll_weight per key."""
    n_in = p.assign(inw=p["days"] <= POLL_WINDOW_DAYS).groupby(keys)["inw"].transform("sum")
    rec = recency_weight(p["days"], n_in, mode)
    n = pd.to_numeric(p["sample_size"], errors="coerce").fillna(500).clip(200, 3000)
    partisan = p["partisan_sponsor"].fillna("").astype(str).str.len() > 0
    p = p.assign(w=np.sqrt(n / 600.0) * rec * np.where(partisan, 0.5, 1.0))
    p = p[p["w"] > 0]
    if POLLSTER_DAMP and "pollster" in p:  # v3 option (kavi-e): a pollster's k polls in a race weigh 1/sqrt(k) each
        k = p.groupby(keys + [p["pollster"].fillna("?").str.lower()])["w"].transform("size")
        p = p.assign(w=p["w"] / np.sqrt(k))
    side = sponsor_side(p["partisan_sponsor"])
    p = p.assign(ws=p["w"] * p["share"], wd=p["w"] * (side == "D"), wr=p["w"] * (side == "R"))
    g = p.groupby(keys).agg(ws=("ws", "sum"), w=("w", "sum"), wd=("wd", "sum"), wr=("wr", "sum"),
                            n_polls=("share", "size"))
    g["poll_share"] = g["ws"] / g["w"]
    g["poll_weight"] = g["w"]
    # v3: weight fractions of D- and R-sponsored polls, so the sponsor shift can be applied per fit cutoff
    g["poll_frac_d"] = g["wd"] / g["w"]
    g["poll_frac_r"] = g["wr"] / g["w"]
    return g[["poll_share", "n_polls", "poll_weight", "poll_frac_d", "poll_frac_r"]].reset_index()


def sponsor_side(sponsor: pd.Series) -> pd.Series:
    """'D' / 'R' / '' from the sponsor tag (history: DEM, REP, 'REP,REF'; 2026: D, R, D/R = mixed -> '')."""
    s = sponsor.fillna("").astype(str).str.upper().str.strip()
    d = s.str.startswith("D") & ~s.str.contains("R")
    r = s.str.startswith("R") & ~s.str.contains("D")
    return pd.Series(np.where(d, "D", np.where(r, "R", "")), index=sponsor.index)


QUANT_OUTLETS_HIST = ["other_ddhq", "other_economist", "other_split_ticket"]  # + other_fte (538), kept apart
QUANT_OUTLETS_2026 = ["other_ddhq", "other_economist", "other_fiftyplusone", "other_silver_bulletin",
                      "other_split_ticket"]


def ratings_table() -> pd.DataFrame:
    """Core rating (mean of Cook/Sabato/Inside, + = D) and the quantitative outlets' mean category.

    rating_quant_nofte: DDHQ / Economist / Split Ticket only (the one usable in the 2018/2022 backtests: 538's own
    category is the benchmark's); rating_quant: the same plus 538, used only to tune the blend on 2020/2024."""
    r = pd.read_csv(CLEAN / "ratings_hist.csv")
    r["rating"] = -pd.to_numeric(r["mean_rating"], errors="coerce")  # flip so positive = D
    q = r[QUANT_OUTLETS_HIST].apply(pd.to_numeric, errors="coerce")
    r["rating_quant_nofte"] = -q.mean(axis=1)
    r["rating_quant"] = -pd.concat([q, pd.to_numeric(r["other_fte"], errors="coerce")], axis=1).mean(axis=1)
    r["special"] = r["special"].astype(int)
    return r[["year", "race_id", "rating", "rating_quant", "rating_quant_nofte"]]


def build_history() -> pd.DataFrame:
    res = load_results()
    nat = national_table()
    lean = lean_columns(pd.read_csv(CLEAN / "lean_hist.csv"))
    fec = fec_lookup()
    df = res[res["year"].isin(YEARS)].copy()
    df["lean_key"] = np.where(df["chamber"] == "Senate", df["state"] + "-SEN", df["race_id"])
    df = df.merge(lean.rename(columns={"race_id": "lean_key"}).drop(columns="chamber"),
                  on=["year", "lean_key"], how="left")
    df["d_role_ind"] = 0
    note = df["notes"].fillna("")
    ind = note.str.contains("major non-D/R") & (df["dem_votes"].fillna(0) == 0)
    df.loc[ind, "d_role_ind"] = 1  # e.g. KS 2014 Orman, UT 2022 McMullin, NE 2024 Osborn, VT/ME (King/Sanders)
    rows = []
    for r in df.itertuples():
        di, dm = match_fec(fec, r.year, r.race_id, r.dem_cand, "DEM")
        ri, rm = match_fec(fec, r.year, r.race_id, r.rep_cand, "REP")
        rows.append((di, dm, ri, rm))
    df[["dem_ici", "dem_rec", "rep_ici", "rep_rec"]] = pd.DataFrame(rows, index=df.index)
    for (yr, rid), surname in APPOINTED_INCUMBENTS.items():
        for side in ("dem", "rep"):
            hit = (df["year"] == yr) & (df["race_id"] == rid) & df[f"{side}_cand"].fillna("").str.lower().str.contains(surname)
            df.loc[hit, f"{side}_ici"] = "I"
    df["inc"] = (df["dem_ici"] == "I").astype(int) - (df["rep_ici"] == "I").astype(int)
    df["open"] = ((df["dem_ici"] != "I") & (df["rep_ici"] != "I")).astype(int)
    df["money"] = money_feature(df["dem_rec"], df["rep_rec"])
    df = add_prev_house(df, res, nat, lean)
    df["prev_same_inc"] = [int(isinstance(pw, str) and isinstance(w, str) and last_name(pw) == last_name(w))
                           for pw, w in zip(df["prev_winner"], np.where(df["inc"] > 0, df["dem_cand"],
                                                                        np.where(df["inc"] < 0, df["rep_cand"], "")))]
    polls = pd.read_csv(CLEAN / "polls_hist.csv", low_memory=False)
    polls["special"] = polls["special"].astype(str).str.lower().eq("true").astype(int)
    df["poll_key"] = np.where(df["chamber"] == "Senate", df["state"] + "-SEN", df["race_id"])
    for mode in POLL_MODES:  # every mode is stored (poll_share_<mode> ...); poll_share = POLL_MODE
        pa = pd.concat([poll_average(polls, y, ELECTION_DAY[y], mode).assign(year=y) for y in YEARS],
                       ignore_index=True).rename(columns={"race_id": "poll_key"})
        sfx = "" if mode == POLL_MODE else f"_{mode}"
        pa = pa.rename(columns={c: c + sfx for c in ("poll_share", "n_polls", "poll_weight", "poll_frac_d",
                                                     "poll_frac_r")})
        df = df.merge(pa, on=["year", "poll_key", "special"], how="left")
    for c in ("poll_share", "n_polls", "poll_weight"):
        df[f"{c}_{POLL_MODE}"] = df[c]
    df = df.merge(ratings_table(), on=["year", "race_id"], how="left")
    df["g"] = df["year"].map(nat["generic_margin_dr"]) / 200.0
    df["midterm"] = df["year"].map(nat["midterm"])
    df["pres_d"] = df["year"].map(nat["pres_d"])
    df["approve_net"] = df["year"].map(nat["net_approval"])
    return df


def trainable(df: pd.DataFrame) -> pd.Series:
    """Rows fit by the vote-share regression: contested D vs R, plain plurality-style races with lean."""
    note = df["notes"].fillna("")
    ok = (df["uncontested"] == 0) & df["dem_share_2p"].between(0.03, 0.97) & df["lean"].notna()
    ok &= ~note.str.contains("major non-D/R")
    ok &= ~((df["state"] == "AK") & (df["year"] >= 2022))  # RCV first-round sums
    ok &= ~((df["state"] == "LA") & (df["chamber"] == "House"))  # jungle party sums
    return ok


if __name__ == "__main__":
    hist = build_history()
    hist.to_csv(CLEAN.parent.parent / "out" / "hist_features.csv", index=False)
    t = hist[trainable(hist)]
    print(hist.shape, t.shape)
    print(t.groupby(["year", "chamber"]).size().unstack())
    print(t[["lean", "inc", "money", "prev_dev", "poll_share", "rating"]].describe().T)
