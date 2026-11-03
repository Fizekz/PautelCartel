"""Race-level feature table for kavi-d.

One row per (year, race_id) contest between an "anti-R" side (the Democrat, or the main independent when he is the
non-Republican challenger) and the Republican. Every feature is signed so that positive favours the anti-R side.

Structure (who the two sides are) is passed in as a DataFrame with columns
    year, race_id, chamber, state, special, d_name, r_name, d_is_ind
so that training years can take it from results and prediction years from pre-election candidate lists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (CLEAN, ELECTION_DAY, FEC_LR_CLIP, POLL_SPARSE_MIN, POLL_WINDOW_DAYS, POLL_WINDOW_SPARSE_DAYS,
                    fec_surname_tokens, generic_margin, midterm_outparty_sign, name_matches, national_house_margin,
                    prefer_given_name, redrawn, tokens)

FEATURES = ["r_mean", "r_cook", "r_sab", "r_ins", "r_spread", "imputed_safe", "lean_m", "base_m", "generic",
            "inc", "seat", "prev_rel", "fec_share", "fec_lr", "fec_logtot", "poll_m", "poll_n", "senate", "op", "oth_dev",
            "rcp_dev"]


def base_race_id(race_id: str) -> str:
    return race_id[:-2] if race_id.endswith("-SEN-S") else race_id


# ----------------------------------------------------------------------------------------------- ratings / lean
RATING_COLS = ["year", "race_id", "cook", "sabato", "inside", "imputed_safe", "pvi_wiki", "incumbent_party", "oth",
               "rcp"]
# Outlets excluded from the "other raters" mean: 538 (the backtest benchmark itself).
OTHER_EXCLUDE = {"other_fte"}


def _with_other_raters(r: pd.DataFrame) -> pd.DataFrame:
    """oth = mean of the non-core outlets that rated the race (blank = none did); rcp = RealClearPolitics alone."""
    cols = [c for c in r.columns if c.startswith("other_") and c not in OTHER_EXCLUDE]
    return r.assign(oth=r[cols].mean(axis=1), rcp=r["other_rcp"])


def ratings_hist() -> pd.DataFrame:
    return _with_other_raters(pd.read_csv(CLEAN / "ratings_hist.csv"))[RATING_COLS]


def ratings_2026() -> pd.DataFrame:
    r = pd.read_csv(CLEAN / "ratings_2026.csv")
    r["year"] = 2026
    return _with_other_raters(r)[RATING_COLS]


def add_ratings(df: pd.DataFrame, rat: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(rat, on=["year", "race_id"], how="left")
    for src, dst in [("cook", "r_cook"), ("sabato", "r_sab"), ("inside", "r_ins")]:
        m[dst] = -m[src]  # rating scale is -3 Safe D .. +3 Safe R; flip so + favours anti-R
    m["r_mean"] = m[["r_cook", "r_sab", "r_ins"]].mean(axis=1)
    m["r_spread"] = m[["r_cook", "r_sab", "r_ins"]].max(axis=1) - m[["r_cook", "r_sab", "r_ins"]].min(axis=1)
    m["imputed_safe"] = m["imputed_safe"].fillna(0)
    # Second opinion: other outlets' mean minus the core mean (D-positive). Blank where no other outlet rated it.
    m["oth_dev"] = -m["oth"] - m["r_mean"]
    m["rcp_dev"] = -m["rcp"] - m["r_mean"]
    return m.drop(columns=["cook", "sabato", "inside", "oth", "rcp"])


def lean_table(year: int) -> pd.DataFrame:
    """lean_m = 200 x (prior presidential D 2-party share - national), i.e. a relative margin in points."""
    if year == 2026:
        l26 = pd.read_csv(CLEAN / "lean_2026.csv")
        return pd.DataFrame({"race_id": l26["race_id"], "chamber": l26["chamber"],
                             "lean_m": 200 * (l26["pres24_dem_2p"] - 0.49250),
                             "changed": l26["district_changed_2026"]})
    lh = pd.read_csv(CLEAN / "lean_hist.csv")
    lh = lh[lh["year"] == year]
    return pd.DataFrame({"race_id": lh["race_id"], "chamber": lh["chamber"],
                         "lean_m": 200 * (lh["pres_prev_dem_2p"] - lh["national_pres_dem_2p"]), "changed": np.nan})


def add_lean(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for year, g in df.groupby("year"):
        lt = lean_table(int(year))
        house = lt[lt["chamber"] == "House"][["race_id", "lean_m", "changed"]]
        sen = lt[lt["chamber"] == "Senate"].assign(state=lambda d: d["race_id"].str[:2])[["state", "lean_m"]]
        gh = g[g["chamber"] == "House"].merge(house, on="race_id", how="left")
        gs = g[g["chamber"] == "Senate"].merge(sen.drop_duplicates("state"), on="state", how="left")
        parts += [gh, gs]
    m = pd.concat(parts, ignore_index=True)
    # Fallback for blank lean (CA 2012, FL 2016): Cook PVI printed on that year's page (R positive, share points).
    fill = -2.0 * m["pvi_wiki"]
    m["lean_m"] = m["lean_m"].fillna(fill)
    return m


# ----------------------------------------------------------------------------------------------- incumbency / prev
def add_prev_result(df: pd.DataFrame) -> pd.DataFrame:
    """House only: previous cycle's D margin in the same district, relative to that cycle's national House margin.
    Blank where the state's lines changed this cycle (same id != same district) or the previous race was one-sided."""
    h = pd.read_csv(CLEAN / "results_house.csv")
    h = h[h["special"] == 0]
    nat = national_house_margin()
    h = h.assign(prev_rel=np.where((h["uncontested"] == 0) & h["dem_share_2p"].notna(),
                                   100 * (2 * h["dem_share_2p"] - 1) - h["year"].map(nat), np.nan))
    prev = h[["year", "race_id", "prev_rel"]].assign(year=lambda d: d["year"] + 2)
    m = df.merge(prev, on=["year", "race_id"], how="left")
    bad = (m["chamber"] != "House") | np.array([redrawn(int(y), s) for y, s in zip(m["year"], m["state"])])
    if "changed" in m:  # 2026: unchanged districts inside redrawn states keep their history
        keep = (m["year"] == 2026) & (m["changed"] == 0) & (m["chamber"] == "House")
        bad = bad & ~keep
    m.loc[bad, "prev_rel"] = np.nan
    return m


def seat_party_feature(df: pd.DataFrame) -> pd.Series:
    """+1 seat held by D, -1 by R (House only; blank where lines changed)."""
    sp = df["incumbent_party"].map({"D": 1.0, "R": -1.0, "I": 0.0})
    bad = (df["chamber"] != "House") | np.array([redrawn(int(y), s) for y, s in zip(df["year"], df["state"])])
    if "changed" in df:
        bad = bad & ~((df["year"] == 2026) & (df["changed"] == 0))
    return sp.where(~bad)


# ----------------------------------------------------------------------------------------------- FEC
def fec_match(fec: pd.DataFrame, name: str, party_bucket: str | None) -> pd.Series | None:
    """Best FEC row for a nominee: surname match within the race (prefer party bucket, then most receipts)."""
    if not isinstance(name, str) or not name:
        return None
    hit = prefer_given_name(fec[fec["_tok"].map(lambda t: name_matches(t, name))], name)
    if party_bucket is not None and (hit["party_bucket"] == party_bucket).any():
        hit = hit[hit["party_bucket"] == party_bucket]
    if hit.empty:
        return None
    return hit.sort_values("receipts", ascending=False, na_position="last").iloc[0]


def add_fec_hist(df: pd.DataFrame) -> pd.DataFrame:
    fec = pd.read_csv(CLEAN / "fec_hist.csv")
    fec["_tok"] = fec["cand_name"].map(fec_surname_tokens)
    groups = {k: g for k, g in fec.groupby(["year", "race_id"])}
    rows = []
    for r in df.itertuples():
        g = groups.get((r.year, base_race_id(r.race_id)), fec.iloc[0:0])
        d = fec_match(g, r.d_name, None if r.d_is_ind else "DEM")
        rr = fec_match(g, r.r_name, "REP")
        rows.append({"d_rec": np.nan if d is None else d["receipts"], "r_rec": np.nan if rr is None else rr["receipts"],
                     "d_ici": None if d is None else d["ici"], "r_ici": None if rr is None else rr["ici"]})
    x = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, x], axis=1)


def finish_fec(m: pd.DataFrame) -> pd.DataFrame:
    d = m["d_rec"].fillna(0).clip(lower=0)
    r = m["r_rec"].fillna(0).clip(lower=0)
    have = m["d_rec"].notna() | m["r_rec"].notna()
    m["fec_share"] = np.where(have & (d + r > 0), d / (d + r).replace(0, np.nan), np.nan)
    m["fec_lr"] = np.where(have, np.clip(np.log((d + 1e4) / (r + 1e4)), -FEC_LR_CLIP, FEC_LR_CLIP), np.nan)
    m["fec_logtot"] = np.where(have, np.log10(d + r + 1e4), np.nan)
    return m


# ----------------------------------------------------------------------------------------------- polls
def poll_average(polls: pd.DataFrame, eday: str) -> tuple[float, int]:
    """Recency- and sample-weighted D-R margin over the final window; each pollster's weight is shared."""
    if polls.empty:
        return np.nan, 0
    days = (pd.Timestamp(eday) - pd.to_datetime(polls["end_date"])).dt.days.clip(lower=0)
    n = polls["sample_size"].fillna(500).clip(100, 3000)
    w = np.exp(-days / 21.0) * np.sqrt(n / 600.0)
    w = w / polls.groupby("pollster")["pollster"].transform("size")
    return float(np.average(polls["margin"], weights=w)), int(len(polls))


def hist_poll_rows(year: int) -> pd.DataFrame:
    p = pd.read_csv(CLEAN / "polls_hist.csv")
    p = p[(p["year"] == year) & (p["stage"] != "runoff") & (~p["is_hypothetical"].fillna(False).astype(bool))]
    eday = pd.Timestamp(ELECTION_DAY[year])
    end = pd.to_datetime(p["end_date"])
    p = p[(end <= eday) & (end >= eday - pd.Timedelta(days=POLL_WINDOW_SPARSE_DAYS))].copy()
    return p


def windowed(polls: pd.DataFrame, eday: str) -> pd.DataFrame:
    """Final POLL_WINDOW_DAYS; if fewer than POLL_SPARSE_MIN matched polls fall in it, extend to
    POLL_WINDOW_SPARSE_DAYS (summer polls in under-polled races). 2010-2016 538 files only keep ~60 days, so the
    extension can only bite from 2018 on."""
    if polls.empty or POLL_WINDOW_SPARSE_DAYS <= POLL_WINDOW_DAYS:
        cut = pd.Timestamp(eday) - pd.Timedelta(days=POLL_WINDOW_DAYS)
        return polls[pd.to_datetime(polls["end_date"]) >= cut]
    cut = pd.Timestamp(eday) - pd.Timedelta(days=POLL_WINDOW_DAYS)
    near = polls[pd.to_datetime(polls["end_date"]) >= cut]
    return near if len(near) >= POLL_SPARSE_MIN else polls


def match_poll_margin(p: pd.DataFrame, d_name: str, r_name: str, other_col: str = "other_pct",
                      other_cand_col: str = "other_cand") -> pd.DataFrame:
    """Keep polls whose R matches the R nominee and whose D (or 'other' independent) matches the anti-R nominee."""
    out = []
    for r in p.itertuples():
        if not name_matches(tokens(r.rep_cand), r_name):
            continue
        if isinstance(r.dem_cand, str) and name_matches(tokens(r.dem_cand), d_name):
            dpct = r.dem_pct
        elif isinstance(getattr(r, other_cand_col), str) and name_matches(tokens(getattr(r, other_cand_col)), d_name):
            dpct = getattr(r, other_col)
        else:
            continue
        if pd.notna(dpct) and pd.notna(r.rep_pct):
            out.append((r.pollster, r.end_date, r.sample_size, dpct - r.rep_pct))
    return pd.DataFrame(out, columns=["pollster", "end_date", "sample_size", "margin"])


def add_polls_hist(df: pd.DataFrame) -> pd.DataFrame:
    vals = {}
    for year, g in df.groupby("year"):
        p = hist_poll_rows(int(year))
        sen_two = set(g.loc[g["race_id"].str.endswith("-SEN-S"), "race_id"].str[:2])
        rid = np.where(p["special"].fillna(False).astype(bool) & p["race_id"].str[:2].isin(sen_two),
                       p["race_id"] + "-S", p["race_id"])
        p = p.assign(race_id=rid)
        by = {k: v for k, v in p.groupby("race_id")}
        for r in g.itertuples():
            pr = by.get(r.race_id)
            if pr is None:
                vals[r.Index] = (np.nan, 0)
                continue
            eday = ELECTION_DAY[int(year)]
            vals[r.Index] = poll_average(windowed(match_poll_margin(pr, r.d_name, r.r_name), eday), eday)
    df = df.copy()
    df["poll_m"] = [vals[i][0] for i in df.index]
    df["poll_n"] = [vals[i][1] for i in df.index]
    return df


# ----------------------------------------------------------------------------------------------- assemble
def common_finish(m: pd.DataFrame) -> pd.DataFrame:
    m["generic"] = m["year"].map(lambda y: generic_margin(int(y)))
    m["base_m"] = m["lean_m"] + m["generic"]
    m["senate"] = (m["chamber"] == "Senate").astype(float)
    m["seat"] = seat_party_feature(m)
    # Midterm out-party indicator, D-positive: +1 when the anti-R side is the out-party in a midterm (R president),
    # -1 when it is the president's party in a midterm, 0 in presidential years. Mirror-symmetric by construction.
    m["op"] = m["year"].map(lambda y: midterm_outparty_sign(int(y)))
    return m


def build_hist_features(struct: pd.DataFrame) -> pd.DataFrame:
    """Features for historical years (2010-2024) given a structure table."""
    m = add_ratings(struct, ratings_hist())
    m = add_lean(m)
    m = add_prev_result(m)
    m = add_fec_hist(m)
    m = finish_fec(m)
    m["inc"] = np.where(m["d_ici"] == "I", 1.0, 0.0) - np.where(m["r_ici"] == "I", 1.0, 0.0)
    m = add_polls_hist(m)
    return common_finish(m)


# ----------------------------------------------------------------------------------------------- 2026
def poll_rows_2026() -> pd.DataFrame:
    """Usable 2026 race polls (polls_2026.md rule) in the final window before election day."""
    p = pd.read_csv(CLEAN / "polls_2026.csv")
    nm = p["nominee_match"]
    usable = (nm == True) | (nm.isna() & (p["is_hypothetical"] == False))  # noqa: E712
    usable &= ~p["rcv_round"].astype(str).isin(["2", "3", "BA"])
    eday = pd.Timestamp(ELECTION_DAY[2026])
    end = pd.to_datetime(p["end_date"])
    return p[usable & (end <= eday) & (end >= eday - pd.Timedelta(days=POLL_WINDOW_SPARSE_DAYS))].copy()


def add_polls_2026(df: pd.DataFrame) -> pd.DataFrame:
    p = poll_rows_2026()
    by = {k: v for k, v in p.groupby("race_id")}
    pm, pn = [], []
    for r in df.itertuples():
        pr = by.get(r.race_id)
        if pr is None:
            pm.append(np.nan)
            pn.append(0)
            continue
        mp = match_poll_margin(pr, r.d_name, r.r_name, "top_other_pct", "top_other_cand")
        m, n = poll_average(windowed(mp, ELECTION_DAY[2026]), ELECTION_DAY[2026])
        pm.append(m)
        pn.append(n)
    return df.assign(poll_m=pm, poll_n=pn)


def build_2026_features(struct: pd.DataFrame) -> pd.DataFrame:
    """struct needs year=2026, race_id, chamber, state, special, d_name, r_name, d_is_ind, d_inc, r_inc,
    d_rec, r_rec (receipts of the two sides)."""
    m = add_ratings(struct, ratings_2026())
    m = add_lean(m)
    m = add_prev_result(m)
    m = finish_fec(m)
    m["inc"] = m["d_inc"].astype(float) - m["r_inc"].astype(float)
    m = add_polls_2026(m)
    return common_finish(m)
