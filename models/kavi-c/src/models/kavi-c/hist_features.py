"""Race-level feature table for 2010-2024 (training and backtest rows).

One row per (year, race_id) from MIT results. Features are things knowable before election day:
partisan lean (previous presidential result), expert ratings (final pre-election), incumbency (FEC I/C/O
flag on the nominee), fundraising (nominee receipts, see METHODS.md for the post-election-report caveat),
previous House result in the same district when lines are unchanged, national generic ballot.
The target (dem_share_2p) and winner come from the same results rows and are only used as labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from functools import lru_cache

from common import CLEAN, FEC_2018_PRE, FEC_2022_PRE, HIST_YEARS, last_name, money_feature, name_tokens

USE_FEC_2018_PRE = True   # False: v1 behaviour (full-cycle 2018 totals), for the ablation only
# v3: 2022 test-year receipts from the latest pre-election cut (Wayback webl22.zip of 2022-08-03, coverage to
# 2022-06-30; borrowed from kavi-a). Switched on only while backtesting 2022 (`backtest.run_year`), so 2022 stays a
# full-cycle training year for 2024/2026 like 2010-16 and 2020-24.
USE_FEC_2022_PRE = False


def _replace_receipts(fec: pd.DataFrame, year: int, snap: pd.DataFrame) -> None:
    """Overwrite `year` receipts in place by the snapshot: cand_id, else (race, surname), else 0."""
    snap = snap.assign(ln=snap["cand_name"].map(last_name),
                       race_id_raw=snap["race_id_raw"].str.replace(r"-00$", "-AL", regex=True))
    by_id = snap.drop_duplicates("cand_id").set_index("cand_id")["receipts_pre"]
    by_name = snap.sort_values("receipts_pre", ascending=False).drop_duplicates(["race_id_raw", "ln"])
    by_name = by_name.set_index(["race_id_raw", "ln"])["receipts_pre"]
    m = fec["year"] == year
    pre = fec.loc[m, "cand_id"].map(by_id)
    alt = pd.Series([by_name.get((k, n), np.nan) for k, n in zip(fec.loc[m, "race_key"], fec.loc[m, "ln"])],
                    index=pre.index)
    fec.loc[m, "receipts"] = pre.fillna(alt).fillna(0.0).astype(float)


@lru_cache(maxsize=2)
def _load_fec_hist(use_2022_pre: bool) -> pd.DataFrame:
    fec = pd.read_csv(CLEAN / "fec_hist.csv")
    fec["race_key"] = fec["race_id"].str.replace("-SEN-S", "-SEN", regex=False)
    fec["ln"] = fec["cand_name"].map(last_name)
    fec["receipts"] = pd.to_numeric(fec["receipts"], errors="coerce").fillna(0.0)
    if USE_FEC_2018_PRE:
        _replace_receipts(fec, 2018, pd.read_csv(FEC_2018_PRE))
    if use_2022_pre:
        s22 = pd.read_csv(FEC_2022_PRE).rename(columns={"race_id": "race_id_raw"})
        _replace_receipts(fec, 2022, s22)
    return fec


def load_fec_hist() -> pd.DataFrame:
    """fec_hist.csv with 2018 receipts replaced by the pre-election (election-day) snapshot, and (only while the
    2022 backtest runs) 2022 receipts by the June 30 2022 cut.

    Other years keep full-cycle totals (through Dec 31, a mild post-election look-ahead in training only)."""
    return _load_fec_hist(USE_FEC_2022_PRE)


load_fec_hist.cache_clear = _load_fec_hist.cache_clear  # type: ignore[attr-defined]  (ablation scripts)


def pick_by_first_name(hit: pd.DataFrame, name: str) -> pd.DataFrame:
    """Several FEC rows share the surname (WA-09 2018 Adam v Sarah Smith): keep those whose first names match."""
    if hit["cand_id"].nunique() <= 1:
        return hit
    toks = name_tokens(name)
    if len(toks) < 2:
        return hit
    first = toks[0]
    sub = hit[hit["cand_name"].map(lambda s: first in name_tokens(str(s).split(",", 1)[-1]))]
    return sub if len(sub) else hit


def load_results() -> pd.DataFrame:
    house = pd.read_csv(CLEAN / "results_house.csv").assign(chamber="House")
    sen = pd.read_csv(CLEAN / "results_senate.csv").assign(chamber="Senate")
    res = pd.concat([house, sen], ignore_index=True)
    res = res[res["year"].isin(HIST_YEARS) & (res["special"].fillna(0).astype(int).eq(0) | res["chamber"].eq("Senate"))]
    res = res.copy()
    notes = res["notes"].fillna("")
    res["same_party"] = res["dem_share_2p"].isna()
    res["major_other"] = notes.str.contains("major non-D/R candidate")
    res["rcv"] = notes.str.contains("ranked-choice")
    res["contested"] = (res["uncontested"].fillna(0).astype(int) == 0) & ~res["same_party"] & ~res["major_other"]
    # trainable: a two-party share that means what it says (RCV first-round party sums do not)
    res["trainable"] = res["contested"] & ~res["rcv"]
    return res


def lean_table() -> pd.DataFrame:
    lean = pd.read_csv(CLEAN / "lean_hist.csv")
    lean["lean"] = lean["pres_prev_dem_2p"] - lean["national_pres_dem_2p"]
    lean["lean2"] = lean["pres_prev2_dem_2p"] - lean["national_pres_prev2_dem_2p"]
    return lean


def lines_unchanged(lean: pd.DataFrame) -> pd.DataFrame:
    """(year, race_id, unchanged): same House district as two years earlier, judged by a shared presidential
    result on both years' lines agreeing within 0.3 points."""
    h = lean[lean["chamber"] == "House"]
    vals = {}
    for _, r in h.iterrows():
        d = {}
        if pd.notna(r["pres_prev_dem_2p"]):
            d[int(r["pres_prev_year"])] = r["pres_prev_dem_2p"]
        if pd.notna(r["pres_prev2_dem_2p"]) and pd.notna(r["pres_prev2_year"]):
            d[int(r["pres_prev2_year"])] = r["pres_prev2_dem_2p"]
        vals[(int(r["year"]), r["race_id"])] = d
    rows = []
    for (year, rid), d in vals.items():
        prev = vals.get((year - 2, rid))
        same = False
        if prev:
            common = set(d) & set(prev)
            same = bool(common) and all(abs(d[k] - prev[k]) < 0.003 for k in common)
        rows.append((year, rid, same))
    return pd.DataFrame(rows, columns=["year", "race_id", "unchanged"])


def nominee_fec(res: pd.DataFrame) -> pd.DataFrame:
    """Match each nominee to FEC by (year, race, party bucket, surname). Returns receipts and incumbency."""
    fec = load_fec_hist()
    best = (fec.sort_values("receipts", ascending=False)
            .drop_duplicates(["year", "race_key", "party_bucket", "ln"]))
    idx = best.set_index(["year", "race_key", "party_bucket", "ln"])
    top1 = best.drop_duplicates(["year", "race_key", "party_bucket"])
    rank1 = {(y, k, p): row for y, k, p, row in zip(top1["year"], top1["race_key"], top1["party_bucket"],
                                                    [r for _, r in top1.iterrows()])}
    out = []
    for _, r in res.iterrows():
        key = r["race_id"].replace("-SEN-S", "-SEN")
        rec = {"year": r["year"], "race_id": r["race_id"]}
        for side, party in (("dem", "DEM"), ("rep", "REP")):
            name = r[f"{side}_cand"]
            k = (r["year"], key, party, last_name(name)) if isinstance(name, str) else None
            hit = idx.loc[k] if k is not None and k in idx.index else None
            if hit is None and k is not None:  # MEDSL misspellings: fall back to that party's top fundraiser
                top = rank1.get((r["year"], key, party))
                hit = top
            rec[f"{side}_receipts"] = float(hit["receipts"]) if hit is not None else np.nan
            rec[f"{side}_inc"] = int(hit is not None and hit["ici"] == "I")
            rec[f"{side}_fec_match"] = int(hit is not None)
        out.append(rec)
    return pd.DataFrame(out)


def prev_house(res: pd.DataFrame, unchanged: pd.DataFrame) -> pd.DataFrame:
    """Previous-cycle House result in the same district (lines unchanged), relative to that year's national vote."""
    nat = pd.read_csv(CLEAN / "national_hist.csv")
    nat_share = dict(zip(nat["year"], 0.5 + nat["house_margin_dr"] / 200.0))
    house = pd.read_csv(CLEAN / "results_house.csv")
    house = house[house["special"].fillna(0).eq(0)]
    prev = house[["year", "race_id", "dem_share_2p", "uncontested", "winner_name"]].copy()
    prev["year"] = prev["year"] + 2
    prev = prev.rename(columns={"dem_share_2p": "prev_share", "uncontested": "prev_unc", "winner_name": "prev_winner"})
    prev["prev_rel"] = prev["prev_share"] - (prev["year"] - 2).map(nat_share)
    m = unchanged.merge(prev, on=["year", "race_id"], how="left")
    ok = m["unchanged"] & m["prev_unc"].fillna(1).eq(0) & m["prev_share"].notna()
    m["prev_rel"] = np.where(ok, m["prev_rel"], 0.0)
    m["has_prev"] = ok.astype(int)
    return m[["year", "race_id", "prev_rel", "has_prev", "prev_winner", "unchanged"]]


def national() -> pd.DataFrame:
    nat = pd.read_csv(CLEAN / "national_hist.csv")
    nat["env_generic"] = nat["generic_dem"] / (nat["generic_dem"] + nat["generic_rep"]) - 0.5
    nat["env_actual"] = nat["house_dem_share"] / (nat["house_dem_share"] + nat["house_rep_share"]) - 0.5
    nat["pres_d"] = (nat["president_party"] == "D").astype(int)
    nat["midterm"] = (nat["cycle_type"] == "midterm").astype(int)
    nat["out_d"] = 1 - 2 * nat["pres_d"]   # +1 when the Democrats are the out-party (Republican president)
    return nat[["year", "env_generic", "env_actual", "net_approval", "pres_d", "midterm", "out_d", "election_date"]]


# Other outlets' ratings (Wikipedia "other" columns), same -3..+3 scale. 538 is excluded: it is the benchmark.
OTHER_EXCLUDE = {"other_fte"}


def other_dev(rat: pd.DataFrame) -> pd.Series:
    """Mean of the available other outlets' ratings minus the core-3 mean; 0 where no other outlet rated it."""
    cols = [c for c in rat.columns if c.startswith("other_") and c not in OTHER_EXCLUDE]
    others = rat[cols].apply(pd.to_numeric, errors="coerce")
    return (others.mean(axis=1) - rat["mean_rating"]).fillna(0.0)


def build() -> pd.DataFrame:
    res = load_results()
    lean = lean_table()
    rat = pd.read_csv(CLEAN / "ratings_hist.csv")
    rat = rat.assign(other_dev=other_dev(rat))[["year", "race_id", "mean_rating", "pvi_wiki", "cook", "sabato",
                                                "inside", "other_dev"]]
    df = res.merge(rat, on=["year", "race_id"], how="left")
    # Lean: House by (year, race_id); Senate by state-year.
    df["lean_key"] = np.where(df["chamber"] == "Senate", df["state"] + "-SEN", df["race_id"])
    df = df.merge(lean[["year", "race_id", "lean", "lean2"]].rename(columns={"race_id": "lean_key"}),
                  on=["year", "lean_key"], how="left")
    fallback = -df["pvi_wiki"] / 100.0
    df["lean_filled"] = df["lean"].isna().astype(int)
    df["lean"] = df["lean"].fillna(fallback)
    df = df.merge(nominee_fec(res), on=["year", "race_id"], how="left")
    unchanged = lines_unchanged(lean)
    df = df.merge(prev_house(res, unchanged), on=["year", "race_id"], how="left")
    df["prev_rel"] = df["prev_rel"].fillna(0.0)
    df["has_prev"] = df["has_prev"].fillna(0).astype(int)
    # Incumbency fallback: nominee is last cycle's winner in an unchanged district.
    for side in ("dem", "rep"):
        same = df["prev_winner"].notna() & df["unchanged"].fillna(False).astype(bool) & (
            df[f"{side}_cand"].map(lambda x: last_name(x) if isinstance(x, str) else "x")
            == df["prev_winner"].map(lambda x: last_name(x) if isinstance(x, str) else "y"))
        df[f"{side}_inc"] = (df[f"{side}_inc"].fillna(0).astype(bool) | same).astype(int)
    df["inc"] = df["dem_inc"] - df["rep_inc"]
    df["money"] = money_feature(df["dem_receipts"], df["rep_receipts"])
    df = df.merge(national(), on="year", how="left")
    df["rating"] = df["mean_rating"].fillna(np.sign(-df["lean"]) * 3)
    df["other_dev"] = df["other_dev"].fillna(0.0)
    df["senate"] = (df["chamber"] == "Senate").astype(int)
    return df


if __name__ == "__main__":
    d = build()
    print(d.shape)
    print(d.groupby(["year", "chamber"]).agg(n=("race_id", "size"), contested=("contested", "sum"),
                                              lean_na=("lean", lambda s: s.isna().sum()),
                                              rat_na=("mean_rating", lambda s: s.isna().sum()),
                                              dmatch=("dem_fec_match", "mean"), rmatch=("rep_fec_match", "mean"),
                                              inc_d=("dem_inc", "sum"), inc_r=("rep_inc", "sum"),
                                              prev=("has_prev", "sum")))
