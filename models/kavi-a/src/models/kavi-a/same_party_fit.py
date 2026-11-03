"""v3: fitted same-party top-two model (CA / WA House D v D and R v R generals).

Borrowed data: kavi-e's top-two primary table (data/kavi-a/borrowed/primaries_top_two_kavi-e.csv, copied from
origin/kavi/kavi-e-1101 9efb6c7, data/kavi-e/primaries_top_two.csv: CA 2012-2026 and WA 2012-2024 House primary
votes from revision-pinned Wikipedia election-results tables; vote counts, not link existence, so not the P016
trap). kavi-e's finding: the primary leader won 86% of 50 same-party generals.

My model (my own fit, not kavi-e's): for the two finalists A, B of a same-party general,
  logit P(A wins) = B_SHARE * 10 (s_A - 0.5) + B_INC (inc_A - inc_B)
s_A = A's share of the two finalists' primary votes, inc = FEC incumbent flag. No intercept (the labels A/B are
arbitrary: both orientations enter at weight 0.5), mild L2 (C = 1). Fitted on same-party generals of the training
years only (2018 backtest: 2012-2016, 25 races; 2022: 2012-2020; 2026: 2012-2024). Winners come from MEDSL
results (winner_name).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import CLEAN, ROOT, last_name, tokens

PRIMARY_FILE = ROOT / "data" / "kavi-a" / "borrowed" / "primaries_top_two_kavi-e.csv"
C_REG = 1.0
SHARE_UNIT = 10.0     # coefficient per 10 points of two-finalist primary share


@lru_cache(maxsize=None)
def finalists() -> pd.DataFrame:
    """Top two of each primary that are of the same party: year, race_id, cand_a, cand_b, s_a, party."""
    p = pd.read_csv(PRIMARY_FILE).drop_duplicates(["year", "race_id", "candidate"])
    p = p[~p["candidate"].str.contains("write-in", case=False)]
    top = p.sort_values("votes", ascending=False).groupby(["year", "race_id"]).head(2)
    rows = []
    for (yr, rid), g in top.groupby(["year", "race_id"]):
        if len(g) < 2 or g["party"].nunique() != 1 or g["party"].iloc[0] not in ("DEM", "REP"):
            continue
        a, b = g.iloc[0], g.iloc[1]
        rows.append({"year": int(yr), "race_id": rid, "cand_a": a["candidate"], "cand_b": b["candidate"],
                     "s_a": a["votes"] / (a["votes"] + b["votes"]), "party": a["party"]})
    return pd.DataFrame(rows)


def primary_share(year: int, race_id: str, name_a: str, name_b: str) -> float | None:
    """A's two-finalist primary share, or None if the race/candidates are not in the table."""
    f = finalists()
    f = f[(f["year"] == year) & (f["race_id"] == race_id)]
    if f.empty:
        return None
    r = f.iloc[0]
    if same_person(name_a, r["cand_a"]) and same_person(name_b, r["cand_b"]):
        return float(r["s_a"])
    if same_person(name_a, r["cand_b"]) and same_person(name_b, r["cand_a"]):
        return float(1 - r["s_a"])
    return None


def same_person(x: str, y: str) -> bool:
    """Surname match, allowing hyphenated / compound surnames (Kamlager vs Kamlager-Dove)."""
    return last_name(x) in tokens(y) or last_name(y) in tokens(x)


def training_rows(train_years: tuple) -> pd.DataFrame:
    from features import fec_lookup, match_fec_full
    f = finalists()
    f = f[f["year"].isin(train_years)]
    res = pd.read_csv(CLEAN / "results_house.csv")[["year", "race_id", "winner_name"]]
    f = f.merge(res, on=["year", "race_id"], how="inner")
    fec = fec_lookup()
    rows = []
    for r in f.itertuples():
        win = last_name(str(r.winner_name))
        if win not in (last_name(r.cand_a), last_name(r.cand_b)) or last_name(r.cand_a) == last_name(r.cand_b):
            continue
        inc = [int(match_fec_full(fec, r.year, r.race_id, n, r.party)[0] == "I") for n in (r.cand_a, r.cand_b)]
        rows.append({"year": r.year, "race_id": r.race_id, "x_share": SHARE_UNIT * (r.s_a - 0.5),
                     "x_inc": inc[0] - inc[1], "a_won": int(win == last_name(r.cand_a))})
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def fit(train_years: tuple) -> dict:
    t = training_rows(train_years)
    X = t[["x_share", "x_inc"]].to_numpy(dtype=float)
    y = t["a_won"].to_numpy()
    Xs, ys = np.vstack([X, -X]), np.concatenate([y, 1 - y])  # both orientations, half weight each
    m = LogisticRegression(fit_intercept=False, C=C_REG, max_iter=1000).fit(Xs, ys, sample_weight=np.full(len(ys), 0.5))
    b_share, b_inc = (float(c) for c in m.coef_[0])
    return {"b_share": b_share, "b_inc": b_inc, "n": int(len(t)), "leader_won": float(
        np.mean(np.where(t["x_share"] > 0, t["a_won"], 1 - t["a_won"]))), "years": list(train_years)}


def prob_a(params: dict, s_a: float, inc_a: int, inc_b: int) -> float:
    z = params["b_share"] * SHARE_UNIT * (s_a - 0.5) + params["b_inc"] * (inc_a - inc_b)
    return float(1 / (1 + np.exp(-z)))


if __name__ == "__main__":
    for yrs in (tuple(range(2012, 2017, 2)), tuple(range(2012, 2021, 2)), tuple(range(2012, 2025, 2))):
        print(fit(yrs))
