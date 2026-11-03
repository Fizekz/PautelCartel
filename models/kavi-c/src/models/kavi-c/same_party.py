"""v3: within-party model for CA/WA top-two same-party generals, from the June primary.

logit P(A beats B) = b_pct (primary% A - primary% B) / 10 + b_inc (inc_A - inc_B), no intercept, both orderings of
every pair, L2 logistic, fitted on same-party generals of the training years only. Primary results are June
information, public months before the general. Data: kavi-b's parse of the CA/WA top-two primaries
(`data/kavi-c/primaries_top_two.csv`, copied from kavi-e's branch); the fitted-model idea is kavi-b's / kavi-e's,
this fit is my own. No money term (training-year receipts are full-cycle, and the primary already measures strength).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import ROOT, last_name, name_tokens

PRIMARIES = ROOT / "data" / "kavi-c" / "primaries_top_two.csv"
L2_C = 10.0


@lru_cache(maxsize=1)
def primaries() -> pd.DataFrame:
    p = pd.read_csv(PRIMARIES)
    p = p[~p["candidate"].str.contains("write-in", case=False, na=False)].copy()
    p["candidate"] = p["candidate"].str.replace(r"\(.*?\)", "", regex=True).str.strip()
    p["ln"] = p["candidate"].map(last_name)
    p["pct"] = pd.to_numeric(p["pct"], errors="coerce")
    return p


def primary_pct(year: int, race_id: str, name: str) -> float:
    p = primaries()
    hit = p[(p["year"] == year) & (p["race_id"] == race_id) & (p["ln"] == last_name(name))]
    if len(hit) > 1:  # same surname (WA-09 2018 Adam v Sarah Smith): first-name tie-break, else unknown
        first = name_tokens(name)[0] if name_tokens(name) else ""
        hit = hit[hit["candidate"].map(lambda c: (name_tokens(c) or [""])[0] == first)]
        if len(hit) != 1:
            return float("nan")
    return float(hit["pct"].iloc[0]) if len(hit) else float("nan")


@dataclass
class SameParty:
    b_pct: float
    b_inc: float
    n_pairs: int
    leader_win_rate: float

    def prob(self, pct_a: float, pct_b: float, inc_a: int, inc_b: int) -> float:
        z = self.b_pct * (pct_a - pct_b) / 10.0 + self.b_inc * (inc_a - inc_b)
        return float(1 / (1 + np.exp(-z)))


def training_pairs(years: list[int], results: pd.DataFrame, inc_lookup) -> pd.DataFrame:
    """One row per historical same-party general with both finalists in the primary file."""
    p = primaries()
    sp = results[results["year"].isin(years) & results["same_party"].astype(bool)
                 & results["race_id"].str[:2].isin(["CA", "WA"])]
    rows = []
    for _, r in sp.iterrows():
        g = p[(p["year"] == r["year"]) & (p["race_id"] == r["race_id"])].sort_values("votes", ascending=False)
        if len(g) < 2 or not isinstance(r["winner_name"], str):
            continue
        a, b = g.iloc[0], g.iloc[1]
        if a["party"] != b["party"]:
            continue
        wl = last_name(r["winner_name"])
        if wl not in (a["ln"], b["ln"]) or a["ln"] == b["ln"]:
            continue
        rows.append({"year": r["year"], "race_id": r["race_id"], "gap": (a["pct"] - b["pct"]) / 10.0,
                     "inc": inc_lookup(r["year"], r["race_id"], a["candidate"]) -
                     inc_lookup(r["year"], r["race_id"], b["candidate"]),
                     "leader_won": int(wl == a["ln"])})
    return pd.DataFrame(rows)


def fit(years: list[int], results: pd.DataFrame, inc_lookup) -> SameParty:
    d = training_pairs(years, results, inc_lookup)
    X = np.vstack([d[["gap", "inc"]].to_numpy(float), -d[["gap", "inc"]].to_numpy(float)])
    y = np.concatenate([d["leader_won"].to_numpy(), 1 - d["leader_won"].to_numpy()])
    lr = LogisticRegression(C=L2_C, fit_intercept=False).fit(X, y)
    return SameParty(b_pct=float(lr.coef_[0, 0]), b_inc=float(lr.coef_[0, 1]), n_pairs=len(d),
                     leader_win_rate=float(d["leader_won"].mean()) if len(d) else float("nan"))
