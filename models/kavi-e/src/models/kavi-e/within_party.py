"""v2: within-party split of same-party top-two generals (CA/WA), fitted on history instead of v1's judgment weights.

Data: June top-two primary results 2012-2026 parsed from Wikipedia by kavi-b (copied, unchanged, to
data/kavi-e/primaries_top_two.csv; built by kavi-b's fetch_primaries.py). Primaries are pre-general information.
Training rows: every same-party top-two House general in CA/WA 2012-2024 (results_house note "same-party top-two")
whose two finalists are found in the primary file; outcome = which finalist won (surname match to the winner).
Model (no intercept, both orderings of each pair so it is symmetric):
    logit P(A beats B) = b_pct * (primary % A - primary % B) / 10 + b_inc * (inc_A - inc_B)
fitted by maximum likelihood with a weak ridge (0.1) on years < target year only.
Money variant (+ b_lr * ln((receipts_A + 1e4) / (receipts_B + 1e4)), clipped +-3): the TRAINING receipts are FEC
full-cycle totals (they run past election day; kavi-b measured the pre-election vs full-cycle log ratio gap at
0.07 on average in 2018, so the coefficient leak is small). It is applied only where a PRE-election cut of the
target year exists: 2018 (Wayback capture of weball18.zip from 2018-11-07 found by kavi-b, copied to
data/kavi-e/fec_2018_preelection.csv) and 2026 (fec_2026, through 2026-10-14). The 2022 backtest uses the
no-money model. For 2026 the late same-party polls are added on top (POLL_WITHIN, judgment) and the Nov 1 dossiers
adjust the result (dossier_adjust.py).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import ROOT, last_name, read, same_person

PRIMARIES = ROOT / "data" / "kavi-e" / "primaries_top_two.csv"
FEC_2018_PRE = ROOT / "data" / "kavi-e" / "fec_2018_preelection.csv"
RIDGE = 0.1
LR_CLIP = 3.0


def log_ratio(a: float, b: float) -> float:
    return float(np.clip(np.log((a + 1e4) / (b + 1e4)), -LR_CLIP, LR_CLIP))


@lru_cache(maxsize=None)
def _fec_hist() -> pd.DataFrame:
    f = read("fec_hist.csv")
    return f.assign(sn=f["cand_name"].map(last_name))


def hist_receipts(year: int, race_id: str, name: str) -> float:
    f = _fec_hist()
    sub = f[(f["year"] == year) & (f["race_id"] == race_id) & (f["sn"] == last_name(name))]
    return float(sub["receipts"].fillna(0).max()) if not sub.empty else 0.0


@lru_cache(maxsize=None)
def _fec_2018_pre() -> pd.DataFrame:
    f = pd.read_csv(FEC_2018_PRE)
    f["race_id"] = f["race_id_raw"].str.replace("-00", "-AL", regex=False)
    return f.assign(sn=f["cand_name"].map(last_name))


def pre_receipts_2018(race_id: str, name: str) -> float:
    f = _fec_2018_pre()
    sub = f[(f["race_id"] == race_id) & (f["sn"] == last_name(name))]
    return float(sub["receipts_pre"].fillna(0).max()) if not sub.empty else 0.0


def primaries() -> pd.DataFrame:
    p = pd.read_csv(PRIMARIES)
    p["sn"] = p["candidate"].map(last_name)
    return p


def primary_pct(year: int, race_id: str, name: str, pr: pd.DataFrame | None = None) -> float:
    pr = primaries() if pr is None else pr
    sub = pr[(pr["year"] == year) & (pr["race_id"] == race_id)]
    hit = sub[sub["candidate"].map(lambda n: same_person(n, name))]
    if hit.empty:
        hit = sub[sub["sn"] == last_name(name)]
    return float(hit["pct"].iloc[0]) if not hit.empty else np.nan


def training_rows() -> pd.DataFrame:
    res = read("results_house.csv")
    sp = res[res["notes"].astype(str).str.contains("same-party top-two") & res["state"].isin(["CA", "WA"])
             & (res["year"] >= 2012)]
    pr = primaries()
    rows = []
    for _, r in sp.iterrows():
        sub = pr[(pr["year"] == r["year"]) & (pr["race_id"] == r["race_id"])].sort_values("votes", ascending=False)
        if len(sub) < 2:
            continue
        a, b = sub.iloc[0], sub.iloc[1]
        if a["party"] != b["party"]:
            continue
        prev = res[(res["year"] == r["year"] - 2) & (res["state"] == r["state"])]["winner_name"].dropna()
        inc = [int(any(same_person(c, w) for w in prev)) for c in (a["candidate"], b["candidate"])]
        a_won = same_person(a["candidate"], r["winner_name"]) or a["sn"] == last_name(r["winner_name"])
        d_lr = log_ratio(hist_receipts(r["year"], r["race_id"], a["candidate"]),
                         hist_receipts(r["year"], r["race_id"], b["candidate"]))
        rows.append({"year": r["year"], "race_id": r["race_id"], "a": a["candidate"], "b": b["candidate"],
                     "d_pct": (a["pct"] - b["pct"]) / 10, "d_inc": inc[0] - inc[1], "d_lr": d_lr,
                     "a_won": int(a_won)})
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def _rows() -> pd.DataFrame:
    return training_rows()


@lru_cache(maxsize=None)
def fit(target_year: int, money: bool = False) -> tuple[float, ...]:
    """(b_pct, b_inc[, b_lr], n_train)"""
    t = _rows()
    t = t[t["year"] < target_year]
    cols = ["d_pct", "d_inc"] + (["d_lr"] if money else [])
    X = np.vstack([t[cols].to_numpy(float), -t[cols].to_numpy(float)])
    y = np.concatenate([t["a_won"].to_numpy(float), 1 - t["a_won"].to_numpy(float)])

    def nll(b: np.ndarray) -> float:
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z) + RIDGE * np.sum(b ** 2))

    b = minimize(nll, np.zeros(len(cols)), method="BFGS").x
    return tuple(float(v) for v in b) + (int(len(t)),)


def scores(year: int, race_id: str, cands: pd.DataFrame, receipts: pd.Series | None = None) -> pd.Series:
    """Within-party logit score per candidate (NaN when a primary share is missing). receipts: PRE-election
    receipts per candidate (only 2018 snapshot / 2026); None = no-money model."""
    pct = cands["candidate"].map(lambda n: primary_pct(year, race_id, n))
    inc = cands["is_incumbent"].fillna(0).astype(float)
    if receipts is None:
        b_pct, b_inc, _ = fit(year)
        return b_pct * pct / 10 + b_inc * inc
    b_pct, b_inc, b_lr, _ = fit(year, True)
    lr = np.log(receipts.astype(float).fillna(0) + 1e4)
    lr = np.clip(lr - lr.mean(), -LR_CLIP / 2, LR_CLIP / 2)
    return b_pct * pct / 10 + b_inc * inc + b_lr * lr


if __name__ == "__main__":
    t = training_rows()
    print(t.groupby("year").size().to_dict(), "favourite-by-primary won", round(float(((t.d_pct > 0) == t.a_won).mean()), 3))
    for y in (2018, 2022, 2026):
        print(y, [round(v, 3) for v in fit(y)], "money", [round(v, 3) for v in fit(y, True)])
    print(_fec_2018_pre()["race_id"].drop_duplicates().head(8).tolist())
