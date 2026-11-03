"""Candidate-level model for contests the party model cannot see: same-party top-two finals (CA/WA), and the split
of one party's probability among several of its candidates (Louisiana jungle, AK RCV with two Republicans).

Strength of candidate i: s_i = a * incumbent_i + b * log10(receipts_i + 1e4) + c * (primary_i - pair mean)
(+ hand-set 0.08 x poll points above the pair mean, 2026 only). P(i wins | contest of that party) = softmax(s).
(a, b, c) fit by ridge maximum likelihood on 2012-2024 same-party finals (MIT results; FEC receipts and I/C/O flags;
top-two primary shares from kavi-b's Wikipedia parse, data/extra/primaries_top_two.csv; a final without a parsed
primary contributes 0 to c), using only years before the predicted one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import CACHE, CLEAN, ROOT, fec_surname_tokens, name_matches, prefer_given_name, tokens

RAW_MIT = ROOT / "data" / "raw" / "mit"
L2 = 0.5  # ridge on (a, b, c): only ~50 training contests (c is per primary point, so scaled by 10 in the ridge)
PRIMARIES = Path(__file__).resolve().parents[3] / "data" / "extra" / "primaries_top_two.csv"  # borrowed from kavi-b
POLL_WEIGHT = 0.08  # logit per poll point above the pair mean


def _finalists() -> pd.DataFrame:
    res = pd.concat([pd.read_csv(CLEAN / "results_house.csv"), pd.read_csv(CLEAN / "results_senate.csv")])
    sp = res[res["dem_share_2p"].isna() & (res["year"] >= 2012)][["year", "race_id", "winner_name"]]
    h = pd.read_csv(RAW_MIT / "1976-2024-house.tab", low_memory=False)
    h = h[(h["stage"] == "GEN") & (h["writein"] == False)]  # noqa: E712
    h["race_id"] = h["state_po"] + "-" + h["district"].astype(int).map(lambda d: "AL" if d == 0 else f"{d:02d}")
    s = pd.read_csv(RAW_MIT / "1976-2024-senate-state.csv", low_memory=False)
    s = s[(s["stage"].str.upper() == "GEN") & (s["writein"] == False)]  # noqa: E712
    s["race_id"] = s["state_po"] + "-SEN"
    raw = pd.concat([h[["year", "race_id", "candidate", "candidatevotes"]],
                     s[["year", "race_id", "candidate", "candidatevotes"]]])
    x = raw.merge(sp, on=["year", "race_id"])
    g = x.groupby(["year", "race_id", "candidate", "winner_name"], as_index=False)["candidatevotes"].sum()
    g = g.sort_values("candidatevotes", ascending=False).groupby(["year", "race_id"]).head(2)
    g["won"] = (g["candidate"] == g["winner_name"]).astype(int)
    return g[g.groupby(["year", "race_id"])["won"].transform("sum") == 1]


def fec_lookup(fec: pd.DataFrame, year: int, race_id: str, name: str) -> tuple[float, int]:
    g = fec[(fec["year"] == year) & (fec["race_id"] == race_id)]
    hit = prefer_given_name(g[g["_tok"].map(lambda t: name_matches(t, name))], name)
    if hit.empty:
        return np.nan, 0
    row = hit.sort_values("receipts", ascending=False, na_position="last").iloc[0]
    return float(row["receipts"]) if pd.notna(row["receipts"]) else np.nan, int(row["ici"] == "I")


def training_rows() -> pd.DataFrame:
    path = CACHE / "sameparty_train.csv"
    if path.exists():
        return pd.read_csv(path)
    fec = pd.read_csv(CLEAN / "fec_hist.csv")
    fec["_tok"] = fec["cand_name"].map(fec_surname_tokens)
    f = _finalists()
    vals = [fec_lookup(fec, r.year, r.race_id, r.candidate) for r in f.itertuples()]
    f["receipts"] = [v[0] for v in vals]
    f["inc"] = [v[1] for v in vals]
    f["primary"] = [primary_pct(r.year, r.race_id, r.candidate) for r in f.itertuples()]
    CACHE.mkdir(parents=True, exist_ok=True)
    f.to_csv(path, index=False)
    return f


_PRIM = None


def primary_pct(year: int, race_id: str, name: str) -> float:
    """Top-two primary share (points) of a candidate, NaN when the primary was not parsed."""
    global _PRIM
    if _PRIM is None:
        _PRIM = pd.read_csv(PRIMARIES) if PRIMARIES.exists() else pd.DataFrame(columns=["year", "race_id"])
    g = _PRIM[(_PRIM["year"] == year) & (_PRIM["race_id"] == race_id)]
    hit = g[g["candidate"].map(lambda c: name_matches(tokens(c), name))]
    return float(hit["pct"].iloc[0]) if len(hit) else np.nan


def strength(inc, receipts, params, primary=None) -> np.ndarray:
    a, b = params[0], params[1]
    s = a * np.asarray(inc, float) + b * np.log10(np.nan_to_num(np.asarray(receipts, float), nan=0.0) + 1e4)
    if len(params) > 2 and primary is not None:
        s = s + params[2] * _dev(primary)
    return s


def fit(before_year: int, use_money: bool = True, use_primary: bool = True) -> np.ndarray:
    """(a, b, c) by ridge MLE on finals before `before_year`. use_money=False fixes b = 0, for backtests whose
    test-year receipts are not available pre-election (2022). use_primary=False fits without primary shares (c = 0),
    for contests with no top-two primary (LA jungle, AK, CA-SEN): there incumbency must carry what the primary would."""
    t = training_rows()
    t = t[t["year"] < before_year]
    if not use_primary:
        t = t.assign(primary=np.nan)
    groups = [g for _, g in t.groupby(["year", "race_id"])]

    def obj(x):
        p = np.array([x[0], x[1] if use_money else 0.0, x[-1]])
        return _nll(groups, p) + L2 * (p[0] ** 2 + p[1] ** 2 + (10 * p[2]) ** 2)

    x0 = np.array([1.0, 1.0, 0.03]) if use_money else np.array([1.0, 0.03])
    x = minimize(obj, x0=x0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 4000}).x
    return np.array([x[0], x[1] if use_money else 0.0, x[-1]])


def _nll(groups, p) -> float:
    total = 0.0
    for g in groups:
        s = strength(g["inc"], g["receipts"], p, g["primary"])
        s = s - s.max()
        total -= (s[g["won"].to_numpy() == 1] - np.log(np.exp(s).sum())).sum()
    return total


def _dev(x) -> np.ndarray:
    x = np.asarray(x, float)
    if not np.isfinite(x).any():
        return np.zeros_like(x)
    return np.nan_to_num(x - np.nanmean(x), nan=0.0)


def split(inc, receipts, params, poll_share=None, primary_share=None, poll_weight: float = POLL_WEIGHT) -> np.ndarray:
    """Softmax probabilities over candidates of one contest. Primary shares use the fitted weight params[2]; poll
    shares (points) a hand-set weight (no historical same-party poll data to fit it on)."""
    s = strength(inc, receipts, params, primary_share)
    if poll_share is not None:
        s = s + poll_weight * _dev(poll_share)
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


if __name__ == "__main__":
    t = training_rows()
    print(t.groupby("year").size().to_dict(), "inc-won rate", t[t.inc == 1]["won"].mean(),
          "primary parsed", int(t["primary"].notna().sum()), "of", len(t))
    for y in (2018, 2022, 2026):
        print(y, "money", fit(y), "no money", fit(y, use_money=False), "no primary", fit(y, use_primary=False),
              "neither", fit(y, use_money=False, use_primary=False))


def pick_params(both: tuple, primary_share: dict | None) -> np.ndarray:
    """both = (params with primary, params without); use the first when the contest has a parsed primary."""
    has = primary_share is not None and any(np.isfinite(v) for v in primary_share.values())
    return both[0] if has else both[1]
