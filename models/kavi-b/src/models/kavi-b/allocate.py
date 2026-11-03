"""Turn a race-level P(D side wins) into per-candidate q.

Within a side, candidates of the same party as that side's principal share the side's probability through a
within-party model fitted on past same-party top-two generals (CA/WA): P(A beats B) = sigmoid(a*dInc + b*dLogReceipts).
With several same-party candidates that becomes a softmax of a*inc + b*log(receipts + FEC_SMOOTH). The principal
gets no extra weight (in jungle races the 'principal' from the results table would leak the outcome).
Minor candidates (other parties, 'Others' rows) get MINOR_Q each, taken from their side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import FEC_SMOOTH, MINOR_Q, OWN, last_name, name_tokens

_PRIMARIES: pd.DataFrame | None = None


def primaries() -> pd.DataFrame:
    """CA/WA top-two primary results (data/kavi-b/primaries_top_two.csv, built by fetch_primaries.py)."""
    global _PRIMARIES
    if _PRIMARIES is None:
        p = pd.read_csv(OWN / "primaries_top_two.csv")
        p["last"] = p["candidate"].map(last_name)
        _PRIMARIES = p
    return _PRIMARIES


def primary_pct(year: int, race_id: str, name: str) -> float:
    p = primaries()
    hit = p[(p["year"] == year) & (p["race_id"] == race_id) & (p["last"] == last_name(name))]
    return float(hit["pct"].max()) if len(hit) else np.nan


def within_party_data(hist: pd.DataFrame, fec: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Past same-party generals: top-2 fundraisers of the winner's party, y = 1 if the top fundraiser won."""
    rows = []
    sp = hist[(hist["same_party"] == 1) & hist["year"].isin(years)]
    for r in sp.itertuples():
        bucket = "DEM" if r.winner_party == "DEM" else "REP"
        f = fec[(fec["year"] == r.year) & (fec["race_id"] == r.race_id.replace("-SEN-S", "-SEN"))
                & (fec["party_bucket"] == bucket)].sort_values("receipts", ascending=False).head(2)
        if len(f) < 2:
            continue
        wl = last_name(r.winner_name)
        lasts = list(f["last"])
        if wl not in lasts:
            continue
        a, b = f.iloc[0], f.iloc[1]
        pa, pb = (primary_pct(r.year, r.race_id, x["cand_name"]) for x in (a, b))
        rows.append({"year": r.year, "race_id": r.race_id, "y": int(lasts[0] == wl),
                     "d_inc": int(a["ici"] == "I") - int(b["ici"] == "I"),
                     "d_lr": np.log(a["receipts"] + FEC_SMOOTH) - np.log(b["receipts"] + FEC_SMOOTH),
                     "d_prim": (pa - pb) / 10.0})
    return pd.DataFrame(rows)


def _sym_logit(data: pd.DataFrame, cols: list[str]) -> dict:
    mirror = data.assign(y=1 - data["y"], **{c: -data[c] for c in cols})
    d = pd.concat([data, mirror], ignore_index=True)
    res = sm.GLM(d["y"], d[cols].astype(float), family=sm.families.Binomial()).fit_regularized(alpha=0.01, L1_wt=0.0)
    return {c: float(res.params[c]) for c in cols}


def fit_within_party(data: pd.DataFrame, use_money: bool = True) -> dict:
    """Symmetric logistics without intercept (mirrored pairs), mild ridge:
    'money' = dInc + dLogReceipts (used when no primary result is known),
    'prim'  = dInc + dLogReceipts + dPrimaryPct/10 (CA/WA House races with a parsed top-two primary)."""
    out = {"n": len(data), "money": use_money}
    if not use_money:   # v2: test year without a pre-election FEC cut -> incumbency + primary share only
        m = _sym_logit(data, ["d_inc"])
        dp = data.dropna(subset=["d_prim"])
        p = _sym_logit(dp, ["d_inc", "d_prim"]) if len(dp) >= 10 else {"d_inc": m["d_inc"], "d_prim": 0.0}
        out.update({"inc": m["d_inc"], "lr": 0.0, "p_inc": p["d_inc"], "p_lr": 0.0, "p_prim": p["d_prim"],
                    "n_prim": len(dp)})
        return out
    if len(data) < 10:
        out.update({"inc": 1.5, "lr": 0.8, "p_inc": 0.5, "p_lr": 0.3, "p_prim": 1.0, "n_prim": 0})
        return out
    m = _sym_logit(data, ["d_inc", "d_lr"])
    out.update({"inc": m["d_inc"], "lr": m["d_lr"]})
    dp = data.dropna(subset=["d_prim"])
    out["n_prim"] = len(dp)
    if len(dp) >= 10:
        p = _sym_logit(dp, ["d_inc", "d_lr", "d_prim"])
        out.update({"p_inc": p["d_inc"], "p_lr": p["d_lr"], "p_prim": p["d_prim"]})
    else:
        out.update({"p_inc": out["inc"], "p_lr": out["lr"], "p_prim": 0.0})
    return out


def split_side(cands: pd.DataFrame, side_q: float, principal_party: str, wp: dict) -> np.ndarray:
    """cands: rows of one side with columns party, receipts, is_inc, is_minor. Returns q per row summing to side_q."""
    q = np.zeros(len(cands))
    minor = cands["is_minor"].to_numpy() | (cands["party"].to_numpy() != principal_party)
    n_minor = int(minor.sum())
    if n_minor == len(cands):          # nobody of the principal's party listed: treat the richest as principal
        minor = np.ones(len(cands), bool)
        minor[int(np.argmax(cands["receipts"].to_numpy()))] = False
        n_minor -= 1
    # v2: minor / withdrawn candidates take at most 10% of a small side (ID/SD/MT-SEN withdrawn Democrats were
    # taking a quarter of the independent's mass in v1)
    eps = min(MINOR_Q, 0.1 * side_q / max(n_minor, 1))
    q[minor] = eps
    rest = side_q - eps * n_minor
    lr = np.log(cands["receipts"].fillna(0).to_numpy(float) + FEC_SMOOTH)
    inc = cands["is_inc"].to_numpy(float)
    prim = cands["prim_pct"].to_numpy(float) if "prim_pct" in cands else np.full(len(cands), np.nan)
    if (~minor).sum() > 1 and not np.isnan(prim[~minor]).any():
        score = (wp["p_inc"] * inc + wp["p_lr"] * lr + wp["p_prim"] * prim / 10.0)[~minor]
    else:
        score = (wp["inc"] * inc + wp["lr"] * lr)[~minor]
    w = np.exp(score - score.max())
    q[~minor] = rest * w / w.sum()
    return q


def allocate_race(cands: pd.DataFrame, q_dside: float, d_party: str, same_party: str | None, wp: dict) -> np.ndarray:
    """cands for one race; columns party, receipts, is_inc, is_minor, side ('D'/'R').
    same_party: 'DEM'/'REP' for a same-party top-two race (whole race is one side)."""
    q = np.zeros(len(cands))
    if same_party:
        return split_side(cands, 1.0, same_party, wp)
    for side, sq, party in (("D", q_dside, d_party), ("R", 1 - q_dside, "REP")):
        m = (cands["side"] == side).to_numpy()
        if m.any():
            q[m] = split_side(cands[m], sq, party, wp)
    if q.sum() <= 0:
        return np.full(len(cands), 1.0 / len(cands))
    return q / q.sum()


def fec_lookup(fec: pd.DataFrame, year: int, race_id: str, name: str, party: str) -> tuple[float, int]:
    rows = fec[(fec["year"] == year) & (fec["race_id"] == race_id.replace("-SEN-S", "-SEN"))]
    ln = last_name(name)
    hit = rows[rows["last"] == ln]
    if len(hit) > 1 and (hit["party_bucket"] == party).any():
        hit = hit[hit["party_bucket"] == party]
    if len(hit) > 1:   # same surname twice (WA-09 2018: Adam Smith vs Sarah Smith): use the first name
        first = name_tokens(name)[0] if name_tokens(name) else ""
        same_first = hit["cand_name"].map(lambda n: first in name_tokens(n)[:-1])
        if same_first.any():
            hit = hit[same_first]
    if hit.empty:
        return 0.0, 0
    hit = hit.sort_values("receipts", ascending=False).iloc[0]
    return float(hit["receipts"]), int(hit["ici"] == "I")
