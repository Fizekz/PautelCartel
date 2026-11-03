"""Candidate-level helpers for races the two-party model cannot see directly.

- `split_within_party`: several candidates of one party (CA/WA top-two lockouts, Louisiana jungle ballots,
  Mississippi 2018 special): incumbency and money, or head-to-head polls when they exist.
- `two_way_poll_share`: poll average of candidate A's share of the A+B vote by surname, for independents and
  same-party finals, read from dem/rep/other columns of the poll files.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import ELECTION_DAY, POLL_HALFLIFE_DAYS, POLL_WINDOW_DAYS, last_name
from hist_features import load_fec_hist, pick_by_first_name

INC_LOGIT = 2.0          # incumbent vs non-incumbent of the same party, before money
MONEY_LOGIT = 0.8        # per log-unit of receipts ratio
MONEY_FLOOR = 50_000.0
# Years whose receipts may not be used (no pre-election FEC cut): the split then rests on incumbency alone.
NO_RECEIPT_YEARS: set[int] = set()
# v3: fitted primary-share model (same_party.SameParty) for two-candidate CA/WA same-party finals; set by the caller
# with a fit on the training years only. None = v2 rule (incumbency + money softmax).
SP_MODEL = None


def fec_lookup(year: int, race_id: str, name: str) -> tuple[float, int]:
    fec = load_fec_hist()
    key = race_id.replace("-SEN-S", "-SEN")
    hit = fec[(fec["year"] == year) & (fec["race_key"] == key) & (fec["ln"] == last_name(name))]
    hit = pick_by_first_name(hit, name)
    if hit.empty:
        return 0.0, 0
    top = hit.sort_values("receipts", ascending=False).iloc[0]
    receipts = 0.0 if year in NO_RECEIPT_YEARS else float(top["receipts"])
    return receipts, int(top["ici"] == "I")


def split_scores(receipts: list[float], inc: list[int]) -> np.ndarray:
    rec = np.asarray(receipts, float)
    logit = INC_LOGIT * np.asarray(inc, float) + MONEY_LOGIT * np.log((rec + MONEY_FLOOR) / (rec.max() + MONEY_FLOOR))
    e = np.exp(logit - logit.max())
    return e / e.sum()


def split_within_party(year: int, race_id: str, names: list[str]) -> np.ndarray:
    looked = [fec_lookup(year, race_id, n) for n in names]
    if SP_MODEL is not None and len(names) == 2:
        import same_party
        pa, pb = (same_party.primary_pct(year, race_id, n) for n in names)
        if not (np.isnan(pa) or np.isnan(pb)):
            p = SP_MODEL.prob(pa, pb, looked[0][1], looked[1][1])
            return np.array([p, 1 - p])
    return split_scores([r for r, _ in looked], [i for _, i in looked])


def _pct_by_surname(row: pd.Series) -> dict[str, float]:
    """Surname -> share for one poll row; other_cand only when it names a single candidate."""
    out = {}
    for c, v in (("dem_cand", "dem_pct"), ("rep_cand", "rep_pct")):
        if isinstance(row.get(c), str) and pd.notna(row.get(v)):
            out[last_name(row[c])] = float(row[v])
    oc = row.get("other_cand")
    if isinstance(oc, str) and ";" not in oc and pd.notna(row.get("other_pct")):
        out[last_name(re.sub(r"\(.*?\)", "", oc))] = float(row["other_pct"])
    return out


def two_way_poll_share(polls: pd.DataFrame, year: int, a: str, b: str) -> tuple[float, float]:
    """(share of A in A+B, effective n of polls) from rows that name both surnames; NaN if none."""
    eday = pd.Timestamp(ELECTION_DAY[year])
    la, lb = last_name(a), last_name(b)
    vals, wts = [], []
    for _, r in polls.iterrows():
        d = _pct_by_surname(r)
        if la in d and lb in d and d[la] + d[lb] > 0:
            days = (eday - pd.to_datetime(r["end_date"])).days
            if 0 <= days <= POLL_WINDOW_DAYS:
                vals.append(d[la] / (d[la] + d[lb]))
                wts.append(np.exp(-np.log(2) * days / POLL_HALFLIFE_DAYS))
    if not vals:
        return float("nan"), 0.0
    return float(np.average(vals, weights=wts)), float(np.sum(wts))
