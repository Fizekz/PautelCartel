"""Turn a race-level P(D side wins) into per-candidate q, for a benchmark / Kalshi candidate list.

Rules (applied per race):
- The D side is the Democrat, or the designated major independent (MAJOR_IND) when there is one; the R side is the
  Republican(s). A race with only one of the two sides gets SOLO_P for that side.
- Two or more candidates on one side (top-two same party, Louisiana jungle) share that side's probability by a
  softmax of INC_WEIGHT * is_incumbent + POLL_WEIGHT * poll lead (when given).
- Every other candidate gets MINOR_Q; q is renormalised to sum to 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import last_name

MINOR_Q = 0.001
SOLO_P = 0.998
INC_WEIGHT = 2.5
# Major independents that take the D side (pre-election knowledge: the main non-R challenger / incumbent).
MAJOR_IND = {
    2018: {"VT-SEN": "Sanders", "ME-SEN": "King"},
    2022: {"UT-SEN": "McMullin"},
    2026: {"NE-SEN": "Osborn", "ID-SEN": "Achilles", "SD-SEN": "Bengs", "MT-SEN": "Bodnar", "AK-AL": "Hill"},
}
# Races where an independent takes the R side (CA-06 2026: Kiley, ex-R incumbent, vs Pan (D)).
R_SIDE_IND = {2026: {"CA-06": "Kiley"}}


def sides(year: int, race_id: str, cands: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    ind_d = MAJOR_IND.get(year, {}).get(race_id)
    ind_r = R_SIDE_IND.get(year, {}).get(race_id)
    lname = cands["candidate"].map(last_name)
    if ind_d:
        d_idx = cands.index[lname == ind_d.lower()]
    else:
        d_idx = cands.index[cands["party"] == "DEM"]
    if ind_r:
        r_idx = cands.index[lname == ind_r.lower()]
    else:
        r_idx = cands.index[cands["party"] == "REP"]
    return d_idx, r_idx


def split_side(cands: pd.DataFrame, idx: pd.Index, total: float) -> pd.Series:
    if len(idx) == 0:
        return pd.Series(dtype=float)
    score = INC_WEIGHT * cands.loc[idx, "is_incumbent"].fillna(0).astype(float)
    # v2: same-party top-two sides with a June primary share for every candidate use the fitted within-party
    # model (within_party.py) instead of the incumbency rule
    if len(idx) > 1 and "sp_score" in cands and cands.loc[idx, "sp_score"].notna().all():
        score = cands.loc[idx, "sp_score"].astype(float)
    if "within_score" in cands:
        score = score + cands.loc[idx, "within_score"].fillna(0).astype(float)
    e = np.exp(score - score.max())
    return total * e / e.sum()


def allocate(year: int, race_id: str, cands: pd.DataFrame, p_d: float | None,
             same_side_p: float | None = None) -> pd.Series:
    """cands: rows of one race with columns candidate, party, is_incumbent (and optional within_score)."""
    d_idx, r_idx = sides(year, race_id, cands)
    q = pd.Series(MINOR_Q, index=cands.index, dtype=float)
    if len(d_idx) and len(r_idx) and p_d is not None:
        pd_side = float(p_d)
    elif len(d_idx) and not len(r_idx):
        pd_side = SOLO_P
    elif len(r_idx) and not len(d_idx):
        pd_side = 1 - SOLO_P
    else:
        pd_side = 0.5 if p_d is None else float(p_d)
    q.loc[d_idx] = split_side(cands, d_idx, pd_side)
    q.loc[r_idx] = split_side(cands, r_idx, 1 - pd_side)
    return q / q.sum()
