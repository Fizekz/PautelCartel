"""Turn a candidate list per race into (a) the party-model structure row and (b) per-candidate q.

Each candidate needs: race_id, candidate, party, receipts (FEC, may be NaN), inc (0/1), viable_ind (bool).
Sides: R-side = REP candidates; anti-R side = DEM candidates plus viable independents. The principal of each side
is the candidate with the largest same-party strength (incumbency + money).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import EPS_MINOR
from sameparty import split, strength

IND_VIABLE_RECEIPTS = 1.0e6
ONE_SIDED_Q = 0.998  # q for the only viable side when the other major side has no candidate


def classify(c: pd.DataFrame) -> pd.DataFrame:
    """Add `side` (D / R / minor) to one race's candidates."""
    c = c.copy()
    viable_ind = (~c["party"].isin(["DEM", "REP"])) & (c["viable_ind"].astype(bool))
    c["side"] = np.where(c["party"] == "REP", "R", np.where((c["party"] == "DEM") | viable_ind, "D", "minor"))
    return c


def principal(c: pd.DataFrame, side: str, params) -> pd.Series | None:
    s = c[c["side"] == side]
    if s.empty:
        return None
    st = strength(s["inc"], s["receipts"], params)
    return s.iloc[int(np.argmax(st))]


def structure_row(race_id: str, chamber: str, year: int, c: pd.DataFrame, params, special: int = 0) -> dict | None:
    d, r = principal(c, "D", params), principal(c, "R", params)
    if d is None or r is None:
        return None
    return {"year": year, "race_id": race_id, "chamber": chamber, "state": race_id[:2], "special": special,
            "d_name": d["candidate"], "r_name": r["candidate"], "d_is_ind": d["party"] != "DEM"}


def candidate_q(c: pd.DataFrame, p_anti_r: float | None, params, poll_share: dict | None = None,
                primary_share: dict | None = None) -> np.ndarray:
    """q per candidate for one race. p_anti_r is the party model's P(anti-R side wins), None if one-sided."""
    q = np.zeros(len(c))
    sides = c["side"].to_numpy()
    has_d, has_r = (sides == "D").any(), (sides == "R").any()
    if not has_d and not has_r:
        return np.full(len(c), 1.0 / len(c))
    if has_d and has_r:
        mass = {"D": float(p_anti_r), "R": 1.0 - float(p_anti_r)}
    else:
        only = "D" if has_d else "R"
        n_minor = int((sides == "minor").sum())
        mass = {only: ONE_SIDED_Q if n_minor else 1.0}
    for side, m in mass.items():
        idx = np.where(sides == side)[0]
        names = [c.iloc[i]["candidate"] for i in idx]
        ps = None if poll_share is None else [poll_share.get(n, np.nan) for n in names]
        pr = None if primary_share is None else [primary_share.get(n, np.nan) for n in names]
        w = split(c.iloc[idx]["inc"], c.iloc[idx]["receipts"], params, ps, pr)
        q[idx] = m * w
    minor = np.where(sides == "minor")[0]
    q[minor] = EPS_MINOR
    return q / q.sum()
