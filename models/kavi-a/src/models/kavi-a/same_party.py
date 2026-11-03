"""Same-party top-two races (CA D v D, R v R) in 2026: fitted primary model + race polls.

v3: the prior is the fitted logistic of same_party_fit.py (two-finalist primary share and incumbency, 48
same-party generals 2012-2024 from kavi-e's primary table), turned into an equivalent expected share
m_A = 0.5 + SD probit(p_A). Polls (two-way share, last 60 days) are averaged in with POLL_WEIGHT each against the
prior of weight 1, and q_A = Phi((m_A - 0.5) / SD). v2's hand-set shrink / incumbency / money terms
(PRIMARY_SHRINK, INC_BONUS, MONEY_COEF) are kept below for reference only and no longer used.
Primary shares: kavi-e's table first (all 9 races), my own Ballotpedia parse as the fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import CLEAN, ELECTION_DAY, ROOT, last_name

PRIMARY_SHRINK = 0.6
INC_BONUS = 0.02
MONEY_COEF = 0.01
POLL_WEIGHT = 1.5
SD = 0.09
PRIMARIES = ROOT / "data" / "kavi-a" / "ca_primaries_2026.csv"


def poll_shares(race_id: str, a: str, b: str) -> list[float]:
    p = pd.read_csv(CLEAN / "polls_2026.csv")
    p = p[(p["race_id"] == race_id) & p["cand_detail"].notna()]
    days = (pd.Timestamp(ELECTION_DAY[2026]) - pd.to_datetime(p["end_date"], errors="coerce")).dt.days
    p = p[(days >= 0) & (days <= 60)]
    out = []
    for detail in p["cand_detail"]:
        shares = {}
        for part in str(detail).split(";"):
            name, _, val = part.rpartition(":")
            shares[last_name(name.split("(")[0])] = float(val)
        va, vb = shares.get(last_name(a)), shares.get(last_name(b))
        if va is not None and vb is not None and va + vb > 0:
            out.append(va / (va + vb))
    return out


def own_primary_share(race_id: str, name_a: str, name_b: str) -> float | None:
    """A's two-finalist share from my own Ballotpedia parse (v1/v2 source; 7 of 9 races)."""
    if not PRIMARIES.exists():
        return None
    prim = pd.read_csv(PRIMARIES)
    prim = prim[prim["race_id"] == race_id]
    pct = []
    for name in (name_a, name_b):
        hit = prim[prim["candidate"].map(last_name) == last_name(name)]
        pct.append(float(hit["primary_pct"].iloc[0]) if len(hit) else np.nan)
    return None if any(np.isnan(pct)) else pct[0] / (pct[0] + pct[1])


def same_party_q(sub: pd.DataFrame) -> pd.Series | None:
    """q for the two main same-party candidates of one race; None if no primary data.

    v3: the primary + incumbency prior is the fitted logistic of same_party_fit.py (2012-2024 same-party
    generals, kavi-e's primary table), turned into an equivalent share m = 0.5 + SD probit(p) so that the race
    polls can be averaged in exactly as in v2. The v2 money term is dropped (no fitted coefficient)."""
    from same_party_fit import fit, primary_share, prob_a
    main = sub[sub["side"] != ""].sort_values("candidate").head(2)  # order is irrelevant to q
    if len(main) < 2:
        return None
    a, b = main.iloc[0], main.iloc[1]
    s_a = primary_share(2026, a["race_id"], a["candidate"], b["candidate"])
    source = "kavi-e table"
    if s_a is None:
        s_a, source = own_primary_share(a["race_id"], a["candidate"], b["candidate"]), "own Ballotpedia"
    if s_a is None:
        return None
    params = fit(tuple(range(2012, 2025, 2)))
    p_prior = prob_a(params, s_a, int(a["is_inc"]), int(b["is_inc"]))
    m = 0.5 + SD * float(stats.norm.ppf(np.clip(p_prior, 1e-4, 1 - 1e-4)))
    polls = poll_shares(a["race_id"], a["candidate"], b["candidate"])
    m = (m + POLL_WEIGHT * sum(polls)) / (1 + POLL_WEIGHT * len(polls))
    q_a = float(stats.norm.cdf((m - 0.5) / SD))
    return pd.Series({a.name: q_a, b.name: 1 - q_a}), {"primary_share_a": s_a, "n_polls": len(polls),
                                                        "m_a": m, "cand_a": a["candidate"],
                                                        "p_prior_a": p_prior, "primary_source": source}
