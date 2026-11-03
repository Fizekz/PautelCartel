"""Turn a candidate list plus race features into per-candidate win probabilities.

Each race gets a D-role and an R-role main candidate:
  D-role = the Democrat (incumbent first, then most receipts), unless an independent is the real non-R
  challenger (incumbent independent, a forced independent from the candidates file, or a well-funded independent
  with no Democrat on the ballot). R-role = the Republican the same way.
Race types:
  two_party     D-role vs R-role: q from the vote-share model (mu, error components).
  same_party    top-two lockout (D v D, R v R): heuristic on incumbency and receipts.
  one_sided     only one side has a candidate: that candidate 1 - minor mass.
Jungle/RCV races with several candidates of one party: the party probability is split inside the party
by the same incumbency/receipts heuristic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import last_name, money_feature

MINOR_Q = 0.001           # each minor candidate
INTRA_INC = 1.6           # logit per incumbency difference (same-party contests)
INTRA_MONEY = 0.7         # logit per log receipts ratio (clipped at +-3)
STRONG_IND_RECEIPTS = 5e5


def intra_party_probs(sub: pd.DataFrame) -> np.ndarray:
    """Win shares among same-party candidates: softmax of incumbency and log receipts."""
    rec = np.log(sub["receipts"].fillna(0).clip(lower=0).to_numpy() + 2.5e4)
    score = INTRA_INC * sub["is_inc"].fillna(0).to_numpy() + INTRA_MONEY * np.clip(rec - rec.max(), -3, 0)
    e = np.exp(score - score.max())
    return e / e.sum()


SAME_PARTY_FIT = True     # v3: fitted primary-share model (same_party_fit.py) where the primary is known


def fitted_same_party(main: pd.DataFrame) -> np.ndarray | None:
    """q of the two same-party finalists from the fitted primary model (training years < the race's year)."""
    if len(main) != 2 or "year" not in main:
        return None
    from same_party_fit import fit, primary_share, prob_a
    year = int(main["year"].iloc[0])
    a, b = main.iloc[0], main.iloc[1]
    s_a = primary_share(year, a["race_id"], a["candidate"], b["candidate"])
    if s_a is None:
        return None
    params = fit(tuple(range(2012, min(year, 2026), 2)))
    p = prob_a(params, s_a, int(a["is_inc"]), int(b["is_inc"]))
    return np.array([p, 1 - p])


def pick_main(sub: pd.DataFrame) -> int | None:
    if sub.empty:
        return None
    s = sub.sort_values(["is_inc", "receipts"], ascending=False)
    return s.index[0]


def assign_roles(c: pd.DataFrame) -> pd.DataFrame:
    """Adds columns side ('D', 'R' or '') and main (1 for the D-role / R-role candidate).

    If the table has a 'force_side' column (2026: built from the candidates file's is_major flags), it is used as
    the side directly; otherwise the side is inferred from party, incumbency and FEC filing (backtests)."""
    c = c.copy()
    c["side"] = ""
    c["main"] = 0
    if "force_side" in c:
        c["side"] = c["force_side"].fillna("")
        for _, sub in c.groupby("race_id"):
            for side in ("D", "R"):
                m = pick_main(sub[sub["side"] == side])
                if m is not None:
                    c.loc[m, "main"] = 1
        return c
    for rid, sub in c.groupby("race_id"):
        dem = sub[sub["party"] == "DEM"]
        rep = sub[sub["party"] == "REP"]
        ind = sub[~sub["party"].isin(["DEM", "REP"])]
        forced = ind[ind.get("force_drole", pd.Series(False, index=ind.index)).fillna(False).astype(bool)]
        # a true independent (FEC files them outside D/R): a Democrat who lost the primary and stays on a minor
        # line (Crowley NY-14 2018, WFP) is not a D-role candidate
        true_ind = ~ind.get("fec_bucket", pd.Series("IND", index=ind.index)).isin(["DEM", "REP"])
        strong = ind[true_ind & ((ind["is_inc"] == 1) |
                                 ((ind["receipts"].fillna(0) >= STRONG_IND_RECEIPTS) & dem.empty))]
        d_pool = forced if not forced.empty else (strong if not strong.empty else dem)
        if not forced.empty or not strong.empty:
            c.loc[dem.index, "side"] = ""  # the Democrat becomes a minor candidate
        c.loc[d_pool.index, "side"] = "D"
        c.loc[rep.index, "side"] = "R"
        for side_pool in (d_pool, rep):
            m = pick_main(side_pool)
            if m is not None:
                c.loc[m, "main"] = 1
    return c


def race_frame(c: pd.DataFrame) -> pd.DataFrame:
    """Race-level inc / money from the main candidates, and the race type."""
    rows = []
    for rid, sub in c.groupby("race_id"):
        d = sub[(sub["side"] == "D") & (sub["main"] == 1)]
        r = sub[(sub["side"] == "R") & (sub["main"] == 1)]
        parties = sub.loc[sub["party"].isin(["DEM", "REP"]), "party"]
        if not d.empty and not r.empty:
            kind = "two_party"
        elif len(sub) >= 2 and parties.nunique() == 1 and (parties == parties.iloc[0]).sum() >= 2 and \
                (d.empty or r.empty):
            kind = "same_party"
        else:
            kind = "one_sided"
        d_inc = int(d["is_inc"].iloc[0]) if not d.empty else 0
        r_inc = int(r["is_inc"].iloc[0]) if not r.empty else 0
        rows.append({"race_id": rid, "kind": kind, "inc": d_inc - r_inc, "open": int(d_inc == 0 and r_inc == 0),
                     "d_rec": d["receipts"].iloc[0] if not d.empty else 0.0,
                     "r_rec": r["receipts"].iloc[0] if not r.empty else 0.0,
                     "d_name": d["candidate"].iloc[0] if not d.empty else "",
                     "r_name": r["candidate"].iloc[0] if not r.empty else "",
                     "inc_name": (d["candidate"].iloc[0] if d_inc else (r["candidate"].iloc[0] if r_inc else "")),
                     "d_is_ind": int(not d.empty and d["party"].iloc[0] != "DEM")})
    out = pd.DataFrame(rows).set_index("race_id")
    out["money"] = money_feature(out["d_rec"], out["r_rec"])
    return out


def prev_same(races: pd.DataFrame) -> pd.Series:
    pw = races["prev_winner"].fillna("")
    return pd.Series([int(bool(a) and bool(b) and last_name(a) == last_name(b))
                      for a, b in zip(pw, races["inc_name"].fillna(""))], index=races.index)


def candidate_q(c: pd.DataFrame, races: pd.DataFrame) -> pd.Series:
    """Per-candidate q given races['p_d'] (P the D-role side wins) for two_party races."""
    q = pd.Series(0.0, index=c.index)
    for rid, sub in c.groupby("race_id"):
        kind = races.at[rid, "kind"]
        n_minor_pool = sub[sub["side"] == ""]
        if kind == "two_party":
            pd_ = races.at[rid, "p_d"]
            for side, p_side in (("D", pd_), ("R", 1 - pd_)):
                grp = sub[sub["side"] == side]
                q[grp.index] = p_side * intra_party_probs(grp)
        elif kind == "same_party":
            main = sub[sub["party"].isin(["DEM", "REP"])]
            fitted = fitted_same_party(main) if SAME_PARTY_FIT else None
            q[main.index] = intra_party_probs(main) if fitted is None else fitted
        else:
            main = sub[sub["side"].isin(["D", "R"])]
            if main.empty:
                main = sub
            q[main.index] = intra_party_probs(main)
        q[n_minor_pool.index] = np.where(q[n_minor_pool.index] > 0, q[n_minor_pool.index], MINOR_Q)
        tot = q[sub.index].sum()
        q[sub.index] = q[sub.index] / tot
    return q
