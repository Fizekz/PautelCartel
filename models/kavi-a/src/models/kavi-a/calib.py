"""Win-probability calibration layer, fitted on nested leave-one-year-out predictions.

The vote-share model fits shares by least squares, which under-uses expert ratings in the toss-up/lean band
(LOYO 2010-2016 alone: House 'Likely D' q 0.87 vs won 1.00, 'Lean R' 0.14 vs 0.07). This layer refits
  logit P(D wins) = c0 + c1 z + c2 r + c3 s z + c4 s r [+ c5 M + ...]
with z = probit of the model q, r = rating clipped to +-1.5 (D positive), s = Senate dummy, and (v2) the
cycle term M = midterm x o, o = +1 when the Democrats are the out-party (Republican president), -1 when they hold
the White House. M lets the layer learn, from past midterms only, how much more the out-party wins than the
share model (which already moves with the generic ballot) expects. Fitted on out-of-sample predictions for the
training years only; the calibrated q is turned back into an equivalent mean share (bisection) so the simulation
stays coherent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

Z_CLIP = 3.5
R_CLIP = 1.5         # v3 tested 3.0 (no clip): better on pre-2018 LOYO and 2020/2024, worse on the scorecard -> kept 1.5
C_REG = 1.0
MIN_YEARS_INNER = 3
CALIB_SPECS = ("base", "mo", "mo_z", "mo_r", "mid_r", "mid_z", "mid_rz", "mid")
V3_SPECS = ("inc", "poll", "inc_poll")   # v3 candidates: symmetric incumbency and race-poll terms
POLL_Z_SCALE = 0.03    # poll lead in share units per unit of the calibration's poll term
POLL_Z_CLIP = 3.0
CAL_SPEC = "base"      # chosen on the pre-2018 two-level LOYO (select_v2.py, LOG.md v2)


def design(frame: pd.DataFrame, spec: str | None = None) -> np.ndarray:
    """frame columns: z, rating, senate, midterm, o_d (+ inc, pollz for the v3 specs)."""
    spec = spec or CAL_SPEC
    z = np.clip(frame["z"].to_numpy(dtype=float), -Z_CLIP, Z_CLIP)
    r = np.clip(np.nan_to_num(frame["rating"].to_numpy(dtype=float)), -R_CLIP, R_CLIP)
    s = frame["senate"].to_numpy(dtype=float)
    mid = frame["midterm"].to_numpy(dtype=float)
    m = mid * frame["o_d"].to_numpy(dtype=float)
    cols = [z, r, s * z, s * r]
    if spec == "mo":
        cols += [m]
    elif spec == "mo_z":
        cols += [m, mid * z]
    elif spec == "mo_r":
        cols += [m, m * r]
    elif spec == "mid":      # separate D intercept for midterms (pooled c0 carries the presidential-year D misses)
        cols += [mid]
    elif spec == "mid_r":    # ratings sharper (or flatter) in midterms, symmetric in party
        cols += [mid * r]
    elif spec == "mid_z":
        cols += [mid * z]
    elif spec == "mid_rz":
        cols += [mid * r, mid * z]
    elif spec in ("inc", "inc_poll"):   # does the share model over- or under-rate incumbency in the close band?
        cols += [frame["inc"].to_numpy(dtype=float)]
    elif spec != "base" and spec != "poll":
        raise ValueError(spec)
    if spec in ("poll", "inc_poll"):    # extra weight on the race polls beyond the OLS stage-2 slope
        cols += [frame["pollz"].to_numpy(dtype=float)]
    return np.column_stack(cols)


def to_z(q: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(np.clip(q, 1e-4, 1 - 1e-4))


def cycle_columns(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(midterm 0/1, o_d = +1 if the D side is the out-party) for prediction rows (hist or 2026 features)."""
    mid = rows["midterm"].to_numpy(dtype=float)
    o_d = -rows["pres_d"].to_numpy(dtype=float)
    return mid, o_d


def poll_term(rows: pd.DataFrame, poll_k: float) -> np.ndarray:
    """Poll lead scaled by the poll-weight saturation (0 without polls), clipped."""
    share = rows["poll_share"].to_numpy(dtype=float)
    pw = np.nan_to_num(rows["poll_weight"].to_numpy(dtype=float))
    lead = np.nan_to_num((share - 0.5) / POLL_Z_SCALE)
    return np.clip(lead, -POLL_Z_CLIP, POLL_Z_CLIP) * pw / (pw + poll_k)


def loyo_frame(hist: pd.DataFrame, years: list, use_polls: bool, use_ratings: bool) -> pd.DataFrame:
    """Out-of-sample (leave-one-year-out within `years`) uncalibrated predictions for trainable rows."""
    from features import trainable
    from model import fit_all, predict_chamber, win_prob
    rows = []
    for held in sorted(years):
        inner = [y for y in years if y != held]
        fits = fit_all(hist, inner, use_polls, use_ratings, calibrate=False)
        test = hist[trainable(hist) & (hist["year"] == held)]
        for ch, fit in fits.items():
            sub = test[test["chamber"] == ch]
            pr = predict_chamber(fit, sub, float(sub["g"].iloc[0]))
            q = win_prob(pr["mu"], pr["sd_nat"], pr["sd_state"], pr["sd_race"])
            mid, o_d = cycle_columns(sub)
            from sponsor import shift_polls
            shifted = shift_polls(sub, fit.sponsor)
            rows.append(pd.DataFrame({"year": held, "race_id": sub["race_id"].to_numpy(), "chamber": ch,
                                      "q": q, "z": to_z(q), "rating": sub["rating"].to_numpy(),
                                      "senate": float(ch == "Senate"), "midterm": mid, "o_d": o_d,
                                      "inc": sub["inc"].to_numpy(dtype=float),
                                      "pollz": poll_term(shifted, fit.poll_k),
                                      "won": (sub["dem_share_2p"] > 0.5).astype(int).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def fit_on_frame(d: pd.DataFrame, spec: str | None = None) -> LogisticRegression:
    spec = spec or CAL_SPEC
    model = LogisticRegression(C=C_REG, max_iter=1000).fit(design(d, spec), d["won"])
    model.spec_ = spec
    return model


def fit_calibration(hist: pd.DataFrame, train_years: list, use_polls: bool, use_ratings: bool):
    """Logistic calibration on LOYO predictions within train_years (no data from outside them)."""
    if len(train_years) < MIN_YEARS_INNER + 1:
        return None
    return fit_on_frame(loyo_frame(hist, sorted(train_years), use_polls, use_ratings))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrated_q(model, q: np.ndarray, rating: np.ndarray, senate: bool, midterm: np.ndarray, o_d: np.ndarray,
                 weight: float = 1.0, inc: np.ndarray | None = None,
                 pollz: np.ndarray | None = None) -> np.ndarray:
    """Calibrated q; weight < 1 mixes the calibrated and raw logits."""
    if model is None:
        return q
    n = len(np.asarray(q))
    frame = pd.DataFrame({"z": to_z(np.asarray(q)), "rating": np.asarray(rating, dtype=float),
                          "senate": float(senate), "midterm": midterm, "o_d": o_d,
                          "inc": np.zeros(n) if inc is None else np.asarray(inc, dtype=float),
                          "pollz": np.zeros(n) if pollz is None else np.asarray(pollz, dtype=float)})
    cal = model.predict_proba(design(frame, model.spec_))[:, 1]
    mix = weight * logit(cal) + (1 - weight) * logit(q)
    return 1 / (1 + np.exp(-mix))


def mu_for_q(target: np.ndarray, sd_nat, sd_state, sd_race, lo: float = -0.6, hi: float = 0.6) -> np.ndarray:
    """Mean D-side share - 0.5 that gives win_prob == target under the race's error components."""
    from model import win_prob
    lo_v = np.full(len(target), lo)
    hi_v = np.full(len(target), hi)
    for _ in range(30):
        mid = (lo_v + hi_v) / 2
        q = win_prob(mid, sd_nat, sd_state, sd_race)
        up = q < target
        lo_v = np.where(up, mid, lo_v)
        hi_v = np.where(up, hi_v, mid)
    return (lo_v + hi_v) / 2
