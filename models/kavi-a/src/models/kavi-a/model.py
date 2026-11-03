"""kavi-a: fundamentals vote-share model with a fitted three-level error.

Stage 1 (fundamentals, per chamber): dem two-party share - 0.5 = X beta + alpha_year + e, with year fixed effects.
  X = presidential lean (last two elections, relative to the nation), incumbency, receipts log-ratio, and for
  House seats on unchanged lines the previous result's deviation from lean + national vote.
National: alpha_year is predicted from the final generic-ballot average, alpha = a + b * g (g = margin / 200),
  b shrunk toward 0.8; the residual sd of that mapping is the national error.
Stage 2 (optional): fundamentals forecast mu_F (with the generic-ballot alpha, as at forecast time) is updated with
  the race poll average and expert ratings by an OLS fitted on the same training years.
Error: total sd per race = sqrt(nat^2 + state^2 + race^2); state and race parts come from the stage-2 residuals
  (method of moments on same-state pairs), national from the alpha mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

B_PRIOR, B_PRIOR_SD = 0.8, 0.25
NAT_SD_FLOOR = 0.015
POLL_K_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)  # poll weight half-saturation, picked by SSE per chamber
RACE_SD_FLOOR = 0.015
RATING_KNOT = 1.0
CAL_WEIGHT = 0.75     # 1 = full calibration; < 1 mixes it with the raw q in logit space (v2: pre-2018 LOYO; v3 kept)
T_DF = 6              # race-level error is Student t with this many df (fat tails)

HOUSE_X_FULL = ["lean", "lean2f", "inc", "money", "prevf", "prev_inc", "hasprev"]
HOUSE_X_BY_PREV = {"full": HOUSE_X_FULL,
                   "inc_only": ["lean", "lean2f", "inc", "money", "prev_inc", "hasprev"],
                   "none": ["lean", "lean2f", "inc", "money"]}
SENATE_X = ["lean", "lean2f", "inc", "money"]
# v3 switches (select_v3.py decides them on pre-2018 LOYO + 2020/2024 forward tests)
PREV_MODE = "full"      # previous-House-result features: full / inc_only / none
STAGE2_WLS = 0.0        # > 0: stage-2 OLS weighted by 1 + STAGE2_WLS * exp(-|mu_F| / WLS_SCALE)
WLS_SCALE = 0.05
GCLOSE = False          # True: House stage 1 adds g x exp(-|lean| / GCLOSE_SCALE) (competitive seats swing more;
GCLOSE_SCALE = 0.10     # a symmetric version of pau-a's GB x midterm x non-safe term)


@dataclass
class ChamberFit:
    chamber: str
    xcols: list
    beta: pd.Series
    alpha: dict
    a: float
    b: float
    nat_sd: float
    stage2: dict = field(default_factory=dict)
    sd_state: float = 0.0
    sd_race: dict = field(default_factory=dict)
    poll_k: float = 1.0
    calib: object = None
    resid: pd.Series = None
    frame: pd.DataFrame = None
    sponsor: tuple = None     # v3: (bias_D, bias_R) of sponsored polls, from the training years


def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lean2f"] = out["lean2"].fillna(out["lean"])
    out["hasprev"] = out["prev_dev"].notna().astype(int)
    out["prevf"] = out["prev_dev"].fillna(0.0)
    out["prev_inc"] = out["prevf"] * out["prev_same_inc"].fillna(0)
    out["has_poll"] = out["poll_share"].notna().astype(int)
    out["pw"] = out["poll_weight"].fillna(0.0)
    out["rating_f"] = out["rating"].fillna(0.0)
    out["y"] = out["dem_share_2p"] - 0.5 if "dem_share_2p" in out else np.nan
    out["gclose"] = out["g"] * np.exp(-out["lean"].abs() / GCLOSE_SCALE) if "g" in out else 0.0
    return out


def fit_stage1(d: pd.DataFrame, xcols: list) -> tuple[pd.Series, dict]:
    years = sorted(d["year"].unique())
    dummies = pd.get_dummies(d["year"], prefix="yr").astype(float)
    X = pd.concat([d[xcols].astype(float), dummies], axis=1)
    m = sm.OLS(d["y"], X).fit()
    alpha = {y: float(m.params[f"yr_{y}"]) for y in years}
    return m.params[xcols], alpha


def fit_national(alpha: dict, g: dict) -> tuple[float, float, float]:
    """alpha_t = a + b g_t; b has a normal prior; returns (a, b, residual sd)."""
    ys = np.array([alpha[y] for y in sorted(alpha)])
    gs = np.array([g[y] for y in sorted(alpha)])
    gc, yc = gs - gs.mean(), ys - ys.mean()
    s2 = max(np.var(yc - B_PRIOR * gc, ddof=1), 1e-5)
    prec = (gc @ gc) / s2 + 1 / B_PRIOR_SD ** 2
    b = ((gc @ yc) / s2 + B_PRIOR / B_PRIOR_SD ** 2) / prec
    a = ys.mean() - b * gs.mean()
    resid = ys - a - b * gs
    sd = np.sqrt((resid @ resid) / max(len(ys) - 1, 1) * (1 + 1 / len(ys)))
    return float(a), float(b), float(max(sd, NAT_SD_FLOOR))


def mu_fund(d: pd.DataFrame, beta: pd.Series, alpha_year: pd.Series) -> pd.Series:
    return d[beta.index].astype(float) @ beta + alpha_year


def stage2_design(d: pd.DataFrame, mu_f: pd.Series, use_polls: bool, use_ratings: bool,
                  poll_k: float = 1.0) -> pd.DataFrame:
    X = pd.DataFrame({"mu_f": mu_f}, index=d.index)
    if use_polls:
        w = d["pw"] / (d["pw"] + poll_k)
        X["poll_gap"] = w * (d["poll_share"].fillna(0.5) - 0.5 - mu_f)
    if use_ratings:  # piecewise-linear in the rating: the toss-up..lean band carries most of the information
        r = d["rating_f"]
        inner = r.clip(-RATING_KNOT, RATING_KNOT)
        X["rating"] = inner / 100.0
        X["rating_outer"] = (r - inner) / 100.0
    return X


def stage2_weights(mu_f: pd.Series) -> np.ndarray:
    """v3 option: up-weight races the fundamentals call close, so the rating/poll slopes fit the competitive band."""
    return 1.0 + STAGE2_WLS * np.exp(-np.abs(mu_f.to_numpy(dtype=float)) / WLS_SCALE)


def fit_chamber(train: pd.DataFrame, chamber: str, g: dict, use_polls: bool, use_ratings: bool) -> ChamberFit:
    xcols = HOUSE_X_BY_PREV[PREV_MODE] if chamber == "House" else SENATE_X
    if GCLOSE and chamber == "House":
        xcols = xcols + ["gclose"]
    d = prep(train[train["chamber"] == chamber])
    beta, alpha = fit_stage1(d, xcols)
    a, b, nat_sd = fit_national(alpha, g)
    fit = ChamberFit(chamber, xcols, beta, alpha, a, b, nat_sd)
    # stage 2 on forecast-time mu_F: alpha from leave-one-year-out generic mapping
    alpha_oos = {}
    for y in alpha:
        rest = {k: v for k, v in alpha.items() if k != y}
        a_, b_, _ = fit_national(rest, g) if len(rest) >= 2 else (a, b, nat_sd)
        alpha_oos[y] = a_ + b_ * g[y]
    mu_f = mu_fund(d, beta, d["year"].map(alpha_oos))
    best = None
    for k in (POLL_K_GRID if use_polls else (1.0,)):
        X = stage2_design(d, mu_f, use_polls, use_ratings, k)
        wts = stage2_weights(mu_f)
        m = sm.WLS(d["y"], X, weights=wts).fit()
        if best is None or m.ssr < best[0]:
            best = (m.ssr, k, m.params, d["y"] - X @ m.params)
    _, fit.poll_k, params, resid = best
    fit.stage2 = params.to_dict()
    fit.resid = resid
    fit.frame = d
    return fit


def state_cov(d: pd.DataFrame, resid: pd.Series) -> float:
    """Mean product of year-demeaned residuals over pairs of races in the same state and year (both chambers)."""
    r = resid - resid.groupby(d["year"]).transform("mean")
    key = d["year"].astype(str) + d["state"]
    cov_sum, n_pairs = 0.0, 0
    for _, grp in r.groupby(key):
        v = grp.to_numpy()
        if len(v) > 1:
            s = v.sum()
            cov_sum += (s * s - (v * v).sum()) / 2
            n_pairs += len(v) * (len(v) - 1) / 2
    return max(cov_sum / n_pairs, 0.0) if n_pairs else 0.0


def race_sd(d: pd.DataFrame, resid: pd.Series, cov: float) -> dict:
    r = resid - resid.groupby(d["year"]).transform("mean")
    sd_race = {}
    for grp_name, mask in (("poll", d["has_poll"] == 1), ("nopoll", d["has_poll"] == 0)):
        var = float((r[mask] ** 2).mean()) if mask.any() else float((r ** 2).mean())
        sd_race[grp_name] = float(np.sqrt(max(var - cov, RACE_SD_FLOOR ** 2)))
    return sd_race


def predict_chamber(fit: ChamberFit, rows: pd.DataFrame, g_year: float) -> pd.DataFrame:
    from sponsor import shift_polls
    d = prep(shift_polls(rows, fit.sponsor))
    alpha_hat = fit.a + fit.b * g_year
    mu_f = mu_fund(d, fit.beta, pd.Series(alpha_hat, index=d.index))
    use_polls, use_ratings = "poll_gap" in fit.stage2, "rating" in fit.stage2
    X = stage2_design(d, mu_f, use_polls, use_ratings, fit.poll_k)
    mu = X @ pd.Series(fit.stage2)[X.columns]
    sd_race = np.where(d["has_poll"] == 1, fit.sd_race["poll"], fit.sd_race["nopoll"])
    out = pd.DataFrame({"mu": mu, "mu_f": mu_f, "sd_race": sd_race, "sd_state": fit.sd_state,
                        "sd_nat": fit.nat_sd}, index=d.index)
    out["sd_total"] = np.sqrt(out["sd_nat"] ** 2 + out["sd_state"] ** 2 + out["sd_race"] ** 2)
    out["mu_raw"] = out["mu"]
    if fit.calib is not None:
        from calib import calibrated_q, cycle_columns, mu_for_q, poll_term
        q_raw = win_prob(out["mu"], out["sd_nat"], out["sd_state"], out["sd_race"])
        mid, o_d = cycle_columns(d)
        q_cal = calibrated_q(fit.calib, q_raw, d["rating"].to_numpy(), fit.chamber == "Senate", mid, o_d,
                             CAL_WEIGHT, inc=d["inc"].to_numpy(dtype=float), pollz=poll_term(d, fit.poll_k))
        out["mu"] = mu_for_q(q_cal, out["sd_nat"].to_numpy(), out["sd_state"].to_numpy(), out["sd_race"].to_numpy())
    return out


def win_prob(mu: np.ndarray, sd_nat: np.ndarray, sd_state: np.ndarray, sd_race: np.ndarray,
             n_draw: int = 4000, seed: int = 7) -> np.ndarray:
    """P(dem share > 0.5): normal national+state error, Student-t race error (scaled to the same sd)."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_draw)
    shared = np.sqrt(np.asarray(sd_nat) ** 2 + np.asarray(sd_state) ** 2)
    scale = np.asarray(sd_race) / np.sqrt(T_DF / (T_DF - 2))
    m = np.asarray(mu)[:, None] + shared[:, None] * z[None, :]
    return stats.t.cdf(m / scale[:, None], df=T_DF).mean(axis=1)


def fit_all(hist: pd.DataFrame, train_years: list, use_polls: bool, use_ratings: bool,
            calibrate: bool = False) -> dict:
    from features import trainable
    from sponsor import shift_polls, sponsor_bias
    nat_g = hist.groupby("year")["g"].first().to_dict()
    bias = sponsor_bias(tuple(train_years))
    t = shift_polls(hist[trainable(hist) & hist["year"].isin(train_years)], bias)
    fits = {ch: fit_chamber(t, ch, nat_g, use_polls, use_ratings) for ch in ("House", "Senate")}
    for f in fits.values():
        f.sponsor = bias
    frame = pd.concat([f.frame for f in fits.values()])
    resid = pd.concat([f.resid for f in fits.values()])
    cov = state_cov(frame, resid)  # pooled: House pairs dominate; Senate-House pairs add cross-chamber info
    for f in fits.values():
        f.sd_state = float(np.sqrt(cov))
        f.sd_race = race_sd(f.frame, f.resid, cov)
    if calibrate:
        from calib import fit_calibration
        cal = fit_calibration(hist, train_years, use_polls, use_ratings)
        for f in fits.values():
            f.calib = cal
    return fits
