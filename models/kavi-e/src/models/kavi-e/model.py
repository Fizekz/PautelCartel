"""Swing model: fundamentals regression on the D-R margin, a poll layer, and a calibrated error model.

Stage A  margin_i = b0 + b_L L + b_Lp L*pres_year + b_N N + b_inc inc + ... + e_i   (OLS on contested D-v-R races,
         training years only, N = actual national House margin of the training year)
Stage B  mu_i = muA_i + w_i (poll_i - muA_i),  w_i = neff_i / (neff_i + k)   (muA with the PRE-election N_hat)
Error    y_i - mu_i = (b_N (1 - w_i) + w_i) eps_N + u_i,  eps_N ~ N(0, sigma_N^2) shared by all races,
         sd(u_i) = tau * sqrt(s_e^2 (1 - w_i)^2 + s_p^2 w_i^2)
P(D side wins) = Phi(mu_i / sqrt(((b_N (1-w) + w) sigma_N)^2 + sd(u_i)^2)).
s_e from stage-A residuals, (k, s_p) by least squares / residual sd on training years, tau by maximum likelihood of
the training years' winners (leave-one-year-out stage-A fits), so calibration is learned before the target year.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm, t as student_t

import national

# v2: + rgap (other raters' mean minus the core-3 mean; ratings_extra.py)
HOUSE_FEATURES = ("L", "L_pres", "N", "inc", "rating")
HOUSE_FEATURES_PREV = ("L", "L_pres", "N", "inc", "prev_resid_inc", "prev_resid_open", "rating")
SENATE_FEATURES = ("L", "L_pres", "N", "inc", "rating")
K_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0)
Q_CLIP = 1e-3
POLL_SHIFT_K = 2.0  # shrinkage (in years) of the historical mean poll miss
USE_POLL_SHIFT = False
TAU_MODE = "blend"
TAU_BOUNDS = (0.3, 2.5)
CLOSE_MU = 10.0
CLOSE_RATING = 1.5  # |mean rating| below this = Tilt/Lean/Toss-up territory (v2 out_mid_close)
MIDTERM_TAU = False  # tried (v2): Senate-only gain 0.0015 pre-2018, House none; not adopted
ERR_DF = 6  # 0 = normal errors; > 2 = Student-t with this many df, scaled to the same sd
# v3 options (env KAVI_E_V3="hetero,wls,..." overrides for the selection harness only)
SAFE_R = 2.5  # |mean rating| >= this = the three raters (nearly) all say Safe
V3_DEFAULT = "pool"  # v3 adopted: classifier pool (selected on non-test LOYO, eval_v3.py)
V3 = set(filter(None, os.environ.get("KAVI_E_V3", V3_DEFAULT).split(",")))
WLS_SAFE_WEIGHT = 0.5  # stage-A weight of safe-rated rows under the "wls" option (tested, not adopted)
# v3 "pool": logit weight of the mirrored winner classifier (classifier.py), chosen on non-test LOYO
POOL_A = {"House": 0.4, "Senate": 0.0}  # Senate 0: a 0.2 (classifier on House + Senate rows) won non-test LOYO but worsened both backtests; reverted (LOG v3)


def z_for(p: np.ndarray) -> np.ndarray:
    """Inverse of win_prob: the standardised mean that gives P(D) = p."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    if ERR_DF and ERR_DF > 2:
        return student_t.ppf(p, ERR_DF) / np.sqrt(ERR_DF / (ERR_DF - 2))
    return norm.ppf(p)


def win_prob(z: np.ndarray) -> np.ndarray:
    """P(margin > 0) for standardised mean z = mu / sigma."""
    if ERR_DF and ERR_DF > 2:
        return student_t.cdf(z * np.sqrt(ERR_DF / (ERR_DF - 2)), ERR_DF)
    return norm.cdf(z)  # tried: hurt both backtests (see LOG.md)


@dataclass
class Fitted:
    features: tuple[str, ...]
    coef: np.ndarray
    s_e: float
    k: float = 5.0
    s_p: float = 5.0
    tau: float = 1.0
    tau_mid: float | None = None  # v2 option: tau fitted on the midterm training years only
    poll_shift: float = 0.0
    s_e_close: float | None = None  # v3 "hetero": stage-A residual sd of non-safe-rated rows
    s_e_safe: float | None = None
    clf: object | None = None  # v3 "pool": fitted classifier.LogisticRegression
    pool_a: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def b_n(self) -> float:
        return float(self.coef[1 + self.features.index("N")])


def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["L_pres"] = out["L"] * out["pres_year"]
    # v2 midterm / out-party terms. outdir = +1 when the D side is the out-party (R president), -1 otherwise.
    outdir = out["year"].map(lambda y: 1.0 if national.pres_party(int(y)) == "R" else -1.0)
    mid = 1.0 - out["pres_year"].astype(float)
    out["out_mid"] = mid * outdir
    r = out["rating"].fillna(0.0) if "rating" in out else 0.0
    out["out_mid_close"] = out["out_mid"] * (np.abs(r) < CLOSE_RATING)
    out["out_mid_r"] = out["out_mid"] * np.clip(1.0 - np.abs(r) / 3.0, 0.0, 1.0)
    out["rating_mid"] = r * mid
    out["inc_out_mid"] = out["inc"].fillna(0.0) * out["out_mid"] if "inc" in out else 0.0
    out["grp_close"] = np.abs(r) < SAFE_R
    out["r_hinge"] = np.sign(r) * np.clip(np.abs(r) - 1.5, 0.0, None)
    out["L_safe"] = out["L"] * (np.abs(r) >= SAFE_R)
    for col in ("prev_resid_inc", "prev_resid_open", "rating", "inc", "rgap"):
        if col in out:
            out[col] = out[col].fillna(0.0)
    if "poll_neff" not in out:
        out["poll_neff"] = np.nan
        out["poll_margin"] = np.nan
    return out


def design(df: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([np.ones(len(df))] + [df[f].to_numpy(float) for f in features])


def stage_a_weights(train: pd.DataFrame) -> np.ndarray:
    if "wls" not in V3 or "grp_close" not in train:
        return np.ones(len(train))
    return np.where(train["grp_close"].to_numpy(bool), 1.0, WLS_SAFE_WEIGHT)


def fit_stage_a(train: pd.DataFrame, features: tuple[str, ...]) -> tuple[np.ndarray, float]:
    X, y = design(train, features), train["margin"].to_numpy(float)
    sw = np.sqrt(stage_a_weights(train))
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ coef
    return coef, float(np.sqrt(np.sum(resid ** 2) / (len(y) - X.shape[1])))


def group_s_e(train: pd.DataFrame, features: tuple[str, ...], coef: np.ndarray) -> tuple[float, float]:
    """v3: residual sd of stage A separately for non-safe-rated (close) and safe-rated rows."""
    resid = train["margin"].to_numpy(float) - design(train, features) @ coef
    close = train["grp_close"].to_numpy(bool)
    return float(np.sqrt(np.mean(resid[close] ** 2))), float(np.sqrt(np.mean(resid[~close] ** 2)))


def stage_a(df: pd.DataFrame, features: tuple[str, ...], coef: np.ndarray, n_value: float | pd.Series) -> np.ndarray:
    tmp = df.copy()
    tmp["N"] = n_value
    return design(tmp, features) @ coef


def poll_weight(neff: pd.Series, k: float) -> np.ndarray:
    n = neff.fillna(0.0).to_numpy(float)
    return n / (n + k)


def combine(mu_a: np.ndarray, df: pd.DataFrame, k: float) -> tuple[np.ndarray, np.ndarray]:
    w = poll_weight(df["poll_neff"], k)
    poll = df["poll_margin"].fillna(0.0).to_numpy(float)
    return mu_a + w * (poll - mu_a), w


def n_hat_map(years) -> dict[int, dict]:
    return {int(y): national.estimate(int(y)) for y in sorted(set(years))}


def poll_shift(train: pd.DataFrame) -> float:
    """Shrunk mean of the yearly mean poll miss (actual - poll margin) over training years; 0 if disabled."""
    if not USE_POLL_SHIFT:
        return 0.0
    polled = train[train["poll_neff"].fillna(0) > 0]
    if polled.empty:
        return 0.0
    by_year = (polled["margin"] - polled["poll_margin"]).groupby(polled["year"]).mean()
    return float(by_year.mean() * len(by_year) / (len(by_year) + POLL_SHIFT_K))


def loyo_frame(train: pd.DataFrame, features: tuple[str, ...], nh: dict[int, dict]) -> pd.DataFrame:
    """For each training year, stage-A fitted on the OTHER training years, evaluated with that year's N_hat;
    that year's polls shifted by the poll miss of the other years."""
    parts = []
    for y in sorted(train["year"].unique()):
        others = train[train["year"] != y]
        coef, s_e = fit_stage_a(others, features)
        part = train[train["year"] == y].copy()
        part["poll_margin"] = part["poll_margin"] + poll_shift(others)
        part["mu_a"] = stage_a(part, features, coef, nh[y]["n_hat"])
        part["b_n"] = coef[1 + features.index("N")]
        part["s_e"] = s_e
        part["sigma_n"] = nh[y]["sigma_n"]
        parts.append(part)
    return pd.concat(parts)


def fit_poll_layer(lf: pd.DataFrame) -> tuple[float, float]:
    polled = lf[lf["poll_neff"].fillna(0) > 0]
    if len(polled) < 20:
        return 5.0, 6.0
    best = None
    for k in K_GRID:
        mu, _ = combine(polled["mu_a"].to_numpy(), polled, k)
        sse = float(np.sum((polled["margin"] - mu) ** 2))
        if best is None or sse < best[1]:
            best = (k, sse)
    k = best[0]
    # pure-poll idiosyncratic sd: residual of the poll itself after removing each year's mean poll miss
    miss = polled["margin"] - polled["poll_margin"]
    miss = miss - miss.groupby(polled["year"]).transform("mean")
    return float(k), float(miss.std())


def sigma_total(fit: Fitted, w: np.ndarray, sigma_n: float, tau: float | None = None,
                mult: np.ndarray | float = 1.0, close: np.ndarray | None = None) -> np.ndarray:
    t = fit.tau if tau is None else tau
    nat = (fit.b_n * (1 - w) + w) * sigma_n
    s_e = fit.s_e
    if fit.s_e_close is not None and close is not None:
        s_e = np.where(close, fit.s_e_close, fit.s_e_safe)
    idio = mult * t * np.sqrt((s_e * (1 - w)) ** 2 + (fit.s_p * w) ** 2)
    return np.sqrt(nat ** 2 + idio ** 2)


def log_loss_d(p_d: np.ndarray, d_won: np.ndarray) -> float:
    p = np.clip(np.where(d_won == 1, p_d, 1 - p_d), Q_CLIP, 1 - Q_CLIP)
    return float(-np.mean(np.log(p)))


def fit_tau(fit: Fitted, lf: pd.DataFrame) -> float:
    """tau from the LOYO frame. 'likelihood': maximise the winners' likelihood; 'margin': make the margin z-scores
    of close races (|mu| < CLOSE_MU) have sd 1; 'blend': geometric mean of the two."""
    mu, w = combine(lf["mu_a"].to_numpy(), lf, fit.k)
    d_won = (lf["winner_party"] == "DEM").to_numpy(int)
    sig_n = lf["sigma_n"].to_numpy(float)
    grp = lf["grp_close"].to_numpy(bool)

    def loss(t: float) -> float:
        return log_loss_d(win_prob(mu / sigma_total(fit, w, sig_n, t, close=grp)), d_won)

    t_ll = float(minimize_scalar(loss, bounds=TAU_BOUNDS, method="bounded").x)
    close = np.abs(mu) < CLOSE_MU
    resid = (lf["margin"].to_numpy(float) - mu)[close]

    def zgap(t: float) -> float:
        return abs(float(np.std(resid / sigma_total(fit, w[close], sig_n[close], t, close=grp[close]))) - 1.0)

    t_m = float(minimize_scalar(zgap, bounds=TAU_BOUNDS, method="bounded").x)
    fit.info["tau_likelihood"], fit.info["tau_margin"] = round(t_ll, 3), round(t_m, 3)
    return {"likelihood": t_ll, "margin": t_m, "blend": float(np.sqrt(t_ll * t_m))}[TAU_MODE]


def fit(train: pd.DataFrame, features: tuple[str, ...], clf_extra: pd.DataFrame | None = None) -> Fitted:
    """clf_extra: other-chamber training rows added to the classifier's training set (v3: House rows for the Senate)."""
    train = prep(train[train["contested"] & train["L"].notna()])
    coef, s_e = fit_stage_a(train, features)
    nh = n_hat_map(train["year"])
    lf = loyo_frame(train, features, nh)
    k, s_p = fit_poll_layer(lf)
    f = Fitted(features, coef, s_e, k, s_p, poll_shift=poll_shift(train))
    f.info = {"n_train": int(len(train)), "years": sorted(int(y) for y in train["year"].unique())}
    if "hetero" in V3:
        f.s_e_close, f.s_e_safe = group_s_e(train, features, coef)
    if "pool" in V3:
        import classifier
        chamber = "Senate" if train["race_id"].str.contains("SEN").all() else "House"
        f.pool_a = POOL_A[chamber]
        clf_train = train
        if clf_extra is not None:
            extra = prep(clf_extra[clf_extra["contested"] & clf_extra["L"].notna()])
            clf_train = pd.concat([train, extra], ignore_index=True)
        f.clf = classifier.fit(clf_train, k)
        f.info["clf_n_train"] = int(len(clf_train))
        f.info["pool_a"] = f.pool_a
        f.info["clf_coef"] = dict(zip(classifier.FEATURE_NAMES, np.round(f.clf.coef_[0], 4).tolist()))
    f.tau = fit_tau(f, lf)
    if MIDTERM_TAU:
        mid = lf[lf["pres_year"] == 0]
        if len(mid):
            info = dict(f.info)
            f.tau_mid = fit_tau(f, mid)
            f.info = {**info, "tau_mid": round(f.tau_mid, 3)}
    return f


def predict(f: Fitted, df: pd.DataFrame, n_hat: float, sigma_n: float) -> pd.DataFrame:
    df = prep(df)
    is_mid = bool(len(df)) and int(df["pres_year"].iloc[0]) == 0
    tau = f.tau_mid if (MIDTERM_TAU and is_mid and f.tau_mid is not None) else None
    df["poll_margin"] = df["poll_margin"] + f.poll_shift
    mu_a = stage_a(df, f.features, f.coef, n_hat)
    mu, w = combine(mu_a, df, f.k)
    mult = df["sigma_mult"].fillna(1.0).to_numpy(float) if "sigma_mult" in df else 1.0
    sig = sigma_total(f, w, sigma_n, tau=tau, mult=mult, close=df["grp_close"].to_numpy(bool))
    out = df.copy()
    out["mu_a"], out["mu"], out["w_poll"], out["sigma"] = mu_a, mu, w, sig
    out["p_d"] = win_prob(mu / sig)
    if f.clf is not None and f.pool_a > 0:
        import classifier
        # independents (sigma_mult > 1) keep the margin model: the classifier never saw such races
        a = np.where(np.asarray(mult) * np.ones(len(out)) > 1.0, 0.0, f.pool_a)
        out["p_margin"], out["mu_margin"] = out["p_d"], out["mu"]
        out["p_clf"] = classifier.predict(f.clf, df, n_hat, f.k)
        out["p_d"] = classifier.pool(out["p_margin"].to_numpy(), out["p_clf"].to_numpy(), a)
        out["mu"] = sig * z_for(out["p_d"].to_numpy())  # implied margin: keeps sims / dossier pool coherent
    return out


def describe(f: Fitted) -> dict:
    return {"coef": dict(zip(("const",) + f.features, np.round(f.coef, 4).tolist())), "s_e": round(f.s_e, 3),
            "k": f.k, "s_p": round(f.s_p, 3), "tau": round(f.tau, 3),
            "poll_shift": round(f.poll_shift, 3),
            "s_e_close": None if f.s_e_close is None else round(f.s_e_close, 3),
            "s_e_safe": None if f.s_e_safe is None else round(f.s_e_safe, 3), **f.info}
