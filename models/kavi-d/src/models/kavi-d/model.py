"""Party-level classifier for kavi-d: P(anti-R side wins).

Main model: monotone-constrained gradient boosting (sklearn HistGradientBoostingClassifier) on the race features,
followed by Platt calibration fitted on rolling-origin out-of-sample predictions from earlier cycles.
Comparators: L2 logistic regression; a margin regressor turned into a probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from common import logit, sigmoid

GBM_FEATURES = ["r_mean", "r_cook", "r_sab", "r_ins", "r_spread", "imputed_safe", "lean_m", "base_m", "inc", "seat",
                "prev_rel", "fec_share", "fec_lr", "fec_logtot", "poll_m", "poll_n", "senate"]
MONOTONE = {"r_mean": 1, "r_cook": 1, "r_sab": 1, "r_ins": 1, "lean_m": 1, "base_m": 1, "inc": 1, "seat": 1,
            "prev_rel": 1, "fec_share": 1, "fec_lr": 1, "poll_m": 1, "op": 1, "oth_dev": 1, "rcp_dev": 1}
LOGIT_FEATURES = ["r_mean", "base_m", "inc", "fec_lr", "poll_m", "prev_rel"]
B_MIN, B_MAX = 0.5, 1.25  # calibration slope bounds
FLIP_SIGNED = ["r_mean", "r_cook", "r_sab", "r_ins", "lean_m", "base_m", "inc", "seat", "prev_rel", "fec_lr",
               "poll_m", "margin", "generic", "op", "oth_dev", "rcp_dev", "env_n"]


@dataclass(frozen=True)
class Config:
    kind: str = "gbm"                       # gbm | logit | margin | blend
    features: tuple = tuple(GBM_FEATURES)
    max_iter: int = 300
    learning_rate: float = 0.03
    max_depth: int = 3
    min_samples_leaf: int = 25
    l2: float = 1.0
    logit_c: float = 0.3
    calibrate: bool = True
    platt_intercept: bool = False
    symmetric: bool = False                 # train on the data plus its D<->R mirror image
    floor_nonsafe: float = 0.0              # clip q to [f, 1-f] unless all three raters say Safe
    min_calib_years: int = 2
    extra: dict = field(default_factory=dict)


def _gbm(cfg: Config) -> HistGradientBoostingClassifier:
    cst = [MONOTONE.get(f, 0) for f in cfg.features]
    return HistGradientBoostingClassifier(max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
                                          max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
                                          l2_regularization=cfg.l2, monotonic_cst=cst, random_state=0)


def _logit_design(df: pd.DataFrame, feats) -> np.ndarray:
    cols = []
    for f in feats:
        x = df[f].astype(float) if f in df else None
        scale = {"base_m": 20.0, "poll_m": 10.0, "prev_rel": 20.0, "r_mean": 1.5, "fec_lr": 1.5}.get(f, 1.0)
        if f == "op_close":  # out-party shift concentrated in competitive races: op x (1 - |r_mean|/3)
            x = df["op"].astype(float) * (1 - df["r_mean"].abs().fillna(3) / 3)
        if f == "gb_mid":  # v3 (pau-a/pau-b idea): generic ballot x midterm, D-positive, /10
            x = df["generic"].astype(float) * (df["op"].astype(float) != 0) / 10.0
        if f == "env_mid":  # v3: national environment N x midterm, /20 (env_n set by environment.py)
            x = df["env_n"].astype(float) * (df["op"].astype(float) != 0) / 20.0
        if f == "base_sen":  # v3: separate base_m slope for Senate races
            x = df["base_m"].astype(float) * df["senate"].astype(float) / 20.0
        if f == "gb_mid_close":  # v3 test (kavi-c idea): GB x midterm concentrated in competitive seats
            x = (df["generic"].astype(float) * (df["op"].astype(float) != 0) / 10.0
                 * (1 - df["r_mean"].abs().fillna(3) / 3))
        if f == "poll_strong":  # v3 test: poll margin x min(poll count, 10)/10 (more weight to well-polled races)
            x = np.nan_to_num(df["poll_m"].astype(float) / 10.0) * df["poll_n"].fillna(0).clip(0, 10) / 10.0
        raw = f in ("op_close", "gb_mid", "env_mid", "base_sen", "gb_mid_close", "poll_strong")
        cols.append(np.nan_to_num(x.to_numpy() / scale, nan=0.0) if not raw else x.to_numpy())
        if f in ("poll_m", "prev_rel", "fec_lr", "oth_dev", "rcp_dev"):
            cols.append(x.isna().to_numpy().astype(float))
    return np.column_stack(cols)


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """D<->R mirror image of the rows: signed features negated, shares flipped, outcome flipped."""
    m = df.copy()
    for f in FLIP_SIGNED:
        if f in m:
            m[f] = -m[f]
    m["fec_share"] = 1 - m["fec_share"]
    m["y"] = 1 - m["y"]
    return m


def raw_fit_predict(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Uncalibrated P(anti-R wins) for test rows, fitted on train rows."""
    env = cfg.extra.get("env")  # v3: train on the actual national margin, integrate N_hat uncertainty at test
    if env and cfg.kind == "logit":
        from environment import env_draws, with_true_env
        train = with_true_env(train)
        sub = Config(kind="logit", logit_c=cfg.logit_c, symmetric=cfg.symmetric,
                     extra={k: v for k, v in cfg.extra.items() if k != "env"})
        if sub.symmetric:
            train = pd.concat([train, mirror(train)], ignore_index=True)
        feats = sub.extra.get("logit_features", LOGIT_FEATURES)
        m = LogisticRegression(C=sub.logit_c, max_iter=2000).fit(_logit_design(train, feats), train["y"])
        return sum(w * m.predict_proba(_logit_design(t, feats))[:, 1] for w, t in env_draws(test, env))
    if cfg.symmetric:
        train = pd.concat([train, mirror(train)], ignore_index=True)
    if cfg.kind == "gbm":
        m = _gbm(cfg).fit(train[list(cfg.features)], train["y"])
        return m.predict_proba(test[list(cfg.features)])[:, 1]
    if cfg.kind == "logit":
        feats = cfg.extra.get("logit_features", LOGIT_FEATURES)
        m = LogisticRegression(C=cfg.logit_c, max_iter=2000).fit(_logit_design(train, feats), train["y"])
        return m.predict_proba(_logit_design(test, feats))[:, 1]
    if cfg.kind == "margin":
        tr = train[train["margin"].notna()]
        reg = HistGradientBoostingRegressor(max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
                                            max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
                                            l2_regularization=cfg.l2, loss="absolute_error",
                                            monotonic_cst=[MONOTONE.get(f, 0) for f in cfg.features], random_state=0)
        reg.fit(tr[list(cfg.features)], tr["margin"].clip(-60, 60))
        sd = cfg.extra.get("margin_sd", 7.0)
        return norm.cdf(reg.predict(test[list(cfg.features)]) / sd)
    if cfg.kind == "blend":
        w = cfg.extra.get("w_gbm", 0.5)
        g = raw_fit_predict(train, test, Config(kind="gbm", features=cfg.features, symmetric=cfg.symmetric))
        lg = raw_fit_predict(train, test, Config(kind="logit", logit_c=cfg.logit_c, extra=cfg.extra,
                                                 symmetric=cfg.symmetric))
        return sigmoid(w * logit(g) + (1 - w) * logit(lg))
    raise ValueError(cfg.kind)


def oof_predictions(df: pd.DataFrame, cfg: Config, before_year: int, min_train_years: int = 2) -> pd.DataFrame:
    """Rolling-origin predictions for each cycle c < before_year, fitted on cycles < c."""
    years = sorted(y for y in df["year"].unique() if y < before_year)
    out = []
    for i, c in enumerate(years):
        if i < min_train_years:
            continue
        tr, te = df[df["year"] < c], df[df["year"] == c]
        out.append(te.assign(p_raw=raw_fit_predict(tr, te, cfg)))
    return pd.concat(out) if out else df.iloc[0:0].assign(p_raw=np.nan)


def platt(p_raw: np.ndarray, y: np.ndarray, intercept: bool = False) -> tuple[float, float]:
    """Fit logit(p) -> a + b*logit(p_raw). Default: slope only (a = 0). A free intercept learns the partisan error
    of the calibration cycles (2014/2016 R years), which is a year effect, not a calibration property."""
    x = logit(p_raw).reshape(-1, 1)
    m = LogisticRegression(C=10.0, max_iter=1000, fit_intercept=intercept).fit(x, y)
    a = float(m.intercept_[0]) if intercept else 0.0
    return a, float(np.clip(m.coef_[0, 0], B_MIN, B_MAX))


def fit_predict(df: pd.DataFrame, test: pd.DataFrame, cfg: Config, year: int) -> tuple[np.ndarray, dict]:
    """Calibrated P(anti-R wins) for `test` (cycle `year`) using only training cycles < year."""
    train = df[df["year"] < year]
    p_raw = raw_fit_predict(train, test, cfg)
    info = {"a": 0.0, "b": 1.0}
    if cfg.calibrate:
        oof = oof_predictions(df, cfg, year)
        if oof["year"].nunique() >= cfg.min_calib_years:
            a, b = platt(oof["p_raw"].to_numpy(), oof["y"].to_numpy(), cfg.platt_intercept)
            info = {"a": a, "b": b, "calib_years": sorted(oof["year"].unique().tolist())}
    p = sigmoid(info["a"] + info["b"] * logit(p_raw))
    if cfg.floor_nonsafe > 0:
        nonsafe = test["r_mean"].abs().fillna(0).to_numpy() < 3
        p = np.where(nonsafe, np.clip(p, cfg.floor_nonsafe, 1 - cfg.floor_nonsafe), p)
    return p, info


def log_loss(p, y) -> float:
    p = np.clip(p, 0.001, 0.999)
    return float(-np.mean(np.where(y == 1, np.log(p), np.log(1 - p))))
