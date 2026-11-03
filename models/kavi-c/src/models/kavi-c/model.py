"""Polls-first Bayesian aggregator: fundamentals prior + poll average + explicit error model.

Pipeline for a set of training years T (all strictly before the target year):
1. Fundamentals prior: weighted least squares of side A's two-party share on lean, ratings, incumbency, money,
   previous result and the national generic-ballot environment (fit on contested races in T).
2. Poll update: mu = prior + w * (poll_avg - prior), w = neff / (neff + k_chamber). k is fitted on
   leave-one-year-out (LOYO) prior residuals within T, so the prior/poll balance is learned out of sample.
3. Error model: LOYO residuals of mu give a national (year-level) component and a local component whose scale
   depends on chamber, how well polled the race is and how safe it is (maximum likelihood).
4. Win probability: Student-t on the margin, df and a global scale tuned on LOYO log loss of who won.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

SAFE_WEIGHT = 0.3        # WLS weight on races all raters call Safe (their shares matter little for who wins)
RIDGE = 1e-3
USE_ENV_MIDTERM = True    # extra env x midterm slope: the generic ballot carries more weight in midterms (see LOG)
# v2 options (selected on pre-2018 leave-one-year-out, see experiments.py and LOG.md):
USE_MID_OUT = False       # midterm x out-party sign: share shift toward the party not holding the White House
USE_MID_OUT_COMP = True   # the same shift, only in competitive seats (weight 1 at toss-up, 0 beyond Likely)
USE_OTHER_DEV = False     # other raters' mean rating minus the core-3 mean (faster-moving outlets)
# v3 option: the generic-ballot environment x midterm, only in competitive seats (symmetric in the environment;
# no out-party sign). Borrowed from pau-a's gb_mid = GB x midterm x (1 - Safe), adapted to the comp weight.
USE_ENV_MID_COMP = False
OTHER_DEV_CLIP = 1.5
SD_NAT_FLOOR = 0.01
TUNE_SCALE = True        # False: trust the MLE error scale, tune only the t df
NU_GRID = (3.0, 4.0, 5.0, 7.0, 10.0, 20.0)
SCALE_GRID = np.arange(0.5, 2.01, 0.05)
SCALE_SHRINK_N0 = 200.0  # tuned scale shrunk toward 1: s = 1 + (s_hat - 1) n / (n + n0), guards small samples
MLE_NU = 5.0            # df of the t likelihood used to fit the local error scale


def design(df: pd.DataFrame) -> np.ndarray:
    r = df["rating"].to_numpy(float)
    senate = df["senate"].to_numpy(float)
    cols = [
        np.ones(len(df)),
        senate,
        df["lean"].to_numpy(float),
        df["lean"].to_numpy(float) * senate,
        np.clip(r, -2.0, 2.0),
        np.sign(r) * (np.abs(r) > 2.5),
        np.clip(r, -2.0, 2.0) * senate,
        df["inc"].to_numpy(float),
        df["money"].to_numpy(float),
        df["prev_rel"].to_numpy(float),
        df["env"].to_numpy(float),
    ]
    mid_out = df["midterm"].to_numpy(float) * df["out_d"].to_numpy(float)
    if USE_ENV_MIDTERM:
        cols.append(df["env"].to_numpy(float) * df["midterm"].to_numpy(float))
    if USE_MID_OUT:
        cols.append(mid_out)
    if USE_MID_OUT_COMP:
        cols.append(mid_out * competitiveness(r))
    if USE_ENV_MID_COMP:
        cols.append(df["env"].to_numpy(float) * df["midterm"].to_numpy(float) * competitiveness(r))
    if USE_OTHER_DEV:
        cols.append(np.clip(df["other_dev"].to_numpy(float), -OTHER_DEV_CLIP, OTHER_DEV_CLIP))
    return np.column_stack(cols)


def competitiveness(r: np.ndarray) -> np.ndarray:
    """1 at a toss-up, falling linearly to 0 at |mean rating| 2 (Likely)."""
    return np.clip(1.0 - np.abs(r) / 2.0, 0.0, 1.0)


def coef_names() -> list[str]:
    names = ["const", "senate", "lean", "lean_x_sen", "rating", "safe", "rating_x_sen", "inc", "money",
             "prev_rel", "env"]
    for flag, name in ((USE_ENV_MIDTERM, "env_x_midterm"), (USE_MID_OUT, "mid_out"),
                       (USE_MID_OUT_COMP, "mid_out_x_comp"), (USE_ENV_MID_COMP, "env_x_mid_x_comp"),
                       (USE_OTHER_DEV, "other_dev")):
        if flag:
            names.append(name)
    return names


def fit_fund(df: pd.DataFrame) -> np.ndarray:
    X = design(df)
    y = df["share"].to_numpy(float) - 0.5
    w = np.where(df["rating"].abs() > 2.5, SAFE_WEIGHT, 1.0)
    XtW = X.T * w
    A = XtW @ X + RIDGE * np.eye(X.shape[1]) * len(df)
    A[0, 0] -= RIDGE * len(df)
    return np.linalg.solve(A, XtW @ y)


def predict_fund(beta: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    return 0.5 + design(df) @ beta


def poll_weight(neff: np.ndarray, senate: np.ndarray, k_house: float, k_sen: float) -> np.ndarray:
    k = np.where(senate == 1, k_sen, k_house)
    neff = np.nan_to_num(neff, nan=0.0)
    return neff / (neff + k)


def loyo_prior(df: pd.DataFrame) -> np.ndarray:
    """Prior for each training row from a fundamentals fit that leaves its own year out."""
    out = np.full(len(df), np.nan)
    for y in df["year"].unique():
        m = (df["year"] == y).to_numpy()
        beta = fit_fund(df[~m])
        out[m] = predict_fund(beta, df[m])
    return out


def fit_k(df: pd.DataFrame, prior: np.ndarray) -> tuple[float, float]:
    """Choose k per chamber minimising squared error of the poll-updated mean (LOYO prior, polled races)."""
    ks = {}
    for sen in (0, 1):
        m = (df["senate"].to_numpy() == sen) & df["poll_share"].notna().to_numpy()
        if m.sum() < 10:
            ks[sen] = 1.0
            continue
        y, pr = df["share"].to_numpy()[m], prior[m]
        ps, ne = df["poll_share"].to_numpy()[m], df["poll_neff"].to_numpy()[m]
        def loss(logk: float) -> float:
            w = ne / (ne + np.exp(logk))
            return float(np.mean((y - (pr + w * (ps - pr))) ** 2))
        r = optimize.minimize_scalar(loss, bounds=(np.log(0.02), np.log(50)), method="bounded")
        ks[sen] = float(np.exp(r.x))
    return ks[0], ks[1]


def local_scale_design(df: pd.DataFrame) -> np.ndarray:
    ne = np.nan_to_num(df["poll_neff"].to_numpy(float), nan=0.0)
    r = np.abs(df["rating"].to_numpy(float))
    return np.column_stack([np.ones(len(df)), df["senate"].to_numpy(float), np.log1p(ne),
                            np.clip(r, 0, 3), (r > 2.5).astype(float)])


@dataclass
class ErrorModel:
    gamma: np.ndarray            # log-scale coefficients for the poll/rating-dependent local component
    floor: float                 # irreducible local scale (candidate-specific surprises), added in quadrature
    sd_nat: float                # national (shared, year-level) sd of side A's share
    nu: float = 8.0              # Student-t degrees of freedom for the margin
    scale_house: float = 1.0     # chamber multipliers on the total scale, tuned on LOYO win log loss
    scale_sen: float = 1.0
    state_frac: float = 0.0      # share of local variance common to races in one state-year (for simulation)

    def sd_local(self, df: pd.DataFrame) -> np.ndarray:
        return np.sqrt(self.floor ** 2 + np.exp(local_scale_design(df) @ self.gamma) ** 2)

    def scale(self, df: pd.DataFrame) -> np.ndarray:
        return np.where(df["senate"].to_numpy() == 1, self.scale_sen, self.scale_house)

    def sd_total(self, df: pd.DataFrame) -> np.ndarray:
        return self.scale(df) * np.sqrt(self.sd_local(df) ** 2 + self.sd_nat ** 2)


def fit_errors(df: pd.DataFrame, resid: np.ndarray) -> ErrorModel:
    """National sd from year-mean residuals of competitive races; local scale by Student-t MLE after removing them."""
    d = df.assign(res=resid)
    comp = d["rating"].abs() <= 2.0
    ymean = d[comp].groupby("year")["res"].mean()
    n_years = len(ymean)
    # root-mean-square of year means (mean taken as 0), floored at 1 pt of share
    sd_nat = float(np.sqrt(max((ymean ** 2).sum() / max(n_years, 1), SD_NAT_FLOOR ** 2)))
    local = (d["res"] - d["year"].map(ymean).fillna(0.0)).to_numpy()
    Z = local_scale_design(df)

    def nll(theta: np.ndarray) -> float:  # Student-t likelihood: robust to the occasional 10-point miss
        s = np.sqrt(np.exp(theta[0]) ** 2 + np.exp(Z @ theta[1:]) ** 2)
        return float(-np.sum(stats.t.logpdf(local / s, df=MLE_NU) - np.log(s)))

    th0 = np.array([np.log(0.01), np.log(0.04), 0, 0, 0, 0])
    r = optimize.minimize(nll, th0, method="L-BFGS-B")
    return ErrorModel(gamma=r.x[1:], floor=float(np.exp(r.x[0])), sd_nat=sd_nat,
                      state_frac=state_fraction(d.assign(local=local)[comp.to_numpy()]))


def state_fraction(d: pd.DataFrame) -> float:
    """Mean within-state-year pairwise product of local residuals over their mean square (competitive races)."""
    num, n_pairs = 0.0, 0
    for _, g in d.groupby(["year", "state"]):
        r = g["local"].to_numpy()
        if len(r) < 2:
            continue
        tot = r.sum() ** 2 - (r ** 2).sum()
        num += tot
        n_pairs += len(r) * (len(r) - 1)
    if n_pairs == 0:
        return 0.0
    cov = num / n_pairs
    return float(np.clip(cov / np.mean(d["local"] ** 2), 0.0, 0.8))


def win_prob(mu: np.ndarray, sd: np.ndarray, nu: float) -> np.ndarray:
    return stats.t.cdf((mu - 0.5) / sd, df=nu)


@dataclass
class Model:
    beta: np.ndarray
    k_house: float
    k_sen: float
    err: ErrorModel
    diag: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        prior = predict_fund(self.beta, df)
        w = poll_weight(df["poll_neff"].to_numpy(float), df["senate"].to_numpy(), self.k_house, self.k_sen)
        ps = df["poll_share"].to_numpy(float)
        mu = np.where(np.isnan(ps), prior, prior + w * (np.nan_to_num(ps) - prior))
        sd = self.err.sd_total(df)
        return df.assign(prior=prior, poll_w=w, mu=mu, sd=sd, sd_local=self.err.scale(df) * self.err.sd_local(df),
                         sd_nat=self.err.scale(df) * self.err.sd_nat, qA=win_prob(mu, sd, self.err.nu))


def fit(train: pd.DataFrame) -> Model:
    """train: contested races of the training years with columns share, features, poll_share, poll_neff."""
    prior = loyo_prior(train)
    kh, ks = fit_k(train, prior)
    w = poll_weight(train["poll_neff"].to_numpy(float), train["senate"].to_numpy(), kh, ks)
    ps = train["poll_share"].to_numpy(float)
    mu = np.where(np.isnan(ps), prior, prior + w * (np.nan_to_num(ps) - prior))
    resid = train["share"].to_numpy() - mu
    err = fit_errors(train, resid)
    # tune df (shared) and a scale per chamber on LOYO win log loss
    won = (train["share"].to_numpy() > 0.5).astype(float)
    sd0 = np.sqrt(err.sd_local(train) ** 2 + err.sd_nat ** 2)
    sen = train["senate"].to_numpy() == 1
    best = (np.inf, 8.0, 1.0, 1.0)
    for nu in NU_GRID:
        res = {}
        for ch in (False, True):
            m = sen == ch
            cands = []
            for sc in (SCALE_GRID if TUNE_SCALE else [1.0]):
                q = np.clip(win_prob(mu[m], sc * sd0[m], nu), 1e-4, 1 - 1e-4)
                cands.append((-np.sum(won[m] * np.log(q) + (1 - won[m]) * np.log(1 - q)), sc))
            res[ch] = min(cands)
        tot = (res[False][0] + res[True][0]) / len(train)
        if tot < best[0]:
            best = (tot, nu, res[False][1], res[True][1])
    n_h, n_s = int((~sen).sum()), int(sen.sum())
    shrink = lambda s_hat, n: 1.0 + (s_hat - 1.0) * n / (n + SCALE_SHRINK_N0)  # noqa: E731
    err.nu = best[1]
    err.scale_house, err.scale_sen = float(shrink(best[2], n_h)), float(shrink(best[3], n_s))
    beta = fit_fund(train)
    diag = {"k_house": kh, "k_sen": ks, "sd_nat": err.sd_nat, "gamma": err.gamma.round(3).tolist(),
            "scale_house_raw": best[2], "scale_sen_raw": best[3], "nu": err.nu, "scale_house": err.scale_house, "scale_sen": err.scale_sen, "floor": err.floor, "state_frac": err.state_frac, "loyo_logloss": best[0],
            "beta": dict(zip(coef_names(), np.round(beta, 4).tolist())),
            "loyo_rmse_prior": float(np.sqrt(np.mean((train["share"] - prior) ** 2))),
            "loyo_rmse_mu": float(np.sqrt(np.mean(resid ** 2)))}
    return Model(beta=beta, k_house=kh, k_sen=ks, err=err, diag=diag)
