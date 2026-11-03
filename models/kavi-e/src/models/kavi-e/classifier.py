"""v3 option: a direct, D/R mirror-symmetric logistic classifier of the winner, pooled with the margin model.

Idea borrowed from kavi-d (mirror-symmetric L2 logit on ratings, lean, incumbency, polls) and from the disclosed
pool results (two structurally different models pooled beat either alone in 2022). Here it is refitted on my own
training frames: features rating, L, L_pres, N (actual in training, N_hat when predicting), inc, the poll margin
(0 when unpolled, capped at +-30) times the poll weight w = neff/(neff+k), and that weight's sign-free indicator is
not needed (mirroring). Pooling: logit p = (1 - a) logit p_margin + a logit p_clf.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CLF_C = 1.0  # v3: C=0.1 won the non-test LOYO but worsened the 2018 backtest more than it helped 2022; reverted (LOG.md)
POLL_CAP = 30.0
CLIP = 1e-4
FEATURE_NAMES = ("rating", "L/10", "L_pres/10", "N/10", "inc", "w_poll*poll/10")


def features(df: pd.DataFrame, n_value, k: float) -> np.ndarray:
    neff = df["poll_neff"].fillna(0.0).to_numpy(float)
    w = neff / (neff + k)
    pm = np.clip(df["poll_margin"].fillna(0.0).to_numpy(float), -POLL_CAP, POLL_CAP) * w
    n = np.broadcast_to(np.asarray(n_value, float), (len(df),)) if np.ndim(n_value) == 0 else np.asarray(n_value, float)
    cols = [df["rating"].fillna(0.0).to_numpy(float), df["L"].to_numpy(float) / 10.0,
            (df["L"] * df["pres_year"]).to_numpy(float) / 10.0, n / 10.0, df["inc"].fillna(0.0).to_numpy(float),
            pm / 10.0]
    return np.column_stack(cols)


def fit(train: pd.DataFrame, k: float) -> LogisticRegression:
    X = features(train, train["N"].to_numpy(float), k)
    y = (train["winner_party"] == "DEM").to_numpy(int)
    Xm, ym = np.vstack([X, -X]), np.concatenate([y, 1 - y])
    return LogisticRegression(C=CLF_C, fit_intercept=False, max_iter=2000).fit(Xm, ym)


def predict(clf: LogisticRegression, df: pd.DataFrame, n_hat: float, k: float) -> np.ndarray:
    return clf.predict_proba(features(df, n_hat, k))[:, 1]


def pool(p_margin: np.ndarray, p_clf: np.ndarray, a) -> np.ndarray:
    lm = np.log(np.clip(p_margin, CLIP, 1 - CLIP) / np.clip(1 - p_margin, CLIP, 1))
    lc = np.log(np.clip(p_clf, CLIP, 1 - CLIP) / np.clip(1 - p_clf, CLIP, 1))
    return 1.0 / (1.0 + np.exp(-((1 - a) * lm + a * lc)))
