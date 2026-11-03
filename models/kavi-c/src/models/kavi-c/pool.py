"""v3: log-linear pool of the model's q with a ratings-only logistic (idea: baselines.py showed the ratings logistic
beating the model on 2014-2024 on average; pooling borrowed from the Kavi audit ensembles, K016).

q_pool = logit^-1((1 - w) logit q_model + w logit q_rating). w is fixed from pre-2018 LOYO (pool_experiment.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

POOL_W = 0.25         # chosen on pre-2018 LOYO by pool_experiment.py (0.0728 vs 0.0745 at w=0); LOG.md v3
Q_EPS = 1e-4


def _logit(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, Q_EPS, 1 - Q_EPS)
    return np.log(q / (1 - q))


def log_pool(q_model: np.ndarray, q_rating: np.ndarray, w: float) -> np.ndarray:
    z = (1 - w) * _logit(np.asarray(q_model, float)) + w * _logit(np.asarray(q_rating, float))
    return 1 / (1 + np.exp(-z))


FEATURES = ("rating",)   # baselines.py ratings-only logistic; ("rating", "env") tied on pre-2018 LOYO and was rejected


def _x(df: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return df[list(features)].to_numpy(float)


@dataclass
class RatingLogit:
    """P(side A wins) from the mean core-3 rating (as in baselines.py), optionally plus the generic-ballot env."""
    lr: LogisticRegression
    features: tuple[str, ...] = FEATURES

    @classmethod
    def fit(cls, train: pd.DataFrame, features: tuple[str, ...] | None = None) -> "RatingLogit":
        feats = FEATURES if features is None else features
        lr = LogisticRegression(C=10.0).fit(_x(train, feats), (train["share"] > 0.5).astype(int))
        return cls(lr, feats)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(_x(df, self.features))[:, 1]


def apply(det: pd.DataFrame, rl: RatingLogit, nu: float, eligible: np.ndarray | None = None,
          w: float | None = None) -> pd.DataFrame:
    """Pool qA with the ratings logistic in eligible rated rows; move mu so that T_nu((mu - 0.5) / sd) = pooled q
    (keeps the simulation consistent). Adds q_unpooled and q_rating columns."""
    w = POOL_W if w is None else w
    ok = det["rating"].notna().to_numpy().copy()
    if eligible is not None:
        ok &= np.asarray(eligible, bool)
    q0 = det["qA"].to_numpy(float)
    qr = np.full(len(det), np.nan)
    if ok.any():
        qr[ok] = rl.predict(det[ok])
    qp = np.where(ok, log_pool(q0, np.nan_to_num(qr, nan=0.5), w), q0)
    qp_c = np.clip(qp, 1e-6, 1 - 1e-6)
    mu = np.where(ok, 0.5 + det["sd"].to_numpy(float) * stats.t.ppf(qp_c, df=nu), det["mu"].to_numpy(float))
    return det.assign(q_unpooled=q0, q_rating=qr, qA=qp, mu=mu)
