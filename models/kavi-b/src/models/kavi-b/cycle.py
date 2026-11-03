"""v2: hierarchical model of the cycle (year) effect.

The side model's year fixed effects a_y are the shared logit error of every race in cycle y. v1 treated them as
a_y ~ N(c, tau^2): one mean for all cycles. v2 lets the mean depend on the cycle type,

    a_y = z_y' theta + e_y,   e_y ~ N(0, tau^2),   a_y observed with sampling noise s_y^2,

with z_y one of (see CYCLE_DESIGNS)
    "const"  : [1]                                   (v1)
    "mo"     : [1, midterm * out_d]                  out-party midterm shift (out_d = +1 if D is the out-party)
    "mid_mo" : [presidential, midterm, midterm * out_d]
  v3 (generic-ballot conditioning; idea from pau-b's year-level regression a + b GB x midterm and the
  report's finding that GB-tied environments were stable in 2018 and 2022):
    "gb_mid"    : [1, midterm * GB/10]                      GB = final generic-ballot margin D-R (pts)
    "mid_gb"    : [presidential, midterm, midterm * GB/10]
    "mid_mo_gb" : [presidential, midterm, midterm * out_d, midterm * GB/10]

Priors: theta ~ N(0, THETA_SD^2) per coefficient, tau ~ half-Cauchy(0, TAU_SCALE). tau is integrated on a grid,
so the predictive distribution of a new cycle's effect is a mixture of normals
    sum_k w_k N(z*' m_k, tau_k^2 + z*' V_k z*)
which the side model integrates for q and the simulation draws from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THETA_SD = 1.0
TAU_SCALE = 1.0
TAU_GRID = np.linspace(0.05, 3.0, 60)
CYCLE_DESIGNS = ("const", "mo", "mid_mo", "gb_mid", "mid_gb", "mid_mo_gb", "bma")
# v3 "bma": Bayesian model average of these designs, equal prior weight, weights = marginal likelihood of the
# observed year effects (theta and tau integrated out). Chosen so that no single 4-midterm design is picked by hand.
BMA_DESIGNS = ("const", "mo", "mid_mo", "gb_mid", "mid_gb", "mid_mo_gb")


def cycle_z(design: str, midterm: int, out_d: int, gb: float = 0.0) -> np.ndarray:
    g = midterm * gb / 10.0
    if design == "bma":     # raw covariates; CycleBMA maps them to each member design
        return np.array([float(midterm), float(out_d), float(gb)])
    if design == "const":
        return np.array([1.0])
    if design == "mo":
        return np.array([1.0, midterm * out_d])
    if design == "mid_mo":
        return np.array([1.0 - midterm, float(midterm), midterm * out_d])
    if design == "gb_mid":
        return np.array([1.0, g])
    if design == "mid_gb":
        return np.array([1.0 - midterm, float(midterm), g])
    if design == "mid_mo_gb":
        return np.array([1.0 - midterm, float(midterm), midterm * out_d, g])
    raise ValueError(design)


@dataclass
class CycleModel:
    design: str
    weights: np.ndarray = None     # posterior weight of each tau grid point
    means: list = None             # posterior mean of theta per grid point
    covs: list = None              # posterior covariance of theta per grid point

    def fit(self, a: np.ndarray, s2: np.ndarray, Z: np.ndarray) -> "CycleModel":
        p = Z.shape[1]
        prior_cov = np.eye(p) * THETA_SD ** 2
        logw, means, covs = [], [], []
        for tau in TAU_GRID:
            sig = np.diag(tau ** 2 + s2)
            marg = Z @ prior_cov @ Z.T + sig
            sign, logdet = np.linalg.slogdet(marg)
            ll = -0.5 * (logdet + a @ np.linalg.solve(marg, a))
            lp = np.log(2 / (np.pi * TAU_SCALE * (1 + (tau / TAU_SCALE) ** 2)))
            logw.append(ll + lp)
            si = np.linalg.inv(sig)
            V = np.linalg.inv(Z.T @ si @ Z + np.linalg.inv(prior_cov))
            means.append(V @ Z.T @ si @ a)
            covs.append(V)
        logw = np.array(logw)
        self.log_evidence = float(logw.max() + np.log(np.exp(logw - logw.max()).sum()))
        w = np.exp(logw - logw.max())
        self.weights, self.means, self.covs = w / w.sum(), means, covs
        return self

    def predictive(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(weights, means, sds) of the normal mixture for a new cycle with covariates z."""
        mu = np.array([z @ m for m in self.means])
        sd = np.sqrt(TAU_GRID ** 2 + np.array([z @ V @ z for V in self.covs]))
        return self.weights, mu, sd

    def summary(self, z: np.ndarray) -> dict:
        w, mu, sd = self.predictive(z)
        mean = float(w @ mu)
        var = float(w @ (sd ** 2 + mu ** 2) - mean ** 2)
        theta = sum(wk * m for wk, m in zip(w, self.means))
        return {"design": self.design, "theta": [float(t) for t in theta], "tau_post_mean": float(w @ TAU_GRID),
                "pred_mean": mean, "pred_sd": float(np.sqrt(var))}

    def draw(self, z: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        w, mu, sd = self.predictive(z)
        k = rng.choice(len(w), size=n, p=w)
        return rng.normal(mu[k], sd[k])


@dataclass
class CycleBMA:
    """v3: mixture over BMA_DESIGNS weighted by each design's marginal likelihood (equal prior odds)."""
    design: str = "bma"
    members: list = None
    post: np.ndarray = None        # posterior design probabilities

    def fit(self, a: np.ndarray, s2: np.ndarray, Zraw: np.ndarray) -> "CycleBMA":
        self.members = []
        for d in BMA_DESIGNS:
            Z = np.vstack([cycle_z(d, int(m), int(o), g) for m, o, g in Zraw])
            self.members.append(CycleModel(d).fit(a, s2, Z))
        le = np.array([m.log_evidence for m in self.members])
        p = np.exp(le - le.max())
        self.post = p / p.sum()
        return self

    def predictive(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m_, o_, g_ = z
        ws, mus, sds = [], [], []
        for pd_, mem in zip(self.post, self.members):
            w, mu, sd = mem.predictive(cycle_z(mem.design, int(m_), int(o_), float(g_)))
            ws.append(pd_ * w), mus.append(mu), sds.append(sd)
        return np.concatenate(ws), np.concatenate(mus), np.concatenate(sds)

    def summary(self, z: np.ndarray) -> dict:
        w, mu, sd = self.predictive(z)
        mean = float(w @ mu)
        var = float(w @ (sd ** 2 + mu ** 2) - mean ** 2)
        return {"design": "bma", "design_weights": {d: round(float(p), 4) for d, p in zip(BMA_DESIGNS, self.post)},
                "theta": [], "tau_post_mean": float(sum(p * (m.weights @ TAU_GRID) for p, m in zip(self.post, self.members))),
                "pred_mean": mean, "pred_sd": float(np.sqrt(var))}

    def draw(self, z: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        w, mu, sd = self.predictive(z)
        k = rng.choice(len(w), size=n, p=w / w.sum())
        return rng.normal(mu[k], sd[k])


def make_cycle(design: str):
    return CycleBMA() if design == "bma" else CycleModel(design)
