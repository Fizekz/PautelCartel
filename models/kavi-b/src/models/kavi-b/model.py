"""Side model: P(D side wins) from expert ratings corrected by lean, incumbency, money and polls.

logit P(D side wins | year effect u) = x'beta + u,   u ~ N(0, tau^2) shared by every race in a cycle.

beta is fitted with year fixed effects (so it is the within-cycle, conditional effect); tau^2 is the spread of
those fixed effects net of their sampling noise. The forecast for a new cycle integrates u out
(Gauss-Hermite), and the simulation draws u explicitly, so the marginal q and the simulated seat counts agree.

Races where all three forecasters say Safe (|mean rating| = 3) are not fitted: 0 of ~2,600 such contested races
flipped in 2010-2024. They get a fixed upset probability SAFE_UPSET.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import sigmoid
from cycle import CycleModel, cycle_z, make_cycle

SAFE_UPSET = 0.003
TAIL_Q = 0.98
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(41)
GH_W = GH_W / GH_W.sum()


@dataclass(frozen=True)
class Config:
    name: str = "base"
    feats: tuple[str, ...] = ("r", "r_hinge", "lean", "inc", "fec", "poll_s", "poll_h")
    ridge: float = 0.002        # L2 penalty per observation on the non-FE coefficients (statsmodels, L1_wt=0)
    tau_floor: float = 0.25
    floor: bool = True
    cycle: str = "const"        # v2: year-effect mean model, see cycle.py ("const" = v1)
    poll_k: float = 1.0         # v2: "poll_sh" shrinks the poll margin by W / (W + poll_k), W = summed poll weight


def design(df: pd.DataFrame, feats: tuple[str, ...], poll_k: float = 1.0) -> pd.DataFrame:
    r = df["r_mean"].astype(float)
    has_poll = df["n_polls"].fillna(0) > 0
    pm = (df["poll_margin"].fillna(0).clip(-30, 30) / 10.0).where(has_poll, 0.0)
    sen = df["chamber"].eq("Senate")
    rr = r
    cols = {
        "r": r,
        "r_hinge": np.sign(r) * np.maximum(r.abs() - 1.0, 0.0),
        "r_cook": df["r_cook"], "r_sabato": df["r_sabato"], "r_inside": df["r_inside"],
        "r_x_spread": r * df["r_spread"],
        "lean": df["lean"].fillna(0) / 10.0,
        "lean_s": (df["lean"].fillna(0) / 10.0).where(sen, 0.0),
        "lean_h": (df["lean"].fillna(0) / 10.0).where(~sen, 0.0),
        "inc": df["inc"].astype(float),
        "fec": df["fec_lr"].astype(float),
        "poll_s": pm.where(sen, 0.0),
        "poll_h": pm.where(~sen, 0.0),
        "poll": pm,
        "poll_sh": pm * df.get("poll_w", pd.Series(0.0, index=df.index)).fillna(0)
                   / (df.get("poll_w", pd.Series(0.0, index=df.index)).fillna(0) + poll_k),
        "oth": ((df["r_oth"] - r).clip(-3, 3)).where(df["n_oth"] > 0, 0.0) if "r_oth" in df else 0.0 * r,
        "r_mid": r * df["midterm"],
        "lean_mid": df["lean"].fillna(0) / 10.0 * df["midterm"],
        "env": df["generic_margin_dr"] / 10.0,
        "env_x_close": df["generic_margin_dr"] / 10.0 * np.exp(-r.abs()),
        "senate": sen.astype(float),
    }
    return pd.DataFrame({f: cols[f] for f in feats}, index=df.index).astype(float)


def fit_set(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the side model learns from: contested D-side vs R-side races not rated Safe by all three."""
    m = (df["same_party"] == 0) & (df["no_contest"] == 0) & (df["r_mean"].abs() < 3)
    return df[m]


@dataclass
class SideModel:
    cfg: Config
    beta: pd.Series = field(default=None)
    year_fx: pd.Series = field(default=None)     # uncentred year intercepts a_y (logit, D+)
    cyc: CycleModel = field(default=None)
    tau: float = 0.0                             # predictive sd of the cycle effect (summary only)
    tail_floor: float = 0.0
    train_years: tuple = ()

    def fit(self, train: pd.DataFrame) -> "SideModel":
        d = fit_set(train)
        X = design(d, self.cfg.feats, self.cfg.poll_k)
        years = sorted(d["year"].unique())
        self.train_years = tuple(int(y) for y in years)
        fe = pd.get_dummies(d["year"].astype(int), prefix="y").astype(float)
        Xf = pd.concat([X, fe], axis=1)
        alpha = np.r_[np.full(X.shape[1], self.cfg.ridge), np.zeros(fe.shape[1])]
        res = sm.GLM(d["y"].astype(float), Xf, family=sm.families.Binomial()).fit_regularized(
            alpha=alpha, L1_wt=0.0, maxiter=500)
        params = pd.Series(res.params, index=Xf.columns)
        self.year_fx = params[fe.columns]
        self.beta = params[X.columns].copy()
        # sampling noise of each year intercept: 1 / sum p(1-p) over that year's races
        p = sigmoid(Xf.to_numpy() @ params.to_numpy())
        info = pd.Series(p * (1 - p), index=d.index).groupby(d["year"]).sum()
        s2 = (1.0 / info).to_numpy()
        yinfo = d.groupby("year")[["midterm", "out_d", "generic_margin_dr"]].first()
        Z = np.vstack([cycle_z(self.cfg.cycle, int(m), int(o), float(g)) for m, o, g in yinfo.to_numpy()])
        self.cyc = make_cycle(self.cfg.cycle).fit(self.year_fx.to_numpy(), s2, Z)
        # Tail floor: training-set upset rate among races the model calls at > TAIL_Q, Beta(1, 1) prior
        q = np.concatenate([self._marginal(self.eta(g), g) for _, g in d.groupby("year")])
        yy = np.concatenate([g["y"].to_numpy() for _, g in d.groupby("year")])
        qf, fav_won = np.maximum(q, 1 - q), np.where(q > 0.5, yy, 1 - yy)
        sure = qf > TAIL_Q
        self.tail_floor = float((np.sum(~fav_won[sure].astype(bool)) + 1) / (sure.sum() + 2)) if self.cfg.floor else 0.0
        return self

    def z_of(self, df: pd.DataFrame) -> np.ndarray:
        m, o, g = df["midterm"].unique(), df["out_d"].unique(), df["generic_margin_dr"].unique()
        assert len(m) == 1 and len(o) == 1 and len(g) == 1, "predict one cycle at a time"
        return cycle_z(self.cfg.cycle, int(m[0]), int(o[0]), float(g[0]))

    def cycle_summary(self, df: pd.DataFrame) -> dict:
        return self.cyc.summary(self.z_of(df))

    def _marginal(self, eta: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        w, mu, sd = self.cyc.predictive(self.z_of(df))
        out = np.zeros(len(eta))
        for wk, mk, sk in zip(w, mu, sd):
            if wk < 1e-6:
                continue
            out += wk * (sigmoid(eta[:, None] + mk + sk * GH_X[None, :]) * GH_W[None, :]).sum(axis=1)
        return out / w[w >= 1e-6].sum()

    def eta(self, df: pd.DataFrame) -> np.ndarray:
        """Race logit WITHOUT the cycle effect (v2: the cycle mean lives in the cycle model)."""
        X = design(df, self.cfg.feats, self.cfg.poll_k)
        return X.to_numpy() @ self.beta[X.columns].to_numpy()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Marginal P(D side wins), integrating the cycle effect; Safe-by-all races get SAFE_UPSET."""
        if len(df) == 0:
            return np.zeros(0)
        q = np.clip(self._marginal(self.eta(df), df), self.tail_floor, 1 - self.tail_floor)
        safe = df["r_mean"].abs().to_numpy() >= 3
        return np.where(safe, np.where(df["r_mean"].to_numpy() > 0, 1 - SAFE_UPSET, SAFE_UPSET), q)


def log_loss(y: np.ndarray, q: np.ndarray) -> float:
    q = np.clip(q, 1e-3, 1 - 1e-3)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def loyo(hist: pd.DataFrame, cfg: Config, years: list[int]) -> pd.DataFrame:
    """Leave-one-year-out inside `years`: fit on the others, score the held-out year (side model only)."""
    rows = []
    for y in years:
        tr = hist[hist["year"].isin([v for v in years if v != y])]
        te = hist[(hist["year"] == y) & (hist["same_party"] == 0) & (hist["no_contest"] == 0)]
        m = SideModel(cfg).fit(tr)
        q = m.predict(te)
        rows.append({"year": y, "n": len(te), "ll": log_loss(te["y"].to_numpy(), q),
                     "ll_nonsafe": log_loss(te["y"].to_numpy()[te["r_mean"].abs() < 3],
                                            q[te["r_mean"].abs().to_numpy() < 3])})
    return pd.DataFrame(rows)
