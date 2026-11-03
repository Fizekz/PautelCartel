"""pau-r: within-cycle district-type pattern from that cycle's own district polls (cfg["pr_pat_lam"]).
For each year in the frame being predicted: gap_i = clip(poll_i - yhat_i, -25, 25) over polled House races (poll =
the component's own poll average, yhat = its stage-1 margin with the year effect). Weighted ridge
  gap = a + X gamma, weights n/(n + 1.2), penalty lam |gamma|^2 (intercept unpenalised and NOT applied: the
  cycle's level stays with v5.1's year model)
X = standardised district features (race/features.csv; missing -> 0 = the mean). Then every House race of that
year gets yhat + X gamma before v5.1's poll shrinkage. lam = inf is v5.1. Uses only that year's pre-election polls."""
from pathlib import Path
import numpy as np, pandas as pd
import model as M

F = pd.read_csv(Path(__file__).resolve().parents[2] / "race" / "features.csv")
for r in ("MW", "S", "W"):
    F[f"reg_{r}"] = (F.region == r).astype(float)
F["loginc"] = F.loginc.replace([np.inf, -np.inf], np.nan)
ALL = ["college", "hispanic", "black", "age", "loginc", "owner", "lean", "ltrend", "inc", "reg_MW", "reg_S", "reg_W"]
mu, sd = F[ALL].mean(), F[ALL].std()
Z = ((F[ALL] - mu) / sd).fillna(0.0)
Z.index = pd.MultiIndex.from_arrays([F.year, F.race_id])
SETS = {"all": ALL, "demo": ["college", "hispanic", "black", "age", "loginc", "owner"],
        "lean": ["lean", "ltrend", "inc"], "demo_reg": ["college", "hispanic", "black", "age", "loginc", "owner", "reg_MW", "reg_S", "reg_W"],
        "college": ["college"], "edu_eth": ["college", "hispanic", "black"]}
_orig = M.WinModel._mu
LOG = []
RATED_ONLY = [False]


def pattern(df, lam, cols, unpolled_only=False):
    add = np.zeros(len(df))
    idx = pd.MultiIndex.from_arrays([df.year.values, df.race_id.values])
    X = Z.reindex(idx)[cols].fillna(0.0).values
    house = df.senate.values == 0
    pol = house & (df.poll_pollsters.fillna(0).values > 0) & np.isfinite(df.poll_margin.values.astype(float))
    est = pol & (np.abs(df.rating.values) < 2.5) if RATED_ONLY[0] else pol
    for y in np.unique(df.year.values):
        m = est & (df.year.values == y)
        if m.sum() < 10:
            continue
        g = np.clip(df.poll_margin.values[m].astype(float) - df.yhat.values[m], -25, 25)
        n = df.poll_pollsters.values[m].astype(float)
        w = n / (n + 1.2)
        A = np.column_stack([np.ones(m.sum()), X[m]])
        P = np.diag(np.r_[0.0, np.full(len(cols), lam)])
        b = np.linalg.solve(A.T @ (A * w[:, None]) + P, A.T @ (w * g))
        sel = house & (df.year.values == y)
        if unpolled_only:
            sel = sel & ~pol
        add[sel] = X[sel] @ b[1:]
        LOG.append(dict(year=int(y), n=int(m.sum()), lam=lam, **dict(zip(cols, b[1:]))))
    return add


def _mu(self, df):
    lam = self.cfg.get("pr_pat_lam")
    if lam is None:
        return _orig(self, df)
    d = df.copy()
    d["yhat"] = d.yhat.values + pattern(d, lam, SETS[self.cfg.get("pr_pat_set", "all")], self.cfg.get("pr_pat_unp", False))
    return _orig(self, d)


M.WinModel._mu = _mu
