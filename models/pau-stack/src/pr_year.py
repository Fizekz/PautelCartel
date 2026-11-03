"""pau-r year-level hooks for pau-b v5.1's TwoLevelMarginModel (set via cfg["pr_ye"]).

The race-level part is fitted exactly as v5.1 on the training years. Only the year-level regression
(year effect on generic ballot x midterm) is replaced:
  pr_ye = "base"   : the same regression refitted here (sanity check: must equal v5.1)
  pr_ye = "old"    : extra environments: year effects of earlier cycles read through the fitted race-level model,
                     fe_t = weighted mean over that year's contested House rows of clip(y) - x'beta (weights as the
                     race fit), added as rows of the year regression with weight pr_old_w. Years in pr_old_years.
  pr_ye = "gbsep"  : separate GB slopes for midterms and presidential years, ridge pr_lam on the presidential slope.
Options: pr_house_fe (recompute the modern training years' effects House-only, like the old ones),
         pr_b_prior / pr_b_lam (ridge of the midterm slope toward a prior value).
"""
import numpy as np
import pandas as pd

import model as M
from common import WORK, ROOT

_OLD = None
_EXT = None


def old_rows():
    global _OLD
    if _OLD is None:
        d = pd.read_csv(ROOT / "data" / "expanded_r" / "house_2002_2008.csv")
        for c in ("money", "money_pa", "last_rel", "last_miss", "rimp", "ltrend", "plean", "plean2", "prim_dev",
                  "has_prim", "indep_dside", "senate"):
            d[c] = 0.0
        d["pres"] = -1
        _OLD = d[d.train_ok == 1].copy()
    return _OLD


def ext():
    global _EXT
    if _EXT is None:
        _EXT = pd.read_csv(WORK / "national_ext.csv").set_index("year")
    return _EXT


def _fe_from_rows(self, d):
    X = M.design(d, self.cfg)
    X = X.drop(columns=[c for c in M.YEAR_LEVEL if c in X])
    for c in self.beta_.index:
        if c not in X:
            X[c] = 0.0
    race = X[self.beta_.index].values @ self.beta_.values
    w = np.where(np.abs(d.rating.values) < 2.5, self.cfg.get("comp_weight", 1.0), 1.0)
    r = np.clip(d.y.values, -60, 60) - race
    t = pd.DataFrame({"year": d.year.values, "wr": w * r, "w": w}).groupby("year").sum()
    return t.wr / t.w


def _fit(self):
    cfg = self.cfg
    kind = cfg["pr_ye"]
    fe = self.fe_.copy()
    if cfg.get("pr_house_fe") and getattr(self, "_pr_train", None) is not None:
        tr = self._pr_train
        fe = _fe_from_rows(self, tr[tr.senate == 0])
    years = list(fe.index)
    vals = list(fe.values)
    wts = [1.0] * len(years)
    if kind == "old":
        o = old_rows()
        o = o[o.year.isin(cfg.get("pr_old_years", [2006, 2008]))]
        if len(o):
            feo = _fe_from_rows(self, o)
            if cfg.get("pr_old_mid_only"):
                feo = feo[[y for y in feo.index if y % 4 == 2]]
            years += list(feo.index)
            vals += list(feo.values)
            wts += [cfg.get("pr_old_w", 1.0)] * len(feo)
        self._pr_old_fe = dict(zip(years, vals))
    e = ext()
    gtr = self._pr_train.groupby("year").gb.first()
    gb = np.array([gtr.get(y, e.gb.get(y)) for y in years], float)
    mid = np.array([int(y % 4 == 2) for y in years], float)
    cols = [np.ones(len(years)), gb * mid]
    pen = [0.0, cfg.get("pr_b_lam", 1e-3)]
    prior = [0.0, cfg.get("pr_b_prior", 0.0)]
    if kind == "gbsep" or cfg.get("pr_gbsep"):
        cols.append(gb * (1 - mid))
        pen.append(cfg.get("pr_lam", 1e-3))
        prior.append(0.0)
    if cfg.get("pr_mid_icpt"):        # separate intercept for midterms
        cols.append(mid)
        pen.append(cfg.get("pr_mid_icpt_lam", 1e-3))
        prior.append(0.0)
    A = np.column_stack(cols)
    W = np.diag(wts)
    P = np.diag(pen)
    b = np.linalg.solve(A.T @ W @ A + P, A.T @ W @ np.array(vals) + P @ np.array(prior))
    self._pr_b = b
    return b


def hook(self, df, race, ye):
    if not self.cfg.get("pr_ye"):
        return ye
    if not hasattr(self, "_pr_b"):
        _fit(self)
    b = self._pr_b
    yr = df.groupby("year").first()
    gb = yr.gb.values.astype(float)
    mid = yr.midterm.values.astype(float)
    cols = [np.ones(len(yr)), gb * mid]
    if self.cfg["pr_ye"] == "gbsep" or self.cfg.get("pr_gbsep"):
        cols.append(gb * (1 - mid))
    if self.cfg.get("pr_mid_icpt"):
        cols.append(mid)
    return pd.Series(np.column_stack(cols) @ b, index=yr.index)


M.TwoLevelMarginModel._ye_hook = hook
_orig_fit = M.TwoLevelMarginModel.fit


def _fit_store(self, df):
    self._pr_train = df
    return _orig_fit(self, df)


M.TwoLevelMarginModel.fit = _fit_store
