"""pau-x: harness for new-angle add-ons on top of v6+nl (pau-r's non-linear pattern model).

An add-on is a set of race-level feature columns f_j(year, race_id), each built only from pre-election information.
The hook adds them to the stage-1 margin BEFORE v5.1's poll shrinkage:
    yhat' = yhat + sum_j k_j f_j      (so a race's own poll still overrides the feature, by v5.1's poll weight)
k is fitted inside every WinModel instance (main, Senate and pool components) on its training years' inner-LOYO
predictions: ridge of (y - mu) on the per-unit effects d_j = shrink(yhat + f_j) - shrink(yhat), over rated races
(|rating| < 2.5) where the feature is non-zero; ridge `px_lam` toward 0. The test cycle never enters k.
Features are centred within (year, scope) by their builders, so they cannot shift a cycle's level.

Usage from scripts: H.register(name, series) ; H.evaluate(tag, feats=[...], where=..., lam=...).
"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import numpy as np
import pandas as pd

import evaluate4 as E
import pr_eval as PE
import pr_pattern as PP
import model as M
from common import WORK
from evaluate import metrics

# --- v6+nl: the 4 product columns, as in pr_theta_nl2.py lines 8-11
Z = PP.Z
for a, b in (("college", "college"), ("hispanic", "hispanic"), ("black", "black"), ("college", "hispanic")):
    v = Z[a] * Z[b]
    Z[f"{a}_x_{b}"] = (v - v.mean()) / v.std()
PP.SETS["nl"] = ["college", "hispanic", "black", "college_x_college", "hispanic_x_hispanic", "black_x_black",
                 "college_x_hispanic"]
BASE = PE.all_(PE.V["v51"], pr_pat_lam=1.0, pr_pat_set="nl", pr_pat_unp=True)
HIST = pd.read_csv(WORK / "features_hist.csv", low_memory=False)

FEATS = {}      # name -> pd.Series indexed by MultiIndex (year, race_id)
LOGK = []
_prev_mu = M.WinModel._mu


def register(name, s):
    s = s.copy()
    s.index = pd.MultiIndex.from_tuples([(int(a), str(b)) for a, b in s.index])
    FEATS[name] = s.astype(float)


def _X(df, names):
    idx = pd.MultiIndex.from_arrays([df.year.values.astype(int), df.race_id.astype(str).values])
    return np.column_stack([FEATS[n].reindex(idx).fillna(0.0).values for n in names])


def _mask(df, where):
    sen = df.senate.values == 1
    pol = (df.poll_pollsters.fillna(0).values > 0) & np.isfinite(df.poll_margin.values.astype(float))
    if where == "all":
        return np.ones(len(df), bool)
    if where == "house":
        return ~sen
    if where == "senate":
        return sen
    if where == "unp":
        return ~pol
    if where == "house_unp":
        return (~sen) & (~pol)
    raise ValueError(where)


def _mu(self, df):
    names = self.cfg.get("px_feats")
    if not names:
        return _prev_mu(self, df)
    mu0 = _prev_mu(self, df)
    X = _X(df, names) * _mask(df, self.cfg.get("px_where", "all"))[:, None]
    D = np.zeros_like(X)
    for j in range(X.shape[1]):     # per-unit effect through v5.1's poll shrinkage (exact: shrink is linear in yhat)
        d = df.copy()
        d["yhat"] = d.yhat.values + X[:, j]
        D[:, j] = _prev_mu(self, d) - mu0
    if not hasattr(self, "pxk_"):
        m = (np.abs(df.rating.values) < 2.5) & (np.abs(D).sum(1) > 0)
        if self.cfg.get("px_fit_all"):
            m = np.abs(D).sum(1) > 0
        if "y" in df and m.sum() >= 8:
            r = df.y.values[m] - mu0[m]
            A = D[m]
            lam = self.cfg.get("px_lam", 10.0)
            self.pxk_ = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ r)
            if self.cfg.get("px_pos"):
                self.pxk_ = np.clip(self.pxk_, 0, None)
        else:
            self.pxk_ = np.zeros(X.shape[1])
        LOGK.append(dict(tr=tuple(self.train_years), n=int(m.sum()), **{n: float(k) for n, k in zip(names, self.pxk_)}))
    return mu0 + D @ self.pxk_


M.WinModel._mu = _mu


def subsets(pr):
    rated = pr[pr.rating.abs() < 2.5]
    pol = pr.poll_pollsters.fillna(0) > 0
    unp = pr[(~pol) & (pr.rating.abs() < 2.5) & (pr.senate == 0)]
    return pr, rated, unp


def ll(pr):
    a, r, u = subsets(pr)
    return [round(metrics(x.p_dside, x.dwin)["ll"], 4) for x in (a, r, u)]


def evaluate(tag, feats=None, where="all", lam=10.0, schemes=("six", "pre18", "bt", "loyo8"), save=True, **kw):
    cfg = BASE if not feats else PE.all_(BASE, px_feats=list(feats), px_where=where, px_lam=lam, **kw)
    out = {}
    for sch in schemes:
        pr = pd.concat([E.one(HIST, cfg, T, tr) for T, tr in E.splits(sch)])
        out[sch] = ll(pr)
        out[sch + "_yr"] = {int(y): round(metrics(g.p_dside, g.dwin)["ll"], 4) for y, g in pr.groupby("year")}
        if save and sch == "loyo8":
            (WORK / "px").mkdir(exist_ok=True)
            pr.to_csv(WORK / "px" / f"loyo8_{tag}.csv", index=False)
    out["k"] = [d for d in LOGK[-50:]]
    (WORK / "px").mkdir(exist_ok=True)
    rd = WORK.parent.parent / "results" / "px"
    rd.mkdir(parents=True, exist_ok=True)
    json.dump(dict(tag=tag, feats=feats, where=where, lam=lam, **{k: v for k, v in out.items() if k != "k"}),
              open(rd / f"{tag}.json", "w"), indent=1)
    line = " | ".join(f"{s} {out[s][0]:.4f}/{out[s][1]:.4f}/{out[s][2]:.4f}" for s in schemes)
    print(f"{tag:40s} {line}", flush=True)
    return out


def insample(tag, feats=None, where="all", lam=10.0, **kw):
    """In-sample: train on all 8 cycles (test cycle included), predict each; compare with loyo8."""
    cfg = BASE if not feats else PE.all_(BASE, px_feats=list(feats), px_where=where, px_lam=lam, **kw)
    Y = list(range(2010, 2026, 2))
    wm = M.WinModel(Y, cfg).fit(HIST)
    te = HIST[HIST.year.isin(Y) & (HIST.train_ok == 1)]
    pr = wm.predict(te)
    v = ll(pr)
    print(f"{tag:40s} in-sample {v}", flush=True)
    return v


def permute_within_year(name, seed, group=None):
    """Placebo: shuffle a feature's values within year. group='state' shuffles the state -> value map."""
    s = FEATS[name]
    rng = np.random.default_rng(seed)
    out = s.copy()
    yrs = s.index.get_level_values(0)
    for y in np.unique(yrs):
        m = yrs == y
        if group == "state":
            rid = s.index.get_level_values(1)[m]
            st = np.array([r[:2] for r in rid])
            us = np.unique(st)
            vals = {u: s[m][st == u].values for u in us}
            perm = dict(zip(us, rng.permutation(us)))
            # each state's races take the (state-mean) value of another state
            means = {u: np.mean(v) for u, v in vals.items()}
            out[m] = [means[perm[u]] for u in st]
        else:
            out[m] = s[m].values[rng.permutation(m.sum())]
    return out
