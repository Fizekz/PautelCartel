"""pau-t step 5: the Senate race as a state node fed by its House districts (pre-election polls only).
For each Senate row: h_S = sum_j w_j e_j / (sum_j w_j + c0) over the state's polled House races, e_j = residual poll
gap after the cycle's pattern ridge (as pt_prop), w_j = n_j / (n_j + 1.2), c0 = cfg pt_sen_c0 (shrinkage, default 2).
mu_S += kappa_S * h_S, kappa_S fitted on the Senate rows of the training cycles' inner-LOYO predictions
(least squares, clipped to [-1, 1.5]) before the Senate scales and calibration. Applied in the Senate sub-model
only (cfg key pt_senh). Placebo: pt_senh_shuffle = seed permutes h_S across states within cycle."""
import numpy as np, pandas as pd
import model as M
import pr_pattern as PP

_orig = M.WinModel._mu
LOG = []


def house_signal(df, cfg):
    out = np.zeros(len(df)); cnt = np.zeros(len(df))
    cols = PP.SETS[cfg.get("pr_pat_set", "nl")]
    idx = pd.MultiIndex.from_arrays([df.year.values, df.race_id.values])
    X = PP.Z.reindex(idx)[cols].fillna(0.0).values
    pm = df.poll_margin.values.astype(float)
    npl = df.poll_pollsters.fillna(0).values.astype(float)
    pol = (npl > 0) & np.isfinite(pm)
    house = df.senate.values == 0
    yv = df.year.values
    st = df.race_id.str[:2].values
    for y in np.unique(yv):
        o = house & pol & (yv == y)
        if o.sum() < 10:
            continue
        g = np.clip(pm[o] - df.yhat.values[o], -25, 25)
        w = npl[o] / (npl[o] + 1.2)
        A = np.column_stack([np.ones(o.sum()), X[o]])
        P = np.diag(np.r_[0.0, np.full(len(cols), 1.0)])
        b = np.linalg.solve(A.T @ (A * w[:, None]) + P, A.T @ (w * g))
        e = g - A @ b
        so = st[o]
        sens = np.where((yv == y) & (df.senate.values == 1))[0]
        vals = {}
        for s_ in np.unique(st[sens]):
            mm = so == s_
            vals[s_] = (float((w[mm] * e[mm]).sum() / (w[mm].sum() + cfg.get("pt_sen_c0", 2.0))), int(mm.sum()))
        keys = list(vals)
        if cfg.get("pt_senh_shuffle") is not None:
            perm = np.random.default_rng(cfg["pt_senh_shuffle"] * 100 + int(y)).permutation(len(keys))
            vals = {k: vals[keys[p]] for k, p in zip(keys, perm)}
        if cfg.get("pt_senh_demean"):      # control: remove the cycle mean of the signal (state-level part only)
            mv = np.mean([v[0] for v in vals.values() if v[1] > 0]) if vals else 0.0
            vals = {k: (v[0] - mv, v[1]) if v[1] > 0 else v for k, v in vals.items()}
        for k in sens:
            out[k], cnt[k] = vals.get(st[k], (0.0, 0))
    return out, cnt


def _mu(self, df):
    mu = _orig(self, df)
    c = self.cfg
    if not c.get("pt_senh"):
        return mu
    h, cnt = house_signal(df, c)
    sen = df.senate.values == 1
    if not hasattr(self, "ks_"):
        m = sen & (cnt > 0)
        if "y" in df and m.sum() >= 10:
            r = np.clip(df.y.values[m], -60, 60) - mu[m]
            self.ks_ = float(np.clip((h[m] * r).sum() / ((h[m] ** 2).sum() + 1e-9), -1, 1.5))
        else:
            self.ks_ = 0.0
        LOG.append(dict(kappa=self.ks_, n=int(m.sum())))
    return mu + np.where(sen, self.ks_ * h, 0.0)


M.WinModel._mu = _mu
