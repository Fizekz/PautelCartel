"""pau-r: state poll gap for unpolled House races (cfg["pr_sgap"]).
For every race with its own poll average, gap = clip(poll - yhat, -25, 25) (yhat = the model's own stage-1 margin,
year effect included). An unpolled House race gets the leave-own-out mean gap of the other polled races in its
state and year (House and Senate), times k: mu += k * sgap. k is fitted on the training years' inner-LOYO
predictions (unpolled rated House races; ridge pr_sgap_lam toward 0), before sigma and the calibration layer.
All inputs are pre-election polls and the model's own predictions."""
import numpy as np
import pandas as pd
import model as M

_orig_mu = M.WinModel._mu


def sgap(df):
    pol = (df.poll_pollsters.fillna(0).values > 0) & np.isfinite(df.poll_margin.values.astype(float))
    gap = np.where(pol, np.clip(np.nan_to_num(df.poll_margin.values.astype(float)) - df.yhat.values, -25, 25), 0.0)
    key = pd.Series(list(zip(df.year.values, df.state.values)))
    s = pd.Series(gap).groupby(key).transform("sum").values
    c = pd.Series(pol.astype(float)).groupby(key).transform("sum").values
    cnt = c - pol
    out = np.where(cnt > 0, (s - gap * pol) / np.where(cnt > 0, cnt, 1), np.nan)
    return out, pol


def _mu(self, df):
    mu = _orig_mu(self, df)
    if not self.cfg.get("pr_sgap"):
        return mu
    sg, pol = sgap(df)
    unp = (~pol) & (df.senate.values == 0) & np.isfinite(sg)
    if not hasattr(self, "kst_"):
        m = unp & (np.abs(df.rating.values) < 2.5)
        if "y" in df and m.sum() >= 10:
            r = df.y.values[m] - mu[m]
            x = sg[m]
            self.kst_ = float((x * r).sum() / ((x * x).sum() + self.cfg.get("pr_sgap_lam", 0.0)))
            self.kst_ = float(np.clip(self.kst_, 0, 1))
        else:
            self.kst_ = 0.0
    return mu + np.where(unp, self.kst_ * np.nan_to_num(sg), 0.0)


M.WinModel._mu = _mu
