"""Upset analysis, calibration table and feature importance (all on rolling-origin out-of-sample predictions).

Upset = the expert-rating favourite (|mean rating| >= 0.5) loses. Outputs:
  out/analysis_calibration.csv   OOF calibration by probability bin (2014-2024)
  out/analysis_upsets.csv        upset rates by rating bucket and by poll/rating disagreement, model vs actual
  out/analysis_upset_logit.csv   logistic regression of upset on favourite-signed features (pooled OOF years)
  out/analysis_importance.csv    permutation importance (log-loss increase) of the GBM, averaged over val years
Usage: python analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backtest import chosen_config
from common import OUT
from history import training_table
from model import Config, fit_predict, log_loss, raw_fit_predict

VAL = [2014, 2016, 2018, 2020, 2022, 2024]


def oof(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    parts = []
    for y in VAL:
        te = df[df["year"] == y]
        p, _ = fit_predict(df, te, cfg, y)
        parts.append(te.assign(p=p))
    return pd.concat(parts, ignore_index=True)


def calibration(o: pd.DataFrame) -> pd.DataFrame:
    bins = [0, .01, .05, .15, .3, .5, .7, .85, .95, .99, 1]
    o = o.assign(bin=pd.cut(o["p"], bins, include_lowest=True))
    return o.groupby("bin", observed=True).agg(n=("y", "size"), mean_p=("p", "mean"), actual=("y", "mean")).reset_index()


def upsets(o: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = o[o["r_mean"].abs() >= 0.5].copy()
    fav_d = f["r_mean"] > 0
    sgn = np.where(fav_d, 1.0, -1.0)
    f["upset"] = np.where(fav_d, 1 - f["y"], f["y"])
    f["p_upset"] = np.where(fav_d, 1 - f["p"], f["p"])
    f["rating_strength"] = f["r_mean"].abs()
    f["poll_vs_fav"] = sgn * f["poll_m"]           # + = polls agree with the favourite
    f["base_vs_fav"] = sgn * f["base_m"]           # lean + generic in the favourite's direction
    f["inc_fav"] = sgn * f["inc"]                  # +1 favourite is the incumbent, -1 the underdog is
    f["prev_vs_fav"] = sgn * f["prev_rel"]
    f["env_vs_fav"] = sgn * f["generic"]           # national environment with/against the favourite
    f["fav_is_d"] = fav_d.astype(float)
    f["spread"] = f["r_spread"]
    f["bucket"] = pd.cut(f["rating_strength"], [0.49, 1.01, 2.01, 3.01], labels=["lean", "likely", "safe"])
    f["poll_flag"] = np.select([f["poll_m"].isna(), f["poll_vs_fav"] < 0, f["poll_vs_fav"] < 5],
                               ["no polls", "polls favour underdog", "polls fav < +5"], "polls fav >= +5")
    t1 = f.groupby(["bucket", "poll_flag"], observed=True).agg(n=("upset", "size"), upset_rate=("upset", "mean"),
                                                             model_p_upset=("p_upset", "mean")).reset_index()
    feats = ["rating_strength", "base_vs_fav", "inc_fav", "env_vs_fav", "fav_is_d", "spread"]
    g = f[f["rating_strength"] <= 2].copy()
    x = g[feats].fillna(0).to_numpy()
    xp = np.column_stack([np.nan_to_num(g["poll_vs_fav"].to_numpy() / 10), g["poll_m"].isna().astype(float),
                          np.nan_to_num(g["prev_vs_fav"].to_numpy() / 20), g["prev_rel"].isna().astype(float)])
    X = np.column_stack([x, xp])
    names = feats + ["poll_vs_fav_per10", "no_poll", "prev_vs_fav_per20", "no_prev"]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, g["upset"])
    t2 = pd.DataFrame({"feature": names, "std_coef": m.coef_[0], "mean": mu, "sd": sd})
    t2.loc[len(t2)] = ["(n races, upset rate)", float(len(g)), float(g["upset"].mean()), np.nan]
    return t1, t2.sort_values("std_coef", key=np.abs, ascending=False)


def importance(df: pd.DataFrame, cfg: Config, n_rep: int = 3) -> pd.DataFrame:
    """Permutation importance of the chosen model's inputs (log-loss increase on the held-out cycle, refitted
    model per validation year, no calibration step)."""
    rng = np.random.default_rng(0)
    feats = cfg.extra.get("logit_features", list(cfg.features)) if cfg.kind == "logit" else list(cfg.features)
    rows = []
    for y in VAL:
        tr, te = df[df["year"] < y], df[df["year"] == y]
        base = log_loss(raw_fit_predict(tr, te, cfg), te["y"].to_numpy())
        for f in feats:
            if f not in te or te[f].nunique(dropna=False) <= 1:  # op is constant within a cycle: skip
                continue
            inc = []
            for _ in range(n_rep):
                x = te.copy()
                x[f] = rng.permutation(x[f].to_numpy())
                inc.append(log_loss(raw_fit_predict(tr, x, cfg), te["y"].to_numpy()) - base)
            rows.append({"year": y, "feature": f, "delta_logloss": float(np.mean(inc))})
    r = pd.DataFrame(rows)
    return r.pivot_table(index="feature", columns="year", values="delta_logloss").assign(
        mean=lambda d: d.mean(axis=1)).sort_values("mean", ascending=False).reset_index()


def midterm_table(o: pd.DataFrame) -> pd.DataFrame:
    """Midterm OOF predictions by who the ratings favour: out-party vs president's party, by rating bucket."""
    m = o[o["op"] != 0].copy()
    out_fav = np.sign(m["r_mean"].fillna(0)) * m["op"]  # + = ratings favour the out-party
    m["fav"] = np.select([m["r_mean"].abs() < 0.5, out_fav > 0], ["toss-up", "out-party"], "in-party")
    m["bucket"] = pd.cut(m["r_mean"].abs(), [-0.01, 0.49, 1.01, 2.01, 3.01], labels=["tossup", "lean", "likely", "safe"])
    m["out_won"] = np.where(m["op"] > 0, m["y"], 1 - m["y"])
    m["p_out"] = np.where(m["op"] > 0, m["p"], 1 - m["p"])
    return m.groupby(["year", "fav", "bucket"], observed=True).agg(n=("y", "size"), out_won=("out_won", "mean"),
                                                                   model_p_out=("p_out", "mean")).reset_index()


def main() -> None:
    df = training_table()
    cfg = chosen_config()
    o = oof(df, cfg)
    cal = calibration(o)
    cal.to_csv(OUT / "analysis_calibration.csv", index=False)
    t1, t2 = upsets(o)
    t1.to_csv(OUT / "analysis_upsets.csv", index=False)
    t2.to_csv(OUT / "analysis_upset_logit.csv", index=False)
    mt = midterm_table(o)
    mt.to_csv(OUT / "analysis_midterm.csv", index=False)
    print(mt.round(3).to_string())
    imp = importance(df, cfg)
    imp.to_csv(OUT / "analysis_importance.csv", index=False)
    pd.set_option("display.width", 200)
    print(cal.round(3).to_string(), "\n", t1.round(3).to_string(), "\n", t2.round(3).to_string(), "\n",
          imp.round(4).to_string())


if __name__ == "__main__":
    main()
