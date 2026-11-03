"""Rolling-origin comparison of model configurations on contested 2014-2024 races (train on earlier cycles only).

Usage: python experiments.py            -> table of log loss per validation year for each configuration
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import OUT
from history import training_table
from model import GBM_FEATURES, Config, fit_predict, log_loss

VAL_YEARS = [2014, 2016, 2018, 2020, 2022, 2024]

NO_RATINGS = tuple(f for f in GBM_FEATURES if not f.startswith("r_") and f != "imputed_safe")
CORE = ("r_mean", "base_m", "inc", "fec_lr", "poll_m", "poll_n", "prev_rel", "senate")
CORE_NOFEC = tuple(f for f in CORE if not f.startswith("fec"))
CONFIGS = {
    "final": Config(kind="blend", features=CORE_NOFEC, symmetric=True, floor_nonsafe=0.01,
                    extra={"logit_features": ["r_mean", "base_m", "inc", "poll_m", "prev_rel"]}),
    "blend_core_sym_nofec": Config(kind="blend", features=CORE_NOFEC, symmetric=True,
                                   extra={"logit_features": ["r_mean", "base_m", "inc", "poll_m", "prev_rel"]}),
    "logit_sym_nofec": Config(kind="logit", symmetric=True,
                              extra={"logit_features": ["r_mean", "base_m", "inc", "poll_m", "prev_rel"]}),
    "gbm_icpt": Config(platt_intercept=True),
    "gbm_sym": Config(symmetric=True),
    "gbm_core_sym": Config(features=CORE, symmetric=True),
    "logit_sym": Config(kind="logit", symmetric=True),
    "blend_sym": Config(kind="blend", symmetric=True),
    "blend_core_sym": Config(kind="blend", features=CORE, symmetric=True),
    "gbm": Config(),
    "gbm_nocal": Config(calibrate=False),
    "gbm_d2": Config(max_depth=2, max_iter=400),
    "gbm_slow": Config(learning_rate=0.015, max_iter=500),
    "gbm_leaf50": Config(min_samples_leaf=50),
    "gbm_noratings": Config(features=NO_RATINGS),
    "gbm_core": Config(features=("r_mean", "base_m", "inc", "fec_lr", "poll_m", "poll_n", "prev_rel", "senate")),
    "logit": Config(kind="logit"),
    "logit_nocal": Config(kind="logit", calibrate=False),
    "margin": Config(kind="margin", calibrate=True),
    "blend": Config(kind="blend"),
}


def evaluate(df: pd.DataFrame, cfg: Config) -> dict:
    out = {}
    for y in VAL_YEARS:
        te = df[df["year"] == y]
        p, _ = fit_predict(df, te, cfg, y)
        out[y] = log_loss(p, te["y"].to_numpy())
    return out


def main(names: list[str]) -> None:
    df = training_table()
    rows = []
    for name in names or CONFIGS:
        r = evaluate(df, CONFIGS[name])
        r["name"] = name
        r["pre2018_mean"] = np.mean([r[2014], r[2016]])
        r["all_mean"] = np.mean([r[y] for y in VAL_YEARS])
        rows.append(r)
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    path = OUT / "experiments_rolling.csv"
    new = pd.DataFrame(rows)
    if path.exists():  # merge: keep earlier configs, replace re-run ones
        old = pd.read_csv(path)
        old.columns = [int(c) if c.isdigit() else c for c in old.columns]
        new = pd.concat([old[~old["name"].isin(new["name"])], new], ignore_index=True)
    new.to_csv(path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
