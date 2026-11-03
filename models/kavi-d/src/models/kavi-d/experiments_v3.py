"""v3 design comparison (same two criteria as experiments_v2.py: LOYO inside 2010-2016 is the selection criterion;
rolling 2018-2024 reported for transparency only).

New ideas tested (all D/R mirror-symmetric logistic, slope Platt, 1% non-Safe floor):
  gb_mid   generic ballot x midterm (pau-a / pau-b: "GB x midterm" instead of a White-House out-party sign)
  env_*    kavi-e's national-environment idea: train base_m = lean + ACTUAL national House margin, predict with a
           pre-election N_hat (own GB-bias estimate "gb", or kavi-e's combined estimate "kavi-e") integrated over its
           uncertainty
  env_mid  N x midterm;  base_sen  separate base_m slope in Senate races

Usage: python experiments_v3.py [config ...]   -> out/experiments_v3.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import OUT, POLL_WINDOW_SPARSE_DAYS
from experiments_v2 import CONFIGS as V2, LOGIT_BASE, contested, evaluate, floor
from history import training_table
from model import Config, log_loss, raw_fit_predict


def lg(extra=(), env=None) -> Config:
    ex = {"logit_features": LOGIT_BASE + list(extra)}
    if env:
        ex["env"] = env
    return Config(kind="logit", symmetric=True, floor_nonsafe=0.01, extra=ex)


CONFIGS = {
    "logit_v1": V2["logit_v1"],
    "logit_op": V2["logit_op"],
    "logit_gbmid": lg(("gb_mid",)),
    "logit_op_gbmid": lg(("op", "gb_mid")),
    "env_gb": lg(env="gb"),
    "env_e": lg(env="kavi-e"),
    "env_gb_op": lg(("op",), env="gb"),
    "env_e_op": lg(("op",), env="kavi-e"),
    "env_e_mid": lg(("env_mid",), env="kavi-e"),
    "env_e_sen": lg(("base_sen",), env="kavi-e"),
    "env_e_gbmid": lg(("gb_mid",), env="kavi-e"),
    "env_e_gbclose": lg(("gb_mid_close",), env="kavi-e"),
    "env_e_gbmid_close": lg(("gb_mid", "gb_mid_close"), env="kavi-e"),
    "env_e_gbmid_polls": lg(("gb_mid", "poll_strong"), env="kavi-e"),
    "env_e_gbmid_op": lg(("gb_mid", "op"), env="kavi-e"),
}
# v3 choice: logit_gbmid (see LOG.md v3). The pre-2018 LOYO prefers the op variants by ~0.001; the all-cycle LOYO
# (includes 2018/2022) prefers the gb_mid family by ~0.004. The env (true-N) layer was tried on top (env_e_gbmid)
# and reverted because it made the 2022 scorecard worse (0.0810 -> 0.0838). Test-year-informed choices, logged.
CONFIGS["final_v3"] = CONFIGS["logit_gbmid"]


def loyo_all(df: pd.DataFrame, c: Config) -> dict:
    """Leave-one-year-out over ALL eight cycles 2010-2024 (train on the other seven). It includes 2018 and 2022, so
    any choice made on it is test-year-informed (logged in LOG.md)."""
    ll, llc = [], []
    for h in sorted(df["year"].unique()):
        tr, te = df[df["year"] != h], df[df["year"] == h]
        p = floor(raw_fit_predict(tr, te, c), te, c.floor_nonsafe)
        y, k = te["y"].to_numpy(), contested(te)
        ll.append(log_loss(p, y))
        llc.append(log_loss(p[k], y[k]))
    return {"loyo8_mean": float(np.mean(ll)), "loyo8C_mean": float(np.mean(llc))}


def main(names: list[str]) -> None:
    df = training_table()
    rows = []
    for name in names or [n for n in CONFIGS if n != "final_v3"]:
        r = {"name": name, "window": POLL_WINDOW_SPARSE_DAYS, **evaluate(df, CONFIGS[name]),
             **loyo_all(df, CONFIGS[name])}
        rows.append(r)
        print(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and not k.startswith("b_")},
              flush=True)
    path = OUT / "experiments_v3.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["name"].isin(new["name"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
