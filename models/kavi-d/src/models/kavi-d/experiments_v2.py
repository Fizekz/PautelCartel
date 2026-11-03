"""v2 design comparison.

Two criteria per configuration:
  loyo_pre2018: leave-one-year-out inside 2010-2016 only (train on the other three cycles, no calibration step, the
                1% non-Safe floor). This never sees 2018+ and is the primary selection criterion.
  rolling:      year Y in 2018-2024 predicted by the full pipeline (fit + slope Platt on OOF) trained on cycles < Y.
                Reported for transparency; any choice that leans on it is logged as test-year selection.

Usage: python experiments_v2.py [config ...]   -> out/experiments_v2.csv (merged by name and poll window)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import OUT, POLL_WINDOW_SPARSE_DAYS
from history import training_table
from model import Config, fit_predict, log_loss, raw_fit_predict

PRE = [2010, 2012, 2014, 2016]
ROLL = [2018, 2020, 2022, 2024]
GBM_BASE = ("r_mean", "base_m", "inc", "poll_m", "poll_n", "prev_rel", "senate")
LOGIT_BASE = ["r_mean", "base_m", "inc", "poll_m", "prev_rel"]


def cfg(gbm_extra=(), logit_extra=(), **kw) -> Config:
    return Config(kind="blend", features=GBM_BASE + tuple(gbm_extra), symmetric=True, floor_nonsafe=0.01,
                  extra={"logit_features": LOGIT_BASE + list(logit_extra)}, **kw)


CONFIGS = {
    "v1_final": cfg(),
    "op": cfg(("op",), ("op",)),
    "op_close": cfg(("op",), ("op_close",)),
    "op_both": cfg(("op",), ("op", "op_close")),
    "oth": cfg(("oth_dev",), ("oth_dev",)),
    "rcp": cfg(("rcp_dev",), ("rcp_dev",)),
    "op_oth": cfg(("op", "oth_dev"), ("op", "oth_dev")),
    "op_rcp": cfg(("op", "rcp_dev"), ("op", "rcp_dev")),
    "logit_v1": Config(kind="logit", symmetric=True, floor_nonsafe=0.01, extra={"logit_features": LOGIT_BASE}),
    "logit_op": Config(kind="logit", symmetric=True, floor_nonsafe=0.01,
                       extra={"logit_features": LOGIT_BASE + ["op"]}),
    "logit_op_rcp": Config(kind="logit", symmetric=True, floor_nonsafe=0.01,
                           extra={"logit_features": LOGIT_BASE + ["op", "rcp_dev"]}),
    "logit_oth": Config(kind="logit", symmetric=True, floor_nonsafe=0.01,
                        extra={"logit_features": LOGIT_BASE + ["oth_dev"]}),
    "logit_op_oth": Config(kind="logit", symmetric=True, floor_nonsafe=0.01,
                           extra={"logit_features": LOGIT_BASE + ["op", "oth_dev"]}),
}
# v2 choice (lowest pre-2018 LOYO log loss among the candidates; rcp_dev's extra 0.0002 not worth a feature).
CONFIGS["final_v2"] = CONFIGS["logit_op"]


def floor(p: np.ndarray, test: pd.DataFrame, f: float) -> np.ndarray:
    nonsafe = test["r_mean"].abs().fillna(0).to_numpy() < 3
    return np.where(nonsafe, np.clip(p, f, 1 - f), p) if f > 0 else p


def contested(te: pd.DataFrame) -> np.ndarray:
    return te["r_mean"].abs().fillna(3).to_numpy() < 3


def evaluate(df: pd.DataFrame, c: Config) -> dict:
    out = {}
    for h in PRE:
        tr, te = df[df["year"].isin([y for y in PRE if y != h])], df[df["year"] == h]
        p = floor(raw_fit_predict(tr, te, c), te, c.floor_nonsafe)
        y = te["y"].to_numpy()
        out[f"loyo_{h}"] = log_loss(p, y)
        k = contested(te)
        out[f"loyoC_{h}"] = log_loss(p[k], y[k])
    for yr in ROLL:
        te = df[df["year"] == yr]
        p, info = fit_predict(df, te, c, yr)
        y = te["y"].to_numpy()
        out[f"roll_{yr}"] = log_loss(p, y)
        k = contested(te)
        out[f"rollC_{yr}"] = log_loss(p[k], y[k])
        out[f"b_{yr}"] = info["b"]
    out["loyo_mean"] = np.mean([out[f"loyo_{h}"] for h in PRE])
    out["loyoC_mean"] = np.mean([out[f"loyoC_{h}"] for h in PRE])
    out["roll_mean"] = np.mean([out[f"roll_{y}"] for y in ROLL])
    return out


def main(names: list[str]) -> None:
    df = training_table()
    rows = []
    for name in names or CONFIGS:
        r = {"name": name, "window": POLL_WINDOW_SPARSE_DAYS, **evaluate(df, CONFIGS[name])}
        rows.append(r)
        print(name, POLL_WINDOW_SPARSE_DAYS, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
              flush=True)
    path = OUT / "experiments_v2.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        key = set(zip(new["name"], new["window"]))
        old = old[[(n, w) not in key for n, w in zip(old["name"], old["window"])]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
