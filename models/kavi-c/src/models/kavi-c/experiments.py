"""Design selection for v2: leave-one-year-out (LOYO) over the pre-2018 years, plus a forward check.

Selection metric (pre-registered): mean over held-out years 2010/2012/2014/2016 of the win log loss on contested
races, each year predicted from a fit on the other three pre-2018 years only. The forward check (each of
2018-2024 predicted from earlier years) is printed for information; when a choice used it, LOG.md says so.

Usage: python src/models/kavi-c/experiments.py [--forward]  -> out/experiments_v2.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import model  # noqa: E402
import polls  # noqa: E402
from common import HIST_YEARS, OUT  # noqa: E402

PRE = [2010, 2012, 2014, 2016]
FLAGS = ("USE_ENV_MIDTERM", "USE_MID_OUT", "USE_MID_OUT_COMP", "USE_OTHER_DEV", "USE_ENV_MID_COMP")
POLL_FLAGS = ("USE_HOUSE_EFFECTS", "USE_SPONSOR_OFFSET")
V1 = dict(USE_ENV_MIDTERM=True, USE_MID_OUT=False, USE_MID_OUT_COMP=False, USE_OTHER_DEV=False,
          USE_ENV_MID_COMP=False)
# Every config starts from v1 (poll house effects and sponsor offsets on) and changes only what it names.
BASE = dict(V1, USE_HOUSE_EFFECTS=True, USE_SPONSOR_OFFSET=True)
CONFIGS = {
    "v1 (env x mid)": dict(USE_ENV_MIDTERM=True, USE_MID_OUT=False, USE_MID_OUT_COMP=False, USE_OTHER_DEV=False),
    "v1 - env x mid": dict(USE_ENV_MIDTERM=False, USE_MID_OUT=False, USE_MID_OUT_COMP=False, USE_OTHER_DEV=False),
    "+ mid_out": dict(USE_ENV_MIDTERM=True, USE_MID_OUT=True, USE_MID_OUT_COMP=False, USE_OTHER_DEV=False),
    "+ mid_out x comp": dict(USE_ENV_MIDTERM=True, USE_MID_OUT=False, USE_MID_OUT_COMP=True, USE_OTHER_DEV=False),
    "+ mid_out + mid_out x comp": dict(USE_ENV_MIDTERM=True, USE_MID_OUT=True, USE_MID_OUT_COMP=True,
                                       USE_OTHER_DEV=False),
    "mid_out, no env x mid": dict(USE_ENV_MIDTERM=False, USE_MID_OUT=True, USE_MID_OUT_COMP=False,
                                  USE_OTHER_DEV=False),
    "+ other_dev": dict(USE_ENV_MIDTERM=True, USE_MID_OUT=False, USE_MID_OUT_COMP=False, USE_OTHER_DEV=True),
    "+ mid_out + other_dev": dict(USE_ENV_MIDTERM=True, USE_MID_OUT=True, USE_MID_OUT_COMP=False,
                                  USE_OTHER_DEV=True),
    "v1, no house effects": dict(V1, USE_HOUSE_EFFECTS=False),
    "v1, no sponsor offset": dict(V1, USE_SPONSOR_OFFSET=False),
    "v1, neither poll adj": dict(V1, USE_HOUSE_EFFECTS=False, USE_SPONSOR_OFFSET=False),
    "nosp + mid_out": dict(V1, USE_SPONSOR_OFFSET=False, USE_MID_OUT=True),
    "nosp + mid_out x comp": dict(V1, USE_SPONSOR_OFFSET=False, USE_MID_OUT_COMP=True),
    "nosp + other_dev": dict(V1, USE_SPONSOR_OFFSET=False, USE_OTHER_DEV=True),
    "nosp + mid_out x comp + other_dev": dict(V1, USE_SPONSOR_OFFSET=False, USE_MID_OUT_COMP=True,
                                              USE_OTHER_DEV=True),
    "nosp - env x mid": dict(V1, USE_SPONSOR_OFFSET=False, USE_ENV_MIDTERM=False),
    "nosp + mid_out x comp - env x mid": dict(V1, USE_SPONSOR_OFFSET=False, USE_MID_OUT_COMP=True,
                                              USE_ENV_MIDTERM=False),
    # v3 candidates (v2 = nosp + mid_out x comp)
    "v3a: v2 - mid_out x comp + env x mid x comp": dict(V1, USE_SPONSOR_OFFSET=False, USE_ENV_MID_COMP=True),
    "v3b: v2 + env x mid x comp": dict(V1, USE_SPONSOR_OFFSET=False, USE_MID_OUT_COMP=True, USE_ENV_MID_COMP=True),
    "v3c: v3a - env x mid": dict(V1, USE_SPONSOR_OFFSET=False, USE_ENV_MID_COMP=True, USE_ENV_MIDTERM=False),
}


def log_loss(q: np.ndarray, won: np.ndarray) -> float:
    q = np.clip(q, 1e-3, 1 - 1e-3)
    return float(-np.mean(won * np.log(q) + (1 - won) * np.log(1 - q)))


def score(train_years: list[int], test_year: int) -> dict:
    m = model.fit(dataset.training(train_years))
    te = dataset.year_table(test_year)
    te = te[te["trainable"]].reset_index(drop=True)
    q = m.predict(te)["qA"].to_numpy()
    won = (te["share"] > 0.5).to_numpy(float)
    comp = (te["rating"].abs() <= 1.5).to_numpy()
    return {"ll": log_loss(q, won), "ll_comp": log_loss(q[comp], won[comp]), "n_comp": int(comp.sum()),
            "e_d_minus_actual_comp": float(q[comp].sum() - won[comp].sum())}


def run_config(name: str, flags: dict, forward: bool) -> list[dict]:
    saved = {f: getattr(model, f) for f in FLAGS}
    saved_p = {f: getattr(polls, f) for f in POLL_FLAGS}
    for f, v in {**BASE, **flags}.items():
        setattr(polls if f in POLL_FLAGS else model, f, v)
    dataset.poll_table.cache_clear()
    rows = []
    try:
        for y in PRE:
            rows.append({"config": name, "mode": "loyo_pre2018", "year": y,
                         **score([t for t in PRE if t != y], y)})
        if forward:
            for y in (2018, 2020, 2022, 2024):
                rows.append({"config": name, "mode": "forward", "year": y,
                             **score([t for t in HIST_YEARS if t < y], y)})
    finally:
        for f, v in saved.items():
            setattr(model, f, v)
        for f, v in saved_p.items():
            setattr(polls, f, v)
        dataset.poll_table.cache_clear()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--only", default="", help="'|'-separated config names")
    ap.add_argument("--out", default="experiments_v2.csv", help="file name under out/")
    args = ap.parse_args()
    rows = []
    only = set(args.only.split("|")) if args.only else None
    for name, flags in CONFIGS.items():
        if only and name not in only:
            continue
        rows += run_config(name, flags, args.forward)
        print(name, "done", flush=True)
    out = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / args.out, index=False)
    piv = out.pivot_table(index="config", columns=["mode", "year"], values="ll").round(4)
    piv[("mean", "loyo_pre2018")] = out[out["mode"] == "loyo_pre2018"].groupby("config")["ll"].mean().round(4)
    if args.forward:
        piv[("mean", "forward")] = out[out["mode"] == "forward"].groupby("config")["ll"].mean().round(4)
    print(piv.to_string())
    comp = out.pivot_table(index="config", columns=["mode", "year"], values="ll_comp").round(4)
    print("competitive (|rating|<=1.5):")
    print(comp.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
