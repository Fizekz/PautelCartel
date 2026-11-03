"""Pool several models' win probabilities into one forecast (log-linear pooling).

For each candidate i in a race, with model probabilities q_i^(1..M):
    q_i = exp(mean_m ln q_i^(m)) / sum_j exp(mean_m ln q_j^(m))
i.e. the geometric mean of the models' probabilities, renormalised so each race sums to 1. Each model's q is
clipped to [1e-4, 1 - 1e-4] first so one model's 0 cannot veto a candidate. A candidate a model leaves out is
pooled over the models that do list it.

Usage:
    python src/final/ensemble.py --out pooled.csv MODEL1.csv MODEL2.csv ...
Inputs: race_id,candidate,party,q (+ any other columns; the first file's extra columns are kept).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Q_CLIP = 1e-4
JOIN = ["race_id", "candidate"]


def read_model(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"race_id", "candidate", "party", "q"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    # Same candidate, different whitespace across models (e.g. TX-01 "Yolanda R.  Prince") must join as one.
    df = df.assign(candidate=df["candidate"].astype(str).str.split().str.join(" "))
    if df.duplicated(JOIN).any():
        raise SystemExit(f"{path}: duplicate (race_id, candidate) rows")
    return df


def pool(models: list[pd.DataFrame]) -> pd.DataFrame:
    """Log-linear pool of the models' q, on the union of their (race_id, candidate) rows."""
    keys = pd.concat([m[JOIN + ["party"]] for m in models]).drop_duplicates(JOIN)
    extra = [c for c in models[0].columns if c not in ("q", "party", *JOIN)]
    base = keys.merge(models[0][JOIN + extra], on=JOIN, how="left")
    logs = pd.concat(
        [base[JOIN].merge(m[JOIN + ["q"]], on=JOIN, how="left")["q"].clip(Q_CLIP, 1 - Q_CLIP).map(np.log)
         for m in models],
        axis=1,
    )
    geo = np.exp(logs.mean(axis=1, skipna=True))
    return base.assign(q=geo / geo.groupby(base["race_id"]).transform("sum"), n_models=logs.notna().sum(axis=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pooled = pool([read_model(p) for p in args.models])
    pooled.to_csv(args.out, index=False)
    print(f"pooled {len(args.models)} models: {pooled['race_id'].nunique()} races, {len(pooled)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
