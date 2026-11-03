"""Invariant checks on the 2026 and backtest outputs. Usage: python check_outputs.py (exit 1 on failure)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from common import KALSHI_RACES, OUT, TEMPLATE


def main() -> int:
    errors = []
    pred = pd.read_csv(OUT / "pred_2026.csv")
    tmpl = set(pd.read_csv(TEMPLATE)["race_id"])
    if set(pred["race_id"]) != tmpl:
        errors.append("pred_2026 race coverage differs from the template")
    n_slots = sum(len(str(c).split(" | ")) for c in pd.read_csv(KALSHI_RACES)["candidates"])
    if len(pred) != n_slots:
        errors.append(f"pred_2026 has {len(pred)} rows, Kalshi has {n_slots} slots")
    if not np.allclose(pred.groupby("race_id")["q"].sum(), 1.0):
        errors.append("q does not sum to 1 in every race")
    if pred["q"].lt(0).any() or pred["q"].isna().any() or pred["p_kalshi"].isna().any():
        errors.append("negative / missing q or p_kalshi")
    if (pred["party"] == "UNK").any():
        errors.append("UNK party labels remain")
    if pred.duplicated(["race_id", "candidate"]).any():
        errors.append("duplicate candidates")
    for y in (2018, 2022):
        b = pd.read_csv(OUT / f"backtest_{y}.csv")
        if not np.allclose(b.groupby("race_id")["q"].sum(), 1.0):
            errors.append(f"backtest {y}: q does not sum to 1")
    sims = json.loads((OUT / "sims_summary.json").read_text())
    if sims["n_sims"] < 10000:
        errors.append("fewer than 10k simulations")
    print("OK" if not errors else "\n".join(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
