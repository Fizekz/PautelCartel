"""v3: paired bootstrap of the log-loss difference between two backtest files (REPORT.md 8.4 item 17: no
uncertainty intervals). Per-race loss = -ln(q_winner) exactly as src/scorecard.py computes it; races resampled in
state clusters (a state's House and Senate races move together). Year-level correlation cannot be resampled with
two years, so the intervals understate the true uncertainty.
Run: .venv/bin/python src/models/kavi-e/paired_bootstrap.py out/v2 out   (dirs holding backtest_<year>.csv)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import scorecard  # noqa: E402

N_BOOT = 5000
SEED = 7
CONTESTED_FAV = 0.95


def race_losses(year: int, pred_path: Path) -> pd.DataFrame:
    bench = pd.read_csv(str(scorecard.BENCH).format(year=year))
    bench["key"] = scorecard.candidate_key(bench)
    bench["won"] = scorecard.actual_winner_flags(bench, scorecard.winners(year))
    ok = bench.groupby("race_id")["won"].sum()
    bench = bench[bench["race_id"].isin(ok[ok == 1].index)].copy()
    pred = pd.read_csv(pred_path)
    pred["key"] = scorecard.candidate_key(pred)
    m = bench.merge(pred[["race_id", "key", "q"]], on=["race_id", "key"], how="left")
    m["q"] = m["q"].fillna(0.0)
    m["q"] = m["q"] / m.groupby("race_id")["q"].transform("sum").replace(0, 1)
    fav = m.groupby("race_id")["p_fte"].transform("max")
    w = m[m["won"] == 1].assign(fav=fav[m["won"] == 1])
    return pd.DataFrame({"race_id": w["race_id"], "loss": -np.log(w["q"].clip(scorecard.Q_CLIP, 1 - scorecard.Q_CLIP)),
                         "contested": w["fav"] < CONTESTED_FAV})


def boot(diff: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    states = diff["race_id"].str[:2]
    groups = [g["d"].to_numpy() for _, g in diff.groupby(states)]
    sums = np.array([g.sum() for g in groups]); counts = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), (N_BOOT, len(groups)))
    stats = sums[idx].sum(1) / counts[idx].sum(1)
    return float(diff["d"].mean()), float(np.percentile(stats, 5)), float(np.percentile(stats, 95))


def main(a: str, b: str) -> int:
    rng = np.random.default_rng(SEED)
    for year in (2018, 2022):
        la = race_losses(year, Path(a) / f"backtest_{year}.csv")
        lb = race_losses(year, Path(b) / f"backtest_{year}.csv")
        d = la.merge(lb, on="race_id", suffixes=("_a", "_b"))
        d["d"] = d["loss_b"] - d["loss_a"]
        for name, sub in (("all", d), ("contested", d[d["contested_a"]])):
            mean, lo, hi = boot(sub, rng)
            print(f"{year} {name:9s} n={len(sub):3d} LL {sub['loss_a'].mean():.4f} -> {sub['loss_b'].mean():.4f}  "
                  f"diff {mean:+.4f} (90% state-cluster bootstrap {lo:+.4f} .. {hi:+.4f}); "
                  f"races better {int((sub['d'] < 0).sum())}, worse {int((sub['d'] > 0).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
