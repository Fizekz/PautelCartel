"""v3 design check: pool the model's q with a ratings-only logistic, weight chosen on pre-2018 LOYO.

For each held-out pre-2018 year, both the model and the ratings logistic are fitted on the other three pre-2018 years;
the pooled q is logit^-1((1 - w) logit q_model + w logit q_rating). The weight w is picked by mean held-out log loss
over 2010/12/14/16 (contested trainable races, the v2 selection metric). Forward years 2018-2024 are printed for
information only.

Usage: python src/models/kavi-c/pool_experiment.py -> out/pool_experiment_v3.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import model  # noqa: E402
import pool  # noqa: E402
from common import HIST_YEARS, OUT  # noqa: E402

PRE = [2010, 2012, 2014, 2016]
FEATS = ("rating",)
W_GRID = np.round(np.arange(0.0, 0.61, 0.05), 2)


def log_loss(q: np.ndarray, won: np.ndarray) -> float:
    q = np.clip(q, 1e-3, 1 - 1e-3)
    return float(-np.mean(won * np.log(q) + (1 - won) * np.log(1 - q)))


def held_out(train_years: list[int], year: int) -> pd.DataFrame:
    tr = dataset.training(train_years)
    te = dataset.year_table(year)
    te = te[te["trainable"]].reset_index(drop=True)
    q_m = model.fit(tr).predict(te)["qA"].to_numpy()
    q_r = pool.RatingLogit.fit(tr, FEATS).predict(te)
    return pd.DataFrame({"year": year, "q_model": q_m, "q_rating": q_r, "won": (te["share"] > 0.5).to_numpy(float),
                         "comp": (te["rating"].abs() <= 1.5).to_numpy()})


def main() -> int:
    global FEATS
    if len(sys.argv) > 1:
        FEATS = tuple(sys.argv[1].split(","))
    print("ratings-logit features:", FEATS)
    parts = [held_out([t for t in PRE if t != y], y).assign(mode="loyo_pre2018") for y in PRE]
    parts += [held_out([t for t in HIST_YEARS if t < y], y).assign(mode="forward") for y in (2018, 2020, 2022, 2024)]
    d = pd.concat(parts, ignore_index=True)
    rows = []
    for w in W_GRID:
        q = pool.log_pool(d["q_model"].to_numpy(), d["q_rating"].to_numpy(), w)
        for (mode, y), g in d.assign(q=q).groupby(["mode", "year"]):
            rows.append({"w": w, "mode": mode, "year": y, "ll": log_loss(g["q"].to_numpy(), g["won"].to_numpy()),
                         "ll_comp": log_loss(g["q"][g["comp"]].to_numpy(), g["won"][g["comp"]].to_numpy())})
    out = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    out.assign(features="+".join(FEATS)).to_csv(OUT / f"pool_experiment_v3_{'_'.join(FEATS)}.csv", index=False)
    piv = out.pivot_table(index="w", columns=["mode", "year"], values="ll").round(4)
    piv[("mean", "loyo_pre2018")] = out[out["mode"] == "loyo_pre2018"].groupby("w")["ll"].mean().round(4)
    piv[("mean", "forward")] = out[out["mode"] == "forward"].groupby("w")["ll"].mean().round(4)
    print(piv.to_string())
    best = piv[("mean", "loyo_pre2018")].idxmin()
    print("w chosen on pre-2018 LOYO:", best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
