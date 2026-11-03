"""Diagnostic: our model vs a ratings-only logistic baseline and a no-poll (prior-only) variant, every year
2014-2024 out of sample (fit on earlier years only), contested races, scored on who won.

Usage: python src/models/kavi-c/baselines.py  -> out/baselines.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import model  # noqa: E402
from common import HIST_YEARS, OUT  # noqa: E402


def ll(q: np.ndarray, won: np.ndarray) -> float:
    q = np.clip(q, 1e-3, 1 - 1e-3)
    return float(-np.mean(won * np.log(q) + (1 - won) * np.log(1 - q)))


def main() -> int:
    rows = []
    for y in HIST_YEARS[2:]:
        tr = dataset.training([t for t in HIST_YEARS if t < y])
        te = dataset.year_table(y)
        te = te[te["contested"]].reset_index(drop=True)
        won = (te["winner_party"] == "DEM").to_numpy(float)
        m = model.fit(tr)
        q_full = m.predict(te)["qA"].to_numpy()
        no_poll = te.assign(poll_share=np.nan, poll_neff=np.nan)
        q_prior = m.predict(no_poll)["qA"].to_numpy()
        lr = LogisticRegression(C=10.0).fit(tr[["rating"]], (tr["share"] > 0.5).astype(int))
        q_rat = lr.predict_proba(te[["rating"]])[:, 1]
        comp = (te["rating"].abs() <= 1.5).to_numpy()
        rows.append({"year": y, "n": len(te), "n_competitive": int(comp.sum()),
                     "ll_model": ll(q_full, won), "ll_prior_only": ll(q_prior, won), "ll_ratings_logit": ll(q_rat, won),
                     "ll_model_comp": ll(q_full[comp], won[comp]), "ll_prior_only_comp": ll(q_prior[comp], won[comp]),
                     "ll_ratings_logit_comp": ll(q_rat[comp], won[comp])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "baselines.csv", index=False)
    print(out.round(4).to_string())
    print(out.mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
