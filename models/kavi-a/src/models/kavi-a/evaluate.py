"""Quick evaluation on contested D-vs-R races of the historical table (not the official scorecard).

Leave-one-year-out within a training window, and forward tests (fit on years < Y, test on Y).
Usage: python evaluate.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT
from features import trainable
from model import fit_all, predict_chamber, win_prob

VARIANTS = {"F": (False, False, False), "F+P+R": (True, True, False), "F+P+R+C": (True, True, True),
            "F+C": (False, False, True)}


def score(hist: pd.DataFrame, train_years: list, test_year: int, polls: bool, ratings: bool, cal: bool) -> dict:
    fits = fit_all(hist, train_years, polls, ratings, calibrate=cal)
    test = hist[trainable(hist) & (hist["year"] == test_year)]
    out = {}
    for ch, fit in fits.items():
        rows = test[test["chamber"] == ch]
        pr = predict_chamber(fit, rows, rows["g"].iloc[0])
        q = win_prob(pr["mu"], pr["sd_nat"], pr["sd_state"], pr["sd_race"])
        won = (rows["dem_share_2p"] > 0.5).to_numpy()
        qw = np.clip(np.where(won, q, 1 - q), 1e-3, 1 - 1e-3)
        out[ch] = (len(rows), float(-np.log(qw).mean()), float(((q > 0.5) == won).mean()),
                   float((pr["mu"] - rows["y"] if "y" in rows else pr["mu"] - (rows["dem_share_2p"] - 0.5)).mean()))
    return out


def main() -> None:
    hist = pd.read_csv(OUT / "hist_features.csv")
    lines = []
    for name, (p, r, cal) in VARIANTS.items():
        for test_year, window in ((2014, [2010, 2012, 2016]), (2016, [2010, 2012, 2014]),
                                  (2018, [2010, 2012, 2014, 2016]), (2020, list(range(2010, 2019, 2))),
                                  (2022, list(range(2010, 2021, 2))), (2024, list(range(2010, 2023, 2)))):
            res = score(hist, window, test_year, p, r, cal)
            for ch, (n, ll, acc, bias) in res.items():
                lines.append({"variant": name, "test": test_year, "chamber": ch, "n": n, "logloss": round(ll, 4),
                              "acc": round(acc, 3), "mean_mu_err": round(bias, 4)})
    df = pd.DataFrame(lines)
    print(df.pivot_table(index=["chamber", "test"], columns="variant", values="logloss").round(4))
    print(df.pivot_table(index=["chamber", "test"], columns="variant", values="mean_mu_err").round(4))


if __name__ == "__main__":
    main()
