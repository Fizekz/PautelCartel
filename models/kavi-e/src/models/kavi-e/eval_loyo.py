"""v2 selection harness: leave-one-year-out over the PRE-2018 years only (2010, 2012, 2014, 2016).

For each held-out year Y the full model (stage A, k, s_p, tau) is fitted on the other pre-2018 years and Y is
predicted with Y's pre-election N_hat. Scores = mean log loss of P(D side wins) on contested D-v-R races, House and
Senate, all years and midterms only (2010, 2014). Used to choose v2 design options without touching 2018/2022.
Run: .venv/bin/python src/models/kavi-e/eval_loyo.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import model
from backtest import load_frames

YEARS = (2010, 2012, 2014, 2016)


def loyo(frame: pd.DataFrame, features: tuple[str, ...], years=YEARS, clf_extra: pd.DataFrame | None = None) -> pd.DataFrame:
    import national
    frame = frame[frame["year"].isin(years)]
    parts = []
    for y in years:
        extra = None if clf_extra is None else clf_extra[clf_extra["year"].isin(years) & (clf_extra["year"] != y)]
        f = model.fit(frame[frame["year"] != y], features, clf_extra=extra)
        env = national.estimate(y)
        tgt = frame[(frame["year"] == y) & frame["contested"] & frame["L"].notna()]
        parts.append(model.predict(f, tgt, env["n_hat"], env["sigma_n"]))
    return pd.concat(parts)


def score(pred: pd.DataFrame) -> dict:
    d = (pred["winner_party"] == "DEM").to_numpy(int)
    ll = lambda m: model.log_loss_d(pred["p_d"].to_numpy()[m], d[m])  # noqa: E731
    mid = pred["pres_year"].to_numpy() == 0
    close = (np.abs(pred["rating"].fillna(3)) < 2.5).to_numpy()
    return {"all": round(ll(np.ones(len(d), bool)), 4), "mid": round(ll(mid), 4),
            "close": round(ll(close), 4), "close_mid": round(ll(close & mid), 4), "n": len(d)}


def main() -> int:
    house, senate = load_frames()
    base_h, base_s = model.HOUSE_FEATURES, model.SENATE_FEATURES
    variants = {
        "base": ((), ()),
        "out_mid": (("out_mid",), ("out_mid",)),
        "out_mid_close": (("out_mid_close",), ("out_mid_close",)),
        "out_mid_r": (("out_mid_r",), ("out_mid_r",)),
        "out_mid+close": (("out_mid", "out_mid_close"), ("out_mid", "out_mid_close")),
        "out_mid+rating_mid": (("out_mid", "rating_mid"), ("out_mid", "rating_mid")),
        "rgap": (("rgap",), ("rgap",)),
        "inc_out_mid": (("inc_out_mid",), ("inc_out_mid",)),
        "rgap_house_only": (("rgap",), ()),
    }
    for name, (eh, es) in variants.items():
        ph = loyo(house, base_h + eh)
        ps = loyo(senate, base_s + es)
        print(f"{name:20s} House {score(ph)}  Senate {score(ps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
