"""v3: national environment N (national House D-R vote margin, points) for the party model.

Idea borrowed from kavi-e (METHODS section 3): train on the ACTUAL national House margin of each training cycle, and
at forecast time use a pre-election estimate N_hat with its uncertainty, integrated out (Gauss-Hermite).

  N_true(Y)  national_hist.csv house_margin_dr (MIT House vote), training cycles only.
  N_hat(Y)   source "gb": my own estimate = final generic ballot - shrunk mean generic-ballot error of EARLIER cycles
             (e = GB - actual; bias = mean(e) n/(n+2); sd^2 = (2*3^2 + sum (e-bias)^2)/(2+n), floored at 2.5), the
             same shrinkage rule kavi-e uses for its gb signal.
             source "kavi-e": kavi-e's combined N_hat (generic ballot + special elections + off-year governors +
             fundamentals, each signal's bias/sd from earlier cycles only) for the midterms 2010/2014/2018/2022/2026
             (borrowed: data/extra/kavi-e_national_env.json); presidential cycles fall back to "gb".
Every N_hat(Y) uses only information from before cycle Y's election (strictly earlier GB errors), so it is valid in
the leave-one-year-out and rolling designs as well as in the backtests.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from common import CLEAN, ROOT, generic_margin

KAVI_E_ENV = ROOT / "data" / "extra" / "kavi-e_national_env.json"
PRIOR_SD = 3.0
SD_FLOOR = 2.5
N_NODES = 9


@lru_cache(maxsize=None)
def _hist() -> pd.DataFrame:
    return pd.read_csv(CLEAN / "national_hist.csv").set_index("year")


def n_true(year: int) -> float:
    """Actual national House D-R margin of a completed cycle (training rows only)."""
    return float(_hist().loc[year, "house_margin_dr"])


def _gb_estimate(year: int) -> tuple[float, float]:
    h = _hist()
    earlier = h[h.index < year]
    e = (earlier["generic_margin_dr"] - earlier["house_margin_dr"]).to_numpy()
    n = len(e)
    bias = float(e.mean() * n / (n + 2)) if n else 0.0
    sd = float(np.sqrt((2 * PRIOR_SD ** 2 + ((e - bias) ** 2).sum()) / (2 + n)))
    return generic_margin(year) - bias, max(sd, SD_FLOOR)


@lru_cache(maxsize=None)
def _kavi_e() -> dict:
    return json.loads(KAVI_E_ENV.read_text())


@lru_cache(maxsize=None)
def n_hat(year: int, source: str = "gb") -> tuple[float, float]:
    """(N_hat, sd) for cycle `year`, using only pre-election information."""
    if source == "kavi-e" and str(year) in _kavi_e():
        r = _kavi_e()[str(year)]
        return float(r["n_hat"]), max(float(r["sigma_n"]), SD_FLOOR)
    if source not in ("gb", "kavi-e"):
        raise ValueError(f"unknown N_hat source {source!r}")
    return _gb_estimate(year)


def with_true_env(train: pd.DataFrame) -> pd.DataFrame:
    """Training rows: base_m = lean_m + actual national House margin of the row's cycle."""
    n = train["year"].map(lambda y: n_true(int(y)))
    return train.assign(base_m=train["lean_m"] + n, env_n=n)


def env_draws(test: pd.DataFrame, source: str) -> list[tuple[float, pd.DataFrame]]:
    """Test rows at each Gauss-Hermite node of N ~ Normal(N_hat, sd): [(weight, frame with base_m set)]."""
    est = test["year"].map(lambda y: n_hat(int(y), source))
    mu = est.map(lambda t: t[0]).to_numpy()
    sd = est.map(lambda t: t[1]).to_numpy()
    z, w = np.polynomial.hermite_e.hermegauss(N_NODES)
    w = w / w.sum()
    return [(float(wk), test.assign(base_m=test["lean_m"].to_numpy() + mu + sd * zk, env_n=mu + sd * zk))
            for zk, wk in zip(z, w)]


if __name__ == "__main__":
    for y in [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026]:
        g, e = n_hat(y, "gb"), n_hat(y, "kavi-e")
        act = n_true(y) if y < 2026 else float("nan")
        print(y, f"gb {g[0]:+.2f} sd {g[1]:.2f}  kavi-e {e[0]:+.2f} sd {e[1]:.2f}  actual {act:+.2f}")
