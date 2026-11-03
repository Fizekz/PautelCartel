"""Unit tests for the pooling and the expected-value pick rule."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "final"))
from ensemble import pool  # noqa: E402
from make_submission import ev_picks  # noqa: E402


def frame(rows):
    return pd.DataFrame(rows, columns=["race_id", "candidate", "party", "q", "p_kalshi"])


def test_pool_is_renormalised_geometric_mean():
    m1 = frame([("X-01", "A", "DEM", 0.8, 0.7), ("X-01", "B", "REP", 0.2, 0.3)])
    m2 = frame([("X-01", "A", "DEM", 0.5, 0.7), ("X-01", "B", "REP", 0.5, 0.3)])
    out = pool([m1, m2]).set_index("candidate")["q"]
    a, b = np.sqrt(0.8 * 0.5), np.sqrt(0.2 * 0.5)
    assert out["A"] == pytest.approx(a / (a + b))
    assert out.sum() == pytest.approx(1.0)


def test_ev_rule_takes_underdog_when_q_times_points_is_higher():
    # Favourite: 0.6 * -log2(0.8) = 0.193; underdog: 0.4 * -log2(0.2) = 0.929 -> underdog.
    pooled = frame([("X-01", "Fav", "DEM", 0.6, 0.8), ("X-01", "Dog", "REP", 0.4, 0.2)]).assign(chamber="House")
    pick = ev_picks(pooled).iloc[0]
    assert pick["candidate"] == "Dog" and bool(pick["is_underdog"])
    assert pick["ev"] == pytest.approx(0.4 * -np.log2(0.2))


def test_ev_rule_keeps_favourite_when_underdog_q_is_tiny():
    # Underdog EV 0.001 * 6.64 < favourite EV 0.999 * 0.0145.
    pooled = frame([("X-01", "Fav", "DEM", 0.999, 0.99), ("X-01", "Dog", "REP", 0.001, 0.01)]).assign(chamber="House")
    assert ev_picks(pooled).iloc[0]["candidate"] == "Fav"


def test_cprime_keeps_favourite_when_underdog_q_is_below_its_price():
    # EV prefers the dog (0.10 * 3.32 = 0.33 > 0.90 * 0.152 = 0.14), but q 0.10 < Kalshi 0.12 -> C' keeps the favourite.
    pooled = frame([("X-01", "Fav", "DEM", 0.90, 0.88), ("X-01", "Dog", "REP", 0.10, 0.12)]).assign(chamber="House")
    assert ev_picks(pooled).iloc[0]["candidate"] == "Dog"
    pick = ev_picks(pooled, dog_needs_q_at_price=True).iloc[0]
    assert pick["candidate"] == "Fav" and not bool(pick["is_underdog"])
    assert pick["ev"] == pytest.approx(0.90 * -np.log2(0.88))


def test_cprime_takes_underdog_when_q_is_at_least_its_price():
    pooled = frame([("X-01", "Fav", "DEM", 0.70, 0.80), ("X-01", "Dog", "REP", 0.30, 0.20)]).assign(chamber="House")
    assert ev_picks(pooled, dog_needs_q_at_price=True).iloc[0]["candidate"] == "Dog"


def test_forced_favourite_race_keeps_the_kalshi_favourite():
    pooled = frame([("X-01", "Fav", "DEM", 0.40, 0.80), ("X-01", "Dog", "REP", 0.60, 0.20),
                    ("X-02", "Fav", "DEM", 0.40, 0.80), ("X-02", "Dog", "REP", 0.60, 0.20)]).assign(chamber="House")
    picks = ev_picks(pooled, dog_needs_q_at_price=True, force_favourite={"X-01"}).set_index("race_id")
    assert picks.at["X-01", "candidate"] == "Fav" and picks.at["X-02", "candidate"] == "Dog"


def test_withdrawn_candidate_is_never_picked():
    pooled = frame([("NE-SEN", "Withdrawn Dem", "DEM", 0.05, 0.004), ("NE-SEN", "Ricketts", "REP", 0.95, 0.996)])
    assert ev_picks(pooled.assign(chamber="Senate")).iloc[0]["candidate"] == "Ricketts"


def test_missing_kalshi_price_is_an_error():
    pooled = frame([("X-01", "A", "DEM", 0.5, 0.5), ("X-01", "B", "REP", 0.5, np.nan)]).assign(chamber="House")
    with pytest.raises(SystemExit):
        ev_picks(pooled)
