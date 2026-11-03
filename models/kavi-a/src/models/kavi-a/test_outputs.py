"""Invariant tests for the kavi-a outputs (run after the pipeline): pytest src/models/kavi-a/test_outputs.py"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import BENCH, OUT, TEMPLATE, KALSHI_RACES
from inputs_2026 import kalshi_slots
from races import intra_party_probs


def test_backtest_q_sums_to_one_and_covers_benchmark():
    for year in (2018, 2022):
        pred = pd.read_csv(OUT / f"backtest_{year}.csv")
        bench = pd.read_csv(BENCH / f"fte_final_{year}.csv")
        assert np.allclose(pred.groupby("race_id")["q"].sum(), 1.0, atol=1e-6)
        assert set(pred["race_id"]) == set(bench["race_id"])
        assert pred["q"].between(0, 1).all()


def test_pred_2026_covers_template_and_kalshi_slots():
    pred = pd.read_csv(OUT / "pred_2026.csv")
    assert set(pd.read_csv(TEMPLATE)["race_id"]) <= set(pred["race_id"])
    assert np.allclose(pred.groupby("race_id")["q"].sum(), 1.0, atol=1e-4)
    slots = kalshi_slots()
    merged = slots.merge(pred, on=["race_id", "candidate"], how="left")
    assert merged["q"].notna().all()
    assert np.allclose(merged["p_kalshi_x"], merged["p_kalshi_y"], atol=1e-4)


def test_sims_summary_shape():
    s = json.loads((OUT / "sims_summary.json").read_text())
    assert s["n_sims"] >= 10000
    assert 0 <= s["house"]["p_dem_majority"] <= 1
    assert sum(s["house"]["dem_seat_histogram"].values()) == s["n_sims"]


def test_intra_party_prefers_incumbent_with_equal_money():
    sub = pd.DataFrame({"is_inc": [1, 0], "receipts": [1e6, 1e6]})
    p = intra_party_probs(sub)
    assert p[0] > 0.8 and abs(p.sum() - 1) < 1e-9


def test_kalshi_file_exists():
    assert KALSHI_RACES.exists()


def test_pred_2026_has_real_parties():
    pred = pd.read_csv(OUT / "pred_2026.csv")
    assert not (pred["party"] == "UNK").any()
    ca_dd = pred[pred["race_id"].isin(["CA-04", "CA-07", "CA-11", "CA-12", "CA-14", "CA-29", "CA-34", "CA-37"])]
    assert (ca_dd["party"] == "DEM").all()
    assert (pred.loc[pred["race_id"] == "CA-40", "party"] == "REP").all()


def test_dossier_adjustment_matched_every_race():
    log = pd.read_csv(OUT / "dossier_adjustments_2026.csv")
    assert len(log) == 85
    assert set(log["status"]) <= {"two_party", "same_party"}


def test_backtest_receipts_are_pre_election():
    from features import fec_lookup
    for year in (2018, 2022):
        full, pre = fec_lookup(), fec_lookup(pre_year=year)
        m = full["year"] == year
        # a pre-election cut can only be smaller than the two-year total, up to amended-report noise
        assert (pre.loc[m, "receipts"] <= full.loc[m, "receipts"] * 1.05 + 1e4).mean() > 0.98
        assert (pre.loc[~m, "receipts"] == full.loc[~m, "receipts"]).all()


def test_same_party_fit_is_symmetric_and_leader_favoured():
    """v3: the fitted top-two model gives the primary leader > 0.5 and A/B labels do not matter."""
    from same_party_fit import fit, prob_a
    params = fit(tuple(range(2012, 2017, 2)))
    assert params["n"] >= 20 and params["b_share"] > 0
    p = prob_a(params, 0.6, 0, 0)
    assert p > 0.5
    assert abs(p + prob_a(params, 0.4, 0, 0) - 1) < 1e-12


def test_same_surname_candidates_get_their_own_fec_rows():
    """v3: WA-09 2018 Adam Smith (incumbent) vs Sarah Smith used to share one FEC row (both incumbents)."""
    b = pd.read_csv(OUT / "backtest_2018.csv")
    q = b[b["race_id"] == "WA-09"].set_index("candidate")["q"]
    assert q["Adam Smith"] > 0.7
