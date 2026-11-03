"""Correlated simulation of the 2026 House and Senate from the fitted error model.

Each partisan race's side-A share = mu + national error (shared by every race) + state error (shared by the
races of one state, House and Senate) + local error. The scales come from the model: sd_nat, and the local sd
split into a state part (state_frac of its variance, estimated from 2010-2024 residuals) and an idiosyncratic
part. National and local draws are Student-t with the model's df; state draws are normal.
Same-party finals and one-party races are independent draws with their q.

Usage: python src/models/kavi-c/simulate.py [--n 20000]  -> out/sims_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402

SEED = 20261103
SENATE_HOLDOVER = {"DEM": 34, "REP": 31}   # 47 D-caucus / 53 R now, minus the 13 D and 22 R seats up in 2026
HOUSE_MAJORITY = 218
SENATE_MAJORITY = 51                        # VP is a Republican, so 50 is not D control


def side_parties(det: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    a = np.where(det["ind_side_a"], "IND", "DEM")
    b = np.where(det["ind_side_b"], "IND", "REP")
    return a, b


def summarise(seats: np.ndarray, majority: int) -> dict:
    vals, cnt = np.unique(seats, return_counts=True)
    return {"mean": float(seats.mean()), "median": float(np.median(seats)),
            "p10": float(np.percentile(seats, 10)), "p90": float(np.percentile(seats, 90)),
            "p05": float(np.percentile(seats, 5)), "p95": float(np.percentile(seats, 95)),
            "p_majority": float((seats >= majority).mean()),
            "distribution": {int(v): float(c / len(seats)) for v, c in zip(vals, cnt)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    det = pd.read_csv(OUT / "race_detail_2026.csv")
    pred = pd.read_csv(OUT / "pred_2026.csv")
    diag = json.loads((OUT / "model_2026_diag.json").read_text())
    nu, frac = float(diag["nu"]), float(diag["state_frac"])
    rng = np.random.default_rng(SEED)
    n = args.n
    partisan = ~(det["non_partisan_final"] | det["uncontested_d"]).to_numpy()
    states = det["state"].to_numpy()
    st_index = {s: i for i, s in enumerate(sorted(set(states)))}
    st = np.array([st_index[s] for s in states])

    z_nat = rng.standard_t(nu, size=(n, 1))
    z_state = rng.standard_normal(size=(n, len(st_index)))[:, st]
    z_loc = rng.standard_t(nu, size=(n, len(det)))
    sd_loc = det["sd_local"].to_numpy()
    share = (det["mu"].to_numpy() + det["sd_nat"].to_numpy() * z_nat
             + np.sqrt(frac) * sd_loc * z_state + np.sqrt(1 - frac) * sd_loc * z_loc)
    a_wins = share > 0.5
    pa, pb = side_parties(det)
    winner = np.where(a_wins, pa, pb).astype(object)

    # non-partisan finals and one-party races: independent draws with the published q
    for i in np.where(~partisan)[0]:
        rid = det["race_id"].iloc[i]
        g = pred[pred["race_id"] == rid]
        parties = g["party"].replace({"UNK": "REP" if rid == "CA-40" else "DEM"}).to_numpy()
        winner[:, i] = rng.choice(parties, size=n, p=g["q"].to_numpy() / g["q"].sum())

    house = (det["chamber"] == "House").to_numpy()
    sen = ~house
    out = {"n_sims": n, "seed": SEED, "nu": nu, "state_frac": frac, "sd_nat_share": float(det["sd_nat"].iloc[0])}
    h_d = (winner[:, house] == "DEM").sum(1)
    h_i = (winner[:, house] == "IND").sum(1)
    out["house"] = {"dem_seats": summarise(h_d, HOUSE_MAJORITY),
                    "rep_seats_median": float(np.median((winner[:, house] == "REP").sum(1))),
                    "ind_seats_mean": float(h_i.mean()), "p_dem_control": float((h_d >= HOUSE_MAJORITY).mean())}
    s_d = (winner[:, sen] == "DEM").sum(1) + SENATE_HOLDOVER["DEM"]
    s_i = (winner[:, sen] == "IND").sum(1)
    s_r = (winner[:, sen] == "REP").sum(1) + SENATE_HOLDOVER["REP"]
    out["senate"] = {"dem_caucus_seats": summarise(s_d, SENATE_MAJORITY),
                     "rep_seats": summarise(s_r, SENATE_MAJORITY),
                     "ind_new_seats_mean": float(s_i.mean()),
                     "p_dem_control": float((s_d >= SENATE_MAJORITY).mean()),
                     "p_dem_plus_new_ind_51": float((s_d + s_i >= SENATE_MAJORITY).mean()),
                     "p_rep_control": float((s_r >= 50).mean()),
                     "holdovers": SENATE_HOLDOVER}
    out["p_dem_both"] = float(((h_d >= HOUSE_MAJORITY) & (s_d >= SENATE_MAJORITY)).mean())
    # consistency: simulated side-A win rate vs the analytic q used in pred_2026
    sim_q = a_wins[:, partisan].mean(0)
    out["max_abs_sim_vs_analytic_q"] = float(np.max(np.abs(sim_q - det["qA"].to_numpy()[partisan])))
    (OUT / "sims_summary.json").write_text(json.dumps(out, indent=2))
    h, s = out["house"]["dem_seats"], out["senate"]["dem_caucus_seats"]
    print(f"House D seats median {h['median']:.0f} (80% {h['p10']:.0f}-{h['p90']:.0f}), P(D control) {out['house']['p_dem_control']:.3f}")
    print(f"Senate D-caucus seats median {s['median']:.0f} (80% {s['p10']:.0f}-{s['p90']:.0f}), P(D>=51) {out['senate']['p_dem_control']:.3f}, "
          f"P(D+new IND>=51) {out['senate']['p_dem_plus_new_ind_51']:.3f}; max |sim q - q| {out['max_abs_sim_vs_analytic_q']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
