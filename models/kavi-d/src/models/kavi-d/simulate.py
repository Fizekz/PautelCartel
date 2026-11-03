"""Correlated simulation of the 2026 House and Senate from out/pred_2026.csv.

Gaussian copula on each race's anti-R-vs-R outcome: X_i = sqrt(rho_n) Z_nat + sqrt(rho_s) Z_state + sqrt(1-rho_n-rho_s) e_i,
anti-R side wins when X_i < Phi^-1(q_anti_R,i). Marginals equal the model's q exactly; rho_n is estimated from the
year-to-year shifts of the rolling-origin out-of-sample predictions 2014-2024 (probit random effect), rho_s is
hand-set. Within a side (same-party races, minor slots) the candidate is drawn independently with the q split.

Usage: python simulate.py [--n 20000]  -> out/sims_summary.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from backtest import chosen_config
from common import OUT
from history import training_table
from model import fit_predict

RHO_STATE = 0.05
SENATE_NOT_UP = {"DEM": 34, "REP": 31}  # 47 D-caucus (incl. King, Sanders) - 13 up; 53 R - 22 up
HOUSE_MAJORITY, SENATE_MAJORITY = 218, 51  # VP is a Republican, so 50-50 is R control
SEED = 20261103


def year_shift_sd() -> tuple[float, dict]:
    """MLE probit shift per validation year of the calibrated rolling OOF predictions; return RMS shift."""
    df = training_table()
    cfg = chosen_config()
    shifts = {}
    for y in [2014, 2016, 2018, 2020, 2022, 2024]:
        te = df[df["year"] == y]
        p, _ = fit_predict(df, te, cfg, y)
        a = norm.ppf(np.clip(p, 1e-4, 1 - 1e-4))
        yy = te["y"].to_numpy()

        def nll(d, a=a, yy=yy):
            pp = np.clip(norm.cdf(a + d), 1e-9, 1 - 1e-9)
            return -np.sum(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

        shifts[y] = float(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)
    sd = float(np.sqrt(np.mean(np.square(list(shifts.values())))))
    return sd, shifts


def summarize(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
            "hist": {int(k): int(v) for k, v in zip(*np.unique(x, return_counts=True))}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    sd, shifts = year_shift_sd()
    rho_n = sd ** 2 / (1 + sd ** 2)
    pred = pd.read_csv(OUT / "pred_2026.csv")
    races = list(pred.groupby("race_id", sort=False))
    states = sorted({rid[:2] for rid, _ in races})
    rng = np.random.default_rng(SEED)
    n = args.n
    z_nat = rng.standard_normal(n)
    z_state = {s: rng.standard_normal(n) for s in states}
    w_e = np.sqrt(1 - rho_n - RHO_STATE)
    house_d = np.zeros(n, int)
    house_ind = np.zeros(n, int)
    senate_d = np.full(n, SENATE_NOT_UP["DEM"])
    senate_newind = np.zeros(n, int)
    for rid, g in races:
        sides = g["side"].to_numpy()
        q = g["q"].to_numpy()
        q_d = q[sides == "D"].sum() / max(q[sides != "minor"].sum(), 1e-12) if (sides != "minor").any() else 0
        x = np.sqrt(rho_n) * z_nat + np.sqrt(RHO_STATE) * z_state[rid[:2]] + w_e * rng.standard_normal(n)
        anti_r_wins = x < norm.ppf(np.clip(q_d, 1e-9, 1 - 1e-9))
        winner_party = np.empty(n, dtype=object)
        for side, mask in (("D", anti_r_wins), ("R", ~anti_r_wins)):
            idx = np.where(sides == side)[0]
            if len(idx) == 0:
                idx = np.where(sides != "minor")[0] if (sides != "minor").any() else np.arange(len(g))
            w = q[idx] / q[idx].sum()
            pick = rng.choice(idx, size=int(mask.sum()), p=w)
            winner_party[mask] = g["party"].to_numpy()[pick]
        is_d = winner_party == "DEM"
        is_i = (winner_party != "DEM") & (winner_party != "REP")
        if g["chamber"].iloc[0] == "House":
            house_d += is_d
            house_ind += is_i
        else:
            senate_d += is_d
            senate_newind += is_i
    out = {
        "n_sims": n, "seed": SEED, "rho_national": rho_n, "rho_state": RHO_STATE,
        "national_probit_shift_rms": sd, "oof_year_shifts_probit": shifts,
        "house_dem_seats": summarize(house_d), "house_ind_seats_mean": float(house_ind.mean()),
        "p_house_dem_control": float((house_d >= HOUSE_MAJORITY).mean()),
        "house_expected_dem_from_q": float(pred[pred.chamber == "House"].query("party == 'DEM'")["q"].sum()),
        "senate_dem_seats_incl_king_sanders": summarize(senate_d),
        "senate_new_independents_mean": float(senate_newind.mean()),
        "p_senate_dem_control": float((senate_d >= SENATE_MAJORITY).mean()),
        "p_senate_dem_plus_new_ind_ge_51": float((senate_d + senate_newind >= SENATE_MAJORITY).mean()),
        "notes": "Senate D count includes the 34 D-caucus seats not up (King, Sanders included). New independents "
                 "(Osborn, Bodnar, Achilles, Bengs) counted separately. House D excludes independents (Kiley, Hill).",
    }
    (OUT / "sims_summary.json").write_text(json.dumps(out, indent=2))
    h, s = out["house_dem_seats"], out["senate_dem_seats_incl_king_sanders"]
    print(f"rho_n {rho_n:.3f} (shifts {shifts})")
    print(f"House D median {h['median']:.0f} 80% [{h['p10']:.0f},{h['p90']:.0f}] P(D control) {out['p_house_dem_control']:.3f}")
    print(f"Senate D median {s['median']:.0f} 80% [{s['p10']:.0f},{s['p90']:.0f}] P(D control) {out['p_senate_dem_control']:.3f}")


if __name__ == "__main__":
    main()
