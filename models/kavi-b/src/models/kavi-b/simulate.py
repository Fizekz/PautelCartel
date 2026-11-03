"""Correlated simulation of the 2026 House and Senate from the kavi-b model.

Each simulated election draws
  u ~ cycle mixture        v3: one national cycle effect on the logit scale from the fitted generic-ballot x midterm
                           predictive mixture (model_2026.json "cycle_mixture_2026": mean ~+0.35, sd ~1.40; v2 mid_mo: +0.67, 1.53)
  s_state ~ N(0, TAU_S^2)  one shock per state, shared by its House seats and its Senate race (assumed, see below)
and race i goes to the D side with probability sigmoid(c*eta_i + c*u + s_state), where eta_i is the race logit
that reproduces the final q (after allocation and the dossier pool; race_eta_2026.csv "eta_sim") and
c = sqrt(1 + pi*TAU_S^2/8) keeps each race's probability given u equal to the fitted sigmoid(eta_i + u)
(probit approximation), so the state shock adds correlation without changing the race marginals.
Races with no real opponent, and same-party CA races, have a known party.

Usage: python src/models/kavi-b/simulate.py [--n 20000] [--seed 2026]
Writes out/sims_summary.json and out/sims_marginals_2026.csv (simulated vs analytic q per race).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, sigmoid  # noqa: E402

TAU_S = 0.35            # state shock sd (logit); assumption, not fitted (see METHODS.md)
SENATE_HOLDOVER = {"R": 31, "D_caucus": 34}   # seats not up in 2026: R 53-22, D 45 + I 2 (King, Sanders) - 13
HOUSE_MAJORITY, SENATE_MAJORITY_D = 218, 51   # VP (R) breaks a 50-50 tie


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    races = pd.read_csv(OUT / "race_eta_2026.csv")
    model = json.loads((OUT / "model_2026.json").read_text())
    mix = model["cycle_mixture_2026"]
    w, mu, sd = (np.array(mix[k]) for k in ("w", "mu", "sd"))
    c = np.sqrt(1 + np.pi * TAU_S ** 2 / 8)
    races["state"] = races["race_id"].str[:2]
    states = sorted(races["state"].unique())
    st_idx = races["state"].map({s: i for i, s in enumerate(states)}).to_numpy()

    k = rng.choice(len(w), size=args.n, p=w / w.sum())
    u = rng.normal(mu[k], sd[k])[:, None]
    s = rng.normal(0, TAU_S, size=(args.n, len(states)))[:, st_idx]
    eta = races["eta_sim"].to_numpy()[None, :]
    p = sigmoid(c * eta + c * u + s)
    fixed = (races["no_contest"] == 1).to_numpy()
    p[:, fixed] = races.loc[fixed, "q_dside_final"].to_numpy()
    same = (races["same_party"] == 1).to_numpy()
    p[:, same] = np.where(races.loc[same, "same_party_final"].eq("D").to_numpy(), 1.0, 0.0)
    dwin = rng.random(p.shape) < p

    house = (races["chamber"] == "House").to_numpy()
    senate = ~house
    ind_side = (races["dside_is_ind"] == 1).to_numpy()   # AK-AL Hill; NE/MT/ID/SD-SEN independents
    house_d = (dwin[:, house] & ~ind_side[house]).sum(axis=1)
    house_i = (dwin[:, house] & ind_side[house]).sum(axis=1)
    house_r = house.sum() - house_d - house_i
    sen_d = (dwin[:, senate] & ~ind_side[senate]).sum(axis=1)
    sen_i = (dwin[:, senate] & ind_side[senate]).sum(axis=1)
    sen_d_caucus = SENATE_HOLDOVER["D_caucus"] + sen_d
    sen_r = SENATE_HOLDOVER["R"] + (senate.sum() - sen_d - sen_i)

    def dist(x: np.ndarray) -> dict:
        q = np.percentile(x, [10, 50, 90])
        return {"mean": float(x.mean()), "median": float(q[1]), "p10": float(q[0]), "p90": float(q[2]),
                "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
                "hist": {int(k): int(v) for k, v in zip(*np.unique(x, return_counts=True))}}

    summary = {
        "n_sims": args.n, "seed": args.seed, "cycle_2026": model["cycle_2026"], "tau_state": TAU_S,
        "u_draws": {"mean": float(u.mean()), "sd": float(u.std())},
        "house": {"dem_seats": dist(house_d), "rep_seats": dist(house_r), "ind_seats_mean": float(house_i.mean()),
                  "p_dem_control": float((house_d >= HOUSE_MAJORITY).mean()),
                  "p_rep_control": float((house_r >= HOUSE_MAJORITY).mean())},
        "senate": {"dem_caucus_seats_excl_new_independents": dist(sen_d_caucus), "rep_seats": dist(sen_r),
                   "new_independent_wins_mean": float(sen_i.mean()),
                   "p_dem_control": float((sen_d_caucus >= SENATE_MAJORITY_D).mean()),
                   "p_dem_control_if_new_independents_caucus_D": float((sen_d_caucus + sen_i >= SENATE_MAJORITY_D).mean()),
                   "p_rep_control": float((sen_r >= 50).mean()),
                   "holdover": SENATE_HOLDOVER},
        "assumptions": ["Senate holdover from the 2025 composition R 53 / D 45 / I 2 minus the 35 seats up "
                        "(22 R, 13 D per candidates_2026 incumbent_party_of_seat)",
                        "independents NE/MT/ID/SD-SEN (Osborn, Bodnar, Achilles, Bengs) counted separately; "
                        "AK-AL Hill counted as independent",
                        "D control of the Senate needs 51 because the Vice President is a Republican"],
    }
    (OUT / "sims_summary.json").write_text(json.dumps(summary, indent=2))
    marg = races[["race_id", "chamber", "q_dside_final"]].assign(q_sim=dwin.mean(axis=0))
    marg.to_csv(OUT / "sims_marginals_2026.csv", index=False)
    h, se = summary["house"], summary["senate"]
    print(f"House D seats median {h['dem_seats']['median']:.0f} (80% {h['dem_seats']['p10']:.0f}-{h['dem_seats']['p90']:.0f}), "
          f"P(D control) {h['p_dem_control']:.3f}")
    print(f"Senate D caucus median {se['dem_caucus_seats_excl_new_independents']['median']:.0f} "
          f"(80% {se['dem_caucus_seats_excl_new_independents']['p10']:.0f}-{se['dem_caucus_seats_excl_new_independents']['p90']:.0f}), "
          f"P(D control) {se['p_dem_control']:.3f}; with new I's caucusing D {se['p_dem_control_if_new_independents_caucus_D']:.3f}")
    chk = marg[(races["same_party"] == 0).to_numpy()]
    print(f"max |q_sim - q| = {np.abs(chk['q_sim'] - chk['q_dside_final']).max():.3f} (non-same-party races)")


if __name__ == "__main__":
    main()
