"""Correlated simulation of the 2026 House and Senate (N_SIMS draws), from the fitted race margins.

Per draw: one national error eps_N ~ N(0, sigma_N) shared by every race (loading c_i = b_N (1 - w_i) + w_i, as
in model.sigma_total), one state error per state shared by that state's House and Senate races, and a race error
(scaled Student-t, ERR_DF). The state share of the idiosyncratic variance (ICC) is estimated from stage-A
residuals of House races in the same state and year (training data 2010-2024).
Races without a D-v-R margin (same-party top-two) are drawn independently from their q.
Writes out/sims_summary.json. Run: .venv/bin/python src/models/kavi-e/simulate.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import model
import national
from allocate import MAJOR_IND
from backtest import load_frames
from common import OUT, WORK, read

N_SIMS = 20000
SEED = 20261103
HOUSE_MAJORITY = 218
SENATE_MAJORITY = 51  # 50-50 goes to the R vice president


def state_icc() -> float:
    house, _ = load_frames()
    tr = model.prep(house[house["contested"] & house["L"].notna()])
    coef, _ = model.fit_stage_a(tr, model.HOUSE_FEATURES)
    tr = tr.assign(r=tr["margin"] - model.design(tr, model.HOUSE_FEATURES) @ coef)
    g = tr.groupby(["year", "state"])["r"]
    sizes, means = g.size(), g.mean()
    keep = sizes >= 2
    within = float((g.var(ddof=1)[keep] * (sizes[keep] - 1)).sum() / (sizes[keep] - 1).sum())
    n0 = float(sizes[keep].mean())
    between = float(means[keep].var(ddof=1)) - within / n0
    return float(np.clip(between / (between + within), 0.0, 0.5))


def senate_baseline() -> tuple[int, int]:
    """Seats NOT up in 2026: (D caucus incl. King and Sanders, R). 2025-27 Senate: R 53, D 45 + 2 I."""
    cand = read("candidates_2026.csv")
    up = cand[cand["chamber"] == "Senate"].drop_duplicates("race_id")["incumbent_party_of_seat"].value_counts()
    return 47 - int(up.get("DEM", 0)), 53 - int(up.get("REP", 0))


def summarise(x: np.ndarray, majority: int | None = None) -> dict:
    out = {"mean": round(float(x.mean()), 2), "median": float(np.median(x)),
           "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
           "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}
    vals, counts = np.unique(x, return_counts=True)
    out["distribution"] = {str(int(v)): int(c) for v, c in zip(vals, counts)}
    if majority is not None:
        out["p_majority"] = round(float((x >= majority).mean()), 4)
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    fit_info = json.loads((OUT / "fit_2026.json").read_text())
    env = national.estimate(2026)
    races = pd.read_csv(WORK / "races_2026.csv")
    pred = pd.read_csv(OUT / "pred_2026.csv")
    icc = state_icc()
    df = model.ERR_DF
    b_n = {"House": fit_info["house"]["coef"]["N"], "Senate": fit_info["senate"]["coef"]["N"]}
    races["c"] = races.apply(lambda r: b_n[r["chamber"]] * (1 - r["w_poll"]) + r["w_poll"], axis=1)
    idio = np.sqrt(np.clip(races["sigma"] ** 2 - (races["c"] * env["sigma_n"]) ** 2, 1e-6, None)).to_numpy()
    states = sorted(races["state"].unique())
    s_idx = races["state"].map({s: i for i, s in enumerate(states)}).to_numpy()

    eps_n = rng.normal(0, env["sigma_n"], N_SIMS)
    eps_s = rng.normal(0, 1, (N_SIMS, len(states)))
    t_scale = np.sqrt((df - 2) / df) if df and df > 2 else 1.0
    eps_r = (rng.standard_t(df, (N_SIMS, len(races))) * t_scale) if df and df > 2 else rng.normal(0, 1, (N_SIMS, len(races)))
    margin = (races["mu"].to_numpy()[None, :] + races["c"].to_numpy()[None, :] * eps_n[:, None]
              + idio[None, :] * (np.sqrt(icc) * eps_s[:, s_idx] + np.sqrt(1 - icc) * eps_r))
    d_side_wins = margin > 0

    # winner party of each race in each draw
    seats = {"House": {"DEM": 0, "REP": 0, "IND": 0}, "Senate": {"DEM": 0, "REP": 0, "IND": 0}}
    counts = {ch: {p: np.zeros(N_SIMS, int) for p in ("DEM", "REP", "IND")} for ch in seats}
    rid_pos = {r: i for i, r in enumerate(races["race_id"])}
    for rid, g in pred.groupby("race_id"):
        ch = g["chamber"].iloc[0]
        parties = set(g.loc[g["q"] > 0.01, "party"])
        dside = "IND" if rid in MAJOR_IND[2026] else "DEM"
        rside = "IND" if rid == "CA-06" else "REP"
        if {dside, rside} <= parties and rid in rid_pos and len(parties) == 2:
            win = d_side_wins[:, rid_pos[rid]]
            counts[ch][dside] += win
            counts[ch][rside] += ~win
        else:  # same-party or one-sided: draw a party from q
            pq = g.groupby("party")["q"].sum()
            draw = rng.choice(pq.index.to_numpy(), size=N_SIMS, p=(pq / pq.sum()).to_numpy())
            for p in ("DEM", "REP", "IND"):
                counts[ch][p] += draw == p
    d_base, r_base = senate_baseline()
    sen_d = counts["Senate"]["DEM"] + d_base
    sen_r = counts["Senate"]["REP"] + r_base
    out = {
        "n_sims": N_SIMS, "seed": SEED, "national": {"n_hat": env["n_hat"], "sigma_n": env["sigma_n"]},
        "state_icc": round(icc, 3), "err_df": df,
        "house": {"dem_seats": summarise(counts["House"]["DEM"], HOUSE_MAJORITY),
                  "rep_seats": summarise(counts["House"]["REP"], HOUSE_MAJORITY),
                  "ind_seats_mean": round(float(counts["House"]["IND"].mean()), 3),
                  "p_dem_control": round(float((counts["House"]["DEM"] >= HOUSE_MAJORITY).mean()), 4)},
        "senate": {"seats_not_up": {"dem_caucus": d_base, "rep": r_base},
                   "dem_caucus_seats": summarise(sen_d, SENATE_MAJORITY), "rep_seats": summarise(sen_r),
                   "ind_wins_mean": round(float(counts["Senate"]["IND"].mean()), 3),
                   "p_dem_control": round(float((sen_d >= SENATE_MAJORITY).mean()), 4),
                   "p_dem_plus_new_ind_ge_51": round(float((sen_d + counts["Senate"]["IND"] >= 51).mean()), 4),
                   "p_rep_control": round(float((sen_r >= 50).mean()), 4)},
        "check_mean_vs_q": {
            "house_dem_q_sum": round(float(pred.loc[(pred.chamber == "House") & (pred.party == "DEM"), "q"].sum()), 2),
            "senate_dem_q_sum": round(float(pred.loc[(pred.chamber == "Senate") & (pred.party == "DEM"), "q"].sum()), 2)},
    }
    (OUT / "sims_summary.json").write_text(json.dumps(out, indent=2))
    h, s = out["house"], out["senate"]
    print(f"House D seats median {h['dem_seats']['median']:.0f} (80% {h['dem_seats']['p10']:.0f}-{h['dem_seats']['p90']:.0f}),"
          f" P(D control) {h['p_dem_control']:.3f}; Senate D caucus median {s['dem_caucus_seats']['median']:.0f}"
          f" (80% {s['dem_caucus_seats']['p10']:.0f}-{s['dem_caucus_seats']['p90']:.0f}), P(D control) {s['p_dem_control']:.3f};"
          f" icc {icc:.3f}; check {out['check_mean_vs_q']}, sim means H {h['dem_seats']['mean']} S {s['dem_caucus_seats']['mean'] - d_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
