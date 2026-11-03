"""2026 forecast: fit on 2010-2024, predict all 470 races, compare with Kalshi, simulate the chambers.

Writes out/pred_2026.csv, out/disagreements_2026.csv, out/races_2026.csv (race-level detail) and
out/sims_summary.json.  Usage: python predict_2026.py [--variant F+P+R] [--sims 20000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from backtest import VARIANTS, predict_races
from common import OUT, TEMPLATE
from inputs_2026 import candidate_table, race_features
from model import T_DF, fit_all
from races import MINOR_Q
from late_info import DOSSIER_WEIGHT, apply_dossiers
from same_party import same_party_q

SENATE_NOT_UP = {"DEM": 34, "REP": 31}   # 47 D-caucus (incl. King, Sanders) - 13 up; 53 R - 22 up
HOUSE_MAJORITY, SENATE_MAJORITY = 218, 51  # VP is a Republican: a 50-50 Senate is R control
DISAGREE_GAP = 0.10
IND_BONUS = 0.045  # mean fundamentals residual of KS-14 / UT-22 / NE-24 independents; only used if unpolled


def rating_label(r: float) -> str:
    if pd.isna(r):
        return "no rating"
    r = -r  # back to the file's scale, + = R
    names = {-3: "Safe D", -2: "Likely D", -1: "Lean D", 0: "Toss-up", 1: "Lean R", 2: "Likely R", 3: "Safe R"}
    return f"{names[int(np.clip(round(r), -3, 3))]} ({r:+.2f})"


def apply_same_party(c: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Replace the incumbency/receipts heuristic with primary + polls in same-party top-two races."""
    c = c.copy()
    races["sp_note"] = ""
    for rid in races.index[races["kind"] == "same_party"]:
        sub = c[c["race_id"] == rid]
        res = same_party_q(sub)
        if res is None:
            continue
        q, info = res
        new = pd.Series(MINOR_Q, index=sub.index)
        new[q.index] = q * (1 - MINOR_Q * (len(sub) - len(q)))
        c.loc[sub.index, "q"] = new / new.sum()
        races.at[rid, "sp_note"] = (f"same-party top-two: {info['cand_a']} primary two-finalist share "
                                    f"{info['primary_share_a']:.3f}, {info['n_polls']} polls, expected share "
                                    f"{info['m_a']:.3f} +- 0.09")
    return c


def reason(row: pd.Series) -> str:
    if row.get("sp_note", ""):
        return row["sp_note"]
    if row["kind"] != "two_party":
        return f"{row['kind']} race: q from incumbency/receipts heuristic"
    parts = [f"fundamentals D-side share {0.5 + row['mu_f']:.3f}"]
    if pd.notna(row["poll_share"]):
        parts.append(f"polls {row['poll_share']:.3f} (n={int(row['n_polls'])})")
    else:
        parts.append("no polls")
    parts.append(f"ratings {rating_label(row['rating'])}")
    pre = row.get("p_d_pre_dossier", np.nan)
    if pd.notna(pre) and abs(pre - row["p_d"]) > 1e-4:
        parts.append(f"model P(D side) {pre:.3f}, dossier-adjusted (w {DOSSIER_WEIGHT}) -> {row['p_d']:.3f}")
    else:
        parts.append(f"final {0.5 + row['mu']:.3f} +- {row['sd_total']:.3f} -> P(D side) {row['p_d']:.3f}")
    return "; ".join(parts)


def candidate_notes(c: pd.DataFrame, races: pd.DataFrame) -> pd.Series:
    notes = []
    for r in c.itertuples():
        kind = races.at[r.race_id, "kind"]
        bits = [kind]
        if r.placeholder and r.force_side == "":
            bits.append("party slot with no nominee on the ballot")
        elif r.force_side == "" and not r.matched:
            bits.append("not on Wikipedia's ballot list (withdrawn/write-in): minor")
        elif r.force_side == "":
            bits.append("minor / not the main challenger")
        elif r.party == "IND":
            bits.append(f"independent on the {r.force_side} side")
        if r.race_id.startswith("LA-") and r.chamber == "House":
            bits.append("Nov 3 jungle; party price = eventual seat winner")
        notes.append("; ".join(bits))
    return pd.Series(notes, index=c.index)


def disagreements(c: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, sub in c.groupby("race_id"):
        fav = sub.loc[sub["p_kalshi"].idxmax()]
        top = sub.loc[sub["q"].idxmax()]
        gap = fav["q"] - fav["p_kalshi"]
        if top.name == fav.name and abs(gap) <= DISAGREE_GAP:
            continue
        ev = sub.assign(ev=sub["q"] * -np.log2(sub["p_kalshi"].clip(lower=0.01)))
        evp = ev.loc[ev["ev"].idxmax()]
        rows.append({"race_id": rid, "chamber": fav["chamber"],
                     "kalshi_fav": fav["candidate"], "kalshi_fav_party": fav["party"],
                     "p_kalshi_fav": round(fav["p_kalshi"], 4), "q_kalshi_fav": round(fav["q"], 4),
                     "our_top": top["candidate"], "our_top_party": top["party"], "q_our_top": round(top["q"], 4),
                     "p_kalshi_our_top": round(top["p_kalshi"], 4), "against_kalshi": int(top.name != fav.name),
                     "ev_pick": evp["candidate"], "ev_pick_points": round(evp["ev"], 3),
                     "reasoning": reason(races.loc[rid])})
    return pd.DataFrame(rows).sort_values(["against_kalshi", "q_kalshi_fav"], ascending=[False, True])


def simulate(c: pd.DataFrame, races: pd.DataFrame, fits: dict, n: int, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    z_nat = rng.standard_normal(n)
    states = sorted(races["state"].unique())
    z_state = pd.DataFrame(rng.standard_normal((n, len(states))), columns=states)
    seats = {ch: {p: np.zeros(n, dtype=int) for p in ("DEM", "REP", "IND")} for ch in ("House", "Senate")}
    for rid, sub in c.groupby("race_id"):
        r = races.loc[rid]
        ch = r["chamber"]
        if r["kind"] == "two_party":
            scale = r["sd_race"] / np.sqrt(T_DF / (T_DF - 2))
            share = (r["mu"] + r["sd_nat"] * z_nat + r["sd_state"] * z_state[r["state"]].to_numpy()
                     + scale * rng.standard_t(T_DF, n))
            d_win = share > 0
            winners = []
            for side, mask in (("D", d_win), ("R", ~d_win)):
                grp = sub[sub["side"] == side]
                w = (grp["q"] / grp["q"].sum()).to_numpy()
                pick = rng.choice(len(grp), size=n, p=w)
                winners.append((mask, grp["party"].to_numpy()[pick]))
            party = np.where(winners[0][0], winners[0][1], winners[1][1])
        else:
            w = (sub["q"] / sub["q"].sum()).to_numpy()
            party = sub["party"].to_numpy()[rng.choice(len(sub), size=n, p=w)]
        for p in seats[ch]:
            seats[ch][p] += (party == p) if p != "IND" else ~np.isin(party, ["DEM", "REP"])
    return summarise(seats, n, z_nat, fits)


def dist(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "p2.5": float(np.percentile(x, 2.5)),
            "p97.5": float(np.percentile(x, 97.5))}


def summarise(seats: dict, n: int, z_nat: np.ndarray, fits: dict) -> dict:
    h, s = seats["House"], seats["Senate"]
    sen_d = s["DEM"] + SENATE_NOT_UP["DEM"]
    sen_r = s["REP"] + SENATE_NOT_UP["REP"]
    hist_h = pd.Series(h["DEM"]).value_counts().sort_index()
    hist_s = pd.Series(sen_d).value_counts().sort_index()
    return {
        "n_sims": n,
        "correlation": "shared national shock (sd House %.4f, Senate %.4f share), state shock shared across "
                       "House+Senate in a state (sd %.4f), Student-t(%d) race noise; non-two-party races drawn "
                       "independently from q" % (fits["House"].nat_sd, fits["Senate"].nat_sd,
                                                  fits["House"].sd_state, T_DF),
        "house": {"dem_seats": dist(h["DEM"]), "rep_seats": dist(h["REP"]), "ind_seats": dist(h["IND"]),
                  "p_dem_majority": float((h["DEM"] >= HOUSE_MAJORITY).mean()),
                  "p_rep_majority": float((h["REP"] >= HOUSE_MAJORITY).mean()),
                  "dem_seat_histogram": {int(k): int(v) for k, v in hist_h.items()}},
        "senate": {"dem_seats_total": dist(sen_d), "rep_seats_total": dist(sen_r), "ind_wins_2026": dist(s["IND"]),
                   "dem_wins_2026": dist(s["DEM"]), "rep_wins_2026": dist(s["REP"]),
                   "p_dem_control": float((sen_d >= SENATE_MAJORITY).mean()),
                   "p_dem_plus_new_ind_ge_51": float((sen_d + s["IND"] >= SENATE_MAJORITY).mean()),
                   "p_rep_control": float((sen_r >= 50).mean()),
                   "note": "dem_seats_total counts King and Sanders (not up) as D; new independents (Osborn, "
                           "Bodnar, Achilles, Bengs) are counted separately",
                   "dem_seat_histogram": {int(k): int(v) for k, v in hist_s.items()}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="F+P+R+C", choices=list(VARIANTS))
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--quant-lambda", type=float, default=0.0)
    ap.add_argument("--dossier-weight", type=float, default=DOSSIER_WEIGHT)
    ap.add_argument("--suffix", default="", help="sensitivity runs: write out/sens/*<suffix> instead of out/")
    args = ap.parse_args()
    out_dir = OUT if not args.suffix else OUT / "sens"
    out_dir.mkdir(exist_ok=True)
    sfx = args.suffix
    hist = pd.read_csv(OUT / "hist_features.csv")
    polls, ratings, cal = VARIANTS[args.variant]
    fits = fit_all(hist, list(range(2010, 2025, 2)), polls, ratings, calibrate=cal)
    c = candidate_table()
    feats = race_features(c, args.quant_lambda)
    c, races = predict_races(c, feats, fits, ind_bonus=IND_BONUS)
    c = apply_same_party(c, races)
    if args.dossier_weight > 0:
        c, races, dlog = apply_dossiers(c, races, args.dossier_weight)
        dlog.to_csv(out_dir / f"dossier_adjustments_2026{sfx}.csv", index=False)
    template = pd.read_csv(TEMPLATE)
    missing = set(template["race_id"]) - set(c["race_id"])
    if missing:
        raise SystemExit(f"races missing from the prediction: {sorted(missing)}")
    c["note"] = candidate_notes(c, races)
    pred = c[["race_id", "chamber", "candidate", "party", "q", "p_kalshi", "note"]].copy()
    pred["q"] = pred["q"].round(5)
    pred["p_kalshi"] = pred["p_kalshi"].round(5)
    pred.to_csv(out_dir / f"pred_2026{sfx}.csv", index=False)
    races["reason"] = [reason(r) for _, r in races.iterrows()]
    races.to_csv(out_dir / f"races_2026{sfx}.csv")
    dis = disagreements(c, races)
    dis.to_csv(out_dir / f"disagreements_2026{sfx}.csv", index=False)
    summary = simulate(c, races, fits, args.sims)
    summary["generic_margin_2026"] = float(feats["generic_margin"].iloc[0])
    house = c[c["chamber"] == "House"]
    summary["house"]["expected_dem_seats_sum_q"] = float(house.loc[house["party"] == "DEM", "q"].sum())
    summary["house"]["expected_dem_seats_sum_p_kalshi"] = float(house.loc[house["party"] == "DEM", "p_kalshi"].sum())
    summary["settings"] = {"cal_spec": "base", "cal_weight": 0.75, "poll_mode": "v1",
                           "quant_lambda": args.quant_lambda, "dossier_weight": args.dossier_weight}
    summary["alpha_hat"] = {ch: f.a + f.b * float(feats["g"].iloc[0]) for ch, f in fits.items()}
    (out_dir / f"sims_summary{sfx}.json").write_text(json.dumps(summary, indent=2))
    print(f"races {races.shape[0]}, against Kalshi {int(dis['against_kalshi'].sum())}, flagged {len(dis)}")
    print(json.dumps({k: summary[k] for k in ("house", "senate")}, indent=1)[:1500])


if __name__ == "__main__":
    main()
