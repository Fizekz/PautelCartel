"""2026 forecast: fit on 2010-2024, predict all 470 races, write out/pred_2026.csv and out/disagreements_2026.csv.

Candidates are Kalshi's (primary event of each race in the Nov 1 snapshot); p_kalshi = mid normalised within the
race. Same-party races (CA top-two) split by incumbency + FEC receipts + polls (allocate.within_score).
Run: .venv/bin/python src/models/kavi-e/predict_2026.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import independents
import model
import national
import dossier_adjust
import within_party
from allocate import INC_WEIGHT, MAJOR_IND, R_SIDE_IND, allocate
from backtest import load_frames
from common import KALSHI_MARKETS, KALSHI_RACES, OUT, ROOT, WORK, ensure_dirs, last_name, read, same_person
from polls import polls_2026
from ratings_extra import rgap_2026

FEC_WITHIN = 0.6  # logit per unit ln(receipts ratio), same-party races only (judgment, no training data)
POLL_WITHIN = 0.15  # logit per point of poll lead, same-party races only (judgment)
INC_WITH_POLL = 0.5  # with polls, incumbency is mostly already in the poll: its logit weight drops 2.5 -> 0.5
DISAGREE_GAP = 0.10
# v2: the independent poll shift is estimated in independents.py (net of the year's Senate poll miss, shrunk):
# -2.8 for 2026 (v1 hand-set -5); idiosyncratic sd x1.25 (v1 x1.5).


def kalshi_candidates() -> pd.DataFrame:
    races = pd.read_csv(KALSHI_RACES)[["race_id", "event_ticker"]]
    m = pd.read_csv(KALSHI_MARKETS)
    m = m.merge(races, on=["race_id", "event_ticker"])
    m["p_kalshi"] = m["mid"] / m.groupby("race_id")["mid"].transform("sum")
    return m[["race_id", "chamber", "candidate", "party", "p_kalshi"]].reset_index(drop=True)


def attach_ballot(k: pd.DataFrame) -> pd.DataFrame:
    """Party for UNK rows and incumbency, from candidates_2026 (name match; party-only slots use the party)."""
    cand = read("candidates_2026.csv")
    fec = read("fec_2026.csv")
    parties, incs, receipts = [], [], []
    for _, r in k.iterrows():
        cs = cand[cand["race_id"] == r["race_id"]]
        hit = cs[cs["candidate_name"].map(lambda n: same_person(n, r["candidate"]))]
        party = r["party"]
        if party == "UNK" and not hit.empty:
            party = hit["party"].iloc[0]
        if r["candidate"].endswith("party"):
            hit = cs[(cs["party"] == party) & (cs["is_major"] == 1)]
        incs.append(int(hit["is_incumbent"].max()) if not hit.empty else 0)
        parties.append(party)
        f = fec[(fec["race_id"] == r["race_id"]) & (fec["kalshi_name"] == r["candidate"])]
        receipts.append(float(f["receipts"].fillna(0).sum()) if not f.empty else 0.0)
    return k.assign(party=parties, is_incumbent=incs, receipts=receipts)


def same_party_poll_leads() -> dict[tuple[str, str], float]:
    """Mean share of each named candidate in same-party (UNK) race polls since 2026-07-01."""
    p = read("polls_2026.csv")
    p = p[(p["end_date"] >= "2026-07-01") & p["cand_detail"].astype(str).str.contains(r"\(\?\)")]
    shares: dict[tuple[str, str], list[float]] = {}
    for _, r in p.iterrows():
        for part in str(r["cand_detail"]).split(";"):
            if ":" not in part:
                continue
            name, val = part.rsplit(":", 1)
            shares.setdefault((r["race_id"], last_name(name.replace("(?)", ""))), []).append(float(val))
    return {k: float(np.mean(v)) for k, v in shares.items()}


def within_scores(k: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """v2: same-party races. sp_score = fitted within-party model (June primary lead + incumbency, within_party.py);
    within_score = late same-party poll lead x POLL_WITHIN (judgment; no same-party poll history), or, where no
    primary share exists, v1's FEC + incumbency rule."""
    leads = same_party_poll_leads()
    score = pd.Series(0.0, index=k.index)
    sp = pd.Series(np.nan, index=k.index)
    for rid, g in k.groupby("race_id"):
        major = g[g["party"].isin(["DEM", "REP"])]
        dup = major[major.duplicated("party", keep=False)]
        if dup.empty:
            continue
        if rid[:2] in ("CA", "WA"):
            sp.loc[dup.index] = within_party.scores(2026, rid, dup, dup["receipts"])
        poll = dup["candidate"].map(lambda n: leads.get((rid, last_name(n)), np.nan))
        if sp.loc[dup.index].isna().any():
            rec = np.log(dup["receipts"] + 1e4)
            score.loc[dup.index] += FEC_WITHIN * (rec - rec.mean())
            if poll.notna().all():
                score.loc[dup.index] -= (INC_WEIGHT - INC_WITH_POLL) * dup["is_incumbent"].astype(float)
        if poll.notna().all():
            score.loc[dup.index] += POLL_WITHIN * (poll - poll.mean())
    return score, sp


def same_party_races(k: pd.DataFrame) -> set[str]:
    major = k[k["party"].isin(["DEM", "REP"]) & (k["p_kalshi"] > 0.02)]
    return {rid for rid, g in major.groupby("race_id") if g["party"].duplicated().any()}


def frames_2026() -> tuple[pd.DataFrame, pd.DataFrame]:
    h = pd.read_csv(WORK / "house_2026.csv")
    pol = polls_2026()
    hp = pol[pol["chamber"] == "House"][["race_id", "poll_margin", "poll_neff"]]
    h = h.merge(hp, on="race_id", how="left")
    cand = read("candidates_2026.csv")
    for rid, surname in {**MAJOR_IND[2026], **R_SIDE_IND[2026]}.items():
        if not rid.endswith("SEN") and rid in set(h["race_id"]):
            cs = cand[cand["race_id"] == rid]
            ind_inc = int(cs.loc[cs["candidate_name"].str.contains(surname), "is_incumbent"].max())
            other = "REP" if rid in MAJOR_IND[2026] else "DEM"
            oth_inc = int(cs.loc[cs["party"] == other, "is_incumbent"].max())
            h.loc[h["race_id"] == rid, "inc"] = (ind_inc - oth_inc) if other == "REP" else (oth_inc - ind_inc)
    s = pd.read_csv(WORK / "senate_2026.csv")
    gap = rgap_2026()
    h, s = h.merge(gap, on="race_id", how="left"), s.merge(gap, on="race_id", how="left")
    shift = independents.poll_shift(2026, pd.read_csv(WORK / "senate_hist.csv"))
    for df in (h, s):
        ind = df["race_id"].isin(list(MAJOR_IND[2026]))
        df.loc[ind, "poll_margin"] = df.loc[ind, "poll_margin"] + shift
        df["sigma_mult"] = np.where(ind, independents.SIGMA_MULT, 1.0)
    return h, s


def note_for(rid: str, chamber: str, row: pd.Series | None) -> str:
    if row is None:
        return "same-party or one-sided race"
    bits = [f"mu {row['mu']:+.1f} sd {row['sigma']:.1f}", f"lean {row['L']:+.1f}", f"inc {int(row['inc']):+d}",
            f"rating {row['rating']:+.2f}"]
    if pd.notna(row.get("poll_margin")):
        bits.append(f"polls {row['poll_margin']:+.1f} (w {row['w_poll']:.2f})")
    if rid in MAJOR_IND[2026]:
        bits.append(f"D side = {MAJOR_IND[2026][rid]} (I)")
    if rid in R_SIDE_IND[2026]:
        bits.append(f"R side = {R_SIDE_IND[2026][rid]} (I)")
    if row.get("dossier_lam", 0) > 0:
        bits.append(f"dossier P(D) {row['q_d_dossier']:.2f} at weight {row['dossier_lam']:.2f}: model mu "
                    f"{row['mu_model']:+.1f} -> {row['mu']:+.1f} (P(D) {row['p_d_model']:.2f} -> {row['p_d']:.2f})")
    return "; ".join(bits)


def main() -> int:
    ensure_dirs()
    house_hist, senate_hist = load_frames()
    env = national.estimate(2026)
    fh = model.fit(house_hist, model.HOUSE_FEATURES)
    fs = model.fit(senate_hist, model.SENATE_FEATURES, clf_extra=house_hist)
    h26, s26 = frames_2026()
    ph = model.predict(fh, h26, env["n_hat"], env["sigma_n"]).assign(chamber="House")
    ps = model.predict(fs, s26, env["n_hat"], env["sigma_n"]).assign(chamber="Senate")
    races = pd.concat([ph, ps], ignore_index=True)
    k = attach_ballot(kalshi_candidates())
    k["within_score"], k["sp_score"] = within_scores(k)
    same_party = same_party_races(k)
    # v2 late-information adjustment from the Nov 1 dossiers (dossier_adjust.py)
    dside = {rid: "IND" for rid in MAJOR_IND[2026]}
    races = dossier_adjust.adjust_races(races, dside, skip=same_party)
    races.to_csv(WORK / "races_2026.csv", index=False)
    by_race = races.set_index("race_id")
    template = pd.read_csv(ROOT / "data" / "templates" / "full_congress_template.csv")
    rows = []
    for rid in template["race_id"]:
        cands = k[k["race_id"] == rid]
        row = by_race.loc[rid] if rid in by_race.index else None
        q = allocate(2026, rid, cands, None if row is None else float(row["p_d"]))
        note = note_for(rid, cands["chamber"].iloc[0], row)
        major = cands[cands["party"].isin(["DEM", "REP"]) & (cands["p_kalshi"] > 0.02)]
        if major["party"].duplicated().any():
            b_pct, b_inc, b_lr, _ = within_party.fit(2026, True)
            note = (f"same-party top-two: fitted within-party model ({b_pct:.2f} x primary lead/10 + {b_inc:.2f} x inc "
                    f"+ {b_lr:.2f} x ln receipts ratio) "
                    f"+ late poll lead x 0.15; scores {', '.join(f'{n}: {a:+.2f}/{b:+.2f}' for n, a, b in zip(cands['candidate'], cands['sp_score'].fillna(0), cands['within_score']))}")
        for i, c in cands.iterrows():
            rows.append({"race_id": rid, "chamber": c["chamber"], "candidate": c["candidate"], "party": c["party"],
                         "q": round(float(q.loc[i]), 6), "p_kalshi": round(float(c["p_kalshi"]), 6), "note": note})
    pred = dossier_adjust.adjust_same_party(pd.DataFrame(rows), same_party)
    pred["q"] = (pred["q"] / pred.groupby("race_id")["q"].transform("sum")).round(6)
    pred.to_csv(OUT / "pred_2026.csv", index=False)
    write_disagreements(pred)
    (OUT / "fit_2026.json").write_text(json.dumps({"env": env, "house": model.describe(fh),
                                                   "senate": model.describe(fs)}, indent=2, default=float))
    print("N_hat", round(env["n_hat"], 2), "house", model.describe(fh), "\nsenate", model.describe(fs))
    return 0


def write_disagreements(pred: pd.DataFrame) -> None:
    rows = []
    for rid, g in pred.groupby("race_id", sort=False):
        ours = g.loc[g["q"].idxmax()]
        fav = g.loc[g["p_kalshi"].idxmax()]
        gap = float(fav["q"] - fav["p_kalshi"])
        if ours["candidate"] == fav["candidate"] and abs(gap) <= DISAGREE_GAP:
            continue
        against = ours["candidate"] != fav["candidate"]
        pts = -np.log2(max(float(ours["p_kalshi"]), 0.01))
        rows.append({
            "race_id": rid, "chamber": g["chamber"].iloc[0], "kalshi_fav": fav["candidate"],
            "kalshi_fav_party": fav["party"], "p_kalshi_fav": round(float(fav["p_kalshi"]), 3),
            "q_kalshi_fav": round(float(fav["q"]), 3), "our_top": ours["candidate"], "our_top_party": ours["party"],
            "q_our_top": round(float(ours["q"]), 3), "p_kalshi_our_top": round(float(ours["p_kalshi"]), 3),
            "type": "against_favourite" if against else "same_pick_gap>0.10",
            "ev_our_top_pts": round(float(ours["q"]) * pts, 3), "reasoning": reason(g, ours, fav, against),
        })
    pd.DataFrame(rows).to_csv(OUT / "disagreements_2026.csv", index=False)


def reason(g: pd.DataFrame, ours: pd.Series, fav: pd.Series, against: bool) -> str:
    direction = "more" if fav["q"] > fav["p_kalshi"] else "less"
    head = (f"model picks {ours['candidate']} ({ours['party']}) at q={ours['q']:.2f} vs Kalshi {ours['p_kalshi']:.2f}"
            if against else f"model is {direction} confident in {fav['candidate']} ({fav['q']:.2f} vs {fav['p_kalshi']:.2f})")
    return f"{head}; inputs: {g['note'].iloc[0]}"


if __name__ == "__main__":
    sys.exit(main())
