"""2026 forecast: fit on 2010-2024, predict all 470 races, compare with Kalshi's Nov 1 snapshot.

Usage (from the worktree root): python src/models/kavi-b/predict_2026.py
Writes out/pred_2026.csv, out/disagreements_2026.csv, out/race_eta_2026.csv (inputs for simulate.py).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allocate import allocate_race, fit_within_party, primary_pct, within_party_data  # noqa: E402
from backtest import FINAL  # noqa: E402
from common import CLEAN, OUT, OWN, last_name, name_tokens, sigmoid  # noqa: E402
from features import _fec_hist, hist_table, kalshi_candidates, table_2026  # noqa: E402
from model import SideModel  # noqa: E402
import dossiers  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

warnings.filterwarnings("ignore")
# Same-party races: no expert rating absorbs the polls, so the poll average is read as evidence with error
# sd = sqrt(SYS^2 + SINGLE^2 / n) points (shared house/turnout error + per-poll noise); a lead L then moves the
# logit by 1.6 * L / sd (probit-to-logit factor 1.6). One poll: sd 10 -> 1.6 per 10 pts; three polls: ~2.1.
POLL_SD_SYS, POLL_SD_SINGLE = 6.0, 8.0
# v2 (audit: same-party nudge ignored sponsorship): campaign-released polls get half weight in n_eff and are moved
# INTERNAL_BIAS points against their sponsor. Sponsors from the Nov 1 dossiers (CA-04: both polls released by the
# Jones campaign); the undated Binder poll's field dates (Oct 15-21) also come from the CA-04 dossier.
INTERNAL_BIAS = 4.0
CAMPAIGN_POLLS_2026 = {("CA-04", "fulcrum"): "jones", ("CA-04", "david"): "jones"}
UNDATED_FIX_2026 = {("CA-04", "David Binder Research (D)"): "2026-10-21"}
DISAGREE_GAP = 0.10


def _match(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    hit = df[df[col].fillna("").map(last_name) == last_name(name)]
    if len(hit) > 1:
        first = name_tokens(name)[0]
        sub = hit[hit[col].map(lambda n: first in name_tokens(n))]
        hit = sub if len(sub) else hit
    return hit


def candidate_info(k: pd.DataFrame, row: pd.Series, fec: pd.DataFrame, fec_race: pd.DataFrame,
                   cand: pd.DataFrame) -> pd.DataFrame:
    """Kalshi candidates of one race with party (UNK resolved via FEC), receipts, incumbency, side, minor flag."""
    k = k.copy()
    fr, cc = fec[fec["race_id"] == row.race_id], cand[cand["race_id"] == row.race_id]
    rec, inc, party = [], [], []
    for name, p in zip(k["candidate"], k["party"]):
        if name.endswith(" party"):
            pp = "DEM" if name.startswith("Democratic") else "REP"
            if row.la_jungle:
                fx = fec_race[fec_race["race_id"] == row.race_id].iloc[0]
                rec.append(float(fx["dem_receipts"] if pp == "DEM" else fx["rep_receipts"]))
            else:
                rec.append(float(fr.loc[fr["party"] == pp, "receipts"].fillna(0).sum()))
            inc.append(int((cc["party"] == pp).any() and cc.loc[cc["party"] == pp, "is_incumbent"].max() == 1))
            party.append(pp)
            continue
        hf, hc = _match(fr, "kalshi_name", name), _match(cc, "candidate_name", name)
        rec.append(float(hf["receipts"].fillna(0).sum()) if len(hf) else 0.0)
        inc.append(int(len(hc) > 0 and hc["is_incumbent"].max() == 1))
        if row.same_party:   # both finalists are of the race's party (UNK on Kalshi)
            p = {"D": "DEM", "R": "REP"}[row.same_party_final]
        elif p == "UNK":
            p = hf["party"].iloc[0] if len(hf) else (hc["party"].iloc[0] if len(hc) else "UNK")
        party.append(p)
    k["party_resolved"], k["receipts"], k["is_inc"] = party, rec, inc
    k["is_minor"] = False
    rside_ind = row.race_id == "CA-06"
    if row.dside_is_ind:
        k["side"] = np.where(k["party_resolved"] == "REP", "R", "D")
    elif rside_ind:   # Pan (D) vs Kiley (I, former R incumbent); the 'Republican party' slot is a minor
        k["side"] = np.where(k["party_resolved"] == "DEM", "D", "R")
    else:
        k["side"] = np.where(k["party_resolved"] == "REP", "R", "D")
    k["party"] = k["party_resolved"]
    return k


def same_party_polls(race_id: str) -> pd.DataFrame:
    """Two-way head-to-heads for a same-party race from the shared file (cand_detail) and the CA sub-pages,
    final 60 days, one row per pollster + end date."""
    rows = []
    p = pd.read_csv(CLEAN / "polls_2026.csv")
    for r in p[p["race_id"] == race_id].itertuples():
        if isinstance(r.cand_detail, str):
            shares = {last_name(x.split("(")[0]): float(x.rsplit(":", 1)[1]) for x in r.cand_detail.split(";")}
            rows.append({"pollster": r.pollster, "end_date": r.end_date, "shares": shares})
    c = pd.read_csv(OWN / "ca_polls_2026.csv")
    c["end_date"] = [UNDATED_FIX_2026.get((r, p), e) for r, p, e in zip(c["race_id"], c["pollster"], c["end_date"])]
    for r in c[(c["race_id"] == race_id) & c["end_date"].notna()].itertuples():
        rows.append({"pollster": r.pollster, "end_date": r.end_date,
                     "shares": {last_name(r.cand_a): r.pct_a, last_name(r.cand_b): r.pct_b}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df[(df["end_date"] >= pd.Timestamp("2026-09-04")) & (df["end_date"] <= pd.Timestamp("2026-11-02"))]
    df["pkey"] = df["pollster"].str.replace(r"\s*\((?:D|R)\)", "", regex=True).str.split().str[0].str.lower()
    df["sponsor"] = [CAMPAIGN_POLLS_2026.get((race_id, k)) for k in df["pkey"]]
    df["internal"] = df["sponsor"].notna() | df["pollster"].str.contains(r"\((?:D|R)\)", regex=True)
    return df.drop_duplicates(["pkey", "end_date"], keep="first")


def same_party_poll_adjust(k: pd.DataFrame, race_id: str, q: np.ndarray) -> tuple[np.ndarray, float | None, int]:
    """Bayes-style nudge of a same-party split by recent polls (recency-weighted lead, half-life 14 days)."""
    df = same_party_polls(race_id)
    if df.empty or len(k) != 2:
        return q, None, 0
    a, b = (last_name(n) for n in k["candidate"])
    df = df[df["shares"].map(lambda s: a in s and b in s)]
    if df.empty:
        return q, None, 0
    lead = df["shares"].map(lambda s: s[a] - s[b])
    bias = np.where(df["sponsor"].eq(a), INTERNAL_BIAS, np.where(df["sponsor"].eq(b), -INTERNAL_BIAS, 0.0))
    lead = lead - bias
    w = 0.5 ** ((pd.Timestamp("2026-11-03") - df["end_date"]).dt.days / 14.0) * np.where(df["internal"], 0.5, 1.0)
    lead = float(np.average(lead, weights=w))
    q = np.clip(q, 1e-4, 1 - 1e-4)
    n_eff = float(np.where(df["internal"], 0.5, 1.0).sum())
    sd = float(np.sqrt(POLL_SD_SYS ** 2 + POLL_SD_SINGLE ** 2 / n_eff))
    q0 = float(sigmoid(np.log(q[0] / q[1]) + 1.6 * lead / sd))
    return np.array([q0, 1 - q0]), lead, len(df)


def main() -> None:
    hist = hist_table(pre2018_fec=True)
    fec_h = _fec_hist(pre2018=True)
    side = SideModel(FINAL).fit(hist)
    wp = fit_within_party(within_party_data(hist, fec_h, sorted(hist["year"].unique())))
    tab = table_2026()
    k_all = kalshi_candidates()
    fec = pd.read_csv(CLEAN / "fec_2026.csv")
    fec_race = pd.read_csv(CLEAN / "fec_race_2026.csv")
    cand = pd.read_csv(CLEAN / "candidates_2026.csv")
    tab["eta"] = side.eta(tab)      # v2: race logit without the cycle effect
    tab["q_dside"] = side.predict(tab)
    tab.loc[tab["no_contest"] == 1, "q_dside"] = np.where(tab.loc[tab["no_contest"] == 1, "dside_name"].notna(), 0.999, 0.001)
    rows = []
    for row in tab.itertuples():
        k = candidate_info(k_all[k_all["race_id"] == row.race_id], row, fec, fec_race, cand)
        k["prim_pct"] = [primary_pct(2026, row.race_id, n) for n in k["candidate"]]
        same = {"D": "DEM", "R": "REP"}.get(row.same_party_final) if row.same_party else None
        d_party = "IND" if row.dside_is_ind else "DEM"
        if row.race_id == "CA-06":
            # R side is Kiley (IND); reuse allocate_race by labelling his slot REP for the split only
            kk = k.assign(party=np.where(k["candidate"] == "Kevin Kiley", "REP", np.where(k["party"] == "REP", "OTHER", k["party"])))
            q = allocate_race(kk, row.q_dside, d_party, None, wp)
        else:
            q = allocate_race(k, row.q_dside, d_party, same, wp)
        note = []
        if same:
            q2, lead, npolls = same_party_poll_adjust(k, row.race_id, q)
            if lead is not None:
                note.append(f"split nudged by {npolls} Sep-Oct polls ({k['candidate'].iloc[0]} {lead:+.1f})")
            q = q2
            note.append(f"same-party {same} top-two: within-party model on June primary share + money + incumbency")
        if row.no_contest:
            note.append("no real opponent on the ballot")
        if row.la_jungle:
            note.append("LA all-party primary Nov 3 (runoff Dec 12): party-level forecast")
        if row.dside_is_ind:
            note.append("independent is the main non-R challenger")
        if row.r_mean ** 2 >= 9 and not same:
            note.append("Safe by all three raters")
        k["q"] = q
        k["note"] = "; ".join(note)
        k["chamber"] = row.chamber
        rows.append(k)
    pred = pd.concat(rows, ignore_index=True)
    pred["q"] = pred["q"] / pred.groupby("race_id")["q"].transform("sum")
    # v2: late-information pool with the Nov 1 dossiers (dossiers.py), then renormalise
    pred, dlog = dossiers.apply(pred)
    pred["q"] = pred["q"] / pred.groupby("race_id")["q"].transform("sum")
    dlog.to_csv(OUT / "dossier_adjustments_2026.csv", index=False)
    tmpl = pd.read_csv(Path(CLEAN).parent / "templates" / "full_congress_template.csv")
    assert set(tmpl["race_id"]) == set(pred["race_id"]), "race list mismatch with template"
    assert (pred["party"] != "UNK").all(), "unresolved party label"
    out = pred[["race_id", "chamber", "candidate", "party", "q", "p_kalshi", "q_model", "note"]]
    out.to_csv(OUT / "pred_2026.csv", index=False)
    # D-side probability after allocation + dossier pool, and the race logit that reproduces it in the simulation
    qd = pred[pred["side"] == "D"].groupby("race_id")["q"].sum()
    tab["q_dside_final"] = tab["race_id"].map(qd).fillna(0.0).clip(1e-4, 1 - 1e-4)
    tab["eta_sim"] = [solve_eta(side, tab, q) for q in tab["q_dside_final"]]
    tab.to_csv(OUT / "race_eta_2026.csv", index=False)
    w, mu, sd = side.cyc.predictive(side.z_of(tab))
    (OUT / "model_2026.json").write_text(json.dumps({
        "config": FINAL.name, "features": list(FINAL.feats), "cycle_design": FINAL.cycle,
        "beta": side.beta.to_dict(), "year_effects_uncentred": side.year_fx.to_dict(),
        "cycle_2026": side.cycle_summary(tab), "cycle_mixture_2026": {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist()},
        "tail_floor": side.tail_floor, "train_years": list(side.train_years), "within_party": wp,
        "dossier_weight": dossiers.W_DOSSIER}, indent=2))
    write_disagreements(pred, tab, side.beta.to_dict(), side.cycle_summary(tab))
    print(f"side model beta {side.beta.round(3).to_dict()} cycle 2026 {side.cycle_summary(tab)}; within-party {wp}")
    print(f"wrote {len(out)} candidate rows for {out['race_id'].nunique()} races; dossier-adjusted races {len(dlog)}")
    house = pred[(pred["chamber"] == "House")]
    print(f"House E[D seats] (DEM rows): final {house.loc[house['party'] == 'DEM', 'q'].sum():.1f}, "
          f"model {house.loc[house['party'] == 'DEM', 'q_model'].sum():.1f}, Kalshi {house.loc[house['party'] == 'DEM', 'p_kalshi'].sum():.1f}")


def solve_eta(side: SideModel, tab: pd.DataFrame, q: float) -> float:
    """Race logit (without cycle effect) whose cycle-integrated probability equals q."""
    probe = tab.iloc[:1]
    f = lambda e: float(side._marginal(np.array([e]), probe)[0]) - q  # noqa: E731
    return float(brentq(f, -25, 25, xtol=1e-6))


def write_disagreements(pred: pd.DataFrame, tab: pd.DataFrame, beta: dict, cyc: dict) -> None:
    t = tab.set_index("race_id")
    rows = []
    for rid, g in pred.groupby("race_id"):
        ours, theirs = g.loc[g["q"].idxmax()], g.loc[g["p_kalshi"].idxmax()]
        fav_gap = float(g.loc[theirs.name, "q"] - theirs["p_kalshi"])
        if ours.name == theirs.name and abs(fav_gap) <= DISAGREE_GAP:
            continue
        r = t.loc[rid]
        pts = -np.log2(max(float(ours["p_kalshi"]), 0.01))
        rows.append({
            "race_id": rid, "chamber": ours["chamber"], "our_pick": ours["candidate"], "our_pick_party": ours["party"],
            "q_our_pick": round(float(ours["q"]), 4), "p_kalshi_our_pick": round(float(ours["p_kalshi"]), 4),
            "kalshi_fav": theirs["candidate"], "p_kalshi_fav": round(float(theirs["p_kalshi"]), 4),
            "q_kalshi_fav": round(float(g.loc[theirs.name, "q"]), 4), "gap_on_fav": round(fav_gap, 4),
            "top_pick_differs": int(ours.name != theirs.name),
            "ev_pick_our_top": round(float(ours["q"]) * pts, 3),
            "ev_pick_kalshi_fav": round(float(g.loc[theirs.name, "q"]) * -np.log2(max(float(theirs["p_kalshi"]), 0.01)), 3),
            "cook": r["cook_raw"], "sabato": r["sabato_raw"], "inside": r["inside_raw"],
            "lean_pts": round(float(r["lean"]), 1),
            "poll_margin": None if pd.isna(r["poll_margin"]) else round(float(r["poll_margin"]), 1),
            "n_polls": int(r["n_polls"]), "q_model_our_pick": round(float(ours["q_model"]), 4),
            "reasoning": reason(r, g, beta, cyc),
        })
    pd.DataFrame(rows).sort_values(["top_pick_differs", "gap_on_fav"], ascending=[False, True]) \
        .to_csv(OUT / "disagreements_2026.csv", index=False)


def reason(r: pd.Series, g: pd.DataFrame, beta: dict, cyc: dict) -> str:
    if r["same_party"]:
        return ("Same-party top-two: our split is the June primary share + money + incumbency, nudged by Sep-Oct "
                "polls; it cannot see news after the primary (scandals, endorsements), so treat it as weak.")
    if r["no_contest"]:
        return "No real opponent on the ballot; Kalshi's residual price on the other slot looks like noise."
    pm = 0.0 if r["n_polls"] == 0 else float(np.clip(r["poll_margin"], -30, 30))
    parts = {"ratings": beta["r"] * r["r_mean"], "lean": beta["lean"] * r["lean"] / 10, "polls": beta["poll"] * pm / 10}
    txt = ", ".join(f"{k} {v:+.2f}" for k, v in parts.items())
    q_d = float(g.loc[g["side"] == "D", "q"].sum())
    k_d = float(g.loc[g["side"] == "D", "p_kalshi"].sum())
    out = (f"logit contributions to the D side: {txt}; 2026 cycle effect (v3: midterm x generic ballot D+{r['generic_margin_dr']:.1f}, fitted "
           f"on 2010-2024 year effects) mean {cyc['pred_mean']:+.2f}, sd {cyc['pred_sd']:.2f}, integrated out. ")
    out += (f"P(D side) {q_d:.2f} vs Kalshi {k_d:.2f}: ")
    if abs(r["r_mean"]) < 0.5:
        out += ("raters call it a toss-up; lean and polls decide it against a generic-ballot cycle shift "
                "toward D.")
    elif q_d < k_d:
        out += ("raters have the R side ahead; even in the 2018 wave Lean R seats held 90% and Likely R 93%, so "
                "after the generic-ballot shift we stay below the market's price on the D side.")
    else:
        out += "ratings, lean and polls together are warmer on the D side than the market."
    if r["n_polls"] == 0:
        out += " No usable polls."
    elif r["n_polls"] <= 2:
        out += f" Only {int(r['n_polls'])} poll(s), probably partisan."
    return out


if __name__ == "__main__":
    main()
