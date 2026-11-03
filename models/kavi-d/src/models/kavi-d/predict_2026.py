"""2026 predictions for all 470 races, on Kalshi's candidate slots.

Party model: fitted on all 2010-2024 contested races (calibration slope from rolling-origin OOF 2014-2024).
Candidate split inside a side: same-party model fitted on 2012-2024 same-party finals (+ 2026 polls where present).
Kalshi prices are read only to list the slots and to fill the p_kalshi column; they are never a model input.

Usage: python predict_2026.py  -> out/pred_2026.csv, out/disagreements_2026.csv, out/cache/features_2026.csv
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import late_info
from assemble import candidate_q, classify, principal
from backtest import CHOSEN, chosen_config
from common import CACHE, CLEAN, KALSHI_RACES, OUT, ROOT, TEMPLATE, edit1, name_matches, tokens
from features import build_2026_features
from history import training_table
from model import fit_predict
from sameparty import fit as fit_sameparty
from sameparty import pick_params

PARTY_LABEL = {"Democratic party": "DEM", "Republican party": "REP"}
SAME_PARTY_POLL_START = "2026-06-03"  # after the CA top-two primary
CA_PRIMARY = ROOT / "data" / "extra" / "ca_primary_2026.csv"
CA_POLLS = ROOT / "data" / "extra" / "ca_polls_2026.csv"  # borrowed from kavi-b


def kalshi_slots() -> pd.DataFrame:
    k = pd.read_csv(KALSHI_RACES)
    rows = []
    for r in k.itertuples():
        for part in str(r.candidates).split(" | "):
            party, rest = part.split(":", 1)
            name, mid = rest.rsplit(":", 1)
            rows.append({"race_id": r.race_id, "kalshi_party": party, "candidate": name, "mid": float(mid)})
    s = pd.DataFrame(rows)
    s["p_kalshi"] = s["mid"] / s.groupby("race_id")["mid"].transform("sum")
    return s


def attach_attributes(slots: pd.DataFrame) -> pd.DataFrame:
    cands = pd.read_csv(CLEAN / "candidates_2026.csv")
    fec = pd.read_csv(CLEAN / "fec_2026.csv")
    fec_race = pd.read_csv(CLEAN / "fec_race_2026.csv").set_index("race_id")
    tmpl = pd.read_csv(TEMPLATE)[["race_id", "chamber"]]
    out = []
    for r in slots.itertuples():
        c = cands[cands["race_id"] == r.race_id]
        rec = fec[(fec["race_id"] == r.race_id) & (fec["kalshi_name"] == r.candidate)]["receipts"]
        receipts = float(rec.iloc[0]) if len(rec) and pd.notna(rec.iloc[0]) else np.nan
        if r.candidate in PARTY_LABEL:  # party-only slot (AK-SEN, LA House, phantom R slots)
            party = PARTY_LABEL[r.candidate]
            same = c[c["party"] == party]
            major = same[same["is_major"] == 1]
            is_major = int(len(same) > 0)
            inc = int(same["is_incumbent"].max()) if len(same) else 0
            name = major["candidate_name"].iloc[0] if len(major) == 1 else r.candidate
            if r.race_id.startswith("LA-") and r.race_id != "LA-SEN":
                receipts = float(fec_race.loc[r.race_id, "dem_receipts" if party == "DEM" else "rep_receipts"])
            note = "party-only Kalshi slot"
        else:
            hit = c[c["candidate_name"].map(lambda n: name_matches(tokens(r.candidate), n))]
            if hit.empty:  # one-letter spelling variants (KY-03: Kalshi "Rodriguez", Wikipedia "Rodriquez")
                kt = tokens(r.candidate)
                hit = c[c["candidate_name"].map(lambda n: bool(kt) and bool(tokens(n)) and kt[0] == tokens(n)[0]
                                                and edit1(kt[-1], tokens(n)[-1]))]
            if hit.empty:
                party, is_major, inc, name = (r.kalshi_party if r.kalshi_party != "UNK" else "IND"), 0, 0, r.candidate
                note = "not on Wikipedia ballot list (withdrawn / write-in): minor"
            else:
                h = hit.iloc[0]
                party, is_major, inc, name = h["party"], int(h["is_major"]), int(h["is_incumbent"]), r.candidate
                note = ""
        out.append({**r._asdict(), "party": party, "is_major": is_major, "inc": inc, "receipts": receipts,
                    "model_name": name, "note": note})
    s = pd.DataFrame(out).drop(columns=["Index"])
    return s.merge(tmpl, on="race_id", how="left")


def sides(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["viable_ind"] = (g["party"] != "DEM") & (g["party"] != "REP") & (g["is_major"] == 1)
    g = classify(g.assign(candidate_orig=g["candidate"]))
    g.loc[g["is_major"] == 0, "side"] = "minor"
    if g["race_id"].iloc[0] == "CA-06":  # Kiley: former-R incumbent running as independent; no R on the ballot
        g.loc[g["candidate"] == "Kevin Kiley", "side"] = "R"
        g.loc[g["candidate"] == "Kevin Kiley", "note"] = "ex-Republican incumbent, modelled as the R side"
    return g


def same_party_polls(race_id: str, names: list[str]) -> dict:
    """Mean poll share per candidate in same-party races: shared polls_2026 (cand_detail) plus the CA sub-page polls
    collected by kavi-b (data/extra/ca_polls_2026.csv); a sub-page poll is skipped when the shared file has a poll
    by the same pollster ending within 2 days."""
    p = pd.read_csv(CLEAN / "polls_2026.csv")
    p = p[(p["race_id"] == race_id) & (p["end_date"] >= SAME_PARTY_POLL_START) & p["cand_detail"].notna()]
    shares = {n: [] for n in names}
    for detail in p["cand_detail"]:
        for item in str(detail).split(";"):
            m = re.match(r"\s*(.+?)\s*\((.*?)\):\s*([\d.]+)", item)
            if not m:
                continue
            for n in names:
                if name_matches(tokens(m.group(1)), n):
                    shares[n].append(float(m.group(3)))
    if CA_POLLS.exists() and race_id.startswith("CA-"):
        sub = pd.read_csv(CA_POLLS)
        sub = sub[(sub["race_id"] == race_id) & (sub["end_date"] >= SAME_PARTY_POLL_START)]
        seen = [(tokens(a)[:1], pd.Timestamp(e)) for a, e in zip(p["pollster"], p["end_date"])]
        for r in sub.itertuples():
            dup = any(t == tokens(r.pollster)[:1] and abs((pd.Timestamp(r.end_date) - e).days) <= 2 for t, e in seen)
            if dup:
                continue
            for cand, pct in ((r.cand_a, r.pct_a), (r.cand_b, r.pct_b)):
                for n in names:
                    if name_matches(tokens(cand), n):
                        shares[n].append(float(pct))
    return {n: float(np.mean(v)) for n, v in shares.items() if v}


def ca_primary(race_id: str, names: list[str]) -> dict:
    """June 2026 top-two primary share per candidate (data/extra/ca_primary_2026.csv)."""
    if not CA_PRIMARY.exists() or not race_id.startswith("CA-"):
        return {}
    d = pd.read_csv(CA_PRIMARY)
    d = d[d["race_id"] == race_id]
    out = {}
    for n in names:
        hit = d[d["candidate"].map(lambda c: name_matches(tokens(c), n))]
        if len(hit):
            out[n] = float(hit["pct"].iloc[0])
    return out


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = (fit_sameparty(2026), fit_sameparty(2026, use_primary=False))
    params = pair[1]
    slots = attach_attributes(kalshi_slots())
    races = {rid: sides(g) for rid, g in slots.groupby("race_id", sort=False)}
    struct = []
    for rid, g in races.items():
        gm = g.assign(candidate=g["model_name"])
        d, r = principal(gm, "D", params), principal(gm, "R", params)
        if d is None or r is None:
            continue
        struct.append({"year": 2026, "race_id": rid, "chamber": g["chamber"].iloc[0], "state": rid[:2],
                       "special": int(rid in ("FL-SEN", "OH-SEN")), "d_name": d["candidate"], "r_name": r["candidate"],
                       "d_is_ind": d["party"] != "DEM", "d_inc": d["inc"], "r_inc": r["inc"],
                       "d_rec": d["receipts"], "r_rec": r["receipts"]})
    feats = build_2026_features(pd.DataFrame(struct))
    p, info = fit_predict(training_table(), feats, chosen_config(), 2026)
    print("2026 calibration", info, "config", CHOSEN)
    feats = feats.assign(p_anti_r=p)
    p_by_race = dict(zip(feats["race_id"], p))
    out = []
    for rid, g in races.items():
        same_side = rid not in p_by_race
        poll = same_party_polls(rid, list(g["candidate"])) if same_side else None
        prim = ca_primary(rid, list(g["candidate"])) if same_side else None
        q = candidate_q(g, p_by_race.get(rid), pick_params(pair, prim), poll, prim)
        note = g["note"].copy()
        if same_side and (g["side"] != "minor").sum() > 1:
            note = note.where(note != "", f"same-party contest: incumbency+money split; polls {poll or 'none'}; "
                                          f"June primary % {prim or 'none'}")
        out.append(g.assign(q=q, note=note))
    pred = pd.concat(out, ignore_index=True)
    return pred, feats


def disagreements(pred: pd.DataFrame, feats: pd.DataFrame, late: pd.DataFrame | None = None) -> pd.DataFrame:
    f = feats.set_index("race_id")
    rows = []
    for rid, g in pred.groupby("race_id", sort=False):
        ours, fav = g.loc[g["q"].idxmax()], g.loc[g["p_kalshi"].idxmax()]
        q_fav = float(fav["q"])
        if ours["candidate"] == fav["candidate"] and abs(q_fav - fav["p_kalshi"]) <= 0.10:
            continue
        reason = explain(rid, g, f.loc[rid] if rid in f.index else None)
        if late is not None and rid in set(late["race_id"]):
            lr = late[late["race_id"] == rid].iloc[0]
            reason += (f"; dossier: Kalshi underdog {lr['underdog']} model {lr['q_model_dog']:.3f} -> "
                       f"{lr['q_final_dog']:.3f} (dossier {lr['q_dossier_dog']:.3f}, w {lr['w']})")
        rows.append({"race_id": rid, "chamber": g["chamber"].iloc[0], "kalshi_fav": fav["candidate"],
                     "p_kalshi_fav": round(float(fav["p_kalshi"]), 3), "q_kalshi_fav": round(q_fav, 3),
                     "our_top": ours["candidate"], "q_our_top": round(float(ours["q"]), 3),
                     "p_kalshi_our_top": round(float(ours["p_kalshi"]), 3),
                     "against_kalshi": ours["candidate"] != fav["candidate"], "reasoning": reason})
    return pd.DataFrame(rows)


def explain(rid: str, g: pd.DataFrame, f) -> str:
    if f is None:
        return ("same-party / one-sided race: candidate split from incumbency, FEC receipts, top-two primary share "
                "(fitted weights) and post-primary polls (hand-set weight)")
    bits = [f"ratings mean {f['r_mean']:+.2f} (D+)", f"lean {f['lean_m']:+.1f}", f"lean+generic {f['base_m']:+.1f}"]
    if f["poll_n"] > 0:
        bits.append(f"polls {f['poll_m']:+.1f} (n={int(f['poll_n'])})")
    else:
        bits.append("no polls")
    if pd.notna(f["prev_rel"]):
        bits.append(f"prev rel margin {f['prev_rel']:+.1f}")
    bits.append(f"inc {int(f['inc']):+d}")
    return f"model P(anti-R)={f['p_anti_r']:.3f}; " + ", ".join(bits)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-late-info", action="store_true", help="skip the dossier adjustment (model-only q)")
    args = ap.parse_args()
    pred, feats = build()
    log = None
    if args.no_late_info:
        pred = pred.assign(q_model=pred["q"])
    else:
        pred, log = late_info.adjust(pred)
        log.to_csv(OUT / "late_info_2026.csv", index=False)
        print(f"late-info adjustment: {int((log['status'] == 'adjusted').sum())} races; "
              f"unmatched {list(log.loc[log['status'] != 'adjusted', 'race_id'])}")
    CACHE.mkdir(parents=True, exist_ok=True)
    feats.to_csv(CACHE / "features_2026.csv", index=False)
    cols = ["race_id", "chamber", "candidate", "party", "q", "p_kalshi", "q_model", "note", "kalshi_party", "side",
            "inc", "receipts"]
    pred = pred.rename(columns={"party": "model_party"}).assign(party=lambda d: d["kalshi_party"])
    pred["party"] = np.where(pred["party"] == "UNK", pred["model_party"], pred["party"])
    pred[cols].to_csv(OUT / "pred_2026.csv", index=False)
    dis = disagreements(pred, feats, log)
    dis.to_csv(OUT / "disagreements_2026.csv", index=False)
    tmpl = pd.read_csv(TEMPLATE)
    assert set(tmpl["race_id"]) == set(pred["race_id"]), "race coverage"
    assert np.allclose(pred.groupby("race_id")["q"].sum(), 1.0)
    print(f"races {pred['race_id'].nunique()}, slots {len(pred)}, disagreements {len(dis)}, "
          f"against Kalshi favourite {int(dis['against_kalshi'].sum())}")


if __name__ == "__main__":
    main()
