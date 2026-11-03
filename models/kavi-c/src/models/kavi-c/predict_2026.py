"""2026 forecast: fit on 2010-2024, predict all 470 races, map to Kalshi candidates, list disagreements.

Usage: python src/models/kavi-c/predict_2026.py
Writes out/pred_2026.csv, out/disagreements_2026.csv, out/model_2026_diag.json, out/race_detail_2026.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import dossiers  # noqa: E402
import pool  # noqa: E402
import features_2026 as f26  # noqa: E402
import hist_features  # noqa: E402
import model  # noqa: E402
import same_party  # noqa: E402
import special  # noqa: E402
from scipy import stats  # noqa: E402

from common import CLEAN, HIST_YEARS, KALSHI_RACES, OUT, ROOT, TEMPLATE, last_name, norm_pollster  # noqa: E402

EPS_MINOR = 0.001        # minor / withdrawn candidate
UNCONTESTED_Q = 0.998    # only one party's candidate on the ballot
DISAGREE_GAP = 0.10
# Independent challengers to a Republican faded against their final poll average in every comparable past race
# in our data (two-way share miss: KS-SEN 2014 Orman -6.5, UT-SEN 2022 McMullin -2.7, NE-SEN 2024 Osborn -2.6
# points; about -3.8 after removing each year's average poll error). Half of that, applied to the poll part of
# the posterior, plus extra uncertainty because the lean/rating prior was fitted on D-vs-R races.
SP_MODEL_2026 = None         # v3: same_party.SameParty fitted on 2012-2024 in main()
SAME_PARTY_REP = {"CA-40"}   # Young Kim v Ken Calvert; the other 8 same-party finals are D v D
IND_POLL_FADE = 0.02
IND_EXTRA_SD = 0.025


def parse_kalshi() -> pd.DataFrame:
    k = pd.read_csv(KALSHI_RACES)
    rows = []
    for _, r in k.iterrows():
        for part in str(r["candidates"]).split(" | "):
            head, price = part.rsplit(":", 1)
            party, name = head.split(":", 1)
            rows.append({"race_id": r["race_id"], "party": party, "candidate": name, "mid": float(price)})
    out = pd.DataFrame(rows)
    out["p_kalshi"] = out["mid"] / out.groupby("race_id")["mid"].transform("sum")
    return out


def cand_info_2026(race_id: str, name: str) -> tuple[float, int]:
    fec = pd.read_csv(CLEAN / "fec_2026.csv")
    cand = pd.read_csv(CLEAN / "candidates_2026.csv")
    f = fec[(fec["race_id"] == race_id) & (fec["kalshi_name"].map(last_name) == last_name(name))]
    c = cand[(cand["race_id"] == race_id) & (cand["candidate_name"].map(last_name) == last_name(name))]
    rec = float(pd.to_numeric(f["receipts"], errors="coerce").fillna(0).max()) if len(f) else 0.0
    inc = int(c["is_incumbent"].max()) if len(c) else 0
    return rec, inc


PAU_SP_POLLS = ROOT / "data" / "kavi-c" / "pau_polls_2026_same_party.csv"


def pau_same_party_polls(race_id: str) -> pd.DataFrame:
    """v3: same-party-final polls from Pau's disclosed 2026 poll layer (origin/pau/model-1101
    data/features/polls_2026.csv, frozen copy), in this model's column names. Any named sponsor counts as partisan."""
    p = pd.read_csv(PAU_SP_POLLS)
    p = p[p["race_id"] == race_id]
    detail = p["cand_pcts"].fillna("").str.replace("=", ": ", regex=False)
    sponsored = p["sponsor"].notna() | p["partisan_sponsor"].isin(["D", "R"])
    return pd.DataFrame({"race_id": race_id, "pollster": p["pollster"], "end_date": p["end_date"],
                         "sample_size": p["sample_size"], "population": p["population"],
                         "partisan_sponsor": np.where(sponsored, "D", ""), "cand_detail": detail})


def two_way_2026(race_id: str, a: str, b: str) -> tuple[float, float]:
    """A's share of A+B from cand_detail of usable 2026 polls (window and recency as in history). v3: same-party
    finals add Pau's poll rows not already present (same pollster key and end date)."""
    p = f26.usable_polls()
    p = p[p["race_id"] == race_id]
    if race_id in f26.NON_PARTISAN_FINAL:
        extra = pau_same_party_polls(race_id)
        have = set(zip(p["pollster"].map(norm_pollster), p["end_date"]))
        extra = extra[[(norm_pollster(x), d) not in have for x, d in zip(extra["pollster"], extra["end_date"])]]
        p = pd.concat([p, extra], ignore_index=True)
    rows = []
    for _, r in p.iterrows():
        d = f26.detail_pct(r["cand_detail"])
        la, lb = last_name(a), last_name(b)
        if la in d and lb in d:
            rows.append({"race_id": race_id, "pollster": r["pollster"], "end_date": r["end_date"],
                         "sample_size": r["sample_size"], "population": r["population"],
                         "partisan_sponsor": r["partisan_sponsor"] if r["partisan_sponsor"] in ("D", "R") else "",
                         "share": d[la] / (d[la] + d[lb])})
    if not rows:
        return float("nan"), 0.0
    import polls
    st = polls._standardise(pd.DataFrame(rows), 2026)
    if st.empty:
        return float("nan"), 0.0
    ag, _ = polls.aggregate(st, {})
    return float(ag["poll_share"].iloc[0]), float(ag["poll_neff"].iloc[0])


# Same-party finals: no historical training set of this kind with 2026-style polls, so a small conjugate-normal
# model on A's two-way share. Prior mean from incumbency and money, wide prior; polls of same-party finals carry
# a large systematic error (many undecided, no party cue), so a 4-point floor on top of sampling noise.
SP_PRIOR_SD = 0.07
SP_INC_SHIFT = 0.04
SP_MONEY_SHIFT = 0.025
SP_POLL_FLOOR = 0.04
SP_POLL_NOISE = 0.035     # per poll of neff 1


def non_partisan_prob(m: model.Model, race_id: str, a: str, b: str) -> tuple[float, str]:
    """P(A beats B) in a D-vs-D or R-vs-R final."""
    (ra, ia), (rb, ib) = cand_info_2026(race_id, a), cand_info_2026(race_id, b)
    money = float(np.clip(np.log((ra + 5e4) / (rb + 5e4)), -3, 3))
    prior = 0.5 + SP_INC_SHIFT * (ia - ib) + SP_MONEY_SHIFT * money
    pa_pct, pb_pct = same_party.primary_pct(2026, race_id, a), same_party.primary_pct(2026, race_id, b)
    prim_txt = ""
    if SP_MODEL_2026 is not None and not (np.isnan(pa_pct) or np.isnan(pb_pct)):
        # v3: prior from the fitted June-primary model, mapped to a share with the same prior sd
        p0 = SP_MODEL_2026.prob(pa_pct, pb_pct, ia, ib)
        prior = float(0.5 + SP_PRIOR_SD * stats.t.ppf(np.clip(p0, 1e-4, 1 - 1e-4), df=m.err.nu))
        prim_txt = f"primary {pa_pct:.1f}-{pb_pct:.1f} -> P0 {p0:.2f}; "
    share, neff = two_way_2026(race_id, a, b)
    if np.isnan(share):
        mu, var = prior, SP_PRIOR_SD ** 2
        txt = "no head-to-head poll"
    else:
        v_poll = SP_POLL_FLOOR ** 2 + SP_POLL_NOISE ** 2 / max(neff, 0.05)
        var = 1 / (1 / SP_PRIOR_SD ** 2 + 1 / v_poll)
        mu = var * (prior / SP_PRIOR_SD ** 2 + share / v_poll)
        txt = f"poll avg {share:.3f} two-way (neff {neff:.2f})"
    p = float(model.win_prob(np.array([mu]), np.array([np.sqrt(var)]), m.err.nu)[0])
    return p, (f"same-party final: {prim_txt}prior share {prior:.3f} (inc {ia}/{ib}, receipts {ra:,.0f}/{rb:,.0f}); {txt}; "
               f"posterior {mu:.3f} +/- {np.sqrt(var):.3f}")


def race_rows(m: model.Model, race: pd.Series, kal: pd.DataFrame) -> list[dict]:
    rid = race["race_id"]
    names, parties = kal["candidate"].tolist(), kal["party"].tolist()
    q = np.full(len(kal), EPS_MINOR)
    side = np.array(["minor"] * len(kal), dtype=object)
    note = ""
    if race["non_partisan_final"]:
        named = [i for i, n in enumerate(names) if not n.endswith(" party")]
        ia, ib = named[0], named[1]
        p, note = non_partisan_prob(m, rid, names[ia], names[ib])
        q[ia], q[ib] = p, 1 - p
        # Kalshi labels both finalists UNK: use their real party (8 D-v-D finals, CA-40 R-v-R)
        real = "REP" if rid in SAME_PARTY_REP else "DEM"
        parties = [real if pt == "UNK" else pt for pt in parties]
        side[[ia, ib]] = "np"
    elif race["uncontested_d"]:
        q[[i for i, pt in enumerate(parties) if pt == "DEM"]] = UNCONTESTED_Q
        side[[i for i, pt in enumerate(parties) if pt == "DEM"]] = "np"
        note = "no Republican on the ballot"
    else:
        qa = float(race["qA"])
        if race["ind_side_a"]:
            a_idx = [i for i, n in enumerate(names) if last_name(n) == last_name(f26.IND_SIDE_A[rid])]
            note = f"side A = independent {f26.IND_SIDE_A[rid]}; Democrat priced as minor"
        else:
            a_idx = [i for i, pt in enumerate(parties) if pt == "DEM"]
        if race["ind_side_b"]:
            b_idx = [i for i, n in enumerate(names) if last_name(n) == last_name(f26.IND_SIDE_B[rid])]
            note = f"side B = independent {f26.IND_SIDE_B[rid]} (no Republican on the ballot)"
        else:
            b_idx = [i for i, pt in enumerate(parties) if pt == "REP"]
        for idx, mass, lab in ((a_idx, qa, "A"), (b_idx, 1 - qa, "B")):
            for i in idx:
                q[i] = mass / len(idx)
                side[i] = lab
        poll_txt = (f"poll avg {race['poll_share']:.3f} (n={int(race['poll_n'])}, neff {race['poll_neff']:.2f}, "
                    f"weight {race['poll_w']:.2f})" if pd.notna(race["poll_share"]) else "no usable poll")
        note = (note + "; " if note else "") + (f"prior {race['prior']:.3f}, {poll_txt}, mu {race['mu']:.3f}, "
                                                f"sd {race['sd']:.3f}, rating {race['rating']:+.2f}")
    q = q / q.sum()
    return [{"race_id": rid, "chamber": race["chamber"], "candidate": n, "party": pt, "q": qi,
             "p_kalshi": pk, "side": sd, "note": note}
            for n, pt, qi, pk, sd in zip(names, parties, q, kal["p_kalshi"], side)]


def disagreements(pred: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in pred.groupby("race_id", sort=False):
        top = g.loc[g["q"].idxmax()]
        fav = g.loc[g["p_kalshi"].idxmax()]
        gap = fav["q"] - fav["p_kalshi"]
        if top["candidate"] == fav["candidate"] and abs(gap) <= DISAGREE_GAP:
            continue
        ev = g.assign(ev=g["q"] * -np.log2(g["p_kalshi"].clip(lower=0.01)))
        evp = ev.loc[ev["ev"].idxmax()]
        kind = "different favourite" if top["candidate"] != fav["candidate"] else "same favourite, |q-p|>0.10"
        d = detail.set_index("race_id").loc[rid]
        reason = reasoning(d, top, fav)
        rows.append({"race_id": rid, "chamber": g["chamber"].iloc[0], "kind": kind,
                     "kalshi_fav": fav["candidate"], "kalshi_fav_party": fav["party"], "p_kalshi_fav": fav["p_kalshi"],
                     "q_kalshi_fav": fav["q"], "our_top": top["candidate"], "our_top_party": top["party"],
                     "q_our_top": top["q"], "p_kalshi_our_top": top["p_kalshi"],
                     "ev_pick": evp["candidate"], "ev_points": evp["ev"], "reasoning": reason})
    return pd.DataFrame(rows)


def reasoning(d: pd.Series, top: pd.Series, fav: pd.Series) -> str:
    if d["non_partisan_final"] or d["uncontested_d"]:
        return top["note"]
    parts = [f"fundamentals prior A-share {d['prior']:.3f} (lean {d['lean']:+.3f}, rating {d['rating']:+.2f}, "
             f"inc {int(d['inc'])}, money {d['money']:+.2f}, env {d['env']:+.3f})"]
    if pd.notna(d["poll_share"]):
        parts.append(f"poll avg {d['poll_share']:.3f} from {int(d['poll_n'])} polls (neff {d['poll_neff']:.2f}, "
                     f"weight {d['poll_w']:.2f})")
    else:
        parts.append("no usable poll in the last 75 days, so the prior decides")
    parts.append(f"posterior A-share {d['mu']:.3f} +/- {d['sd']:.3f} -> q(A) {d['qA']:.2f}")
    parts.append(f"Kalshi favourite {fav['candidate']} p={fav['p_kalshi']:.2f}, our q={fav['q']:.2f}")
    return "; ".join(parts)


def adjust_independents(det: pd.DataFrame, m: model.Model) -> pd.DataFrame:
    ind = det["ind_side_a"].to_numpy()
    mu = det["mu"].to_numpy() - np.where(ind, IND_POLL_FADE * det["poll_w"].to_numpy(), 0.0)
    sd = np.sqrt(det["sd"].to_numpy() ** 2 + np.where(ind, IND_EXTRA_SD ** 2, 0.0))
    sd_local = np.sqrt(det["sd_local"].to_numpy() ** 2 + np.where(ind, IND_EXTRA_SD ** 2, 0.0))
    return det.assign(mu=mu, sd=sd, sd_local=sd_local, qA=model.win_prob(mu, sd, m.err.nu))


def sync_side_a(det: pd.DataFrame, pred: pd.DataFrame, nu: float) -> pd.DataFrame:
    """After the dossier blend, move each partisan race's mean so the simulation reproduces the final q."""
    a = pred[pred["side"] == "A"].groupby("race_id")["q"].sum()
    b = pred[pred["side"] == "B"].groupby("race_id")["q"].sum()
    qa = (a / (a + b)).reindex(det["race_id"]).to_numpy()
    partisan = ~(det["non_partisan_final"] | det["uncontested_d"]).to_numpy() & ~np.isnan(qa)
    qa_c = np.clip(np.nan_to_num(qa, nan=0.5), 1e-6, 1 - 1e-6)
    mu_new = 0.5 + det["sd"].to_numpy() * stats.t.ppf(qa_c, df=nu)
    changed = partisan & (np.abs(qa_c - det["qA"].to_numpy()) > 1e-9)
    return det.assign(mu=np.where(changed, mu_new, det["mu"]), qA=np.where(changed, qa_c, det["qA"]),
                      dossier_adjusted=changed)


def main() -> int:
    res = hist_features.load_results()
    train = dataset.training(HIST_YEARS)
    m = model.fit(train)
    global SP_MODEL_2026
    SP_MODEL_2026 = same_party.fit(HIST_YEARS, res, lambda y, r, n: special.fec_lookup(y, r, n)[1])
    feat = f26.build(res)
    ag, effects = f26.poll_table_2026(res)
    feat = feat.merge(ag, on="race_id", how="left")
    det = adjust_independents(m.predict(feat), m)
    # v3: pool with the ratings-only logistic in the D-vs-R races (not the independent-side races)
    eligible = ~(det["ind_side_a"] | det["ind_side_b"] | det["non_partisan_final"] | det["uncontested_d"]).to_numpy()
    det = pool.apply(det, pool.RatingLogit.fit(train), m.err.nu, eligible=eligible)
    kal = parse_kalshi()
    rows = []
    for _, race in det.iterrows():
        rows += race_rows(m, race, kal[kal["race_id"] == race["race_id"]].reset_index(drop=True))
    pred = pd.DataFrame(rows)
    pred_model = pred.copy()
    pred, dlog = dossiers.blend(pred, dossiers.load())
    det = sync_side_a(det, pred, m.err.nu)
    tmpl = pd.read_csv(TEMPLATE)
    assert set(pred["race_id"]) == set(tmpl["race_id"]), "pred must cover every template race"
    assert np.allclose(pred.groupby("race_id")["q"].sum(), 1.0)
    OUT.mkdir(exist_ok=True)
    pred = pred.merge(pred_model[["race_id", "candidate", "q"]].rename(columns={"q": "q_model"}),
                      on=["race_id", "candidate"], how="left")
    cols = ["race_id", "chamber", "candidate", "party", "q", "p_kalshi", "q_model", "side", "note"]
    pred = pred[cols]
    pred.to_csv(OUT / "pred_2026.csv", index=False)
    dlog.to_csv(OUT / "dossier_adjustments_2026.csv", index=False)
    det.to_csv(OUT / "race_detail_2026.csv", index=False)
    dis = disagreements(pred, det.assign(note=None))
    dis.to_csv(OUT / "disagreements_2026.csv", index=False)
    diag = {**m.diag, "same_party_model": vars(SP_MODEL_2026), "pool_w": pool.POOL_W, "env_2026": float(feat["env"].iloc[0]),
            "poll_sponsor_offsets_2026": effects.get("sponsor", {}),
            "n_races_polled_2026": int(feat["poll_share"].notna().sum()),
            "n_dossier_adjusted": int(len(dlog)),
            "e_dem_house_model": float(pred_model[(pred_model["chamber"] == "House")
                                                  & (pred_model["party"] == "DEM")]["q"].sum()),
            "e_dem_house_final": float(pred[(pred["chamber"] == "House") & (pred["party"] == "DEM")]["q"].sum()),
            "n_different_favourite": int((dis["kind"] == "different favourite").sum()) if len(dis) else 0}
    (OUT / "model_2026_diag.json").write_text(json.dumps(diag, indent=2, default=float))
    print(json.dumps({k: diag[k] for k in ("env_2026", "n_races_polled_2026", "n_different_favourite",
                                           "n_dossier_adjusted", "e_dem_house_model", "e_dem_house_final")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
