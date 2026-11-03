"""Out-of-sample backtests on the 538 benchmark race lists (2018, 2022).

For year Y: the party model and the same-party model are fitted on cycles < Y only; Y's inputs are its final
ratings, the prior presidential lean, the previous cycle's result, FEC receipts and race polls in the final 60 days.
538's probabilities are never read; only its candidate list (race_id, candidate, party) is used as the ballot.

Usage: python backtest.py [--config blend_core_sym]   -> out/backtest_2018.csv, out/backtest_2022.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from assemble import candidate_q, classify, structure_row
from common import BENCH, CLEAN, OUT, ROOT, fec_surname_tokens, name_matches, prefer_given_name
from features import base_race_id, build_hist_features
from history import training_table
from model import Config, fit_predict
from sameparty import fit as fit_sameparty
from sameparty import pick_params, primary_pct

CHOSEN = "final_v3"


def chosen_config(name: str = CHOSEN) -> Config:
    from experiments import CONFIGS as V1
    from experiments_v2 import CONFIGS as V2
    from experiments_v3 import CONFIGS as V3
    return {**V1, **V2, **V3}[name]


FEC_2018_PRE = ROOT / "data" / "extra" / "fec_2018_preelection.csv"
# Backtest money policy (v2, audit fix): 2018 uses the pre-election Wayback snapshot of weball18 (borrowed from
# kavi-b); 2022 has no pre-election cut, so money is dropped (receipts NaN, incumbency-only same-party split).
MONEY_SOURCE = {2018: "preelection_snapshot", 2022: None}


def _snapshot_2018() -> pd.DataFrame:
    s = pd.read_csv(FEC_2018_PRE)
    s = s[s["office"].isin(["H", "S"])].copy()
    s["race_id"] = s["race_id_raw"].str.replace(r"-00$", "-AL", regex=True)
    s["_tok"] = s["cand_name"].map(fec_surname_tokens)
    return s


def _pre_receipts(snap: pd.DataFrame, race_id: str, cand_id, name: str) -> float:
    """Pre-election receipts: by FEC candidate id, else surname (+ given name) within the race."""
    hit = snap[snap["cand_id"] == cand_id]
    if hit.empty:
        g = snap[snap["race_id"] == base_race_id(race_id)]
        hit = prefer_given_name(g[g["_tok"].map(lambda t: name_matches(t, name))], name)
    return float(hit["receipts_pre"].max()) if not hit.empty else np.nan


def bench_candidates(year: int) -> pd.DataFrame:
    b = pd.read_csv(BENCH / f"fte_final_{year}.csv")[["year", "race_id", "chamber", "special", "candidate", "party"]]
    fec = pd.read_csv(CLEAN / "fec_hist.csv")
    fec = fec[fec["year"] == year].copy()
    fec["_tok"] = fec["cand_name"].map(fec_surname_tokens)
    groups = {k: g for k, g in fec.groupby("race_id")}
    snap = _snapshot_2018() if MONEY_SOURCE.get(year) == "preelection_snapshot" else None
    rec, inc = [], []
    for r in b.itertuples():
        g = groups.get(base_race_id(r.race_id))
        hit = None if g is None else prefer_given_name(g[g["_tok"].map(lambda t: name_matches(t, r.candidate))],
                                                      r.candidate)
        if hit is not None and not hit.empty and r.candidate != "Others":
            want = {"DEM": "DEM", "REP": "REP"}.get(r.party, "IND")
            if (hit["party_bucket"] == want).any():
                hit = hit[hit["party_bucket"] == want]
            row = hit.sort_values("receipts", ascending=False, na_position="last").iloc[0]
            inc.append(int(row["ici"] == "I"))  # FEC status flag: known before the election
            rec.append(np.nan if snap is None else _pre_receipts(snap, r.race_id, row["cand_id"], r.candidate))
        else:
            rec.append(np.nan)
            inc.append(0)
    b["receipts"], b["inc"] = rec, inc
    # Viable independent (money-free rule, v2): the race has no Democrat and the raters do not call it Safe R for
    # all three (UT-SEN 2022 McMullin), or the raters rate the race for an independent ("Safe I": King, Sanders)
    # and he is the side's leading independent (pre-election money, else the first listed). A big-name independent
    # next to a Democratic nominee (Crowley, WFP line, NY-14 2018) is not viable. Then a nominal Democrat is minor.
    rat = pd.read_csv(CLEAN / "ratings_hist.csv")
    rat = rat[rat["year"] == year]
    rated_ind = set(rat.loc[rat[["cook_raw", "sabato_raw", "inside_raw"]].astype(str)
                            .apply(lambda s: s.str.contains(r"\bI\b")).any(axis=1), "race_id"])
    not_safe = set(rat.loc[rat[["cook", "sabato", "inside"]].min(axis=1) < 3, "race_id"])
    has_dem = b.groupby("race_id")["party"].transform(lambda s: (s == "DEM").any())
    ind = ~b["party"].isin(["DEM", "REP"]) & (b["candidate"] != "Others")
    money = b["receipts"].where(ind).fillna(-1)
    top_ind = ind & (money == money.groupby(b["race_id"]).transform("max"))
    top_ind &= ~top_ind.groupby(b["race_id"]).cumsum().gt(1)  # one per race
    b["viable_ind"] = top_ind & ((~has_dem & b["race_id"].isin(not_safe)) | b["race_id"].isin(rated_ind))
    b["nominal_dem"] = (b["party"] == "DEM") & b["race_id"].isin(rated_ind) & \
        b.groupby("race_id")["viable_ind"].transform("any")
    return b


def predict_year(year: int, cfg: Config) -> pd.DataFrame:
    money = MONEY_SOURCE.get(year) is not None
    pair = (fit_sameparty(year, use_money=money), fit_sameparty(year, use_money=money, use_primary=False))
    params = pair[1]  # for choosing each side's principal (no primary shares there)
    cands = bench_candidates(year)
    races = {rid: classify(g) for rid, g in cands.groupby("race_id", sort=False)}
    for rid, g in races.items():
        g.loc[g["nominal_dem"], "side"] = "minor"
    struct = [structure_row(rid, g["chamber"].iloc[0], year, g, params, int(g["special"].iloc[0]))
              for rid, g in races.items()]
    struct = pd.DataFrame([s for s in struct if s is not None])
    feats = build_hist_features(struct)
    train = training_table()
    p, info = fit_predict(train, feats, cfg, year)
    p_by_race = dict(zip(feats["race_id"], p))
    print(year, "calibration", info, "party-model races", len(feats))
    out = []
    for rid, g in races.items():
        prim = {c: primary_pct(year, rid, c) for c in g["candidate"]} if rid[:2] in ("CA", "WA") else None
        q = candidate_q(g, p_by_race.get(rid), pick_params(pair, prim), primary_share=prim)
        out.append(g.assign(q=q))
    res = pd.concat(out, ignore_index=True)
    return res, feats.assign(p=p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CHOSEN)
    ap.add_argument("--outdir", default=None, help="write variant backtests here instead of out/")
    args = ap.parse_args()
    cfg = chosen_config(args.config)
    outdir = OUT if args.outdir is None else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for year in (2018, 2022):
        res, feats = predict_year(year, cfg)
        res[["year", "race_id", "candidate", "party", "q", "side"]].to_csv(outdir / f"backtest_{year}.csv", index=False)
        if args.outdir is None:
            feats.to_csv(OUT / "cache" / f"backtest_features_{year}.csv", index=False)


if __name__ == "__main__":
    main()
