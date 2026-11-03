"""Out-of-sample backtests: for Y in (2018, 2022) fit on years < Y only, predict Y, write out/backtest_<Y>.csv.

Inputs used for year Y: Y's lean (previous presidential result), ballot (who the nominees are, incumbency), final
pre-election expert ratings, polls ending before election day, and the national-environment estimate built from
pre-election signals (national.py). Not used: Y results (except as the scoring truth), FEC (the historical FEC
totals run to Dec 31 of the election year, i.e. past election day), 538's forecast.
Run: .venv/bin/python src/models/kavi-e/backtest.py [--variant NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import independents
import model
import national
import within_party
from allocate import MAJOR_IND, allocate
from common import BENCH, OUT, WORK, ensure_dirs, override_incumbent, read, same_person
from polls import history_polls
from ratings_extra import rgap_hist

# ablation switches for the v1 -> v2 effect table only (KAVI_E_ABLATE="no_within,no_money")
ABLATE = set(filter(None, os.environ.get("KAVI_E_ABLATE", "").split(",")))
VARIANTS = {
    "main": {"house": model.HOUSE_FEATURES, "senate": model.SENATE_FEATURES},
    "no_rating": {"house": tuple(f for f in model.HOUSE_FEATURES if f not in ("rating", "rgap")),
                  "senate": tuple(f for f in model.SENATE_FEATURES if f not in ("rating", "rgap"))},
    "with_prev": {"house": model.HOUSE_FEATURES_PREV, "senate": model.SENATE_FEATURES},
}


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    house = pd.read_csv(WORK / "house_hist.csv")
    hp = history_polls()
    hp = hp[hp["chamber"] == "House"][["year", "race_id", "poll_margin", "poll_neff"]]
    house = house.merge(hp, on=["year", "race_id"], how="left")
    senate = pd.read_csv(WORK / "senate_hist.csv")
    gap = rgap_hist()
    house = house.merge(gap, on=["year", "race_id"], how="left")
    senate = senate.merge(gap, on=["year", "race_id"], how="left")
    return house, senate


def bench_incumbency(bench: pd.DataFrame, year: int) -> pd.Series:
    """1 if the benchmark candidate won this chamber's seat in the state in an earlier cycle (name match), or FEC
    calls them the incumbent."""
    rh, rs = read("results_house.csv"), read("results_senate.csv")
    fec = read("fec_hist.csv")
    fec = fec[(fec["year"] == year) & (fec["ici"] == "I")]
    flags = []
    for _, b in bench.iterrows():
        st = b["race_id"][:2]
        if b["chamber"] == "House":
            prev = rh[(rh["year"] == year - 2) & (rh["state"] == st)]["winner_name"]
            f = fec[fec["race_id"].str.startswith(st) & ~fec["race_id"].str.contains("SEN")]["cand_name"]
        else:
            prev = rs[(rs["year"] < year) & (rs["year"] >= year - 6) & (rs["state"] == st)]["winner_name"]
            f = fec[fec["race_id"] == f"{st}-SEN"]["cand_name"]
        hit = any(same_person(b["candidate"], w) for w in prev.dropna())
        hit = hit or any(same_person(b["candidate"], n) for n in f.dropna())
        hit = hit or override_incumbent(year, b["race_id"], b["party"], b["candidate"])
        flags.append(int(hit))
    return pd.Series(flags, index=bench.index)


def race_probs(frame: pd.DataFrame, year: int, features: tuple[str, ...], env: dict,
               bench: pd.DataFrame, clf_extra: pd.DataFrame | None = None) -> tuple[pd.DataFrame, model.Fitted]:
    extra = None if clf_extra is None else clf_extra[clf_extra["year"] < year]
    fit = model.fit(frame[frame["year"] < year], features, clf_extra=extra)
    target = frame[(frame["year"] == year) & frame["L"].notna()].copy()
    # major-independent races: incumbency from the benchmark ballot, polls of the D-v-R pair are meaningless
    for rid, surname in MAJOR_IND.get(year, {}).items():
        sel = target["race_id"] == rid
        if not sel.any():
            continue
        cands = bench[bench["race_id"] == rid]
        d_inc = int(cands.loc[cands["candidate"].str.contains(surname), "is_incumbent"].max())
        r_inc = int(cands.loc[cands["party"] == "REP", "is_incumbent"].max())
        target.loc[sel, "inc"] = d_inc - r_inc
        # v2: polls are the independent-vs-R polls (v1 dropped them: McMullin got 0.0009 from fundamentals);
        # challengers' polls shifted by the shrunk independent-specific miss of EARLIER cases (independents.py)
        if (year, rid) not in independents.INCUMBENT_IND:
            target.loc[sel, "poll_margin"] += independents.poll_shift(year, frame)
            target.loc[sel, "sigma_mult"] = independents.SIGMA_MULT
    pred = model.predict(fit, target, env["n_hat"], env["sigma_n"])
    return pred, fit


def run_year(year: int, variant: str, house: pd.DataFrame, senate: pd.DataFrame) -> dict:
    bench = pd.read_csv(BENCH / f"fte_final_{year}.csv")
    bench["is_incumbent"] = bench_incumbency(bench, year)
    env = national.estimate(year)
    feats = VARIANTS[variant]
    ph, fh = race_probs(house, year, feats["house"], env, bench)
    ps, fs = race_probs(senate, year, feats["senate"], env, bench, clf_extra=house)
    p_d = pd.concat([ph, ps]).set_index("race_id")["p_d"]
    rows = []
    for rid, cands in bench.groupby("race_id", sort=False):
        major = cands[cands["party"].isin(["DEM", "REP"])]
        if rid[:2] in ("CA", "WA") and major["party"].duplicated().any() and "no_within" not in ABLATE:
            # money only where a pre-election FEC cut exists (2018 Wayback snapshot); 2022: no-money model
            rec = cands["candidate"].map(lambda n: within_party.pre_receipts_2018(rid, n)) if (year == 2018 and "no_money" not in ABLATE) else None
            cands = cands.assign(sp_score=within_party.scores(year, rid, cands, rec))
        q = allocate(year, rid, cands, p_d.get(rid))
        for i, c in cands.iterrows():
            rows.append({"year": year, "race_id": rid, "candidate": c["candidate"], "party": c["party"],
                         "q": round(float(q.loc[i]), 6)})
    out = pd.DataFrame(rows)
    suffix = "" if variant == "main" else f"_{variant}"
    out.to_csv(OUT / f"backtest_{year}{suffix}.csv", index=False)
    pd.concat([ph.assign(chamber="House"), ps.assign(chamber="Senate")]).to_csv(
        WORK / f"backtest_races_{year}{suffix}.csv", index=False)
    return {"year": year, "variant": variant, "env": env, "house_fit": model.describe(fh),
            "senate_fit": model.describe(fs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main", choices=sorted(VARIANTS))
    args = ap.parse_args()
    ensure_dirs()
    house, senate = load_frames()
    info = [run_year(y, args.variant, house, senate) for y in (2018, 2022)]
    suffix = "" if args.variant == "main" else f"_{args.variant}"
    (OUT / f"backtest_fits{suffix}.json").write_text(json.dumps(info, indent=2, default=float))
    for i in info:
        print(i["year"], "N_hat", round(i["env"]["n_hat"], 2), "house", i["house_fit"]["coef"], "tau",
              i["house_fit"]["tau"], "| senate tau", i["senate_fit"]["tau"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
