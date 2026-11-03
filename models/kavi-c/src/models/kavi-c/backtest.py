"""Out-of-sample backtests: predict 2018 and 2022 using only elections before that year.

For target year Y the model is fitted on contested races of 2010..Y-2 (features, polls and results of those
years); Y's own features are pre-election inputs (lean, final ratings, incumbency, receipts, Y's polls, Y's final
generic-ballot average). Y's results are used only to know who was on the ballot (nominee names, which seats were
uncontested), which is public before election day. 538's forecast is never an input.

Usage: python src/models/kavi-c/backtest.py   -> out/backtest_2018.csv, out/backtest_2022.csv, out/backtest_diag.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import model  # noqa: E402
import polls  # noqa: E402
import pool  # noqa: E402
import same_party  # noqa: E402
import special  # noqa: E402
from common import BENCH, HIST_YEARS, OUT, last_name  # noqa: E402

EPS_MINOR = 0.001    # probability mass for a minor candidate
# Backtest years with no pre-election FEC cut (2018 uses the election-day Wayback snapshot instead): receipts are
# removed from every input of that backtest, i.e. the model is refitted without the money feature and same-party
# splits use incumbency only. Set to empty to reproduce the leaky full-cycle variant.
NO_MONEY_BACKTEST: set[int] = set()   # v2: {2022}; v3 uses the June 30 2022 cut instead
PRE_CUT_2022 = True
USE_PRIMARY_SPLIT = True                  # v3: same-party splits from the June top-two primary (same_party.py)                       # v3: 2022 test-year receipts from the pre-election June cut
UNCONTESTED_Q = 0.998

# Races whose real contest is not D vs R, known before election day: (A, B) finalists by name.
SPECIAL_HIST = {
    (2018, "ME-SEN"): ("Angus S. King Jr.", "Eric L. Brakey"),       # independent vs R
    (2018, "VT-SEN"): ("Bernard Sanders", "Lawrence Zupan"),          # independent vs R
    (2018, "CA-SEN"): ("Dianne Feinstein", "Kevin de Leon"),          # D vs D top-two
    (2022, "UT-SEN"): ("Evan McMullin", "Mike Lee"),                  # independent (no D) vs R
    (2022, "AK-SEN"): ("Lisa Murkowski", "Kelly C. Tshibaka"),        # R vs R under RCV
}


def special_two_way(m: model.Model, year: int, row: pd.Series, a: str, b: str, polls: pd.DataFrame,
                    national_shared: bool) -> float:
    """P(A beats B) from the A-vs-B poll average; falls back to the rating when unpolled."""
    share, neff = special.two_way_poll_share(polls, year, a, b)
    if np.isnan(share):
        return float(1 / (1 + np.exp(1.8 * row["rating"])))  # A is the non-R side by convention
    pseudo = pd.DataFrame({"poll_neff": [neff], "rating": [0.0], "senate": [1]})
    sd_loc = m.err.sd_local(pseudo)[0]
    sd = m.err.scale_sen * np.sqrt(sd_loc ** 2 + (m.err.sd_nat ** 2 if national_shared else 0.0))
    return float(model.win_prob(np.array([share]), np.array([sd]), m.err.nu)[0])


def race_probs(m: model.Model, year: int, bench: pd.DataFrame, feat: pd.Series | None,
               polls: pd.DataFrame) -> np.ndarray:
    """q for each benchmark candidate of one race."""
    rid = bench["race_id"].iloc[0]
    names, parties = bench["candidate"].tolist(), bench["party"].tolist()
    q = np.full(len(bench), EPS_MINOR)
    if (year, rid) in SPECIAL_HIST:
        a, b = SPECIAL_HIST[(year, rid)]
        ia = [i for i, n in enumerate(names) if last_name(n) == last_name(a)][0]
        ib = [i for i, n in enumerate(names) if last_name(n) == last_name(b)][0]
        pa = special_two_way(m, year, feat, a, b, polls[polls["race_id"] == rid],
                             national_shared=parties[ib] == "REP" and parties[ia] != "REP")
        q[ia], q[ib] = pa, 1 - pa
        return q / q.sum()
    d_idx = [i for i, p in enumerate(parties) if p == "DEM"]
    r_idx = [i for i, p in enumerate(parties) if p == "REP"]
    if feat is None:
        return np.full(len(bench), 1 / len(bench))
    if feat["same_party"]:
        idx = d_idx if len(d_idx) >= len(r_idx) else r_idx
        if len(idx) < 2:  # the benchmark lists only one finalist of the locked-out party
            q[idx] = UNCONTESTED_Q
            return q / q.sum()
        split = special.split_within_party(year, rid, [names[i] for i in idx])
        if len(idx) == 2:
            share, neff = special.two_way_poll_share(polls[polls["race_id"] == rid], year, names[idx[0]], names[idx[1]])
            if not np.isnan(share):
                pa = special_two_way(m, year, feat, names[idx[0]], names[idx[1]], polls[polls["race_id"] == rid], False)
                split = np.array([pa, 1 - pa])
        for i, s in zip(idx, split):
            q[i] = s
        return q / q.sum()
    if feat["contested"] or feat["rcv"] or feat["major_other"]:
        # v2: races with a major third candidate use the model's pre-election qA (v1 read the actual result, audit L2)
        qd = float(np.clip(feat["qA"], 1 - UNCONTESTED_Q, UNCONTESTED_Q))
    else:  # one major party on the ballot: which one is ballot information, known before election day
        qd = UNCONTESTED_Q if isinstance(feat["dem_cand"], str) else 1 - UNCONTESTED_Q
    for idx, mass in ((d_idx, qd), (r_idx, 1 - qd)):
        if not idx:
            continue
        split = special.split_within_party(year, rid, [names[i] for i in idx]) if len(idx) > 1 else np.ones(1)
        for i, s in zip(idx, split):
            q[i] = max(mass * s, EPS_MINOR * 0.1)
    return q / q.sum()


def set_fec_cut(use_2022_pre: bool) -> None:
    """Switch 2022 receipts between full-cycle (training) and the pre-election cut (2022 test year)."""
    if dataset.hist_features.USE_FEC_2022_PRE != use_2022_pre:
        dataset.hist_features.USE_FEC_2022_PRE = use_2022_pre
        dataset.features.cache_clear()


def run_year(year: int) -> tuple[pd.DataFrame, dict]:
    set_fec_cut(PRE_CUT_2022 and year == 2022)
    train = dataset.training([y for y in HIST_YEARS if y < year])
    test = dataset.year_table(year)
    no_money = year in NO_MONEY_BACKTEST
    special.NO_RECEIPT_YEARS = {year} if no_money else set()
    # v3: same-party finals from the June primary, model fitted on earlier years only
    special.SP_MODEL = same_party.fit([y for y in HIST_YEARS if y < year], dataset.hist_features.load_results(),
                                      lambda y, r, n: special.fec_lookup(y, r, n)[1]) if USE_PRIMARY_SPLIT else None
    if no_money:
        train, test = train.assign(money=0.0), test.assign(money=0.0)
    m = model.fit(train)
    # v3: log-linear pool with a ratings-only logistic fitted on the same training years (pool.POOL_W)
    feat = pool.apply(m.predict(test), pool.RatingLogit.fit(train), m.err.nu).set_index("race_id")
    bench = pd.read_csv(BENCH / f"fte_final_{year}.csv")
    hp = pd.read_csv(dataset.hist_features.CLEAN / "polls_hist.csv")
    hp = hp[(hp["year"] == year) & (hp["stage"] == "general") & ~hp["is_hypothetical"].fillna(False).astype(bool)]
    hp = polls.rcv_final_only(hp)
    rows = []
    for rid, b in bench.groupby("race_id", sort=False):
        f = feat.loc[rid] if rid in feat.index else None
        q = race_probs(m, year, b.reset_index(drop=True), f, hp)
        rows.append(pd.DataFrame({"year": year, "race_id": rid, "candidate": b["candidate"].to_numpy(),
                                  "party": b["party"].to_numpy(), "q": q}))
    out = pd.concat(rows, ignore_index=True)
    special.NO_RECEIPT_YEARS = set()
    sp_diag = vars(special.SP_MODEL) if special.SP_MODEL is not None else None
    special.SP_MODEL = None
    set_fec_cut(False)
    return out, {"year": year, "train_years": sorted(train["year"].unique().tolist()), "money_used": not no_money, "same_party_model": sp_diag,
                 **m.diag}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    diags = []
    for year in (2018, 2022):
        out, diag = run_year(year)
        out.to_csv(OUT / f"backtest_{year}.csv", index=False)
        diags.append(diag)
        print(year, json.dumps({k: diag[k] for k in ("k_house", "k_sen", "sd_nat", "nu", "scale_house", "scale_sen", "floor")}))
    (OUT / "backtest_diag.json").write_text(json.dumps(diags, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
