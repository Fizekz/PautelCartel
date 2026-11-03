"""pau-k: 2026 probabilities of v6+nl, the final stack and the no-primaries variant (all trained on 2010-2024), against
Kalshi's Nov 1 price (the earliest Nov 1 snapshot, .git/wt/market/data/kalshi/kalshi_races_2026-11-01T074812Z.csv,
07:48Z; read-only; it is the benchmark-day price, long before the 23:00Z Nov 3 cut).
pau-e's rater innovation is rebuilt from the 2026 rating pages (E-35 = 2026-09-29, eve = 2026-11-02T00:00Z, pau-e's
MediaWiki-API revisions, copied to work/stack/wiki) against the poll file in use (pe_rchg_2026.py's regression).
Usage: pk_2026.py [FEATURES_2026_CSV] [TAG]  -> results/pk/pk_2026_<TAG>.csv and .json"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pk_stack as K
import pk_run as R
from model import WinModel
from common import WORK

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ratings"))
from parse import parse  # noqa: E402

KAL = ROOT / "data" / "benchmark" / "kalshi_races_2026-11-01T074812Z.csv"  # reference column only
WIKI = WORK / "stack" / "wiki"
MODELS = {"v6nl": "base", "stack": "prim+node+geo+field+rchg:clip2", "noprim": "node+geo+field+rchg:clip2"}


def rchg_2026(f26):
    rows = {}
    for tag in ("e35", "eve"):
        rows[tag] = []
        for k in ("house", "senate"):
            rows[tag] += parse((WIKI / f"2026_{k}_{tag}.wikitext").read_text(), k == "senate", 2026)
    e = pd.DataFrame(rows["e35"]).drop_duplicates(["race_id", "outlet"], keep="last")
    l = pd.DataFrame(rows["eve"]).drop_duplicates(["race_id", "outlet"], keep="last")
    m = e.merge(l, on=["race_id", "outlet"], suffixes=("_e", "_l"))
    m["chg"] = -(m.num_l - m.num_e)
    g = m.groupby("race_id").chg.mean().rename("rchg").reset_index()
    d = f26.merge(g, on="race_id", how="inner")
    late = d.poll_margin.where(d.poll_pollsters.fillna(0) > 0)
    early = d.pm4.where(d.pn4.fillna(0) > 0)
    d["dpoll"] = (late - early).fillna(0.0).clip(-15, 15)
    d["pol"] = (d.poll_pollsters.fillna(0) > 0).astype(float)
    X = np.column_stack([np.ones(len(d)), d[["dpoll", "pol"]].values])
    b = np.linalg.lstsq(X, d.rchg.values, rcond=None)[0]
    d["rchg_res"] = d.rchg.values - X @ b
    return d[["race_id", "rchg", "rchg_res"]]


def kalshi():
    k = pd.read_csv(KAL)
    k = k[k.covered == 1]
    pD = np.where(k.fav_party == "REP", 1 - k.fav_norm, k.fav_norm)   # D-side = DEM, or IND in R_vs_IND races
    return dict(zip(k.race_id, pD))


def main(fpath, tag):
    f26 = pd.read_csv(fpath, low_memory=False)
    std = f26[f26.kind.isin(["D_vs_R", "R_vs_IND", "jungle_party"])].copy()
    std["year"] = 2026
    r = rchg_2026(std)
    std = std.merge(r[["race_id", "rchg_res"]], on="race_id", how="left")
    unp = ((std.poll_pollsters.fillna(0) == 0) & (std.pn4.fillna(0) == 0)).astype(float)
    std["rchg_unp"] = std.rchg_res.fillna(0) * unp * (std.rating.abs() < 2.5)
    Y = list(range(2010, 2026, 2))
    d = std[["race_id", "senate", "rating", "poll_pollsters", "rchg_unp"]].copy()
    d["polled"] = d.poll_pollsters.fillna(0) > 0
    for k, name in MODELS.items():
        comps, kw, _, _ = R.parse(name)
        d[f"p_{k}"] = WinModel(Y, K.build(comps, **kw)).fit(K.HIST).predict(std).p_dside.values
    kp = kalshi()
    d["k_pD"] = d.race_id.map(kp)
    d["prim_dev"] = [K.FEATS["prim_dev"].get((2026, r_), np.nan) for r_ in d.race_id]
    d = d.drop(columns=["poll_pollsters"])
    d.to_csv(K.RES / f"pk_2026_{tag}.csv", index=False)
    h, s, ok = d.senate == 0, d.senate == 1, d.k_pD.notna()
    pts = lambda p: -np.log2(np.clip(p, 0.01, 1))
    res = {"features": str(fpath), "n": int(len(d)), "n_kalshi": int(ok.sum()),
           "kalshi_expected_D_house": round(float(d.k_pD[h].sum()), 1), "kalshi_expected_D_senate": round(float(d.k_pD[s].sum()), 2)}
    for k in MODELS:
        p = d[f"p_{k}"]
        dis = d[ok & ((p >= 0.5) != (d.k_pD >= 0.5))]
        evD = p * pts(d.k_pD)
        evR = (1 - p) * pts(1 - d.k_pD)
        pickD = evD > evR
        res[k] = dict(expected_D_house=round(float(p[h].sum()), 1), expected_D_senate=round(float(p[s].sum()), 2),
                      fav_disagree_kalshi=dis[["race_id", f"p_{k}", "k_pD"]].round(3).values.tolist(),
                      mean_abs_vs_kalshi_house=round(float((p - d.k_pD)[h & ok].abs().mean()), 4),
                      ev_picks_against_kalshi_fav=int((ok & (pickD != (d.k_pD >= 0.5))).sum()),
                      expected_points_own_model=round(float(np.where(pickD, evD, evR)[ok].sum()), 1))
    d["dp"] = d.p_stack - d.p_v6nl
    top = d.reindex(d.dp.abs().sort_values(ascending=False).index).head(15)
    res["biggest_moves"] = top[["race_id", "rating", "polled", "p_v6nl", "p_stack", "p_noprim", "k_pD", "rchg_unp", "prim_dev"]].round(3).values.tolist()
    json.dump(res, open(K.RES / f"pk_2026_{tag}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "biggest_moves"}, indent=1, default=float))
    print(top[["race_id", "rating", "polled", "p_v6nl", "p_stack", "p_noprim", "k_pD", "rchg_unp", "prim_dev"]].round(3).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]) if a else WORK / "features_2026.csv", a[1] if len(a) > 1 else "current")
