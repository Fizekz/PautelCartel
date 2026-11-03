"""pau-k: standard outputs (proto/pk_stack/out: backtest_2018.csv, backtest_2022.csv, loyo_backtest.csv) of the final
stack, the referee-style LOYO (pau-t's scorer, pt_proto.referee_loyo: all races with a winner except same-party
contests, 3-way q, 1% floor; it reproduces the referee's 0.0654 for proto/pattern_nl), and the 2018/2022 backtest
summaries. Usage: pk_proto.py [NAME]   (NAME as in pk_run.py, default the final stack)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pk_stack as K
import pk_run as R
import backtest as B
from common import FEAT

ROOT = Path(__file__).resolve().parents[2]
FINAL_NAME = "prim+node+geo+field+rchg:clip2"


def referee_loyo(path, hist):
    l = pd.read_csv(path)
    m = l.merge(hist[["year", "race_id", "winner_party", "contest_type"]], on=["year", "race_id"], how="left")
    m = m[~m.contest_type.fillna("").isin(["D_vs_D", "R_vs_R"]) & m.winner_party.notna()]
    q = np.where(m.winner_party == "DEM", m.q_dem, np.where(m.winner_party == "REP", m.q_rep, m.q_other))
    ll = -np.log(np.clip(q.astype(float), 0.01, 1))
    return round(float(ll.mean()), 4), {int(y): round(float(ll[(m.year == y).values].mean()), 4) for y in sorted(m.year.unique())}


def main(name, outname="pk_stack"):
    comps, kw, _, _ = R.parse(name)
    cfg = K.build(comps, **kw)
    out = ROOT / "proto" / outname / "out"
    out.mkdir(parents=True, exist_ok=True)
    B.OUT = out
    hist = K.HIST
    f538 = pd.read_csv(FEAT / "forecast538_history.csv")
    res = {"variant": name}
    for T in (2018, 2022):
        r = B.run_year(hist, T, f538, cfg=cfg)
        r = r[0] if isinstance(r, tuple) else r
        res[f"bt{T}"] = dict(all=r["All"]["model"], close=r["All"]["competitive_538lt95"]["model"],
                             points=r["All"]["points_vs_538_market"])
    B.loyo_all(hist, cfg=cfg)
    res["referee_loyo"] = {n: referee_loyo(p, hist) for n, p in
                           ((outname, out / "loyo_backtest.csv"),
                            ("v6+nl (proto/pattern_nl)", ROOT / "proto" / "pattern_nl" / "out" / "loyo_backtest.csv"))}
    json.dump(res, open(K.RES / f"proto_{outname}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else FINAL_NAME, a[1] if len(a) > 1 else "pk_stack")
