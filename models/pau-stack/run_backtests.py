"""Re-run the evidence files in backtests/ for the no-primaries stack ('node+geo+field+rchg:clip2'), as pk_proto.py did:
  loyo_backtest.csv  every cycle 2010-2024 predicted by the model trained on the other seven (backtest.loyo_all)
  backtest_2018.csv, backtest_2022.csv  trained on 2010..T-2 only, all races in 538's list (backtest.run_year)
and score them: log loss (natural log, 1% floor) over every race won by a D or an R, and pau-t's referee-style LOYO
(all races with a winner except same-party contests, 3-way q). Writes out/backtests/<model>/ and compares with the
committed files in backtests/ (no-primaries stack), backtests/stack/ (with primaries) or backtests/v6nl/ (base).
Usage: python run_backtests.py [--name NAME]   (NAME as in src/pk_run.py; default the no-primaries stack;
       'prim+node+geo+field+rchg:clip2' = the stack with primaries, 'base' = v6+nl)"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import pk_stack as K  # noqa: E402
import pk_run as R    # noqa: E402
import backtest as B  # noqa: E402
from common import FEAT  # noqa: E402

REFS = {"node+geo+field+rchg:clip2": "", "prim+node+geo+field+rchg:clip2": "stack", "base": "v6nl"}


def ll_won(path, hist):
    """log loss as reported in README.md (and reproduced by Kavi's side, board K025): every race won by a Democrat or
    a Republican (n = 3,756 for 2010-2024), -ln max(q of the winner's party, 0.01)"""
    l = pd.read_csv(path).merge(hist[["year", "race_id", "winner_party"]], on=["year", "race_id"], how="left")
    m = l[l.winner_party.isin(["DEM", "REP"])]
    q = np.where(m.winner_party == "DEM", m.q_dem, m.q_rep).astype(float)
    ll = -np.log(np.clip(q, 0.01, 1))
    return dict(ll=round(float(ll.mean()), 4), n=int(len(m)),
                per_year={int(y): round(float(ll[(m.year == y).values].mean()), 4) for y in sorted(m.year.unique())})


def referee_loyo(path, hist):
    """pk_proto.referee_loyo (pau-t's scorer): all races with a winner except same-party contests, 3-way q"""
    l = pd.read_csv(path)
    m = l.merge(hist[["year", "race_id", "winner_party", "contest_type"]], on=["year", "race_id"], how="left")
    m = m[~m.contest_type.fillna("").isin(["D_vs_D", "R_vs_R"]) & m.winner_party.notna()]
    q = np.where(m.winner_party == "DEM", m.q_dem, np.where(m.winner_party == "REP", m.q_rep, m.q_other))
    ll = -np.log(np.clip(q.astype(float), 0.01, 1))
    return dict(ll=round(float(ll.mean()), 4), n=int(len(m)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="node+geo+field+rchg:clip2")
    ap.add_argument("--out", help="output folder (default out/backtests/<model>)")
    a = ap.parse_args()
    sub = REFS.get(a.name, "other")
    out = Path(a.out).resolve() if a.out else HERE / "out" / "backtests" / (sub or "stack_noprim")
    out.mkdir(parents=True, exist_ok=True)
    B.OUT = B.WORK = out                 # backtest.py writes its files here (not into data/)
    comps, kw, _, _ = R.parse(a.name)
    cfg = K.build(comps, **kw)
    hist = K.HIST
    f538 = pd.read_csv(FEAT / "forecast538_history.csv")
    t0 = time.time()
    res = {"name": a.name}
    for T in (2018, 2022):
        r = B.run_year(hist, T, f538, cfg=cfg)
        r = r[0] if isinstance(r, tuple) else r
        res[f"bt{T}"] = dict(all=r["All"]["model"], close_538_fav_lt95=r["All"]["competitive_538lt95"]["model"],
                             points_vs_538_market=r["All"]["points_vs_538_market"])
    B.loyo_all(hist, cfg=cfg)
    res["secs"] = round(time.time() - t0, 1)
    res["loyo_ll"] = ll_won(out / "loyo_backtest.csv", hist)
    res["loyo_referee"] = referee_loyo(out / "loyo_backtest.csv", hist)
    res["vs_backtests_folder"] = {}
    for f in ("loyo_backtest", "backtest_2018", "backtest_2022"):
        ref = HERE / "backtests" / sub / f"{f}.csv"
        if ref.exists():
            x = pd.read_csv(out / f"{f}.csv").merge(pd.read_csv(ref), on=["year", "race_id", "chamber"],
                                                     suffixes=("", "_ref"), how="outer", indicator=True)
            res["vs_backtests_folder"][f] = dict(
                n=len(x), unmatched=int((x._merge != "both").sum()),
                max_abs=float(max(np.nanmax(np.abs(x[c] - x[c + "_ref"])) for c in ("q_dem", "q_rep", "q_other"))))
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
