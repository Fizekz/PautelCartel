"""pau-e: observer innovations in expert ratings. Rating change over the final 5 weeks (eve revision minus the E-35
revision of the same Wikipedia pages, both pre-election by construction; pau-r's parser), and the part of it the
public polls and the national environment do not explain.
- rchg: mean over outlets of (eve - early), on the D-side scale (+ = moved toward the D-side), races rated on both
  revisions (else 0).
- rchg_res: residual of rchg after a within-year regression on the race's poll movement over the window (late
  21-day poll average minus the 120-day average, D-side, 0 if unpolled), a polled flag and the year mean
  (so the national drift is removed). Within-year, so no outcome enters.
Outputs work/pe_rchg_features.csv; the eve ratings used here are the eve revisions, not the retrospective layer."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[2] / "ratings"
sys.path.insert(0, str(R))
from parse import parse  # noqa: E402
from common import WORK  # noqa: E402
from pe_util import ols  # noqa: E402
EARLY = Path(__file__).resolve().parents[1] / "data" / "raw_e" / "wiki_early"


def load(days=35):
    rows_e, rows_l = [], []
    for y in range(2010, 2026, 2):
        for kind, sen in (("house", False), ("senate", True)):
            f = EARLY / f"{y}_{kind}_e{days}.wikitext"
            if f.exists():
                rows_e += parse(f.read_text(), sen, y)
        fs = [("2018_house_template", False), ("2018_senate_template", True)] if y == 2018 else [(f"{y}_house", False), (f"{y}_senate", True)]
        for k, sen in fs:
            rows_l += parse((R / "asof" / f"{k}.wikitext").read_text(), sen, y)
    e = pd.DataFrame(rows_e).drop_duplicates(["year", "race_id", "outlet"], keep="last")
    l = pd.DataFrame(rows_l).drop_duplicates(["year", "race_id", "outlet"], keep="last")
    m = e.merge(l, on=["year", "race_id", "outlet"], suffixes=("_e", "_l"))
    m["chg"] = -(m.num_l - m.num_e)                 # parser scale is R-positive; D-side + = toward D
    return e, l, m


def main():
    e, l, m = load()
    print("early ratings parsed per year:", e.groupby("year").size().to_dict())
    print("matched race-outlets per year:", m.groupby("year").size().to_dict(), "| changed share", (m.chg != 0).mean().round(3))
    g = m.groupby(["year", "race_id"]).chg.mean().rename("rchg").reset_index()
    h = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
    d = h[["year", "race_id", "senate", "poll_margin", "poll_pollsters", "pm4", "pn4", "y", "dwin", "rating", "dside_party"]].merge(g, on=["year", "race_id"], how="inner")
    late = d.poll_margin.where(d.poll_pollsters.fillna(0) > 0)
    early = d.pm4.where(d.pn4.fillna(0) > 0)
    d["dpoll"] = (late - early).fillna(0.0).clip(-15, 15)
    d["pol"] = (d.poll_pollsters.fillna(0) > 0).astype(float)
    d["rchg_res"] = 0.0
    for y, gg in d.groupby("year"):
        b, se, t = ols(gg.rchg, gg[["dpoll", "pol"]])
        X = np.column_stack([np.ones(len(gg)), gg[["dpoll", "pol"]].values])
        d.loc[gg.index, "rchg_res"] = gg.rchg.values - X @ b
        print(y, "n", len(gg), "changed", int((gg.rchg != 0).sum()), "mean change %.3f" % gg.rchg.mean(),
              "| rchg ~ dpoll %.3f (t %.1f)" % (b[1], t[1]))
    d[["year", "race_id", "rchg", "rchg_res", "dpoll"]].to_csv(WORK / "pe_rchg_features.csv", index=False)


if __name__ == "__main__":
    main()
