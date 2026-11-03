"""pau-e: 2026 rater innovation. Fetch the 2026 rating pages as of E-35 (2026-09-29) and as of 2026-11-02T00:00Z
(MediaWiki API, the client in pe_rchg_fetch.py), parse with pau-r's parser, build rchg_res for unpolled rated races
the same way as history (within-year regression on poll movement), and show the shift it implies (coefficient from
the loyo8-years fit of v6+nl + rchg_unp). Output work/pe_rchg_2026.csv."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import pe_rchg_fetch as F
from pe_rchg import R, parse
from common import WORK
OUT = Path(__file__).resolve().parents[1] / "data" / "raw_e" / "wiki_early"
pages = {"2026_house": "2026 United States House of Representatives election ratings",
         "2026_senate": "2026 United States Senate elections"}
rows = {}
for tag, ts in (("e35", "2026-09-29T00:00:00Z"), ("eve", "2026-11-02T00:00:00Z")):
    rows[tag] = []
    for k, t in pages.items():
        f = OUT / f"{k}_{tag}.wikitext"
        if not f.exists():
            r = F.rev_asof(t, ts)
            if r is None:
                print(k, tag, "no revision"); continue
            f.write_text(r["content"]); print(k, tag, r["revid"], r["timestamp"], len(r["content"]), flush=True)
        rows[tag] += parse(f.read_text(), k.endswith("senate"), 2026)
e = pd.DataFrame(rows["e35"]).drop_duplicates(["race_id", "outlet"], keep="last")
l = pd.DataFrame(rows["eve"]).drop_duplicates(["race_id", "outlet"], keep="last")
print("parsed 2026 ratings: E-35", len(e), "eve", len(l))
m = e.merge(l, on=["race_id", "outlet"], suffixes=("_e", "_l"))
m["chg"] = -(m.num_l - m.num_e)
g = m.groupby("race_id").chg.mean().rename("rchg").reset_index()
f26 = pd.read_csv(WORK / "features_2026.csv", low_memory=False)
d = f26.merge(g, on="race_id", how="inner")
late = d.poll_margin.where(d.poll_pollsters.fillna(0) > 0)
early = d.pm4.where(d.pn4.fillna(0) > 0)
d["dpoll"] = (late - early).fillna(0.0).clip(-15, 15)
d["pol"] = (d.poll_pollsters.fillna(0) > 0).astype(float)
X = np.column_stack([np.ones(len(d)), d[["dpoll", "pol"]].values])
b = np.linalg.lstsq(X, d.rchg.values, rcond=None)[0]
d["rchg_res"] = d.rchg.values - X @ b
unp = (d.poll_pollsters.fillna(0) == 0) & (d.pn4.fillna(0) == 0) & (d.rating.abs() < 2.5) & (d.senate == 0)
coef = float(sys.argv[1]) if len(sys.argv) > 1 else 4.5
d["shift_pts"] = np.where(unp, coef * d.rchg_res, 0.0)
print("2026: races rated on both revisions", len(d), "changed", int((d.rchg != 0).sum()), "mean change %.3f" % d.rchg.mean(),
      "| unpolled rated House races", int(unp.sum()), "with a change", int((unp & (d.rchg != 0)).sum()))
k = pd.read_csv(WORK / "pr_nl_2026.csv")[["race_id", "p_nl", "k_pD"]]
o = d[unp & (d.rchg != 0)][["race_id", "rating", "rchg", "rchg_res", "shift_pts"]].merge(k, on="race_id", how="left")
print(o.sort_values("shift_pts").round(3).to_string(index=False))
d[["race_id", "rchg", "rchg_res", "dpoll", "shift_pts"]].to_csv(WORK / "pe_rchg_2026.csv", index=False)
