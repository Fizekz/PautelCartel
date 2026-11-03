"""pau-f: v6+nl for 2026 with the Nov 1 poll file vs the refreshed (Nov 2) one; per-race P(D), Kalshi and EV picks.

v6+nl = PE.all_(PE.V["v51"], pr_pat_lam=1.0, pr_pat_set="nl", pr_pat_unp=True) with the nl product columns of
pr_theta_nl2.py lines 8-11. The 2026 feature table is rebuilt by features.build_2026() unchanged, with two inputs
redirected on read (nothing shared is written):
  features/polls_2026.csv  -> the poll file given (old = the shared Nov 1 file, new = the refreshed one)
  work/national.csv        -> 2026 generic ballot from the national file given (old: unchanged national.csv)
and its side-effect write (work/polls_v4_rows_2026.csv) redirected into the output folder.

Usage: refresh_v6nl_2026.py OUT_DIR NEW_POLLS [NEW_NATIONAL_2026]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import pr_pattern as PP
Z = PP.Z
for a, b in (("college", "college"), ("hispanic", "hispanic"), ("black", "black"), ("college", "hispanic")):
    v = Z[a] * Z[b]; Z[f"{a}_x_{b}"] = (v - v.mean()) / v.std()
PP.SETS["nl"] = ["college", "hispanic", "black", "college_x_college", "hispanic_x_hispanic", "black_x_black", "college_x_hispanic"]
import pr_eval as PE
from model import WinModel
from common import WORK, FEAT
import features
import topttwo
from bs4 import BeautifulSoup as _BS
# lxml is not installed here: same parse with html.parser (the old-rebuild check below verifies the result)
topttwo.BeautifulSoup = lambda html, *_a, **_k: _BS(html, "html.parser")

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
NEW_POLLS = Path(sys.argv[2])
NEW_NAT = Path(sys.argv[3]) if len(sys.argv) > 3 else None
KDIR = Path(__file__).resolve().parents[4] / "data" / "kalshi"  # repository copy of the Kalshi snapshots
K_NOV1 = KDIR / "races_2026-11-01T074812Z.csv"     # earliest Nov 1 snapshot (= the benchmark rebuild)
K_LAST = sorted(KDIR.glob("races_2026-11-0[12]T*.csv"))[-1]
assert K_LAST.name < "races_2026-11-03T230000Z.csv"

_read, _write = pd.read_csv, pd.DataFrame.to_csv


def gb_from(nat_file):
    n = _read(nat_file)
    agg = n[(n.metric == "generic_ballot") & (n.pollster == "Average")].sort_values("as_of").iloc[-1]
    return float(agg.dem_pct - agg.rep_pct), str(agg.as_of)


def build(polls_file, nat_file, tag):
    natw = _read(WORK / "national.csv")
    if nat_file is not None:
        gb, asof = gb_from(nat_file)
        print(f"[{tag}] 2026 gb {natw.loc[natw.year == 2026, 'gb'].iloc[0]:.2f} -> {gb:.2f} (as of {asof})")
        natw.loc[natw.year == 2026, "gb"] = gb
    natf = OUT / f"national_{tag}.csv"; _write(natw, natf, index=False)

    def rd(path, *a, **k):
        p = str(path)
        if p == str(FEAT / "polls_2026.csv"):
            path = polls_file
        elif p == str(WORK / "national.csv"):
            path = natf
        return _read(path, *a, **k)

    def wr(self, path=None, *a, **k):
        if path is not None and str(path).startswith(str(WORK)):
            path = OUT / f"{tag}_{Path(path).name}"
        return _write(self, path, *a, **k)

    pd.read_csv, pd.DataFrame.to_csv = rd, wr
    try:
        f = features.build_2026()
    finally:
        pd.read_csv, pd.DataFrame.to_csv = _read, _write
    _write(f, OUT / f"features_2026_{tag}.csv", index=False)
    return f


def kalshi(path, sides):
    """Normalised Kalshi price of the D-side and of the R-side candidate (matched by surname, else by party)."""
    from common import surname
    k = _read(path)
    out = []
    for r in k.itertuples():
        c = []
        for part in str(r.candidates).split("|"):
            b = part.strip().split(":")
            if len(b) >= 3:
                try:
                    c.append((b[0], ":".join(b[1:-1]).strip(), float(b[-1])))
                except ValueError:
                    pass
        tot = sum(x[2] for x in c) or np.nan
        dn, rn, ind = sides.get(r.race_id, (None, None, False))

        def price(name, party):
            for pa, nm, pr in c:
                if name and not nm.endswith(" party") and surname(nm) == surname(name):
                    return pr / tot
            return sum(pr for pa, nm, pr in c if pa == party) / tot
        out.append(dict(race_id=r.race_id, pD=price(dn, "IND" if ind else "DEM"), pR=price(rn, "REP"),
                        fav=r.fav_candidate, fav_party=r.fav_party))
    return pd.DataFrame(out)


hist = _read(WORK / "features_hist.csv", low_memory=False)
cfg = PE.all_(PE.V["v51"], pr_pat_lam=1.0, pr_pat_set="nl", pr_pat_unp=True)
Y = list(range(2010, 2026, 2))
res = {}
for tag, pf, nf in (("old", FEAT / "polls_2026.csv", None), ("new", NEW_POLLS, NEW_NAT), ("newpolls_oldgb", NEW_POLLS, None)):
    if tag == "newpolls_oldgb" and NEW_NAT is None:
        continue
    f26 = build(pf, nf, tag)
    if tag == "old":   # the rebuild must reproduce the feature table the 2026 runs used
        cur = _read(WORK / "features_2026.csv", low_memory=False)
        def _eq(a, b):
            if a.dtype.kind in "fi" and b.dtype.kind in "fi":
                return bool(np.isclose(a, b, equal_nan=True, rtol=0, atol=1e-9).all())
            return a.astype(str).equals(b.astype(str))
        back = _read(OUT / "features_2026_old.csv", low_memory=False)   # compare as written (dtypes)
        same = list(cur.columns) == list(back.columns) and all(_eq(cur[c], back[c]) for c in cur.columns)
        print("old rebuild == work/features_2026.csv:", same, flush=True)
    std = f26[f26.kind.isin(["D_vs_R", "R_vs_IND", "jungle_party"])].copy(); std["year"] = 2026
    p = WinModel(Y, cfg).fit(hist).predict(std)
    res[tag] = pd.DataFrame({"race_id": std.race_id.values, "pD": p.p_dside.values, "mu": p.mu.values,
                             "poll_margin": std.poll_margin.values, "poll_n": std.poll_pollsters.values,
                             "pm4": std.pm4.values, "pn4": std.pn4.values, "gb": std.gb.values,
                             "dside": std.dside_name.values, "rside": std.rside_name.values,
                             "chamber": std.chamber.values})
    print(tag, "done", flush=True)

d = res["old"][["race_id", "chamber", "dside", "rside"]].copy()
for tag, r in res.items():
    for c in ("pD", "mu", "poll_margin", "poll_n", "pm4", "pn4"):
        d[f"{c}_{tag}"] = r[c].values
sides = {r: (dn, rn, r in features.DSIDE_OVERRIDE_2026) for r, dn, rn in zip(d.race_id, d.dside, d.rside)}
k1, kl = kalshi(K_NOV1, sides), kalshi(K_LAST, sides)
d = d.merge(k1.rename(columns={"pD": "kD_nov1", "pR": "pR_nov1", "fav": "fav_nov1", "fav_party": "favp_nov1"}),
            on="race_id", how="left")
d = d.merge(kl[["race_id", "pD", "pR"]].rename(columns={"pD": "kD_last", "pR": "pR_last"}), on="race_id", how="left")
pts = lambda p: -np.log2(np.maximum(p, 0.01))
for tag in res:
    q = d[f"pD_{tag}"]
    evD, evR = q * pts(d.kD_nov1), (1 - q) * pts(d.pR_nov1)
    d[f"pick_{tag}"] = np.where(evD >= evR, "D", "R")
    d[f"ev_{tag}"] = np.maximum(evD, evR)
d["dpD"] = d.pD_new - d.pD_old
d.to_csv(OUT / "v6nl_2026_old_vs_new.csv", index=False)
print("Kalshi Nov 1:", K_NOV1.name, "| latest:", K_LAST.name)
show = ["race_id", "pD_old", "pD_new", "dpD", "kD_nov1", "kD_last", "pick_old", "pick_new", "poll_margin_old", "poll_margin_new",
        "poll_n_old", "poll_n_new"]
print(d.reindex(d.dpD.abs().sort_values(ascending=False).index).head(30)[show].round(3).to_string(index=False))
ch = d[d.pick_old != d.pick_new]
print("\nEV pick changes vs the Nov 1 price:", len(ch))
print(ch[show].round(3).to_string(index=False))
print("\nexpected D-side House wins old/new:", round(d[d.chamber == "House"].pD_old.sum(), 2), round(d[d.chamber == "House"].pD_new.sum(), 2))
