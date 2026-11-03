"""pau-x item 3: House primaries as within-cycle indicators (2010-2022).

Source: FEC "Federal Elections 20XX" (fec.gov, public, no account), the copies in the shared raw/fec folder
(federalelections2010-2022). Each House candidate's primary votes per district; "Unopposed" = no vote count.
There is no FEC primary file for 2024 in the folder (2024congressgecands.xlsx lists general-ballot candidates only),
so 2024 gets no feature. All primaries precede the general election; the vote counts are certified results of a
pre-election event (as-of safe). Louisiana (November jungle) is excluded.

District features (race_id of that cycle's map):
  prim_dev   = 100 (D - R) / (D + R) primary votes - 2 * lean, clipped +-40, where both parties' primaries were
               counted (D, R > 0); centred within year.  CA and WA top-two primaries are included (they are the
               v4 `prim_dev` feature that v5.1 does not use)
  prim_dev_s = the same, centred within state-year (removes state turnout rules / statewide primaries)
  prim_dev_x = the same, without CA and WA
  outp_cont  = +1 if the D primary had 2+ candidates with votes and the seat is R-held, -1 for the mirror, centred
  inc_weak   = (1 - incumbent's primary share) signed against the incumbent's party (D incumbent: negative),
               0 if unopposed; centred
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work" / "pylib"))
from common import SHARED

FEC = SHARED / "raw" / "fec"
DP = {"D", "DEM", "DFL", "D/WF", "D/IP", "DEM/WF"}
RP = {"R", "REP", "R/CRV", "R/C"}


def _sheet(y):
    f = FEC / (f"federalelections{y}.xls" if y <= 2014 else f"federalelections{y}.xlsx")
    xs = pd.ExcelFile(f)
    nm = [s for s in xs.sheet_names if "House" in s and ("Result" in s or "Resut" in s) and "Party" not in s][0]
    return pd.read_excel(f, sheet_name=nm, dtype=str)


def parse(y):
    x = _sheet(y)
    x.columns = [str(c).strip().upper() for c in x.columns]
    x = x.rename(columns={"D": "DISTRICT"})
    pc = [c for c in x.columns if c in ("PRIMARY VOTES", "PRIMARY")][0]
    st = "STATE ABBREVIATION"
    idc = [c for c in x.columns if c.startswith("FEC ID")][0]
    inc = [c for c in x.columns if c.startswith("(I)") or c.startswith("INCUMBENT")][0]
    x[inc] = x[inc].fillna("").str.replace("INCUMBENT INDICATOR", "")
    x = x[x[idc].notna() & x["DISTRICT"].notna()].copy()
    x["DISTRICT"] = x["DISTRICT"].str.strip()
    x = x[x["DISTRICT"].str.match(r"^\d+$")]
    x["party"] = x["PARTY"].fillna("").str.strip().str.upper()
    x["unopp"] = x[pc].fillna("").str.contains("Unopposed", case=False)
    x["pv"] = pd.to_numeric(x[pc], errors="coerce")
    x["ppct"] = pd.to_numeric(x["PRIMARY %"], errors="coerce")
    x["inc"] = x[inc].fillna("").str.contains("I")
    x["side"] = np.where(x.party.isin(DP), "D", np.where(x.party.isin(RP), "R", "O"))
    x["year"] = y
    x["state"] = x[st].str.strip()
    x["dist"] = x["DISTRICT"].astype(int)
    return x[["year", "state", "dist", "side", "pv", "ppct", "unopp", "inc", "party"]]


def districts():
    rows = []
    for y in range(2010, 2024, 2):
        x = parse(y)
        x = x[x.state != "LA"]
        for (s, d), g in x.groupby(["state", "dist"]):
            D = g[g.side == "D"]
            R = g[g.side == "R"]
            dv, rv = D.pv.sum(min_count=1), R.pv.sum(min_count=1)
            incs = g[g.inc & g.side.isin(["D", "R"])]
            isd = incs.side.iloc[0] if len(incs) else None
            ish = incs.ppct.iloc[0] if len(incs) else np.nan
            if len(incs) and incs.unopp.iloc[0]:
                ish = 1.0
            rows.append(dict(year=y, state=s, dist=d, dv=dv, rv=rv, nD=int(D.pv.notna().sum()), nR=int(R.pv.notna().sum()),
                             inc_side=isd, inc_share=ish))
    return pd.DataFrame(rows)


def race_ids(df, hist):
    ids = set(zip(hist.year, hist.race_id))
    def rid(y, s, d):
        if (y, f"{s}-AL") in ids:
            return f"{s}-AL"
        return f"{s}-{d:02d}"
    df["race_id"] = [rid(y, s, d) for y, s, d in zip(df.year, df.state, df.dist)]
    return df


def _centre(v, key):
    v = v.copy()
    has = v.notna()
    k = pd.Series(key, index=v.index)
    m = v[has].groupby(k[has]).transform("mean")
    v[has] = v[has] - m
    return v.fillna(0.0)


def build(hist):
    p = race_ids(districts(), hist)
    h = hist[hist.senate == 0][["year", "race_id", "lean", "incumbent_party"]]
    p = p.merge(h, on=["year", "race_id"], how="inner")
    both = (p.dv > 0) & (p.rv > 0)
    pm = 100 * (p.dv - p.rv) / (p.dv + p.rv)
    p["prim_margin"] = np.where(both, pm, np.nan)
    p["raw"] = np.clip(p.prim_margin - 2 * p.lean, -40, 40)
    idx = pd.MultiIndex.from_arrays([p.year.values, p.race_id.values])
    out = {}
    raw = pd.Series(p.raw.values, index=idx)
    out["prim_dev"] = _centre(raw, p.year.values)
    out["prim_dev_s"] = _centre(raw, (p.year.astype(str) + p.state).values)
    rx = raw.copy()
    rx[p.state.isin(["CA", "WA"]).values] = np.nan
    out["prim_dev_x"] = _centre(rx, p.year.values)
    oc = np.where((p.incumbent_party == "REP") & (p.nD >= 2), 1.0, np.where((p.incumbent_party == "DEM") & (p.nR >= 2), -1.0, 0.0))
    out["outp_cont"] = _centre(pd.Series(oc, index=idx), p.year.values)
    iw = np.where(p.inc_side == "D", -(1 - p.inc_share), np.where(p.inc_side == "R", (1 - p.inc_share), np.nan))
    out["inc_weak"] = _centre(pd.Series(np.nan_to_num(iw, nan=0.0) * 1.0, index=idx), p.year.values)
    return p, out


if __name__ == "__main__":
    import px_harness as H
    p, out = build(H.HIST)
    p.to_csv(H.WORK / "px" / "primaries.csv", index=False)
    print(p.groupby("year").agg(n=("race_id", "size"), both=("prim_margin", lambda s: s.notna().sum())))
    ok = p.prim_margin.notna()
    print("corr(prim_margin, 2*lean)", np.corrcoef(p.prim_margin[ok], p.lean[ok])[0, 1])
