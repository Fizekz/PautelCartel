"""pau-r race-level features: district demographics (ACS 1-year table-based summary files, www2.census.gov bulk
download, no key: 2021 = 116th Congress lines, 2022 = 118th, 2024 = 119th) joined to each House race-year.
Mapping (lines in force; demographics are as of the ACS year, not the election year, noted as a caveat):
  2012, 2014: 116th-line ACS 2021 for states whose lines did not change 2012 -> 2018 (PA, NC, FL, VA missing)
  2016: ACS 2021 except PA (2016 still on the 2012 PA map) ; 2018: ACS 2021 ; 2020: ACS 2021 except NC (redrawn 2020)
  2022: ACS 2022 ; 2024: ACS 2024 ; 2026: ACS 2024 except the states redrawn for 2026 ; 2010: none (2002 lines)
Features: college (BA+ share of 25+, B15003), hispanic, black NH, white NH (B03002), median age (B01002),
log median household income (B19013), owner-occupied share (B25003). Also from pau-b's features: lean (-PVI),
ltrend, incumbency, region. Output race/features.csv."""
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).parent
FIPS = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
        "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
        "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
        "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
        "56": "WY", "72": "PR"}
REGION = {**{s: "NE" for s in "CT ME MA NH RI VT NJ NY PA".split()}, **{s: "MW" for s in "IL IN MI OH WI IA KS MN MO NE ND SD".split()},
          **{s: "S" for s in "DE FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX DC".split()},
          **{s: "W" for s in "AZ CO ID MT NV NM UT WY AK CA HI OR WA".split()}}
AL = {"AK", "DE", "MT", "ND", "SD", "VT", "WY"}


def acs(year):
    def rd(t):
        d = pd.read_csv(R / "raw" / f"acsdt1y{year}-{t}.dat", sep="|", dtype={"GEO_ID": str})
        return d[d.GEO_ID.str.match(r"^500\d\d00US")].set_index("GEO_ID")
    e, r, a, i, o = rd("b15003"), rd("b03002"), rd("b01002"), rd("b19013"), rd("b25003")
    out = pd.DataFrame(index=e.index)
    out["college"] = e[[f"B15003_E0{k}" for k in (22, 23, 24, 25)]].sum(1) / e.B15003_E001
    out["hispanic"] = r.B03002_E012 / r.B03002_E001
    out["black"] = r.B03002_E004 / r.B03002_E001
    out["white"] = r.B03002_E003 / r.B03002_E001
    out["age"] = a.B01002_E001
    out["loginc"] = np.log(pd.to_numeric(i.B19013_E001, errors="coerce"))
    out["owner"] = o.B25003_E002 / o.B25003_E001
    st = out.index.str[9:11].map(FIPS)
    dd = out.index.str[11:13]
    out["race_id"] = [f"{s}-AL" if (d in ("00", "98") or s in AL) else f"{s}-{d}" for s, d in zip(st, dd)]
    return out.reset_index(drop=True)


A = {y: acs(y) for y in (2021, 2022, 2024)}
h = pd.read_csv(R.parent / "pbase" / "work" / "features_hist.csv", low_memory=False)
f26 = pd.read_csv(R.parent / "pbase" / "work" / "features_2026.csv", low_memory=False).assign(year=2026)
base = pd.concat([h[["year", "race_id", "senate", "lean", "ltrend", "inc", "state"]],
                  f26[["year", "race_id", "senate", "lean", "ltrend", "inc"]].assign(state=f26.race_id.str[:2])])
base = base[base.senate == 0].copy()
REDRAWN26 = set("AL CA FL LA NC OH TN TX UT ".split())
try:
    rd26 = pd.read_csv(R.parent / "pbase" / "work" / "features_2026.csv", low_memory=False)
    if "redrawn" in rd26:
        REDRAWN26 = set(rd26.loc[rd26.redrawn == 1, "race_id"].str[:2])
except Exception:  # noqa: BLE001
    pass
src = {2012: 2021, 2014: 2021, 2016: 2021, 2018: 2021, 2020: 2021, 2022: 2022, 2024: 2024, 2026: 2024}
miss = {2012: {"PA", "NC", "FL", "VA"}, 2014: {"PA", "NC", "FL", "VA"}, 2016: {"PA"}, 2020: {"NC"}, 2026: REDRAWN26}
rows = []
for y, g in base.groupby("year"):
    if y in src:
        g = g.merge(A[src[y]], on="race_id", how="left")
        bad = g.state.isin(miss.get(y, set()))
        for c in ("college", "hispanic", "black", "white", "age", "loginc", "owner"):
            g.loc[bad, c] = np.nan
    rows.append(g)
F = pd.concat(rows, ignore_index=True)
F["region"] = F.state.map(REGION)
F["has_acs"] = F.college.notna().astype(int)
F.to_csv(R / "features.csv", index=False)
print(F.groupby("year").agg(n=("race_id", "size"), acs=("has_acs", "sum")).to_string())
print("2026 redrawn states without ACS:", sorted(REDRAWN26))
