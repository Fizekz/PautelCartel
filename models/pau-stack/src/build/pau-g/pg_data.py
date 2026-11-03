"""pau-g: county election table and county-to-district weights for the swing-geometry model.

Outputs (pbase/work/):
  pg_county_votes.csv  one row per (source, county): source id, year, office (pres / sen / gov), state, fips, dem, rep
  pg_cd_county.csv     population-weighted county parts of each district: lines (116 / 118 / 119), race_id, fips, pop
County units: 2010-vintage county FIPS, with Bedford city VA folded into Bedford County, Shannon SD -> Oglala Lakota,
and Alaska as one unit (02000; its House and Senate races are statewide). DC is dropped.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pg"
WORK = ROOT / "work"
FIPS2ST = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
           "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
           "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
           "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
           "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
           "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}
ST2FIPS = {v: k for k, v in FIPS2ST.items()}
NAME2ST = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
           "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
           "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
           "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
           "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
           "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
           "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
           "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
           "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
           "Wisconsin": "WI", "Wyoming": "WY"}
AL_STATES = {"AK", "DE", "MT", "ND", "SD", "VT", "WY"}


def canon_fips(f):
    f = int(f)
    if f // 1000 == 2:
        return 2000
    return {51515: 51019, 46113: 46102, 51560: 51005, 15005: 15009}.get(f, f)


def norm(s):
    import unicodedata
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.lower().replace("&", "and")
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"\(.*?city.*?\)", " city", s)
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("saint ", "st ").replace("st. ", "st ").replace("ste. ", "ste ")
    s = re.sub(r"[^a-z ]", "", s)
    s = re.sub(r"\b(county|parish|borough|census area|municipality|city and borough)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------- MEDSL county presidential 2000-2016 ----------
def medsl_pres():
    m = pd.read_csv(RAW / "countypres_2000-2016.csv")
    m = m[m.FIPS.notna() & m.party.isin(["democrat", "republican"])].copy()
    m["fips"] = m.FIPS.astype(int).map(canon_fips)
    m = m[m.state_po != "DC"]
    p = m.pivot_table(index=["year", "state_po", "fips"], columns="party", values="candidatevotes", aggfunc="sum").reset_index()
    p = p.rename(columns={"state_po": "state", "democrat": "dem", "republican": "rep"})
    p["office"] = "pres"
    return p


def name_table():
    """(state, normalised county name) -> canonical fips, from MEDSL's names; VA cities get a ' city' variant."""
    m = pd.read_csv(RAW / "countypres_2000-2016.csv")
    m = m[m.FIPS.notna() & (m.year == 2016)]
    t = {}
    for st, c, f in zip(m.state_po, m.county, m.FIPS.astype(int)):
        n = norm(c)
        if st == "VA" and f >= 51500:
            t[(st, n + " city")] = canon_fips(f)
            t.setdefault((st, n), canon_fips(f))
        else:
            t[(st, n)] = canon_fips(f)
    extra = {("SD", "oglala lakota"): 46102, ("SD", "shannon"): 46102, ("VA", "bedford city"): 51019,
             ("NM", "dona ana"): 35013, ("LA", "la salle"): 22059, ("LA", "lasalle"): 22059,
             ("IL", "dekalb"): 17037, ("IL", "de kalb"): 17037, ("IN", "dekalb"): 18033, ("IN", "de kalb"): 18033,
             ("IL", "dupage"): 17043, ("IL", "du page"): 17043, ("IN", "laporte"): 18091, ("IN", "la porte"): 18091,
             ("IN", "lagrange"): 18087, ("IN", "la grange"): 18087, ("MO", "st louis city"): 29510,
             ("MO", "saint louis city"): 29510, ("MD", "baltimore city"): 24510, ("MO", "kansas city"): None,
             ("MD", "st marys"): 24037, ("MO", "ste genevieve"): 29186, ("NV", "carson city"): 32510,
             ("VA", "james city"): 51095, ("VA", "charles city"): 51036, ("TX", "dewitt"): 48123,
             ("TX", "de witt"): 48123, ("FL", "desoto"): 12027, ("FL", "de soto"): 12027, ("MS", "desoto"): 28033,
             ("MS", "de soto"): 28033, ("LA", "desoto"): 22031, ("LA", "de soto"): 22031, ("ND", "lamoure"): 38045,
             ("ND", "la moure"): 38045, ("IA", "obrien"): 19141, ("IL", "lasalle"): 17099, ("IL", "la salle"): 17099,
             ("AL", "dekalb"): 1049, ("GA", "dekalb"): 13089, ("TN", "dekalb"): 47041, ("MO", "dekalb"): 29063,
             ("WI", "fond du lac"): 55039, ("HI", "kalawao"): 15009, ("IL", "dewitt"): 17039,
             ("OK", "leflore"): 40079, ("AK", "alaska"): 2000}
    for k, v in extra.items():
        t[k] = v
    return t


# ---------- Wikipedia "By county" tables ----------
def _grid(table):
    """Expand a table's rows into a grid of cell texts, honouring colspan/rowspan; returns (header_rows, body_rows)."""
    rows = table.find_all("tr")
    grid, pending = [], {}
    is_head = []
    for r in rows:
        cells = r.find_all(["th", "td"], recursive=False)
        out, col = [], 0
        ci = 0
        while ci < len(cells) or col in pending:
            if col in pending:
                txt, left = pending[col]
                out.append(txt)
                if left > 1:
                    pending[col] = (txt, left - 1)
                else:
                    del pending[col]
                col += 1
                continue
            c = cells[ci]
            ci += 1
            txt = c.get_text(" ", strip=True)
            cs = int(re.sub(r"\D", "", c.get("colspan", "1")) or 1)
            rs = int(re.sub(r"\D", "", c.get("rowspan", "1")) or 1)
            for k in range(cs):
                out.append(txt)
                if rs > 1:
                    pending[col] = (txt, rs - 1)
                col += 1
        grid.append(out)
        is_head.append(all(c.name == "th" for c in cells) and len(cells) > 0)
    nh = 0
    while nh < len(grid) and is_head[nh]:
        nh += 1
    return grid[:nh], grid[nh:]


def _num(s):
    s = re.sub(r"\[.*?\]", "", s).replace(",", "").replace("−", "-").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_county_table(html):
    """Return a DataFrame (name, dem, rep) from the largest county table in the section HTML."""
    soup = BeautifulSoup(html, "html.parser")
    best = None
    for tb in soup.find_all("table"):
        head, body = _grid(tb)
        if len(head) == 0 or len(body) < 3:
            continue
        ncol = max(len(r) for r in head)
        labels = []
        for j in range(ncol):
            parts = [h[j] for h in head if j < len(h)]
            labels.append(" | ".join(dict.fromkeys(parts)))
        dcols = [j for j, l in enumerate(labels) if re.search(r"Democrat|DFL|Democratic", l) and re.search(r"\| (#|Votes|votes)$", l)]
        rcols = [j for j, l in enumerate(labels) if re.search(r"Republican", l) and re.search(r"\| (#|Votes|votes)$", l)]
        if not dcols or not rcols:
            continue
        rows = []
        for r in body:
            if len(r) < ncol - 1:
                continue
            nm = r[0]
            if re.match(r"(?i)totals?\b|total$", nm.strip()):
                continue
            d = sum(_num(r[j]) for j in dcols if j < len(r))
            rp = sum(_num(r[j]) for j in rcols if j < len(r))
            rows.append((nm, d, rp))
        df = pd.DataFrame(rows, columns=["name", "dem", "rep"]).dropna()
        if best is None or len(df) > len(best):
            best = df
    return best


def wiki_votes(folder, pattern):
    """Parse every saved section file; returns rows (source, year, office, state, fips, dem, rep) and a log."""
    names = name_table()
    out, log = [], []
    for f in sorted((RAW / "wiki" / folder).glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("n"):
            continue
        m = re.match(pattern, f.stem)
        year, office, stname = int(m.group(1)), m.group(2), m.group(3)
        st = NAME2ST[stname]
        best = None
        for h in d["html"]:
            t = parse_county_table(h)
            if t is not None and (best is None or len(t) > len(best)):
                best = t
        if best is None:
            log.append((f.stem, "no table"))
            continue
        if st == "AK":
            best = pd.DataFrame([("alaska", best.dem.sum(), best.rep.sum())], columns=["name", "dem", "rep"])
        best["fips"] = [names.get((st, norm(n)), names.get((st, norm(n).replace(" city", "")))) for n in best.name]
        miss = best[best.fips.isna()]
        if len(miss):
            log.append((f.stem, "unmatched: " + "; ".join(miss.name.head(8))))
        best = best[best.fips.notna()]
        best = best.groupby("fips", as_index=False)[["dem", "rep"]].sum()
        out.append(best.assign(year=year, office=office, state=st))
    return pd.concat(out, ignore_index=True), log


def county_votes():
    p = medsl_pres()
    w, log = wiki_votes("pres", r"(\d{4})_()(.*)")
    w["office"] = "pres"
    allp = pd.concat([p[["year", "office", "state", "fips", "dem", "rep"]], w[["year", "office", "state", "fips", "dem", "rep"]]])
    allp["fips"] = allp.fips.astype(int)
    allp = allp.groupby(["year", "office", "state", "fips"], as_index=False)[["dem", "rep"]].sum()
    allp["source"] = allp.office + "_" + allp.year.astype(str)
    parts = [allp]
    if (RAW / "wiki" / "sw").exists():
        s, log2 = wiki_votes("sw", r"(\d{4})_(sen|sens|gov)_(.*)")
        log += log2
        s["source"] = s.office + "_" + s.year.astype(str) + "_" + s.state
        parts.append(s)
    if (RAW / "2024-senate-county.csv").exists():
        m = pd.read_csv(RAW / "2024-senate-county.csv", dtype={"county_fips": str})
        m = m[m.party_simplified.isin(["DEMOCRAT", "REPUBLICAN"]) & m.county_fips.notna() & (m.stage == "GEN")]
        m["fips"] = m.county_fips.astype(float).astype(int).map(canon_fips)
        m["special"] = m.special.astype(str).str.upper() == "TRUE"
        q = m.pivot_table(index=["state_po", "fips", "special"], columns="party_simplified", values="votes", aggfunc="sum").reset_index()
        q = q.rename(columns={"state_po": "state", "DEMOCRAT": "dem", "REPUBLICAN": "rep"}).dropna(subset=["dem", "rep"])
        q["office"] = np.where(q.special, "sens", "sen")
        q["year"] = 2024
        q["source"] = q.office + "_2024_" + q.state
        q = q[~q.source.isin(set(parts[-1].source) if len(parts) > 1 else set())]
        parts.append(q[["year", "office", "state", "fips", "dem", "rep", "source"]])
    v = pd.concat(parts, ignore_index=True)
    v["fips"] = v.fips.astype(int)
    v = v[(v.dem > 0) | (v.rep > 0)]
    v.to_csv(WORK / "pg_county_votes.csv", index=False)
    return v, log


# ---------- county-to-district weights (ACS 5-year summary level 510) ----------
def cd_county():
    rows = []
    for acs_y, lines, code in ((2021, 116, "5101600US"), (2022, 118, "5101800US"), (2024, 119, "5101900US")):
        d = pd.read_csv(RAW / "acs5" / f"acsdt5y{acs_y}-b01003.dat", sep="|", dtype={"GEO_ID": str})
        d = d[d.GEO_ID.str.startswith(code)]
        g = d.GEO_ID.str[9:]
        st = g.str[:2].map(FIPS2ST)
        cd = g.str[2:4]
        fips = (g.str[:2] + g.str[4:7]).astype(int)
        al = AL_STATES - ({"MT"} if lines >= 118 else set())
        rid = [f"{s}-AL" if (c in ("00", "98") or s in al) else f"{s}-{c}" for s, c in zip(st, cd)]
        rows.append(pd.DataFrame(dict(lines=lines, race_id=rid, state=st, fips=fips, pop=d.B01003_E001.astype(float))))
    w = pd.concat(rows)
    w = w[w.state.notna() & (w.state != "DC") & ~w.race_id.str.endswith("-ZZ")]
    w["fips"] = w.fips.map(canon_fips)
    # Connecticut: ACS 2022+ uses planning regions, not the counties results are reported by. Use the 116th-line
    # (2021, county-based) weights for CT on every map (CT's lines barely changed in 2022).
    ct = w[(w.lines == 116) & (w.state == "CT")]
    w = pd.concat([w[w.state != "CT"]] + [ct.assign(lines=l) for l in (116, 118, 119)])
    w = w.groupby(["lines", "race_id", "state", "fips"], as_index=False)["pop"].sum()
    w.to_csv(WORK / "pg_cd_county.csv", index=False)
    return w


if __name__ == "__main__":
    v, log = county_votes()
    for l in log:
        print(l)
    print(v.groupby(["office", "year"]).agg(n=("fips", "size"), src=("source", "nunique")).to_string())
    w = cd_county()
    print(w.groupby("lines").agg(n=("race_id", "nunique"), parts=("fips", "size")))


def county_acs():
    """County demographics, ACS 5-year 2022 (2018-2022): college (BA+ of 25+), Hispanic, Black NH, white NH, Asian NH."""
    raw = RAW / "acs5"
    e = pd.read_csv(raw / "acsdt5y2022-b15003.dat", sep="|", dtype={"GEO_ID": str}, usecols=lambda c: c == "GEO_ID" or c.endswith(("E001", "E022", "E023", "E024", "E025")))
    r = pd.read_csv(raw / "acsdt5y2022-b03002.dat", sep="|", dtype={"GEO_ID": str}, usecols=["GEO_ID", "B03002_E001", "B03002_E003", "B03002_E004", "B03002_E006", "B03002_E012"])
    e = e[e.GEO_ID.str.startswith("0500000US")].set_index("GEO_ID")
    r = r[r.GEO_ID.str.startswith("0500000US")].set_index("GEO_ID")
    out = pd.DataFrame(index=r.index)
    out["college"] = e[[f"B15003_E0{k}" for k in (22, 23, 24, 25)]].sum(1) / e.B15003_E001
    out["hispanic"] = r.B03002_E012 / r.B03002_E001
    out["black"] = r.B03002_E004 / r.B03002_E001
    out["white"] = r.B03002_E003 / r.B03002_E001
    out["asian"] = r.B03002_E006 / r.B03002_E001
    out["pop"] = r.B03002_E001.astype(float)
    out["fips"] = out.index.str[9:].astype(int)
    out = out[out.fips // 1000 != 72].reset_index(drop=True)
    out["state"] = (out.fips // 1000).map(lambda s: FIPS2ST.get(f"{s:02d}"))
    out["fips"] = out.fips.map(canon_fips)
    # Connecticut planning regions (09110-09190) do not match the election counties: CT gets NaN (filled later)
    out = out[out.state.notna() & (out.state != "DC")]
    agg = out.groupby("fips").apply(lambda g: pd.Series({c: np.average(g[c], weights=g["pop"]) for c in ("college", "hispanic", "black", "white", "asian")} | {"pop": g["pop"].sum()}))
    agg.reset_index().to_csv(WORK / "pg_county_acs.csv", index=False)
    return agg
