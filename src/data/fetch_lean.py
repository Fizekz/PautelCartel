"""Build district partisan-lean tables (presidential results by congressional district, on each cycle's lines).

Outputs
  data/clean/lean_2026.csv  one row per 2026 race_id (435 House + 35 Senate): 2024/2020 presidential two-party
                            shares on the lines used in Nov 2026, Cook PVI (Wikipedia transcription), own PVI calc.
  data/clean/lean_hist.csv  one row per (year, race_id) for House and Senate general elections 2010-2024: the most
                            recent presidential result on THAT year's lines (2010->2008, 2012->2008, 2014->2012,
                            2016->2012, 2018->2016, 2020->2016, 2022->2020, 2024->2020) and the one before it.

Sources (raw copies kept under data/raw/lean/; rebuild offline unless --refresh)
  The Downballot / Daily Kos Elections presidential-results-by-CD Google Sheets (CSV export of the public sheet)
  MIT Election Data + Science Lab, U.S. President 1976-2024 (Harvard Dataverse doi:10.7910/DVN/42MVDX)
  Wikipedia "Cook Partisan Voting Index" (MediaWiki parse API; revision id recorded in the cached JSON)

Usage: .venv/bin/python src/data/fetch_lean.py [--refresh]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "lean"
CLEAN = ROOT / "data" / "clean"
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"
USER_AGENT = "DemocracyDayForecastResearch/0.1 (Stanford Model Citizens student contest; python-requests)"

GSHEET = "https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"
# file name -> (sheet id, gid, what it is)
SHEETS = {
    "db_pres24_2026lines.csv": ("1eZfaFI-c-PFOoKx1-zZA2MP0_dxRq_LVK0re3BOQqy0", "1491069057",
                                "2024 (and 2020 where unchanged) pres results on the lines used in 2026"),
    "db_pres24_2024lines.csv": ("1ng1i_Dm_RMDnEvauH44pgE6JCUsapcuu8F2pCfeLWFo", "1491069057",
                                "2024 and 2020 pres results on the lines used in 2024"),
    "db_pres20_2022lines.csv": ("1CKngqOp8fzU22JOlypoxNsxL6KSAH920Whc-rd7ebuM", "1871835782",
                                "2020 pres results on the lines used in 2022"),
    "db_pres08-20_2020lines.csv": ("1XbUXnI9OyfAuhP5P3vWtMuGc5UJlrhXbzZo3AwMuHtk", "0",
                                   "2008/2012/2016/2020 pres results on the lines used in 2020"),
    "db_pres08-16_2018lines.csv": ("1zLNAuRqPauss00HDz4XbTH2HqsCzMe0pR8QmD1K8jk8", "0",
                                   "2008/2012/2016 pres results on the lines used in 2018"),
    "db_pres08-16_2016lines.csv": ("1VfkHtzBTP5gf4jAu8tcVQgsBJ1IDvXEHjuMqYlOgYbA", "0",
                                   "2008/2012/2016 pres results on the lines used in 2016"),
    "db_pres08-12_2012-2014lines.csv": ("1xn6nCNM97oFDZ4M-HQgoUT3X4paOiSDsRMSuxbaOBdg", "0",
                                        "2008/2012 pres results on the lines used in 2012-2014"),
    "db_pres08_2006-2010lines.csv": ("1l7W130tPRF6dQ4JoqlQJSJexnEg4rZct-7kwDXqgoLE", "429358610",
                                     "2000/2004/2008 pres results (integer %) on the lines used in 2006-2010"),
}
MIT_PRES_URL = "https://dataverse.harvard.edu/api/access/datafile/13887042"
MIT_PRES_FILE = "mit_1976-2024-president.csv"
MIT_SENATE_FILE = ROOT / "data" / "raw" / "mit" / "1976-2024-senate-state.csv"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PVI_TITLE = "Cook_Partisan_Voting_Index"
WIKI_PVI_FILE = "wiki_Cook_Partisan_Voting_Index.json"

# States whose Nov-2026 House map differs from the 2024 map (The Downballot, 2026-07-09 post + its 2026 sheet).
# Missouri passed a 2025 map, but the U.S. Supreme Court ordered the 2022 map used for Nov 2026 (KCUR, 2026-09-29);
# the Downballot 2026 sheet shows MO on its 2022/2024 lines, so MO is NOT in this set.
REDRAWN_2026 = {"AL", "CA", "FL", "LA", "NC", "OH", "TN", "TX", "UT"}

# Cook PVI (2022 formula onward): 75% most recent presidential, 25% the one before. With 0.75 our pvi_calc,
# rounded, reproduces the Wikipedia-transcribed Cook values for 288 of 290 districts with both years available
# (0.5 matches only 180), so the transcription and our two-party shares agree.
PVI_W_RECENT = 0.75

DISTRICT_RE = r"^[A-Z]{2}-(\d{2}|AL)$"

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}


# ----------------------------------------------------------------------------------------------- downloads
def download(url: str, dest: Path, refresh: bool) -> Path:
    if dest.exists() and not refresh:
        return dest
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    resp.raise_for_status()
    if dest.suffix == ".csv" and resp.text.lstrip().startswith("<"):
        raise RuntimeError(f"{url} returned HTML, not CSV (sheet private or moved?)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


def fetch_all(refresh: bool) -> None:
    for fname, (sid, gid, _) in SHEETS.items():
        download(GSHEET.format(sid=sid, gid=gid), RAW / fname, refresh)
    download(MIT_PRES_URL, RAW / MIT_PRES_FILE, refresh)
    wiki = RAW / WIKI_PVI_FILE
    if refresh or not wiki.exists():
        params = {"action": "parse", "page": WIKI_PVI_TITLE, "prop": "text|revid", "format": "json",
                  "formatversion": "2", "redirects": "1"}
        resp = requests.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        data = resp.json()["parse"]
        page = {"title": data["title"], "revid": data["revid"], "html": data["text"],
                "fetched_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}
        wiki.write_text(json.dumps(page))


# ----------------------------------------------------------------------------------------------- parsing
def to_num(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.replace(r"[,%\s]", "", regex=True)
    return pd.to_numeric(cleaned.replace("", None), errors="coerce")


def read_districts(fname: str) -> pd.DataFrame:
    """Raw sheet -> rows whose first cell is a district id, indexed by that id; columns are positional ints."""
    raw = pd.read_csv(RAW / fname, header=None, dtype=str, keep_default_na=False)
    rows = raw[raw[0].str.strip().str.match(DISTRICT_RE)].copy()
    rows[0] = rows[0].str.strip()
    if rows[0].duplicated().any():
        raise ValueError(f"{fname}: duplicate district ids {rows[0][rows[0].duplicated()].tolist()}")
    return rows.set_index(0)


def two_party(dem: pd.Series, rep: pd.Series) -> pd.Series:
    return (dem / (dem + rep)).round(5)


# column positions (0 = district id, consumed as index) of (dem, rep) per presidential year in each sheet
SHEET_COLS = {
    "db_pres24_2026lines.csv": {2024: (4, 5), 2020: (11, 12)},           # vote counts
    "db_pres24_2024lines.csv": {2024: (3, 4), 2020: (10, 11)},           # vote counts
    "db_pres20_2022lines.csv": {2020: (3, 4)},                           # percentages
    "db_pres08-20_2020lines.csv": {2020: (3, 4), 2016: (5, 6), 2012: (7, 8), 2008: (9, 10)},
    "db_pres08-16_2018lines.csv": {2016: (3, 4), 2012: (5, 6), 2008: (7, 8)},
    "db_pres08-16_2016lines.csv": {2016: (3, 4), 2012: (5, 6), 2008: (7, 8)},
    "db_pres08-12_2012-2014lines.csv": {2012: (3, 4), 2008: (5, 6)},
    "db_pres08_2006-2010lines.csv": {2008: (3, 4), 2004: (5, 6), 2000: (7, 8)},  # integer percentages
}


def sheet_shares(fname: str) -> pd.DataFrame:
    """District x presidential-year Dem two-party share for one sheet."""
    rows = read_districts(fname)
    out = {yr: two_party(to_num(rows[d]), to_num(rows[r])) for yr, (d, r) in SHEET_COLS[fname].items()}
    return pd.DataFrame(out, index=rows.index)


def mit_pres() -> pd.DataFrame:
    """State x year Dem two-party share (MIT; fusion lines already folded into party_simplified)."""
    df = pd.read_csv(RAW / MIT_PRES_FILE)
    df = df[df["party_simplified"].isin(["DEMOCRAT", "REPUBLICAN"])]
    votes = df.pivot_table(index=["year", "state_po"], columns="party_simplified", values="candidatevotes",
                           aggfunc="sum")
    return votes


def national_shares(votes: pd.DataFrame) -> dict[int, float]:
    tot = votes.groupby(level="year").sum()
    return (tot["DEMOCRAT"] / (tot["DEMOCRAT"] + tot["REPUBLICAN"])).round(5).to_dict()


def state_shares(votes: pd.DataFrame) -> pd.Series:
    return (votes["DEMOCRAT"] / (votes["DEMOCRAT"] + votes["REPUBLICAN"])).round(5)


def wiki_pvi() -> tuple[pd.Series, pd.Series, int]:
    """(district PVI text by race_id, state PVI text by state, revid) from the cached Wikipedia page."""
    page = json.loads((RAW / WIKI_PVI_FILE).read_text())
    tables = re.findall(r"<table.*?</table>", page["html"], re.S)

    def cells(table: str) -> list[list[str]]:
        rows = re.findall(r"<tr.*?</tr>", table, re.S)
        clean = lambda c: re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c))).strip()
        return [[clean(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)] for r in rows]

    district, state = {}, {}
    for table in tables:
        rows = cells(table)
        if not rows or len(rows[0]) < 2 or rows[0][1] != "PVI":
            continue
        for row in rows[1:]:
            name, pvi = row[0], row[1].replace("−", "-")
            if rows[0][0] == "District":
                m = re.match(r"^(.*?) (\d+|at-large)$", name)
                if not m or m.group(1) not in STATE_ABBR:
                    continue
                num = "AL" if m.group(2) == "at-large" else f"{int(m.group(2)):02d}"
                district[f"{STATE_ABBR[m.group(1)]}-{num}"] = pvi
            elif rows[0][0] == "State" and name in STATE_ABBR:
                state[STATE_ABBR[name]] = pvi
    if len(district) != 435 or len(state) != 50:
        raise ValueError(f"Wikipedia PVI parse: {len(district)} districts, {len(state)} states")
    return pd.Series(district), pd.Series(state), page["revid"]


def pvi_to_num(text: str | float) -> float:
    if not isinstance(text, str) or not text:
        return float("nan")
    if text.upper() == "EVEN":
        return 0.0
    m = re.match(r"^([DR])\+(\d+(?:\.\d+)?)$", text.strip())
    if not m:
        raise ValueError(f"unparseable PVI {text!r}")
    return float(m.group(2)) * (1 if m.group(1) == "D" else -1)


def pvi_calc(d24: pd.Series, d20: pd.Series, nat: dict[int, float]) -> pd.Series:
    """Cook-style PVI (D positive, points) = weighted mean of two-party share minus national two-party share."""
    rel24, rel20 = (d24 - nat[2024]) * 100, (d20 - nat[2020]) * 100
    both = PVI_W_RECENT * rel24 + (1 - PVI_W_RECENT) * rel20
    return both.round(1)


# ----------------------------------------------------------------------------------------------- 2026 table
def build_2026(nat: dict[int, float], st: pd.Series) -> tuple[pd.DataFrame, dict]:
    template = pd.read_csv(TEMPLATE)[["race_id", "chamber"]]
    new = sheet_shares("db_pres24_2026lines.csv")
    old = sheet_shares("db_pres24_2024lines.csv")
    new_votes = read_districts("db_pres24_2026lines.csv")[[4, 5]]
    old_votes = read_districts("db_pres24_2024lines.csv")[[3, 4]].set_axis([4, 5], axis=1)
    unchanged = (new_votes == old_votes.reindex(new_votes.index)).all(axis=1)
    pvi_d, pvi_s, revid = wiki_pvi()

    rows = []
    for race_id, chamber in template.itertuples(index=False):
        state = race_id[:2]
        if chamber == "Senate":
            d24, d20 = st.get((2024, state)), st.get((2020, state))
            rows.append(dict(race_id=race_id, chamber=chamber, pres24_dem_2p=d24, pres20_dem_2p=d20,
                             cook_pvi=pvi_s.get(state), lines="statewide", district_changed_2026=0,
                             source=f"MIT president 1976-2024 (statewide); Cook PVI via Wikipedia rev {revid}"))
            continue
        redrawn = state in REDRAWN_2026
        d24, d20 = new.at[race_id, 2024], new.at[race_id, 2020]
        src = "Downballot 2026-lines sheet"
        if pd.isna(d20) and redrawn and unchanged.get(race_id, False):
            d20 = old.at[race_id, 2020]
            src += "; pres20 from 2024-lines sheet (district geography unchanged: identical 2024 vote totals)"
        rows.append(dict(race_id=race_id, chamber=chamber, pres24_dem_2p=d24, pres20_dem_2p=d20,
                         cook_pvi=pvi_d.get(race_id), lines="new-2026-lines" if redrawn else "2024-lines",
                         district_changed_2026=int(redrawn and not unchanged.get(race_id, False)),
                         source=src + f"; Cook PVI via Wikipedia rev {revid}"))
    out = pd.DataFrame(rows)
    out["pres24_rep_2p"] = (1 - out["pres24_dem_2p"]).round(5)
    out["pvi_num"] = out["cook_pvi"].map(pvi_to_num)
    out["pvi_calc"] = pvi_calc(out["pres24_dem_2p"], out["pres20_dem_2p"], nat)
    cols = ["race_id", "chamber", "pres24_dem_2p", "pres24_rep_2p", "pres20_dem_2p", "cook_pvi", "pvi_num",
            "lines", "source", "pvi_calc", "district_changed_2026"]
    return out[cols], {"wiki_revid": revid}


# ----------------------------------------------------------------------------------------------- history
# House election year -> (sheet with that year's lines, previous pres year, pres year before that)
HOUSE_HIST = {
    2010: ("db_pres08_2006-2010lines.csv", 2008, 2004),
    2012: ("db_pres08-12_2012-2014lines.csv", 2008, None),
    2014: ("db_pres08-12_2012-2014lines.csv", 2012, 2008),
    2016: ("db_pres08-16_2016lines.csv", 2012, 2008),
    2018: ("db_pres08-16_2018lines.csv", 2016, 2012),
    2020: ("db_pres08-20_2020lines.csv", 2016, 2012),
    2022: ("db_pres20_2022lines.csv", 2020, None),
    2024: ("db_pres24_2024lines.csv", 2020, None),
}
SENATE_YEARS = range(2010, 2025, 2)
# Senate general elections missing from the MIT senate-state file (v. in repo): CT 2022 (Blumenthal v. Levy, class 3).
# Verified against the class-3 seat list; the 2022 file has 33 states but 34 held a Senate general election.
SENATE_MISSING_FROM_MIT = {(2022, "CT")}


def prev_pres(year: int) -> tuple[int, int]:
    """Most recent presidential year strictly before `year`, and the one before that."""
    p = year - 4 if year % 4 == 0 else year - 2
    return p, p - 4


def build_hist(nat: dict[int, float], st: pd.Series) -> pd.DataFrame:
    rows = []
    for year, (fname, p1, p2) in HOUSE_HIST.items():
        shares = sheet_shares(fname)
        for race_id, r in shares.iterrows():
            rows.append(dict(year=year, race_id=race_id, chamber="House", pres_prev_year=p1,
                             pres_prev_dem_2p=r[p1], pres_prev2_year=p2 if p2 else None,
                             pres_prev2_dem_2p=r[p2] if p2 else None, source=f"Downballot {fname}"))
    sen = pd.read_csv(MIT_SENATE_FILE)
    sen = sen[sen["year"].isin(SENATE_YEARS) & (sen["stage"].str.lower() == "gen")]
    for year, state in sorted(set(zip(sen["year"], sen["state_po"])) | SENATE_MISSING_FROM_MIT):
        p1, p2 = prev_pres(year)
        rows.append(dict(year=year, race_id=f"{state}-SEN", chamber="Senate", pres_prev_year=p1,
                         pres_prev_dem_2p=st.get((p1, state)), pres_prev2_year=p2,
                         pres_prev2_dem_2p=st.get((p2, state)), source="MIT president 1976-2024 (statewide)"))
    out = pd.DataFrame(rows)
    out["national_pres_dem_2p"] = out["pres_prev_year"].map(nat)
    out["national_pres_prev2_dem_2p"] = out["pres_prev2_year"].map(nat)
    out["pres_prev2_year"] = out["pres_prev2_year"].astype("Int64")
    cols = ["year", "race_id", "chamber", "pres_prev_dem_2p", "pres_prev2_dem_2p", "national_pres_dem_2p",
            "pres_prev_year", "pres_prev2_year", "national_pres_prev2_dem_2p", "source"]
    return out[cols].sort_values(["year", "chamber", "race_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- checks
def validate(lean: pd.DataFrame, hist: pd.DataFrame) -> list[str]:
    msgs = []
    if len(lean) != 470 or lean["race_id"].duplicated().any():
        raise ValueError(f"lean_2026 has {len(lean)} rows / duplicates")
    if lean["pres24_dem_2p"].isna().any():
        raise ValueError(f"missing pres24: {lean.loc[lean.pres24_dem_2p.isna(), 'race_id'].tolist()}")
    if not lean["pres24_dem_2p"].between(0.05, 0.95).all():
        raise ValueError("pres24 share out of range")
    if hist.duplicated(["year", "race_id"]).any():
        raise ValueError("duplicate (year, race_id) in lean_hist")
    # Senate state-years with a general election (regular classes 1/2/3 cycle: 33, 33/34 states + specials)
    expected_sen = {2010: 36, 2012: 33, 2014: 34, 2016: 34, 2018: 33, 2020: 34, 2022: 34, 2024: 33}
    got_sen = hist[hist.chamber == "Senate"].groupby("year").size().to_dict()
    if got_sen != expected_sen:
        raise ValueError(f"Senate state-years per year {got_sen} != expected {expected_sen}")
    house = lean[lean.chamber == "House"]
    diff = (house["pvi_num"] - house["pvi_calc"]).abs()
    msgs.append(f"pvi_calc vs Wikipedia Cook PVI (House, both present): mean |diff| {diff.mean():.2f}, "
                f"max {diff.max():.1f}, n>1.5: {(diff > 1.5).sum()}")
    for yr, g in hist.groupby(["year", "chamber"]):
        msgs.append(f"hist {yr}: {len(g)} rows, prev missing {g.pres_prev_dem_2p.isna().sum()}")
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download all raw files")
    args = ap.parse_args()
    fetch_all(args.refresh)
    votes = mit_pres()
    nat, st = national_shares(votes), state_shares(votes)
    lean, meta = build_2026(nat, st)
    hist = build_hist(nat, st)
    for line in validate(lean, hist):
        print(line)
    CLEAN.mkdir(parents=True, exist_ok=True)
    lean.to_csv(CLEAN / "lean_2026.csv", index=False)
    hist.to_csv(CLEAN / "lean_hist.csv", index=False)
    print(f"wrote lean_2026.csv ({len(lean)} rows), lean_hist.csv ({len(hist)} rows); wiki rev {meta['wiki_revid']}")
    print("national two-party D share:", {k: v for k, v in nat.items() if k >= 2000})
    return 0


if __name__ == "__main__":
    sys.exit(main())
