#!/usr/bin/env python3
"""Build data/features/race_info_2026.csv: one row per 2026 race (435 House + 35 Senate).

Sources (all English Wikipedia, via the MediaWiki parse API; raw JSON cached in data/raw/wiki_2026/):
  * "2026 United States House of Representatives elections"  -> per-district 2026 Cook PVI (on 2026 lines),
    incumbents (+ status), candidates.
  * "2026 United States House of Representatives election ratings" -> Cook / Inside Elections / Sabato
    (+ Economist, Split Ticket, DDHQ, FiftyPlusOne, Fox, Silver, RCP) for every seat that at least one outlet
    rates as competitive. Seats missing from that table are rated Safe/Solid by every outlet (per the page);
    we impute "Safe R"/"Safe D" for them and flag ratings_imputed=1.
  * "2026 United States Senate elections" -> state PVI, incumbents, candidates, ratings for all 35 races.
  * "2024 United States House of Representatives elections" -> last House result (2024 general). In the 9
    states with new 2026 maps, last_margin is left blank; the old same-numbered district's result is in
    last_margin_old_lines and a rough new-lines estimate (old margin + 2 * PVI change) in last_margin_newlines_est.
  * "Cook Partisan Voting Index" at revision 1292130997 (May 2025) -> 2025 Cook PVI on the 2024 lines
    (pvi_old_lines). It equals the 2026 PVI in every district of the 41 states that kept their maps.
  * "2020/2022/2024 United States Senate elections" -> last result for each Senate seat.

Run with an interpreter that has requests + bs4, e.g. /usr/local/bin/python3 on Pau's Mac.

Usage:
  python3 src/fetch_race_info_2026.py            # use cached raw pages if present, else download
  python3 src/fetch_race_info_2026.py --refresh  # re-download latest revisions
  python3 src/fetch_race_info_2026.py --pinned   # re-download the exact revisions recorded in the meta file

Requires: requests, beautifulsoup4 (html.parser backend; lxml not needed). pandas is not needed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "wiki_2026"
OUT_CSV = ROOT / "data" / "features" / "race_info_2026.csv"
OUT_META = ROOT / "data" / "features" / "race_info_2026_meta.json"
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"

API = "https://en.wikipedia.org/w/api.php"
UA = ("ModelCitizensForecast/0.1 (Stanford Democracy Day 'Model Citizens' student election-forecast project; "
      "research use, low request rate) python-requests")

PAGES = {
    "house2026": "2026 United States House of Representatives elections",
    "house2026_ratings": "2026 United States House of Representatives election ratings",
    "senate2026": "2026 United States Senate elections",
    "house2024": "2024 United States House of Representatives elections",
    "senate2020": "2020 United States Senate elections",
    "senate2022": "2022 United States Senate elections",
    "senate2024": "2024 United States Senate elections",
    "cookpvi2025": "Cook Partisan Voting Index",
}
# Pages fetched at a fixed old revision: the Cook PVI article as of 2025-05-25 lists the 2025 Cook PVI
# (2020+2024 presidential results) on the 2024 district lines, i.e. before any mid-decade redistricting.
FIXED_OLDID = {"cookpvi2025": 1292130997}

# States that use new congressional maps in 2026 (2026 House page, "Mid-decade redistricting changes",
# status "New districts enacted"). Missouri's 2025 map and Virginia's amendment were blocked -> old maps.
REDISTRICTED_2026 = {"AL", "CA", "FL", "LA", "NC", "OH", "TN", "TX", "UT"}

STATES = {
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

RATING_NUM = {"Safe D": -3, "Likely D": -2, "Lean D": -1, "Tilt D": -0.5, "Tossup": 0,
              "Tilt R": 0.5, "Lean R": 1, "Likely R": 2, "Safe R": 3}

# Hand-checked facts from the prose of the 2026 Senate page ("Republican-held seats with independent
# challengers" and the state sections), which the candidate tables alone do not show. Kalshi still lists the
# withdrawn Democrats in ID/NE/SD as the "DEM" side of its markets.
MANUAL = {
    "MT-SEN": {"other_major_candidate": "Seth Bodnar (Independent)",
               "note": "three-way race: independent Seth Bodnar (endorsed by Jon Tester; ex-MT Democratic chair "
                       "resigned to back him) is the main challenger; Democrat Alani Bankhead stays on the ballot"},
    "ID-SEN": {"note": "Democratic nominee David Roth withdrew (Jul 28); independent Todd Achilles is the main "
                       "challenger"},
    "NE-SEN": {"note": "Democratic nominee Cindy Burbank withdrew (Jul 17); state Democrats back independent Dan "
                       "Osborn"},
    "SD-SEN": {"note": "Democratic nominee Julian Beaudion withdrew (Aug 4); independent Brian Bengs (2022 D "
                       "nominee) is the main challenger"},
}

# Outlet header prefix (as in the Wikipedia ratings tables) -> output column stem.
OUTLETS = {"Cook": "cook", "IE": "inside_elections", "Sabato": "sabato", "Econ": "economist",
           "ST": "splitticket", "DDHQ": "ddhq", "FPO": "fiftyplusone", "Fox": "fox", "Silver": "silver",
           "RCP": "rcp", "NBC": "nbc"}
MAIN_OUTLETS = ["cook", "inside_elections", "sabato"]
EXTRA_OUTLETS = ["economist", "splitticket", "ddhq", "fiftyplusone", "fox", "silver", "rcp", "nbc"]


# --------------------------------------------------------------------------------------------------- fetching
def fetch(key: str, refresh: bool = False, oldid: int | None = None) -> dict:
    title = PAGES[key]
    fixed = FIXED_OLDID.get(key)
    if fixed:
        oldid = fixed
    path = RAW_DIR / (title.replace(" ", "_") + (f"_oldid{fixed}" if fixed else "") + ".json")
    if fixed and path.exists():
        return json.loads(path.read_text())
    if path.exists() and not refresh and oldid is None:
        return json.loads(path.read_text())
    params = {"action": "parse", "prop": "text|revid", "format": "json", "formatversion": "2",
              "redirects": "1"}
    if oldid:
        params["oldid"] = str(oldid)
    else:
        params["page"] = title
    for attempt in range(4):
        r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=60)
        if r.status_code == 200 and "parse" in r.json():
            break
        time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"failed to fetch {title}: {r.status_code} {r.text[:200]}")
    data = r.json()
    data["_fetched_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    time.sleep(1.0)  # be polite
    return data


def soup_of(data: dict) -> BeautifulSoup:
    return BeautifulSoup(data["parse"]["text"], "html.parser")


# --------------------------------------------------------------------------------------------------- parsing
def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"\[\s*[^\]]{1,6}\s*\]", "", text)  # [12], [a], [ n ]
    return re.sub(r"\s+", " ", text).strip()


def cell_text(cell: Tag) -> str:
    c = BeautifulSoup(str(cell), "html.parser")
    for sup in c.find_all("sup", class_="reference"):
        sup.decompose()
    for hidden in c.find_all(style=re.compile(r"display:\s*none")):
        hidden.decompose()
    return clean(c.get_text(" "))


def table_grid(table: Tag) -> list[list[Tag]]:
    """Expand a wikitable into a rectangular grid of cell Tags (handles rowspan/colspan)."""
    grid: list[list[Tag | None]] = []
    pending: dict[tuple[int, int], Tag] = {}
    for ri, tr in enumerate(table.find_all("tr")):
        if tr.find_parent("table") is not table:
            continue
        row: list[Tag | None] = []
        ci = 0
        cells = tr.find_all(["th", "td"], recursive=False)
        it = iter(cells)
        while True:
            if (len(grid), ci) in pending:
                row.append(pending.pop((len(grid), ci)))
                ci += 1
                continue
            try:
                cell = next(it)
            except StopIteration:
                # flush any trailing spanned cells
                while (len(grid), ci) in pending:
                    row.append(pending.pop((len(grid), ci)))
                    ci += 1
                break
            rs = int(re.sub(r"\D", "", cell.get("rowspan", "1")) or 1)
            cs = int(re.sub(r"\D", "", cell.get("colspan", "1")) or 1)
            for k in range(cs):
                row.append(cell)
                for d in range(1, rs):
                    pending[(len(grid) + d, ci)] = cell
                ci += 1
        grid.append(row)
    return grid


def party_code(label: str) -> str:
    lab = label.lower()
    if re.search(r"democrat|dfl|farmer", lab) and "independent democrat" not in lab:
        return "DEM"
    if "republican" in lab:
        return "REP"
    if lab.startswith("independent") or lab == "nonpartisan":
        return "IND"
    return "OTH"


COLOR_PARTY = {"#0044c9": "DEM", "#3333ff": "DEM", "#e81b23": "REP", "#ff0000": "REP"}


def parse_candidates(cell: Tag) -> list[dict]:
    """Parse a plainlist candidate cell. If the cell has rounds (first round / runoff / instant runoff),
    only the final round is returned."""
    sections: list[list[dict]] = [[]]
    items = cell.find_all("li") or [cell]  # single (unopposed) candidates are not in a list
    for li in items:
        raw = cell_text(li)
        if re.match(r"^(first round|runoff|instant runoff|second round|final round|round \d)\b", raw,
                    re.I) or (raw.endswith(":") and len(raw) < 30):
            if sections[-1]:
                sections.append([])
            continue
        color = None
        sp = li.find("span", style=re.compile("color"))
        if sp:
            m = re.search(r"color:\s*(#[0-9A-Fa-f]{6})", sp.get("style", ""))
            color = m.group(1).lower() if m else None
        txt = raw.replace("▌", "").strip()
        txt = re.sub(r"^Y\s+", "", txt)
        m = re.match(r"^(.*?)\s*\(\s*([^()]*?)\s*\)\s*(?:([\d.]+)\s*%)?", txt)
        if not m:
            continue
        name, plabel, pct = m.group(1).strip(), m.group(2).strip(), m.group(3)
        if not name:
            continue
        pc = party_code(plabel)
        if pc == "OTH" and color in COLOR_PARTY and not plabel:
            pc = COLOR_PARTY[color]
        withdrawn = bool(re.search(r"withdr|disqualif|deceased|died", txt, re.I))
        href = None
        for a in li.find_all("a", href=re.compile(r"^/wiki/")):
            if clean(a.get_text()) and clean(a.get_text()) in name:
                href = a["href"]
                break
        sections[-1].append({"name": name, "party_label": plabel, "party": pc, "href": href,
                             "pct": float(pct) if pct else None, "withdrawn": withdrawn})
    sections = [s for s in sections if s]
    return sections[-1] if sections else []


def parse_pvi(text: str) -> float | None:
    t = clean(text).upper().replace("−", "-").replace("–", "-")
    if t.startswith("EVEN"):
        return 0.0
    m = re.match(r"^([RD])\s*\+\s*(\d+(?:\.\d+)?)", t)
    if not m:
        return None
    v = float(m.group(2))
    return v if m.group(1) == "R" else -v


def norm_rating(text: str) -> str | None:
    t = clean(text)
    t = re.sub(r"\(flip\)", "", t, flags=re.I).strip()
    t = t.replace("Solid", "Safe").replace("Toss-up", "Tossup").replace("Toss up", "Tossup")
    t = re.sub(r"\s+", " ", t)
    if t.lower().startswith("tossup"):
        return "Tossup"
    m = re.match(r"^(Safe|Likely|Lean|Tilt)\s+([DRI])", t)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return None


def rating_num(r: str | None) -> float | None:
    if r is None:
        return None
    if r in RATING_NUM:
        return RATING_NUM[r]
    if r.endswith(" I"):  # independent favoured: treat as anti-R (no such case expected)
        return RATING_NUM[r.replace(" I", " D")]
    return None


def district_id(label: str) -> tuple[str, str] | None:
    label = clean(label)
    m = re.match(r"^(.+?)\s+(\d+|at-large)$", label, re.I)
    if not m or m.group(1) not in STATES:
        return None
    st = STATES[m.group(1)]
    d = "AL" if m.group(2).lower() == "at-large" else f"{int(m.group(2)):02d}"
    return st, d


def state_tables(soup: BeautifulSoup) -> list[tuple[str, Tag]]:
    """(state name, wikitable) for each per-state table under an h2 named after a state."""
    out = []
    for t in soup.find_all("table", class_="wikitable"):
        h = t.find_previous("h2")
        if h and clean(h.get_text()) in STATES:
            out.append((clean(h.get_text()), t))
    return out


def header_index(grid: list[list[Tag]], patterns: dict[str, str], nrows: int = 3) -> dict[str, int]:
    idx = {}
    for row in grid[:nrows]:
        for ci, cell in enumerate(row):
            if cell is None or cell.name != "th":
                continue
            txt = cell_text(cell)
            for key, pat in patterns.items():
                if key not in idx and re.search(pat, txt, re.I):
                    idx[key] = ci
    return idx


# -------------------------------------------------------------------------------- House 2026 (main page)
def parse_house_2026(soup: BeautifulSoup) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for state, t in state_tables(soup):
        grid = table_grid(t)
        hi = header_index(grid, {"pvi": r"PVI", "member": r"^Member", "party": r"^Party",
                                 "first": r"First", "status": r"^Status", "cands": r"^Candidates"})
        for row in grid:
            if not row or row[0] is None or row[0].name != "th":
                continue
            did = district_id(cell_text(row[0]))
            if not did:
                continue
            rid = f"{did[0]}-{did[1]}"
            rec = out.setdefault(rid, {"state": did[0], "district": did[1], "incumbents": [],
                                       "pvi_text": cell_text(row[hi["pvi"]]),
                                       "candidates": parse_candidates(row[hi["cands"]])})
            mcell = row[hi["member"]]
            mtxt = cell_text(mcell)
            if mtxt.lower().startswith("none"):
                rec["new_seat"] = True
                continue
            if mtxt.lower().startswith("vacant"):
                st_txt = cell_text(row[hi["status"]])
                m = re.search(r"Rep\.\s+(.+?)\s+\(([DRI])\)", st_txt)
                rec["vacancy"] = {"previous": m.group(1) if m else "", "party":
                                  {"D": "DEM", "R": "REP", "I": "IND"}.get(m.group(2), "") if m else "",
                                  "status": st_txt}
                continue
            a = mcell.find("span", class_="fn") or mcell.find("a")
            name = clean(a.get_text()) if a else re.split(r"Redistricted", mtxt)[0].strip()
            moved = re.search(r"Redistricted from the (\S+) district", mtxt)
            rec["incumbents"].append({
                "name": name, "href": a.get("href") if a and a.name == "a" else (
                    a.find("a").get("href") if a and a.find("a") else None), "party_label": cell_text(row[hi["party"]]),
                "party": party_code(cell_text(row[hi["party"]])),
                "first_elected": cell_text(row[hi["first"]]), "status": cell_text(row[hi["status"]]),
                "redistricted_from": moved.group(1) if moved else None})
    return out


def incumbent_running(status: str) -> bool:
    s = status.lower()
    if re.search(r"retir|lost|resign|died|ineligible|withdr|not running|running in the|run in the|"
                 r"running for|appointed to|term-limited|defeated", s):
        return False
    return bool(re.search(r"renominated|advanced|running|nominated|seeking|re-elected|unopposed", s))


# ------------------------------------------------------------------------------------ ratings tables
def parse_ratings_table(t: Tag, senate: bool) -> tuple[dict[str, dict], dict[str, str]]:
    grid = table_grid(t)
    # second header row holds outlet names with dates
    hdr_row = None
    for row in grid[:3]:
        if any(c is not None and re.match(r"^Cook", cell_text(c)) for c in row):
            hdr_row = row
            break
    outlet_cols, outlet_dates = {}, {}
    for ci, c in enumerate(hdr_row):
        txt = cell_text(c)
        for pre, stem in OUTLETS.items():
            if re.match(rf"^{pre}\b", txt) and stem not in outlet_cols:
                outlet_cols[stem] = ci
                m = re.search(r"([A-Z][a-z]{2}\.? \d{1,2}, \d{4})", txt)
                outlet_dates[stem] = m.group(1) if m else ""
    out = {}
    for row in grid:
        if not row or row[0] is None or row[0].name != "th":
            continue
        label = cell_text(row[0])
        if senate:
            st_name = re.sub(r"\s*\(.*\)$", "", label).strip()
            if st_name not in STATES:
                continue
            rid = f"{STATES[st_name]}-SEN"
        else:
            did = district_id(label)
            if not did:
                continue
            rid = f"{did[0]}-{did[1]}"
        rec = {"pvi_text": cell_text(row[1]), "incumbent_text": cell_text(row[2]),
               "last_text": cell_text(row[3]), "label": label}
        for stem, ci in outlet_cols.items():
            rec[stem] = norm_rating(cell_text(row[ci])) if ci < len(row) else None
            rec[stem + "_raw"] = cell_text(row[ci]) if ci < len(row) else ""
        out[rid] = rec
    return out, outlet_dates


# --------------------------------------------------------------------------------------------- Senate 2026
def parse_senate_2026(soup: BeautifulSoup) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in soup.find_all("table", class_="wikitable"):
        grid = table_grid(t)
        hi = header_index(grid, {"pvi": r"^PVI", "senator": r"^Senator", "party": r"^Party",
                                 "status": r"^(Status|Results)$", "cands": r"^Candidates", "last": r"^Last race"})
        if not {"senator", "party", "cands", "status"} <= hi.keys():
            continue
        for row in grid:
            if not row or row[0] is None or row[0].name != "th":
                continue
            label = cell_text(row[0])
            st_name = re.sub(r"\s*\(.*\)$", "", label).strip()
            if st_name not in STATES:
                continue
            special = bool(re.search(r"special|Class 3", label, re.I)) and st_name in ("Florida", "Ohio")
            rid = f"{STATES[st_name]}-SEN"
            # the template's FL-SEN / OH-SEN are the specials (no regular race in FL/OH in 2026)
            if rid in out and not special:
                continue
            scell = row[hi["senator"]]
            a = scell.find("a")
            out[rid] = {
                "label": label, "special": special, "pvi_text": cell_text(row[hi["pvi"]]),
                "incumbent": clean(a.get_text()) if a else cell_text(scell),
                "incumbent_href": a.get("href") if a else None,
                "party_label": cell_text(row[hi["party"]]), "party": party_code(cell_text(row[hi["party"]])),
                "status": cell_text(row[hi["status"]]), "candidates": parse_candidates(row[hi["cands"]]),
                "table_section": clean(t.find_previous(["h2", "h3"]).get_text())}
    return out


# ------------------------------------------------------------------------------------------ past results
def margin_from_candidates(cands: list[dict]) -> tuple[float | None, str]:
    """R-minus-D two-party margin in points (100*(R-D)/(R+D)), summing all candidates of each party (jungle
    primaries). If one major party is absent, the strongest other candidate stands in for it (e.g. AK-SEN 2020
    Al Gross, an independent who was the Democratic nominee). Unopposed / same-party races give +-100."""
    if not cands or all(c["pct"] is None for c in cands):
        return None, "no result parsed"
    r = sum(c["pct"] for c in cands if c["party"] == "REP" and c["pct"] is not None)
    d = sum(c["pct"] for c in cands if c["party"] == "DEM" and c["pct"] is not None)
    oth = sorted([c for c in cands if c["party"] not in ("REP", "DEM") and c["pct"]], key=lambda c: -c["pct"])
    if r > 0 and d > 0:
        return round(100 * (r - d) / (r + d), 2), "two-party"
    if r > 0 and oth:
        o = oth[0]["pct"]
        return round(100 * (r - o) / (r + o), 2), f"R vs top non-D ({oth[0]['name']}, {oth[0]['party_label']})"
    if d > 0 and oth:
        o = oth[0]["pct"]
        return round(-100 * (d - o) / (d + o), 2), f"D vs top non-R ({oth[0]['name']}, {oth[0]['party_label']})"
    n_r = sum(1 for c in cands if c["party"] == "REP")
    n_d = sum(1 for c in cands if c["party"] == "DEM")
    if r > 0:
        return 100.0, "R vs R (same-party)" if n_r > 1 else "uncontested R"
    if d > 0:
        return -100.0, "D vs D (same-party)" if n_d > 1 else "uncontested D"
    return None, "no D or R"


def parse_pvi_table(soup: BeautifulSoup) -> dict[str, float | None]:
    """District -> PVI from the 'By congressional district' table of the Cook PVI article."""
    out = {}
    for t in soup.find_all("table", class_="wikitable"):
        grid = table_grid(t)
        for row in grid:
            if len(row) < 2 or row[0] is None:
                continue
            did = district_id(cell_text(row[0]))
            if did:
                out[f"{did[0]}-{did[1]}"] = parse_pvi(cell_text(row[1]))
    return out


def parse_house_2024(soup: BeautifulSoup) -> dict[str, dict]:
    out = {}
    for state, t in state_tables(soup):
        grid = table_grid(t)
        hi = header_index(grid, {"cands": r"^Candidates", "pvi": r"PVI"})
        for row in grid:
            if not row or row[0] is None or row[0].name != "th":
                continue
            did = district_id(cell_text(row[0]))
            if not did:
                continue
            rid = f"{did[0]}-{did[1]}"
            if rid in out:
                continue
            cands = parse_candidates(row[hi["cands"]])
            m, kind = margin_from_candidates(cands)
            out[rid] = {"margin": m, "kind": kind, "pvi2024_text": cell_text(row[hi["pvi"]]),
                        "summary": "; ".join(f"{c['name']} ({c['party']}) {c['pct']}" for c in cands)}
    return out


def parse_senate_results(soup: BeautifulSoup) -> dict[tuple[str, bool], dict]:
    """(state, is_special) -> last result, from a past Senate elections page."""
    out = {}
    for t in soup.find_all("table", class_="wikitable"):
        grid = table_grid(t)
        hi = header_index(grid, {"cands": r"candidates$"})
        if "cands" not in hi:
            continue
        for row in grid:
            if not row or row[0] is None or row[0].name != "th":
                continue
            label = cell_text(row[0])
            st_name = re.sub(r"\s*\(.*\)$", "", label).strip()
            if st_name not in STATES:
                continue
            special = bool(re.search(r"special|Class", label, re.I))
            cands = parse_candidates(row[hi["cands"]])
            if not cands or all(c["pct"] is None for c in cands):
                continue
            m, kind = margin_from_candidates(cands)
            key = (st_name, special, label)
            out[key] = {"margin": m, "kind": kind, "label": label,
                        "summary": "; ".join(f"{c['name']} ({c['party']}) {c['pct']}" for c in cands)}
    return out


# ------------------------------------------------------------------------------------------- assembling
def _norm_name(n: str) -> str:
    n = re.sub(r"\b(Jr|Sr|II|III|IV)\b\.?", "", n)
    n = re.sub(r"\b[A-Z]\.\s*", "", n)          # middle initials
    n = re.sub(r"[\"'“”‘’.,()]", " ", n)
    return re.sub(r"\s+", " ", n).strip().lower()


def is_same_person(c: dict, name: str | None, href: str | None) -> bool:
    if not name:
        return False
    if href and c.get("href"):
        return c["href"] == href
    return c["name"] == name or _norm_name(c["name"]) == _norm_name(name)


def order_party(cands: list[dict], party: str, inc_name: str | None, inc_href: str | None) -> list[dict]:
    """Candidates of one party, incumbent first, then candidates with a Wikipedia article, then the rest."""
    lst = [c for c in cands if c["party"] == party and not c["withdrawn"]]
    return sorted(lst, key=lambda c: (not is_same_person(c, inc_name, inc_href), c.get("href") is None))


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:g}"
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download latest page revisions")
    ap.add_argument("--pinned", action="store_true", help="re-download revisions recorded in the meta file")
    args = ap.parse_args()

    pinned = {}
    if args.pinned and OUT_META.exists():
        pinned = {k: v["revid"] for k, v in json.loads(OUT_META.read_text())["pages"].items()}
    raw = {k: fetch(k, refresh=args.refresh, oldid=pinned.get(k)) for k in PAGES}
    soups = {k: soup_of(v) for k, v in raw.items()}

    template = list(csv.DictReader(open(TEMPLATE, newline="", encoding="utf-8-sig")))
    tmpl_ids = [r["race_id"].strip() for r in template]
    tmpl_chamber = {r["race_id"].strip(): r["chamber"].strip() for r in template}

    house = parse_house_2026(soups["house2026"])
    rt_tables = [t for t in soups["house2026_ratings"].find_all("table", class_="wikitable")
                 if re.search(r"Ratings", t.get_text()[:300])]
    house_ratings, house_rating_dates = parse_ratings_table(rt_tables[0], senate=False)
    senate = parse_senate_2026(soups["senate2026"])
    st_tables = [t for t in soups["senate2026"].find_all("table", class_="wikitable")
                 if t.find("th") and re.search(r"Ratings", t.get_text()[:300])]
    senate_ratings, senate_rating_dates = parse_ratings_table(st_tables[0], senate=True)
    h24 = parse_house_2024(soups["house2024"])
    s20 = parse_senate_results(soups["senate2020"])
    s22 = parse_senate_results(soups["senate2022"])
    s24 = parse_senate_results(soups["senate2024"])
    pvi_old = parse_pvi_table(soups["cookpvi2025"])

    def senate_last(state_name: str, rid: str) -> tuple[dict | None, str]:
        # Which past election last filled each 2026 seat.
        if rid == "FL-SEN":
            src, special, year = s22, False, 2022          # Rubio's class 3 seat, 2022 regular
        elif rid == "OH-SEN":
            src, special, year = s22, False, 2022          # Vance's class 3 seat, 2022 regular
        elif rid == "OK-SEN":
            src, special, year = s22, True, 2022           # Mullin, 2022 special (class 2)
        elif rid == "NE-SEN":
            src, special, year = s24, True, 2024           # Ricketts, 2024 special (class 2)
        else:
            src, special, year = s20, False, 2020          # class 2, 2020 (GA: Jan 2021 runoff)
        for (st, sp, label), rec in src.items():
            if st == state_name and sp == special:
                return rec, f"{year} {'special' if special else 'general'}"
        return None, ""

    rows, problems = [], []
    for rid in tmpl_ids:
        chamber = tmpl_chamber[rid]
        st = rid.split("-")[0]
        notes: list[str] = []
        row = {"race_id": rid, "chamber": chamber, "state": st}
        ratings_src: dict | None = None
        if chamber == "House":
            rec = house.get(rid)
            row["district"] = rid.split("-")[1]
            if rec is None:
                problems.append(f"{rid}: not found on House page")
                rows.append(row)
                continue
            row["pvi"] = parse_pvi(rec["pvi_text"])
            row["pvi_source"] = ("Cook PVI 2026 (July 29, 2026 release, 2026 district lines) via Wikipedia "
                                 "'2026 United States House of Representatives elections'")
            incs = rec["incumbents"]
            running = [i for i in incs if incumbent_running(i["status"])]
            inc = running[0] if running else (incs[0] if incs else None)
            vac = rec.get("vacancy")
            row["incumbent_name"] = inc["name"] if inc else ""
            row["_inc_href"] = inc.get("href") if inc else None
            row["incumbent_party"] = inc["party"] if inc else (vac["party"] if vac and vac["party"] else "none")
            row["incumbent_running"] = int(bool(running))
            row["open_seat"] = int(not running)
            row["incumbent_status"] = "; ".join(
                [f"{i['name']}: {i['status']}" for i in incs] + ([f"Vacant: {vac['status']}"] if vac else [])
            ) or "new seat"
            if vac:
                notes.append(f"seat vacant ({vac['status']})" + ("; incumbent_party = party of previous holder"
                                                                  if not inc else ""))
            if len(incs) > 1:
                notes.append("multiple incumbents listed: " + "; ".join(
                    f"{i['name']} ({i['party']}, {i['status']})" for i in incs))
            if len(running) > 1:
                notes.append("incumbent-vs-incumbent race")
            if rec.get("new_seat"):
                notes.append("newly created seat (no incumbent)")
            for i in incs:
                if i["redistricted_from"]:
                    notes.append(f"{i['name']} redistricted from the {i['redistricted_from']} district")
            cands = rec["candidates"]
            ratings_src = house_ratings.get(rid)
            last = h24.get(rid)
            same_lines = st not in REDISTRICTED_2026
            old_m = last["margin"] if last else None
            row["pvi_old_lines"] = pvi_old.get(rid)
            row["last_margin_old_lines"] = old_m
            row["last_margin"] = old_m if same_lines else None
            row["last_margin_kind"] = (last["kind"] + ", 2024 general" + ("" if same_lines else
                                       ", OLD lines (same district number)")) if last else ""
            row["last_margin_same_lines"] = int(same_lines)
            row["redistricted_2026"] = int(not same_lines)
            # rough new-lines estimate: shift the old-lines result by twice the PVI change (PVI ~ half margin)
            if same_lines:
                row["last_margin_newlines_est"] = old_m
            elif old_m is not None and last["kind"] == "two-party" and row["pvi_old_lines"] is not None \
                    and row["pvi"] is not None:
                row["last_margin_newlines_est"] = round(max(-100, min(100, old_m + 2 * (row["pvi"] -
                                                                                     row["pvi_old_lines"]))), 2)
            if not same_lines:
                d_pvi = (row["pvi"] - row["pvi_old_lines"]) if row.get("pvi_old_lines") is not None else None
                notes.append(f"{st} redrew its map for 2026: pvi is on the 2026 lines (old-lines 2025 PVI "
                             f"{fmt(row.get('pvi_old_lines'))}, change {fmt(d_pvi)}); last_margin left blank, the 2024 "
                             f"result in the old same-numbered district is in last_margin_old_lines")
            elif row.get("pvi_old_lines") is not None and row["pvi"] != row["pvi_old_lines"]:
                notes.append(f"2026 PVI {fmt(row['pvi'])} differs from 2025 PVI {fmt(row['pvi_old_lines'])} though "
                             f"{st} kept its map")
            if ratings_src and parse_pvi(ratings_src["pvi_text"]) != row["pvi"]:
                notes.append(f"ratings-page CPVI {ratings_src['pvi_text']} differs from House-page 2026 PVI "
                             f"{rec['pvi_text']}")
            if st in ("CA", "WA"):
                notes.append("top-two primary state (general is top-two)")
            if st == "LA":
                notes.append("Louisiana: Nov 3 is a jungle (all-party) primary on new lines; runoff Dec 12 if "
                             "no one >50%")
            if st == "GA":
                notes.append("Georgia: runoff (Dec) if no candidate >50%")
            if st in ("AK",):
                notes.append("Alaska: top-four primary + ranked-choice general")
            if st == "ME":
                notes.append("Maine: ranked-choice voting")
        else:
            rec = senate.get(rid)
            row["district"] = "SEN"
            if rec is None:
                problems.append(f"{rid}: not found on Senate page")
                rows.append(row)
                continue
            row["pvi"] = parse_pvi(rec["pvi_text"])
            row["pvi_source"] = "Cook PVI 2025 (statewide) via Wikipedia '2026 United States Senate elections'"
            run = incumbent_running(rec["status"])
            row["incumbent_name"] = rec["incumbent"]
            row["_inc_href"] = rec.get("incumbent_href")
            row["incumbent_party"] = rec["party"]
            row["incumbent_running"] = int(run)
            row["open_seat"] = int(not run)
            row["incumbent_status"] = rec["status"]
            if rec["special"]:
                notes.append("special election (class 3 seat; incumbent is an appointee)")
            elif re.search(r"appoint", rec["status"], re.I):
                notes.append("incumbent is an interim appointee")
            cands = rec["candidates"]
            ratings_src = senate_ratings.get(rid)
            state_name = [k for k, v in STATES.items() if v == st][0]
            last, last_desc = senate_last(state_name, rid)
            row["last_margin"] = last["margin"] if last else None
            row["last_margin_old_lines"] = row["last_margin"]
            row["last_margin_newlines_est"] = row["last_margin"]
            row["pvi_old_lines"] = row["pvi"]
            if rid == "GA-SEN":
                last_desc = "Jan 2021 runoff"
            row["last_margin_kind"] = f"{last['kind']}, {last_desc}" if last else ""
            row["last_margin_same_lines"] = 1
            row["redistricted_2026"] = 0
            if st == "LA":
                notes.append("Louisiana: closed party primaries were held; general Nov 3")
            if st == "GA":
                notes.append("Georgia: runoff (Dec) if no candidate >50%")
            if st == "AK":
                notes.append("Alaska: top-four primary + ranked-choice general")
            if st == "ME":
                notes.append("Maine: ranked-choice voting")

        # candidates
        inc_name, inc_href = row.get("incumbent_name") or None, row.pop("_inc_href", None)
        dem_c = order_party(cands, "DEM", inc_name, inc_href)
        rep_c = order_party(cands, "REP", inc_name, inc_href)
        dems, reps = [c["name"] for c in dem_c], [c["name"] for c in rep_c]
        others = [c for c in cands if c["party"] not in ("DEM", "REP") and not c["withdrawn"]]
        jungle = st == "LA" and chamber == "House"  # all-party primary on Nov 3: list everyone
        top_two = st in ("CA", "WA") and chamber == "House"
        row["dem_candidate"] = "; ".join(dems) if jungle else (dems[0] if dems else "")
        row["rep_candidate"] = "; ".join(reps) if jungle else (reps[0] if reps else "")
        other_major, other_minor = [], [c for c in others]
        if not jungle:
            extra_same = [(c, "DEM") for c in dem_c[1:]] + [(c, "REP") for c in rep_c[1:]]
            if top_two:  # same-party top-two general: the second candidate is a real opponent
                other_major += [f"{c['name']} ({p})" for c, p in extra_same]
            else:  # e.g. Alaska top-four: extra same-party names are minor
                other_minor = [c for c, _ in extra_same] + other_minor
            if not dems or not reps:
                # a non-major-party candidate matters when one major party has no nominee: take those with a
                # Wikipedia article (notable), else the sole other candidate
                notable = [c for c in others if c.get("href")]
                pick = notable or (others if len(others) == 1 else [])
                other_major += [f"{c['name']} ({c['party_label'] or c['party']})" for c in pick]
                other_minor = [c for c in other_minor if c not in pick]
        row["other_major_candidate"] = "; ".join(other_major)
        row["other_candidates"] = "; ".join(f"{c['name']} ({c['party_label']})" for c in other_minor)
        if not dems and not reps:
            notes.append("no D or R candidate parsed")
        elif not dems:
            notes.append("no Democratic candidate" + (" (non-major opponent: see other_major_candidate)"
                                                      if other_major else " (R effectively unopposed)"))
        elif not reps:
            notes.append("no Republican candidate" + (" (D vs non-R/same-party)" if others or len(dems) > 1
                                                      else " (D unopposed)"))
        if not jungle and (len(dems) > 1 or len(reps) > 1):
            notes.append("same-party top-two general" if top_two else
                         "extra same-party candidates on general ballot (in other_candidates)")
        if len(cands) and sum(1 for c in cands if not c["withdrawn"]) == 1:
            notes.append("uncontested (single candidate listed)")
        if top_two and (len(dems) > 1 or len(reps) > 1) and not (dems and reps):
            notes.append("party-level rating is trivially safe; the contest is between same-party candidates")
        man = MANUAL.get(rid, {})
        if "other_major_candidate" in man:
            row["other_major_candidate"] = man["other_major_candidate"]
            row["other_candidates"] = "; ".join(x for x in (row["other_candidates"] or "").split("; ")
                                                if x and x.split(" (")[0] not in man["other_major_candidate"])
        if "note" in man:
            notes.append(man["note"])

        # ratings
        if ratings_src:
            row["ratings_imputed"] = 0
            for stem in MAIN_OUTLETS + EXTRA_OUTLETS:
                if stem in ratings_src:
                    row[stem] = ratings_src[stem]
            row["last_result_text"] = ratings_src["last_text"]
        else:
            row["ratings_imputed"] = 1
            pvi = row.get("pvi")
            if dems and not reps and not others:
                implied = "Safe D"
            elif reps and not dems and not others:
                implied = "Safe R"
            elif pvi is not None and pvi != 0:
                implied = "Safe R" if pvi > 0 else "Safe D"
            else:
                implied = None
            for stem in MAIN_OUTLETS:
                row[stem] = implied
            inc_party = row.get("incumbent_party")
            if implied and inc_party in ("DEM", "REP") and implied[-1] != inc_party[0]:
                notes.append(f"imputed {implied} differs from incumbent party {inc_party} (redistricting flip?)")
            notes.append("not in Wikipedia competitive-ratings table: all outlets Safe/Solid; rating imputed "
                         "from PVI sign")
        nums = []
        for stem in MAIN_OUTLETS + EXTRA_OUTLETS:
            v = rating_num(row.get(stem))
            row[stem + "_num"] = v
            if v is None and row.get(stem) is not None:
                problems.append(f"{rid}: unparsed rating {stem}={row.get(stem)}")
        main_nums = [row[s + "_num"] for s in MAIN_OUTLETS if row.get(s + "_num") is not None]
        all_nums = [row[s + "_num"] for s in MAIN_OUTLETS + EXTRA_OUTLETS if row.get(s + "_num") is not None]
        row["rating_mean"] = round(sum(main_nums) / len(main_nums), 3) if main_nums else None
        row["rating_mean_all"] = round(sum(all_nums) / len(all_nums), 3) if all_nums else None
        row["n_ratings_all"] = len(all_nums)
        if row.get("pvi") is None:
            notes.append("PVI missing")
        row["notes"] = " | ".join(dict.fromkeys(notes))
        rows.append(row)

    cols = ["race_id", "chamber", "state", "district", "pvi", "pvi_source", "incumbent_name", "incumbent_party",
            "incumbent_running", "open_seat", "last_margin", "cook", "inside_elections", "sabato", "cook_num",
            "inside_elections_num", "sabato_num", "rating_mean", "dem_candidate", "rep_candidate",
            "other_major_candidate", "notes",
            # extras
            "ratings_imputed", "redistricted_2026", "pvi_old_lines", "last_margin_same_lines", "last_margin_kind",
            "last_margin_old_lines", "last_margin_newlines_est",
            "last_result_text", "incumbent_status", "other_candidates", "rating_mean_all", "n_ratings_all"] + \
           [s + "_num" for s in EXTRA_OUTLETS]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c)) for c in cols})

    meta = {
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "pages": {k: {"title": PAGES[k], "revid": v["parse"].get("revid"),
                      "fetched_utc": v.get("_fetched_utc", "")} for k, v in raw.items()},
        "house_rating_dates": house_rating_dates, "senate_rating_dates": senate_rating_dates,
        "redistricted_2026": sorted(REDISTRICTED_2026),
        "rating_scale": RATING_NUM,
        "problems": problems,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_CSV.relative_to(ROOT)}: {len(rows)} rows")
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)


if __name__ == "__main__":
    main()
