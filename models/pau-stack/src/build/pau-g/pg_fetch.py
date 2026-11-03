"""pau-g: refetch the raw inputs of the swing-geometry model into pbase/data/raw/pg/ (gitignored).

Sources and terms (checked 2026-11-02):
  1. MEDSL county presidential returns 2000-2016: MEDSL's own GitHub (github.com/MEDSL/county-returns), the same
     data MEDSL releases on Harvard Dataverse under CC0. (The 2000-2024 Dataverse file sits behind a required
     guestbook form, refused without it: not used.)
  2. Wikipedia (MediaWiki parse API; generic User-Agent, no e-mail; 1 request/s; backs off on 429): the
     "By county" result tables of the 2020 and 2024 presidential elections per state, and of the Senate and
     governor elections 2000-2024 (statewide sources for the geometry). Only result tables of past elections.
  3. Census ACS 5-year table-based summary files (www2.census.gov bulk, no key; robots.txt has no rule against
     generic agents): B01003 total population, whose summary level 510 (state-CD-county part) is the
     population-weighted county-to-district relationship: 2021 = 116th Congress, 2022 = 118th, 2024 = 119th.
     Plus county demographics (B03002, B15003) for the demographic map.
Usage: pg_fetch.py [pres|statewide|acs|all]
"""
import json
import sys
import time
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "pg"
WIKI = RAW / "wiki"
UA = "Mozilla/5.0 (compatible; research-script/0.1)"
API = "https://en.wikipedia.org/w/api.php"
STATES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
          "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
          "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
          "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
          "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
          "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
S = requests.Session()
S.headers["User-Agent"] = UA
_last = [0.0]


def api(**params):
    params = dict(params, format="json", formatversion=2)
    for attempt in range(6):
        wait = 2.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        r = S.get(API, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("Retry-After", 10 * (attempt + 1))))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("API kept refusing")


def county_section(title, out):
    """Save the HTML of the page's 'By county' result section(s) to out (json: title, revid, html list)."""
    if out.exists():
        return json.loads(out.read_text()).get("n", 0)
    d = api(action="parse", page=title, prop="sections|revid", redirects=1)
    if "error" in d:
        out.write_text(json.dumps(dict(title=title, n=0, error=d["error"].get("code"))))
        return 0
    secs = d["parse"]["sections"]
    keys = ("by county", "county results", "results by county", "by county or county-equivalent", "by parish",
            "by borough", "by county and independent city", "by county or independent city")
    pick = [s for s in secs if s["line"].strip().lower() in keys or s["line"].strip().lower().startswith("by county")]
    html = []
    for s in pick:
        t = api(action="parse", page=d["parse"]["title"], prop="text", section=s["index"], redirects=1)
        if "parse" in t:
            html.append(t["parse"]["text"])
    out.write_text(json.dumps(dict(title=d["parse"]["title"], revid=d["parse"].get("revid"), n=len(html),
                                   sections=[s["line"] for s in pick], html=html,
                                   all_sections=[s["line"] for s in secs] if not pick else None)))
    return len(html)


KEYS = ("by county", "county results", "results by county", "by county or county-equivalent", "by parish",
        "by borough", "by county and independent city", "by county or independent city")


def extract_sections(html):
    """From a full rendered page, the HTML of each section whose heading is a 'By county'-type title."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    out, names = [], []
    for h in soup.find_all(["h2", "h3", "h4", "h5"]):
        t = h.get_text(" ", strip=True).lower()
        if not (t in KEYS or t.startswith("by county")):
            continue
        lvl = int(h.name[1])
        node = h.parent if (h.parent is not None and "mw-heading" in " ".join(h.parent.get("class", []))) else h
        parts = []
        for sib in node.next_siblings:
            name = getattr(sib, "name", None)
            if name in ("h2", "h3", "h4", "h5") and int(name[1]) <= lvl:
                break
            if name == "div" and "mw-heading" in " ".join(sib.get("class", [])):
                hh = sib.find(["h2", "h3", "h4", "h5"])
                if hh is not None and int(hh.name[1]) <= lvl:
                    break
            parts.append(str(sib))
        out.append("".join(parts))
        names.append(h.get_text(" ", strip=True))
    return names, out


def county_page(title, out):
    """One request per page: the full rendered page, then the 'By county' section(s) cut out locally."""
    if out.exists():
        return json.loads(out.read_text()).get("n", 0)
    d = api(action="parse", page=title, prop="text|revid", redirects=1)
    if "error" in d:
        out.write_text(json.dumps(dict(title=title, n=0, error=d["error"].get("code"))))
        return 0
    names, html = extract_sections(d["parse"]["text"])
    out.write_text(json.dumps(dict(title=d["parse"]["title"], revid=d["parse"].get("revid"), n=len(html),
                                   sections=names, html=html)))
    return len(html)


def pres():
    (WIKI / "pres").mkdir(parents=True, exist_ok=True)
    for y in (2020, 2024):
        for st in STATES:
            nm = "Washington (state)" if st == "Washington" else st
            n = county_section(f"{y} United States presidential election in {nm}", WIKI / "pres" / f"{y}_{st}.json")
            print(y, st, n, flush=True)


SPECIALS = {2002: ["Minnesota"], 2008: ["Mississippi", "Wyoming"], 2010: ["Delaware", "New York", "West Virginia", "Illinois"],
            2014: ["Hawaii", "Oklahoma", "South Carolina"], 2018: ["Minnesota", "Mississippi"], 2020: ["Arizona", "Georgia"],
            2022: ["Oklahoma", "California"]}
# Senate classes: which states hold a regular Senate election in a year (y % 6)
SEN_CLASS = {0: ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Florida", "Georgia",
                 "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maryland", "Missouri",
                 "Nevada", "New Hampshire", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
                 "Pennsylvania", "South Carolina", "South Dakota", "Utah", "Vermont", "Washington", "Wisconsin"],
             2: ["Arizona", "California", "Connecticut", "Delaware", "Florida", "Hawaii", "Indiana", "Maine", "Maryland",
                 "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
                 "New Jersey", "New Mexico", "New York", "North Dakota", "Ohio", "Pennsylvania", "Rhode Island",
                 "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"],
             4: ["Alabama", "Alaska", "Arkansas", "Colorado", "Delaware", "Georgia", "Idaho", "Illinois", "Iowa", "Kansas",
                 "Kentucky", "Louisiana", "Maine", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Montana",
                 "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "North Carolina", "Oklahoma", "Oregon",
                 "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Virginia", "West Virginia",
                 "Wyoming"]}


def statewide(kinds=("sen",), years=(2022, 2020, 2018, 2016, 2014, 2012, 2010, 2008, 2006, 2004, 2002, 2000)):
    """Senate (regular + known specials) first, newest years first; governors only if asked ('gov')."""
    (WIKI / "sw").mkdir(parents=True, exist_ok=True)
    for y in years:
        todo = []
        if "sen" in kinds:
            todo += [("sen", st, f"{y} United States Senate election in {st}") for st in SEN_CLASS[y % 6]]
            todo += [("sens", st, f"{y} United States Senate special election in {st}") for st in SPECIALS.get(y, [])]
        if "gov" in kinds:
            todo += [("gov", st, f"{y} {st} gubernatorial election") for st in STATES]
        for kind, st, title in todo:
            n = county_page(title, WIKI / "sw" / f"{y}_{kind}_{st}.json")
            print(y, kind, st, n, flush=True)


def acs():
    d = RAW / "acs5"
    d.mkdir(parents=True, exist_ok=True)
    base = "https://www2.census.gov/programs-surveys/acs/summary_file/{y}/table-based-SF/data/5YRData/acsdt5y{y}-{t}.dat"
    for y, t in [(2021, "b01003"), (2022, "b01003"), (2024, "b01003"), (2022, "b03002"), (2022, "b15003")]:
        out = d / f"acsdt5y{y}-{t}.dat"
        if out.exists():
            continue
        r = S.get(base.format(y=y, t=t), timeout=600)
        r.raise_for_status()
        out.write_bytes(r.content)
        print("acs", y, t, len(r.content), flush=True)
        time.sleep(1)


def medsl():
    out = RAW / "countypres_2000-2016.csv"
    if not out.exists():
        r = S.get("https://raw.githubusercontent.com/MEDSL/county-returns/master/countypres_2000-2016.csv", timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    RAW.mkdir(parents=True, exist_ok=True)
    if what in ("medsl", "all"):
        medsl()
    if what in ("acs", "all"):
        acs()
    if what in ("pres", "all"):
        pres()
    if what in ("statewide", "all"):
        statewide()
    if what == "sw_old":
        statewide(years=(2018, 2016, 2014, 2012, 2010, 2008, 2006, 2004, 2002, 2000))
    if what == "gov":
        statewide(kinds=("gov",))
