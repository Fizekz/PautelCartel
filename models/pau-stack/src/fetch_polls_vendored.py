#!/usr/bin/env python3
"""Fetch 2026 race-level and national polls from Wikipedia (MediaWiki API).

Outputs
  data/features/polls_2026.csv              one row per poll (race level, Senate + House)
  data/features/race_poll_aggregates_2026.csv  per-race aggregator averages (270toWin, DDHQ, RCP, ...)
                                               as quoted in the Wikipedia race articles
  data/features/national_2026.csv           Trump approval polls (one row per poll) plus generic-ballot and
                                               approval aggregator snapshots (Wikipedia has no per-poll
                                               generic-ballot list for 2026; the generic-ballot rows are daily
                                               snapshots of the aggregator table, from the article's revision
                                               history, so the same aggregate repeats until it is updated)
  data/features/polls_2026_district_generic.csv  district-level "Generic Democrat vs Generic Republican" polls
  data/raw/polls/wikipedia/*.json           cached API responses (re-used unless --refresh)
  data/raw/polls/polls_2026_all_versions.csv every poll version row before de-duplication

Sources and why
  * Wikipedia per-race articles ("2026 United States Senate election in X", "2026 United States House of
    Representatives elections in X", standalone district articles). CC BY-SA, read through the API with a
    descriptive User-Agent and a delay between calls.
  * FiftyPlusOne: its CSV API is a paid product and its terms forbid automated harvesting of the site, so it is
    NOT scraped here (needs a human decision / an API key passed via the env var FPO_API_KEY, not implemented).
  * RealClearPolling: blocked by a bot-protection captcha (HTTP 403); not circumvented.
  * 270toWin: robots.txt disallows Anthropic agents (anthropic-ai, Claude-Web); not scraped.

Name matching uses the Kalshi snapshot (candidate names per race) only to pick the general-election matchup.

Usage: python3 src/fetch_polls.py [--refresh] [--only-senate] [--days-national 45]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "polls" / "wikipedia"
FEAT = ROOT / "data" / "features"
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"
KALSHI_GLOB = "kalshi_races_*.csv"

API = "https://en.wikipedia.org/w/api.php"
UA = ("ModelCitizensResearch/1.0 (https://github.com/Pausb08/DemocracyDay; Stanford Democracy Day student "
      "election-forecast project) python-requests/" + requests.__version__)
DELAY = 1.5  # seconds between API calls

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
NAME2ABBR = {v: k for k, v in STATES.items()}
SENATE_SPECIAL = {"FL", "OH"}  # 2026 specials; the template ids are plain FL-SEN / OH-SEN

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9

POLL_COLS = ["race_id", "pollster", "start_date", "end_date", "sample_size", "population", "dem_pct", "rep_pct",
             "other_pct", "dem_candidate", "rep_candidate", "other_candidate", "undecided_pct",
             "unnamed_other_pct", "moe", "partisan_sponsor", "sponsor", "version_note", "n_versions",
             "cand_pcts", "wiki_page", "wiki_section", "source_url"]


# --------------------------------------------------------------------------------------------- HTTP / cache
class Wiki:
    def __init__(self, refresh: bool = False):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.refresh = refresh
        self.last = 0.0
        RAW.mkdir(parents=True, exist_ok=True)

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json", "formatversion": 2}
        for attempt in range(6):
            wait = DELAY - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            r = self.s.get(API, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("retry-after")
                sleep = float(ra) if ra and ra.isdigit() else 10 * (attempt + 1)
                print(f"  HTTP {r.status_code}; sleeping {sleep}s", file=sys.stderr)
                time.sleep(sleep)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"giving up on {params}")

    @staticmethod
    def slug(title: str) -> str:
        s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
        return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:150]

    def parse(self, title: str) -> dict | None:
        """Rendered HTML of a page (action=parse), cached as JSON."""
        f = RAW / f"{self.slug(title)}.json"
        if f.exists() and not self.refresh:
            return json.loads(f.read_text())
        d = self._get(dict(action="parse", page=title, prop="text|sections|revid|displaytitle",
                           redirects=1, disableeditsection=1, disabletoc=1))
        if "error" in d:
            print(f"  missing page: {title} ({d['error'].get('code')})", file=sys.stderr)
            return None
        out = {"title": d["parse"]["title"], "revid": d["parse"]["revid"],
               "fetched": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "text": d["parse"]["text"]}
        f.write_text(json.dumps(out))
        return out

    def query(self, **params) -> dict:
        return self._get(dict(action="query", **params))


# --------------------------------------------------------------------------------------------- text helpers
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s.replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def last_name(name: str) -> str:
    toks = [t for t in norm(name).replace("'", "").split() if t not in SUFFIX]
    return toks[-1] if toks else ""


def cell_text(el: Tag) -> str:
    """Visible text of a cell, without footnote markers."""
    el = BeautifulSoup(str(el), "html.parser")
    for sup in el.find_all("sup"):
        sup.decompose()
    for st in el.find_all(["style", "script"]):
        st.decompose()
    for br in el.find_all("br"):
        br.replace_with(" ")
    t = el.get_text(" ", strip=True).replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


def parse_pct(t: str) -> float | None:
    t = t.replace(",", ".").strip()
    if not t or t in {"–", "—", "-", "N/A", "n/a", "?"}:
        return None
    m = re.search(r"(<\s*)?(\d+(?:\.\d+)?)\s*%?", t)
    if not m:
        return None
    v = float(m.group(2))
    if m.group(1):  # "<1%"
        v = v / 2
    return v if 0 <= v <= 100 else None


def parse_sample(t: str):
    t = t.replace("\xa0", " ")
    pop = None
    m = re.search(r"\(?\b(LV|RV|A|V|RVs|LVs)\b\)?", t)
    if m:
        pop = m.group(1).lower().rstrip("s")
    n = re.search(r"(\d[\d,]*)", t)
    size = int(n.group(1).replace(",", "")) if n else None
    if size is not None and size < 50:  # e.g. footnote numbers
        size = None
    return size, pop


def _md(part: str):
    """'October 16' -> (10, 16, None); '16' -> (None, 16, None); 'October 16, 2026' -> (10,16,2026)."""
    part = part.strip().strip(",").strip()
    y = re.search(r"(\d{4})\s*$", part)
    year = int(y.group(1)) if y else None
    if y:
        part = part[: y.start()].strip().strip(",")
    mon = None
    day = None
    for tok in re.findall(r"[A-Za-z]+\.?|\d+", part):
        tl = tok.lower().rstrip(".")
        if tl in MONTHS:
            mon = MONTHS[tl]
        elif tok.isdigit():
            day = int(tok)
    return mon, day, year


def parse_dates(t: str, default_year: int | None = None):
    """Return (start, end) ISO dates from strings such as 'September 30 – October 2, 2026'."""
    t = t.replace("\xa0", " ")
    t = re.sub(r"\s*(?:–|—|−|‒|-)\s*|\s+to\s+|\s+through\s+", " – ", t)
    parts = [p for p in t.split(" – ") if p.strip()]
    if not parts:
        return None, None
    if len(parts) > 2:
        parts = [parts[0], parts[-1]]
    em, ed, ey = _md(parts[-1])
    sm, sd, sy = _md(parts[0]) if len(parts) == 2 else (em, ed, ey)
    ey = ey or default_year
    if sm is None:
        sm = em
    if em is None:
        em = sm
    if sy is None:
        sy = ey
        if sm and em and sm > em and sy:
            sy = ey - 1
    if ed is None and em:  # "October 2026"
        ed = (dt.date(ey + (em == 12), em % 12 + 1, 1) - dt.timedelta(days=1)).day if ey else None
        sd = sd or 1
    if sd is None:
        sd = ed

    def mk(y, m, d):
        try:
            return dt.date(y, m, d).isoformat()
        except Exception:
            return None

    return mk(sy, sm, sd), mk(ey, em, ed)


PARTY_RE = re.compile(r"\(\s*(D|R|I|L|G|DFL|C|WF|WFP|IND|Ind\.?|Lib\.?|Grn\.?|U|NPA|NP|FWD|F|AIP|PF|LBT|"
                      r"Libertarian|Independent|Democratic|Republican|Green|Forward|Constitution)\s*\)\s*$", re.I)


def header_party(h: str):
    m = PARTY_RE.search(h)
    if not m:
        return h.strip(), None
    p = m.group(1).upper().rstrip(".")
    p = {"DFL": "D", "DEMOCRATIC": "D", "REPUBLICAN": "R", "IND": "I", "INDEPENDENT": "I", "LIB": "L",
         "LIBERTARIAN": "L", "LBT": "L", "GRN": "G", "GREEN": "G"}.get(p, p)
    return h[: m.start()].strip(), p


# --------------------------------------------------------------------------------------------- tables
def table_grid(tbl: Tag):
    """Expand row/colspans. Returns list of rows; each row a list of (tag, is_th)."""
    grid: list[list] = []
    pending: dict[int, list] = {}  # col -> [remaining_rows, cell]
    rows = [tr for tr in tbl.find_all("tr") if tr.find_parent("table") is tbl]
    for tr in rows:
        cells = [c for c in tr.find_all(["td", "th"], recursive=False)]
        row: list = []
        col = 0

        def fill_pending():
            nonlocal col
            while col in pending and pending[col][0] > 0:
                row.append(pending[col][1])
                pending[col][0] -= 1
                col += 1

        for c in cells:
            fill_pending()
            try:
                cs = int(re.sub(r"\D", "", c.get("colspan", "1")) or 1)
            except ValueError:
                cs = 1
            try:
                rs = int(re.sub(r"\D", "", c.get("rowspan", "1")) or 1)
            except ValueError:
                rs = 1
            cs = min(cs, 50)
            rs = min(rs, 500)
            item = (c, c.name == "th")
            for k in range(cs):
                row.append(item)
                if rs > 1:
                    pending[col] = [rs - 1, item]
                col += 1
        fill_pending()
        live = [k for k, v in pending.items() if v[0] > 0 and k > col]
        while live and col <= max(live):
            if col in pending and pending[col][0] > 0:
                fill_pending()
            else:
                row.append(None)
                col += 1
        grid.append(row)
    return grid


def classify_header(h: str) -> str:
    hl = h.lower()
    if re.search(r"source of poll|aggregat", hl):
        return "aggregator"
    if re.search(r"poll source|^source$|pollster", hl):
        return "pollster"
    if re.search(r"dates? updated", hl):
        return "updated"
    if re.search(r"date", hl):
        return "dates"
    if re.search(r"sample", hl):
        return "sample"
    if re.search(r"margin of error|^moe$|error", hl):
        return "moe"
    if re.search(r"^(other|others)\b|other/|^other$", hl):
        return "other"
    if re.search(r"undecided|unsure|don.?t know|not sure|no opinion|would not vote", hl):
        return "undecided"
    if re.search(r"^lead$|^margin$|^net$", hl):
        return "lead"
    if re.search(r"^approve|^disapprove", hl):
        return hl.split()[0]
    if hl in {"", "#", "no."} or re.search(r"\bround\b|^rcv|^type$|^question|^scenario|^version", hl):
        return "skip"
    return "candidate"


def parse_poll_table(tbl: Tag):
    """Return (kind, columns, data_rows). columns: list of (role, header_text). rows: list of cell lists."""
    grid = table_grid(tbl)
    if not grid:
        return None
    nhead = 0
    for row in grid:
        if row and all(c is None or c[1] for c in row):
            nhead += 1
        else:
            break
    if nhead == 0 or nhead >= len(grid):
        return None
    ncol = max(len(r) for r in grid)
    headers = []
    for j in range(ncol):
        parts = []
        for r in grid[:nhead]:
            if j < len(r) and r[j] is not None:
                t = cell_text(r[j][0])
                if t and t not in parts:
                    parts.append(t)
        headers.append(" ".join(parts))
    roles = [classify_header(h) for h in headers]
    kind = "aggregate" if "aggregator" in roles else ("poll" if "pollster" in roles else None)
    if kind is None:
        return None
    data = []
    for r in grid[nhead:]:
        cells = [r[j][0] if j < len(r) and r[j] is not None else None for j in range(ncol)]
        # skip separator / note rows (a single cell spanning the table)
        distinct = {id(c) for c in cells if c is not None}
        if len(distinct) <= 2:
            continue
        data.append(cells)
    return kind, list(zip(roles, headers)), data


# --------------------------------------------------------------------------------------------- page walk
EXCLUDE_SECTION = re.compile(r"primar|runoff|convention|caucus|nominat|jungle|blanket|top.two primary|"
                             r"special election.*primary|first round", re.I)


def ref_map(soup: BeautifulSoup) -> dict:
    out = {}
    for li in soup.select('li[id^="cite_note"]'):
        span = li.select_one("span.reference-text") or li
        url = None
        a = span.select_one("a.external[href]")
        if a:
            url = a["href"]
        out[li["id"]] = (re.sub(r"\s+", " ", span.get_text(" ", strip=True)), url)
    return out


def cell_refs(cell: Tag, refs: dict):
    out = []
    for a in cell.select('sup.reference a[href^="#cite_note"]'):
        out.append(refs.get(a["href"][1:], ("", None)))
    return out


def walk_tables(html: str):
    """Yield (heading_path, table_tag, label, refs) for every wikitable in document order."""
    soup = BeautifulSoup(html, "html.parser")
    refs = ref_map(soup)
    root = soup.select_one("div.mw-parser-output") or soup
    path: dict[int, str] = {}
    for el in root.find_all(["h2", "h3", "h4", "h5", "h6", "table"]):
        if el.name.startswith("h"):
            lvl = int(el.name[1])
            path = {k: v for k, v in path.items() if k < lvl}
            path[lvl] = el.get_text(" ", strip=True)
            continue
        if "wikitable" not in (el.get("class") or []):
            continue
        if el.find_parent("table") is not None:
            continue
        label = ""
        coll = el.find_parent(class_=re.compile("mw-collapsible|NavFrame|toccolours"))
        if coll is not None:
            ttl = coll.find(class_=re.compile("NavHead|mw-collapsible-toggle|title"))
            label = ttl.get_text(" ", strip=True) if ttl else ""
        cap = el.find("caption")
        if cap:
            label = (label + " " + cap.get_text(" ", strip=True)).strip()
        yield [path[k] for k in sorted(path)], el, label, refs


# --------------------------------------------------------------------------------------------- kalshi names
def load_kalshi() -> dict:
    files = sorted((ROOT / "data" / "benchmark").glob(KALSHI_GLOB))
    k = pd.read_csv(files[-1])
    out = {}
    for _, r in k.iterrows():
        cands = []
        for part in str(r["candidates"]).split("|"):
            bits = part.strip().split(":")
            if len(bits) < 3:
                continue
            party, name, price = bits[0], ":".join(bits[1:-1]), bits[-1]
            try:
                price = float(price)
            except ValueError:
                price = float("nan")
            placeholder = bool(re.fullmatch(r"(Democratic|Republican|Independent) party", name.strip(), re.I))
            cands.append(dict(party=party, name=name.strip(), price=price, placeholder=placeholder))
        cands.sort(key=lambda c: -(c["price"] if c["price"] == c["price"] else 0))
        out[r["race_id"]] = cands
    return out


PARTY_K2W = {"DEM": "D", "REP": "R", "IND": "I", "LIB": "L", "GRN": "G"}


def match_columns(cols, kcands):
    """Map Kalshi candidates to candidate column indexes. Returns {kalshi_idx: col_idx}."""
    cand_cols = [(j, *header_party(h)) for j, (role, h) in enumerate(cols) if role == "candidate"]
    m = {}
    used = set()
    for ki, kc in enumerate(kcands):
        if kc["placeholder"]:
            continue
        ln = last_name(kc["name"])
        full = norm(kc["name"])
        best = None
        for j, nm, p in cand_cols:
            if j in used:
                continue
            nn = norm(nm)
            toks = nn.replace("'", "").split()
            if ln and (ln in toks or nn == full or ln.replace(" ", "") == nn.replace(" ", "")):
                best = j
                break
            # spelling variants ("Okapareke" vs "Okpareke"): close last name AND same first initial
            hl = last_name(nm)
            if (ln and hl and len(ln) >= 5 and SequenceMatcher(None, ln, hl).ratio() >= 0.85
                    and toks and full and toks[0][0] == full[0]):
                best = j
                break
            # hyphenated surnames: "hyde smith"
            if ln and len(full.split()) >= 2 and " ".join(full.split()[-2:]) in nn:
                best = j
                break
        if best is not None:
            m[ki] = best
            used.add(best)
    # placeholders: match by party letter if exactly one unused column of that party
    for ki, kc in enumerate(kcands):
        if not kc["placeholder"] or ki in m:
            continue
        p = PARTY_K2W.get(kc["party"])
        options = [j for j, nm, pp in cand_cols if pp == p and j not in used]
        if options:
            m[ki] = options[0]
            used.add(options[0])
    return m


# --------------------------------------------------------------------------------------------- race polls
def race_from_house_path(state: str, path: list[str], at_large: bool):
    if at_large:
        return f"{state}-AL"
    for h in path:
        m = re.match(r"District\s+(\d+)", h, re.I)
        if m:
            return f"{state}-{int(m.group(1)):02d}"
        if re.match(r"At[- ]large", h, re.I):
            return f"{state}-AL"
    return None


def extract_race_polls(page: dict, race_for_path, kalshi: dict, rows: list, aggs: list, generic: list | None = None):
    title = page["title"]
    url_page = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    resolved: dict[str, dict] = {}  # race -> {kalshi_idx: column-name} for placeholders
    for path, tbl, label, refs in walk_tables(page["text"]):
        if any(EXCLUDE_SECTION.search(h) for h in path):
            continue
        race = race_for_path(path)
        if race is None or race not in kalshi:
            continue
        parsed = parse_poll_table(tbl)
        if not parsed:
            continue
        kind, cols, data = parsed
        kc = kalshi[race]
        cand_heads = [h for r, h in cols if r == "candidate"]
        if len([h for h in cand_heads if not re.match(r"generic|other|someone", norm(h))]) >= 5:
            continue  # 5+ named candidates: a top-two / jungle primary poll filed under "General election"
        if kind == "poll" and generic is not None and cand_heads and all(
                re.match(r"generic (democrat|republican)", norm(h)) for h in cand_heads):
            _generic_rows(race, cols, data, refs, url_page, " > ".join(path), generic)
            continue
        mapping = match_columns(cols, kc)
        top2 = [i for i in range(min(2, len(kc)))]
        if not all(i in mapping for i in top2):
            continue
        # placeholder consistency: once resolved to a column header name, require the same name later
        ok = True
        for ki in top2:
            if kc[ki]["placeholder"]:
                nm = last_name(header_party(cols[mapping[ki]][1])[0])
                prev = resolved.setdefault(race, {}).setdefault(ki, nm)
                if prev != nm:
                    ok = False
        if not ok:
            continue
        sect = " > ".join(path + ([label] if label else []))
        cand_idx = [j for j, (r, h) in enumerate(cols) if r == "candidate"]
        k_of_col = {j: ki for ki, j in mapping.items()}
        role_idx = {r: j for j, (r, h) in reversed(list(enumerate(cols)))}

        for cells in data:
            def get(role):
                j = role_idx.get(role)
                return cells[j] if j is not None and j < len(cells) else None

            src_cell = get("aggregator" if kind == "aggregate" else "pollster")
            if src_cell is None:
                continue
            pollster_raw = cell_text(src_cell)
            if not pollster_raw or re.match(r"^(average|aggregate)", pollster_raw, re.I) and kind == "poll":
                continue
            # candidate values
            vals = {}
            for j in cand_idx:
                if cells[j] is None:
                    continue
                vals[j] = parse_pct(cell_text(cells[j]))
            if not any(v is not None for v in vals.values()):
                continue
            dem = rep = None
            dem_c = rep_c = None
            others = []
            same_party = []
            # matched columns first, in Kalshi order (favourite first), so dem/rep = that party's leading nominee
            order = sorted(cand_idx, key=lambda j: (k_of_col.get(j, 999), j))
            for j in order:
                v = vals.get(j)
                nm, hp = header_party(cols[j][1])
                if j in k_of_col:
                    kk = kc[k_of_col[j]]
                    party = PARTY_K2W.get(kk["party"], hp)
                    cname = kk["name"] if not kk["placeholder"] else nm
                else:
                    party, cname = hp, nm
                if party == "D" and dem_c is None and j in k_of_col:
                    dem, dem_c = v, cname
                elif party == "R" and rep_c is None and j in k_of_col:
                    rep, rep_c = v, cname
                else:
                    if v is not None and v > 0:
                        others.append((cname, v))
                    if party in ("D", "R") and j in k_of_col:
                        same_party.append(cname)
            refs_src = cell_refs(src_cell, refs)
            src_url = None
            a = src_cell.select_one("a.external[href]")
            if a:
                src_url = a["href"]
            for txt, u in refs_src:
                if u and not src_url:
                    src_url = u
            sponsor = "; ".join(t for t, u in refs_src if re.search(r"(sponsor|conducted for|commissioned)", t, re.I)
                                and not u)
            notes = []
            for j in cand_idx:
                if cells[j] is not None:
                    for t, u in cell_refs(cells[j], refs):
                        if t and not u and len(t) < 120:
                            notes.append(t)
            tags = sorted(set(re.findall(r"\((D|R)\)", pollster_raw)))
            partisan = "+".join(tags) if tags else None  # "D+R" = bipartisan pollster pair
            pollster = re.sub(r"\s*\((D|R|I)\)", "", pollster_raw).strip()

            if kind == "aggregate":
                d0, d1 = parse_dates(cell_text(get("dates")) if get("dates") is not None else "")
                upd = parse_dates(cell_text(get("updated")))[1] if get("updated") is not None else None
                aggs.append(dict(race_id=race, aggregator=pollster, start_date=d0, end_date=d1, updated=upd,
                                 dem_candidate=dem_c, dem_pct=dem, rep_candidate=rep_c, rep_pct=rep,
                                 other_candidate="; ".join(n for n, v in others), other_pct=sum(v for n, v in others)
                                 if others else None, wiki_page=url_page, wiki_section=sect, source_url=src_url))
                continue
            dates_t = cell_text(get("dates")) if get("dates") is not None else ""
            d0, d1 = parse_dates(dates_t)
            size, pop = parse_sample(cell_text(get("sample"))) if get("sample") is not None else (None, None)
            moe = parse_pct(cell_text(get("moe"))) if get("moe") is not None else None
            und = parse_pct(cell_text(get("undecided"))) if get("undecided") is not None else None
            oth = parse_pct(cell_text(get("other"))) if get("other") is not None else None
            named_other = [(n, v) for n, v in others]
            rows.append(dict(
                race_id=race, pollster=pollster, start_date=d0, end_date=d1, sample_size=size, population=pop,
                dem_pct=dem, rep_pct=rep,
                other_pct=round(sum(v for n, v in named_other), 2) if named_other else None,
                dem_candidate=dem_c, rep_candidate=rep_c,
                other_candidate="; ".join(n for n, v in named_other) or None,
                undecided_pct=und, unnamed_other_pct=oth, moe=moe,
                partisan_sponsor=partisan, sponsor=sponsor or None,
                version_note="; ".join(dict.fromkeys(notes)) or None,
                cand_pcts="; ".join(f"{header_party(cols[j][1])[0]}"
                                    f"{' (' + header_party(cols[j][1])[1] + ')' if header_party(cols[j][1])[1] else ''}"
                                    f"={vals[j]:g}" for j in cand_idx if vals.get(j) is not None),
                same_party=";".join(same_party) or None,
                dates_raw=dates_t, wiki_page=url_page, wiki_section=sect, source_url=src_url,
                _tbl=id(tbl), _rowkey=(pollster_raw, dates_t)))


def _generic_rows(race, cols, data, refs, url_page, sect, out):
    """District-level generic-ballot questions ("Generic Democrat" vs "Generic Republican")."""
    roles = {r: j for j, (r, h) in reversed(list(enumerate(cols)))}
    for cells in data:
        src = cells[roles["pollster"]] if "pollster" in roles else None
        if src is None:
            continue
        name = cell_text(src)
        vals = {}
        for j, (r, h) in enumerate(cols):
            if r == "candidate" and cells[j] is not None:
                vals["dem_pct" if "democrat" in h.lower() else "rep_pct"] = parse_pct(cell_text(cells[j]))
        d0, d1 = parse_dates(cell_text(cells[roles["dates"]])) if "dates" in roles else (None, None)
        size, pop = parse_sample(cell_text(cells[roles["sample"]])) if "sample" in roles else (None, None)
        url = None
        for t, u in cell_refs(src, refs):
            url = url or u
        tags = sorted(set(re.findall(r"\((D|R)\)", name)))
        out.append(dict(race_id=race, pollster=re.sub(r"\s*\((D|R|I)\)", "", name).strip(), start_date=d0,
                        end_date=d1, sample_size=size, population=pop, dem_pct=vals.get("dem_pct"),
                        rep_pct=vals.get("rep_pct"), partisan_sponsor="+".join(tags) or None,
                        wiki_page=url_page, wiki_section=sect, source_url=url))


POP_RANK = {"lv": 0, "rv": 1, "v": 2, "a": 3, None: 4}


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per poll: same race + pollster + end date. Prefer LV>RV>A, then first table, then first row."""
    if df.empty:
        return df
    df = df.copy()
    df["_order"] = range(len(df))
    df["_pop"] = df["population"].map(lambda p: POP_RANK.get(p, 4))
    df["_pkey"] = df["pollster"].map(lambda s: norm(s))
    key = ["race_id", "_pkey", "end_date"]
    df["n_versions"] = df.groupby(key, dropna=False)["_order"].transform("size")
    df = df.sort_values(["_pop", "_order"]).drop_duplicates(key, keep="first").sort_values("_order")
    return df


# --------------------------------------------------------------------------------------------- page lists
def senate_pages():
    out = []
    for st, name in STATES.items():
        if st in SENATE_SPECIAL:
            out.append((f"{st}-SEN", f"2026 United States Senate special election in {name}"))
        else:
            out.append((f"{st}-SEN", f"2026 United States Senate election in {name}"))
    return out


def house_pages(wiki: Wiki):
    titles = []
    d = wiki.query(list="allpages", apprefix="2026 United States House of Representatives election",
                   aplimit=500, apfilterredir="nonredirects")
    for p in d["query"]["allpages"]:
        t = p["title"]
        m = re.match(r"2026 United States House of Representatives elections? in (?:the )?(.+?)(?: \(districts.*\))?$", t)
        if not m:
            continue
        st = NAME2ABBR.get(m.group(1))
        if not st:
            continue
        if t.endswith("in California"):  # split into two district pages
            continue
        titles.append((st, t, " election in " in t))
    # standalone district articles, e.g. "2026 New York's 13th congressional district election"
    d = wiki.query(list="search", srsearch='intitle:"congressional district election" intitle:2026',
                   srlimit=200, srnamespace=0)
    extra = []
    for p in d["query"]["search"]:
        t = p["title"]
        m = re.match(r"2026 (.+?)'s (\d+)(?:st|nd|rd|th) congressional district election$", t)
        if m and m.group(1) in NAME2ABBR:
            extra.append((f"{NAME2ABBR[m.group(1)]}-{int(m.group(2)):02d}", t))
        m = re.match(r"2026 (.+?)'s at-large congressional district election$", t)
        if m and m.group(1) in NAME2ABBR:
            extra.append((f"{NAME2ABBR[m.group(1)]}-AL", t))
    return titles, extra


def main_article_links(page: dict) -> list[tuple[str, str]]:
    """'Main article: 2026 X's Nth congressional district election' hatnotes inside a state page."""
    soup = BeautifulSoup(page["text"], "html.parser")
    out = []
    for a in soup.select("div.hatnote a[title]"):
        t = a["title"]
        m = re.match(r"2026 (.+?)'s (\d+)(?:st|nd|rd|th) congressional district election$", t)
        if m and m.group(1) in NAME2ABBR:
            out.append((f"{NAME2ABBR[m.group(1)]}-{int(m.group(2)):02d}", t))
    return out


# --------------------------------------------------------------------------------------------- national
def parse_wikitext_table(wt: str):
    rows, cur = [], []
    for line in wt.splitlines():
        line = line.strip()
        if line.startswith("|-") or line.startswith("|}"):
            if cur:
                rows.append(cur)
            cur = []
            continue
        if line.startswith("!") or line.startswith("{|"):
            continue
        if line.startswith("|"):
            cells = re.split(r"\|\|", line[1:])
            cur.extend(cells)
    if cur:
        rows.append(cur)
    return rows


def wt_clean(s: str) -> str:
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"\{\{[Ee]fn[^{}]*\}\}", "", s)
    s = re.sub(r"\{\{[Pp]arty shading/[^}]*\}\}\s*\|", "", s)
    s = re.sub(r"^\s*(style|colspan|rowspan)=[^|]*\|", "", s)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = s.replace("'''", "").replace("''", "")
    return s.strip()


def generic_ballot_snapshots(wiki: Wiki, days: int):
    """Daily snapshots of the 'Generic congressional ballot aggregate polls' table (House elections article)."""
    title = "2026 United States House of Representatives elections"
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    revs = []
    cont = {}
    while True:
        d = wiki.query(titles=title, prop="revisions", rvprop="ids|timestamp", rvlimit=500,
                       rvstart=end.strftime("%Y-%m-%dT%H:%M:%SZ"), rvend=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       **cont)
        revs += d["query"]["pages"][0].get("revisions", [])
        if "continue" not in d:
            break
        cont = d["continue"]
    by_day = {}
    for r in revs:  # newest first; keep last revision of each day
        day = r["timestamp"][:10]
        by_day.setdefault(day, r["revid"])
    out = []
    ids = sorted(by_day.items())
    for i in range(0, len(ids), 10):
        chunk = ids[i:i + 10]
        cache = RAW / f"genericballot_revs_{chunk[0][1]}_{chunk[-1][1]}.json"
        if cache.exists() and not wiki.refresh:
            d = json.loads(cache.read_text())
        else:
            d = wiki.query(revids="|".join(str(r) for _, r in chunk), prop="revisions",
                           rvprop="ids|timestamp|content", rvslots="main")
            cache.write_text(json.dumps(d))
        for p in d["query"]["pages"]:
            for rv in p.get("revisions", []):
                c = rv["slots"]["main"]["content"]
                m = re.search(r'<section begin="?GenericBallotAgg"?\s*/>(.*?)<section end', c, re.S)
                if not m:
                    continue
                hdr = re.findall(r"^!(.*)$", m.group(1), re.M)
                hdr = [wt_clean(re.sub(r"^.*\|", "", h)) for h in hdr]
                rcol = next((i for i, h in enumerate(hdr) if h.lower().startswith("republican")), 3)
                dcol = next((i for i, h in enumerate(hdr) if h.lower().startswith("democrat")), 4)
                for cells in parse_wikitext_table(m.group(1)):
                    cl = [wt_clean(x) for x in cells]
                    if len(cl) < 5:
                        continue
                    src = cl[0]
                    if src.lower().startswith("average"):
                        # 'colspan=2 | Average' shifts columns by one
                        cl = [src, "", *cl[1:]]
                    url = re.search(r"url=\s*([^|\s}]+)", cells[0])
                    d0, d1 = parse_dates(cl[1])
                    out.append(dict(
                        metric="generic_ballot", row_type="aggregate", pollster=src, start_date=d0, end_date=d1,
                        as_of=parse_dates(cl[2])[1] or rv["timestamp"][:10], wiki_revision_date=rv["timestamp"][:10],
                        dem_pct=parse_pct(cl[dcol]) if dcol < len(cl) else None,
                        rep_pct=parse_pct(cl[rcol]) if rcol < len(cl) else None,
                        other_pct=parse_pct(cl[5]) if len(cl) > 5 else None,
                        source_url=url.group(1) if url else None,
                        wiki_page=f"https://en.wikipedia.org/w/index.php?oldid={rv['revid']}"))
    return out


def approval_polls(wiki: Wiki):
    page = wiki.parse("Opinion polling on the second Trump presidency")
    out = []
    url_page = "https://en.wikipedia.org/wiki/Opinion_polling_on_the_second_Trump_presidency"
    for path, tbl, label, refs in walk_tables(page["text"]):
        if not path or not re.search(r"nationwide job approval", path[0], re.I):
            continue
        if any(re.search(r"transition|president-elect", h, re.I) for h in path):
            continue
        parsed = parse_poll_table(tbl)
        if not parsed:
            continue
        kind, cols, data = parsed
        roles = [r for r, h in cols]
        hdr = [h.lower() for r, h in cols]
        if "approve" not in roles or "disapprove" not in roles:
            continue
        year = None
        for h in path:
            m = re.search(r"(20\d\d)", h)
            if m:
                year = int(m.group(1))
        ia, idis = roles.index("approve"), roles.index("disapprove")
        for cells in data:
            src_cell = cells[0]
            if src_cell is None:
                continue
            name = cell_text(src_cell)
            if not name:
                continue
            a = src_cell.select_one("a.external[href]")
            src_url = a["href"] if a else None
            if not src_url:
                for t, u in cell_refs(src_cell, refs):
                    src_url = src_url or u

            def g(role):
                return cells[roles.index(role)] if role in roles and cells[roles.index(role)] is not None else None

            ap = parse_pct(cell_text(cells[ia])) if cells[ia] is not None else None
            dis = parse_pct(cell_text(cells[idis])) if cells[idis] is not None else None
            if ap is None or dis is None:
                continue
            if kind == "aggregate":
                d0, d1 = parse_dates(cell_text(g("dates")) if g("dates") is not None else "", year)
                upd = parse_dates(cell_text(g("updated")), year)[1] if g("updated") is not None else None
                out.append(dict(metric="trump_approval", row_type="aggregate", pollster=name, start_date=d0,
                                end_date=d1, as_of=upd, approve=ap, disapprove=dis, source_url=src_url,
                                wiki_page=url_page, wiki_section=" > ".join(path)))
                continue
            dtxt = cell_text(g("dates")) if g("dates") is not None else ""
            d0, d1 = parse_dates(dtxt, year)
            size, pop = parse_sample(cell_text(g("sample"))) if g("sample") is not None else (None, None)
            ps = re.search(r"\((D|R)\)\s*$", name)
            out.append(dict(metric="trump_approval", row_type="poll",
                            pollster=re.sub(r"\s*\((D|R)\)\s*$", "", name), start_date=d0, end_date=d1,
                            sample_size=size, population=pop, approve=ap, disapprove=dis,
                            partisan_sponsor=ps.group(1) if ps else None, source_url=src_url,
                            wiki_page=url_page, wiki_section=" > ".join(path)))
    return out


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download cached pages")
    ap.add_argument("--only-senate", action="store_true")
    ap.add_argument("--days-national", type=int, default=45)
    args = ap.parse_args()

    wiki = Wiki(refresh=args.refresh)
    kalshi = load_kalshi()
    template = pd.read_csv(TEMPLATE)
    valid = set(template["race_id"])
    rows, aggs, log, generic = [], [], [], []

    # ---- Senate
    for race, title in senate_pages():
        if race not in valid:
            continue
        page = wiki.parse(title)
        if page is None:
            log.append((race, title, "missing"))
            continue
        n0 = len(rows)
        extract_race_polls(page, lambda path, r=race: r, kalshi, rows, aggs, generic)
        log.append((race, title, len(rows) - n0))
        print(f"{race}: {len(rows) - n0} poll rows  [{title}]")

    # ---- House
    if not args.only_senate:
        titles, extra = house_pages(wiki)
        done_extra = set()
        for st, title, at_large in titles:
            page = wiki.parse(title)
            if page is None:
                continue
            n0 = len(rows)
            extract_race_polls(page, lambda path, s=st, al=at_large: race_from_house_path(s, path, al),
                               kalshi, rows, aggs, generic)
            print(f"{st} House: {len(rows) - n0} poll rows  [{title}]")
            for race, t in main_article_links(page):
                extra.append((race, t))
        for race, t in extra:
            if t in done_extra or race not in valid:
                continue
            done_extra.add(t)
            page = wiki.parse(t)
            if page is None:
                continue
            n0 = len(rows)
            extract_race_polls(page, lambda path, r=race: r, kalshi, rows, aggs, generic)
            print(f"{race}: {len(rows) - n0} poll rows  [{t}]")

    allv = pd.DataFrame(rows)
    (ROOT / "data" / "raw" / "polls").mkdir(parents=True, exist_ok=True)
    allv.drop(columns=["_tbl", "_rowkey"], errors="ignore").to_csv(
        ROOT / "data" / "raw" / "polls" / "polls_2026_all_versions.csv", index=False)
    polls = dedupe(allv)
    polls["source"] = "wikipedia"
    cols = POLL_COLS + ["same_party", "dates_raw", "source"]
    polls = polls[[c for c in cols if c in polls.columns]].sort_values(["race_id", "end_date"],
                                                                       ascending=[True, False])
    FEAT.mkdir(parents=True, exist_ok=True)
    polls.to_csv(FEAT / "polls_2026.csv", index=False)
    pd.DataFrame(aggs).drop_duplicates().to_csv(FEAT / "race_poll_aggregates_2026.csv", index=False)
    g = pd.DataFrame(generic)
    if not g.empty:
        g = g.drop_duplicates(["race_id", "pollster", "end_date"]).sort_values(["race_id", "end_date"])
    g.to_csv(FEAT / "polls_2026_district_generic.csv", index=False)
    print(f"\npolls_2026.csv: {len(polls)} polls, {polls.race_id.nunique()} races "
          f"({len(allv)} version rows before de-dup)")

    # ---- National
    nat = approval_polls(wiki) + generic_ballot_snapshots(wiki, args.days_national)
    nat = pd.DataFrame(nat)
    ncols = ["metric", "row_type", "pollster", "start_date", "end_date", "as_of", "sample_size", "population",
             "dem_pct", "rep_pct", "other_pct", "approve", "disapprove", "partisan_sponsor", "source_url",
             "wiki_revision_date", "wiki_page", "wiki_section"]
    nat = nat[[c for c in ncols if c in nat.columns]]
    nat.to_csv(FEAT / "national_2026.csv", index=False)
    print(f"national_2026.csv: {len(nat)} rows")
    print(nat.groupby(["metric", "row_type"]).size())


if __name__ == "__main__":
    main()
