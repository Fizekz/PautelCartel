"""Build the POLLS input datasets for the 2026 forecast (and the historical backtest versions).

Outputs (data/clean/):
  polls_2026.csv     race-level 2026 Senate + House general-election polls (Wikipedia race pages + VoteHub API)
  national_2026.csv  2026-cycle generic ballot and Trump job-approval polls (VoteHub API + Wikipedia approval tables)
  polls_hist.csv     race-level Senate + House polls 2010-2024 (FiveThirtyEight archives)
  national_hist.csv  per election year 2010-2024: final generic-ballot average, presidential approval near election
                     day, actual national House popular-vote margin (MIT Election Lab)

Raw downloads are cached under data/raw/polls/ ; rerun with --offline to rebuild only from those files.
Usage:  .venv/bin/python src/data/fetch_polls.py [--offline] [--refresh-wiki]
Only stdlib + pandas + requests. Downloaded files are parsed as data, never executed.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiki_fetch import cache_path, fetch_page  # noqa: E402  (shared polite Wikipedia fetcher with on-disk cache)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "polls"
RAW_WIKI, RAW_538, RAW_VH, RAW_UCSB = RAW / "wiki", RAW / "538", RAW / "votehub", RAW / "ucsb"
CLEAN = ROOT / "data" / "clean"
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"
KALSHI_RACES = ROOT / "data" / "kalshi" / "races_2026-11-01T074812Z.csv"
MIT_HOUSE = ROOT / "data" / "raw" / "mit" / "1976-2024-house.tab"

UA = {"User-Agent": "DemocracyDayForecastResearch/0.1 (Stanford Model Citizens student contest; python-requests)"}
WIKI_URL = "https://en.wikipedia.org/wiki/"
VOTEHUB_URL = "https://api.votehub.com/polls"
WAYBACK = "http://web.archive.org/web/{ts}id_/https://projects.fivethirtyeight.com/{path}"
GH538 = "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
UCSB_OBAMA = "https://www.presidency.ucsb.edu/statistics/data/barack-obama-public-approval"

# 538 files: local name -> (url). The projects.fivethirtyeight.com host is dead (redirects to ABC News), so the
# poll CSVs come from Wayback Machine snapshots taken just before the March 2025 shutdown.
SOURCES_538 = {
    "senate_polls_historical.csv": WAYBACK.format(ts="20250305204704", path="polls/data/senate_polls_historical.csv"),
    "house_polls_historical.csv": WAYBACK.format(ts="20250305204635", path="polls/data/house_polls_historical.csv"),
    "generic_ballot_polls_historical.csv": WAYBACK.format(ts="20250305204633",
                                                          path="polls/data/generic_ballot_polls_historical.csv"),
    "president_approval_polls_historical.csv": WAYBACK.format(
        ts="20250305204636", path="polls/data/president_approval_polls_historical.csv"),
    "generic_ballot_averages.csv": WAYBACK.format(ts="20241203003203", path="polls/data/generic_ballot_averages.csv"),
    "trump1_approval_topline.csv": WAYBACK.format(ts="20241208195423", path="trump-approval-data/approval_topline.csv"),
    "biden_approval_topline.csv": WAYBACK.format(ts="20241102194415", path="biden-approval-data/approval_topline.csv"),
    "generic_topline_historical.csv": GH538 + "congress-generic-ballot/generic_topline_historical.csv",
    "raw_polls.csv": GH538 + "pollster-ratings/raw_polls.csv",
}

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
ABBR_TO_STATE = {v: k for k, v in STATES.items()}
SPECIAL_SENATE_2026 = {"FL", "OH"}
POLL_COLS = ["race_id", "chamber", "pollster", "start_date", "end_date", "sample_size", "population",
             "partisan_sponsor", "dem_cand", "dem_pct", "rep_cand", "rep_pct", "other_cand", "other_pct",
             "source_url"]
EXTRA_2026 = ["undecided_pct", "top_other_cand", "top_other_pct", "cand_detail", "rcv_round", "is_hypothetical",
              "nominee_match", "sponsor_note", "source", "source_page"]
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
NON_CANDIDATE = re.compile(r"^(other|others|undecided|margin|lead|none|someone else|would not vote|refused|"
                           r"don't know|unsure|not sure|no answer|wouldn't vote|blank)", re.I)
PARTY_TAG = re.compile(r"\(([A-Z]{1,4})\)\s*$")
ELECTION_DAYS = {2010: "2010-11-02", 2012: "2012-11-06", 2014: "2014-11-04", 2016: "2016-11-08",
                 2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05"}
AT_LARGE = {"AK", "DE", "ND", "SD", "VT", "WY"}  # plus MT before 2022 (handled in house_race_id)
REGULAR_SENATE_CLASS = {2010: "Class III", 2012: "Class I", 2014: "Class II", 2016: "Class III",
                        2018: "Class I", 2020: "Class II", 2022: "Class III", 2024: "Class I"}
APPROVAL_WINDOW_DAYS = 7  # approval "near election day" = average of the final 7 days of the series


# --------------------------------------------------------------------------------------------- downloads
def download(url: str, dest: Path, offline: bool) -> Path:
    """Fetch url to dest unless it is already cached (or we are offline)."""
    if dest.exists() or offline:
        if not dest.exists():
            raise FileNotFoundError(f"{dest} missing and --offline given")
        return dest
    resp = requests.get(url, headers=UA, timeout=120)
    resp.raise_for_status()
    if resp.content[:200].lstrip().lower().startswith(b"<!doctype html") and dest.suffix == ".csv":
        raise RuntimeError(f"{url} returned HTML, not CSV")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


def fetch_votehub(offline: bool) -> list[dict]:
    dest = RAW_VH / "polls_all.json"
    if not offline:
        resp = requests.get(VOTEHUB_URL, headers=UA, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            raise RuntimeError("VoteHub returned no polls")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data))
    return json.loads(dest.read_text())


# --------------------------------------------------------------------------------------------- HTML tables
class TableParser(HTMLParser):
    """Collects every table.wikitable with its heading path, collapsible context and cell text.

    Footnote superscripts (<sup class="reference">) are dropped from the text but their target ids are kept, so a
    sponsor footnote (upper-case letter, e.g. [A]) can be resolved against the page's notes list.
    """

    def __init__(self, all_tables: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.all_tables = all_tables
        self.tables: list[dict] = []
        self.notes: dict[str, str] = {}
        self.headings: list[tuple[int, str]] = []
        self._stack: list[tuple[str, str]] = []  # (tag, class) of open elements
        self._tbl: list[dict] = []
        self._cell: dict | None = None
        self._heading: list | None = None
        self._sup_depth = 0
        self._sup_label = ""
        self._sup_href = ""
        self._note: list | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag not in ("br", "img", "hr", "meta", "link", "input", "wbr"):
            self._stack.append((tag, cls))
        if tag in ("h2", "h3", "h4", "h5", "h6"):
            self._heading = [int(tag[1]), []]
        elif tag == "li" and (a.get("id") or "").startswith("cite_note-"):
            self._note = [a["id"], []]
        elif tag == "table":
            self._tbl.append({"is_wikitable": "wikitable" in cls, "rows": [], "headings": [h[1] for h in self.headings],
                              "collapsible": any("mw-collapsible" in c or "collapsible" in c for _, c in self._stack),
                              "classes": cls})
        elif tag == "tr" and self._tbl:
            self._tbl[-1]["rows"].append([])
        elif tag in ("td", "th") and self._tbl and self._tbl[-1]["rows"]:
            self._cell = {"th": tag == "th", "text": [], "notes": [], "rowspan": _int(a.get("rowspan"), 1),
                          "colspan": _int(a.get("colspan"), 1), "style": a.get("style") or ""}
        elif tag == "sup" and "reference" in cls:
            self._sup_depth = len(self._stack)
            self._sup_label, self._sup_href = "", ""
        elif tag == "a" and self._sup_depth and a.get("href", "").startswith("#cite_note-"):
            self._sup_href = a["href"][1:]
        elif tag == "br" and self._cell is not None:
            self._cell["text"].append(" ")

    def handle_endtag(self, tag):
        if self._sup_depth and tag == "sup" and len(self._stack) == self._sup_depth:
            if self._cell is not None and self._sup_href:
                self._cell["notes"].append((self._sup_label.strip("[] "), self._sup_href))
            self._sup_depth = 0
        if tag in ("h2", "h3", "h4", "h5", "h6") and self._heading:
            level, text = self._heading[0], _clean(" ".join(self._heading[1]))
            self.headings = [h for h in self.headings if h[0] < level] + [(level, text)]
            self._heading = None
        elif tag == "li" and self._note:
            self.notes[self._note[0]] = _clean(" ".join(self._note[1]))
            self._note = None
        elif tag in ("td", "th") and self._cell is not None:
            self._cell["text"] = _clean(" ".join(self._cell["text"]))
            if self._tbl and self._tbl[-1]["rows"]:
                self._tbl[-1]["rows"][-1].append(self._cell)
            self._cell = None
        elif tag == "table" and self._tbl:
            done = self._tbl.pop()
            if done["is_wikitable"] or self.all_tables:
                self.tables.append(done)
        # pop the open-element stack back to the matching tag
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break

    def handle_data(self, data):
        if self._sup_depth:
            self._sup_label += data
            if self._note is not None:
                self._note[1].append(data)
            return
        if self._heading is not None:
            self._heading[1].append(data)
        if self._note is not None:
            self._note[1].append(data)
        if self._cell is not None:
            self._cell["text"].append(data)


def _int(v, default):
    try:
        return int(str(v).strip().rstrip(";"))
    except (TypeError, ValueError):
        return default


def _clean(s: str) -> str:
    s = html.unescape(s).replace("\xa0", " ").replace("[edit]", "")
    s = re.sub(r"\[\s*edit\s*\|\s*edit source\s*\]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def grid(rows: list[list[dict]]) -> list[list[dict]]:
    """Expand rowspan/colspan into a rectangular grid of cell dicts."""
    out: list[list[dict]] = []
    pending: dict[tuple[int, int], dict] = {}
    for r, row in enumerate(rows):
        line: list[dict] = []
        c = 0
        cells = iter(row)
        while True:
            if (r, c) in pending:
                line.append(pending.pop((r, c)))
                c += 1
                continue
            cell = next(cells, None)
            if cell is None:
                break
            for dc in range(cell["colspan"]):
                line.append(cell)
                for dr in range(1, cell["rowspan"]):
                    pending[(r + dr, c + dc)] = cell
            c += cell["colspan"]
        while (r, c) in pending:
            line.append(pending.pop((r, c)))
            c += 1
        out.append(line)
    return out


def parse_html_tables(page_html: str, all_tables: bool = False) -> TableParser:
    p = TableParser(all_tables)
    p.feed(page_html)
    return p


# --------------------------------------------------------------------------------------------- value parsing
def parse_pct(text: str) -> float | None:
    t = text.replace("%", "").replace("<", "").strip()
    m = re.match(r"^-?\d+(\.\d+)?", t)
    return float(m.group(0)) if m else None


def parse_sample(text: str) -> tuple[float | None, str]:
    pop = ""
    m = re.search(r"\(?\b(LV|RV|A|V)(/(LV|RV))?\b\)?\s*$", text.strip(), re.I) or \
        re.search(r"\((LV|RV|A|V)(/(LV|RV))?\)", text, re.I)
    if m:
        pop = m.group(1).lower()
    n = re.search(r"\d[\d,]*", text.split("(")[0])
    return (float(n.group(0).replace(",", "")) if n else None), pop


def _date_part(part: str) -> tuple[int | None, int | None, int | None]:
    """'October 27' -> (None, 10, 27); '30, 2026' -> (2026, None, 30); 'Jan 3, 2026' -> (2026, 1, 3)."""
    mo = re.search(r"[A-Za-z]+", part)
    month = MONTHS.get(mo.group(0).lower()) if mo else None
    yr = re.search(r"\b(20\d\d)\b", part)
    day = re.search(r"\b(\d{1,2})\b", re.sub(r"\b20\d\d\b", "", part))
    return (int(yr.group(1)) if yr else None), month, (int(day.group(1)) if day else None)


def _iso(y: int | None, m: int | None, d: int | None) -> str:
    try:
        return date(y, m, d or 1).isoformat() if y and m else ""
    except ValueError:
        return ""


def parse_date_range(text: str) -> tuple[str, str]:
    """'October 27–30, 2026' / 'Sep 28 – Oct 2, 2026' / 'Dec 29, 2025 – Jan 3, 2026' -> (start, end) ISO."""
    t = text.replace("–", "-").replace("—", "-").replace(" to ", "-")
    t = re.sub(r"^(through|until|as of)\s+", "", t.strip(), flags=re.I)
    parts = [p.strip() for p in t.split("-") if p.strip()]
    if not parts or not re.search(r"\b20\d\d\b", t):
        return "", ""
    sy, sm, sd = _date_part(parts[0])
    ey, em, ed = _date_part(parts[-1])
    ey = ey or sy
    em = em or sm
    if len(parts) == 1:
        iso = _iso(ey, em, ed)
        return iso, iso
    sm = sm or em
    sy = sy or ey
    start, end = _iso(sy, sm, sd), _iso(ey, em, ed)
    if start and end and start > end:  # 'Dec 29 - Jan 3, 2026' with only one year given
        start = _iso(sy - 1, sm, sd)
    return start, end


def split_name_party(label: str) -> tuple[str, str]:
    lab = re.sub(r"\s+", " ", label).strip()
    m = PARTY_TAG.search(lab)
    if m:
        return lab[:m.start()].strip(), m.group(1)
    return lab, ""


def surname(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()  # Giménez -> Gimenez
    toks = [t for t in re.sub(r"[^A-Za-z\-' ]", " ", name).split() if t.lower() not in ("jr", "sr", "ii", "iii", "iv")]
    return toks[-1].lower() if toks else ""


# --------------------------------------------------------------------------------------------- 2026 Wikipedia
def wiki_title_for_senate(abbr: str) -> str:
    kind = "special election" if abbr in SPECIAL_SENATE_2026 else "election"
    return f"2026 United States Senate {kind} in {ABBR_TO_STATE[abbr]}"


def wiki_title_for_house(abbr: str, n_seats: int) -> str:
    if n_seats == 1:
        return f"2026 United States House of Representatives election in {ABBR_TO_STATE[abbr]}"
    return f"2026 United States House of Representatives elections in {ABBR_TO_STATE[abbr]}"


def find_header(g: list[list[dict]]) -> int | None:
    for i, row in enumerate(g[:4]):
        txt = " ".join(c["text"].lower() for c in row)
        if "poll source" in txt and ("date" in txt or "administered" in txt):
            return i
    return None


def sponsor_note(cells: list[dict], notes: dict[str, str]) -> str:
    """Resolve upper-case-letter footnotes (Wikipedia's sponsor notes) on the pollster cell."""
    found = []
    for cell in cells:
        for label, href in cell["notes"]:
            if re.fullmatch(r"[A-Z]{1,2}", label) and href in notes:
                found.append(re.sub(r"^\^\s*(\w\s)*", "", notes[href]).strip())
    return " | ".join(dict.fromkeys(found))


def interpret_poll_table(tbl: dict, notes: dict[str, str], party_hint: dict[str, str]) -> list[dict]:
    """Turn one Wikipedia poll table into poll dicts with candidate -> pct maps."""
    g = grid(tbl["rows"])
    h = find_header(g)
    if h is None:
        return []
    header = [c["text"] for c in g[h]]
    lower = [x.lower() for x in header]
    try:
        i_src = next(i for i, x in enumerate(lower) if "poll source" in x)
        i_date = next(i for i, x in enumerate(lower) if "date" in x or "administered" in x)
    except StopIteration:
        return []
    i_n = next((i for i, x in enumerate(lower) if "sample" in x), None)
    i_moe = next((i for i, x in enumerate(lower) if "margin of error" in x or x.startswith("margin of")), None)
    i_rcv = next((i for i, x in enumerate(lower) if x in ("rcv round", "round")), None)  # AK ranked-choice tables
    first_cand = max(i for i in (i_src, i_date, i_n, i_moe, i_rcv) if i is not None) + 1
    cols = []
    for j in range(first_cand, len(header)):
        if j > 0 and g[h][j] is g[h][j - 1]:
            continue  # colspan duplicate
        name, party = split_name_party(header[j])
        if not name:
            continue
        kind = "cand"
        low = name.lower()
        if low.startswith("undecided") or low in ("unsure", "not sure", "don't know"):
            kind = "undecided"
        elif NON_CANDIDATE.match(low):
            kind = "other" if low.startswith("other") or low.startswith("someone") else "skip"
        if kind == "cand" and not party:
            party = party_hint.get(surname(name), "")
            if not party:
                kind = "skip_unknown"
        cols.append((j, kind, name, party))
    polls = []
    for row in g[h + 1:]:
        if len(row) <= first_cand or all(c["th"] for c in row):
            continue
        src_cell = row[i_src]
        if src_cell is row[i_date] or "average" in src_cell["text"].lower():
            continue  # aggregate / average / spanning note rows
        start, end = parse_date_range(row[i_date]["text"])
        if not end:
            continue
        n, pop = parse_sample(row[i_n]["text"]) if i_n is not None and i_n < len(row) else (None, "")
        pollster_raw = src_cell["text"]
        tags = sorted(set(re.findall(r"\((D|R)\)", pollster_raw)))  # both tags -> bipartisan team, 'D/R'
        cands, other_generic, und = [], 0.0, None
        for j, kind, name, party in cols:
            if j >= len(row):
                continue
            val = parse_pct(row[j]["text"])
            if kind == "cand" and val is not None:
                cands.append((name, party, val))
            elif kind == "other" and val is not None:
                other_generic += val
            elif kind == "undecided" and val is not None:
                und = val
        if len(cands) < 2:
            continue
        polls.append({"pollster": re.sub(r"\s*\((D|R)\)\s*$", "", pollster_raw).strip(),
                      "partisan_sponsor": "/".join(tags), "start_date": start, "end_date": end,
                      "sample_size": n, "population": pop, "cands": cands, "other_generic": other_generic,
                      "undecided_pct": und, "sponsor_note": sponsor_note([src_cell], notes),
                      "rcv_round": row[i_rcv]["text"] if i_rcv is not None and i_rcv < len(row) else ""})
    return polls


def to_row(p: dict) -> dict:
    """Collapse candidate list to dem/rep/other columns (top D and top R; everyone else is 'other')."""
    dem = max((c for c in p["cands"] if c[1] in ("D", "DFL")), key=lambda c: c[2], default=None)
    rep = max((c for c in p["cands"] if c[1] == "R"), key=lambda c: c[2], default=None)
    others = [c for c in p["cands"] if c is not dem and c is not rep]
    top_other = max(others, key=lambda c: c[2], default=None)
    other_pct = sum(c[2] for c in others) + (p.get("other_generic") or 0.0)
    return {"pollster": p["pollster"], "start_date": p["start_date"], "end_date": p["end_date"],
            "sample_size": p["sample_size"], "population": p["population"],
            "partisan_sponsor": p["partisan_sponsor"],
            "dem_cand": dem[0] if dem else "", "dem_pct": dem[2] if dem else None,
            "rep_cand": rep[0] if rep else "", "rep_pct": rep[2] if rep else None,
            "other_cand": "; ".join(f"{c[0]} ({c[1]})" if c[1] else c[0] for c in others),
            "other_pct": round(other_pct, 2) if (others or p.get("other_generic")) else None,
            "undecided_pct": p.get("undecided_pct"), "sponsor_note": p.get("sponsor_note", ""),
            # other_pct lumps independents with the generic 'Other' column; keep each candidate's own share too
            "top_other_cand": top_other[0] if top_other else "", "top_other_pct": top_other[2] if top_other else None,
            "cand_detail": "; ".join(f"{c[0]} ({c[1] or '?'}): {c[2]:g}" for c in
                                     sorted(p["cands"], key=lambda c: -c[2])),
            "rcv_round": p.get("rcv_round", "")}


def party_hints_from_tables(tables: list[dict], kalshi_party: dict[str, str]) -> dict[str, str]:
    hints = dict(kalshi_party)
    for t in tables:
        for row in t["rows"][:3]:
            for c in row:
                name, party = split_name_party(c["text"])
                if party and name:
                    hints[surname(name)] = party
    return hints


def classify_heading(headings: list[str]) -> tuple[bool, bool]:
    """(is_primary_section, is_hypothetical_section) for a table's heading path."""
    joined = " / ".join(headings).lower()
    return bool(re.search(r"primar|runoff|caucus|convention|jungle", joined)), "hypothetical" in joined


def district_from_headings(headings: list[str], abbr: str, n_seats: int) -> str | None:
    if n_seats == 1:
        return f"{abbr}-AL"
    for hd in reversed(headings):
        m = re.match(r"^District\s+(\d+)\b", hd)
        if m:
            return f"{abbr}-{int(m.group(1)):02d}"
        if re.match(r"^At-large", hd, re.I):
            return f"{abbr}-AL"
    return None


def district_page_titles(page_html: str, abbr: str) -> list[str]:
    state = re.escape(ABBR_TO_STATE[abbr])
    pat = rf'href="/wiki/(2026_{state.replace(" ", "_")}%27s_[^"#]*congressional_district_election)"'
    return sorted({requests.utils.unquote(m).replace("_", " ") for m in re.findall(pat, page_html.replace("\\ ", " "))})


def polls_from_page(page: dict, race_for_table, kalshi_party: dict[str, str]) -> list[dict]:
    parsed = parse_html_tables(page["html"])
    hints = party_hints_from_tables(parsed.tables, kalshi_party)
    rows = []
    for t in parsed.tables:
        race_id = race_for_table(t["headings"])
        if not race_id:
            continue
        is_primary, is_hyp = classify_heading(t["headings"])
        if is_primary:
            continue
        for p in interpret_poll_table(t, parsed.notes, hints):
            r = to_row(p)
            r.update(race_id=race_id, is_hypothetical=bool(is_hyp or t["collapsible"]), source="wikipedia",
                     source_page=WIKI_URL + page["title"].replace(" ", "_"),
                     source_url=wiki_ref(page))
            rows.append(r)
    return rows


def wiki_2026_polls(template: pd.DataFrame, kalshi_party: dict[str, dict[str, str]], offline: bool,
                    refresh: bool) -> list:
    """kalshi_party is keyed by state, so a surname only hints a party within its own state."""
    rows: list[dict] = []

    def get(title):
        return load_wiki(title, offline, refresh)

    senate = sorted(r[:2] for r in template.loc[template.chamber.str.lower() == "senate", "race_id"])
    for abbr in senate:
        page = get(wiki_title_for_senate(abbr))
        if page is None:
            print(f"  [wiki] missing Senate page for {abbr}")
            continue
        rows += polls_from_page(page, lambda hd, a=abbr: f"{a}-SEN", kalshi_party.get(abbr, {}))
    house = template[template.chamber.str.lower() == "house"].assign(st=lambda d: d.race_id.str[:2])
    for abbr, n_seats in house.groupby("st").size().items():
        page = get(wiki_title_for_house(abbr, n_seats))
        if page is None:
            print(f"  [wiki] missing House page for {abbr}")
            continue
        rows += polls_from_page(page, lambda hd, a=abbr, n=n_seats: district_from_headings(hd, a, n),
                                kalshi_party.get(abbr, {}))
        for title in district_page_titles(page["html"], abbr):
            m = re.search(r"'s (\d+)(st|nd|rd|th) congressional|'s at-large", title)
            if not m:
                continue
            rid = f"{abbr}-{int(m.group(1)):02d}" if m.group(1) else f"{abbr}-AL"
            sub = get(title)
            if sub is not None:
                rows += polls_from_page(sub, lambda hd, r=rid: r, kalshi_party.get(abbr, {}))
    return rows


# --------------------------------------------------------------------------------------------- 2026 VoteHub
def votehub_race_id(subject: str, n_seats: dict[str, int]) -> tuple[str, str] | None:
    m = re.fullmatch(r"2026 ([A-Z]{2})-(\d{2})", subject or "")
    if m:
        st, d = m.group(1), int(m.group(2))
        return ("house", f"{st}-AL") if n_seats.get(st) == 1 else ("house", f"{st}-{d:02d}")
    m = re.fullmatch(r"2026 ([A-Za-z ]+)", subject or "")
    if m and m.group(1) in STATES:
        return "senate", f"{STATES[m.group(1)]}-SEN"
    return None  # primaries ('2026 Texas Democratic') and anything else


def votehub_2026_polls(vh: list[dict], n_seats: dict[str, int], kalshi_party: dict[str, dict[str, str]],
                       wiki_hints: dict[str, dict[str, str]]) -> list[dict]:
    """VoteHub answers carry no party; take it from same-state Wikipedia headers, then same-state Kalshi names."""
    rows = []
    for x in vh:
        if x.get("poll_type") not in ("us-senator", "us-representative"):
            continue
        rid = votehub_race_id(x.get("subject", ""), n_seats)
        if not rid:
            continue
        st = rid[1][:2]
        hints = {**kalshi_party.get(st, {}), **wiki_hints.get(st, {})}
        cands, other_generic, und = [], 0.0, None
        for a in x.get("answers") or []:
            name, pct = a.get("choice", ""), a.get("pct")
            if pct is None:
                continue
            low = name.lower()
            if low in ("dem", "rep"):  # district-level generic ballot question
                cands.append((f"Generic {'Democrat' if low == 'dem' else 'Republican'}", low[0].upper(), float(pct)))
            elif low.startswith("undecided"):
                und = pct
            elif NON_CANDIDATE.match(low):
                other_generic += pct
            else:
                cands.append((name, a.get("party") or hints.get(surname(name), ""), float(pct)))
        if len(cands) < 2:
            continue
        p = {"pollster": x.get("pollster", ""), "start_date": x.get("start_date", ""),
             "end_date": x.get("end_date", ""), "sample_size": x.get("sample_size"),
             "population": (x.get("population") or "").lower(),
             "partisan_sponsor": {"DEM": "D", "REP": "R", "D": "D", "R": "R"}.get(str(x.get("partisan") or ""), ""),
             "cands": cands, "other_generic": other_generic, "undecided_pct": und,
             "sponsor_note": "; ".join(s if isinstance(s, str) else str(s.get("name", s))
                                       for s in x.get("sponsors") or [])}
        r = to_row(p)
        r.update(race_id=rid[1], is_hypothetical="", source="votehub",  # VoteHub does not flag hypotheticals
                 source_page=VOTEHUB_URL, source_url=x.get("url") or VOTEHUB_URL)
        rows.append(r)
    return rows


def _same_cand(a, b) -> bool:
    """Surnames agree, or one side has no candidate name (party not resolved)."""
    a, b = (a if isinstance(a, str) else ""), (b if isinstance(b, str) else "")
    return not a or not b or surname(a) == surname(b)


def dedupe_against(primary: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Drop VoteHub rows that duplicate a Wikipedia row: same race, end date +-2 days, D and R within 1.5 pts, and
    the same D and R candidates. Compared against all Wikipedia rows, hypothetical tables included (those hold
    pre-primary polls of matchups that later happened, which VoteHub also lists)."""
    if extra.empty or primary.empty:
        return extra
    key = primary.assign(end=pd.to_datetime(primary.end_date, errors="coerce"))[["race_id", "end", "dem_pct",
                                                                                   "rep_pct", "dem_cand", "rep_cand"]]
    keep = []
    for _, r in extra.iterrows():
        cand = key[key.race_id == r.race_id]
        end = pd.to_datetime(r.end_date, errors="coerce")
        close = cand[(cand.end - end).abs() <= pd.Timedelta(days=2)]
        dup = ((close.dem_pct - r.dem_pct).abs().fillna(99) <= 1.5) & \
              ((close.rep_pct - r.rep_pct).abs().fillna(99) <= 1.5) & \
              pd.Series([_same_cand(c, r.dem_cand) and _same_cand(q, r.rep_cand)
                         for c, q in zip(close.dem_cand, close.rep_cand)], index=close.index, dtype=bool)
        keep.append(not dup.any())
    return extra[keep]


def kalshi_names() -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    """{state: {surname: party}} and surname sets per race from Kalshi's candidate list (name cross-check only)."""
    k = pd.read_csv(KALSHI_RACES)
    party_map = {"DEM": "D", "REP": "R", "IND": "I", "LIB": "L", "GRN": "G"}
    by_surname, per_race = {}, {}
    for _, r in k.iterrows():
        names = set()
        for part in str(r.get("candidates") or "").split("|"):
            bits = part.strip().split(":")
            if len(bits) < 2 or not bits[1].strip():
                continue
            sn, code = surname(bits[1]), bits[0].strip().upper()
            names.add(sn)
            if code in party_map:  # 'UNK' (Kalshi's label in same-party top-two races) gives no party hint
                by_surname.setdefault(r.race_id[:2], {}).setdefault(sn, party_map[code])
        per_race[r.race_id] = names
    return by_surname, per_race


def build_polls_2026(template, vh, offline, refresh) -> pd.DataFrame:
    k_party, k_race = kalshi_names()
    chamber = dict(zip(template.race_id, template.chamber))
    n_seats = template[template.chamber.str.lower() == "house"].race_id.str[:2].value_counts().to_dict()
    wiki = pd.DataFrame(wiki_2026_polls(template, k_party, offline, refresh))
    # Party hints for VoteHub answers: each candidate's party label in the most recent Wikipedia poll that names
    # them (a candidate tagged (D) in an early hypothetical can be an independent nominee later, e.g. MT's Bodnar).
    # cand_detail is sorted by share, so when two candidates in one poll share a surname (AK: Dan S. Sullivan (R)
    # and Dan J. Sullivan (I)) the higher-polling one sets the hint.
    wiki_hints: dict[str, dict[str, str]] = {}
    for rid, detail in zip(*wiki.sort_values("end_date")[["race_id", "cand_detail"]].T.values):
        row_hints: dict[str, str] = {}
        for m in re.finditer(r"([^;]+?) \(([A-Z]{1,4})\): [\d.]+", str(detail)):
            row_hints.setdefault(surname(m.group(1)), m.group(2))
        wiki_hints.setdefault(rid[:2], {}).update(row_hints)
    vhub = pd.DataFrame(votehub_2026_polls(vh, n_seats, k_party, wiki_hints))
    vhub = dedupe_against(wiki, vhub)
    df = pd.concat([wiki, vhub], ignore_index=True)
    df["chamber"] = df.race_id.map(chamber)
    df = df[df.chamber.notna()]

    def match(r):
        names = k_race.get(r.race_id, set())
        if not names or names <= {"party"}:  # Kalshi lists only 'Democratic party'/'Republican party' here
            return ""
        main = [surname(x) for x in (r.dem_cand, r.rep_cand) if x]
        if len(main) < 2:  # e.g. NE-SEN independent vs R: use top two named candidates
            main = main + [surname(x.split(" (")[0]) for x in str(r.other_cand or "").split("; ") if x][:2 - len(main)]
        return bool(names) and len(main) == 2 and all(m in names for m in main)

    df["nominee_match"] = df.apply(match, axis=1)
    df = df.drop_duplicates(subset=["race_id", "pollster", "start_date", "end_date", "sample_size", "population",
                                    "dem_cand", "dem_pct", "rep_cand", "rep_pct", "other_pct", "rcv_round"])
    return df[POLL_COLS + EXTRA_2026].sort_values(["chamber", "race_id", "end_date", "pollster"])


# --------------------------------------------------------------------------------------------- national 2026
_POLLSTER_STOP = {"research", "university", "college", "polling", "group", "insights", "news", "the", "strategies",
                  "associates", "poll", "data", "public", "opinion", "center", "service", "market", "times", "lab"}


def _pollster_overlap(a, b) -> bool:
    """True when two pollster labels share a distinctive word ('Reuters/Ipsos' vs 'Ipsos')."""
    def toks(s):
        return {t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if len(t) >= 3 and t not in _POLLSTER_STOP}
    return bool(toks(a) & toks(b))


def build_national_2026(vh: list[dict], offline: bool, refresh: bool) -> pd.DataFrame:
    rows = []
    for x in vh:
        pt, subj = x.get("poll_type"), x.get("subject")
        ans = {a.get("choice"): a.get("pct") for a in x.get("answers") or []}
        base = {"date": x.get("end_date"), "start_date": x.get("start_date"), "pollster": x.get("pollster"),
                "sample": x.get("sample_size"), "population": (x.get("population") or "").lower(),
                "source": "votehub", "source_url": x.get("url") or VOTEHUB_URL}
        if pt == "generic-ballot" and subj == "2026":
            rows.append({**base, "metric": "generic_ballot", "value_dem": ans.get("Dem"), "value_rep": ans.get("Rep")})
        elif pt == "approval" and subj == "Donald Trump":
            rows.append({**base, "metric": "approve", "value": ans.get("Approve")})
            rows.append({**base, "metric": "disapprove", "value": ans.get("Disapprove")})
    wiki_all = wiki_approval_2026(offline, refresh)
    wiki = wiki_all[wiki_all.metric.isin(["approve", "disapprove"])] if not wiki_all.empty else wiki_all
    aggs = [wiki_all[~wiki_all.metric.isin(["approve", "disapprove"])] if not wiki_all.empty else wiki_all,
            wiki_generic_aggregates(offline, refresh)]
    vh_df = pd.DataFrame(rows).drop_duplicates()  # VoteHub lists a few polls twice with identical fields
    if not wiki.empty:
        vh_app = vh_df[vh_df.metric == "approve"].assign(d=lambda d: pd.to_datetime(d.date))
        new = []
        for (dt, pollster), grp in wiki.groupby(["date", "pollster"]):
            ap = grp[grp.metric == "approve"].value.iloc[0] if (grp.metric == "approve").any() else None
            near = vh_app[(vh_app.d - pd.to_datetime(dt)).abs() <= pd.Timedelta(days=2)]
            near = near[[_pollster_overlap(p, pollster) for p in near.pollster]] if len(near) else near
            if ap is not None and ((near.value - ap).abs() <= 1.0).any():
                continue
            new.append(grp)
        wiki = pd.concat(new) if new else wiki.iloc[0:0]
    out = pd.concat([vh_df, wiki, *aggs], ignore_index=True)
    cols = ["date", "start_date", "metric", "value_dem", "value_rep", "value", "pollster", "sample", "population",
            "source", "source_url"]
    return out.reindex(columns=cols).sort_values(["metric", "date", "pollster"])


def load_wiki(title: str, offline: bool, refresh: bool) -> dict | None:
    if offline:
        path = cache_path(RAW_WIKI, title)
        return json.loads(path.read_text()) if path.exists() else None
    return fetch_page(title, RAW_WIKI, refresh=refresh)


def wiki_ref(page: dict) -> str:
    return WIKI_URL + page["title"].replace(" ", "_") + f"?oldid={page['revid']}"


def col_index(hdr: list[str], *prefixes: str) -> int | None:
    return next((i for i, x in enumerate(hdr) if any(x.startswith(p) for p in prefixes)), None)


def wiki_aggregates(page: dict, heading_has: str, value_cols: dict[str, tuple[str, ...]]) -> list[dict]:
    """Aggregator tables ('Source of poll aggregation' / 'Aggregator') -> rows with metric *_avg values."""
    rows = []
    for t in parse_html_tables(page["html"]).tables:
        if heading_has not in " / ".join(t["headings"]):
            continue
        g = grid(t["rows"])
        hdr = [c["text"].lower() for c in g[0]] if g else []
        i_src = col_index(hdr, "aggregator", "source of poll aggregation")
        i_upd = col_index(hdr, "updated", "dates updated")
        if i_src is None or i_upd is None:
            continue
        idx = {k: col_index(hdr, *v) for k, v in value_cols.items()}
        for row in g[1:]:
            if len(row) <= max(i for i in idx.values() if i is not None) or "average" in row[i_src]["text"].lower():
                continue
            _, upd = parse_date_range(row[i_upd]["text"])
            vals = {k: parse_pct(row[i]["text"]) if i is not None else None for k, i in idx.items()}
            rows.append({"date": upd, "pollster": row[i_src]["text"], "source": "wikipedia_aggregate",
                         "source_url": wiki_ref(page), **vals})
    return rows


def wiki_approval_2026(offline: bool, refresh: bool) -> pd.DataFrame:
    """Individual Trump job-approval polls (monthly tables, 2026 only) plus aggregator rows."""
    page = load_wiki("Opinion polling on the second Trump presidency", offline, refresh)
    if page is None:
        return pd.DataFrame()
    rows = []
    for t in parse_html_tables(page["html"]).tables:
        hds = t["headings"]
        if not hds or hds[0] != "Nationwide job approval ratings" or len(hds) < 3 or not hds[1].isdigit():
            continue
        year = hds[1]
        g = grid(t["rows"])
        hdr = [c["text"].lower() for c in g[0]] if g else []
        i_src, i_date = col_index(hdr, "poll source", "pollster"), col_index(hdr, "date")
        i_ap, i_dis, i_n = col_index(hdr, "approve"), col_index(hdr, "disapprove"), col_index(hdr, "sample")
        if None in (i_src, i_date, i_ap, i_dis):
            continue
        for row in g[1:]:
            if len(row) <= max(i_ap, i_dis) or row[i_src] is row[i_date]:
                continue
            dtxt = row[i_date]["text"]
            start, end = parse_date_range(dtxt if re.search(r"\b20\d\d\b", dtxt) else f"{dtxt}, {year}")
            if not end.startswith("2026"):
                continue
            n, pop = parse_sample(row[i_n]["text"]) if i_n is not None else (None, "")
            base = {"date": end, "start_date": start, "pollster": row[i_src]["text"], "sample": n,
                    "population": pop, "source": "wikipedia", "source_url": wiki_ref(page)}
            for metric, i in (("approve", i_ap), ("disapprove", i_dis)):
                val = parse_pct(row[i]["text"])
                if val is not None:
                    rows.append({**base, "metric": metric, "value": val})
    for r in wiki_aggregates(page, "Aggregate polls / Approval", {"approve": ("approve",),
                                                                  "disapprove": ("disapprove",)}):
        for metric in ("approve", "disapprove"):
            if r[metric] is not None:
                rows.append({"date": r["date"], "metric": f"{metric}_avg", "value": r[metric],
                             "pollster": r["pollster"], "source": r["source"], "source_url": r["source_url"]})
    return pd.DataFrame(rows)


def wiki_generic_aggregates(offline: bool, refresh: bool) -> pd.DataFrame:
    page = load_wiki("2026 United States House of Representatives elections", offline, refresh)
    if page is None:
        return pd.DataFrame()
    rows = wiki_aggregates(page, "Generic congressional ballot aggregate polls",
                           {"value_rep": ("republicans",), "value_dem": ("democrats",)})
    return pd.DataFrame(rows).assign(metric="generic_ballot_avg") if rows else pd.DataFrame()


# --------------------------------------------------------------------------------------------- historical polls
def house_race_id(st: str, seat: int, year: int) -> str:
    if st in AT_LARGE or (st == "MT" and year < 2022) or seat == 0:
        return f"{st}-AL"
    return f"{st}-{seat:02d}"


def pivot_538(path: Path, chamber: str) -> pd.DataFrame:
    """538 *_polls_historical.csv (one row per answer) -> one row per question with dem/rep/other columns."""
    d = pd.read_csv(path, low_memory=False)
    d = d[d.stage.isin(["general", "runoff", "jungle primary"]) & d.state.isin(STATES)]
    d = d[d.cycle.isin(ELECTION_DAYS.keys())]
    out = []
    for qid, q in d.groupby("question_id", sort=False):
        f = q.iloc[0]
        st = STATES[f.state]
        if chamber == "Senate":
            race_id = f"{st}-SEN"
        else:
            seat = int(f.seat_number) if pd.notna(f.seat_number) else 0
            race_id = house_race_id(st, seat, int(f.cycle))
        if q.ranked_choice_round.notna().any():  # RCV tables: keep the final (head-to-head) round
            q = q[q.ranked_choice_round == q.ranked_choice_round.max()]
        dem = q[q.party == "DEM"].sort_values("pct", ascending=False).head(1)
        rep = q[q.party == "REP"].sort_values("pct", ascending=False).head(1)
        oth = q.drop(dem.index).drop(rep.index)
        out.append({
            "year": int(f.cycle), "race_id": race_id, "chamber": chamber, "pollster": f.pollster,
            "start_date": _iso_us(f.start_date), "end_date": _iso_us(f.end_date), "sample_size": f.sample_size,
            "population": str(f.population).lower() if pd.notna(f.population) else "",
            "partisan_sponsor": f.partisan if pd.notna(f.partisan) else "",
            "dem_cand": dem.answer.iloc[0] if len(dem) else "", "dem_pct": dem.pct.iloc[0] if len(dem) else None,
            "rep_cand": rep.answer.iloc[0] if len(rep) else "", "rep_pct": rep.pct.iloc[0] if len(rep) else None,
            "other_cand": "; ".join(f"{a} ({p})" for a, p in zip(oth.answer, oth.party)),
            "other_pct": oth.pct.sum() if len(oth) else None, "source_url": f.url if pd.notna(f.url) else "",
            "stage": f.stage,
            "special": chamber == "Senate" and f.seat_name != REGULAR_SENATE_CLASS.get(int(f.cycle)),
            "seat_name": f.seat_name if pd.notna(f.get("seat_name")) else "",
            "is_hypothetical": bool(f.hypothetical) if pd.notna(f.hypothetical) else False,
            "ranked_choice_reallocated": f.ranked_choice_reallocated,
            "numeric_grade": f.numeric_grade, "pollster_rating_id": f.pollster_rating_id,
            "internal": f.internal, "question_id": qid, "poll_id": f.poll_id, "source": "538_polls_historical",
        })
    return pd.DataFrame(out)


def _iso_us(s) -> str:
    try:
        return datetime.strptime(str(s), "%m/%d/%y").date().isoformat()
    except ValueError:
        try:
            return pd.to_datetime(s).date().isoformat()
        except (ValueError, TypeError):
            return ""


def raw_polls_hist(path: Path, years: set[int]) -> pd.DataFrame:
    """538 pollster-ratings raw_polls.csv (polls from the final ~60 days, with results) for cycles not in the full archives."""
    d = pd.read_csv(path, low_memory=False)
    d = d[d.race.str.split("_").str[1].isin(["Sen-G", "Sen-GS", "House-G"]) & d.cycle.isin(years)]
    d = d[d.location != "US"]  # 2010_House-G_US = national generic-ballot polls, not a race
    out = []
    for _, r in d.iterrows():
        chamber = "Senate" if "_Sen-" in str(r.race) else "House"
        loc = str(r.location)
        if chamber == "Senate":
            race_id = f"{loc[:2]}-SEN"
        else:
            m = re.match(r"([A-Z]{2})-(\d+|AL)", loc)
            seat = 0 if not m or m.group(2) == "AL" else int(m.group(2))
            race_id = house_race_id(m.group(1), seat, int(r.cycle)) if m else loc
        c = [(r.cand1_name, r.cand1_party, r.cand1_pct), (r.cand2_name, r.cand2_party, r.cand2_pct)]
        dem = next((x for x in c if x[1] == "DEM"), None)
        rep = next((x for x in c if x[1] == "REP"), None)
        oth = [x for x in c if x is not dem and x is not rep]
        out.append({
            "year": int(r.cycle), "race_id": race_id, "chamber": chamber, "pollster": r.pollster, "start_date": "",
            "end_date": _iso_us(r.polldate), "sample_size": r.samplesize, "population": "",
            "partisan_sponsor": r.partisan if pd.notna(r.partisan) else "",
            "dem_cand": dem[0] if dem else "", "dem_pct": dem[2] if dem else None,
            "rep_cand": rep[0] if rep else "", "rep_pct": rep[2] if rep else None,
            "other_cand": "; ".join(f"{x[0]} ({x[1]})" for x in oth), "other_pct": sum(x[2] for x in oth) if oth else None,
            "source_url": GH538 + "pollster-ratings/raw_polls.csv", "stage": "general",
            "special": "_Sen-GS_" in str(r.race),
            "seat_name": r.race, "is_hypothetical": False, "numeric_grade": None,
            "pollster_rating_id": r.pollster_rating_id, "internal": "", "question_id": r.question_id,
            "poll_id": r.poll_id, "source": "538_raw_polls", "margin_actual": r.margin_actual,
        })
    return pd.DataFrame(out)


def build_polls_hist() -> pd.DataFrame:
    full = pd.concat([pivot_538(RAW_538 / "senate_polls_historical.csv", "Senate"),
                      pivot_538(RAW_538 / "house_polls_historical.csv", "House")], ignore_index=True)
    have = set(full.year.unique())
    early = raw_polls_hist(RAW_538 / "raw_polls.csv", set(ELECTION_DAYS) - have)
    df = pd.concat([full, early], ignore_index=True)
    lead = ["year"] + POLL_COLS
    return df[lead + [c for c in df.columns if c not in lead]].sort_values(["year", "chamber", "race_id", "end_date"])


# --------------------------------------------------------------------------------------------- national hist
def generic_final(year: int) -> tuple[float | None, float | None, str]:
    eday = ELECTION_DAYS[year]
    if year <= 2016:
        g = pd.read_csv(RAW_538 / "generic_topline_historical.csv")
        g["d"] = pd.to_datetime(g.modeldate, format="%m/%d/%Y")
        g = g[g.d <= eday].sort_values("d")
        if g.empty:
            return None, None, ""
        last = g.iloc[-1]
        return last.dem_estimate, last.rep_estimate, f"538 generic_topline_historical {last.d.date()}"
    g = pd.read_csv(RAW_538 / "generic_ballot_averages.csv")
    g = g[(g.cycle == year) & (g.date <= eday)]
    if g.empty:
        return None, None, ""
    last_date = g.date.max()
    last = g[g.date == last_date].set_index("candidate").pct_estimate
    return last.get("Democrats"), last.get("Republicans"), f"538 generic_ballot_averages {last_date}"


def approval_final(year: int) -> tuple[float | None, float | None, str, str]:
    eday = pd.Timestamp(ELECTION_DAYS[year])
    lo = eday - pd.Timedelta(days=APPROVAL_WINDOW_DAYS)
    if year <= 2016:
        t = obama_gallup()
        t = t[(t.end <= eday) & (t.end > lo - pd.Timedelta(days=7))]  # Gallup weekly: final 1-2 weekly readings
        if t.empty:
            return None, None, "", ""
        return round(t.approve.mean(), 2), round(t.disapprove.mean(), 2), "Barack Obama", "Gallup weekly (UCSB)"
    if year in (2018, 2020):
        t = pd.read_csv(RAW_538 / "trump1_approval_topline.csv")
        t = t[t.subgroup == "All polls"].assign(d=lambda x: pd.to_datetime(x.modeldate, format="%m/%d/%Y"))
        who, src = "Donald Trump", "538 trump approval_topline (All polls)"
    else:
        t = pd.read_csv(RAW_538 / "biden_approval_topline.csv")
        t = t[(t.subgroup == "All polls") & (t.politician == "Joe Biden")].assign(d=lambda x: pd.to_datetime(x.end_date))
        who, src = "Joe Biden", "538 biden approval_topline (All polls)"
    t = t[(t.d <= eday) & (t.d > lo)]
    if t.empty:
        return None, None, who, src
    return round(t.approve_estimate.mean(), 2), round(t.disapprove_estimate.mean(), 2), who, src


def obama_gallup() -> pd.DataFrame:
    page = (RAW_UCSB / "barack-obama-public-approval.html").read_text(errors="replace")
    rows = []
    for t in parse_html_tables(page, all_tables=True).tables:
        for row in t["rows"]:
            txt = [c["text"] for c in row]
            if len(txt) >= 4 and re.match(r"\d{1,2}/\d{1,2}/\d{4}", txt[1] or ""):
                rows.append({"end": pd.to_datetime(txt[1], format="%m/%d/%Y"), "approve": parse_pct(txt[2]),
                             "disapprove": parse_pct(txt[3])})
    return pd.DataFrame(rows)


def house_popular_vote() -> pd.DataFrame:
    """National House popular vote from MIT Election Lab (general elections, all candidate lines)."""
    sep = "\t" if "\t" in MIT_HOUSE.open().readline() else ","  # the .tab download is actually comma-separated
    m = pd.read_csv(MIT_HOUSE, sep=sep, low_memory=False)
    flag = lambda col: m[col].astype(str).str.upper().isin(["TRUE", "1"])  # noqa: E731
    m = m[(m.stage.str.upper() == "GEN") & ~flag("special") & ~flag("runoff")]
    m = m[m.year.isin(ELECTION_DAYS.keys())]
    # Fusion candidates (NY, CT) appear on several lines; count every line toward the candidate's major party.
    m["cand_key"] = m.year.astype(str) + "|" + m.state_po + "|" + m.district.astype(str) + "|" + m.candidate.astype(str)
    # MN's DFL and ND's Democratic-NPL are the Democratic Party's state affiliates.
    m["party1"] = m.party.replace({"DEMOCRATIC-FARMER-LABOR": "DEMOCRAT", "DEMOCRATIC-NPL": "DEMOCRAT"})
    major = m[m.party1.isin(["DEMOCRAT", "REPUBLICAN"])].groupby("cand_key").party1.first()
    m["party2"] = m.cand_key.map(major).fillna("OTHER")
    tot = m.groupby("year").candidatevotes.sum()
    by = m.groupby(["year", "party2"]).candidatevotes.sum().unstack(fill_value=0)
    return pd.DataFrame({"house_dem_share": by.DEMOCRAT / tot * 100, "house_rep_share": by.REPUBLICAN / tot * 100,
                         "house_total_votes": tot}).reset_index()


def build_national_hist() -> pd.DataFrame:
    pv = house_popular_vote().set_index("year")
    rows = []
    for y in sorted(ELECTION_DAYS):
        gd, gr, gsrc = generic_final(y)
        ap, dis, who, asrc = approval_final(y)
        dshare = pv.house_dem_share.get(y)
        rshare = pv.house_rep_share.get(y)
        rows.append({
            "year": y, "cycle_type": "midterm" if y % 4 == 2 else "presidential", "election_date": ELECTION_DAYS[y],
            "generic_dem": _r(gd), "generic_rep": _r(gr), "generic_margin_dr": _r(gd - gr) if gd and gr else None,
            "generic_source": gsrc, "president": who, "president_party": "D" if who in ("Barack Obama", "Joe Biden") else "R",
            "approve": ap, "disapprove": dis, "net_approval": _r(ap - dis) if ap and dis else None,
            "approval_source": asrc,
            "house_dem_share": _r(dshare), "house_rep_share": _r(rshare),
            "house_margin_dr": _r(dshare - rshare) if dshare is not None else None,
            "house_total_votes": pv.house_total_votes.get(y), "house_vote_source": "MIT Election Lab 1976-2024-house.tab",
        })
    return pd.DataFrame(rows)


def _r(x, nd=2):
    return None if x is None or pd.isna(x) else round(float(x), nd)


# --------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="rebuild only from data/raw/polls")
    ap.add_argument("--refresh-wiki", action="store_true", help="re-download cached Wikipedia pages")
    args = ap.parse_args()
    for name, url in SOURCES_538.items():
        download(url, RAW_538 / name, args.offline)
    download(UCSB_OBAMA, RAW_UCSB / "barack-obama-public-approval.html", args.offline)
    vh = fetch_votehub(args.offline)
    template = pd.read_csv(TEMPLATE)
    CLEAN.mkdir(parents=True, exist_ok=True)

    p26 = build_polls_2026(template, vh, args.offline, args.refresh_wiki)
    p26.to_csv(CLEAN / "polls_2026.csv", index=False)
    print(f"polls_2026: {len(p26)} rows, {p26.race_id.nunique()} races")
    n26 = build_national_2026(vh, args.offline, args.refresh_wiki)
    n26.to_csv(CLEAN / "national_2026.csv", index=False)
    print(f"national_2026: {len(n26)} rows")
    ph = build_polls_hist()
    ph.to_csv(CLEAN / "polls_hist.csv", index=False)
    print(f"polls_hist: {len(ph)} rows")
    nh = build_national_hist()
    nh.to_csv(CLEAN / "national_hist.csv", index=False)
    print(nh.to_string())


if __name__ == "__main__":
    main()
