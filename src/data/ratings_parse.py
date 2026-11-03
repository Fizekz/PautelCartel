"""Parse Wikipedia expert-rating tables and House district tables into tidy records (used by fetch_ratings.py)."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from html_tables import extract_tables

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

# Wikipedia column header (first word(s)) -> canonical forecaster key. Core three first.
RATER_ALIASES = {
    "cook": "cook", "sabato": "sabato",
    "ie": "inside", "inside": "inside", "rothenberg": "inside", "roth.": "inside",  # Rothenberg -> Inside Elections
    "rcp": "rcp", "econ.": "economist", "econ": "economist", "economist": "economist", "st": "split_ticket",
    "ddhq": "ddhq", "ddhq/the hill": "ddhq", "fpo": "fiftyplusone", "fox": "fox", "fox news": "fox",
    "silver": "silver_bulletin", "538": "fte", "nyt": "nyt", "cq politics": "cq", "roll call": "rollcall",
    "the hill": "thehill", "daily kos": "dailykos", "politico": "politico", "cnn": "cnn", "cbs": "cbs",
    "nbc": "nbc", "ed": "elections_daily", "princeton": "princeton", "jay desart": "desart", "tpm": "tpm",
}
CORE_RATERS = ("cook", "sabato", "inside")

LEVELS = {"safe": 3.0, "solid": 3.0, "likely": 2.0, "lean": 1.0, "tilt": 0.5}
RATING_RE = re.compile(r"^(safe|solid|likely|lean|tilt)\s*([DRI])\b", re.I)
TOSSUP_RE = re.compile(r"^toss[\s-]?up", re.I)
MONTH_RE = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})")
DIST_RE = re.compile(r"^(?P<state>[A-Z][a-z]+(?: [A-Z][a-z]+)?)\s+(?P<num>\d+|at-large|at large)\b")
CAND_PARTY_RE = re.compile(r"\u258c[^\u258c(]*\(([^)]+)\)")  # '▌Name (Party)'
PVI_RE = re.compile(r"^(?:(?P<p>[DR])\+(?P<n>\d+)|(?P<even>EVEN))")


@dataclass(frozen=True)
class RaterColumn:
    index: int
    key: str
    header: str
    as_of: str  # ISO date or ""


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def encode_rating(raw: str) -> float | None:
    """Map rating text to the D(-) / R(+) scale. Independents (I) sit on the non-Republican side (negative)."""
    text = re.sub(r"\(flip\)", "", norm(raw), flags=re.I).strip()
    text = re.sub(r"^(safe|solid|likely|lean|tilt)(?=[DRI]\b)", r"\1 ", text, flags=re.I)  # "SafeR" typo
    if TOSSUP_RE.match(text):
        return 0.0
    m = RATING_RE.match(text)
    if not m:
        return None
    value = LEVELS[m.group(1).lower()]
    return value if m.group(2).upper() == "R" else -value


def parse_pvi(raw: str) -> float | None:
    m = PVI_RE.match(norm(raw))
    if not m:
        return None
    if m.group("even"):
        return 0.0
    n = float(m.group("n"))
    return n if m.group("p") == "R" else -n


def house_race_id(label: str) -> str | None:
    m = DIST_RE.match(norm(label))
    if not m or m.group("state") not in STATES:
        return None
    num = m.group("num")
    suffix = "AL" if num.startswith("at") else f"{int(num):02d}"
    return f"{STATES[m.group('state')]}-{suffix}"


def senate_label(label: str) -> tuple[str, str] | None:
    """'Georgia (special)' -> ('GA', 'special'); 'New York (regular)' -> ('NY', 'regular'); 'Ohio' -> ('OH', '')."""
    text = norm(label)
    m = re.match(r"^([A-Z][a-z]+(?: [A-Z][a-z]+)?)\s*(?:\(?\s*(special|regular|class \d)\s*\)?)?", text, re.I)
    if not m or m.group(1) not in STATES:
        return None
    kind = (m.group(2) or "").lower()
    return STATES[m.group(1)], "special" if kind == "special" else ("regular" if kind else "")


def parse_header_date(header: str) -> str:
    m = MONTH_RE.search(header)
    if not m:
        return ""
    return dt.datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date().isoformat()


def rater_key(header: str) -> str | None:
    head = MONTH_RE.split(norm(header))[0].strip().lower()
    head = re.sub(r"\s*(january|february|march|april|june|july|august|september|october|november|december)$", "",
                  head).strip()
    return RATER_ALIASES.get(head)


def find_rating_table(html: str) -> tuple[list[list[str]], int, list[RaterColumn]] | None:
    """Return (grid, header_row_index, rater_columns) for the first table carrying Cook and Sabato columns."""
    for grid in extract_tables(html):
        for h_idx, row in enumerate(grid[:3]):
            cells = [norm(c) for c in row]
            if not any(c.startswith("Cook") for c in cells) or not any(c.startswith("Sabato") for c in cells):
                continue
            cols, seen = [], set()
            for i, c in enumerate(cells):
                key = rater_key(c)
                if key and key not in seen:
                    seen.add(key)
                    cols.append(RaterColumn(i, key, c, parse_header_date(c)))
            return grid, h_idx, cols
    return None


def rating_rows(grid: list[list[str]], h_idx: int) -> list[list[str]]:
    rows = []
    for row in grid[h_idx + 1:]:
        label = norm(row[0]) if row else ""
        if label.lower().startswith("overall"):
            break
        if label in ("District", "State", "Constituency", ""):
            continue
        rows.append([norm(c) for c in row])
    return rows


def party_letter(text: str) -> str:
    t = norm(text).lower()
    if t.startswith(("democratic", "dfl")):
        return "D"
    if t.startswith("republican"):
        return "R"
    if t.startswith("independent"):
        return "I"
    return ""


def house_districts(html: str) -> dict[str, dict]:
    """race_id -> {pvi, incumbent_party, incumbent} from the per-state tables of a House elections page.

    Only tables whose header has a PVI column and a Candidates column are used (that excludes the special-election
    and seat-summary tables). The first row seen for a district wins.
    """
    out: dict[str, dict] = {}
    for grid in extract_tables(html):
        header = None
        for row in grid:
            cells = [norm(c) for c in row]
            if header is None:
                if "Party" in cells and any("PVI" in c for c in cells) and "Candidates" in cells:
                    header = cells
                continue
            rid = house_race_id(cells[0]) if cells else None
            if not rid or rid in out or len(cells) < len(header):
                continue
            pvi_idx = max(i for i, c in enumerate(header) if "PVI" in c)  # 'New 2026 PVI' beats 'Original'
            party_idx = header.index("Party")
            member_idx = party_idx - 1
            party = party_letter(cells[party_idx])
            member = cells[member_idx]
            if re.search(r"none \(new seat\)|new seat|vacant", member, re.I) and not party:
                member = "None (new seat)"
            parties = [party_letter(x) or "O" for x in CAND_PARTY_RE.findall(cells[header.index("Candidates")])]
            same = parties[0] if len(parties) >= 2 and len(set(parties)) == 1 and parties[0] in "DR" else ""
            out[rid] = {"pvi": parse_pvi(cells[pvi_idx]), "incumbent_party": party, "incumbent": member,
                        "n_cands_wiki": len(parties), "same_party_final": same}
    return out
