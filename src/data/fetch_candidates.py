"""Build data/clean/candidates_2026.csv and data/clean/redistricting_2026.csv.

Source: English Wikipedia pages "2026 United States House of Representatives elections" and
"2026 United States Senate elections" (rendered HTML via the MediaWiki parse API, cached as JSON under
data/raw/candidates/). Kalshi's race snapshot is used ONLY to cross-check candidate names
(kalshi_name_match column); it never feeds any other field.

Usage:
    .venv/bin/python src/data/fetch_candidates.py            # uses cached raw pages if present
    .venv/bin/python src/data/fetch_candidates.py --refresh  # re-download the Wikipedia pages

stdlib + pandas + requests only. HTML is parsed with html.parser (no lxml/bs4).
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiki_fetch import fetch_page  # noqa: E402  (polite cached MediaWiki fetcher shared in src/data)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "candidates"
CLEAN_DIR = ROOT / "data" / "clean"
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"
KALSHI_RACES = ROOT / "data" / "kalshi" / "races_2026-11-01T074812Z.csv"
HOUSE_PAGE = "2026 United States House of Representatives elections"
SENATE_PAGE = "2026 United States Senate elections"

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

PARTY_MAP = {
    "democratic": "DEM", "dfl": "DEM", "democratic-farmer-labor": "DEM", "democratic–npl": "DEM",
    "democratic-npl": "DEM", "republican": "REP", "independent": "IND", "no party preference": "IND",
    "nonpartisan": "IND", "undeclared": "IND", "unaffiliated": "IND", "libertarian": "LIB", "green": "GRN",
}

# Election system by state (general election). Everything else is plurality.
TOP_TWO_STATES = {"CA", "WA"}
RCV_STATES = {"AK", "ME"}            # AK: top-four primary + RCV general; ME: RCV for federal generals
RUNOFF_STATES = {"GA"}               # GA: majority required in general, runoff Dec 1 2026

# Independents the task names (or that are the de-facto main opposition) - forced is_major=1.
MAJOR_INDEPENDENTS = {
    ("NE-SEN", "Dan Osborn"), ("MT-SEN", "Seth Bodnar"), ("AK-AL", "Bill Hill"),
    ("ID-SEN", "Todd Achilles"), ("SD-SEN", "Brian Bengs"),
}
# Major-party nominees who are NOT one of the top two because a stronger independent leads the opposition.
NOT_MAJOR = {("NE-SEN", "DEM"), ("MT-SEN", "DEM"), ("ID-SEN", "DEM"), ("AK-AL", "DEM")}

SPECIAL_SENATE = {"FL-SEN", "OH-SEN"}
SYSTEM_NOTES = {
    "LA-House": "LA House 2026: Nov 3 is an all-party jungle primary on the new map; runoff Dec 12 if no one >50%",
    "GA": "GA general needs a majority; runoff Dec 1 2026 if no one >50%",
    "AK": "AK: top-four primary, ranked-choice general",
    "ME": "ME: ranked-choice general for federal races",
    "CA": "CA top-two general (same-party pairs possible)",
    "WA": "WA top-two general (same-party pairs possible)",
}

MAJOR_PARTY_ORDER = ["DEM", "REP"]
MINOR_PRIORITY = ["IND", "LIB", "GRN", "OTHER"]


# --------------------------------------------------------------------------------------------- HTML tables
class _TableParser(HTMLParser):
    """Collect every <table> (nested ones included) with the nearest preceding h2/h3/h4 heading."""

    BREAKS = {"br", "li", "p", "div"}
    SKIP = {"style", "script", "sup"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[dict] = []
        self.tables: list[dict] = []
        self.skip_depth = 0
        self.heading = ""
        self.heading_buf: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self.skip_depth += 1
            return
        if tag in ("h2", "h3", "h4"):
            self.heading_buf = []
        if tag == "table":
            self.stack.append({"rows": [], "cur": None, "cell": None, "heading": self.heading})
            return
        if not self.stack:
            return
        t = self.stack[-1]
        if tag == "tr":
            t["cur"] = []
        elif tag in ("td", "th") and t["cur"] is not None:
            t["cell"] = {"text": [], "rs": _span(a.get("rowspan")), "cs": _span(a.get("colspan"))}
        elif tag in self.BREAKS and t["cell"] is not None:
            t["cell"]["text"].append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if tag in ("h2", "h3", "h4") and self.heading_buf is not None:
            self.heading = re.sub(r"\[edit\]", "", "".join(self.heading_buf)).strip()
            self.heading_buf = None
        if not self.stack:
            return
        t = self.stack[-1]
        if tag in ("td", "th") and t["cell"] is not None and t["cur"] is not None:
            cell = t["cell"]
            lines = (re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in "".join(cell["text"]).split("\n"))
            t["cur"].append({**cell, "text": "\n".join(ln for ln in lines if ln)})
            t["cell"] = None
        elif tag == "tr" and t["cur"] is not None:
            t["rows"].append(t["cur"])
            t["cur"] = None
        elif tag == "table":
            self.tables.append(self.stack.pop())

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.heading_buf is not None:
            self.heading_buf.append(data)
        if self.stack and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"]["text"].append(data)


def _span(value) -> int:
    digits = re.sub(r"\D", "", value or "")
    return max(1, int(digits)) if digits else 1


def _grid(raw_rows: list[list[dict]]) -> list[list[str]]:
    """Expand rowspan/colspan into a rectangular text grid."""
    grid, pending = [], {}
    for cells in raw_rows:
        row, col, ci = [], 0, 0
        while ci < len(cells) or col in pending:
            if col in pending:
                text, left = pending.pop(col)
                if left > 1:
                    pending[col] = (text, left - 1)
                row.append(text)
                col += 1
                continue
            cell = cells[ci]
            ci += 1
            for _ in range(cell["cs"]):
                row.append(cell["text"])
                if cell["rs"] > 1:
                    pending[col] = (cell["text"], cell["rs"] - 1)
                col += 1
        grid.append(row)
    return grid


def parse_tables(html: str) -> list[dict]:
    parser = _TableParser()
    parser.feed(html)
    return [{"heading": t["heading"], "rows": _grid(t["rows"])} for t in parser.tables]


# --------------------------------------------------------------------------------------------- helpers
def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)|\".*?\"", " ", s)
    s = re.sub(r"[^a-z\s-]", " ", s).replace("-", " ")
    toks = [t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(toks)


def map_party(label: str) -> str:
    first = re.split(r",|/", label)[0].strip().lower()
    return PARTY_MAP.get(first, "OTHER")


CAND_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<party>[^()]+)\)\s*(?P<rest>.*)$")


def parse_candidates(cell: str) -> list[dict]:
    """'▌Name (Party)' lines -> [{'name','party_label'}]. Drops write-ins and percentage-only lines."""
    out = []
    for line in cell.split("\n"):
        line = line.replace("▌", "").strip()
        line = re.sub(r"^Y\s+", "", line)          # winner check-mark glyph rendered as 'Y'
        m = CAND_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        if not name or name.lower().startswith(("none", "tbd", "candidates")):
            continue
        out.append({"name": name, "party_label": m.group("party").strip(), "extra": m.group("rest").strip()})
    return out


def house_race_id(location: str) -> str | None:
    loc = location.split("\n")[0].strip()
    m = re.match(r"^(.+?)\s+(at-large|\d+)$", loc)
    if not m or m.group(1) not in STATES:
        return None
    st, num = STATES[m.group(1)], m.group(2)
    return f"{st}-AL" if num == "at-large" else f"{st}-{int(num):02d}"


def ordinal_district(text: str) -> int | None:
    m = re.search(r"(?:Redistricted|Moved|Running) (?:from|in) the (\d+)(?:st|nd|rd|th) district", text)
    return int(m.group(1)) if m else None


def system_for(state: str) -> str:
    if state in TOP_TWO_STATES:
        return "top-two"
    if state in RCV_STATES:
        return "rcv"
    if state in RUNOFF_STATES:
        return "runoff-possible"
    return "plurality"


def names_match(a: str, b: str) -> bool:
    """Loose person-name match: exact normalized, or same first+last token (ignores middle names/initials)."""
    na, nb = norm_name(a).split(), norm_name(b).split()
    if not na or not nb:
        return False
    return na == nb or (na[0] == nb[0] and na[-1] == nb[-1])


# Name collisions resolved by hand (race_id, candidate) -> True means this is the incumbent.
INCUMBENT_OVERRIDES = {("AK-SEN", "Dan S. Sullivan"): True, ("AK-SEN", "Dan J. Sullivan"): False}


# --------------------------------------------------------------------------------------------- House
def parse_member_cell(member: str, party: str, status: str) -> dict | None:
    """One incumbent row -> member dict, or None for 'None (new seat)' rows. Vacant rows keep the prior party."""
    first = member.split("\n")[0].strip()
    if first.startswith("None"):
        return None
    if first == "Vacant":
        m = re.search(r"\((D|R|I)\) resigned|\((D|R|I)\) died", status)
        letter = (m.group(1) or m.group(2)) if m else ""
        return {"name": "", "vacant": True, "party": {"D": "DEM", "R": "REP", "I": "IND"}.get(letter, ""),
                "from_district": None, "status": status.replace("\n", " ")}
    return {"name": first, "vacant": False, "party": map_party(party), "from_district": ordinal_district(member),
            "status": status.replace("\n", " ")}


def house_races(tables: list[dict]) -> dict[str, dict]:
    """race_id -> {'members': [...], 'candidates': [...], 'statuses': [...]} from the per-state tables."""
    races: dict[str, dict] = {}
    for t in tables:
        if t["heading"] not in STATES or not t["rows"] or t["rows"][0][-1] != "Candidates":
            continue
        for row in t["rows"][2:]:
            if len(row) < 7:
                continue
            rid = house_race_id(row[0])
            if rid is None:
                continue
            race = races.setdefault(rid, {"members": [], "candidates": parse_candidates(row[6]), "pvi": row[1]})
            member = parse_member_cell(row[2], row[3], row[5])
            if member is not None:
                race["members"].append(member)
    return races


def old_number_holders(races: dict[str, dict]) -> dict[str, dict]:
    """race_id on OLD lines -> the member who holds that district number now, wherever Wikipedia lists them.

    In redrawn states Wikipedia files a member under the new district they run in (with 'Redistricted from the
    Nth district'), so the current holder of a number can sit in another race's row."""
    out: dict[str, dict] = {}
    for rid, race in races.items():
        for m in race["members"]:
            num = m["from_district"]
            key = rid if num is None else f"{rid[:2]}-{num:02d}"
            out.setdefault(key, {**m, "listed_in": rid})
    return out


def seat_holder(rid: str, members: list[dict], holders_by_number: dict[str, dict]) -> tuple[str, str]:
    """(party holding the seat now, note) for a House race on 2026 lines.

    Verifier fix (2026-11-01): the seat's party is ALWAYS the party of the member who now holds this district
    NUMBER (old lines), even when that member runs elsewhere or retires, so no House seat is left blank."""
    h = holders_by_number.get(rid)
    if h is None:
        return "", "no current holder of this district number found on Wikipedia"
    if h["vacant"]:
        return h["party"], f"seat vacant ({h['status']})"
    if h["listed_in"] == rid:
        return h["party"], ""
    movers = [m for m in members if m["from_district"] is not None]
    moved_in = ("; incumbent(s) redistricted in: " + ", ".join(
        f"{m['name']} ({m['party']}, from {rid[:2]}-{m['from_district']:02d})" for m in movers)) if movers else ""
    return h["party"], (f"district number now held by {h['name']} ({h['party']}), listed by Wikipedia under "
                        f"{h['listed_in']} on the new lines ({h['status']})" + moved_in)


def build_house(races: dict[str, dict]) -> list[dict]:
    rows = []
    holders_by_number = old_number_holders(races)
    for rid, race in sorted(races.items()):
        state = rid[:2]
        seat_party, seat_note = seat_holder(rid, race["members"], holders_by_number)
        live_members = [m for m in race["members"] if not m["vacant"]]
        race_rows = []
        for c in race["candidates"]:
            inc = next((m for m in live_members if names_match(m["name"], c["name"])), None)
            diff = int(inc is not None and inc["from_district"] is not None and
                       f"{state}-{inc['from_district']:02d}" != rid)
            notes = []
            if inc is not None and diff:
                notes.append(f"incumbent of {state}-{inc['from_district']:02d} (old lines)")
            race_rows.append(_row(rid, "House", c, inc is not None, seat_party, diff, notes))
        statuses = "; ".join(f"{m['name'] or 'Vacant'}: {m['status']}" for m in race["members"])
        rows.extend(_finish_race(rid, state, race_rows, special=False,
                                 race_note="; ".join(x for x in [seat_note, statuses] if x)))
    return rows


# --------------------------------------------------------------------------------------------- Senate
def senate_races(tables: list[dict]) -> dict[str, dict]:
    races: dict[str, dict] = {}
    for t in tables:
        if not t["rows"] or t["rows"][0][-1] != "Candidates" or t["rows"][1][0] != "State":
            continue
        for row in t["rows"][2:]:
            state_name = row[0].split("\n")[0].strip()
            if state_name not in STATES:
                continue
            rid = f"{STATES[state_name]}-SEN"
            status = row[-2].replace("\n", " ")
            races[rid] = {"incumbent": row[2].split("\n")[0].strip(), "party": map_party(row[3]),
                          "history": row[4].replace("\n", " "), "status": status,
                          "candidates": parse_candidates(row[-1])}
    return races


def build_senate(races: dict[str, dict]) -> list[dict]:
    rows = []
    for rid, race in sorted(races.items()):
        race_rows = []
        for c in race["candidates"]:
            is_inc = INCUMBENT_OVERRIDES.get((rid, c["name"]), names_match(race["incumbent"], c["name"]))
            race_rows.append(_row(rid, "Senate", c, is_inc, race["party"], 0, []))
        note = f"Senate incumbent {race['incumbent']} ({race['party']}, {race['history']}): {race['status']}"
        rows.extend(_finish_race(rid, rid[:2], race_rows, special=rid in SPECIAL_SENATE, race_note=note))
    return rows


# --------------------------------------------------------------------------------------------- shared row logic
def _row(rid: str, chamber: str, cand: dict, is_inc: bool, seat_party: str, diff: int, notes: list[str]) -> dict:
    party = map_party(cand["party_label"])
    extra = [f"party label: {cand['party_label']}"] if party == "OTHER" or party == "IND" and \
        cand["party_label"].lower() != "independent" else []
    return {"race_id": rid, "chamber": chamber, "candidate_name": cand["name"], "party": party,
            "is_incumbent": int(bool(is_inc)), "incumbent_party_of_seat": seat_party,
            "incumbent_running_in_different_district": diff, "_notes": extra + notes}


def _pick_majors(rid: str, rows: list[dict], system: str) -> tuple[set[int], str]:
    """Indices of the (up to) two main contenders, plus a note when the choice is ambiguous."""
    if system == "top-two":
        return set(range(len(rows))), ""
    picks, notes = set(), []
    for party in MAJOR_PARTY_ORDER:
        if (rid, party) in NOT_MAJOR:
            continue
        idx = [i for i, r in enumerate(rows) if r["party"] == party]
        inc = [i for i in idx if rows[i]["is_incumbent"]]
        if len(idx) == 1 or inc:
            picks.add(inc[0] if inc else idx[0])
        elif len(idx) > 1:
            notes.append(f"{len(idx)} {party} candidates on the ballot; is_major not assigned among them")
    picks |= {i for i, r in enumerate(rows) if (rid, r["candidate_name"]) in MAJOR_INDEPENDENTS}
    for party in MINOR_PRIORITY:
        if len(picks) >= 2 or notes:
            break
        idx = [i for i, r in enumerate(rows) if r["party"] == party and i not in picks]
        if len(idx) == 1:
            picks.add(idx[0])
        elif len(idx) > 1:
            notes.append(f"second contender ambiguous among {len(idx)} {party} candidates; none flagged")
            break
    return picks, "; ".join(notes)


def _finish_race(rid: str, state: str, rows: list[dict], special: bool, race_note: str) -> list[dict]:
    system = system_for(state)
    if state == "LA" and rid != "LA-SEN":
        system = "runoff-possible"   # 2026 LA House: jungle primary on Nov 3, runoff Dec 12 (new map)
    if not rows:
        return [{"race_id": rid, "chamber": "Senate" if rid.endswith("SEN") else "House", "candidate_name": "",
                 "party": "", "is_incumbent": 0, "incumbent_party_of_seat": "", "open_seat": "",
                 "incumbent_running_in_different_district": 0, "is_major": 0, "special": int(special),
                 "system": system, "uncontested": "", "kalshi_name_match": 0,
                 "notes": "NO CANDIDATES PARSED; " + race_note}]
    majors, major_note = _pick_majors(rid, rows, system)
    open_seat = int(not any(r["is_incumbent"] for r in rows))
    uncontested = int(len(rows) == 1)
    has_d, has_r = any(r["party"] == "DEM" for r in rows), any(r["party"] == "REP" for r in rows)
    race_flags = [] if uncontested or (has_d and has_r) else ["no " + ("DEM" if not has_d else "REP") + " nominee"]
    if uncontested:
        race_flags.append("only one candidate listed (unopposed)")
    race_flags.append(SYSTEM_NOTES.get("LA-House" if state == "LA" and rid != "LA-SEN" else state, ""))
    out = []
    for i, r in enumerate(rows):
        notes = r.pop("_notes") + race_flags + ([major_note] if major_note else []) + [race_note]
        out.append({**r, "open_seat": open_seat, "is_major": int(i in majors), "special": int(special),
                    "system": system, "uncontested": uncontested, "notes": " | ".join(n for n in notes if n)})
    return out


# --------------------------------------------------------------------------------------------- Kalshi cross-check
def kalshi_names(path: Path) -> dict[str, list[str]]:
    """race_id -> person names Kalshi lists (party-only entries such as 'Democratic party' dropped)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for rid, cands in zip(df["race_id"], df["candidates"].fillna("")):
        names = [p.split(":")[1].strip() for p in cands.split(" | ") if p.count(":") >= 2]
        out[rid] = [n for n in names if not n.lower().endswith(" party")]
    return out


def _last_name_close(a: str, b: str) -> bool:
    """Same surname, allowing one substituted letter (Rodriquez/Rodriguez)."""
    return a == b or (len(a) == len(b) > 4 and sum(x != y for x, y in zip(a, b)) == 1)


def kalshi_match(name: str, k_names: list[str]) -> bool:
    """Exact/first+last match, else a surname match that is unique among Kalshi's names for the race
    (covers nicknames like Tony/Anthony, Gene/Eugene and reordered given names)."""
    if any(names_match(name, k) for k in k_names):
        return True
    toks = norm_name(name).split()
    if not toks:
        return False
    hits = [k for k in k_names if norm_name(k).split() and _last_name_close(norm_name(k).split()[-1], toks[-1])]
    return len(hits) == 1


def add_kalshi(rows: list[dict], kal: dict[str, list[str]]) -> tuple[list[dict], list[str]]:
    out, issues = [], []
    by_race: dict[str, list[dict]] = {}
    for r in rows:
        k_names = kal.get(r["race_id"], [])
        hit = int(bool(r["candidate_name"]) and kalshi_match(r["candidate_name"], k_names))
        by_race.setdefault(r["race_id"], []).append({**r, "kalshi_name_match": hit})
    for rid, rr in by_race.items():
        k_names = kal.get(rid, [])
        missing = [k for k in k_names if not any(kalshi_match(r["candidate_name"], [k]) for r in rr)]
        extra = []
        if not k_names:
            extra.append("Kalshi lists parties only (no names) for this race")
        if missing:
            extra.append("Kalshi names not on Wikipedia ballot list: " + ", ".join(missing))
            issues.append(f"{rid}: {', '.join(missing)}")
        out.extend({**r, "notes": " | ".join([x for x in [r["notes"]] + extra if x])} for r in rr)
    return out, issues


# --------------------------------------------------------------------------------------------- redistricting
# Mechanism + effective date for states whose 2026 lines differ from 2024 (from the Wikipedia summary table
# "Out-of-cycle partisan redistricting efforts" and the per-state 2026 House pages, cached in data/raw/candidates).
NEW_MAPS = {
    "AL": ("legislature", "2026-05", "Legislature redrew after Louisiana v. Callais (Apr 2026); federal three-judge panel blocked "
           "it May 26, SCOTUS stayed the block June 2, 2026; special primaries Aug 11 in affected districts. 2024 "
           "used the court-drawn remedial map. Second source (NPR, 2026-06-02) describes the lines as the "
           "legislature's 2023 plan, which SCOTUS let Alabama use; Wikipedia calls them the May 2026 lines."),
    "CA": ("ballot measure", "2025-11-04", "Prop 50 (Election Rigging Response Act) legislative map approved by "
           "voters ~64%; replaces commission map for 2026-2030."),
    "FL": ("legislature", "2026-05-04", "Special session map passed Apr 29, signed by DeSantis May 4, 2026."),
    "LA": ("legislature", "2026-05-29", "SB 121 signed May 29, 2026 after Callais; removes one of two Black-plurality "
           "Democratic districts. Party primaries cancelled: Nov 3 is a jungle primary, runoff Dec 12."),
    "NC": ("legislature", "2025-10-22", "Mid-decade map making NC-01 more Republican; governor has no veto."),
    "OH": ("commission", "2025-10-31", "Ohio Redistricting Commission unanimous compromise map (required re-draw "
           "because 2021 map lacked bipartisan support)."),
    "TN": ("legislature", "2026-05-07", "HB 7003 signed May 7, 2026; splits Memphis (TN-09), aims 9-0 R."),
    "TX": ("legislature", "2025-08-29", "Signed Aug 29, 2025; district court enjoined Nov 18, SCOTUS allowed map "
           "Dec 4, 2025 (6-3)."),
    "UT": ("court", "2025-11-10", "Judge Gibson rejected legislature's map under Prop 4 and adopted plaintiffs' Map 1 "
           "(creates a Salt Lake County Democratic seat)."),
}
ATTEMPTED_NOT_USED = {
    "MO": "2025 legislative map (new safe R KC seat) signed Sep 2025 but suspended by referendum petition; Missouri "
          "Supreme Court and SCOTUS (Sep 25, 2026) left the 2022 map in place for 2026.",
    "VA": "Constitutional amendment/referendum (passed Apr 21, 2026) invalidated by Virginia Supreme Court May 8, 2026; "
          "2024 map used.",
    "IN": "Indiana Senate rejected mid-decade redraw (Dec 2025); 2024 map used.",
    "MD": "House-passed redraw died in the Senate (Feb 2026); 2024 map used.",
    "KS": "Special session to redraw KS-03 lacked votes; 2024 map used.",
    "SC": "House-passed post-Callais map not advanced by Senate (May 2026); 2024 map used.",
    "NY": "Court-ordered NY-11 redraw blocked by SCOTUS (Mar 2, 2026); 2024 map used.",
    "WI": "Challenge dismissed Mar 31, 2026; 2024 map used.",
    "AR": "Racial-gerrymander suit dismissed; 2024 map used.",
    "GA": "Kemp called a June 2026 special session after Callais, but legislative leaders dropped congressional "
          "redistricting for 2026 (WABE/Bloomberg Government); 2024 map used.",
}


def wiki_redistricting_status(tables: list[dict]) -> dict[str, str]:
    """State abbr -> status text from the House page's mid-decade redistricting summary table."""
    for t in tables:
        if len(t["rows"]) > 2 and t["rows"][0][:2] == ["State\n(linked to\nsummaries below)", "Status"]:
            return {STATES[r[0]]: r[1] for r in t["rows"][2:] if r[0] in STATES}
    return {}


def build_redistricting(template: pd.DataFrame, wiki_status: dict[str, str]) -> pd.DataFrame:
    house = template[template["chamber"] == "House"]
    counts = house["race_id"].str[:2].value_counts()
    abbr_to_name = {v: k for k, v in STATES.items()}
    rows = []
    for st in sorted(counts[counts > 1].index):
        how, date, note = NEW_MAPS.get(st, ("", "", ATTEMPTED_NOT_USED.get(st, "2024 map used (no mid-decade change).")))
        rows.append({"state": st, "state_name": abbr_to_name[st], "n_districts": int(counts[st]),
                     "new_map_for_2026": int(st in NEW_MAPS), "how": how,
                     "wiki_status": wiki_status.get(st, "not discussed (no mid-decade effort listed)"), "date": date, "notes": note})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- main
COLUMNS = ["race_id", "chamber", "candidate_name", "party", "is_incumbent", "incumbent_party_of_seat", "open_seat",
           "incumbent_running_in_different_district", "is_major", "special", "system", "uncontested",
           "kalshi_name_match", "notes"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-download Wikipedia pages")
    args = ap.parse_args()

    template = pd.read_csv(TEMPLATE)
    house_page = fetch_page(HOUSE_PAGE, RAW_DIR, refresh=args.refresh)
    senate_page = fetch_page(SENATE_PAGE, RAW_DIR, refresh=args.refresh)
    if house_page is None or senate_page is None:
        raise SystemExit("Wikipedia page missing")

    house_tables = parse_tables(house_page["html"])
    rows = build_house(house_races(house_tables)) + \
        build_senate(senate_races(parse_tables(senate_page["html"])))
    rows, issues = add_kalshi(rows, kalshi_names(KALSHI_RACES))
    df = pd.DataFrame(rows)

    want = set(template["race_id"])
    have = set(df["race_id"])
    extra = sorted(have - want)
    df = df[df["race_id"].isin(want)]
    missing = sorted(want - have)
    if missing:
        filler = pd.DataFrame([{"race_id": r, "chamber": template.set_index("race_id").loc[r, "chamber"],
                                "notes": "NOT FOUND on Wikipedia page"} for r in missing])
        df = pd.concat([df, filler], ignore_index=True)
    order = {r: i for i, r in enumerate(template["race_id"])}
    df = df.assign(_o=df["race_id"].map(order)).sort_values(["_o", "is_major", "candidate_name"],
                                                             ascending=[True, False, True])
    df = df.drop(columns="_o")[COLUMNS]

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DIR / "candidates_2026.csv", index=False)
    wiki_status = wiki_redistricting_status(house_tables)
    enacted = {st for st, v in wiki_status.items() if v.startswith("New districts enacted")}
    if enacted != set(NEW_MAPS):
        print("WARNING: NEW_MAPS disagrees with Wikipedia summary table:", sorted(enacted ^ set(NEW_MAPS)))
    red = build_redistricting(template, wiki_status)
    red.to_csv(CLEAN_DIR / "redistricting_2026.csv", index=False)

    print(f"house revid {house_page['revid']} fetched {house_page['fetched_at']}; "
          f"senate revid {senate_page['revid']} fetched {senate_page['fetched_at']}")
    print(f"rows {len(df)}, races {df['race_id'].nunique()}/{len(want)}, missing {missing}, dropped {extra}")
    print(f"kalshi names not found ({len(issues)}):", *issues, sep="\n  ")
    print(f"redistricting rows {len(red)}, new maps {int(red['new_map_for_2026'].sum())}")


if __name__ == "__main__":
    main()
