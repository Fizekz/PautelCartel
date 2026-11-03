"""Build clean historical U.S. House and Senate general-election results (2000-2024).

Source: MIT Election Data + Science Lab (MEDSL), Harvard Dataverse
  - U.S. House 1976-2024           doi:10.7910/DVN/IG0UN2  (file 1976-2024-house.tab, v15)
  - U.S. Senate statewide 1976-2024 doi:10.7910/DVN/PEJ5QU (file 1976-2024-senate-state.tab, v8)
Senate special-election class: Wikipedia "List of special elections to the United States Senate".
Races MEDSL omits (CT Senate 2022): official state canvass counts in SENATE_SUPPLEMENT, cross-checked
against saved Wikipedia wikitext.

The House file sits behind a Dataverse guestbook (name/email/institution/position). The script tries
Dataverse first, then falls back to a public GitHub mirror, and accepts a file only when its MD5 equals
the checksum Dataverse publishes for the current version, so the mirror copy is byte-identical.

Usage:
  .venv/bin/python src/data/fetch_results.py            # download if missing, then build
  .venv/bin/python src/data/fetch_results.py --offline  # build from data/raw/mit only

Outputs: data/clean/results_house.csv, data/clean/results_senate.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "raw" / "mit"
CLEAN_DIR = REPO / "data" / "clean"
FIRST_YEAR = 2000
USER_AGENT = "ModelCitizensResearch/1.0 (election forecasting contest)"
DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{fid}?format=original"

RAW_FILES = {
    "1976-2024-house.tab": {
        "fid": 13592823,
        "md5": "a9566ac393913a1af343284e837c17bb",
        "mirrors": [
            "https://raw.githubusercontent.com/sta199-f26/sta199-f26.github.io/main/_data/gerrymander/1976-2024-house.tab",
            "https://raw.githubusercontent.com/Hackquantumcpp/snoutcounter-midterms-2026-model/main/data/1976-2024-house.tab",
        ],
    },
    "1976-2024-senate-state.csv": {
        "fid": 13887039,
        "md5": "0f8b51cf0f77a2ef0bff992f7a64d1b4",
        "mirrors": [],
    },
}
WIKI_SPECIALS = "List_of_special_elections_to_the_United_States_Senate"
WIKI_API = "https://en.wikipedia.org/w/api.php"

DEM_LABELS = {
    "DEMOCRAT", "DEMOCRATIC", "DEMOCRATIC-FARMER-LABOR", "DEMOCRATIC-FARM-LABOR",
    "DEMOCRATIC-NONPARTISAN LEAGUE", "DEMOCRATIC-NPL", "DEMOCRAT (NOT IDENTIFIED ON BALLOT)",
    "DEMOCRATIC/WORKING FAMILIES", "WRITE-IN (DEMOCRATIC)",
}
REP_LABELS = {
    "REPUBLICAN", "GOP", "REPUBLICAN (NOT IDENTIFIED ON BALLOT)", "REPUBLICAN/CONSERVATIVE",
    "REPUBLICAN, LIBERTARIAN", "WRITE-IN (REPUBLICAN)",
}
# Non-candidate rows. INVALID ones (blank/over/under/void/exhausted) are also dropped from total_votes.
PLACEHOLDER_RE = re.compile(
    r"^(WRITEIN|WRITE-IN|WRITE-IN \(MISCELLANEOUS\)|OTHER WRITE-INS|OTHERS?|OTHER CANDIDATES|ALL OTHERS"
    r"|SCATTER(ING)?|NONE OF TH.*|BLANK.*|VOID( VOTE)?|SPOILED|OVER ?VOTES?|UNDER ?VOTES?"
    r"|EXHAUSTED BALLOTS?)$"
)
INVALID_RE = re.compile(r"^(BLANK.*|VOID( VOTE)?|SPOILED|OVER ?VOTES?|UNDER ?VOTES?|EXHAUSTED BALLOTS?)$")
TOP_TWO_STATES = {"CA", "WA"}
RCV_RACES = {"ME": 2018, "AK": 2022}  # state -> first general election using RCV (federal races)
MAJOR_THIRD_SHARE = 0.15
REGULAR_CLASS_BY_MOD6 = {2: 1, 4: 2, 0: 3}  # 2000->I, 2002->II, 2004->III
SENATE_CLASS_SIZES = {1: 33, 2: 33, 3: 34}

# TX 2006: court-ordered (LULAC v. Perry) open elections in redrawn districts, flagged special by MEDSL
# but they were the regular elections for the full term. TX-23 went to a Dec runoff, which MEDSL
# stores as the special=False rows.
TX2006_REDRAWN = {15, 21, 23, 25, 28}
# Candidate party overrides: (year, state, district/0, candidate) -> party, note
# (MEDSL leaves these parties blank or uses a nonpartisan ballot label.) Overridden candidates are also
# treated as on-ballot for the dem/rep columns, so a strong write-in does not make a race "uncontested".
PARTY_OVERRIDES = {
    (2010, "AK", 0, "LISA MURKOWSKI"): ("REP", "Murkowski won as a write-in after losing the GOP primary; coded REP"),
    (2020, "WY", 0, "CYNTHIA M. LUMMIS"): ("REP", "MEDSL party blank; Lummis coded REP"),
    (2020, "WY", 0, "MERAV BEN DAVID"): ("DEM", "MEDSL party blank; Ben David coded DEM"),
    (2022, "AK", 0, "LISA MURKOWSKI"): ("REP", "MEDSL party blank for AK 2022; Murkowski coded REP"),
    (2022, "AK", 0, "KELLY C. TSHIBAKA"): ("REP", "MEDSL party blank for AK 2022; Tshibaka coded REP"),
    (2022, "AK", 0, "PATRICIA R. CHESBRO"): ("DEM", "MEDSL party blank for AK 2022; Chesbro coded DEM"),
    (2022, "AK", 0, "BUZZ A. KELLEY"): ("REP", "MEDSL party blank for AK 2022; Kelley coded REP"),
    (2008, "MS", 0, "RONNIE MUSGROVE"): ("DEM", "MS special ballot is nonpartisan; Musgrove (D) coded DEM"),
    (2002, "VA", 9, "J KATZEN"): ("REP", "MEDSL party blank/write-in flag; Jay Katzen was the GOP nominee"),
    (2006, "TX", 22, "SHELLEY SEKULA GIBBS"): ("REP", "GOP write-in candidate (DeLay withdrew); coded REP"),
}
PARTY_LINE_PSEUDO_CANDIDATES = {"WORKING FAMILIES": "DEM"}  # CT 2022-24: WFP line votes w/o candidate name
# Louisiana Senate runoffs: MEDSL mixes Nov (jungle) and Dec (runoff) rows under stage "gen".
LA_SENATE_RUNOFFS = {
    2002: {"finalists": ["MARY L. LANDRIEU", "SUZANNE HAIK TERRELL"], "mode": "max_row", "fix": 0},
    2014: {"finalists": ["MARY L. LANDRIEU", "BILL CASSIDY"], "mode": "as_is", "fix": -1_000_000},
    2016: {"finalists": ["JOHN KENNEDY", "FOSTER CAMPBELL"], "mode": "as_is", "fix": 0},
}
# Free-text notes for races whose MEDSL result needs context: (year, race_id, special) -> note
RACE_NOTES = {
    (2018, "NC-09", 0): "result never certified (absentee-ballot fraud); seat vacant, 2019 special won by Dan Bishop (R)",
    (2000, "VA-05", 0): "Goode elected as Independent (ex-D), caucused R, joined GOP 2002",
    (2006, "CT-SEN", 0): "Lieberman lost D primary, won as Connecticut for Lieberman; caucused D",
    (2010, "IL-SEN", 0): ("MEDSL flags this race special, but its counts are the regular full-term election, so "
                          "special=0; the same-day special for the rest of Obama's term (Kirk 1,677,729 vs "
                          "Giannoulias 1,641,486) is not in MEDSL"),
    (2022, "CT-SEN", 0): ("race missing from MEDSL; votes added from the CT Secretary of the State canvass "
                          "(electionhistory.ct.gov contest 34715), cross-checked with Wikipedia"),
}
DEM_CAUCUS_INDEPENDENTS = {"BERNARD SANDERS", "BERNIE SANDERS"}
# Senate races where MEDSL holds only the runoff (finalists' runoff votes; Nov jungle round absent).
SENATE_RUNOFF_ONLY = {(2018, "MS", True)}
# LA House jungle races with no Nov majority: MEDSL has only the Nov round. Known Dec runoff winners
# (checked against the Nov plurality leader; the build fails loudly if they disagree).
LA_HOUSE_RUNOFF_WINNERS = {
    (2006, 2): "WILLIAM J JEFFERSON", (2012, 3): "CHARLES W BOUSTANY JR", (2014, 5): "RALPH LEE ABRAHAM",
    (2014, 6): "GARRET GRAVES", (2016, 3): "CLAY HIGGINS", (2016, 4): "\"MIKE\" JOHNSON",
    (2020, 5): "LUKE J LETLOW",
}
# ---- verifier fixes (2026-11-01) ----
# IL 2010: MEDSL has one IL race, flagged special=True, but its counts (Kirk 1,778,698 / Giannoulias
# 1,719,478, total 3,704,473) are the REGULAR full-term election. The same-day special for the
# remainder of Obama's term (Kirk 1,677,729 / Giannoulias 1,641,486) is not in MEDSL.
SENATE_SPECIAL_FLAG_FIXES = {(2010, "IL"): False}
# Senate state-years whose MEDSL names are "LAST FIRST" with no comma (comma form is handled in clean_name).
SENATE_LAST_FIRST_NO_COMMA = {(2022, "OR")}
# Races MEDSL omits entirely. Votes from the official state canvass (cited), cross-checked against the
# Wikipedia wikitext saved under data/raw/mit/wikipedia/ (the build asserts every count appears there).
SENATE_SUPPLEMENT = {
    (2022, "CT"): {
        "wiki": "2022_United_States_Senate_election_in_Connecticut",
        "source": "CT Secretary of the State, electionhistory.ct.gov contest 34715",
        "rows": [  # (candidate, party_detailed, writein, votes)
            ("RICHARD BLUMENTHAL", "DEMOCRAT", False, 701175),
            ("RICHARD BLUMENTHAL", "WORKING FAMILIES", False, 22689),
            ("LEORA R. LEVY", "REPUBLICAN", False, 535943),
            ("WRITEIN", None, True, 80),
        ],
    },
}


# ----------------------------------------------------------------------------- download
def md5sum(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch_bytes(url: str) -> bytes | None:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    except requests.RequestException as exc:
        print(f"  ! {url}: {exc}", file=sys.stderr)
        return None
    if resp.status_code != 200 or resp.content[:1] == b"{":  # Dataverse returns JSON on guestbook errors
        print(f"  ! {url}: HTTP {resp.status_code} {resp.content[:120]!r}", file=sys.stderr)
        return None
    return resp.content


def download_raw(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in RAW_FILES.items():
        dest = raw_dir / name
        if dest.exists() and md5sum(dest) == spec["md5"]:
            print(f"  ok {name} (cached, md5 verified)")
            continue
        for url in [DATAVERSE.format(fid=spec["fid"]), *spec["mirrors"]]:
            content = fetch_bytes(url)
            if content is None:
                continue
            got = hashlib.md5(content).hexdigest()
            if got != spec["md5"]:
                print(f"  ! {url}: md5 {got} != {spec['md5']}", file=sys.stderr)
                continue
            dest.write_bytes(content)
            print(f"  ok {name} <- {url}")
            break
        else:
            raise RuntimeError(f"could not obtain {name} with the expected checksum")
    pages = {"senate_specials.json": WIKI_SPECIALS,
             **{f"{spec['wiki']}.json": spec["wiki"] for spec in SENATE_SUPPLEMENT.values()}}
    for fname, page in pages.items():
        wiki = raw_dir / "wikipedia" / fname
        if wiki.exists():
            continue
        wiki.parent.mkdir(parents=True, exist_ok=True)
        params = {"action": "parse", "page": page, "prop": "wikitext|revid",
                  "format": "json", "formatversion": 2}
        resp = requests.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        wiki.write_bytes(resp.content)
        print(f"  ok {wiki.name} <- Wikipedia")


# ----------------------------------------------------------------------------- normalize
def party_code(label) -> str:
    if not isinstance(label, str):
        return "OTHER"
    if label in DEM_LABELS:
        return "DEM"
    if label in REP_LABELS:
        return "REP"
    return "OTHER"


def clean_name(name) -> str:
    if not isinstance(name, str):
        return "WRITEIN"
    if re.search(r"[ÃÂâ]", name):  # UTF-8 bytes mis-decoded as Latin-1 in some 2022-2024 House rows
        try:
            name = name.encode("latin-1").decode("utf-8")
        except UnicodeError:
            pass
    name = re.sub(r"\s+", " ", name.replace('\\"', '"')).strip().upper()  # MEDSL escapes quotes as \"
    flipped = re.match(r"^([^,]+), ([^,]+)$", name)  # AK/HI/NV 2022 Senate use "LAST, FIRST"
    if flipped and not name.startswith("NONE") and not re.match(r"^(JR|SR|I{2,3}|IV)\.?$", flipped.group(2)):
        name = f"{flipped.group(2)} {flipped.group(1)}"
    return name


def finish_long(df: pd.DataFrame) -> pd.DataFrame:
    out = df.assign(
        candidate=df["candidate"].map(clean_name),
        party=df["party_raw"].map(party_code),
        votes=pd.to_numeric(df["votes"], errors="coerce").fillna(0),
    )
    return out.assign(
        is_placeholder=out["candidate"].str.match(PLACEHOLDER_RE),
        is_invalid=out["candidate"].str.match(INVALID_RE),
    )


def load_house(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[(raw["year"] >= FIRST_YEAR) & (raw["stage"] == "GEN") & (raw["state_po"] != "DC")]
    redrawn = (raw["year"] == 2006) & (raw["state_po"] == "TX") & raw["district"].isin(TX2006_REDRAWN)
    tx23_runoff = redrawn & (raw["district"] == 23) & ~raw["special"]
    is_runoff = (raw["runoff"] == True) | tx23_runoff  # noqa: E712 (column holds True/False/NaN)
    df = pd.DataFrame({
        "year": raw["year"], "state": raw["state_po"], "district": raw["district"].astype(int),
        "special": raw["special"] & ~redrawn, "candidate": raw["candidate"], "party_raw": raw["party"],
        "writein": raw["writein"].astype(bool), "votes": raw["candidatevotes"],
        "unofficial": raw["unofficial"].astype(bool), "fusion": raw["fusion_ticket"].astype(bool),
        "round": is_runoff.map({True: "runoff", False: "nov"}),
        "court_ordered": redrawn,
    })
    return finish_long(df)


def senate_round(raw: pd.DataFrame) -> pd.Series:
    stage = raw["stage"].str.upper()
    key = raw["year"].astype(str) + raw["state_po"] + raw["special"].astype(str)
    has_pre = key.isin(key[stage == "PRE"])  # GA 2008: Nov round coded "pre", Dec runoff coded "gen"
    is_runoff = stage.isin(["RUNOFF", "GEN RUNOFF"]) | ((stage == "GEN") & has_pre)
    return is_runoff.map({True: "runoff", False: "nov"})


def senate_party(raw: pd.DataFrame) -> pd.Series:
    """party_detailed is blank for many 2020-2024 rows; fall back to party_simplified."""
    simple = raw["party_simplified"].where(raw["party_simplified"] != "OTHER")
    return raw["party_detailed"].fillna(simple)


def senate_supplement(columns: pd.Index, wiki_dir: Path) -> pd.DataFrame:
    """Rows for races MEDSL omits (SENATE_SUPPLEMENT), in MEDSL raw format."""
    recs = []
    for (year, state), spec in SENATE_SUPPLEMENT.items():
        wiki = wiki_dir / f"{spec['wiki']}.json"
        if wiki.exists():  # cross-check every count against the saved second source
            text = json.loads(wiki.read_text())["parse"]["wikitext"]
            for cand, _, _, votes in spec["rows"]:
                if f"{votes:,}" not in text and f"={votes}" not in text:
                    raise ValueError(f"{year} {state} {cand}: {votes:,} not found in {wiki.name}")
        else:
            print(f"  ! {wiki.name} missing; {year} {state} supplement not cross-checked", file=sys.stderr)
        total = sum(r[3] for r in spec["rows"])
        for cand, party, writein, votes in spec["rows"]:
            recs.append({"year": year, "state_po": state, "stage": "GEN", "special": False,
                         "candidate": cand, "party_detailed": party, "writein": writein,
                         "candidatevotes": votes, "totalvotes": total, "unofficial": False,
                         "party_simplified": party if party in ("DEMOCRAT", "REPUBLICAN") else "OTHER"})
    return pd.DataFrame(recs).reindex(columns=columns)


def flip_last_first(raw: pd.DataFrame) -> pd.Series:
    """'WYDEN RON' -> 'RON WYDEN' for the state-years in SENATE_LAST_FIRST_NO_COMMA."""
    key = pd.Series(list(zip(raw["year"], raw["state_po"])), index=raw.index)
    flip = key.isin(SENATE_LAST_FIRST_NO_COMMA) & raw["candidate"].notna() & \
        ~raw["candidate"].fillna("").str.upper().str.match(PLACEHOLDER_RE)
    flipped = raw["candidate"].str.split(n=1).map(lambda p: f"{p[1]} {p[0]}" if len(p) == 2 else " ".join(p),
                                                 na_action="ignore")
    return raw["candidate"].where(~flip, flipped)


def load_senate(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["year"] >= FIRST_YEAR].copy()
    missing = [k for k in SENATE_SUPPLEMENT if not ((raw["year"] == k[0]) & (raw["state_po"] == k[1])).any()]
    if missing != list(SENATE_SUPPLEMENT):
        raise ValueError(f"MEDSL now contains a supplemented race; drop it from SENATE_SUPPLEMENT: "
                         f"{set(SENATE_SUPPLEMENT) - set(missing)}")
    raw = pd.concat([raw, senate_supplement(raw.columns, path.parent / "wikipedia")], ignore_index=True)
    raw["special"] = raw["special"].astype(bool)
    for (year, state), flag in SENATE_SPECIAL_FLAG_FIXES.items():
        raw.loc[(raw["year"] == year) & (raw["state_po"] == state), "special"] = flag
    raw["candidate"] = flip_last_first(raw)
    raw["unofficial"] = raw["unofficial"].fillna(False)
    rnd = senate_round(raw)
    year = raw["year"].where(raw["year"] % 2 == 0, raw["year"] - 1)  # GA Jan-2021 runoffs -> 2020 cycle
    df = pd.DataFrame({
        "year": year, "state": raw["state_po"], "district": 0, "special": raw["special"].astype(bool),
        "candidate": raw["candidate"], "party_raw": senate_party(raw), "writein": raw["writein"].astype(bool),
        "votes": raw["candidatevotes"], "unofficial": raw["unofficial"].astype(bool), "fusion": False,
        "round": rnd, "court_ordered": False,
    })
    df = fix_la_senate(finish_long(df))
    runoff_only = pd.Series(list(zip(df["year"], df["state"], df["special"])), index=df.index).isin(
        SENATE_RUNOFF_ONLY)
    return df.assign(round=df["round"].where(~runoff_only, "runoff"))


def fix_la_senate(df: pd.DataFrame) -> pd.DataFrame:
    """Split LA Senate rows into Nov jungle vs Dec runoff (see LA_SENATE_RUNOFFS)."""
    parts = [df[~((df["state"] == "LA") & df["year"].isin(list(LA_SENATE_RUNOFFS)))]]
    for year, spec in LA_SENATE_RUNOFFS.items():
        race = df[(df["state"] == "LA") & (df["year"] == year)]
        fin = race["candidate"].isin(spec["finalists"])
        if spec["mode"] == "max_row":
            is_dec = fin & (race["votes"] == race.groupby("candidate")["votes"].transform("max"))
        else:
            is_dec = fin
        votes = race["votes"].where(~is_dec, race["votes"] + spec["fix"])
        # with as_is, the finalists' Nov rows are absent, so the remaining Nov rows are incomplete
        rnd = is_dec.map({True: "runoff", False: "nov" if spec["mode"] == "max_row" else "nov_partial"})
        parts.append(race.assign(votes=votes, round=rnd))
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- race summary
def attach_party_lines(real: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Give unnamed party-line rows (e.g. CT 'WORKING FAMILIES') to that party's cross-endorsed nominee."""
    pseudo = real["candidate"].isin(list(PARTY_LINE_PSEUDO_CANDIDATES)) & real["party_raw"].isna()
    mapping = {}
    for idx in real.index[pseudo]:
        line = real.at[idx, "candidate"]
        nominees = real[(real["party"] == PARTY_LINE_PSEUDO_CANDIDATES[line]) & ~pseudo]
        if nominees.empty:
            continue
        mapping[idx] = nominees.groupby("candidate")["votes"].sum().idxmax()
        notes.append(f"unnamed {line} line votes attributed to {mapping[idx]}")
    if not mapping:
        return real
    return real.assign(candidate=pd.Series(mapping).reindex(real.index).fillna(real["candidate"]))


def candidate_table(rows: pd.DataFrame, key: tuple) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    real = attach_party_lines(rows[~rows["is_placeholder"]], notes)
    recs = []
    for name, grp in real.groupby("candidate", sort=False):
        by_party = grp.groupby("party")["votes"].sum()
        major = by_party.drop("OTHER", errors="ignore")
        party = major.idxmax() if len(major) else "OTHER"
        writein = bool(grp["writein"].all())
        override = PARTY_OVERRIDES.get((*key, name))
        if override:
            party, writein = override[0], False
            notes.append(override[1])
        n_lines = grp["party_raw"].nunique(dropna=False)
        if n_lines > 1:
            notes.append(f"fusion: {name} on {n_lines} lines, votes summed to {party}")
        recs.append({"candidate": name, "party": party, "votes": grp["votes"].sum(), "writein": writein})
    cands = pd.DataFrame(recs, columns=["candidate", "party", "votes", "writein"])
    return cands.sort_values("votes", ascending=False).reset_index(drop=True), notes


def dem_share(dem: float, rep: float) -> float | None:
    return dem / (dem + rep) if (dem + rep) > 0 else None


def party_block(cands: pd.DataFrame, party: str, winner: str) -> tuple[str, float, int]:
    on_ballot = cands[(cands["party"] == party) & (~cands["writein"] | (cands["candidate"] == winner))]
    if on_ballot.empty:
        return "", 0.0, 0
    return on_ballot.iloc[0]["candidate"], float(on_ballot["votes"].sum()), len(on_ballot)


def summarize(rows: pd.DataFrame, key: tuple) -> dict:
    cands, notes = candidate_table(rows, key)
    if cands.empty:
        return {"notes": "no candidate rows"}
    winner = cands.iloc[0]
    dem_cand, dem_votes, n_dem = party_block(cands, "DEM", winner["candidate"])
    rep_cand, rep_votes, n_rep = party_block(cands, "REP", winner["candidate"])
    total = float(rows.loc[~rows["is_invalid"], "votes"].sum())
    out = {
        "dem_cand": dem_cand, "rep_cand": rep_cand, "dem_votes": dem_votes, "rep_votes": rep_votes,
        "total_votes": total, "dem_share_2p": dem_share(dem_votes, rep_votes),
        "winner_party": winner["party"], "winner_name": winner["candidate"],
        "uncontested": int(n_dem == 0 or n_rep == 0), "winner_share": winner["votes"] / total if total else None,
        "top2_same_party": len(cands) >= 2 and cands.iloc[1]["party"] == winner["party"] != "OTHER",
    }
    if n_dem > 1 or n_rep > 1:
        notes.append(f"multiple same-party candidates (D={n_dem}, R={n_rep}); dem/rep_votes are party sums, "
                     "dem/rep_cand the leader")
    third = cands[cands["party"] == "OTHER"]
    if len(third) and total and third.iloc[0]["votes"] / total >= MAJOR_THIRD_SHARE:
        notes.append(f"major non-D/R candidate {third.iloc[0]['candidate']} "
                     f"({third.iloc[0]['votes'] / total:.1%})")
    if total <= 1:
        # MEDSL records 1 or 0 votes here; keep the uncontested share convention (1.0 D / 0.0 R)
        share = {"DEM": 1.0, "REP": 0.0}.get(winner["party"]) if out["uncontested"] else None
        out.update(dem_votes=None, rep_votes=None, total_votes=None, winner_share=None, dem_share_2p=share)
        notes.append("unopposed and not printed on ballot; MEDSL placeholder vote count, votes blanked")
    out["notes"] = notes
    return out


def race_notes(state: str, year: int, out: dict, rows: pd.DataFrame) -> list[str]:
    notes = list(out.get("notes", []))
    if state in TOP_TWO_STATES and year >= (2012 if state == "CA" else 2008):
        notes.append("top-two general")
        if out["uncontested"] and out.get("top2_same_party"):
            out["dem_share_2p"] = None
            notes.append(f"same-party top-two ({out['winner_party']} vs {out['winner_party']}): dem_share_2p blank")
    if state in RCV_RACES and year >= RCV_RACES[state]:
        notes.append("ranked-choice voting; MEDSL counts may be round 1 or final round (inconsistent by year)")
    if state == "LA":
        notes.append("Louisiana: Nov ballot is a jungle (all-party) election in most years")
    if rows["unofficial"].any():
        notes.append("MEDSL flags some counts unofficial")
    if rows["court_ordered"].any():
        notes.append("TX 2006 court-ordered open election in redrawn district (MEDSL flags it special; "
                     "it elected the full term, so special=0 here)")
    return notes


def build_race(rows: pd.DataFrame, key: tuple, state: str, year: int) -> dict:
    final = rows[rows["round"] == "runoff"]
    is_runoff = not final.empty
    out = summarize(final if is_runoff else rows, key)
    notes = race_notes(state, year, out, rows)
    out["runoff"] = int(is_runoff)
    out["round1_dem_share_2p"] = None
    if is_runoff:
        nov = rows[rows["round"] == "nov"]
        r1 = summarize(nov, key) if not nov.empty else {}
        out["round1_dem_share_2p"] = r1.get("dem_share_2p")
        r1_txt = (f"Nov round leader {r1['winner_name']} ({r1['winner_party']})" if r1.get("winner_name")
                  else "Nov round counts incomplete in MEDSL (finalists' Nov rows missing)")
        notes.append(f"decided in runoff; votes/shares are the runoff; {r1_txt}")
    out["notes"] = "; ".join(dict.fromkeys(notes))
    return out


# ----------------------------------------------------------------------------- tables
def race_id_house(state: str, district: int) -> str:
    return f"{state}-AL" if district == 0 else f"{state}-{district:02d}"


def add_race_note(row: dict) -> dict:
    note = RACE_NOTES.get((row["year"], row["race_id"], row["special"]))
    is_dem_caucus_indep = row["winner_name"] in DEM_CAUCUS_INDEPENDENTS or row["winner_name"].startswith("ANGUS")
    if note is None and is_dem_caucus_indep:
        note = "independent winner who caucuses with Democrats"
    return {**row, "notes": f"{row['notes']}; {note}".strip("; ")} if note else row


def check_la_house(row: dict) -> dict:
    key = (row["year"], row["district"])
    if row["state"] != "LA" or key not in LA_HOUSE_RUNOFF_WINNERS:
        return row
    expected = LA_HOUSE_RUNOFF_WINNERS[key]
    if row["winner_name"] != expected:
        raise ValueError(f"LA-{row['district']} {row['year']}: Nov leader {row['winner_name']} "
                         f"!= known runoff winner {expected}")
    note = "no Nov majority; Dec runoff NOT in MEDSL; votes are the Nov jungle round; winner checked = runoff winner"
    return {**row, "notes": f"{row['notes']}; {note}".strip("; ")}


def build_house(long: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (year, state, district, special), rows in long.groupby(["year", "state", "district", "special"]):
        out = build_race(rows, (year, state, district), state, year)
        row = {"year": year, "state": state, "district": district,
               "race_id": race_id_house(state, district), "special": int(special), **out}
        recs.append(add_race_note(check_la_house(row)))
    return finalize(pd.DataFrame(recs), ["year", "state", "district", "special"])


def parse_special_classes(path: Path) -> dict[tuple[int, str], int]:
    text = json.loads(path.read_text())["parse"]["wikitext"]
    names = {}
    for block in text.split("\n|-"):
        cls = re.search(r"List of United States senators from ([^|\]]+)\|[^\]]*\]\]<br\s*/?>\s*\(Class (\d)\)", block)
        links = r"\[\[(\d{4})(?:–\d{2})? United States Senate (?:special )?elections? in ([^|\]]+)\|"
        for m in re.finditer(links, block):
            if cls:
                names[(int(m.group(1)), m.group(2).strip())] = int(cls.group(2))
    return names


STATE_NAMES = {
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


def build_senate(long: pd.DataFrame, special_classes: dict) -> pd.DataFrame:
    two_races = long.groupby(["year", "state"])["special"].nunique()
    recs = []
    for (year, state, special), rows in long.groupby(["year", "state", "special"]):
        out = build_race(rows, (year, state, 0), state, year)
        if special:
            cls = special_classes.get((year, STATE_NAMES[state]))
            if cls is None:
                out["notes"] = f"{out['notes']}; special-election class not found on Wikipedia".strip("; ")
        else:
            cls = REGULAR_CLASS_BY_MOD6[year % 6]
        suffix = "-S" if special and two_races.loc[(year, state)] > 1 else ""
        recs.append(add_race_note({"year": year, "state": state, "district": 0,
                                   "race_id": f"{state}-SEN{suffix}", "special": int(special), "class": cls, **out}))
    df = pd.DataFrame(recs)
    df["class"] = df["class"].astype("Int64")
    return finalize(df, ["year", "state", "special"])


COLUMNS = ["year", "state", "district", "race_id", "special", "dem_cand", "rep_cand", "dem_votes", "rep_votes",
           "total_votes", "dem_share_2p", "winner_party", "winner_name", "uncontested", "notes",
           "runoff", "round1_dem_share_2p", "winner_share"]


def finalize(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    cols = COLUMNS[:5] + (["class"] if "class" in df else []) + COLUMNS[5:]
    out = df[cols].sort_values(sort_cols).reset_index(drop=True)
    for c in ["dem_votes", "rep_votes", "total_votes"]:
        out[c] = pd.to_numeric(out[c]).round().astype("Int64")
    for c in ["dem_share_2p", "round1_dem_share_2p", "winner_share"]:
        out[c] = pd.to_numeric(out[c]).round(5)
    return out


def validate(house: pd.DataFrame, senate: pd.DataFrame) -> None:
    regular = house[house["special"] == 0]
    counts = regular.groupby("year").size()
    bad = counts[counts != 435]
    if len(bad):
        raise ValueError(f"House regular race counts != 435: {bad.to_dict()}")
    if house.duplicated(["year", "race_id", "special"]).any() or senate.duplicated(["year", "race_id"]).any():
        raise ValueError("duplicate race keys")
    # every regular cycle must hold the full class (I: 33, II: 33, III: 34 states); catches MEDSL gaps
    seats = senate[senate["special"] == 0].groupby(["class", "year"]).size()
    bad = seats[seats != seats.index.get_level_values("class").map(SENATE_CLASS_SIZES)]
    if len(bad):
        raise ValueError(f"Senate regular races per cycle != class size: {bad.to_dict()}")
    print(f"  house rows={len(house)} senate rows={len(senate)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--offline", action="store_true", help="use files already in data/raw/mit")
    args = parser.parse_args()
    if not args.offline:
        download_raw(RAW_DIR)
    house = build_house(load_house(RAW_DIR / "1976-2024-house.tab"))
    classes = parse_special_classes(RAW_DIR / "wikipedia" / "senate_specials.json")
    senate = build_senate(load_senate(RAW_DIR / "1976-2024-senate-state.csv"), classes)
    validate(house, senate)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    house.to_csv(CLEAN_DIR / "results_house.csv", index=False)
    senate.to_csv(CLEAN_DIR / "results_senate.csv", index=False)
    print(f"  wrote {CLEAN_DIR / 'results_house.csv'} and results_senate.csv")


if __name__ == "__main__":
    main()
