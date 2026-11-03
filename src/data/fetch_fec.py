"""Build FEC fundraising tables for 2026 House/Senate races and past cycles.

Usage:
    .venv/bin/python src/data/fetch_fec.py            # use cached zips in data/raw/fec/, download missing ones
    .venv/bin/python src/data/fetch_fec.py --refresh  # re-download every FEC bulk zip

Inputs (FEC bulk data, no key needed; https://www.fec.gov/data/browse-data/?tab=bulk-data):
    weball<yy>.zip  "All candidates" summary: one row per candidate, two-year-cycle totals
    cn<yy>.zip      candidate master: election year, office, state, district, incumbent/challenger/open
    cm<yy>.zip      committee master: which candidate id a principal committee is linked to (dedupes H/S ids)
Plus the 2026 template race_ids and the Kalshi Nov 1 races snapshot (names only, to decide who is on the ballot).

Outputs (data/clean/):
    fec_2026.csv       one row per Kalshi-listed 2026 general-election candidate, with FEC totals where matched
    fec_race_2026.csv  per race: DEM / REP / other receipts, dem share, log ratio
    fec_hist.csv       2010-2024: every House/Senate candidate whose FEC election year equals the cycle
"""
from __future__ import annotations

import argparse
import io
import math
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "fec"
CLEAN_DIR = ROOT / "data" / "clean"
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"
KALSHI_RACES = ROOT / "data" / "kalshi" / "races_2026-11-01T074812Z.csv"

BULK_URL = "https://www.fec.gov/files/bulk-downloads/20{yy}/{kind}{yy}.zip"
CURRENT_CYCLE = 2026
HIST_CYCLES = (2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024)
HTTP_TIMEOUT_S = 120
FILING_THRESHOLD_USD = 5000.0
ACTIVE_COVERAGE_FROM = "2026-09-30"  # last quarterly before the pre-general report
NON_STATES = {"DC", "PR", "AS", "GU", "MP", "VI", "US", "00", ""}
# At-large states per cycle (FEC codes them as district 00, occasionally 01).
AT_LARGE_BASE = {"AK", "DE", "ND", "SD", "VT", "WY"}
AT_LARGE_EXTRA = {2010: {"MT"}, 2012: {"MT"}, 2014: {"MT"}, 2016: {"MT"}, 2018: {"MT"}, 2020: {"MT"}}
# House seats per state that differ from the 2020-census apportionment (used 2022-2026, read from the template).
SEATS_2012_2020 = {"CA": 53, "CO": 7, "FL": 27, "IL": 18, "MI": 14, "MT": 1, "NY": 27, "NC": 13, "OH": 16,
                   "OR": 5, "PA": 18, "TX": 36, "WV": 3}
SEATS_2010 = {**SEATS_2012_2020, "AZ": 8, "FL": 25, "GA": 13, "IL": 19, "IA": 5, "LA": 7, "MA": 10, "MI": 15,
              "MO": 9, "NV": 3, "NJ": 13, "NY": 29, "OH": 18, "PA": 19, "SC": 6, "TX": 32, "UT": 3, "WA": 9}

WEBALL_COLS = [
    "CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION", "TTL_RECEIPTS",
    "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH", "COH_BOP", "COH_COP", "CAND_CONTRIB",
    "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY",
    "TTL_INDIV_CONTRIB", "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "SPEC_ELECTION",
    "PRIM_ELECTION", "RUN_ELECTION", "GEN_ELECTION", "GEN_ELECTION_PRECENT",
    "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT", "INDIV_REFUNDS", "CMTE_REFUNDS",
]
CN_COLS = [
    "CAND_ID", "CAND_NAME", "CAND_PTY_AFFILIATION", "CAND_ELECTION_YR", "CAND_OFFICE_ST",
    "CAND_OFFICE", "CAND_OFFICE_DISTRICT", "CAND_ICI", "CAND_STATUS", "CAND_PCC", "CAND_ST1",
    "CAND_ST2", "CAND_CITY", "CAND_ST", "CAND_ZIP",
]
CM_COLS = [
    "CMTE_ID", "CMTE_NM", "TRES_NM", "CMTE_ST1", "CMTE_ST2", "CMTE_CITY", "CMTE_ST", "CMTE_ZIP",
    "CMTE_DSGN", "CMTE_TP", "CMTE_PTY_AFFILIATION", "CMTE_FILING_FREQ", "ORG_TP", "CONNECTED_ORG_NM", "CAND_ID",
]
MONEY_COLS = ["TTL_RECEIPTS", "TTL_DISB", "COH_COP"]
# States whose Nov 3 2026 House contest is an all-party open primary (Louisiana after Callais: Nov 3 open primary,
# Dec 12 runoff). Kalshi prices only the party there, so the race summary sums every active filer of each party.
OPEN_PRIMARY_HOUSE_STATES = {"LA"}

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "mr", "mrs", "ms", "dr", "hon", "md", "phd", "esq"}
NICKNAMES = {
    "mike": "michael", "jim": "james", "jimmy": "james", "bill": "william", "billy": "william",
    "bob": "robert", "bobby": "robert", "dick": "richard", "rick": "richard", "rich": "richard",
    "joe": "joseph", "tony": "anthony", "ted": "edward", "jack": "john", "chuck": "charles",
    "liz": "elizabeth", "beth": "elizabeth", "betsy": "elizabeth", "peggy": "margaret",
    "jerry": "gerald", "hank": "henry", "larry": "lawrence", "nate": "nathan", "andy": "andrew",
    "buddy": "", "chip": "", "trey": "", "kathy": "katherine", "katie": "katherine",
    "jenn": "jennifer", "jen": "jennifer", "sandy": "sandra", "mandy": "amanda", "becky": "rebecca",
    "steve": "steven", "stephen": "steven", "jon": "jonathan", "johnny": "john", "abe": "abraham",
    "tom": "thomas", "tommy": "thomas", "ed": "edward", "eddie": "edward", "gabe": "gabriel", "dave": "david",
    "nick": "nicholas", "jake": "jacob", "lizzie": "elizabeth", "drew": "andrew", "gene": "eugene",
    "hal": "harold", "marty": "martin", "rosie": "rosemary", "angie": "angela", "ami": "amerish",
    "ro": "rohit", "suzie": "suzanne", "jackie": "jacqueline", "berney": "bernard", "lucy": "lucia",
    "nellie": "nelida", "max": "mahesh", "ace": "ahsan", "pete": "peter", "sid": "sidney",
}
GENERIC_NAME = re.compile(r"\bparty\b", re.IGNORECASE)


# ---------------------------------------------------------------- download / load

def bulk_zip(kind: str, cycle: int, refresh: bool) -> Path:
    """Return the local path of an FEC bulk zip, downloading it if missing or refresh is set."""
    yy = f"{cycle % 100:02d}"
    path = RAW_DIR / f"{kind}{yy}.zip"
    if path.exists() and not refresh:
        return path
    url = BULK_URL.format(yy=yy, kind=kind)
    resp = requests.get(url, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    if not resp.content.startswith(b"PK"):
        raise RuntimeError(f"{url} did not return a zip file")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    print(f"downloaded {url} -> {path.relative_to(ROOT)} ({len(resp.content):,} bytes)")
    return path


def read_pipe_zip(path: Path, cols: list[str]) -> pd.DataFrame:
    """Read the single pipe-delimited text file inside an FEC bulk zip."""
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".txt")]
        if len(members) != 1:
            raise RuntimeError(f"{path}: expected one .txt member, found {members}")
        raw = zf.read(members[0]).decode("latin-1")
    df = pd.read_csv(io.StringIO(raw), sep="|", header=None, names=cols, dtype=str,
                     keep_default_na=False, quoting=3)
    if df.shape[1] != len(cols):
        raise RuntimeError(f"{path}: unexpected column count {df.shape[1]}")
    return df


def drop_shared_pcc_duplicates(cn: pd.DataFrame, cm: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """One person, several candidate ids sharing one principal campaign committee (PCC) in the same election year.

    weball repeats the committee's totals under every such id, so a House member running for Senate (O'Rourke 2018,
    Gardner 2014) also shows up in their House district with the Senate money. Keep the id the committee master
    links the PCC to; if none of the group's ids is that one, keep the ids whose office matches the committee type
    (H/S); if that also fails, keep the whole group.
    """
    links = (cm[["CMTE_ID", "CMTE_TP", "CAND_ID"]].drop_duplicates("CMTE_ID")
             .rename(columns={"CMTE_ID": "CAND_PCC", "CAND_ID": "PCC_CAND_ID"}))
    x = cn.merge(links, on="CAND_PCC", how="left")
    key = [x["CAND_PCC"], x["CAND_ELECTION_YR"]]
    in_group = (x["CAND_PCC"] != "") & x.duplicated(["CAND_PCC", "CAND_ELECTION_YR"], keep=False)
    linked = x["CAND_ID"] == x["PCC_CAND_ID"]
    keep = linked | (~linked.groupby(key).transform("any") & (x["CAND_OFFICE"] == x["CMTE_TP"]))
    keep = keep | ~keep.groupby(key).transform("any")
    drop = in_group & ~keep
    print(f"{cycle}: dropped {int(drop.sum())} duplicate candidate ids sharing a principal committee")
    return x[~drop.to_numpy()][cn.columns]


def load_cycle(cycle: int, refresh: bool) -> pd.DataFrame:
    """Candidate master joined to the all-candidates financial summary, House/Senate only."""
    cn = read_pipe_zip(bulk_zip("cn", cycle, refresh), CN_COLS)
    wb = read_pipe_zip(bulk_zip("weball", cycle, refresh), WEBALL_COLS)
    cm = read_pipe_zip(bulk_zip("cm", cycle, refresh), CM_COLS)
    cn = cn[cn["CAND_OFFICE"].isin(["H", "S"])].drop_duplicates("CAND_ID")
    cn = drop_shared_pcc_duplicates(cn.reset_index(drop=True), cm, cycle)
    wb = wb.drop_duplicates("CAND_ID")
    fin = wb[["CAND_ID", *MONEY_COLS, "CVG_END_DT"]].copy()
    for col in MONEY_COLS:
        fin[col] = pd.to_numeric(fin[col], errors="coerce")
    fin["CVG_END_DT"] = pd.to_datetime(fin["CVG_END_DT"], format="%m/%d/%Y", errors="coerce").dt.strftime("%Y-%m-%d")
    merged = cn.merge(fin, on="CAND_ID", how="left", indicator="fin_src")
    merged["has_fin"] = merged["fin_src"] == "both"
    merged["CAND_ELECTION_YR"] = pd.to_numeric(merged["CAND_ELECTION_YR"], errors="coerce")
    return merged.drop(columns="fin_src")


# ---------------------------------------------------------------- race ids / names

def race_id(state: str, office: str, district: str, cycle: int) -> str | None:
    """Map an FEC office/state/district to the template-style race_id, or None for non-states."""
    if state in NON_STATES:
        return None
    if office == "S":
        return f"{state}-SEN"
    at_large = AT_LARGE_BASE | AT_LARGE_EXTRA.get(cycle, set())
    if state in at_large:
        return f"{state}-AL"
    if not district.isdigit() or int(district) == 0:
        return None
    return f"{state}-{int(district):02d}"


def norm_tokens(text: str) -> list[str]:
    """Lower-case ASCII word tokens; apostrophes and dots are dropped inside words (O'Donnell, J.D.)."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[.'`\u2019]", "", text.lower())
    tokens = re.split(r"[^a-z]+", text)
    return [t for t in tokens if t and t not in NAME_SUFFIXES and len(t) > 1]


def split_fec_name(name: str) -> tuple[list[str], list[str]]:
    """FEC 'LAST, FIRST MIDDLE (NICK)' -> (last tokens, first/middle/nickname tokens)."""
    last, _, first = name.partition(",")
    return norm_tokens(last), norm_tokens(first)


def same_given_name(a: str, b: str) -> bool:
    """Same given name, allowing common nicknames and prefixes (Chris/Christopher)."""
    if a == b:
        return True
    ca, cb = NICKNAMES.get(a) or a, NICKNAMES.get(b) or b
    if ca == cb:
        return True
    return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))


def first_name_score(kalshi_given: list[str], fec_given: list[str]) -> int:
    """2 = a Kalshi given name equals an FEC first/middle/nick name, 1 = first initial agrees, 0 = none."""
    if any(same_given_name(k, f) for k in kalshi_given for f in fec_given):
        return 2
    if kalshi_given and fec_given and kalshi_given[0][0] == fec_given[0][0]:
        return 1
    return 0


def name_match(kalshi_tokens: list[str], fec_name: str) -> int | None:
    """None when surnames disagree, else the first-name score.

    Handles compound surnames (Trone Garriott), married names filed as middle names (ARENHOLZ, ASHLEY HINSON)
    and FEC rows with first/last swapped (NICK, LALOTA).
    """
    if not kalshi_tokens:
        return None
    last, given = split_fec_name(fec_name)
    surname, kalshi_given = kalshi_tokens[-1], kalshi_tokens[:-1]
    if surname in last or (len(kalshi_tokens) > 2 and set(kalshi_tokens[1:-1]) & set(last)):
        return first_name_score(kalshi_given, given)
    if surname in given:
        if set(kalshi_given) & set(last):
            return 2
        if first_name_score(kalshi_given, [g for g in given if g != surname]) == 2:
            return 2
    return None


# ---------------------------------------------------------------- 2026

def parse_kalshi(path: Path) -> pd.DataFrame:
    """One row per (race, Kalshi candidate slot) from the 'PARTY:Name:prob | ...' column."""
    races = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for rec in races.itertuples(index=False):
        for slot in filter(None, (s.strip() for s in rec.candidates.split("|"))):
            party, _, rest = slot.partition(":")
            name = rest.rpartition(":")[0] if rest.count(":") else rest
            rows.append({"race_id": rec.race_id, "kalshi_party": party.strip(), "kalshi_name": name.strip()})
    return pd.DataFrame(rows)


def score_candidate(slot: dict, cand: dict, office: str, district: str) -> tuple | None:
    """Ranking key for an FEC candidate against a Kalshi slot; None when it is not a plausible match.

    A surname-only match (no first-name agreement) is accepted only for the same office, seat, party and 2026.
    A match in another district (pre-redistricting filing) needs full first-name agreement.
    """
    fscore = name_match(norm_tokens(slot["kalshi_name"]), cand["CAND_NAME"])
    if fscore is None:
        return None
    same_office = cand["CAND_OFFICE"] == office
    same_district = office == "S" or cand["CAND_OFFICE_DISTRICT"] == district
    party_ok = party_bucket(cand["CAND_PTY_AFFILIATION"]) == slot["kalshi_party"]
    yr_ok = cand["CAND_ELECTION_YR"] == CURRENT_CYCLE
    if fscore == 0 and not (same_office and same_district and party_ok and yr_ok):
        return None
    if not same_district and fscore < 2:
        return None
    receipts = cand["TTL_RECEIPTS"] if pd.notna(cand["TTL_RECEIPTS"]) else -1.0
    return (same_office, same_district, fscore > 0, yr_ok, fscore, party_ok, bool(cand["has_fin"]), receipts)


def party_bucket(fec_party: str) -> str:
    """FEC party code -> DEM / REP / IND (everything else)."""
    if fec_party in {"DEM", "DFL"}:
        return "DEM"
    if fec_party == "REP":
        return "REP"
    return "IND"


def match_label(key: tuple, office: str) -> str:
    """Human-readable match quality: exact / last_name_initial / last_name_only (+ office/district caveats)."""
    same_office, same_district, _, _, fscore, *_ = key
    label = {2: "exact", 1: "last_name_initial", 0: "last_name_only"}[fscore]
    if not same_office:
        label += "_other_office"
    elif office == "H" and not same_district:
        label += "_other_district"
    return label


def match_slot(slot: dict, pool: pd.DataFrame) -> tuple[dict | None, str]:
    """Best FEC candidate for one Kalshi slot within the same state."""
    rid = slot["race_id"]
    state, _, seat = rid.partition("-")
    office = "S" if seat == "SEN" else "H"
    district = "00" if seat == "AL" else seat
    cands = pool[pool["CAND_OFFICE_ST"] == state]
    if GENERIC_NAME.search(slot["kalshi_name"]):
        return fill_generic_slot(slot, cands, office, district)
    best, best_key = None, None
    for cand in cands.to_dict("records"):
        key = score_candidate(slot, cand, office, district)
        if key is None:
            continue
        if best_key is None or key > best_key:
            best, best_key = cand, key
    if best is None:
        return None, "no_fec_match"
    return best, match_label(best_key, office)


def fill_generic_slot(slot: dict, cands: pd.DataFrame, office: str, district: str) -> tuple[dict | None, str]:
    """Kalshi lists only a party: fill it with that party's active 2026 incumbent for the seat, else with the only
    active 2026 campaign of that party (receipts >= FEC's $5,000 filing threshold and a report covering the general
    period); otherwise leave it unnamed."""
    pool = cands[(cands["CAND_OFFICE"] == office) & (cands["CAND_ELECTION_YR"] == CURRENT_CYCLE)
                 & (cands["CAND_PTY_AFFILIATION"].map(party_bucket) == slot["kalshi_party"])
                 & (cands["TTL_RECEIPTS"].fillna(0) >= FILING_THRESHOLD_USD)
                 & (cands["CVG_END_DT"].fillna("") >= ACTIVE_COVERAGE_FROM)]
    if office == "H":
        pool = pool[pool["CAND_OFFICE_DISTRICT"] == district]
    incumbents = pool[pool["CAND_ICI"] == "I"]
    if len(incumbents) == 1:
        return incumbents.iloc[0].to_dict(), "generic_party_active_incumbent_fec"
    if len(pool) == 1:
        return pool.iloc[0].to_dict(), "generic_party_sole_active_fec"
    return None, "generic_party_no_name"


def build_2026(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = load_cycle(CURRENT_CYCLE, refresh)
    template = pd.read_csv(TEMPLATE, dtype=str)[["race_id", "chamber"]]
    slots = parse_kalshi(KALSHI_RACES).merge(template, on="race_id", how="right")
    out = []
    for slot in slots.to_dict("records"):
        if not isinstance(slot.get("kalshi_name"), str):
            out.append({**slot, "kalshi_name_match": "no_kalshi_slot"})
            continue
        cand, label = match_slot(slot, pool)
        out.append({**slot, **fec_fields(cand), "kalshi_name_match": label})
    cands = pd.DataFrame(out)
    cands = flag_duplicate_ids(cands)
    fec_bucket = cands["fec_party"].map(party_bucket, na_action="ignore")
    known = cands["kalshi_party"].isin(["DEM", "REP", "IND"])
    cands = cands.assign(kalshi_party=cands["kalshi_party"].where(known, fec_bucket))
    cols = ["race_id", "chamber", "cand_id", "candidate_name", "party", "incumbent_challenger_open",
            "receipts", "disbursements", "cash_on_hand", "coverage_end_date", "kalshi_name_match",
            "kalshi_name", "fec_party", "fec_office_district", "fec_election_yr"]
    cands = cands.rename(columns={"kalshi_party": "party"})[cols]
    cands = cands.assign(fec_election_yr=cands["fec_election_yr"].astype("Int64"))
    return cands, race_summary(cands, template, open_primary_party_totals(pool))


def fec_fields(cand: dict | None) -> dict:
    if cand is None:
        return {}
    return {
        "cand_id": cand["CAND_ID"], "candidate_name": cand["CAND_NAME"],
        "incumbent_challenger_open": cand["CAND_ICI"], "receipts": cand["TTL_RECEIPTS"],
        "disbursements": cand["TTL_DISB"], "cash_on_hand": cand["COH_COP"],
        "coverage_end_date": cand["CVG_END_DT"], "fec_party": cand["CAND_PTY_AFFILIATION"],
        "fec_office_district": cand["CAND_OFFICE_DISTRICT"],
        "fec_election_yr": int(cand["CAND_ELECTION_YR"]) if pd.notna(cand["CAND_ELECTION_YR"]) else None,
    }


def flag_duplicate_ids(cands: pd.DataFrame) -> pd.DataFrame:
    """An FEC id matched to two Kalshi slots is ambiguous: keep the label but mark it."""
    dup = cands["cand_id"].notna() & cands.duplicated("cand_id", keep=False)
    label = cands["kalshi_name_match"].where(~dup, cands["kalshi_name_match"] + "_DUPLICATE_ID")
    return cands.assign(kalshi_name_match=label)


def open_primary_party_totals(pool: pd.DataFrame) -> pd.DataFrame:
    """Party receipt totals for House seats decided by an all-party open primary on Nov 3 (see
    OPEN_PRIMARY_HOUSE_STATES): every 2026 filer of the party in the district whose latest report covers the
    general period (drops withdrawn campaigns, e.g. Letlow's old LA-05 committee)."""
    active = pool[pool["CAND_OFFICE_ST"].isin(OPEN_PRIMARY_HOUSE_STATES) & (pool["CAND_OFFICE"] == "H")
                  & (pool["CAND_ELECTION_YR"] == CURRENT_CYCLE)
                  & (pool["CVG_END_DT"].fillna("") >= ACTIVE_COVERAGE_FROM)]
    active = active.assign(race_id=active["CAND_OFFICE_ST"] + "-" + active["CAND_OFFICE_DISTRICT"],
                           bucket=active["CAND_PTY_AFFILIATION"].map(party_bucket))
    sums = active.pivot_table(index="race_id", columns="bucket", values="TTL_RECEIPTS", aggfunc="sum",
                              fill_value=0.0)
    return sums.reindex(columns=["DEM", "REP", "IND"], fill_value=0.0)


def race_summary(cands: pd.DataFrame, template: pd.DataFrame, open_primary: pd.DataFrame) -> pd.DataFrame:
    """Per-race receipts by Kalshi party slot; unmatched slots count as 0 (see n_unmatched).

    Races in `open_primary` (rows race_id, columns DEM/REP/IND) use those all-filer party totals instead.
    """
    df = cands.assign(receipts=pd.to_numeric(cands["receipts"], errors="coerce"))
    df = df.assign(bucket=df["party"].where(df["party"].isin(["DEM", "REP"]), "IND"))
    sums = df.pivot_table(index="race_id", columns="bucket", values="receipts", aggfunc="sum", fill_value=0.0)
    sums = sums.reindex(columns=["DEM", "REP", "IND"], fill_value=0.0)
    is_open = sums.index.isin(open_primary.index)
    sums = pd.concat([sums[~is_open], open_primary.reindex(sums.index[is_open])])
    sums.columns = ["dem_receipts", "rep_receipts", "ind_receipts"]
    sums = sums.assign(receipts_basis=sums.index.map(
        lambda rid: "all_active_party_filers" if rid in open_primary.index else "kalshi_slots"))
    counts = cands.assign(receipts=df["receipts"]).groupby("race_id").agg(
        n_kalshi_slots=("kalshi_name", lambda s: int(s.notna().sum())),
        n_unmatched=("cand_id", lambda s: int(s.isna().sum())),
        n_no_fin=("receipts", lambda s: int(s.isna().sum())),
    )
    out = template[["race_id"]].merge(sums.reset_index(), on="race_id", how="left").merge(
        counts.reset_index(), on="race_id", how="left")
    total = out["dem_receipts"] + out["rep_receipts"]
    out["dem_share_receipts"] = (out["dem_receipts"] / total).where(total > 0)
    out["log_ratio"] = [math.log((d + 1.0) / (r + 1.0)) for d, r in zip(out["dem_receipts"], out["rep_receipts"])]
    return out[["race_id", "dem_receipts", "rep_receipts", "ind_receipts", "dem_share_receipts",
                "log_ratio", "n_kalshi_slots", "n_unmatched", "n_no_fin", "receipts_basis"]]


# ---------------------------------------------------------------- history

def seats_by_state(cycle: int) -> dict[str, int]:
    """House seats per state for a cycle (2000/2010/2020-census apportionment)."""
    template = pd.read_csv(TEMPLATE, dtype=str)
    house = template[template["chamber"] == "House"]["race_id"].str[:2]
    seats = house.value_counts().to_dict()
    if cycle <= 2010:
        return {**seats, **SEATS_2010}
    if cycle <= 2020:
        return {**seats, **SEATS_2012_2020}
    return seats


def valid_seat(rid: str, seats: dict[str, int]) -> bool:
    """Drop FEC district typos (e.g. state-legislative numbers like OH-43, TN-92)."""
    state, _, seat = rid.partition("-")
    if seat in ("SEN", "AL"):
        return state in seats
    return state in seats and 1 <= int(seat) <= seats[state]


def drop_phantom_senate_races(hist: pd.DataFrame) -> pd.DataFrame:
    """Keep a state's Senate rows only when some candidate is the incumbent or in an open-seat race;
    states with only challengers filed under that year are early filers for a later cycle."""
    sen = hist["race_id"].str.endswith("-SEN")
    real = hist[sen].groupby(["CAND_ELECTION_YR", "race_id"])["CAND_ICI"].transform(lambda x: x.isin(["I", "O"]).any())
    keep = ~sen
    keep.loc[real.index] = real.astype(bool)
    print(f"dropped {int((~keep).sum())} Senate candidates in states without a race that year")
    return hist[keep]


def build_hist(refresh: bool) -> pd.DataFrame:
    frames = []
    for cycle in HIST_CYCLES:
        df = load_cycle(cycle, refresh)
        df = df[df["CAND_ELECTION_YR"] == cycle].copy()
        df["race_id"] = [race_id(s, o, d, cycle) for s, o, d in
                         zip(df["CAND_OFFICE_ST"], df["CAND_OFFICE"], df["CAND_OFFICE_DISTRICT"])]
        seats = seats_by_state(cycle)
        ok = df["race_id"].notna() & df["race_id"].map(lambda r: isinstance(r, str) and valid_seat(r, seats))
        print(f"{cycle}: dropped {int((~ok).sum())} candidates with non-state or impossible district codes")
        frames.append(df[ok].assign(year=cycle))
    hist = drop_phantom_senate_races(pd.concat(frames, ignore_index=True))
    hist = hist.rename(columns={"CAND_NAME": "cand_name", "CAND_PTY_AFFILIATION": "party", "CAND_ICI": "ici",
                                "TTL_RECEIPTS": "receipts", "TTL_DISB": "disbursements", "CAND_ID": "cand_id",
                                "COH_COP": "cash_on_hand", "CVG_END_DT": "coverage_end_date"})
    hist["party_bucket"] = hist["party"].map(party_bucket)
    hist["party_rank"] = (hist.assign(r=hist["receipts"].fillna(-1))
                          .groupby(["year", "race_id", "party_bucket"])["r"]
                          .rank(method="first", ascending=False).astype(int))
    cols = ["year", "race_id", "cand_name", "party", "ici", "receipts", "disbursements",
            "cand_id", "party_bucket", "party_rank", "cash_on_hand", "coverage_end_date"]
    return hist[cols].sort_values(["year", "race_id", "party_bucket", "party_rank"]).reset_index(drop=True)


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download all FEC bulk zips")
    args = parser.parse_args(argv)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    cands, races = build_2026(args.refresh)
    cands.to_csv(CLEAN_DIR / "fec_2026.csv", index=False)
    races.to_csv(CLEAN_DIR / "fec_race_2026.csv", index=False)
    hist = build_hist(args.refresh)
    hist.to_csv(CLEAN_DIR / "fec_hist.csv", index=False)
    print(f"fec_2026.csv {len(cands)} rows; match labels: {cands['kalshi_name_match'].value_counts().to_dict()}")
    print(f"fec_race_2026.csv {len(races)} rows")
    print(f"fec_hist.csv {len(hist)} rows; by year: {hist['year'].value_counts().sort_index().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
