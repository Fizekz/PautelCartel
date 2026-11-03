#!/usr/bin/env python3
"""FEC fundraising features for the 2026 House + Senate races and for past cycles.

Outputs
-------
data/features/fec_2026.csv             one row per race_id of data/templates/full_congress_template.csv
data/features/fec_2026_candidates.csv  one row per matched candidate (audit trail of the name matching)
data/features/fec_history.csv          one row per (year, race_id), 2010-2024 general elections

Sources (all public, no API key; downloaded into data/raw/fec/, which is git-ignored)
-------------------------------------------------------------------------------------
* FEC bulk "all candidates" summary  weball{YY}.zip  (one row per candidate, all authorized
  committees summed, full 2-year cycle; columns per
  https://www.fec.gov/campaign-finance-data/all-candidates-file-description/)
* FEC bulk candidate master          cn{YY}.zip      (election year, office, district, party, ICI,
  principal campaign committee)
* FEC "Federal Elections 20XX" results workbooks (2010-2022) and the 2024 "congressional general
  election candidates" list. Both carry the FEC candidate ID of every general-election candidate,
  so past nominees are linked to weball by ID, not by name.
* 2026 nominees are not published by the FEC yet. They are matched by name to the candidate names
  in the Kalshi snapshot (data/benchmark/kalshi_races_*.csv; used ONLY to pick the right FEC
  candidate, never as a feature). For generic Kalshi entries ("Republican party": all of LA's House
  seats, AK-SEN, a few safe seats) the fallback is the top-receipts candidate of that party among
  those whose latest report is the pre-general (coverage through 2026-10-14). That signal is weak:
  general-election candidates must file it, but some primary losers file one too (Cornyn TX-SEN,
  Cameron KY-SEN, Oberweis FL-19), which is why Kalshi names come first.

Comparability: weball totals cover the whole 2-year cycle up to the latest report. For 2026 that
is through ~2026-10-14 (pre-general); for the history files it is through 12/31 of the election
year (post-general and year-end included, e.g. the GA 2020 runoff money). Levels are therefore
not comparable across the two files; shares (receipts_share_dem) roughly are.

Dependencies: Python 3 standard library. The 2010-2014 results workbooks are legacy .xls files and
need `xlrd` (pip install xlrd, e.g. in .venv); without it those years fall back to the "top
receipts per party" nominee heuristic and say so in the nominee_source column.

Usage
-----
    python3 src/fetch_fec.py              # download what is missing, build all CSVs
    python3 src/fetch_fec.py --refresh    # re-download the 2026 files (FEC updates them daily)
    python3 src/fetch_fec.py --only 2026  # or --only history
"""
import argparse
import csv
import difflib
import glob
import io
import re
import sys
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fec"
FEATURES = ROOT / "data" / "features"
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"

BULK = "https://www.fec.gov/files/bulk-downloads/{year}/{name}"
RESULTS_FILES = {  # FEC "Federal Elections 20XX" (election results incl. FEC candidate IDs)
    2010: "https://www.fec.gov/documents/1675/federalelections2010.xls",
    2012: "https://www.fec.gov/documents/1691/federalelections2012.xls",
    2014: "https://www.fec.gov/documents/1700/federalelections2014.xls",
    2016: "https://www.fec.gov/documents/1890/federalelections2016.xlsx",
    2018: "https://www.fec.gov/documents/2706/federalelections2018.xlsx",
    2020: "https://www.fec.gov/documents/4228/federalelections2020.xlsx",
    2022: "https://www.fec.gov/documents/5676/federalelections2022.xlsx",
    2024: "https://www.fec.gov/documents/5548/2024congressgecands.xlsx",  # ballot list, no votes
}
HISTORY_YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
# Senate specials the FEC workbooks label plain "S": (year, state) for a state whose only Senate race
# that year was a special, and (year, FEC ID) for candidates of a special held alongside a regular race.
SOLE_SPECIALS = {(2020, "AZ")}
SPECIAL_CANDIDATES = {(2024, "S6NE00129"), (2024, "S4NE00223")}  # NE 2024 special: Ricketts v. Love

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
STATES = set(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
    "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
)
DEM_PARTIES = {"DEM", "DFL", "D"}
REP_PARTIES = {"REP", "R"}
TOP_TWO_STATES = {"CA", "WA"}
PREGENERAL_CUTOFF_2026 = "2026-10-10"  # pre-general report covers through 2026-10-14
FEC_ID_RE = re.compile(r"^[HSP]\d[A-Z0-9]{2}\d{5}$")
UA = "Mozilla/5.0 (DemocracyDay model-citizens research; fec bulk data)"


# --------------------------------------------------------------------------------------------
# download + parsing helpers
# --------------------------------------------------------------------------------------------
def download(url, dest, refresh=False):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if data[:5] == b"<?xml" and b"NoSuchKey" in data[:300]:
        raise RuntimeError(f"not found on FEC server: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def read_pipe_zip(path, cols):
    """FEC bulk .zip with one pipe-delimited .txt inside, no header row."""
    out = []
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
        with z.open(name) as fh:
            for line in io.TextIOWrapper(fh, encoding="latin-1"):
                parts = line.rstrip("\r\n").split("|")
                if len(parts) < len(cols):
                    parts += [""] * (len(cols) - len(parts))
                out.append(dict(zip(cols, parts)))
    return out


def fnum(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def iso_date(mdy):
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", (mdy or "").strip())
    return f"{m.group(3)}-{m.group(1)}-{m.group(2)}" if m else ""


def load_cycle(yy, refresh=False):
    """{cand_id: merged record} from weball{yy} (financials) + cn{yy} (master)."""
    year = 2000 + yy
    wpath = download(BULK.format(year=year, name=f"weball{yy:02d}.zip"),
                     RAW / f"weball{yy:02d}.zip", refresh)
    cpath = download(BULK.format(year=year, name=f"cn{yy:02d}.zip"), RAW / f"cn{yy:02d}.zip", refresh)
    weball = {r["CAND_ID"]: r for r in read_pipe_zip(wpath, WEBALL_COLS)}
    cn = {r["CAND_ID"]: r for r in read_pipe_zip(cpath, CN_COLS)}
    cands = {}
    for cid in sorted(set(weball) | set(cn)):  # sorted: deterministic tie-breaks downstream
        w, c = weball.get(cid), cn.get(cid)
        rec = {
            "cand_id": cid,
            "name": (c or w)["CAND_NAME"].strip(),
            "office": (c["CAND_OFFICE"] if c and c["CAND_OFFICE"] else cid[0]),
            "state": ((c and c["CAND_OFFICE_ST"]) or (w and w["CAND_OFFICE_ST"]) or "").strip(),
            "district": ((c and c["CAND_OFFICE_DISTRICT"]) or (w and w["CAND_OFFICE_DISTRICT"]) or "").strip(),
            "party": ((c and c["CAND_PTY_AFFILIATION"]) or (w and w["CAND_PTY_AFFILIATION"]) or "").strip(),
            "ici": ((c and c["CAND_ICI"]) or (w and w["CAND_ICI"]) or "").strip(),
            "elec_yr": int(c["CAND_ELECTION_YR"]) if c and c["CAND_ELECTION_YR"].isdigit() else None,
            "status": (c["CAND_STATUS"] if c else ""),
            "pcc": (c["CAND_PCC"].strip() if c else ""),
            "in_weball": w is not None,
        }
        if w:
            rec.update(
                receipts=fnum(w["TTL_RECEIPTS"]) or 0.0,
                disbursements=fnum(w["TTL_DISB"]) or 0.0,
                cash_on_hand=fnum(w["COH_COP"]) or 0.0,
                indiv_contrib=fnum(w["TTL_INDIV_CONTRIB"]) or 0.0,
                self_funding=(fnum(w["CAND_CONTRIB"]) or 0.0) + (fnum(w["CAND_LOANS"]) or 0.0),
                trans_from_auth=fnum(w["TRANS_FROM_AUTH"]) or 0.0,
                debts=fnum(w["DEBTS_OWED_BY"]) or 0.0,
                coverage_end=iso_date(w["CVG_END_DT"]),
                weball_ici=w["CAND_ICI"].strip(),
            )
        else:  # registered (cn) but no financial summary in weball: nothing reported
            rec.update(receipts=0.0, disbursements=0.0, cash_on_hand=0.0, indiv_contrib=0.0,
                       self_funding=0.0, trans_from_auth=0.0, debts=0.0, coverage_end="",
                       weball_ici="")
        if rec["district"] in ("", "0"):
            rec["district"] = "00"
        rec["district"] = rec["district"].zfill(2)
        cands[cid] = rec
    return cands


def party_side(p):
    p = (p or "").strip().upper()
    if p in DEM_PARTIES:
        return "dem"
    if p in REP_PARTIES:
        return "rep"
    return "other"


# --------------------------------------------------------------------------------------------
# spreadsheet readers (xlsx: stdlib; xls: xlrd if installed)
# --------------------------------------------------------------------------------------------
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _col_index(ref):
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def xlsx_sheets(path):
    """{sheet_name: [row lists]} using only zipfile + ElementTree."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    out = {}
    for sh in wb.find("m:sheets", NS):
        t = target[sh.get(RNS)].lstrip("/")
        t = t if t.startswith("xl/") else "xl/" + t
        root = ET.fromstring(z.read(t))
        rows = []
        for row in root.iter(f"{{{NS['m']}}}row"):
            vals = {}
            for c in row.findall("m:c", NS):
                typ, v = c.get("t"), c.find("m:v", NS)
                if typ == "s" and v is not None:
                    val = shared[int(v.text)]
                elif typ == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(f"{{{NS['m']}}}t"))
                elif v is not None:
                    val = v.text
                    if typ not in ("str", "b", "e"):
                        f = fnum(val)
                        val = f if f is not None else val
                else:
                    val = None
                vals[_col_index(c.get("r"))] = val
            rows.append([vals.get(i) for i in range(max(vals) + 1)] if vals else [])
        out[sh.get("name")] = rows
    return out


def xls_sheets(path):
    import xlrd  # optional dependency
    book = xlrd.open_workbook(str(path))
    return {s.name: [s.row_values(i) for i in range(s.nrows)] for s in book.sheets()}


def read_workbook(path):
    return xls_sheets(path) if str(path).endswith(".xls") else xlsx_sheets(path)


# --------------------------------------------------------------------------------------------
# name matching
# --------------------------------------------------------------------------------------------
SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v", "mr", "mrs", "ms", "miss", "dr", "md", "phd", "hon",
          "rev", "esq", "col", "ret", "sgt", "lt", "capt", "cpt", "gen", "maj", "captain",
          "prof", "dds", "cpa", "usn", "usmc", "sen", "rep"}
NICK = {
    "andy": {"andrew"}, "bill": {"william"}, "will": {"william"}, "billy": {"william"},
    "bob": {"robert"}, "rob": {"robert"}, "bobby": {"robert"}, "bert": {"robert", "herbert", "albert"},
    "jim": {"james"}, "jimmy": {"james"}, "jamie": {"james"}, "mike": {"michael"}, "mick": {"michael"},
    "tom": {"thomas"}, "tommy": {"thomas"}, "chris": {"christopher", "christian", "christine", "christina"},
    "nick": {"nicholas"}, "dan": {"daniel"}, "danny": {"daniel"}, "dave": {"david"},
    "joe": {"joseph"}, "joey": {"joseph"}, "jon": {"jonathan", "john"}, "johnny": {"john"},
    "jack": {"john"}, "ted": {"edward", "theodore"}, "ed": {"edward", "edwin"}, "eddie": {"edward"},
    "ken": {"kenneth"}, "kenny": {"kenneth"}, "steve": {"steven", "stephen"}, "rick": {"richard", "eric", "frederick"},
    "dick": {"richard"}, "rich": {"richard"}, "ricky": {"richard"}, "greg": {"gregory"},
    "jeff": {"jeffrey", "jeffery"}, "matt": {"matthew"}, "pat": {"patrick", "patricia"},
    "tony": {"anthony"}, "chuck": {"charles"}, "charlie": {"charles"}, "hank": {"henry"},
    "liz": {"elizabeth"}, "beth": {"elizabeth"}, "betsy": {"elizabeth"}, "betty": {"elizabeth"},
    "kate": {"katherine", "kathryn", "catherine"}, "katie": {"katherine", "kathryn", "catherine", "kathleen"},
    "kathy": {"katherine", "kathleen", "kathryn"}, "cathy": {"catherine"}, "maggie": {"margaret"},
    "peggy": {"margaret"}, "meg": {"margaret"}, "sue": {"susan", "suzanne"}, "susie": {"susan"},
    "debbie": {"deborah", "debra"}, "deb": {"deborah", "debra"}, "abby": {"abigail"},
    "becky": {"rebecca"}, "sam": {"samuel", "samantha"}, "ben": {"benjamin"}, "zach": {"zachary"},
    "alex": {"alexander", "alexandra", "alexandria"}, "al": {"albert", "alan", "allen", "alfred"},
    "don": {"donald"}, "ron": {"ronald"}, "tim": {"timothy"}, "fred": {"frederick"},
    "jerry": {"gerald", "jerome"}, "gerry": {"gerald"}, "larry": {"lawrence"}, "terry": {"terrence", "theresa"},
    "frank": {"francis", "franklin"}, "gabe": {"gabriel"}, "josh": {"joshua"}, "nate": {"nathan", "nathaniel"},
    "drew": {"andrew"}, "doug": {"douglas"}, "russ": {"russell"}, "ray": {"raymond"},
    "vince": {"vincent"}, "val": {"valerie"}, "vicky": {"victoria"}, "trish": {"patricia"},
    "mandy": {"amanda"}, "jen": {"jennifer"}, "jenny": {"jennifer"}, "lou": {"louis"},
    "marty": {"martin"}, "mitch": {"mitchell"}, "phil": {"phillip", "philip"}, "randy": {"randall", "randolph"},
    "ronnie": {"ronald"}, "stan": {"stanley"}, "wes": {"wesley"}, "buddy": {"earl"},
    "abe": {"abraham"}, "jake": {"jacob"}, "jd": {"james"}, "rj": {"robert"}, "tj": {"thomas"},
    "derek": {"derrick"}, "mark": {"marc"}, "brad": {"bradley"}, "pete": {"peter"}, "bart": {"bartholomew"},
}


def norm_tokens(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("’", "").replace("'", "").replace("`", "")
    s = re.sub(r"[^a-z]+", " ", s)
    return s.split()


def parse_fec_name(name):
    """'PAXTON, WARREN KENNETH JR.' -> (last_tokens, other_tokens)."""
    if "," in name:
        last, rest = name.split(",", 1)
    else:
        toks = name.split()
        last, rest = (toks[-1] if toks else ""), " ".join(toks[:-1])
    lt = [t for t in norm_tokens(last) if t not in SUFFIX]
    rt = [t for t in norm_tokens(rest) if t not in SUFFIX]
    return lt, rt


def first_match(kfirst, fec_tokens):
    if not kfirst or not fec_tokens:
        return 0
    if kfirst in fec_tokens:
        return 2
    alts = NICK.get(kfirst, set())
    for t in fec_tokens:
        if t in alts or kfirst in NICK.get(t, set()):
            return 2
        if len(kfirst) >= 3 and len(t) >= 3 and (t.startswith(kfirst) or kfirst.startswith(t)):
            return 1
    return 0


def _score(kt, fl, fr):
    joined_f = "".join(fl)
    last = 0
    for n in (1, 2, 3):
        if len(kt) >= n and "".join(kt[-n:]) == joined_f:
            last = 3
            break
    if not last and kt[-1] in fl and len(kt[-1]) >= 2:
        last = 2
    if not last and len(kt) >= 2 and any(t in fl for t in kt[1:] if len(t) >= 3):
        last = 2  # Kalshi drops / adds part of a compound surname
    if not last and kt[-1] in fr and len(kt[-1]) >= 3:
        last = 1  # maiden name kept as a middle name (Ashley Hinson -> ARENHOLZ, ASHLEY HINSON)
    first = first_match(kt[0], fr)
    if not first and len(kt) > 2 and first_match(kt[1], fr):
        first = 1  # Kalshi uses the middle name
    return last, first


def name_score(kalshi_name, fec_name):
    """(last_level, first_level) of a 'First Last' name against an FEC 'LAST, FIRST' name.
    last: 3 exact surname, 2 surname token, 1 surname found among the given names.
    Also tries the FEC name reversed: some records are keyed 'FIRST, LAST' ('NICK, LALOTA')."""
    kt = [t for t in norm_tokens(kalshi_name) if t not in SUFFIX]
    fl, fr = parse_fec_name(fec_name)
    if not kt or not fl:
        return 0, 0
    best = _score(kt, fl, fr)
    if fr:
        swapped = _score(kt, fr, fl)
        if swapped[0] >= 2 and swapped[1] >= 1 and sum(swapped) > sum(best):
            best = swapped
    return best


# --------------------------------------------------------------------------------------------
# 2026
# --------------------------------------------------------------------------------------------
GENERIC_RE = re.compile(r"^(democratic|republican|independent|other)( party)?$", re.I)

FIN_FIELDS = ["receipts", "indiv_contrib", "self_funding", "disbursements", "cash_on_hand", "debts"]


def parse_race(race_id):
    st, rest = race_id.split("-", 1)
    if rest == "SEN":
        return st, "S", "00"
    if rest == "AL":
        return st, "H", "00"
    return st, "H", rest.zfill(2)


def parse_kalshi(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ents = []
            for part in (r.get("candidates") or "").split(" | "):
                bits = part.strip().split(":")
                if len(bits) < 2:
                    continue
                party = bits[0].strip().upper()  # 'DEM:Barry Moore:0.978' (price ignored)
                name = (":".join(bits[1:-1]) if len(bits) > 2 else bits[1]).strip()
                ents.append({"kalshi_party": party, "kalshi_name": name,
                             "generic": bool(GENERIC_RE.match(name))})
            out[r["race_id"]] = ents
    return out


def build_2026(refresh=False):
    print("== 2026", file=sys.stderr)
    cands = load_cycle(26, refresh=refresh)
    kfiles = sorted(glob.glob(str(ROOT / "data" / "benchmark" / "kalshi_races_*.csv")))
    kalshi = parse_kalshi(kfiles[-1])
    with open(TEMPLATE, newline="") as fh:
        races = [r["race_id"] for r in csv.DictReader(fh)]

    # A House member running for Senate often keeps one committee that FEC then reports under both
    # the H and the S candidate ID (same totals twice). Once a Senate nominee is matched, his/her H
    # IDs (same principal committee, or same name in the same state) are marked as switchers and
    # kept out of the House fallback pools. Senate races are therefore processed first. (Direction
    # matters: e.g. LA's Blake Miguez left the Senate race for LA-06 and keeps both IDs.)
    for c in cands.values():
        c["switcher"] = False

    by_state_office = defaultdict(list)
    for c in cands.values():
        if c["state"] in STATES and c["office"] in ("H", "S"):
            by_state_office[(c["state"], c["office"])].append(c)

    n_dist = defaultdict(int)
    for rid in races:
        st, off, d = parse_race(rid)
        if off == "H":
            n_dist[st] += 1

    def same_district(c, st, off, d):
        if off == "S":
            return True
        if n_dist[st] == 1:
            return c["district"] in ("00", "01")
        return c["district"] == d

    def pregeneral(c):
        return (c.get("coverage_end") or "") >= PREGENERAL_CUTOFF_2026

    rows_by_race, cand_rows = {}, []
    for rid in sorted(races, key=lambda r: (parse_race(r)[1] != "S", races.index(r))):
        st, off, d = parse_race(rid)
        pool = by_state_office[(st, off)]
        ents = kalshi.get(rid, [])
        used = set()
        matched = []  # dicts: role, cand, method, kalshi_name

        # 1) named Kalshi candidates -> best FEC candidate
        for e in ents:
            if e["generic"]:
                continue
            best, best_key, best_tier = None, None, 0
            for c in pool:
                if c["cand_id"] in used:
                    continue
                last, first = name_score(e["kalshi_name"], c["name"])
                if last == 0:
                    continue
                full = (last >= 2 and first >= 1) or (last == 1 and first >= 1)
                sd = same_district(c, st, off, d)
                if sd and full:
                    tier = 3
                elif full and c["elec_yr"] == 2026:
                    tier = 2
                elif sd and last >= 2:
                    tier = 1
                else:
                    continue
                key = (tier, c["elec_yr"] == 2026, last + first, pregeneral(c),
                       c["ici"] == "I", c["receipts"], c["cand_id"])
                if best_key is None or key > best_key:
                    best, best_key, best_tier = c, key, tier
            if best is None:
                # Surname nowhere among this state's FEC candidates for the office: the nominee never
                # registered (registration is required at $5,000 raised or spent) -> money = 0.
                kt = [t for t in norm_tokens(e["kalshi_name"]) if t not in SUFFIX]
                anywhere = any(kt and (kt[-1] in parse_fec_name(c["name"])[0] or kt[-1] in parse_fec_name(c["name"])[1])
                               for c in pool)
                role = {"DEM": "dem", "REP": "rep"}.get(e["kalshi_party"], "other")
                matched.append({"role": role, "cand": None, "kalshi_name": e["kalshi_name"],
                                "kalshi_party": e["kalshi_party"],
                                "method": "kalshi_name_not_matched" if anywhere else "not_registered_with_fec"})
                continue
            used.add(best["cand_id"])
            # same person registered under several IDs (shared committee): block the twins too
            for c in pool:
                if c["pcc"] and c["pcc"] == best["pcc"]:
                    used.add(c["cand_id"])
            method = {3: "name", 2: "name_other_district", 1: "surname_only"}[best_tier]
            kp = e["kalshi_party"]
            role = {"DEM": "dem", "REP": "rep"}.get(kp)
            if role is None:
                role = "other" if kp in ("IND", "OTH", "LIB", "GRE") else party_side(best["party"])
            matched.append({"role": role, "cand": best, "kalshi_name": e["kalshi_name"],
                            "kalshi_party": kp, "method": method})

        # 2) generic Kalshi entries ("Republican party") -> pre-general filer of that party
        have_roles = {m["role"] for m in matched if m["cand"] is not None}
        named_n = sum(1 for e in ents if not e["generic"])
        for e in ents:
            if not e["generic"]:
                continue
            role = {"DEM": "dem", "REP": "rep"}.get(e["kalshi_party"], "other")
            if role in have_roles:
                continue
            if st in TOP_TWO_STATES and named_n >= 2:
                matched.append({"role": role, "cand": None, "kalshi_name": e["kalshi_name"],
                                "kalshi_party": e["kalshi_party"], "method": "generic_top_two_not_on_ballot"})
                continue
            cand_pool = [c for c in pool if same_district(c, st, off, d) and c["elec_yr"] == 2026
                         and party_side(c["party"]) == role and c["cand_id"] not in used
                         and not c["switcher"]]
            pg = [c for c in cand_pool if pregeneral(c)]
            if pg:
                best = max(pg, key=lambda c: (c["receipts"], c["cand_id"]))
                used.add(best["cand_id"])
                matched.append({"role": role, "cand": best, "kalshi_name": e["kalshi_name"],
                                "kalshi_party": e["kalshi_party"], "method": "generic_pregeneral_filer"})
            else:
                matched.append({"role": role, "cand": None, "kalshi_name": e["kalshi_name"],
                                "kalshi_party": e["kalshi_party"],
                                "method": f"generic_no_pregeneral_filer({len(cand_pool)} registered)"})

        # 3) third candidate not listed by Kalshi: non-D/R pre-general filer with money
        if not any(m["role"] == "other" and m["cand"] for m in matched) and st not in TOP_TWO_STATES:
            oth = [c for c in pool if same_district(c, st, off, d) and c["elec_yr"] == 2026
                   and party_side(c["party"]) == "other" and pregeneral(c) and c["receipts"] > 0
                   and c["cand_id"] not in used]
            if oth:
                best = max(oth, key=lambda c: (c["receipts"], c["cand_id"]))
                used.add(best["cand_id"])
                matched.append({"role": "other", "cand": best, "kalshi_name": "",
                                "kalshi_party": "", "method": "fec_pregeneral_third"})

        # assemble ----------------------------------------------------------------------------
        found = [m for m in matched if m["cand"] is not None]
        by_role = defaultdict(list)
        for m in found:
            by_role[m["role"]].append(m)
        same_party = None
        if len([m for m in found if m["role"] in ("dem", "rep")]) >= 2 and \
                (len(by_role["dem"]) >= 2 or len(by_role["rep"]) >= 2):
            same_party = by_role["dem"] if len(by_role["dem"]) >= 2 else by_role["rep"]

        row = {"race_id": rid, "chamber": "Senate" if off == "S" else "House", "state": st,
               "district": "SEN" if off == "S" else ("AL" if n_dist[st] == 1 else d)}

        def fill(prefix, m):
            c = m["cand"] if m else None
            zero = (m is not None and c is None and
                    m["method"].startswith(("not_registered_with_fec", "generic_no_pregeneral_filer")))
            row[f"{prefix}_cand_name_fec"] = c["name"] if c else ""
            row[f"{prefix}_cand_id"] = c["cand_id"] if c else ""
            row[f"{prefix}_party_fec"] = c["party"] if c else ""
            row[f"{prefix}_ici"] = c["ici"] if c else ""
            for f in FIN_FIELDS:
                row[f"{prefix}_{f}"] = round(c[f], 2) if c else (0.0 if zero else "")
            row[f"{prefix}_coverage_end"] = c["coverage_end"] if c else ""
            row[f"{prefix}_match"] = m["method"] if m else ""

        def role_entry(role):
            ms = [m for m in matched if m["role"] == role]
            got = [m for m in ms if m["cand"] is not None]
            if got:
                return max(got, key=lambda m: m["cand"]["receipts"])
            return ms[0] if ms else None

        if same_party:
            fill("dem", None)
            fill("rep", None)
        else:
            fill("dem", role_entry("dem"))
            fill("rep", role_entry("rep"))
        fill("other", role_entry("other"))

        row["same_party_race"] = 1 if same_party else 0
        sp = sorted(same_party, key=lambda m: m["cand"]["name"]) if same_party else [None, None]
        for i, m in enumerate(sp[:2], 1):
            c = m["cand"] if m else None
            row[f"sp{i}_cand_name_fec"] = c["name"] if c else ""
            row[f"sp{i}_cand_id"] = c["cand_id"] if c else ""
            row[f"sp{i}_party_fec"] = c["party"] if c else ""
            row[f"sp{i}_ici"] = c["ici"] if c else ""
            for f in ("receipts", "disbursements", "cash_on_hand"):
                row[f"sp{i}_{f}"] = round(c[f], 2) if c else ""
            row[f"sp{i}_coverage_end"] = c["coverage_end"] if c else ""
        if same_party:
            a, b = sp[0]["cand"]["receipts"], sp[1]["cand"]["receipts"]
            row["receipts_share_sp1"] = round(a / (a + b), 4) if a + b > 0 else ""
        else:
            row["receipts_share_sp1"] = ""

        main_m = [m for m in (same_party or [role_entry("dem"), role_entry("rep")]) if m and m["cand"]]
        oth_m = role_entry("other")
        all_m = main_m + ([oth_m] if oth_m and oth_m["cand"] else [])
        row["coverage_end_date"] = max((m["cand"]["coverage_end"] for m in main_m if m["cand"]["coverage_end"]),
                                       default="")
        dr, rr = row["dem_receipts"], row["rep_receipts"]
        row["receipts_share_dem"] = round(dr / (dr + rr), 4) if dr != "" and rr != "" and dr + rr > 0 else ""
        # incumbency per FEC ICI of the matched nominees, labelled by their role in this race
        # (role = Kalshi party, which overrides FEC party typos such as Risch = 'UNK')
        inc = sorted({m["role"] if m["role"] in ("dem", "rep", "other") else party_side(m["cand"]["party"])
                      for m in all_m if m["cand"]["ici"] == "I"})
        n_inc = sum(1 for m in all_m if m["cand"]["ici"] == "I")
        row["incumbent_party_fec"] = "+".join(inc) if inc else "none"
        row["seat_status_fec"] = ("incumbent_vs_incumbent" if n_inc >= 2 else
                                  f"incumbent_{inc[0]}" if n_inc == 1 else
                                  "open" if all_m else "unknown")
        row["both_dr_matched"] = int(bool(row["dem_cand_id"]) and bool(row["rep_cand_id"]))
        if off == "S":
            for m in found:
                for c in by_state_office[(st, "H")]:
                    if (m["cand"]["pcc"] and c["pcc"] == m["cand"]["pcc"]) or c["name"].upper() == m["cand"]["name"].upper():
                        c["switcher"] = True
        for side in ("dem", "rep"):
            # generic Kalshi entry (party-level market): sum over all pre-general filers of the party
            if row[f"{side}_match"] == "generic_pregeneral_filer":
                seen, tot = set(), 0.0
                for c in pool:
                    if same_district(c, st, off, d) and c["elec_yr"] == 2026 and party_side(c["party"]) == side \
                            and pregeneral(c) and not c["switcher"] and (c["pcc"] or c["cand_id"]) not in seen:
                        seen.add(c["pcc"] or c["cand_id"])
                        tot += c["receipts"]
                row[f"{side}_party_receipts_pregeneral_filers"] = round(tot, 2)
            else:
                row[f"{side}_party_receipts_pregeneral_filers"] = ""
        # other active candidates of the nominee's party (pre-general filers): >0 flags races where
        # Nov 3 still has several same-party candidates (e.g. Louisiana) or a nominee mismatch
        for side in ("dem", "rep"):
            cid = row[f"{side}_cand_id"]
            nom = cands.get(cid)
            row[f"{side}_n_other_pregeneral_same_party"] = (
                sum(1 for c in pool if same_district(c, st, off, d) and c["elec_yr"] == 2026
                    and party_side(c["party"]) == side and pregeneral(c) and c["cand_id"] != cid
                    and not (nom and nom["pcc"] and c["pcc"] == nom["pcc"]) and not c["switcher"])
                if cid else "")
        rows_by_race[rid] = row

        for m in matched:
            c = m["cand"]
            cand_rows.append({
                "race_id": rid, "role": ("same_party" if same_party and m in same_party else m["role"] or ""),
                "kalshi_party": m["kalshi_party"], "kalshi_name": m["kalshi_name"], "match": m["method"],
                "cand_id": c["cand_id"] if c else "", "cand_name_fec": c["name"] if c else "",
                "party_fec": c["party"] if c else "", "ici": c["ici"] if c else "",
                "fec_district": c["district"] if c else "", "election_yr": c["elec_yr"] if c else "",
                "receipts": round(c["receipts"], 2) if c else "",
                "cash_on_hand": round(c["cash_on_hand"], 2) if c else "",
                "disbursements": round(c["disbursements"], 2) if c else "",
                "coverage_end": c["coverage_end"] if c else "",
            })

    rows = [rows_by_race[r] for r in races]
    order = {r: i for i, r in enumerate(races)}
    cand_rows.sort(key=lambda x: order[x["race_id"]])
    FEATURES.mkdir(parents=True, exist_ok=True)
    write_csv(FEATURES / "fec_2026.csv", rows)
    write_csv(FEATURES / "fec_2026_candidates.csv", cand_rows)
    print(f"  wrote {len(rows)} races, {len(cand_rows)} candidate rows", file=sys.stderr)
    return rows, cand_rows


# --------------------------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------------------------
def _find_col(hdr, *preds):
    for i, h in enumerate(hdr):
        h = str(h or "").strip().upper()
        if any(p(h) for p in preds):
            return i
    return None


def results_rows(year):
    """General-election candidate rows from the FEC results workbook: dicts with
    state, office, district, special, fec_id, name, party, votes (None = no count), unopposed."""
    url = RESULTS_FILES[year]
    path = download(url, RAW / url.rsplit("/", 1)[1])
    try:
        sheets = read_workbook(path)
    except ImportError:
        print(f"  WARNING: xlrd not installed, cannot read {path.name}; {year} uses the "
              f"top-receipts nominee heuristic", file=sys.stderr)
        return None
    out = []
    for sname, rows in sheets.items():
        u = sname.upper()
        if year == 2024:
            if "BALLOT" not in u:
                continue
        elif not ("RESULTS" in u or "RESUTS" in u) or ("HOUSE" not in u and "SENATE" not in u) or "PRES" in u:
            continue
        hi = None
        for i, r in enumerate(rows[:5]):
            if any(str(x or "").strip().upper().startswith("FEC ID") for x in r):
                hi = i
                break
        if hi is None:
            continue
        hdr = rows[hi]
        c_st = _find_col(hdr, lambda h: "ABBREVIATION" in h)
        c_d = _find_col(hdr, lambda h: h in ("DISTRICT", "D"))
        c_id = _find_col(hdr, lambda h: h.startswith("FEC ID"))
        c_inc = _find_col(hdr, lambda h: h.startswith("(I)") or h.startswith("INCUMBENT"))
        c_name = _find_col(hdr, lambda h: h in ("CANDIDATE NAME", "CANDIDATE NAME (LAST, FIRST)"))
        c_party = _find_col(hdr, lambda h: h == "PARTY")
        c_gen = _find_col(hdr, lambda h: h in ("GENERAL", "GENERAL VOTES"))
        for r in rows[hi + 1:]:
            r = list(r) + [None] * (len(hdr) - len(r))
            st = str(r[c_st] or "").strip().upper()
            if st not in STATES:
                continue
            draw = str(r[c_d] if r[c_d] is not None else "").strip().upper()
            if not draw:
                continue
            name = str(r[c_name] or "").strip()
            if not name or name.upper().startswith(("SCATTERED", "DISTRICT", "TOTAL", "WRITE-IN", "NONE OF")) \
                    or any(x in name.upper() for x in ("VOTES", "NOT COUNTED", "CHALLENGED", "BLANK", "VOID")):
                continue
            party_raw = str(r[c_party] or "").replace("\xa0", " ").strip().upper()
            if party_raw.startswith(("COMBINED", "W ")) or party_raw in ("W", "PARTY KEY"):
                continue
            special = "UNEXPIRED" in draw or draw in ("SUN",)
            if draw.startswith("S"):
                office, dist = "S", "00"
            elif draw.startswith("H"):
                office, dist = "H", "00"
            else:
                m = re.match(r"^(\d+)", draw.replace(".0", ""))
                if not m:
                    continue
                office, dist = "H", m.group(1).zfill(2)
            fid = str(r[c_id] or "").strip().upper()
            fid = fid if FEC_ID_RE.match(fid) else ""
            if office == "S" and ((year, st) in SOLE_SPECIALS or (year, fid) in SPECIAL_CANDIDATES):
                special = True
            votes, unopposed = None, False
            if c_gen is not None:
                g = r[c_gen]
                if isinstance(g, (int, float)):
                    votes = float(g)
                elif str(g or "").strip().lower().startswith("unopposed"):
                    unopposed = True
                else:
                    votes = fnum(g)
                if not unopposed and not votes:
                    continue  # did not appear on the general ballot (primary loser)
            p = re.sub(r"^[WN]\((.*)\)$", r"\1", party_raw.split("/")[0].strip()).strip()
            out.append({"state": st, "office": office, "district": dist, "special": special,
                        "fec_id": fid, "name": name, "party": p, "votes": votes,
                        "unopposed": unopposed,
                        "inc": "I" in str(r[c_inc] or "").upper() if c_inc is not None else False})
    return out


def as_first_last(name):
    if "," in name:  # 'Last, First' -> 'First Last'
        last, rest = name.split(",", 1)
        return f"{rest.strip()} {last.strip()}"
    return name


def same_person(results_name, fec_name):
    """Does the FEC record behind a results-file ID belong to the named candidate? Tolerates the
    workbooks' spelling slips (Tilllis, Alderholt, Norcoss) but catches wrong IDs (Malinowski ->
    MELE, GREGG)."""
    last, first = name_score(as_first_last(results_name), fec_name)
    if last >= 2 or (last >= 1 and first >= 1):
        return True
    rl = "".join(t for t in norm_tokens(results_name.split(",")[0]) if t not in SUFFIX)
    fl = "".join(parse_fec_name(fec_name)[0])
    ratio = difflib.SequenceMatcher(None, rl, fl).ratio()
    rest = results_name.split(",", 1)[1] if "," in results_name else ""
    fm = first_match((norm_tokens(rest) or [""])[0], parse_fec_name(fec_name)[1])
    return ratio >= 0.8 or (ratio >= 0.7 and fm >= 1)


def build_history(years, refresh=False):
    rows, id_mismatch = [], []
    for year in years:
        print(f"== {year}", file=sys.stderr)
        cands = load_cycle(year % 100, refresh=False)
        res = results_rows(year)
        races = defaultdict(dict)  # (st, office, dist, special) -> key -> candidate agg
        source = "fec_results_general_votes" if year != 2024 else "fec_general_ballot_list"
        if res is None:
            source = "heuristic_top_receipts"
            for c in cands.values():
                if c["elec_yr"] == year and c["office"] in ("H", "S") and c["state"] in STATES \
                        and c["receipts"] > 0:
                    races[(c["state"], c["office"], c["district"] if c["office"] == "H" else "00", False)][
                        c["cand_id"]] = {"fec_id": c["cand_id"], "name": c["name"], "party": c["party"],
                                         "votes": c["receipts"], "unopposed": False}
        else:
            for r in res:
                if r["office"] == "H" and r["special"]:
                    continue  # same-day House special for the unexpired term: not the regular race
                # key by ID *and* surname: fusion lines of one candidate merge, but two candidates
                # given the same ID by a workbook typo (2024 NY-SEN) stay apart
                key = (r["fec_id"], "".join(norm_tokens(r["name"].split(",")[0])))
                rk = (r["state"], r["office"], r["district"], r["special"] if r["office"] == "S" else False)
                agg = races[rk].setdefault(key, {"fec_id": r["fec_id"], "name": r["name"], "party": None,
                                                  "best_row_votes": -1, "votes": 0.0, "unopposed": False})
                agg["votes"] += r["votes"] or 0.0
                agg["unopposed"] |= r["unopposed"]
                if (r["votes"] or 0) > agg["best_row_votes"]:
                    agg["best_row_votes"] = r["votes"] or 0
                    agg["party"] = r["party"]
        # at-large states this year (by number of House districts seen)
        ndist = defaultdict(set)
        for (st, off, dist, sp) in races:
            if off == "H":
                ndist[st].add(dist)
        sen_states = defaultdict(list)
        for (st, off, dist, sp) in races:
            if off == "S":
                sen_states[st].append(sp)
        for (st, off, dist, sp), members in sorted(races.items()):
            if off == "H":
                rid = f"{st}-AL" if len(ndist[st]) == 1 else f"{st}-{dist}"
            else:
                rid = f"{st}-SEN-SPECIAL" if (sp and len(sen_states[st]) > 1) else f"{st}-SEN"
            sides, merged = defaultdict(list), {}
            for m in members.values():
                c = cands.get(m["fec_id"]) if m["fec_id"] else None
                if c is not None and not same_person(m["name"], c["name"]):
                    id_mismatch.append((year, m["name"], m["fec_id"], c["name"]))
                    c = None  # typo'd ID in the results workbook (e.g. 2022 NJ-07 Malinowski)
                if c is None:  # no / unknown ID: try the name within state+office in this cycle
                    c = lookup_by_name(cands, m["name"], st, off, year)
                party = m["party"] if party_side(m["party"]) != "other" or not c else c["party"]
                side = party_side(party)
                if side == "other" and c is not None and party_side(c["party"]) != "other" and m["party"] in ("", "NONE"):
                    side = party_side(c["party"])
                ck = c["cand_id"] if c else m["name"].upper()
                prev = merged.get(ck)
                if prev is not None:  # fusion line without ID resolved to the same candidate
                    prev[1]["votes"] = (prev[1]["votes"] or 0.0) + (m["votes"] or 0.0)
                    prev[1]["unopposed"] |= m["unopposed"]
                    continue
                merged[ck] = [side, dict(m), c]
            for side, m, c in merged.values():
                rank = (1 if m["unopposed"] else 0, m["votes"] or 0.0,
                        c["receipts"] if c else 0.0)
                sides[side].append((rank, m, c))
            row = {"year": year, "race_id": rid, "chamber": "Senate" if off == "S" else "House",
                   "special": int(bool(sp)), "nominee_source": source}
            for side in ("dem", "rep", "other"):
                lst = sorted(sides.get(side, []), key=lambda x: x[0], reverse=True)
                if year == 2024 or source == "heuristic_top_receipts":
                    lst = sorted(sides.get(side, []), key=lambda x: (x[2]["receipts"] if x[2] else 0.0),
                                 reverse=True)
                if lst:
                    _, m, c = lst[0]
                    row[f"{side}_cand_name_fec"] = c["name"] if c else m["name"]
                    row[f"{side}_cand_id"] = c["cand_id"] if c else ""
                    if side == "other":
                        row["other_party"] = (c["party"] if c else "") or m["party"]
                    row[f"{side}_receipts"] = round(c["receipts"], 2) if c else 0.0
                    row[f"{side}_indiv_contrib"] = round(c["indiv_contrib"], 2) if c else 0.0
                    row[f"{side}_self_funding"] = round(c["self_funding"], 2) if c else 0.0
                    row[f"{side}_disbursements"] = round(c["disbursements"], 2) if c else 0.0
                    row[f"{side}_cash_on_hand"] = round(c["cash_on_hand"], 2) if c else 0.0
                    row[f"{side}_ici"] = c["ici"] if c else ""
                    row[f"{side}_coverage_end"] = c["coverage_end"] if c else ""
                    row[f"{side}_fin"] = ("weball" if c and c["in_weball"] else
                                          "registered_no_financials" if c else "no_fec_record")
                    row[f"{side}_n_general_cands"] = len(lst)
                else:
                    for f in ("cand_name_fec", "cand_id", "receipts", "indiv_contrib", "self_funding",
                              "disbursements", "cash_on_hand", "ici", "coverage_end", "fin"):
                        row[f"{side}_{f}"] = ""
                    if side == "other":
                        row["other_party"] = ""
                    row[f"{side}_n_general_cands"] = 0
            dr, rr = row["dem_receipts"], row["rep_receipts"]
            if dr == "" and rr == "":
                row["receipts_share_dem"] = ""
            else:
                d_, r_ = (dr or 0.0), (rr or 0.0)
                row["receipts_share_dem"] = round(d_ / (d_ + r_), 4) if d_ + r_ > 0 else ""
            row["both_dr_present"] = int(dr != "" and rr != "")
            row["same_party_race"] = int(row["dem_n_general_cands"] + row["rep_n_general_cands"] >= 2 and
                                         (row["dem_n_general_cands"] == 0 or row["rep_n_general_cands"] == 0)
                                         and st in TOP_TWO_STATES)
            if row["same_party_race"]:
                row["receipts_share_dem"] = ""  # D-vs-D / R-vs-R top-two race: no D/R split
            row["coverage_end_date"] = max(row["dem_coverage_end"] or "", row["rep_coverage_end"] or "")
            rows.append(row)
    write_csv(FEATURES / "fec_history.csv", rows)
    print(f"  wrote {len(rows)} history rows; {len(id_mismatch)} results-file FEC IDs pointed to a "
          f"different person and were re-matched by name", file=sys.stderr)
    for x in id_mismatch:
        print("    ", x, file=sys.stderr)
    return rows


_name_index_cache = {}


def lookup_by_name(cands, name, st, off, year):
    key = (year, st, off)  # one cands dict per cycle year
    if key not in _name_index_cache:
        _name_index_cache[key] = [c for c in cands.values() if c["state"] == st and c["office"] == off]
    best, bk = None, None
    kalshi_style = as_first_last(name)
    for c in _name_index_cache[key]:
        last, first = name_score(kalshi_style, c["name"])
        if last >= 2 and first >= 1:
            k = (c["elec_yr"] == year, last + first, c["in_weball"], c["receipts"], c["cand_id"])
            if bk is None or k > bk:
                best, bk = c, k
    return best


def write_csv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download the 2026 bulk files")
    ap.add_argument("--only", choices=["2026", "history"])
    ap.add_argument("--years", type=int, nargs="*", default=HISTORY_YEARS)
    a = ap.parse_args()
    if a.only in (None, "2026"):
        build_2026(refresh=a.refresh)
    if a.only in (None, "history"):
        build_history(a.years)


if __name__ == "__main__":
    main()
