#!/usr/bin/env python3
"""Build the training tables from the raw files that src/fetch_history.py downloads:

  data/features/history_races.csv     one row per House / Senate November general election, 2010-2024
  data/features/history_national.csv  one row per cycle: president's party, final generic ballot,
                                      presidential approval, actual national House vote
  data/features/history_races_meta.json  page revisions, rating-column headers (dates), exclusions

Sources and how they are combined:
  * results / vote shares  : MIT Election Data + Science Lab (MEDSL) constituency returns (House, Senate),
                             incl. the Georgia Jan-2021 / Dec-2022 Senate runoffs. Races MEDSL lacks
                             (CT-SEN 2022, IL-SEN-SPECIAL 2010) and runoffs it lacks (Louisiana December
                             runoffs, Maine/Alaska RCV final rounds) use the Wikipedia percentages.
  * winner, incumbent, PVI : Wikipedia "<year> United States House of Representatives elections" state tables
                             (PVI column = the Cook PVI in use before that election, on the lines in use:
                             "Cook PVI (2008)" in 2010, 2004/08-based new-lines PVI in 2012, 2014 PVI in 2014
                             and 2016, 2017 PVI in 2018, 2018 PVI in 2020, 2022 PVI in 2022 and 2024) and
                             "<year> United States Senate elections" result tables.
  * final expert ratings   : Wikipedia "<year> United States House of Representatives election ratings" and the
                             ratings table on the Senate page: Cook, Rothenberg / Inside Elections, Sabato,
                             last pre-election column. Matched to results on (state, district number) for the
                             House (both pages use the lines in force that year) and (state, special) for the
                             Senate. Seats absent from the competitive table are imputed Safe (ratings_imputed=1).
  * Senate PVI             : the ratings table (2014+); 2010/2012: Wikipedia "Cook Partisan Voting Index" as it
                             stood on 15 Oct of the election year (2004/2008-based state PVI).
  * national               : FiveThirtyEight raw_polls.csv (generic ballot 2010-2022) and
                             generic_ballot_polls_historical.csv (2024); Gallup approval via the UCSB American
                             Presidency Project; FiveThirtyEight approval polls (2018-2024); MEDSL House votes.

Conventions (same as data/features/race_info_2026.csv): race_id KS-01 / WY-AL / AZ-SEN, Senate specials
XX-SEN-SPECIAL; pvi_at_time signed, + = Republican; ratings Safe D -3 .. Tossup 0 .. Safe R +3 with Tilt +-0.5;
independents' ratings (King, Sanders, Orman, McMullin, Osborn) count on the D (non-R) side. dem_share /
rep_share are two-party fractions; margin_r_minus_d is in percentage points (like last_margin there).

Run:  /usr/local/bin/python3 src/build_history.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikitable import STATES, STATE_RE, grid, parse_district_label, soup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "history"
WIKI = RAW / "wikipedia"
OUT = ROOT / "data" / "features"
YEARS = list(range(2010, 2026, 2))

DEM_LABELS = {"DEMOCRAT", "DEMOCRATIC", "DEMOCRATIC-FARMER-LABOR", "DEMOCRATIC-NONPARTISAN LEAGUE",
              "DEMOCRATIC-NPL", "WRITE-IN (DEMOCRATIC)"}
REP_LABELS = {"REPUBLICAN", "REPUBLICAN, LIBERTARIAN", "WRITE-IN (REPUBLICAN)"}

RATING_NUM = {"safe d": -3, "likely d": -2, "lean d": -1, "tilt d": -0.5, "tossup": 0,
              "tilt r": 0.5, "lean r": 1, "likely r": 2, "safe r": 3}


# --------------------------------------------------------------------------- helpers
def wpath(year, kind):
    return WIKI / f"{year}_United_States_{kind}.html"


def pvi_num(text):
    """'R+16' -> 16, 'D+5' -> -5, 'EVEN' -> 0, else None."""
    if text is None:
        return None
    t = text.upper().replace(" ", "")
    if "EVEN" in t:
        return 0.0
    m = re.search(r"([RD])\+(\d+(?:\.\d+)?)", t)
    if not m:
        return None
    v = float(m.group(2))
    return v if m.group(1) == "R" else -v


def norm_rating(text):
    """Map a Wikipedia rating cell to (canonical text, numeric) or (None, None)."""
    if not text:
        return None, None
    t = text.lower().replace("(flip)", "").replace("toss-up", "tossup").replace("toss up", "tossup")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("solid", "safe").replace("favored", "likely").replace("leans", "lean")
    t = t.replace("tilts", "tilt")
    m = re.match(r"(safe|likely|lean|tilt)\s+(d|r|dem|rep|democratic|republican|i|ind)\b", t)
    if m:
        party = m.group(2)[0]
        if party == "i":
            # independents in these tables (King, Sanders, Orman, McMullin, Osborn) are the non-R side
            return f"{m.group(1).title()} I", RATING_NUM[f"{m.group(1)} d"]
        key = f"{m.group(1)} {party}"
        return f"{m.group(1).title()} {party.upper()}", RATING_NUM[key]
    if t.startswith("tossup") or t.startswith("pure tossup"):
        # 'Tossup/Tilt R' style (old Rothenberg) -> tilt
        m2 = re.search(r"tilt\s+(d|r)", t)
        if m2:
            return f"Tilt {m2.group(1).upper()}", RATING_NUM[f"tilt {m2.group(1)}"]
        return "Tossup", 0.0
    return None, None


def party_code(text):
    t = (text or "").lower()
    if "democrat" in t or "dfl" in t or t.strip() in ("d",):
        return "DEM"
    if "republican" in t or t.strip() in ("r",):
        return "REP"
    if "independent" in t or t.strip() in ("i",):
        return "IND"
    if "libertarian" in t:
        return "LIB"
    return "OTH" if t.strip() else None


def name_tokens(name):
    """Upper-case name tokens; 'MURKOWSKI, LISA' -> ['LISA', 'MURKOWSKI']; hyphens split; suffixes dropped."""
    import unicodedata
    n = unicodedata.normalize("NFKD", re.sub(r"\(.*?\)", "", name or ""))
    n = "".join(ch for ch in n if not unicodedata.combining(ch))
    if n.count(",") == 1 and not re.search(r",\s*(jr|sr|ii|iii|iv)\b", n, re.I):
        a, b = n.split(",")
        n = f"{b} {a}"
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    n = re.sub(r"[^A-Za-z' ]", " ", n.replace("-", " "))
    return [p.upper() for p in n.split() if len(p) > 1 or p.isalpha()]


def last_name(name):
    parts = name_tokens(name)
    return parts[-1] if parts else ""


# --------------------------------------------------------------------------- candidates cell
def _linear_text(cell):
    """Cell text in document order, winner tick images -> ' \u2714 ', hidden spans dropped."""
    from bs4 import NavigableString, Tag
    parts = []

    def walk(node):
        for ch in node.children:
            if isinstance(ch, NavigableString):
                parts.append(str(ch))
            elif isinstance(ch, Tag):
                if ch.name in ("sup", "style", "link"):
                    continue
                if ch.has_attr("style") and re.search(r"display:\s*none", ch["style"]):
                    continue
                if ch.name == "img":
                    if re.search(r"tick|check", ch.get("alt", ""), re.I):
                        parts.append(" \u2714 ")
                    continue
                if ch.name in ("li", "br", "p", "dd", "dt", "div"):
                    parts.append(" ")
                walk(ch)
                if ch.name in ("li", "p", "dd", "dt", "div"):
                    parts.append(" ")
    walk(cell)
    t = "".join(parts).replace("\xa0", " ").replace("\u2013", "-")
    return re.sub(r"\s+", " ", t).strip()


SEG_RE = re.compile(r"(First round|Instant[- ]runoff|Second round|Final round|Runoff|"
                    r"General election|Jungle primary|Top-two primary|Primary)\s*:?\s*$", re.I)


def parse_candidates(cell):
    """Parse a Wikipedia candidates <td>: dict(rounds=[[cand,...],...], winner=cand, text=...)
    cand = dict(name, party, party_code, pct, winner). Rounds split on 'First round:', 'Runoff:' ..."""
    if cell is None:
        return dict(rounds=[], winner=None, text="")
    text = _linear_text(cell)
    pieces = text.split("\u258c")  # the coloured bar before each candidate
    rounds, cur = [], []
    for piece in pieces[1:]:
        piece = piece.strip()
        label = None
        m = re.search(r"(First round|Instant[- ]runoff|Second round|Final round|Runoff|General election|"
                      r"Jungle primary|Top-two primary|Primary)\s*:\s*$", piece, re.I)
        if m:
            label = m.group(1)
            piece = piece[:m.start()].strip()
        won = "\u2714" in piece
        txt = piece.replace("\u2714", "").strip()
        mm = re.match(r"(.*?)\s*\(([^()]*)\)\s*([\d.]+)\s*%", txt)
        if mm:
            name, party, pct = mm.group(1), mm.group(2), float(mm.group(3))
        else:
            mm = re.match(r"(.*?)\s*\(([^()]*)\)", txt)
            name, party, pct = (mm.group(1), mm.group(2), None) if mm else (txt, "", None)
            if re.search(r"unopposed|uncontested", txt, re.I) or (len(pieces) == 2 and pct is None):
                pct = 100.0
        cur.append(dict(name=name.strip(), party=party.strip(), party_code=party_code(party),
                        pct=pct, winner=won))
        if label:
            rounds.append(cur)
            cur = []
    if cur:
        rounds.append(cur)
    winner = None
    for r in reversed(rounds):
        w = [c for c in r if c["winner"]]
        if w:
            winner = w[0]
            break
    return dict(rounds=rounds, winner=winner, text=text)


# --------------------------------------------------------------------------- Wikipedia: House
def header_map(hdr):
    """hdr: list of header texts; returns dict role -> column index."""
    roles = {}
    for i, h in enumerate(hdr):
        hl = h.lower()
        if "pvi" in hl and "pvi" not in roles:
            roles["pvi"] = i
        elif hl in ("member", "representative", "incumbent", "senator") and "member" not in roles:
            roles["member"] = i
        elif hl == "party" and "party" not in roles:
            roles["party"] = i
        elif hl.startswith("first elected") or hl.startswith("electoral history"):
            roles["first"] = i
        elif hl in ("result", "results", "status") and "status" not in roles:
            roles["status"] = i
        elif hl.startswith("candidates") or hl.startswith("major candidates"):
            roles["cands"] = i
    return roles


def wiki_house_main(year):
    s = soup(wpath(year, "House_of_Representatives_elections"))
    out = {}
    for t in s.find_all("table", class_="wikitable"):
        g = grid(t)
        if len(g) < 2:
            continue
        hrow = None
        for k in range(min(3, len(g))):
            texts = [c[0] for c in g[k]]
            if any("PVI" in x for x in texts) and any(x.lower().startswith("candidates") for x in texts):
                hrow = k
                break
        if hrow is None:
            continue
        roles = header_map([c[0] for c in g[hrow]])
        for r in g[hrow + 1:]:
            if not r or r[0][1] and not parse_district_label(r[0][0]):
                continue
            lab = parse_district_label(r[0][0])
            if not lab:
                continue
            get = lambda role: r[roles[role]] if role in roles and roles[role] < len(r) else (None, None, None)
            rec = out.setdefault(lab, dict(pvi_text=None, incumbents=[], cands=None, status=[]))
            rec["pvi_text"] = get("pvi")[0]
            mem, party, status = get("member")[0], get("party")[0], get("status")[0]
            if mem is not None and (mem, party) not in [(a, b) for a, b, _ in rec["incumbents"]]:
                rec["incumbents"].append((mem, party, status))
            if rec["cands"] is None and get("cands")[2] is not None:
                rec["cands"] = parse_candidates(get("cands")[2])
    return out


# --------------------------------------------------------------------------- Wikipedia: ratings
def rating_cols(hdr):
    cols = {}
    for i, h in enumerate(hdr):
        hl = h.lower()
        if hl.startswith("cook") and "cook" not in cols:
            cols["cook"] = i
        elif (hl.startswith("rothenberg") or hl.startswith("roth.") or hl.startswith("ie ")
              or hl == "ie" or hl.startswith("inside")) and "ie" not in cols:
            cols["ie"] = i
        elif hl.startswith("sabato") and "sabato" not in cols:
            cols["sabato"] = i
        elif "pvi" in hl and "pvi" not in cols:
            cols["pvi"] = i
        elif hl in ("winner", "result") or hl.startswith("winner"):
            cols["winner"] = i
    return cols


def wiki_house_ratings(year):
    s = soup(wpath(year, "House_of_Representatives_election_ratings"))
    out, hdr_info = {}, None
    for t in s.find_all("table", class_="wikitable"):
        g = grid(t)
        if len(g) < 3:
            continue
        hrow = None
        for k in range(min(3, len(g))):
            texts = [c[0] for c in g[k]]
            if any(x.lower().startswith("cook") for x in texts) and any(x.lower().startswith("sabato") for x in texts):
                hrow = k
                break
        if hrow is None:
            continue
        cols = rating_cols([c[0] for c in g[hrow]])
        hdr_info = {k: g[hrow][v][0] for k, v in cols.items()}
        for r in g[hrow + 1:]:
            lab = parse_district_label(r[0][0]) if r else None
            if not lab or len(r) <= max(cols.values()):
                continue
            rec = {k: r[v][0] for k, v in cols.items()}
            out[lab] = rec
    return out, hdr_info


# --------------------------------------------------------------------------- Wikipedia: Senate
def senate_label(text):
    """'Georgia (special)' / 'Hawaii (Class 3)' -> ('GA', is_special_hint)."""
    m = STATE_RE.search(text.replace("\xa0", " "))
    if not m:
        return None, False
    return STATES[m.group(0)], bool(re.search(r"special|class", text, re.I))


def wiki_senate(year):
    s = soup(wpath(year, "Senate_elections"))
    ratings, results = {}, {}
    hdr_info = None
    for t in s.find_all("table", class_="wikitable"):
        g = grid(t)
        if len(g) < 3:
            continue
        for k in range(min(3, len(g))):
            texts = [c[0] for c in g[k]]
            low = [x.lower() for x in texts]
            if any(x.startswith("cook") for x in low) and any(x.startswith("sabato") for x in low):
                cols = rating_cols(texts)
                hdr_info = {kk: texts[v] for kk, v in cols.items()}
                for r in g[k + 1:]:
                    if not r:
                        continue
                    st, sp = senate_label(r[0][0])
                    if not st or len(r) <= max(cols.values()):
                        continue
                    rec = {kk: r[v][0] for kk, v in cols.items()}
                    rec["label"] = r[0][0]
                    rec["special"] = sp
                    ratings.setdefault(st, []).append(rec)
                break
            if any(x.startswith("candidates") or x.startswith("major candidates") for x in low) and ("senator" in low):
                roles = header_map(texts)
                # special-election tables come first on the page and say "Class" / "special"
                cap = t.find_previous(["h2", "h3", "h4"])
                captxt = cap.get_text(" ", strip=True).lower() if cap else ""
                for r in g[k + 1:]:
                    if not r or r[0][1] and STATE_RE.search(r[0][0]) is None:
                        continue
                    st, sp = senate_label(r[0][0])
                    if not st or "cands" not in roles or roles["cands"] >= len(r):
                        continue
                    special = sp or "special" in captxt
                    rec = dict(label=r[0][0], senator=r[roles["member"]][0],
                               party=r[roles["party"]][0], status=r[roles["status"]][0],
                               cands=parse_candidates(r[roles["cands"]][2]), special=special,
                               section=captxt)
                    results.setdefault(st, []).append(rec)
                break
    return ratings, results, hdr_info


def cook_pvi_states(year):
    """State PVIs from the Wikipedia 'Cook Partisan Voting Index' revision live before the election."""
    f = WIKI / f"Cook_Partisan_Voting_Index_asof_{year}-10-15.wikitext"
    if not f.exists():
        return {}
    txt = f.read_text()
    i = txt.find("By state")
    if i < 0:
        return {}
    out = {}
    for m in re.finditer(r"\|\s*\[\[([^\]|]+)(?:\|([^\]]*))?\]\]\s*\n\|[^\n]*?(EVEN|[RD]\+\d+)", txt[i:]):
        name = re.sub(r"\s*\(.*?\)", "", (m.group(2) or m.group(1))).strip()
        if name in STATES and STATES[name] not in out:
            out[STATES[name]] = m.group(3)
    return out


# --------------------------------------------------------------------------- MIT results
UNOPPOSED_NOTE = "unopposed; no votes recorded (state does not print unopposed races)"


def mit_major(labels):
    labels = {str(x).upper() for x in labels if isinstance(x, str)}
    if labels & DEM_LABELS:
        return "DEM"
    if labels & REP_LABELS:
        return "REP"
    if any("INDEPEND" in x or x in ("NONAFFILIATED", "UNAFFILIATED", "NO PARTY", "NONE",
                                    "NO PARTY AFFILIATION", "BY PETITION", "PETITION") for x in labels):
        return "IND"
    return "OTH" if labels else None


def mit_races(df, party_col, chamber):
    """Aggregate MEDSL rows to one record per race. df already filtered to the races wanted."""
    races = {}
    for key, g in df.groupby(["year", "state_po", "district", "special"], sort=False):
        cands = []
        g = g.copy()
        g["cand"] = g["candidate"].fillna("WRITEIN/OTHER").astype(str).str.upper().str.strip()
        for name, cg in g.groupby("cand", sort=False):
            votes = cg["candidatevotes"].fillna(0).clip(lower=0).sum()
            byparty = cg.groupby(cg[party_col].fillna("")).candidatevotes.sum()
            labels = [p for p in byparty.index if p]
            maj = mit_major(labels)
            if {str(x).upper() for x in labels} & DEM_LABELS and {str(x).upper() for x in labels} & REP_LABELS:
                maj = mit_major([byparty.idxmax()])
            cands.append(dict(name=name, votes=float(votes), party=maj,
                              labels="/".join(map(str, labels)),
                              writein=bool(cg["writein"].fillna(False).astype(bool).all())))
        cands.sort(key=lambda c: -c["votes"])
        total = float(g["totalvotes"].max())
        races[key] = dict(cands=cands, total=total,
                          unopposed_novotes=(total <= 1 and len([c for c in cands if not c["writein"]]) <= 1))
    return races


def load_mit():
    h = pd.read_csv(RAW / "mit_1976-2024-house.tab", low_memory=False)
    h = h[h.year.isin(YEARS) & (h.stage.str.upper() == "GEN") & (h.special.astype(str) != "True")
          & (h.runoff.astype(str) != "True")]
    house = mit_races(h, "party", "House")
    s = pd.read_csv(RAW / "mit_1976-2024-senate-state.csv", low_memory=False)
    s["stage"] = s["stage"].str.upper()
    s["district"] = 0
    runoff = s[s.stage.str.contains("RUNOFF")].copy()
    runoff.loc[runoff.year == 2021, "year"] = 2020  # Jan-2021 Georgia runoffs belong to the 2020 cycle
    gen = s[s.year.isin(YEARS) & (s.stage == "GEN")]
    senate = mit_races(gen, "party_detailed", "Senate")
    senate_runoff = mit_races(runoff[runoff.year.isin(YEARS)], "party_detailed", "Senate")
    return house, senate, senate_runoff


# --------------------------------------------------------------------------- race assembly
def status_running(status):
    s = (status or "").lower()
    if re.search(r"lost renomination|withdrew|did not advance|running in the|ran in|retired|resigned|died"
                 r"|ineligible|not seeking", s) and not re.search(r"write-in", s):
        if not re.match(r"incumbent re-elected", s):
            return 0
    if re.search(r"re-elected|lost re-?election|appointee elected|appointee lost election|elected to full", s):
        return 1
    return 0


def inc_party_code(party_text, status):
    p = (party_text or "").lower()
    if "vacant" in p or "none" in p or "new" in p or not p:
        m = re.search(r"\((D|R|I)\)", status or "")
        if m:
            return {"D": "DEM", "R": "REP", "I": "IND"}[m.group(1)], True
        return "none", False
    c = party_code(party_text)
    return ("IND" if c in ("IND", "LIB", "OTH") else c), False


def clean_member(m):
    m = re.sub(r"\s*Redistricted from.*$", "", m or "")
    m = re.sub(r"\s*\((retiring|running.*|lost.*|retired.*)\)\s*$", "", m)
    return m.strip()


def two_party(dv, rv):
    if dv + rv <= 0:
        return None, None
    return dv / (dv + rv), rv / (dv + rv)


def rating_fields(rec, prefix_map=(("cook", "cook"), ("ie", "inside_elections"), ("sabato", "sabato"))):
    out = {}
    for k, col in prefix_map:
        txt, num = norm_rating(rec.get(k)) if rec else (None, None)
        out[f"{col}_final"] = txt
        out[f"{col}_final_num"] = num
        out[f"{col}_raw"] = rec.get(k) if rec else None
    return out


def contest_type(dv, rv, cands, unopposed):
    nd = sum(1 for c in cands if c["party"] == "DEM" and c["votes"] > 0)
    nr = sum(1 for c in cands if c["party"] == "REP" and c["votes"] > 0)
    others = [c for c in cands if c["party"] not in ("DEM", "REP") and c["votes"] > 0 and not c["writein"]]
    if dv > 0 and rv > 0:
        return "D_vs_R"
    side = "D" if dv > 0 else ("R" if rv > 0 else "none")
    if (side == "D" and nd >= 2) or (side == "R" and nr >= 2):
        return f"{side}_vs_{side}"
    if unopposed or not others:
        return f"{side}_unopposed"
    return f"{side}_vs_minor"


def match_wiki_party(mit_cands, wiki_cands):
    """Fill MIT candidates without a D/R label (write-ins, Alaska 2022+ blank labels) from Wikipedia."""
    if not wiki_cands:
        return
    wl = [c for r in wiki_cands["rounds"] for c in r]
    tot = sum(c["votes"] for c in mit_cands) or 1
    for c in mit_cands:
        if c["party"] in (None, "OTH") or (c["party"] == "IND" and not c["labels"]):
            wp = next((w["party_code"] for w in wl if same_person(c["name"], w["name"])), None)
            if wp in ("DEM", "REP") and c["votes"] / tot >= 0.02:
                c["party"] = wp
                c["filled_from_wiki"] = True


NON_CANDIDATES = re.compile(r"EXHAUSTED|BLANK|^OTHERS?$|WRITEIN|WRITE-IN|SCATTERING|OVER ?VOTES|UNDER ?VOTES|VOID")


def same_person(a, b):
    import difflib
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return False
    la, lb = ta[-1], tb[-1]
    if la == lb or la in tb[1:] or lb in ta[1:] or (la in tb and lb in ta) \
            or "".join(ta[-2:]) == "".join(tb[-2:]):
        return True
    return difflib.SequenceMatcher(None, la, lb).ratio() >= 0.75 or \
        difflib.SequenceMatcher(None, (a or "").upper(), (b or "").upper()).ratio() >= 0.75


def leader_mismatch(mit_cands, win):
    """Note when the MIT November vote leader is not the Wikipedia (certified) winner."""
    real = [c for c in mit_cands if not NON_CANDIDATES.search(c["name"])]
    if not real or not win:
        return None
    lead = real[0]
    if lead["party"] in ("DEM", "REP") and win["party_code"] in ("DEM", "REP") and \
            lead["party"] != win["party_code"]:
        return (f"MEDSL November plurality leader {lead['name'].title()} ({lead['party']}) did not win "
                f"(runoff / RCV / write-in)")
    same_party = [c for c in real if c["party"] == lead["party"]]
    if lead["party"] == win["party_code"] and len(same_party) > 1 and not same_person(lead["name"], win["name"]):
        return (f"MEDSL November plurality leader {lead['name'].title()} ({lead['party']}) did not win "
                f"(same-party runoff / RCV)")
    if lead["party"] not in ("DEM", "REP") and win["party_code"] in ("DEM", "REP") and \
            not same_person(lead["name"], win["name"]):
        return f"MEDSL November plurality leader {lead['name'].title()} ({lead['party']}) did not win"
    return None


def final_round(wcands):
    if not wcands or not wcands["rounds"]:
        return None
    return wcands["rounds"][-1]


def real_others(mit_cands):
    """Non-D/R candidates who are people (not write-in pools, blanks, over/under votes)."""
    return [c for c in mit_cands if c["party"] not in ("DEM", "REP") and not c["writein"]
            and not NON_CANDIDATES.search(c["name"])]


STANDIN_MIN = 0.35  # an independent / third-party candidate this strong stands in for a missing major party


def wiki_as_mit(wc):
    """Pseudo-MEDSL race from the Wikipedia candidates cell (final round), used when MEDSL lacks the race."""
    fr = final_round(wc) or []
    cands = [dict(name=c["name"], votes=float(c["pct"] or 0), party=c["party_code"] if c["party_code"] in
                  ("DEM", "REP") else "IND", labels=c["party"], writein=False) for c in fr]
    cands.sort(key=lambda c: -c["votes"])
    tot = sum(c["votes"] for c in cands)
    return dict(cands=cands, total=tot, unopposed_novotes=len(cands) == 1)


def outcome(st, m, wc, win, runoff=None):
    """Vote-share block for one race.

    dem_share / rep_share are two-party shares (fractions), margin_r_minus_d is in percentage points of
    that two-party vote (same unit as last_margin in race_info_2026.csv). Rules:
      * D and R both on the ballot: MEDSL votes summed by party (fusion lines included).
      * an independent won, or a major party is missing and an independent / third-party candidate took
        >= 35%: that candidate stands in for the missing / weaker side (margin_kind says so), uncontested=0.
      * runoff / ranked-choice decided races: shares from the final round when it is D vs R
        (MEDSL runoff votes for Georgia, else the Wikipedia final-round percentages).
      * no candidate of the other major party and no strong independent: uncontested = 1, shares 1/0.
      * race missing from MEDSL: Wikipedia percentages.
    """
    notes = []
    res = dict(dem_share=None, rep_share=None, margin_r_minus_d=None, uncontested=None,
               contest_type=None, margin_kind=None, share_source=None,
               dem_votes=None, rep_votes=None, total_votes=None)
    src = "MEDSL"
    if m is None:
        if not wc or not wc["rounds"]:
            notes.append("no MEDSL row and no Wikipedia result")
            return res, notes
        m = wiki_as_mit(wc)
        src = "Wikipedia % (race missing from MEDSL)"
        notes.append("race missing from MEDSL; shares from Wikipedia percentages")
    else:
        match_wiki_party(m["cands"], wc)
    dv = sum(c["votes"] for c in m["cands"] if c["party"] == "DEM")
    rv = sum(c["votes"] for c in m["cands"] if c["party"] == "REP")
    total = m["total"] if m["total"] and m["total"] > 1 else sum(c["votes"] for c in m["cands"])
    ctype = contest_type(dv, rv, m["cands"], m["unopposed_novotes"])
    if src == "MEDSL":
        res.update(dem_votes=dv, rep_votes=rv, total_votes=total)
    if m["unopposed_novotes"]:
        if src == "MEDSL":
            notes.append(UNOPPOSED_NOTE)
        if dv + rv == 0:
            only = [c for c in m["cands"] if c["party"] in ("DEM", "REP")]
            if not only and win and win["party_code"] in ("DEM", "REP"):
                only = [dict(party=win["party_code"])]
            if only:
                dv, rv = (1.0, 0.0) if only[0]["party"] == "DEM" else (0.0, 1.0)
                ctype = "D_unopposed" if dv else "R_unopposed"
    ds, rs = two_party(dv, rv)
    kind = "two-party"
    others = real_others(m["cands"])
    top = others[0] if others else None
    win_other = bool(win and win["party_code"] not in ("DEM", "REP"))
    if top and total > 0:
        osh = top["votes"] / total
        missing = "DEM" if dv == 0 else ("REP" if rv == 0 else None)
        tname = f"{top['name'].title()}, {top['labels'].title() or 'no party label'}"
        if (missing and osh >= STANDIN_MIN) or (win_other and same_person(top["name"], win["name"])):
            if missing == "REP":
                ds, rs = two_party(dv, top["votes"])
                kind = f"D vs top non-D ({tname})"
            else:
                ds, rs = two_party(top["votes"], rv)
                kind = f"R vs top non-R ({tname})"
                if dv > 0:
                    notes.append(f"Democrat(s) also on the ballot: {dv / total:.1%} of the vote")
            ctype = "R_vs_IND" if missing != "REP" else "D_vs_IND"
        elif osh >= 0.15:
            notes.append(f"third candidate {tname}: {osh:.1%}")
    # final round: MEDSL runoff, else Wikipedia's last round. A one-round Wikipedia cell whose winner has
    # a majority the MEDSL November count does not show is a runoff-only cell (LA 2014/2016 Senate).
    fr = final_round(wc)
    lab = "instant runoff (RCV)" if st in ("ME", "AK") else "runoff"
    multi = bool(wc and len(wc["rounds"]) > 1)
    if not multi and fr and win and src == "MEDSL" and len(fr) == 2 and (win["pct"] or 0) > 50 \
            and st in ("LA", "GA", "MS"):
        wm = next((c for c in m["cands"] if same_person(c["name"], win["name"])), None)
        if wm is not None and total > 0 and wm["votes"] / total < 0.5 and \
                abs(100 * wm["votes"] / total - win["pct"]) > 2:
            multi = True
            notes.append("Wikipedia shows only the runoff; MEDSL has the November first round")
    if runoff is not None or (multi and fr):
        if runoff is not None:
            fd = sum(c["votes"] for c in runoff["cands"] if c["party"] == "DEM")
            frp = sum(c["votes"] for c in runoff["cands"] if c["party"] == "REP")
            fsrc = f"MEDSL {lab} votes"
        else:
            fd = sum(c["pct"] or 0 for c in fr if c["party_code"] == "DEM")
            frp = sum(c["pct"] or 0 for c in fr if c["party_code"] == "REP")
            fsrc = f"Wikipedia final-round % ({lab})"
        if fd > 0 and frp > 0:
            if ds is not None and src == "MEDSL":
                notes.append(f"first-round (November) two-party dem_share {ds:.3f}")
            ds, rs = two_party(fd, frp)
            src, kind, ctype = fsrc, f"two-party, final round ({lab})", "D_vs_R"
        else:
            notes.append(f"decided in {lab} between two "
                         f"{'Republicans' if frp > 0 else 'Democrats'}; shares are first round")
    if ctype in ("D_vs_D", "R_vs_R"):
        notes.append(f"same-party general ({ctype}, top-two/jungle); two-party share set to 1/0")
        kind = "same-party general"
    elif ctype in ("D_unopposed", "D_vs_minor", "R_unopposed", "R_vs_minor"):
        kind = "no major-party opponent"
    uncont = int(ctype is not None and ctype not in ("D_vs_R", "R_vs_IND", "D_vs_IND"))
    if src.startswith("MEDSL") or src.startswith("Wikipedia final"):
        lead_note = leader_mismatch(m["cands"], win) if not src.startswith("Wikipedia %") else None
        if lead_note and "final round" not in kind and "runoff" not in " ".join(notes):
            notes.append(lead_note)
    res.update(dem_share=ds, rep_share=rs, margin_r_minus_d=(100 * (rs - ds)) if ds is not None else None,
               uncontested=uncont, contest_type=ctype, margin_kind=kind, share_source=src)
    return res, notes


def impute_rating(pvi, ctype, inc_party, inc_running=0, redrawn=False):
    """Same rule as race_info_2026.csv: a seat missing from the ratings table is Safe for the party that
    is alone on the ballot, else Safe for the PVI's party (EVEN -> incumbent's party). One addition: an
    incumbent running again on unchanged lines in a seat whose PVI leans the other way (e.g. LoBiondo,
    NJ-02 D+1) was left off the competitive lists because the raters saw it as safe for the incumbent, so
    it is Safe for the incumbent's party (7 House rows 2010-2024; the 2026 table has no such seat)."""
    if ctype in ("D_unopposed", "D_vs_D"):
        return "Safe D", "only Democrats on the general ballot"
    if ctype in ("R_unopposed", "R_vs_R"):
        return "Safe R", "only Republicans on the general ballot"
    if pvi is not None and pvi != 0 and inc_running and not redrawn and inc_party in ("DEM", "REP") \
            and (pvi > 0) != (inc_party == "REP"):
        return ("Safe R" if inc_party == "REP" else "Safe D"), \
            "crossover incumbent running on unchanged lines and absent from the ratings table: imputed Safe " \
            "for the incumbent's party, not the PVI's"
    if pvi is not None and pvi != 0:
        return ("Safe R" if pvi > 0 else "Safe D"), None
    if inc_party in ("DEM", "REP"):
        return ("Safe R" if inc_party == "REP" else "Safe D"), "PVI EVEN: imputed for the incumbent's party"
    return None, "PVI EVEN and no incumbent party: no imputed rating"


RATING_COLS = ("cook", "sabato", "inside_elections")


def apply_ratings(row, rec, pvi, ctype, inc_party, notes, inc_running=0, redrawn=False):
    rf = rating_fields(rec)
    if rec is None:
        txt, why = impute_rating(pvi, ctype, inc_party, inc_running, redrawn)
        for col in RATING_COLS:
            rf[f"{col}_final"] = txt
            rf[f"{col}_final_num"] = RATING_NUM[txt.lower()] if txt else None
        if why:
            notes.append(why)
        if txt and inc_party in ("DEM", "REP") and txt[-1] != inc_party[0]:
            notes.append(f"imputed {txt} but seat held by {inc_party}")
        row["ratings_imputed"] = 1
    else:
        row["ratings_imputed"] = 0
        for col in RATING_COLS:
            if rf[f"{col}_final"] is None:
                notes.append(f"{col}: no rating ('{rf[f'{col}_raw']}')")
    for col in RATING_COLS:
        row[f"{col}_final"] = rf[f"{col}_final"]
        row[f"{col}_final_num"] = rf[f"{col}_final_num"]


def incumbent_block(incs, notes):
    """incs: list of (member, party_text, status). Returns incumbent fields."""
    running = [status_running(c) for a, b, c in incs]
    parties = []
    for a, b, c in incs:
        p, from_status = inc_party_code(b, c)
        parties.append(p)
        if from_status:
            notes.append(f"seat vacant before election (last held by {p})")
    ran = [p for p, r in zip(parties, running) if r and p != "none"]
    held = [p for p in parties if p != "none"]
    # member-vs-member: keep the first-listed incumbent (Wikipedia lists the same-numbered district's
    # member first); the other is in incumbent_name and the notes
    inc_party = (ran or held or ["none"])[0]
    if len(incs) > 1:
        notes.append("multiple incumbents listed: " + "; ".join(
            f"{a} ({p}, {'ran' if r else 'did not run'})" for (a, b, c), p, r in zip(incs, parties, running)))
        if len(set(ran)) > 1:
            notes.append("member-vs-member race between parties")
    if any("new seat" in (c or "").lower() or "new district" in (a or "").lower()
           or "new seat" in (a or "").lower() for a, b, c in incs):
        notes.append("new seat (reapportionment/redistricting)")
    names = "; ".join(a for a, b, c in incs if a and a.lower() not in ("vacant",)
                      and not a.lower().startswith("none") and "new seat" not in a.lower())
    return dict(incumbent_party=inc_party, incumbent_running=int(any(running)),
                open_seat=1 - int(any(running)), incumbent_name=names or None,
                incumbent_status="; ".join(c for a, b, c in incs if c) or None)


def winner_fields(win):
    if not win:
        return None, None
    p = win["party_code"]
    return win["name"], ("IND" if p in ("LIB", "OTH", "IND") else p)


REDISTRICTED = {  # states whose House map changed for that election (pvi and district numbers are on the new lines)
    2012: "all (post-2010 census)", 2016: "FL, NC, VA", 2018: "PA", 2020: "NC",
    2022: "all (post-2020 census)", 2024: "AL, GA, LA, NC, NY",
}
REDISTRICTED_STATES = {2016: {"FL", "NC", "VA"}, 2018: {"PA"}, 2020: {"NC"}, 2024: {"AL", "GA", "LA", "NC", "NY"}}


def build_house(mit_house, meta):
    rows = []
    for year in YEARS:
        main = wiki_house_main(year)
        ratings, rhdr = wiki_house_ratings(year)
        meta.setdefault("rating_columns", {})[f"House {year}"] = rhdr
        mit_keys = {(k[1], int(k[2])): k for k in mit_house if k[0] == year}
        for (st, d), w in sorted(main.items()):
            notes = []
            key = mit_keys.get((st, d)) or (mit_keys.get((st, 1)) if d == 0 else None) or \
                (mit_keys.get((st, 0)) if d == 1 else None)
            m = mit_house.get(key) if key else None
            race_id = f"{st}-AL" if d == 0 else f"{st}-{d:02d}"
            incs = [(clean_member(a), b, c) for a, b, c in w["incumbents"]]
            inc = incumbent_block(incs, notes)
            wc = w["cands"]
            win = wc["winner"] if wc else None
            winner_name, winner_party = winner_fields(win)
            out, onotes = outcome(st, m, wc, win)
            notes += onotes
            pvi = pvi_num(w["pvi_text"])
            rr = ratings.get((st, d))
            if rr and rr.get("pvi") and pvi_num(rr["pvi"]) is not None and pvi is not None \
                    and abs(pvi_num(rr["pvi"]) - pvi) > 0.5:
                notes.append(f"PVI differs: elections page {w['pvi_text']} vs ratings page {rr['pvi']}")
            if pvi is None and rr and rr.get("pvi"):
                pvi = pvi_num(rr["pvi"])
            if st in ("CA", "WA"):
                notes.append("top-two primary state")
            if st == "LA" and year >= 2012:
                notes.append("Louisiana jungle: November all-party election, December runoff if nobody >50%")
            if st == "LA" and year == 2010:
                notes.append("Louisiana 2010: closed party primaries, November general")
            if (st == "ME" and year >= 2018) or (st == "AK" and year >= 2022):
                notes.append("ranked-choice general")
            redis = year in (2012, 2022) or st in REDISTRICTED_STATES.get(year, set())
            if redis:
                notes.append(f"new district lines in {year}; pvi on the new lines")
            if year == 2018 and st == "NC" and d == 9:
                notes.append("result not certified (absentee-ballot fraud); new election 2019 won by "
                             "Dan Bishop (R); winner here = November 2018 vote leader Mark Harris (R)")
                winner_name, winner_party = "Mark Harris", "REP"
            row = dict(year=year, chamber="House", race_id=race_id, state=st,
                       district="AL" if d == 0 else f"{d:02d}", special=0,
                       pvi_at_time=pvi, pvi_text=w["pvi_text"], **inc)
            apply_ratings(row, rr, pvi, out["contest_type"], inc["incumbent_party"], notes,
                          inc["incumbent_running"], redis)
            row.update(winner_party=winner_party, winner_name=winner_name, **out,
                       redistricted=int(redis), notes="; ".join(dict.fromkeys(notes)) or None)
            rows.append(row)
        for (st, d), key in mit_keys.items():
            if st == "DC":
                continue
            if (st, d) not in main and not ((st, 0) in main and d == 1) and not ((st, 1) in main and d == 0):
                print(f"[warn] {year} {st}-{d}: in MEDSL but not on the Wikipedia page")
    return pd.DataFrame(rows)


# MEDSL has a single 2010 Illinois Senate race, flagged special, whose totals are the full-term race's
MEDSL_SPECIAL_FIX = {(2010, "IL"): "MEDSL has one IL 2010 Senate race (flagged special, full-term totals): "
                                   "used for the regular race; the special uses Wikipedia percentages"}
# Senate specials that were not held on the November general-election day (excluded)
OFF_CYCLE_SPECIALS = {(2010, "MA"): "Massachusetts special of 19 Jan 2010 (Scott Brown)"}


def build_senate(mit_senate, mit_runoff, meta):
    rows = []
    for year in YEARS:
        ratings, results, rhdr = wiki_senate(year)
        meta.setdefault("rating_columns", {})[f"Senate {year}"] = rhdr
        state_pvi = cook_pvi_states(year) if year <= 2012 else {}
        for st, recs in sorted(results.items()):
            for rec in recs:
                special = bool(rec["special"])
                if special and (year, st) in OFF_CYCLE_SPECIALS:
                    meta.setdefault("excluded", []).append(f"{year} {OFF_CYCLE_SPECIALS[(year, st)]}")
                    continue
                notes = []
                key = (year, st, 0, special)
                m = mit_senate.get(key)
                ro = mit_runoff.get(key)
                if (year, st) in MEDSL_SPECIAL_FIX:
                    m = mit_senate.get((year, st, 0, True)) if not special else None
                    notes.append(MEDSL_SPECIAL_FIX[(year, st)])
                if m is None:  # MEDSL flags some specials differently; fall back to the other flag
                    alt = mit_senate.get((year, st, 0, not special))
                    if alt is not None and len(recs) == 1:
                        m = alt
                race_id = f"{st}-SEN" + ("-SPECIAL" if special else "")
                inc = incumbent_block([(clean_member(rec["senator"]), rec["party"], rec["status"])], notes)
                win = rec["cands"]["winner"]
                winner_name, winner_party = winner_fields(win)
                out, onotes = outcome(st, m, rec["cands"], win, runoff=ro)
                notes += onotes
                # ratings: same state and same special flag; a special without its own row (same-day
                # partial-term race with the same candidates: IL 2010, CA 2022, CA 2024) copies the regular
                rr_list = ratings.get(st, [])
                rr = next((r for r in rr_list if bool(r["special"]) == special), None)
                if rr is None and special and len(rr_list) == 1:
                    rr = rr_list[0]
                    notes.append("no separate rating for the special; ratings copied from the regular race "
                                 "(same day, same candidates)")
                if rr is None and not special and len(rr_list) == 1 and len(recs) == 1:
                    rr = rr_list[0]
                pvi_text = None
                for r in ([rr] if rr else []) + rr_list:
                    if r and r.get("pvi"):
                        pvi_text = r["pvi"]
                        break
                pvi_src = "Senate ratings table (Wikipedia)"
                if pvi_text is None and st in state_pvi:
                    pvi_text = state_pvi[st]
                    pvi_src = f"Wikipedia 'Cook Partisan Voting Index' as of {year}-10-15 (2004/2008 PVI)"
                pvi = pvi_num(pvi_text)
                if special:
                    notes.append("special election")
                if st == "LA" and year >= 2014:
                    notes.append("Louisiana jungle: November all-party election, December runoff if nobody >50%")
                if st == "GA":
                    notes.append("Georgia: runoff if nobody >50%")
                if (year, st, special) == (2018, "MS", True):
                    notes.append("Mississippi special: nonpartisan November election + 27 Nov runoff")
                if st in ("CA", "WA"):
                    notes.append("top-two primary state")
                if (st == "ME" and year >= 2018) or (st == "AK" and year >= 2022):
                    notes.append("ranked-choice general")
                row = dict(year=year, chamber="Senate", race_id=race_id, state=st, district="SEN",
                           special=int(special), pvi_at_time=pvi, pvi_text=pvi_text, **inc)
                apply_ratings(row, rr, pvi, out["contest_type"], inc["incumbent_party"], notes,
                              inc["incumbent_running"], False)
                row.update(winner_party=winner_party, winner_name=winner_name, **out, redistricted=0,
                           notes="; ".join(dict.fromkeys(notes + ([f"PVI source: {pvi_src}"]
                                                                  if year <= 2012 else []))) or None)
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- national table
PRESIDENT = {2010: ("DEM", "Obama"), 2012: ("DEM", "Obama"), 2014: ("DEM", "Obama"), 2016: ("DEM", "Obama"),
             2018: ("REP", "Trump"), 2020: ("REP", "Trump"), 2022: ("DEM", "Biden"), 2024: ("DEM", "Biden")}
ELECTION_DAY = {2010: "2010-11-02", 2012: "2012-11-06", 2014: "2014-11-04", 2016: "2016-11-08",
                2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05"}
NAT = RAW / "national"
WINDOW_DAYS = 21


def latest_per_pollster_mean(df, value_col):
    """Mean over pollsters of each pollster's latest poll (ties averaged). df: pollster, end, value."""
    last = df.groupby("pollster").end.transform("max")
    d = df[df.end == last].groupby("pollster")[value_col].mean()
    return float(d.mean()), int(len(d))


def generic_ballot(year):
    eday = pd.Timestamp(ELECTION_DAY[year])
    rp = pd.read_csv(NAT / "538_raw_polls.csv", low_memory=False)
    g = rp[(rp.type_simple == "House-G-US") & (rp.cycle == year)].copy()
    if len(g):
        g["end"] = pd.to_datetime(g.polldate)
        g = g[(g.end < eday) & (g.end >= eday - pd.Timedelta(days=WINDOW_DAYS)) & g.partisan.isna()]
        assert (g.cand1_party == "DEM").all() and (g.cand2_party == "REP").all()
        g["r_minus_d"] = g.cand2_pct - g.cand1_pct
        v, n = latest_per_pollster_mean(g, "r_minus_d")
        return v, n, (f"FiveThirtyEight pollster-ratings raw_polls.csv (House-G-US), mean of each pollster's "
                      f"latest non-partisan poll ending in the {WINDOW_DAYS} days before election day")
    f = ROOT / "data" / "raw" / "polls" / "538" / "generic_ballot_polls_historical.csv"
    if f.exists():
        x = pd.read_csv(f, low_memory=False)
        x = x[x.cycle == year].copy()
        x["end"] = pd.to_datetime(x.end_date, format="%m/%d/%y")
        x = x[(x.end < eday) & (x.end >= eday - pd.Timedelta(days=WINDOW_DAYS)) & x.partisan.isna()]
        # one row per question; prefer likely voters, then registered, then adults within a poll
        rank = {"lv": 0, "rv": 1, "v": 2, "a": 3}
        x["pr"] = x.population.map(rank).fillna(4)
        x = x.sort_values("pr").groupby("poll_id").head(1)
        x["r_minus_d"] = x.rep - x.dem
        v, n = latest_per_pollster_mean(x, "r_minus_d")
        return v, n, (f"FiveThirtyEight generic_ballot_polls_historical.csv (archived 2025-03), mean of each "
                      f"pollster's latest non-partisan poll ending in the {WINDOW_DAYS} days before election day "
                      f"(LV > RV > A within a poll)")
    return None, 0, None


def gallup_approval(year):
    pres = PRESIDENT[year][1]
    f = NAT / {"Obama": "ucsb_gallup_approval_obama.html", "Trump": "ucsb_gallup_approval_trump1.html",
               "Biden": "ucsb_gallup_approval_biden.html"}[pres]
    s = soup(f)
    rows = []
    for tr in s.find("table").find_all("tr")[1:]:
        c = [x.get_text(" ", strip=True) for x in tr.find_all(["td", "th"])]
        if len(c) >= 4 and re.match(r"\d\d/\d\d/\d{4}", c[1]) and c[2].isdigit():
            rows.append((pd.Timestamp(pd.to_datetime(c[0])), pd.Timestamp(pd.to_datetime(c[1])),
                         int(c[2]), int(c[3]) if c[3].isdigit() else None))
    eday = pd.Timestamp(ELECTION_DAY[year])
    prior = [r for r in rows if r[1] < eday]
    st, en, ap, dis = max(prior, key=lambda r: r[1])
    return ap, dis, st, en, (f"Gallup via UCSB American Presidency Project ({pres} job approval), last Gallup "
                             f"poll ending before election day: {st.date()}..{en.date()}")


def approval_poll_avg(year):
    """538 approval polls (2017+), same window / pollster rule as the generic ballot."""
    base = ROOT / "data" / "raw" / "polls" / "538"
    f = base / ("president_approval_polls_historical.csv" if year <= 2020 else "president_approval_polls_biden.csv")
    if year < 2018 or not f.exists():
        return None, 0, None
    x = pd.read_csv(f, low_memory=False)
    x["end"] = pd.to_datetime(x.end_date, format="%m/%d/%y")
    eday = pd.Timestamp(ELECTION_DAY[year])
    x = x[(x.end < eday) & (x.end >= eday - pd.Timedelta(days=WINDOW_DAYS)) & x.partisan.isna()].copy()
    name = PRESIDENT[year][1]
    if "politician" in x:
        x = x[x.politician.astype(str).str.contains(name)]
    elif "politician_id" in x and "president" in x:
        x = x[x.president.astype(str).str.contains(name)]
    if not len(x):
        return None, 0, None
    rank = {"a": 0, "rv": 1, "lv": 2, "v": 3}
    x["pr"] = x.population.map(rank).fillna(4)
    x = x.sort_values("pr").groupby("poll_id").head(1)
    v, n = latest_per_pollster_mean(x, "yes")
    return v, n, (f"FiveThirtyEight {f.name}, mean of each pollster's latest non-partisan poll ending in the "
                  f"{WINDOW_DAYS} days before election day (adults > RV > LV within a poll)")


def house_popular_vote(year):
    h = pd.read_csv(RAW / "mit_1976-2024-house.tab", low_memory=False)
    h = h[(h.year == year) & (h.stage.str.upper() == "GEN") & (h.state_po != "DC")
          & (h.special.astype(str) != "True") & (h.runoff.astype(str) != "True")]
    lab = h.party.fillna("").str.upper()
    dv = h.candidatevotes[lab.isin(DEM_LABELS)].clip(lower=0).sum()
    rv = h.candidatevotes[lab.isin(REP_LABELS)].clip(lower=0).sum()
    tot = h.groupby(["state_po", "district"]).totalvotes.max().clip(lower=0).sum()
    return 100 * (rv - dv) / tot, 100 * (rv - dv) / (rv + dv)


def build_national():
    rp = pd.read_csv(NAT / "538_raw_polls.csv", low_memory=False)
    act538 = rp[rp.type_simple == "House-G-US"].groupby("cycle").margin_actual.first()
    rows = []
    for y in YEARS:
        gb, gbn, gbsrc = generic_ballot(y)
        ap, dis, a0, a1, apsrc = gallup_approval(y)
        apa, apn, apasrc = approval_poll_avg(y)
        pv, pv2 = house_popular_vote(y)
        rows.append(dict(
            year=y, midterm=int(y % 4 == 2), president_party=PRESIDENT[y][0], president=PRESIDENT[y][1],
            generic_ballot_margin_r_minus_d=round(gb, 2) if gb is not None else None,
            generic_ballot_n_pollsters=gbn, generic_ballot_source=gbsrc,
            approval_gallup=ap, disapproval_gallup=dis, approval_gallup_net=(ap - dis) if dis else None,
            approval_gallup_source=apsrc,
            approval_poll_avg=round(apa, 2) if apa is not None else None, approval_poll_avg_n_pollsters=apn,
            approval_poll_avg_source=apasrc,
            house_popular_vote_margin_r_minus_d=round(pv, 2),
            house_popular_vote_margin_two_party_r_minus_d=round(pv2, 2),
            house_popular_vote_source=("MEDSL 1976-2024 U.S. House returns, sum of D and R candidate votes over all "
                                       "435 November general elections / total votes (unopposed races that print no "
                                       "votes, e.g. Florida, count as zero)"),
            house_popular_vote_margin_538=(round(-act538[y], 2) if y in act538.index else None),
            house_popular_vote_538_source=("FiveThirtyEight raw_polls.csv margin_actual for House-G-US (sign flipped "
                                           "to R minus D)") if y in act538.index else None,
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
COLS = ["year", "chamber", "race_id", "state", "district", "special", "pvi_at_time", "incumbent_party",
        "incumbent_running", "open_seat", "cook_final", "sabato_final", "inside_elections_final",
        "cook_final_num", "sabato_final_num", "inside_elections_final_num", "ratings_imputed",
        "winner_party", "winner_name", "dem_share", "rep_share", "margin_r_minus_d", "uncontested", "notes",
        # extras
        "rating_mean", "pvi_text", "incumbent_name", "incumbent_status", "contest_type", "margin_kind",
        "share_source", "dem_votes", "rep_votes", "total_votes", "redistricted"]


COLUMN_DOC = {
    "race_id": "KS-01 / WY-AL / AZ-SEN; Senate specials XX-SEN-SPECIAL (538/polls history tables use XX-SEN-S)",
    "district": "01..53, AL (at-large) or SEN, as in race_info_2026.csv",
    "special": "1 = Senate special election (same November ballot)",
    "pvi_at_time": "Cook PVI published before the election on the lines in use, signed (+ = R)",
    "incumbent_party": "DEM/REP/IND/none; member-vs-member races: first-listed (same-numbered district's) member",
    "incumbent_running": "1 if an incumbent was on the November ballot as their party's candidate",
    "open_seat": "1 - incumbent_running",
    "cook_final/sabato_final/inside_elections_final": "last pre-election rating on Wikipedia (Rothenberg = IE)",
    "*_final_num": "Safe D -3, Likely D -2, Lean D -1, Tilt D -0.5, Tossup 0, ... Safe R +3; I = D side",
    "ratings_imputed": "1 = seat absent from the competitive-ratings table, set to Safe (see impute_rating)",
    "winner_party": "certified winner's party (NC-09 2018: November leader Mark Harris, result voided)",
    "dem_share/rep_share": "two-party fractions; see margin_kind for races where an independent stands in",
    "margin_r_minus_d": "100 * (rep_share - dem_share), percentage points",
    "uncontested": "1 = no candidate of the other major party and no independent >=35% (incl. same-party "
                   "top-two/jungle generals); shares then 1/0",
    "rating_mean": "mean of the three *_num columns",
    "contest_type": "D_vs_R, R_vs_IND, D_vs_IND, X_unopposed, X_vs_minor, D_vs_D, R_vs_R",
    "margin_kind": "two-party / final round (runoff, RCV) / R vs top non-R (...) / no major-party opponent / "
                   "same-party general",
    "share_source": "MEDSL, MEDSL runoff votes, Wikipedia final-round %, Wikipedia % (race missing from MEDSL)",
    "dem_votes/rep_votes/total_votes": "MEDSL November votes (D/R summed over fusion lines)",
    "redistricted": "1 = House district on new lines that year (2012, 2022 all; 2016 FL NC VA; 2018 PA; "
                    "2020 NC; 2024 AL GA LA NC NY)",
}


def main():
    meta = dict(columns=COLUMN_DOC, pages=json.loads((WIKI / "revids.json").read_text()),
                medsl=json.loads((RAW / "medsl_provenance.json").read_text()),
                redistricted_house=REDISTRICTED, rating_scale={k.title().replace(" D", " D").replace(" R", " R"): v
                                                               for k, v in RATING_NUM.items()})
    house, senate, senate_runoff = load_mit()
    h = build_house(house, meta)
    s = build_senate(senate, senate_runoff, meta)
    df = pd.concat([h, s], ignore_index=True)
    df["rating_mean"] = df[[f"{c}_final_num" for c in RATING_COLS]].mean(axis=1).round(3)
    for c in ("dem_share", "rep_share"):
        df[c] = df[c].round(4)
    df["margin_r_minus_d"] = df["margin_r_minus_d"].round(2)
    df["pvi_at_time"] = df["pvi_at_time"].astype("Int64") if (df.pvi_at_time.dropna() % 1 == 0).all() \
        else df["pvi_at_time"]
    df = df.sort_values(["year", "chamber", "race_id"], key=lambda c: c.map({"House": 0, "Senate": 1})
                        if c.name == "chamber" else c).reset_index(drop=True)
    df = df[COLS]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "history_races.csv", index=False)
    nat = build_national()
    nat.to_csv(OUT / "history_national.csv", index=False)
    (OUT / "history_races_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(df.groupby(["year", "chamber"]).size().unstack())
    print(nat[["year", "president_party", "generic_ballot_margin_r_minus_d", "approval_gallup",
               "approval_poll_avg", "house_popular_vote_margin_r_minus_d", "house_popular_vote_margin_538"]])


if __name__ == "__main__":
    main()
