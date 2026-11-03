"""Parse California / Washington top-two primary and general results from Wikipedia pages.

History (2012-2024): pages fetched by fetch_sameparty.py into data/raw/wiki/.
2026: the shared Wikipedia JSON dumps in the shared raw/polls/wikipedia/ (primary only).
Output rows: year, state, district (int; 0 = Senate), stage (primary/general), candidate, party,
votes, pct, incumbent.
"""
import json
import re

import pandas as pd
from bs4 import BeautifulSoup

from common import ROOT, RAW, WORK

PARTY = {"Democratic": "DEM", "Republican": "REP"}


def _num(s):
    s = s.replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _rows(table):
    out = []
    marker = None
    for tr in table.find_all("tr"):
        c = [x.get_text(" ", strip=True) for x in tr.find_all(["td", "th"])]
        if len(c) == 1 and c[0].lower() in ("primary election", "general election"):
            marker = c[0].split()[0].lower()
            continue
        if len(c) >= 5 and _num(c[3]) is not None and _num(c[4]) is not None and c[2]:
            nm = c[2]
            inc = int("incumbent" in nm.lower())
            nm = re.sub(r"\s*\((incumbent|write-in)\)", "", nm, flags=re.I).strip()
            if "withdrawn" in nm.lower():
                nm = re.sub(r"\s*\(withdrawn\)", "", nm, flags=re.I).strip()
            out.append(dict(candidate=nm, party=PARTY.get(c[1], c[1] or "OTH"), votes=_num(c[3]),
                            pct=_num(c[4]), incumbent=inc, marker=marker))
    return out


def parse_html(html, year, state, senate=False):
    soup = BeautifulSoup(html, "lxml")
    res = []
    seen = set()
    for t in soup.find_all("table"):
        cap = t.find("caption")
        cap = cap.get_text(" ", strip=True).lower() if cap else ""
        if "←" in t.get_text()[:40] or not _rows(t):
            continue
        head = t.find_previous(["h3", "h4"])
        head = head.get_text(" ", strip=True).lower() if head else ""
        h2 = t.find_previous("h2")
        h2 = h2.get_text(" ", strip=True).lower() if h2 else ""
        if "primary" in cap or ("primary" in head and "general" not in head):
            stage = "primary"
        elif "general" in head or "general" in cap or "congressional district" in cap or "senate election" in cap \
                or "results" in head:
            stage = "general"
        else:
            continue
        if "special" in h2 or "special" in cap or "runoff" in head:
            continue
        if senate:
            dist = 0
        else:
            h = t.find_previous("h2")
            m = re.search(r"District (\d+)", h.get_text(" ", strip=True)) if h else None
            if not m:
                m = re.search(r"(\d+)(st|nd|rd|th) congressional district", cap)
            if not m:
                continue
            dist = int(m.group(1))
        rows = _rows(t)
        for stg in ("primary", "general"):
            sub = [r for r in rows if (r["marker"] or stage) == stg]
            key = (dist, stg)
            if not sub or key in seen:
                continue
            seen.add(key)
            for r in sub:
                r = {k: v for k, v in r.items() if k != "marker"}
                res.append(dict(year=year, state=state, district=dist, stage=stg, **r))
    return res


def history():
    rows = []
    d = ROOT / "data/raw/wiki"
    for y in range(2012, 2026, 2):
        for st, name in (("CA", "California"), ("WA", "Washington")):
            p = d / f"{y}_United_States_House_of_Representatives_elections_in_{name}.html"
            if p.exists():
                rows += parse_html(p.read_text(encoding="utf-8"), y, st)
    for y in (2016, 2018):
        p = d / f"{y}_United_States_Senate_election_in_California.html"
        if p.exists():
            rows += parse_html(p.read_text(encoding="utf-8"), y, "CA", senate=True)
    return pd.DataFrame(rows)


def primaries_2026():
    rows = []
    base = RAW / "polls/wikipedia"
    for fn, st in [("2026_United_States_House_of_Representatives_elections_in_California_districts_126.json", "CA"),
                   ("2026_United_States_House_of_Representatives_elections_in_California_districts_2752.json", "CA"),
                   ("2026_United_States_House_of_Representatives_elections_in_Washington.json", "WA")]:
        p = base / fn
        if p.exists():
            rows += [r for r in parse_html(json.load(open(p))["text"], 2026, st) if r["stage"] == "primary"]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    h = history()
    h.to_csv(WORK / "toptwo_hist.csv", index=False)
    p = primaries_2026()
    p.to_csv(WORK / "toptwo_2026.csv", index=False)
    print(h.groupby(["year", "state", "stage"]).district.nunique().unstack())
    print(p.groupby(["state"]).district.nunique())
