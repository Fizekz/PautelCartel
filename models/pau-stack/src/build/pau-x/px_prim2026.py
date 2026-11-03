"""pau-x item 3, 2026: House primary results per district from the shared Wikipedia JSON dumps
(raw/polls/wikipedia/2026_United_States_House_of_Representatives_election(s)_in_*.json, fetched 2026-11-01 through
the MediaWiki API by the data layer). Tables captioned "<Party> primary results"; top-two / blanket primaries (CA, WA)
are summed by the candidates' party labels. All 2026 primaries were held before election day (as-of safe).
Output: work/px/primaries_2026.csv (state, dist, dv, rv)."""
import json
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import topttwo as T2
from common import RAW, WORK

ORD = re.compile(r"(\d+)(?:st|nd|rd|th) congressional district", re.I)
NAMES = {v: k for k, v in __import__("common").STATE_ABBR.items()}


def parse(html):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    dist = None
    for t in soup.find_all("table"):
        cap = t.find("caption")
        cap = cap.get_text(" ", strip=True) if cap else ""
        m = ORD.search(cap)
        if m and "primary" not in cap.lower():
            dist = int(m.group(1))
            continue
        if "at-large congressional district" in cap.lower() and "primary" not in cap.lower():
            dist = 1
            continue
        if "primary" not in cap.lower() or "runoff" in cap.lower() or "special" in cap.lower():
            continue
        h = t.find_previous(["h2"])
        hm = re.search(r"District (\d+)", h.get_text(" ", strip=True)) if h else None
        d = dist if dist is not None else (int(hm.group(1)) if hm else None)
        if hm and dist is None:
            d = int(hm.group(1))
        if d is None:
            continue
        lc = cap.lower()
        party = "D" if lc.startswith("democratic") else "R" if lc.startswith("republican") else None
        rows = T2._rows(t)
        for r in rows:
            if r.get("marker") == "general":
                continue
            p = party
            if p is None:
                p = "D" if r["party"] == "DEM" else "R" if r["party"] == "REP" else "O"
            out.append(dict(dist=d, side=p, votes=r["votes"], prim_party=party or "all"))
    return pd.DataFrame(out)


def build():
    base = RAW / "polls" / "wikipedia"
    rows = []
    for f in sorted(base.glob("2026_United_States_House_of_Representatives_election*_in_*.json")):
        stname = re.sub(r"_districts_\d+$", "", f.stem.split("_in_")[-1]).replace("_", " ")
        st = __import__("common").STATE_ABBR.get(stname.upper())
        if st is None or st == "LA":
            continue
        d = parse(json.load(open(f))["text"])
        if not len(d):
            continue
        d["state"] = st
        rows.append(d)
    d = pd.concat(rows)
    g = d.pivot_table(index=["state", "dist"], columns="side", values="votes", aggfunc="sum").reset_index()
    g = g.rename(columns={"D": "dv", "R": "rv"})
    g.to_csv(WORK / "px" / "primaries_2026.csv", index=False)
    return g


if __name__ == "__main__":
    g = build()
    print(len(g), "districts;", int(((g.dv > 0) & (g.rv > 0)).sum()), "with both parties counted")
    print(g.head(10).to_string(index=False))
