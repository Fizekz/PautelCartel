"""Top-two primary results for California and Washington House races (2012-2026) from Wikipedia.

Why: nine 2026 California races are same-party (D vs D or R vs R) generals. Expert ratings and presidential
lean say nothing about who wins those; the June primary does. The model learns P(primary leader wins the
same-party general) from 2012-2024 CA/WA races and applies it to 2026.

Sources (MediaWiki parse API, rendered HTML, cached in data/kavi-b/raw/wiki/):
  <Y> United States House of Representatives elections in California / Washington, Y = 2012..2024
  2026 United States House of Representatives elections in California (districts 1-26) / (districts 27-52)
Output: data/kavi-b/primaries_top_two.csv: year, state, race_id, candidate, party, votes, pct, page, revid

Usage: python src/models/kavi-b/fetch_primaries.py [--refresh]
"""
from __future__ import annotations

import argparse
import html as htmllib
import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from common import OWN

API = "https://en.wikipedia.org/w/api.php?action=parse&page={}&prop=text|revid&format=json&formatversion=2&redirects=1"
UA = {"User-Agent": "DemocracyDay-kavi-b-research/0.1 (python urllib)"}
RAW = OWN / "raw" / "wiki"
OUT_CSV = OWN / "primaries_top_two.csv"
STATES = {"California": "CA", "Washington": "WA"}
PAGES = [(y, st, f"{y} United States House of Representatives elections in {st}")
         for y in range(2012, 2025, 2) for st in STATES]
PAGES += [(2026, "California", "2026 United States House of Representatives elections in California (districts 1–26)"),
          (2026, "California", "2026 United States House of Representatives elections in California (districts 27–52)")]
PARTY = {"Democratic": "DEM", "Republican": "REP"}
DIST_RE = re.compile(r"(\d+)(?:st|nd|rd|th) congressional district|at-large congressional district")


def fetch(title: str, refresh: bool) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / (re.sub(r"[^A-Za-z0-9]+", "_", title) + ".json")
    if refresh or not path.exists():
        req = urllib.request.Request(API.format(urllib.parse.quote(title.replace(" ", "_"))), headers=UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
        time.sleep(1.0)
    return json.loads(path.read_text())


def clean(fragment: str) -> str:
    text = re.sub(r"<sup.*?</sup>", "", fragment, flags=re.S)
    text = htmllib.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()


HEAD_RE = re.compile(r'<h[234] id="District_(\d+)')


def parse_page(html: str, year: int, st: str) -> list[dict]:
    """Election boxes: a table is a primary box if its caption says primary, or it holds a 'Primary election'
    header row. The district comes from the nearest preceding 'District N' heading (the caption as fallback)."""
    heads = [(m.start(), m.group(1)) for m in HEAD_RE.finditer(html)]
    rows = []
    for tm in re.finditer(r"<table[^>]*>(.*?)</table>", html, re.S):
        body = tm.group(1)
        cap = re.search(r"<caption>(.*?)</caption>", body, re.S)
        caption = clean(cap.group(1)) if cap else ""
        prior = [d for pos, d in heads if pos < tm.start()]
        dm = DIST_RE.search(caption)
        district = prior[-1] if prior else (dm.group(1) if dm else None)
        if district is None:
            continue
        race_id = f"{STATES[st]}-{int(district):02d}"
        section = "primary" if "primary" in caption.lower() else None
        for tr in re.findall(r"(<tr[^>]*>.*?)</tr>", body, re.S):
            text = clean(tr)
            if re.fullmatch(r"(Primary election|Blanket primary election|Top-two primary|Nonpartisan blanket primary)", text, re.I):
                section = "primary"
                continue
            if re.fullmatch(r"General election", text, re.I):
                section = "general"
                continue
            if section != "primary" or "vcard" not in tr:
                continue
            party_m = re.search(r'class="org"[^>]*>(.*?)</td>', tr, re.S)
            name_m = re.search(r'class="fn"[^>]*>(.*?)</t[hd]>', tr, re.S)
            nums = re.findall(r'<td style="text-align:\s*right[^"]*">(.*?)</td>', tr, re.S)
            if not (party_m and name_m and len(nums) >= 2):
                continue
            votes = re.sub(r"[^\d]", "", clean(nums[0]))
            pct = re.sub(r"[^\d.]", "", clean(nums[1]))
            if not votes:
                continue
            rows.append({"year": year, "state": STATES[st], "race_id": race_id,
                         "candidate": re.sub(r"\s*\((?:incumbent|write-in)\)", "", clean(name_m.group(1))),
                         "party_label": clean(party_m.group(1)), "votes": int(votes),
                         "pct": float(pct) if pct else None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    out = []
    for year, st, title in PAGES:
        try:
            d = fetch(title, args.refresh)
        except Exception as exc:   # a missing page is reported, not fatal
            print(f"skip {title}: {exc}")
            continue
        if "parse" not in d:
            print(f"skip {title}: {d.get('error', {}).get('info')}")
            continue
        rows = parse_page(d["parse"]["text"], year, st)
        for r in rows:
            r["page"], r["revid"] = title, d["parse"].get("revid")
        print(f"{title}: {len(rows)} primary rows, {len({r['race_id'] for r in rows})} districts")
        out.extend(rows)
    df = pd.DataFrame(out).drop_duplicates(["year", "race_id", "candidate"])
    df["party"] = df["party_label"].map(PARTY).fillna("OTHER")
    df.to_csv(OUT_CSV, index=False)
    print(f"{len(df)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
