"""Collect 2026 California top-two primary (June 2, 2026) results per district from Wikipedia.

Source: the MediaWiki parse API pages
  "2026 United States House of Representatives elections in California (districts 1–26)"
  "2026 United States House of Representatives elections in California (districts 27–52)"
cached as JSON in data/extra/raw/ (revid inside). Output: data/extra/ca_primary_2026.csv
  race_id, candidate, party, votes, pct
Only used for the 9 same-party CA generals, where the party model has nothing to say. See data/extra/ca_primary_2026.md.

Usage: python src/models/kavi-d/fetch_ca_primary.py [--refresh]
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "data"))
from wiki_fetch import fetch_page  # noqa: E402

RAW = ROOT / "data" / "extra" / "raw"
OUT = ROOT / "data" / "extra" / "ca_primary_2026.csv"
PAGES = ["2026 United States House of Representatives elections in California (districts 1–26)",
         "2026 United States House of Representatives elections in California (districts 27–52)"]
PARTIES = ["Democratic", "Republican", "No party preference", "Green", "Libertarian", "Peace and Freedom",
           "American Independent"]
ROW = re.compile(r"(" + "|".join(PARTIES) + r") (.+?) (\d{1,3}(?:,\d{3})*) (\d{1,3}\.\d+)")


def plain(h: str) -> str:
    t = re.sub(r"<[^>]+>", " ", h)
    t = html.unescape(t)
    t = re.sub(r"\[\s*\d+\s*\]", " ", t)
    return re.sub(r"\s+", " ", t)


def parse(text: str) -> list[dict]:
    rows = []
    parts = re.split(r"District (\d{1,2}) \[ edit \| edit source \]", text)
    for i in range(1, len(parts), 2):
        dist, body = int(parts[i]), parts[i + 1]
        k = body.find("Primary results")
        if k < 0:
            continue
        seg = body[k: body.find("Total votes", k)]
        for m in ROW.finditer(seg):
            name = re.sub(r"\s*\(incumbent\)", "", m.group(2)).strip()
            rows.append({"race_id": f"CA-{dist:02d}", "candidate": name, "party": m.group(1),
                         "votes": int(m.group(3).replace(",", "")), "pct": float(m.group(4))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    rows, revs = [], []
    for title in PAGES:
        page = fetch_page(title, RAW, refresh=args.refresh)
        revs.append(page["revid"])
        rows += parse(plain(page["html"]))
    df = pd.DataFrame(rows).drop_duplicates(["race_id", "candidate"])
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows, {df['race_id'].nunique()} districts, revids {revs}")


if __name__ == "__main__":
    main()
