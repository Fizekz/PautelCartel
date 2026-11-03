"""Fetch the June 2 2026 top-two primary results for the 9 same-party California House races (Ballotpedia).

Saves raw HTML to data/kavi-a/raw/ballotpedia_CA-XX.html and writes data/kavi-a/ca_primaries_2026.csv
(race_id, candidate, party_label, primary_pct). Pre-election information (the primary was in June).
Usage: python fetch_ca_primaries.py [--offline]
"""
from __future__ import annotations

import argparse
import html
import re
import time
import urllib.request

import pandas as pd

from common import ROOT

RACES = {"CA-04": "4th", "CA-07": "7th", "CA-11": "11th", "CA-12": "12th", "CA-14": "14th", "CA-29": "29th",
         "CA-34": "34th", "CA-37": "37th", "CA-40": "40th"}
RAW = ROOT / "data" / "kavi-a" / "raw"
OUT = ROOT / "data" / "kavi-a" / "ca_primaries_2026.csv"
UA = "Mozilla/5.0 (research script; DemocracyDay forecasting class project)"


def fetch(race_id: str, ordinal: str) -> str:
    path = RAW / f"ballotpedia_{race_id}.html"
    if path.exists():
        return path.read_text()
    url = f"https://ballotpedia.org/California%27s_{ordinal}_Congressional_District"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8", "replace")
    path.write_text(text)
    time.sleep(1.0)
    return text


def primary_rows(race_id: str, page: str) -> list[dict]:
    """Rows of the first 'Nonpartisan primary election ... 2026' results table."""
    start = page.find("Nonpartisan primary election")
    while start != -1 and "2026" not in page[max(0, start - 400):start + 400]:
        start = page.find("Nonpartisan primary election", start + 10)
    if start == -1:
        return []
    seg = page[start:start + 40000]
    end = seg.find("</table>")
    seg = seg[:end] if end > 0 else seg
    rows = []
    for tr in re.findall(r"<tr.*?</tr>", seg, flags=re.S):
        text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", tr))).strip()
        m = re.search(r"^(?:\S+\s)?(.+?)\s\((D|R|Party preference: [^)]+|No party preference|Independent|"
                      r"Democratic Party|Republican Party|[A-Za-z ]+)\)\s+([\d.]+)\s*%?\s+([\d,]+)", text)
        if m:
            rows.append({"race_id": race_id, "candidate": re.sub(r"^photo ", "", m.group(1).strip(" ✔")), "party_label": m.group(2),
                         "primary_pct": float(m.group(3)), "votes": int(m.group(4).replace(",", ""))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for rid, ordinal in RACES.items():
        rows += primary_rows(rid, fetch(rid, ordinal))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
