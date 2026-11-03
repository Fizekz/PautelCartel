"""General-election district polls from Wikipedia's 2026 California House sub-pages.

Data trap found here: the shared polls_2026.csv parsed the statewide "2026 ... elections in California" page,
but since mid-2026 Wikipedia keeps the per-district sections (candidates, primary results, polls) on two
sub-pages, "(districts 1–26)" and "(districts 27–52)". This script reads those two cached pages (see
fetch_primaries.py, which downloads them) and writes every general-election poll table it finds.

Output: data/kavi-b/ca_polls_2026.csv: race_id, pollster, dates, sample, cand_a, pct_a, cand_b, pct_b, page
Usage: python src/models/kavi-b/fetch_ca_polls_2026.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

from common import OWN, ROOT

sys.path.insert(0, str(ROOT / "src" / "data"))
from html_tables import extract_tables  # noqa: E402

PAGES = ["2026_United_States_House_of_Representatives_elections_in_California_districts_1_26_.json",
         "2026_United_States_House_of_Representatives_elections_in_California_districts_27_52_.json"]
HEAD_RE = re.compile(r'<h[234] id="District_(\d+)"')
GEN_RE = re.compile(r'<h[2345] id="(General_election[^"]*|Primary[^"]*|Polling[^"]*)"')
CAND_RE = re.compile(r"^(.+?) \(([DRI])\)$")
PCT_RE = re.compile(r"^(\d+(?:\.\d+)?)%")
OUT_CSV = OWN / "ca_polls_2026.csv"


def district_chunks(html: str):
    heads = [(m.start(), m.group(1)) for m in HEAD_RE.finditer(html)]
    for i, (pos, dist) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(html)
        yield f"CA-{int(dist):02d}", html[pos:end]


def general_part(chunk: str) -> str:
    """Keep only the text after the 'General election' heading (primary polls come before it)."""
    m = re.search(r'<h[234] id="General_election', chunk)
    return chunk[m.start():] if m else ""


def polls_from_chunk(race_id: str, chunk: str, page: str) -> list[dict]:
    rows = []
    for grid in extract_tables(general_part(chunk)):
        if not grid or not grid[0] or not grid[0][0].startswith("Poll source"):
            continue
        header = grid[0]
        cands = [(j, CAND_RE.match(h)) for j, h in enumerate(header)]
        cands = [(j, m.group(1), m.group(2)) for j, m in cands if m]
        if len(cands) != 2:      # keep clean two-way head-to-heads only
            continue
        for r in grid[1:]:
            if len(r) < len(header) or r[0] == header[0]:
                continue
            vals = [PCT_RE.match(r[j]) for j, _, _ in cands]
            if not all(vals):
                continue
            rows.append({"race_id": race_id, "pollster": r[0], "dates": r[1], "sample": r[2],
                         "cand_a": cands[0][1], "party_a": cands[0][2], "pct_a": float(vals[0].group(1)),
                         "cand_b": cands[1][1], "party_b": cands[1][2], "pct_b": float(vals[1].group(1)),
                         "page": page})
    return rows


def end_date(dates: str) -> pd.Timestamp | None:
    """'October 16–24, 2026' -> 2026-10-24; 'September 28 – October 2, 2026' -> 2026-10-02."""
    m = re.search(r"([A-Z][a-z]+)?\s*(\d{1,2}),\s*(\d{4})\s*$", dates.replace("–", " – "))
    month = re.findall(r"[A-Z][a-z]+", dates)
    if not m or not month:
        return None
    try:
        return pd.Timestamp(f"{month[-1]} {m.group(2)} {m.group(3)}")
    except ValueError:
        return None


def main() -> None:
    out = []
    for name in PAGES:
        html = json.loads((OWN / "raw" / "wiki" / name).read_text())["parse"]["text"]
        for race_id, chunk in district_chunks(html):
            out.extend(polls_from_chunk(race_id, chunk, name))
    df = pd.DataFrame(out)
    df["end_date"] = df["dates"].map(end_date)
    df.to_csv(OUT_CSV, index=False)
    shared = pd.read_csv(ROOT / "data" / "clean" / "polls_2026.csv")
    shared = shared[shared["race_id"].str.startswith("CA-")]
    print(f"{len(df)} two-way general polls in {df['race_id'].nunique()} CA districts -> {OUT_CSV}")
    print(f"shared polls_2026.csv has {len(shared)} CA rows in {shared['race_id'].nunique()} districts")
    print(df.groupby("race_id").size().to_string())


if __name__ == "__main__":
    main()
