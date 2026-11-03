"""Parse The Downballot special-election sheets into one clean table.

Input: data/specials/raw/<cycle>_<gid>.csv (from fetch_specials.py). Output: data/specials/specials_clean.csv with
one row per D-vs-R special election: date, state, district, special D-R margin, the most recent presidential D-R
margin in that district, and overperf = special margin - presidential margin (points).

Inclusion follows The Downballot's own averaging rule: rows marked '*' are excluded and D% + R% must be >= 90.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "specials" / "raw"
OUT = ROOT / "data" / "specials" / "specials_clean.csv"

# (file, baseline presidential year of the first presidential block)
TABS = [
    ("2017_18_0.csv", 2016), ("2017_18_2144047916.csv", 2016),
    ("2019_20_0.csv", 2016), ("2019_20_539852412.csv", 2016),
    ("2021_22_0.csv", 2020), ("2021_22_523128020.csv", 2020),
    ("2023_24_0.csv", 2020), ("2023_24_995309150.csv", 2020),
    ("2025_26_415249345.csv", 2024), ("2025_26_1173601967.csv", 2024),
]
MIN_TWO_PARTY = 0.90


def pct(text: str) -> float | None:
    text = (text or "").strip().replace("%", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_tab(path: Path, base_year: int) -> list[dict]:
    rows = list(csv.reader(path.open(encoding="utf-8", errors="ignore")))
    header_idx = next(i for i, r in enumerate(rows) if "Date" in r and "State" in r)
    header = rows[header_idx]
    i_date, i_state, i_dist = header.index("Date"), header.index("State"), header.index("District")
    i_d = header.index("D %age")
    out = []
    for r in rows[header_idx + 1:]:
        if len(r) <= i_d + 7 or not r[i_date].strip() or not r[i_state].strip():
            continue
        try:
            date = pd.to_datetime(r[i_date].strip(), format="%d-%b-%y")
        except ValueError:
            continue
        d, rep = pct(r[i_d]), pct(r[i_d + 1])
        p_d, p_r = pct(r[i_d + 4]), pct(r[i_d + 5])
        if None in (d, rep, p_d, p_r):
            continue
        dist = r[i_dist].strip()
        out.append({
            "date": date.date().isoformat(), "state": r[i_state].strip(), "district": dist.replace("*", "").strip(),
            "excluded_star": "*" in dist, "dem_pct": d, "rep_pct": rep, "special_margin": d - rep,
            "pres_dem_pct": p_d, "pres_rep_pct": p_r, "pres_margin": p_d - p_r, "pres_base_year": base_year,
            "overperf": (d - rep) - (p_d - p_r),
            "is_federal": dist.replace("*", "").strip().startswith(r[i_state].strip() + "-"),
            "source_file": path.name,
        })
    return out


def main() -> int:
    rows = [row for name, base in TABS for row in parse_tab(RAW / name, base)]
    df = pd.DataFrame(rows)
    df["included"] = (~df["excluded_star"]) & ((df["dem_pct"] + df["rep_pct"]) >= MIN_TWO_PARTY * 100)
    df = df.drop_duplicates(["date", "state", "district"]).sort_values("date")
    df.to_csv(OUT, index=False)
    inc = df[df["included"]].copy()
    inc["year"] = inc["date"].str[:4]
    print(inc.groupby("year")["overperf"].agg(["count", "mean", "median"]).round(2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
