"""Shared paths, constants and small helpers for the kavi-a fundamentals model."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CLEAN = ROOT / "data" / "clean"
BENCH = ROOT / "data" / "benchmark"
KALSHI_RACES = ROOT / "data" / "kalshi" / "races_2026-11-01T074812Z.csv"
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"
OUT = ROOT / "out"

SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
ELECTION_DAY = {2010: "2010-11-02", 2012: "2012-11-06", 2014: "2014-11-04", 2016: "2016-11-08",
                2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05",
                2026: "2026-11-03"}
# national two-party D presidential share (MIT), used to make lean relative
NAT_PRES = {2004: 0.48767, 2008: 0.53771, 2012: 0.52010, 2016: 0.51163, 2020: 0.52268, 2024: 0.49250}

# states whose House lines changed going into year Y (relative to Y-2). 2012/2022: all (decennial).
REDRAWN = {2012: "ALL", 2014: set(), 2016: {"FL", "VA", "NC"}, 2018: {"PA"}, 2020: {"NC"}, 2022: "ALL",
           2024: {"AL", "GA", "LA", "NC", "NY"}, 2026: {"AL", "CA", "FL", "LA", "NC", "OH", "TN", "TX", "UT"}}
# appointed senators running in a special whom the FEC flags as open-seat ('O'); every other appointee
# (Bennet, Gillibrand, Heller, Schatz, Smith, Loeffler, McSally, and 2026 Moody/Husted) is flagged 'I'
APPOINTED_INCUMBENTS = {(2018, "MS-SEN-S"): "hyde"}
MONEY_SMOOTH = 2.5e4      # $ added to both sides before the log receipts ratio
MONEY_CLIP = 3.5          # clip |ln ratio| (one-sided races run to +-17 otherwise)
SHARE_CLIP = (0.02, 0.98)


def tokens(name: str) -> list[str]:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return [p for p in re.split(r"[^a-z]+", text) if p and p not in SUFFIXES]


def last_name(name: str) -> str:
    """Surname of 'FIRST M LAST' style names."""
    t = tokens(name)
    return t[-1] if t else ""


def fec_last_name(name: str) -> str:
    """Surname of FEC 'LAST, FIRST' names (last token of the part before the comma)."""
    head = str(name).split(",")[0]
    t = tokens(head)
    return t[-1] if t else ""


def lines_unchanged(year: int, state: str) -> bool:
    red = REDRAWN.get(year, set())
    return not (red == "ALL" or state in red)


def money_feature(dem: pd.Series, rep: pd.Series) -> pd.Series:
    d = pd.to_numeric(dem, errors="coerce").fillna(0.0).clip(lower=0)
    r = pd.to_numeric(rep, errors="coerce").fillna(0.0).clip(lower=0)
    return np.log((d + MONEY_SMOOTH) / (r + MONEY_SMOOTH)).clip(-MONEY_CLIP, MONEY_CLIP)


def national_table() -> pd.DataFrame:
    nat = pd.read_csv(CLEAN / "national_hist.csv")
    nat["house_d2p"] = nat["house_dem_share"] / (nat["house_dem_share"] + nat["house_rep_share"])
    nat["midterm"] = (nat["cycle_type"] == "midterm").astype(int)
    nat["pres_d"] = np.where(nat["president_party"] == "D", 1, -1)
    return nat.set_index("year")
