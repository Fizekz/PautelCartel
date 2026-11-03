"""Shared paths, constants and helpers for the kavi-c polls-first Bayesian aggregator."""
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
# Pre-election FEC totals for 2018 (Wayback capture of weball18.zip, 2018-11-07 08:07 UTC; borrowed from kavi-b,
# see data/kavi-c/README.md). No pre-election capture exists for 2022.
FEC_2018_PRE = ROOT / "data" / "kavi-c" / "fec_2018_preelection.csv"
# v3: 2022 cut (Wayback webl22.zip captured 2022-08-03, coverage to 2022-06-30), borrowed from kavi-a.
FEC_2022_PRE = ROOT / "data" / "kavi-c" / "fec_2022_preelection.csv"

HIST_YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
ELECTION_DAY = {2010: "2010-11-02", 2012: "2012-11-06", 2014: "2014-11-04", 2016: "2016-11-08",
                2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05",
                2026: "2026-11-03"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

# Money feature: ln((D + SMOOTH) / (R + SMOOTH)), clipped (FEC notes: log ratio is extreme in one-sided races).
MONEY_SMOOTH = 25_000.0
MONEY_CLIP = 3.0
# Poll window and weights.
POLL_WINDOW_DAYS = 75          # 2010-16 538 archive keeps ~60 days; keep the same horizon every year
POLL_HALFLIFE_DAYS = 21.0
POP_WEIGHT = {"lv": 1.0, "v": 0.9, "rv": 0.8, "a": 0.5}
PARTISAN_WEIGHT = 0.6


def ascii_lower(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()


def name_tokens(name: str) -> list[str]:
    return [p for p in re.split(r"[^a-z]+", ascii_lower(name)) if p and p not in SUFFIXES]


def last_name(name: str) -> str:
    """Surname of 'First Last', 'LAST, FIRST' (FEC) or 'Last'."""
    text = str(name)
    if "," in text:
        toks = name_tokens(text.split(",")[0])
        return toks[-1] if toks else ""
    toks = name_tokens(text)
    return toks[-1] if toks else ""


def full_name(name: str) -> str:
    return " ".join(name_tokens(name))


def norm_pollster(name: str) -> str:
    """Crude pollster key shared by 538 and Wikipedia spellings."""
    stop = {"the", "inc", "llc", "polling", "poll", "polls", "research", "group", "strategies", "associates",
            "and", "of", "for", "company", "co", "analytics", "insights", "survey", "surveys", "opinion", "center",
            "institute", "partners", "consulting", "communications", "public", "llp", "media"}
    text = re.sub(r"\(.*?\)", " ", ascii_lower(name))
    toks = [t for t in re.split(r"[^a-z0-9]+", text) if t and t not in stop]
    return " ".join(toks[:2]) if toks else ascii_lower(name).strip()


def money_feature(dem: pd.Series, rep: pd.Series) -> pd.Series:
    d = pd.to_numeric(dem, errors="coerce").fillna(0.0).clip(lower=0)
    r = pd.to_numeric(rep, errors="coerce").fillna(0.0).clip(lower=0)
    return np.log((d + MONEY_SMOOTH) / (r + MONEY_SMOOTH)).clip(-MONEY_CLIP, MONEY_CLIP)


def senate_state(race_id: str) -> str:
    return race_id.split("-")[0]
