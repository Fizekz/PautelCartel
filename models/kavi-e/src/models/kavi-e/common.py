"""Shared paths, constants and name helpers for the kavi-e national-environment + swing model."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CLEAN = ROOT / "data" / "clean"
BENCH = ROOT / "data" / "benchmark"
KALSHI = ROOT / "data" / "kalshi"
OUT = ROOT / "out"
WORK = OUT / "work"

ELECTION_DAY = {2010: "2010-11-02", 2012: "2012-11-06", 2014: "2014-11-04", 2016: "2016-11-08",
                2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05",
                2026: "2026-11-03"}
MIDTERMS = (2010, 2014, 2018, 2022, 2026)
# National two-party Dem presidential share (MIT), for converting district shares to relative lean.
NAT_PRES_2P = {2004: 0.48767, 2008: 0.53771, 2012: 0.52010, 2016: 0.51163, 2020: 0.52268, 2024: 0.49250}
# States whose House lines changed going INTO year Y (so Y-2 House results are not on Y lines).
REDRAWN_INTO = {
    2012: "ALL", 2022: "ALL",
    2016: {"FL", "VA", "NC"}, 2018: {"PA"}, 2020: {"NC"}, 2024: {"AL", "GA", "LA", "NC", "NY"},
    2010: set(), 2014: set(),
}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
FEC_CLIP = 3.0  # clip of ln(D receipts / R receipts)
KALSHI_RACES = KALSHI / "races_2026-11-01T074812Z.csv"
KALSHI_MARKETS = KALSHI / "markets_2026-11-01T074812Z.csv"


# v2 (audit bug 1): incumbents the name/FEC match misses: appointed senators filed by FEC as open-seat candidates,
# and House special-election winners (results_house holds no specials; FEC codes Letlow 'C'). Ballot facts known
# before each election. (year, race_id) -> (party, surname token).
INCUMBENT_OVERRIDES = {
    (2018, "MS-SEN-S"): ("REP", "hyde"),     # Cindy Hyde-Smith, appointed April 2018
    (2020, "AZ-SEN-S"): ("REP", "mcsally"),  # Martha McSally, appointed Jan 2019
    (2020, "AZ-SEN"): ("REP", "mcsally"),
    (2022, "LA-05"): ("REP", "letlow"),      # Julia Letlow, special 2021
    (2022, "LA-02"): ("DEM", "carter"),      # Troy Carter, special 2021
    (2022, "NM-01"): ("DEM", "stansbury"),   # Melanie Stansbury, special 2021
}


def override_incumbent(year: int, race_id: str, party: str, name: str) -> bool:
    o = INCUMBENT_OVERRIDES.get((int(year), race_id))
    return bool(o) and o[0] == party and isinstance(name, str) and o[1] in name_tokens(name)


def ascii_lower(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()


def name_tokens(name: str) -> list[str]:
    """Lower-case tokens; handles 'LAST, FIRST' (FEC) by moving the surname to the end."""
    text = ascii_lower(name)
    if "," in text:
        last, _, first = text.partition(",")
        text = f"{first} {last}"
    text = re.sub(r"\([^)]*\)", " ", text)
    return [t for t in re.split(r"[^a-z]+", text) if t and t not in SUFFIXES]


def last_name(name: str) -> str:
    toks = name_tokens(name)
    return toks[-1] if toks else ""


def same_person(a: str, b: str) -> bool:
    """Surname equal and (first initials equal or one side has no first name)."""
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb or ta[-1] != tb[-1]:
        # hyphenated / two-word surnames: accept when the other's surname is in this one's token set
        return bool(ta and tb) and (ta[-1] in tb or tb[-1] in ta) and ta[0][0] == tb[0][0]
    return len(ta) == 1 or len(tb) == 1 or ta[0][0] == tb[0][0] or ta[0] in tb or tb[0] in ta


def clip_log_ratio(dem: float, rep: float) -> float:
    if not np.isfinite(dem) or not np.isfinite(rep):
        return 0.0
    return float(np.clip(np.log((dem + 1e4) / (rep + 1e4)), -FEC_CLIP, FEC_CLIP))


def redrawn(year: int, state: str) -> bool:
    spec = REDRAWN_INTO.get(year, set())
    return spec == "ALL" or state in spec


def ensure_dirs() -> None:
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(CLEAN / name)
