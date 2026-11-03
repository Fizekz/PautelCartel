"""Shared paths and small helpers for the pau-b forecast pipeline."""
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # pau/  (paths below are all inside pau/)
DATA = ROOT / "data"
SHARED = DATA / "build"                             # was the shared model/data folder: inputs of the 2026 feature build
FEAT = SHARED / "features"
RAW = SHARED / "raw"
KALSHI = DATA / "kalshi" / "kalshi_races_2026-11-01T074812Z.csv"
WORK = DATA                                         # was pbase/work: the model's input tables
OUT = ROOT / "out"
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

ELECTION_DATES = {
    2002: "2002-11-05", 2004: "2004-11-02", 2006: "2006-11-07",
    2008: "2008-11-04", 2010: "2010-11-02", 2012: "2012-11-06", 2014: "2014-11-04",
    2016: "2016-11-08", 2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08",
    2024: "2024-11-05", 2026: "2026-11-03",
}
# President's party in each cycle (+1 = D president, -1 = R president)
PRES_PARTY = {2008: -1, 2010: 1, 2012: 1, 2014: 1, 2016: 1, 2018: -1, 2020: -1,
              2022: 1, 2024: 1, 2026: -1}
MIDTERM = {y: int(y % 4 == 2) for y in ELECTION_DATES}

STATE_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT",
    "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT",
    "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
}


def norm_race_id(rid: str) -> str:
    """Normalise history race ids to the 538 file's convention (specials end in -SEN-S)."""
    rid = str(rid)
    rid = re.sub(r"-SEN-SPECIAL$", "-SEN-S", rid)
    rid = re.sub(r"-SEN-SP$", "-SEN-S", rid)
    return rid


def _fix_text(s: str) -> str:
    """Undo UTF-8-read-as-Latin-1 mojibake (MIT files: 'CÃRDENAS') and strip accents."""
    import unicodedata
    if "Ã" in s or "â" in s:
        try:
            s = s.encode("latin1").decode("utf8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm_name(s) -> str:
    """Lower-case surname-ish key for fuzzy candidate matching."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = _fix_text(str(s)).lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z\s\-']", " ", s)
    toks = [t for t in s.replace("-", " ").split() if t not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(toks)


def surname(s) -> str:
    t = norm_name(s).split()
    return t[-1] if t else ""


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))
