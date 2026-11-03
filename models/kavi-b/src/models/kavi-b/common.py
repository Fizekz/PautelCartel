"""Shared paths, constants and name helpers for the kavi-b model (calibrated expert ratings + adjustments)."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CLEAN = ROOT / "data" / "clean"
BENCH = ROOT / "data" / "benchmark"
KALSHI = ROOT / "data" / "kalshi"
OWN = ROOT / "data" / "kavi-b"          # data this model collected itself (see data/kavi-b/README.md)
OUT = ROOT / "out"
KALSHI_STAMP = "2026-11-01T074812Z"

YEARS_HIST = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
ELECTION_DAY = {2010: "2010-11-02", 2012: "2012-11-06", 2014: "2014-11-04", 2016: "2016-11-08",
                2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05",
                2026: "2026-11-03"}
NAT_PRES = {2008: 0.53771, 2012: 0.52010, 2016: 0.51163, 2020: 0.52268, 2024: 0.49250}

FEC_SMOOTH = 5e4        # dollars added to each side before taking log ratios
FEC_CLIP = 4.0          # |log ratio| cap (one-sided races run to +-17 otherwise)
POLL_WINDOW_DAYS = 60   # 2010-2016 polls only exist for the final ~60 days, so use the same window every year
POLL_HALF_LIFE = 14.0
POLL_WINDOW_EXT_DAYS = 120   # v2: widened window when a race has fewer than MIN_POLLS_IN_WINDOW polls in the last 60 days
import os
MIN_POLLS_IN_WINDOW = int(os.environ.get("KB_MIN_POLLS", "2"))  # env override only for the window ablation
# v2: "other outlets" = every non-core rater Wikipedia lists, except 538 (the backtest benchmark is never an input)
OTHER_RATER_EXCLUDE = {"other_fte"}
MINOR_Q = 0.001         # probability given to each minor candidate that is not a principal or same-party rival
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def ascii_lower(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()


def name_tokens(name: str) -> list[str]:
    """Tokens of a name in 'First Last' or 'LAST, FIRST' form, returned as first..last."""
    text = ascii_lower(name)
    if "," in text:
        last, _, first = text.partition(",")
        text = f"{first} {last}"
    text = re.sub(r"\"[^\"]*\"|“[^”]*”|\([^)]*\)", " ", text)
    return [p for p in re.split(r"[^a-z]+", text) if p and p not in SUFFIXES]


def last_name(name: str) -> str:
    toks = name_tokens(name)
    return toks[-1] if toks else ""


def fec_last_name(name: str) -> str:
    """FEC names are 'LAST, FIRST MIDDLE SUFFIX'."""
    text = ascii_lower(name)
    last = text.split(",")[0] if "," in text else text.split()[-1] if text.split() else ""
    toks = [p for p in re.split(r"[^a-z]+", last) if p and p not in SUFFIXES]
    return toks[-1] if toks else ""


def clipped_log_ratio(dem: float, rep: float) -> float:
    d = 0.0 if dem is None or np.isnan(dem) else float(dem)
    r = 0.0 if rep is None or np.isnan(rep) else float(rep)
    return float(np.clip(np.log((d + FEC_SMOOTH) / (r + FEC_SMOOTH)), -FEC_CLIP, FEC_CLIP))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
