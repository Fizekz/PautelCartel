"""Shared helpers for the kavi-d model: paths, name matching, constants."""
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
CACHE = OUT / "cache"

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "md", "phd"}
ELECTION_DAY = {2010: "2010-11-02", 2012: "2012-11-06", 2014: "2014-11-04", 2016: "2016-11-08",
                2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05",
                2026: "2026-11-03"}
# States whose House lines changed in that cycle (same district id != same district).
REDRAW = {2012: "ALL", 2014: set(), 2016: {"FL", "VA", "NC"}, 2018: {"PA"}, 2020: {"NC"}, 2022: "ALL",
          2024: {"AL", "GA", "LA", "NC", "NY"}, 2026: {"AL", "CA", "FL", "LA", "NC", "OH", "TN", "TX", "UT"}}
TRAIN_YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
POLL_WINDOW_DAYS = 60
POLL_WINDOW_SPARSE_DAYS = int(__import__("os").environ.get("KAVID_POLL_SPARSE_DAYS", 60))
POLL_SPARSE_MIN = 2  # fewer matched polls than this in the 60-day window -> use the sparse window
FEC_LR_CLIP = 4.0
EPS_MINOR = 0.0005  # q for a minor / non-viable candidate


def redrawn(year: int, state: str) -> bool:
    r = REDRAW.get(year, set())
    return r == "ALL" or state in r


def tokens(name) -> list[str]:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return []
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\(.*?\)", " ", text)  # drop "(IND)" / nicknames in parentheses
    return [t for t in re.split(r"[^a-z]+", text) if t and t not in SUFFIXES]


def surname(name) -> str:
    t = tokens(name)
    return t[-1] if t else ""


def fec_surname_tokens(fec_name) -> list[str]:
    """FEC names are 'LAST, FIRST MIDDLE'; return the tokens of the LAST part."""
    s = str(fec_name)
    last = s.split(",")[0] if "," in s else s.split()[-1] if s.split() else ""
    return tokens(last)


def name_matches(poll_or_fec_tokens: list[str], cand_name) -> bool:
    """True when any surname token of one appears in the candidate's name tokens."""
    ct = tokens(cand_name)
    if not ct or not poll_or_fec_tokens:
        return False
    if ct[-1] in poll_or_fec_tokens:
        return True
    joined = "".join(poll_or_fec_tokens)
    return joined in {"".join(ct[i:]) for i in range(len(ct))} or poll_or_fec_tokens[-1] in ct


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def national_house_margin() -> pd.Series:
    """National House D-R vote margin (points of total vote) by year, from MIT results."""
    h = pd.read_csv(CLEAN / "results_house.csv")
    g = h.groupby("year")[["dem_votes", "rep_votes", "total_votes"]].sum()
    return 100 * (g["dem_votes"] - g["rep_votes"]) / g["total_votes"]


def generic_margin(year: int) -> float:
    """Final pre-election generic-ballot D-R average (an input known before election day)."""
    if year == 2026:
        n = pd.read_csv(CLEAN / "national_2026.csv")
        a = n[n["metric"] == "generic_ballot_avg"]
        return float((a["value_dem"] - a["value_rep"]).mean())
    nh = pd.read_csv(CLEAN / "national_hist.csv").set_index("year")
    return float(nh.loc[year, "generic_margin_dr"])


def net_approval(year: int) -> float:
    if year == 2026:
        n = pd.read_csv(CLEAN / "national_2026.csv")
        return float(n.loc[n.metric == "approve_avg", "value"].mean() - n.loc[n.metric == "disapprove_avg", "value"].mean())
    nh = pd.read_csv(CLEAN / "national_hist.csv").set_index("year")
    return float(nh.loc[year, "net_approval"])


def pres_party(year: int) -> str:
    if year == 2026:
        return "R"
    nh = pd.read_csv(CLEAN / "national_hist.csv").set_index("year")
    return str(nh.loc[year, "president_party"])


def midterm_outparty_sign(year: int) -> float:
    """+1 if `year` is a midterm under a Republican president (anti-R side = out-party), -1 under a Democratic
    president, 0 in presidential years."""
    if year == 2026:
        return 1.0
    nh = pd.read_csv(CLEAN / "national_hist.csv").set_index("year")
    if nh.loc[year, "cycle_type"] != "midterm":
        return 0.0
    return 1.0 if nh.loc[year, "president_party"] == "R" else -1.0


def fec_given_tokens(fec_name) -> set[str]:
    """Given-name tokens of an FEC 'LAST, FIRST MIDDLE' name."""
    s = str(fec_name)
    return set(tokens(s.split(",", 1)[1])) if "," in s else set()


def prefer_given_name(hit: pd.DataFrame, cand_name, name_col: str = "cand_name") -> pd.DataFrame:
    """Among surname matches, keep those whose given names overlap the candidate's (WA-09 2018: Adam Smith vs
    Sarah Smith both matched 'SMITH'). Falls back to all surname matches when none overlaps."""
    if len(hit) <= 1:
        return hit
    given = set(tokens(cand_name)[:-1])
    if not given:
        return hit
    keep = hit[hit[name_col].map(lambda n: bool(fec_given_tokens(n) & given))]
    money_col = "receipts" if "receipts" in hit else "receipts_pre" if "receipts_pre" in hit else None
    if money_col is not None:  # ignore money-less stub committees (GA-14 2018: "GRAVES, TOM" next to the real one)
        keep = keep[keep[money_col].notna()]
    return keep if not keep.empty else hit


def edit1(a: str, b: str) -> bool:
    """True when a and b differ by at most one substitution, insertion or deletion."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(s == t[:i] + t[i + 1:] for i in range(len(t)))
