"""v2: late-information adjustment from the Nov 1 research dossiers (repo/data/dossiers/summary.csv).

Each of 85 contested races has a researched probability for Kalshi's underdog (q_dog_research), a skeptic's
re-check (q_dog_skeptic) and their mean (q_dog_mean). The dossiers read news, polls and money up to Nov 1 that our
inputs miss (e.g. late scandals, polls not in polls_2026.csv), but they have no track record and are visibly
anchored on the Kalshi price (73 of 85 say "market about right"). So they enter as ONE extra opinion pooled in
logit space with a small fixed weight, only on the two-candidate pair the dossier is about:

    logit p'_dog = (1 - W_DOSSIER) * logit p_dog(model) + W_DOSSIER * logit q_dog_mean

where p_dog is the underdog's share of the pair's mass; the pair's total mass and every other candidate's q are
unchanged. W_DOSSIER = 0.25 is a judgment (no historical analogue to fit it): about the weight of one outlet in a
four-way average of the three core raters plus the dossier.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, last_name

DOSSIER_SUMMARY = ROOT / "data" / "dossiers" / "summary.csv"
W_DOSSIER = 0.25
PARTY_WORDS = {"DEM": "DEM", "REP": "REP", "D": "DEM", "R": "REP", "DFL": "DEM", "IND": "IND", "I": "IND"}


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def parse_fav(text: str) -> tuple[str | None, str | None]:
    """'REP: Ashley Hinson', 'REP:Bob X', 'Mitchell Berman (DEM) (tie ...)', 'Democratic party (Mary Peltola)'."""
    t = str(text).strip()
    m = re.match(r"^(DEM|REP|IND)\s*:\s*(.+)$", t)
    if m:
        return m.group(2).strip(), m.group(1)
    m = re.match(r"^(Democratic|Republican) party", t)
    if m:
        return f"{m.group(1)} party", "DEM" if m.group(1) == "Democratic" else "REP"
    m = re.match(r"^([^()]+?)\s*\(([A-Z]+)\)", t)
    if m:
        return m.group(1).strip(), PARTY_WORDS.get(m.group(2))
    return t, None


def load() -> pd.DataFrame:
    if not DOSSIER_SUMMARY.exists():
        return pd.DataFrame(columns=["race_id", "fav_name", "fav_party", "q_dog_mean"])
    s = pd.read_csv(DOSSIER_SUMMARY)
    parsed = s["kalshi_fav"].map(parse_fav)
    s["fav_name"], s["fav_party"] = [p[0] for p in parsed], [p[1] for p in parsed]
    return s


def adjust_race(g: pd.DataFrame, row: pd.Series) -> tuple[np.ndarray, str]:
    """g: one race's candidates (candidate, party, q, p_kalshi). Returns the adjusted q and a note."""
    q = g["q"].to_numpy(float).copy()
    name = row["fav_name"]
    if name.endswith(" party"):
        hit = np.where(g["candidate"].to_numpy() == name)[0]
    else:
        hit = np.where(g["candidate"].map(last_name).to_numpy() == last_name(name))[0]
    if len(hit) != 1:
        return q, f"dossier not applied (favourite '{name}' not matched)"
    f = int(hit[0])
    others = [i for i in np.argsort(-g["p_kalshi"].to_numpy()) if i != f]
    d = int(others[0])   # Kalshi's underdog = the other priced candidate with the highest price
    mass = q[f] + q[d]
    p_dog = q[d] / mass
    new = 1 / (1 + np.exp(-((1 - W_DOSSIER) * _logit(p_dog) + W_DOSSIER * _logit(float(row["q_dog_mean"])))))
    q[d], q[f] = mass * new, mass * (1 - new)
    note = (f"dossier late-info pool (w {W_DOSSIER}): {g['candidate'].iloc[d]} {p_dog:.3f} -> {new:.3f} "
            f"(dossier {float(row['q_dog_mean']):.3f}, {row['view']}, skeptic {row['skeptic']})")
    return q, note


def apply(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (adjusted pred, per-race log). pred must have race_id, candidate, party, q, p_kalshi, note."""
    s = load()
    pred = pred.copy()
    pred["q_model"] = pred["q"]
    logs = []
    for r in s.itertuples(index=False):
        row = pd.Series(r._asdict())
        idx = pred.index[pred["race_id"] == row["race_id"]]
        if len(idx) < 2:
            continue
        g = pred.loc[idx]
        q, note = adjust_race(g, row)
        pred.loc[idx, "q"] = q
        pred.loc[idx, "note"] = [f"{n}; {note}" if n else note for n in g["note"].fillna("")]
        logs.append({"race_id": row["race_id"], "note": note})
    return pred, pd.DataFrame(logs)
