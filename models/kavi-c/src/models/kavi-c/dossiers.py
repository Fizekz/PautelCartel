"""Late-information adjustment from the Nov 1 research dossiers (repo/data/dossiers/summary.csv, 85 races).

Each dossier gives a researcher's and a skeptic's probability for the Kalshi underdog (`q_dog_research`,
`q_dog_skeptic`; their mean `q_dog_mean`), built from the latest polls, rating moves, money and news as of Nov 1.
The dossiers saw the Kalshi price, so they are not independent of the market; they are also new information that
has no historical analogue, so the weight cannot be fitted or backtested.

The adjustment is a fixed logit-space blend of the model's probability for the Kalshi favourite with the dossier's:
    logit q' = (1 - W) logit q_model + W logit (1 - q_dog_mean),
with W = 0.25 (0.15 when the skeptic weakened or refuted the dossier's view). Other candidates in the race keep
their relative shares. W is a hand-set judgement: small enough that the model, not the dossier, decides every race
where the two differ a lot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import ROOT, full_name, last_name

# Frozen copy of repo/data/dossiers/summary.csv as of 2026-11-01 (the .md dossiers stay in the repo).
DOSSIER_SUMMARY = ROOT / "data" / "kavi-c" / "dossiers_summary_2026-11-01.csv"
W_DEFAULT = 0.25
W_WEAK = 0.15            # skeptic "weakened" or "refuted"
Q_CLIP = 0.001


def _logit(p: float) -> float:
    p = float(np.clip(p, Q_CLIP, 1 - Q_CLIP))
    return float(np.log(p / (1 - p)))


def load() -> pd.DataFrame:
    if not DOSSIER_SUMMARY.exists():
        raise FileNotFoundError(f"dossier summary missing: {DOSSIER_SUMMARY}")
    d = pd.read_csv(DOSSIER_SUMMARY)
    need = {"race_id", "kalshi_fav", "q_dog_mean", "skeptic"}
    missing = need - set(d.columns)
    if missing:
        raise ValueError(f"dossier summary lacks columns {missing}")
    d = d[d["q_dog_mean"].between(0, 1)]
    return d.drop_duplicates("race_id")


def favourite_index(g: pd.DataFrame, fav_text: str) -> int:
    """Row index (in g) of the candidate the dossier calls the Kalshi favourite."""
    text = " ".join(full_name(fav_text).split())
    hits = [i for i, n in zip(g.index, g["candidate"]) if full_name(n) and full_name(n) in text]
    if len(hits) != 1:
        toks = set(text.split())
        hits = [i for i, n in zip(g.index, g["candidate"])
                if last_name(n) in toks and not n.endswith(" party")]
    if len(hits) == 1:
        return hits[0]
    return int(g["p_kalshi"].idxmax())


def blend(pred: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (adjusted pred, one log row per adjusted race)."""
    out = pred.copy()
    log = []
    for _, d in summary.iterrows():
        g = out[out["race_id"] == d["race_id"]]
        if g.empty:
            continue
        fi = favourite_index(g, str(d["kalshi_fav"]))
        w = W_WEAK if str(d["skeptic"]).lower() in ("weakened", "refuted") else W_DEFAULT
        q_fav = float(g.at[fi, "q"])
        q_dos = 1.0 - float(d["q_dog_mean"])
        z = (1 - w) * _logit(q_fav) + w * _logit(q_dos)
        q_new = float(1 / (1 + np.exp(-z)))
        rest = g.index.drop(fi)
        rest_sum = float(g.loc[rest, "q"].sum())
        out.loc[fi, "q"] = q_new
        if rest_sum > 0:
            out.loc[rest, "q"] = g.loc[rest, "q"] * (1 - q_new) / rest_sum
        log.append({"race_id": d["race_id"], "kalshi_fav": g.at[fi, "candidate"], "p_kalshi_fav": g.at[fi, "p_kalshi"],
                    "q_model_fav": q_fav, "q_dossier_fav": q_dos, "w": w, "q_final_fav": q_new})
    return out, pd.DataFrame(log)
