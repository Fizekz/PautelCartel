"""Late-information adjustment for 2026 from the research dossiers (repo/data/dossiers/summary.csv, as of Nov 1).

Each dossier gives q_dog_mean: the mean of a researcher's and a skeptic's probability for Kalshi's underdog in that
race, built from information the model does not see (October polls missing from the data files, news, cash on
hand, early vote). It has no historical analogue, so the weight cannot be fitted. Rule (fixed before looking at its
effect on any total):

    logit(s') = (1 - w) * logit(s_model) + w * logit(q_dog_mean)

where s is the underdog's share of the (favourite + underdog) probability mass; the other slots keep their q.
w = 0.25 for medium/high-confidence dossiers and 0.15 for low-confidence ones (and for a skeptic "refuted" verdict).
Rationale: the dossiers are partly anchored on the Kalshi price the model is meant to be independent of, so the
model keeps at least 75% of the weight in logit space. Exception: same-party contests (favourite and underdog on the
same side, e.g. CA D-vs-D), where the model's split rests on hand-set poll/primary weights with no backtest; there
the validated part of the model has nothing to say and w = 0.5 (an equal pooling of two unvalidated estimates).
This exception was set after seeing CA-34 (model Gomez 0.85, dossier 0.22, scandal not in any model input).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, logit, name_matches, sigmoid, tokens

DOSSIERS = ROOT / "data" / "dossiers" / "summary.csv"
W_BY_CONF = {"high": 0.25, "medium": 0.25, "low": 0.15}
W_REFUTED = 0.15
W_SAME_PARTY = 0.5


def _fav_index(g: pd.DataFrame, fav_text: str) -> int | None:
    """Slot of Kalshi's favourite named in the dossier (free text such as 'REP: Greg Murphy' or 'Kim (REP)')."""
    text = str(fav_text).split("(tie")[0]
    for i, (cand, p) in enumerate(zip(g["candidate"], g["p_kalshi"])):
        if cand in text or name_matches(tokens(text.split(":")[-1]), cand):
            return i
    return None


def weight(row) -> float:
    if row.skeptic == "refuted":
        return W_REFUTED
    return W_BY_CONF.get(str(row.conf), W_BY_CONF["low"])


def adjust(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pred with q adjusted and q_model kept, log of the adjustments)."""
    if not DOSSIERS.exists():
        raise FileNotFoundError(f"dossier summary missing: {DOSSIERS}")
    d = pd.read_csv(DOSSIERS)
    pred = pred.copy()
    pred["q_model"] = pred["q"]
    log = []
    for row in d.itertuples():
        idx = pred.index[pred["race_id"] == row.race_id]
        if len(idx) < 2 or not np.isfinite(row.q_dog_mean):
            continue
        g = pred.loc[idx]
        fi = _fav_index(g, row.kalshi_fav)
        if fi is None:
            log.append({"race_id": row.race_id, "status": f"favourite not matched: {row.kalshi_fav}"})
            continue
        order = np.argsort(-g["p_kalshi"].to_numpy())
        di = next(int(i) for i in order if int(i) != fi)
        fav, dog = idx[fi], idx[di]
        mass = pred.at[fav, "q"] + pred.at[dog, "q"]
        s = pred.at[dog, "q"] / mass
        w = W_SAME_PARTY if pred.at[fav, "side"] == pred.at[dog, "side"] else weight(row)
        s_new = float(sigmoid((1 - w) * logit(s) + w * logit(row.q_dog_mean)))
        pred.at[dog, "q"], pred.at[fav, "q"] = mass * s_new, mass * (1 - s_new)
        log.append({"race_id": row.race_id, "status": "adjusted", "underdog": pred.at[dog, "candidate"],
                    "p_kalshi_dog": round(float(pred.at[dog, "p_kalshi"]), 3), "q_model_dog": round(mass * s, 4),
                    "q_dossier_dog": row.q_dog_mean, "w": w, "q_final_dog": round(mass * s_new, 4),
                    "conf": row.conf, "skeptic": row.skeptic})
    return pred, pd.DataFrame(log)
