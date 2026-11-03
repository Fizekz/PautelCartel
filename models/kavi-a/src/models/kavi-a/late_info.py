"""2026 late-information adjustment from the research dossiers (repo data/dossiers/summary.csv, as of Nov 1).

Each of the 85 dossiers ends in a researcher probability for the Kalshi underdog and a skeptic's re-estimate;
`q_dog_mean` is their mean. They carry news the model does not see (late polls not yet in polls_2026, early vote,
candidate news). There is no historical analogue, so the weight is set a priori, not fitted:
  logit p_new = (1 - DOSSIER_WEIGHT) logit p_model + DOSSIER_WEIGHT logit p_dossier,   DOSSIER_WEIGHT = 0.25
applied to P(D side wins) in two-party races and to the two finalists' q in same-party races. The dossier writers
saw the Kalshi prices (their numbers sit close to them), so a small weight keeps the model's own view dominant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from calib import logit, mu_for_q
from common import ROOT, tokens
from model import win_prob
from races import candidate_q

DOSSIERS = ROOT / "data" / "dossiers" / "summary.csv"
DOSSIER_WEIGHT = 0.25


def expit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def find_favourite(sub: pd.DataFrame, fav_text: str) -> int | None:
    """Row index of the dossier's Kalshi favourite inside one race's candidate rows."""
    main = sub[sub["side"] != ""] if (sub["side"] != "").any() else sub
    # full text first; if it names both candidates (WI-01 "Berman (DEM) (tie with Steil ...)"), the part before "("
    for text in (fav_text, fav_text.split("(")[0]):
        text_tokens = set(tokens(text))
        hits = [i for i, name in main["candidate"].items() if tokens(name) and tokens(name)[-1] in text_tokens]
        if len(hits) == 1:
            return hits[0]
    if "party" in fav_text.lower():  # party slots (AK-SEN "Democratic party (Mary Peltola)")
        party = "DEM" if "democratic" in fav_text.lower() else "REP"
        hit = main[main["party"] == party]
        return hit.index[0] if len(hit) == 1 else None
    return None


def apply_dossiers(c: pd.DataFrame, races: pd.DataFrame, weight: float = DOSSIER_WEIGHT,
                   path=DOSSIERS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns updated (candidates, races, log). Two-party races get a new p_d and the matching mu."""
    if not path.exists():
        raise SystemExit(f"dossier summary missing: {path}")
    dos = pd.read_csv(path)
    c, races = c.copy(), races.copy()
    races["p_d_pre_dossier"] = races["p_d"]
    log, sp_updates = [], {}
    for r in dos.itertuples():
        sub = c[c["race_id"] == r.race_id]
        fav = find_favourite(sub, str(r.kalshi_fav)) if len(sub) else None
        if fav is None:
            log.append({"race_id": r.race_id, "status": "favourite not matched"})
            continue
        p_fav_dossier = 1 - float(r.q_dog_mean)
        kind = races.at[r.race_id, "kind"]
        if kind == "two_party":
            fav_is_d = c.at[fav, "side"] == "D"
            p_d_dos = p_fav_dossier if fav_is_d else 1 - p_fav_dossier
            before = float(races.at[r.race_id, "p_d"])
            races.at[r.race_id, "p_d"] = float(expit((1 - weight) * logit(before) + weight * logit(p_d_dos)))
            log.append({"race_id": r.race_id, "status": "two_party", "fav": c.at[fav, "candidate"],
                        "p_d_model": round(before, 4), "p_d_dossier": round(p_d_dos, 4),
                        "p_d_final": round(races.at[r.race_id, "p_d"], 4)})
        elif kind == "same_party":
            before = float(c.at[fav, "q"])
            sp_updates[r.race_id] = (fav, float(expit((1 - weight) * logit(before) + weight * logit(p_fav_dossier))))
            log.append({"race_id": r.race_id, "status": "same_party", "fav": c.at[fav, "candidate"],
                        "q_fav_model": round(before, 4), "q_fav_dossier": round(p_fav_dossier, 4),
                        "q_fav_final": round(sp_updates[r.race_id][1], 4)})
        else:
            log.append({"race_id": r.race_id, "status": f"{kind}: not adjusted"})
    two = [x["race_id"] for x in log if x.get("status") == "two_party"]
    if two:
        rows = races.loc[two]
        races.loc[two, "mu"] = mu_for_q(rows["p_d"].to_numpy(), rows["sd_nat"].to_numpy(),
                                        rows["sd_state"].to_numpy(), rows["sd_race"].to_numpy())
        races.loc[two, "p_d"] = win_prob(races.loc[two, "mu"], races.loc[two, "sd_nat"],
                                         races.loc[two, "sd_state"], races.loc[two, "sd_race"])
    keep_q = c["q"].copy()
    c["q"] = candidate_q(c, races)
    same = c["race_id"].map(races["kind"]).eq("same_party")
    c.loc[same, "q"] = keep_q[same]  # same-party q came from same_party.py; candidate_q would redo the softmax
    for rid, (fav, q_new) in sp_updates.items():
        idx = c.index[(c["race_id"] == rid) & (c["side"] != "")]
        other = [i for i in idx if i != fav]
        minor = c.index[(c["race_id"] == rid) & (c["side"] == "")]
        mass = 1 - c.loc[minor, "q"].sum()
        c.loc[fav, "q"] = q_new * mass
        c.loc[other, "q"] = (1 - q_new) * mass / max(len(other), 1)
    return c, races, pd.DataFrame(log)
