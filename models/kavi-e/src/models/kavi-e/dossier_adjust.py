"""v2: late-information adjustment of 2026 race margins from the Nov 1 research dossiers.

repo/data/dossiers/summary.csv gives, for 85 contested races, a researched probability for Kalshi's underdog after
a skeptic's review (q_dog_mean). The dossiers read polls, news and money up to Nov 1 that the model's inputs miss,
but they have no track record and are anchored on the Kalshi price (60 of 85 say "market about right"), so they are
a modest extra opinion, not a replacement.

How (kept in the model's own MARGIN space so the simulation stays coherent):
  q_D  = dossier probability that the D side wins (from the favourite's side and q_dog_mean)
  mu_D = sigma_i * Finv(q_D)          (the margin our error model would need to give q_D; F = scaled t6)
  mu'  = (1 - lam) * mu + lam * mu_D,  lam = LAMBDA[dossier confidence]
Same-party races (both finalists on one side) are adjusted in logit space on the pair instead.
LAMBDA = low 0.15 / medium 0.25 / high 0.35 is a judgment: no historical analogue exists to fit it. 0.25 is about
the weight of one more rater next to the core three.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

import model
from common import ROOT, last_name

SUMMARY = ROOT / "data" / "dossiers" / "summary.csv"
LAMBDA = {"low": 0.15, "medium": 0.25, "high": 0.35}
PARTY = {"DEM": "DEM", "D": "DEM", "DFL": "DEM", "REP": "REP", "R": "REP", "IND": "IND", "I": "IND"}


def parse_fav(text: str) -> tuple[str, str]:
    t = str(text).strip()
    m = re.match(r"^(DEM|REP|IND)\s*:\s*(.+)$", t)
    if m:
        return m.group(2).strip(), m.group(1)
    m = re.match(r"^(Democratic|Republican) party \((.+)\)", t)
    if m:
        return m.group(2).strip(), "DEM" if m.group(1) == "Democratic" else "REP"
    m = re.match(r"^(.+?)\s*\((DEM|REP|IND|DFL|D|R|I)\)", t)
    if m:
        return m.group(1).strip(), PARTY[m.group(2)]
    raise ValueError(f"cannot parse dossier favourite {text!r}")


def load() -> pd.DataFrame:
    d = pd.read_csv(SUMMARY)
    parsed = d["kalshi_fav"].map(parse_fav)
    d["fav_name"] = parsed.map(lambda x: x[0])
    d["fav_party"] = parsed.map(lambda x: x[1])
    d["lam"] = d["conf"].map(LAMBDA).fillna(LAMBDA["low"])
    return d


def margin_for(p: np.ndarray | float, sigma: np.ndarray | float) -> np.ndarray:
    """Inverse of model.win_prob: the mu that gives P(D) = p at this sigma."""
    df = model.ERR_DF
    p = np.clip(p, 1e-4, 1 - 1e-4)
    z = student_t.ppf(p, df) / np.sqrt(df / (df - 2)) if df and df > 2 else norm.ppf(p)
    return z * sigma


def adjust_races(races: pd.DataFrame, dside: dict[str, str], skip: set[str]) -> pd.DataFrame:
    """races: model.predict output (race_id, mu, sigma, p_d). dside: race_id -> party label of the D side."""
    d = load().set_index("race_id")
    out = races.copy()
    out["mu_model"], out["p_d_model"], out["dossier_lam"], out["q_d_dossier"] = out["mu"], out["p_d"], 0.0, np.nan
    for i, r in out.iterrows():
        rid = r["race_id"]
        if rid not in d.index or rid in skip:
            continue
        row = d.loc[rid]
        fav_is_d = row["fav_party"] == dside.get(rid, "DEM")
        q_d = 1 - row["q_dog_mean"] if fav_is_d else row["q_dog_mean"]
        mu_d = float(margin_for(q_d, r["sigma"]))
        lam = float(row["lam"])
        out.loc[i, "mu"] = (1 - lam) * r["mu"] + lam * mu_d
        out.loc[i, ["dossier_lam", "q_d_dossier"]] = lam, q_d
    out["p_d"] = model.win_prob(out["mu"].to_numpy() / out["sigma"].to_numpy())
    return out


def adjust_same_party(pred: pd.DataFrame, same_party_races: set[str]) -> pd.DataFrame:
    """Logit pool on the two same-party finalists of a race (the pair's total q is kept)."""
    d = load().set_index("race_id")
    out = pred.copy()
    for rid in same_party_races & set(d.index):
        row = d.loc[rid]
        g = out[out["race_id"] == rid]
        fav = g[g["candidate"].map(last_name) == last_name(row["fav_name"])]
        if fav.empty:
            continue
        fi = fav.index[0]
        pair = g[g["party"] == g.loc[fi, "party"]].nlargest(2, "q").index
        if len(pair) < 2 or fi not in pair:
            continue
        oi = [j for j in pair if j != fi][0]
        mass = out.loc[fi, "q"] + out.loc[oi, "q"]
        p_fav = out.loc[fi, "q"] / mass
        lam = float(row["lam"])
        lg = (1 - lam) * np.log(p_fav / (1 - p_fav)) + lam * np.log((1 - row["q_dog_mean"]) / row["q_dog_mean"])
        p_new = 1 / (1 + np.exp(-lg))
        out.loc[fi, "q"], out.loc[oi, "q"] = mass * p_new, mass * (1 - p_new)
    return out
