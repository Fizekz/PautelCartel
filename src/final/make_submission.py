"""Turn a pooled forecast into the organizers' Full Congress CSV with the expected-value pick rule (C or C').

Contest scoring: a correct pick of candidate i earns pts_i = -log2(max(p_i, 0.01)), p_i = Kalshi's Nov 1 price;
a wrong pick earns 0. With our win probability q_i, the expected points of picking i are
    E_i = q_i * pts_i
and in every race we pick argmax_i E_i. Expected points add across races, so this maximises the expected total
(a shared national swing changes the spread of the total, not its mean).

Input: pooled forecast (race_id,candidate,chamber,party,p_kalshi,q), one row per candidate, q and p_kalshi summing
to 1 per race. Kalshi lists some races by party only ("Republican party"); those get the party's nominee from
data/clean/candidates_2026.csv (incumbent first).

Usage:
    python src/final/make_submission.py POOLED.csv --out predictions.csv [--detail picks_detail.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"
CANDIDATES = ROOT / "data" / "clean" / "candidates_2026.csv"
P_FLOOR = 0.01
PLACEHOLDER = {"Democratic party": "DEM", "Republican party": "REP"}
# Board P007: withdrawn but still listed by Kalshi at 0.1-0.5%; the 1% floor makes them look like 6.64-pt long shots.
WITHDRAWN = {("ID-SEN", "DEM"), ("NE-SEN", "DEM"), ("SD-SEN", "DEM")}
COLUMNS = ["race_id", "chamber", "predicted_winner_party", "candidate_name", "win_probability"]


def points(p: pd.Series) -> pd.Series:
    return -np.log2(p.clip(lower=P_FLOOR))


def ev_picks(pooled: pd.DataFrame, on_ballot: set[tuple[str, str]] | None = None,
             dog_needs_q_at_price: bool = False, force_favourite: set[str] | None = None) -> pd.DataFrame:
    """One row per race: the candidate with the highest q * pts, plus the Kalshi favourite for reference.
    on_ballot: (race_id, party) pairs with a real candidate; a party-level Kalshi market for any other party
    (e.g. "Republican party" in MA-01, where no Republican is on the ballot; board P007) is never picked.
    dog_needs_q_at_price: rule C' -- an underdog is eligible only if our q is at least its Kalshi price, so we never
    take a long shot that we ourselves rate below the market (both sides' models overrated 5-15% long shots in
    past cycles; boards K024, P027).
    force_favourite: race_ids where the Kalshi favourite is picked regardless (a human risk call)."""
    if pooled["p_kalshi"].isna().any():
        bad = sorted(pooled.loc[pooled["p_kalshi"].isna(), "race_id"].unique())
        raise SystemExit(f"candidates without a Kalshi price (name mismatch?) in {bad}")
    df = pooled.assign(pts=points(pooled["p_kalshi"]))
    is_fav = df.index.isin(df.groupby("race_id")["p_kalshi"].idxmax())
    blocked = np.array([(r, p) in WITHDRAWN or (on_ballot is not None and c in PLACEHOLDER and (r, p) not in on_ballot)
                        for r, p, c in zip(df["race_id"], df["party"], df["candidate"])])
    if dog_needs_q_at_price:
        blocked = blocked | (~is_fav & (df["q"] < df["p_kalshi"]).to_numpy())
    if force_favourite:
        blocked = blocked | (~is_fav & df["race_id"].isin(force_favourite).to_numpy())
    df = df.assign(ev=np.where(blocked, -1.0, df["q"] * df["pts"]))
    pick = df.loc[df.groupby("race_id")["ev"].idxmax()]
    fav = df.loc[df.groupby("race_id")["p_kalshi"].idxmax(), ["race_id", "candidate", "p_kalshi", "q"]]
    fav = fav.rename(columns={"candidate": "kalshi_fav", "p_kalshi": "fav_p", "q": "fav_q"})
    return pick.merge(fav, on="race_id").assign(is_underdog=lambda d: d["candidate"] != d["kalshi_fav"])


def nominee(cands: pd.DataFrame, race_id: str, party: str) -> str:
    rows = cands[(cands["race_id"] == race_id) & (cands["party"] == party)]
    if rows.empty:
        raise SystemExit(f"{race_id}: no {party} candidate in {CANDIDATES.name} to name the party-level pick")
    rows = rows.sort_values(["is_incumbent", "is_major"], ascending=False)
    return str(rows.iloc[0]["candidate_name"])


def to_template(picks: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    names = [nominee(cands, r, PLACEHOLDER[c]) if c in PLACEHOLDER else c
             for r, c in zip(picks["race_id"], picks["candidate"])]
    out = picks.assign(candidate_name=names, predicted_winner_party=picks["party"],
                       win_probability=picks["q"].round(4))
    template = pd.read_csv(TEMPLATE)[["race_id", "chamber"]]
    return template.merge(out[["race_id"] + COLUMNS[2:]], on="race_id", how="left")[COLUMNS]


def validate(sub: pd.DataFrame) -> None:
    template = pd.read_csv(TEMPLATE)
    problems = []
    if list(sub.columns) != list(template.columns):
        problems.append(f"columns {list(sub.columns)} != template {list(template.columns)}")
    if len(sub) != len(template) or set(sub["race_id"]) != set(template["race_id"]):
        problems.append("race_ids differ from the template")
    if sub[COLUMNS[2:]].isna().any().any():
        problems.append(f"empty cells in races {sorted(sub[sub.isna().any(axis=1)]['race_id'])}")
    if not sub["win_probability"].between(0, 1).all():
        problems.append("win_probability outside [0, 1]")
    if problems:
        raise SystemExit("submission invalid: " + "; ".join(problems))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pooled", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detail", type=Path, help="also write each pick with q, Kalshi price, points and EV")
    ap.add_argument("--rule", choices=["C", "Cprime"], default="Cprime",
                    help="C: pure expected value; Cprime (default, chosen by Kavi 2026-11-02): expected value, but an "
                         "underdog only where our q >= its Kalshi price")
    ap.add_argument("--favourite", default="",
                    help="comma-separated race_ids forced to the Kalshi favourite (human risk call)")
    args = ap.parse_args()
    cands = pd.read_csv(CANDIDATES)
    forced = {r.strip() for r in args.favourite.split(",") if r.strip()}
    template_ids = set(pd.read_csv(TEMPLATE)["race_id"])
    if forced - template_ids:
        raise SystemExit(f"--favourite: unknown race_ids {sorted(forced - template_ids)}")
    picks = ev_picks(pd.read_csv(args.pooled), on_ballot=set(zip(cands["race_id"], cands["party"])),
                     dog_needs_q_at_price=args.rule == "Cprime", force_favourite=forced)
    sub = to_template(picks, cands)
    validate(sub)
    sub.to_csv(args.out, index=False)
    if args.detail:
        picks.to_csv(args.detail, index=False)
    dogs = picks[picks["is_underdog"]]
    print(f"{len(sub)} races -> {args.out}; {len(dogs)} underdog picks; expected points {picks['ev'].sum():.1f} "
          f"(if right: {picks['pts'].sum():.1f}); chalk expected {(picks['fav_q'] * points(picks['fav_p'])).sum():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
