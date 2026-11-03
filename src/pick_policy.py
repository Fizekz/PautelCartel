"""Compare pick policies on 2026: simulated contest score distributions under a model's q (board P004/K006).

Takes a pred_2026.csv (race_id,candidate,party,q,p_kalshi; q sums to 1 per race) and simulates elections with a
shared national swing, so upsets in one direction happen together. Each policy picks one candidate per race; a
correct pick scores -log2(max(p_kalshi, 0.01)), as in the contest.

Policies:
    chalk        Kalshi's favourite everywhere
    model_top    the candidate our q likes most
    dog_t<T>     Kalshi's favourite, except take the underdog where our q for it is >= T
    ev           the candidate with the highest q * points (expected-value maximiser)

Outcomes are drawn from --truth: "self" (the same q that made the picks, flattering), "kalshi" (Kalshi's prices are
right), or another pred CSV (e.g. a different model). Judge a policy by how it holds up under truths it did not use.

Usage: python src/pick_policy.py PRED.csv [--truth self|kalshi|OTHER.csv] [--sims 20000] [--rho 0.25] [--out picks.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

P_FLOOR = 0.01
THRESHOLDS = (0.35, 0.40, 0.45, 0.50)


def load(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    df = df[df["p_kalshi"].notna()].copy()
    df["q"] = df["q"] / df.groupby("race_id")["q"].transform("sum")
    df["pts"] = -np.log2(df["p_kalshi"].clip(lower=P_FLOOR))
    return df


def two_way(df: pd.DataFrame) -> pd.DataFrame:
    """Per race: Kalshi favourite (A) and the strongest other candidate by q (B); other mass goes to A's side."""
    rows = []
    for rid, g in df.groupby("race_id", sort=False):
        g = g.sort_values("p_kalshi", ascending=False)
        a = g.iloc[0]
        rest = g.iloc[1:]
        b = rest.loc[rest["q"].idxmax()] if len(rest) else None
        rows.append({
            "race_id": rid, "a": a["candidate"], "a_party": a["party"], "a_p": a["p_kalshi"], "a_pts": a["pts"],
            "a_q": a["q"], "b": None if b is None else b["candidate"], "b_party": None if b is None else b["party"],
            "b_p": np.nan if b is None else b["p_kalshi"], "b_pts": 0.0 if b is None else b["pts"],
            "b_q": 0.0 if b is None else b["q"],
        })
    return pd.DataFrame(rows)


def simulate(races: pd.DataFrame, n_sims: int, rho: float, seed: int) -> np.ndarray:
    """Boolean matrix (sims x races): True where B (the Kalshi underdog side) wins. Marginals equal b_q."""
    rng = np.random.default_rng(seed)
    # A national shock moves every race toward the same party; its sign per race depends on B's party.
    direction = np.where(races["b_party"].eq("DEM") | races["b_party"].eq("IND"), 1.0, -1.0)
    direction = np.where(races["b_party"].isin(["DEM", "REP", "IND"]), direction, 0.0)
    thresh = norm.ppf(races["b_q"].clip(1e-6, 1 - 1e-6).to_numpy())
    nat = rng.standard_normal((n_sims, 1))
    eps = rng.standard_normal((n_sims, len(races)))
    rho_vec = np.where(direction != 0, rho, 0.0)
    latent = np.sqrt(rho_vec) * nat * direction + np.sqrt(1 - rho_vec) * eps
    return latent < thresh


def policies(races: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each policy: boolean per race, True = pick B (the underdog side)."""
    out = {"chalk": np.zeros(len(races), bool), "model_top": (races["b_q"] > races["a_q"]).to_numpy()}
    for t in THRESHOLDS:
        out[f"dog_t{t:.2f}"] = (races["b_q"] >= t).to_numpy()
    out["ev"] = (races["b_q"] * races["b_pts"] > races["a_q"] * races["a_pts"]).to_numpy()
    return out


def score(pick_b: np.ndarray, b_wins: np.ndarray, races: pd.DataFrame) -> np.ndarray:
    pts_a, pts_b = races["a_pts"].to_numpy(), races["b_pts"].to_numpy()
    return np.where(pick_b, b_wins * pts_b, ~b_wins * pts_a).sum(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", type=Path)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--rho", type=float, default=0.25, help="share of latent variance from the national swing")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--out", type=Path, help="write each policy's picks per race")
    ap.add_argument("--truth", default="self", help='"self", "kalshi", or a pred CSV to draw outcomes from')
    args = ap.parse_args()

    races = two_way(load(args.pred))
    truth = races.copy()
    if args.truth == "kalshi":
        truth["b_q"] = truth["b_p"].fillna(0.0) / (truth["a_p"] + truth["b_p"].fillna(0.0))
    elif args.truth != "self":
        other = load(Path(args.truth)).set_index(["race_id", "candidate"])["q"]
        truth["b_q"] = [other.get((r, b), np.nan) if b is not None else 0.0
                        for r, b in zip(truth["race_id"], truth["b"])]
        truth["b_q"] = truth["b_q"].fillna(races["b_q"])
    b_wins = simulate(truth, args.sims, args.rho, args.seed)
    pols = policies(races)
    chalk = score(pols["chalk"], b_wins, races)
    print(f"{args.pred} | truth={args.truth} | {len(races)} races | {args.sims} sims | rho {args.rho}")
    print(f"{'policy':<11}{'n_dogs':>7}{'mean':>8}{'sd':>7}{'p10':>7}{'p50':>7}{'p90':>7}{'P>chalk':>9}")
    for name, pick_b in pols.items():
        s = score(pick_b, b_wins, races)
        q10, q50, q90 = np.percentile(s, [10, 50, 90])
        print(f"{name:<11}{pick_b.sum():>7}{s.mean():>8.1f}{s.std():>7.1f}{q10:>7.1f}{q50:>7.1f}{q90:>7.1f}"
              f"{(s > chalk).mean():>9.2f}")
    if args.out:
        table = races[["race_id", "a", "a_party", "a_p", "a_q", "b", "b_party", "b_p", "b_q", "b_pts"]].copy()
        for name, pick_b in pols.items():
            table[name] = np.where(pick_b, table["b"], table["a"])
        table.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
