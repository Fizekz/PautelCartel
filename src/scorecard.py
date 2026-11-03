"""Common scorecard: replay the contest on 2018 and 2022, plus calibration.

A model hands in one file per backtest year with a win probability per candidate:
    year,race_id,candidate,party,q          (q sums to 1 within each race)
race_id as in data/benchmark/fte_final_<year>.csv (House AL-01 / AK-AL, Senate OH-SEN, special <ST>-SEN-S).
The model must be trained only on data available before that election (no <year> results, no later data).

Scoring, per race in the benchmark (all 470 races per year):
- Candidates are matched to the benchmark by (race_id, party); when a race has two candidates of one party
  (top-two), by normalised full name. A race the model leaves out gets equal q across the benchmark's candidates.
- Calibration: log loss = -ln(q_winner) with q clipped to [0.001, 0.999]; Brier = sum over candidates (q - won)^2.
- Contest replay with 538's final forecast as the market: a correct pick earns -log2(max(p_fte, 0.01)), else 0.
  Two pick policies are scored: "top" = the candidate with the highest q; "ev" = the highest q * -log2(p_fte).
- Baselines on the same races: 538 itself (its log loss / Brier) and "chalk" (always 538's favourite).
- Every metric is also reported on the contested subset: races where 538's favourite was below 0.95.

Usage: python src/scorecard.py --pred 2018=path.csv --pred 2022=path.csv [--name kavi-a] [--json out.json]
Actual winners: data/benchmark/winners_<year>.csv (race_id,winner_party,winner_name), frozen from MIT results.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "benchmark" / "fte_final_{year}.csv"
WINNERS = ROOT / "data" / "benchmark" / "winners_{year}.csv"
RESULTS = [ROOT / "data" / "clean" / "results_house.csv", ROOT / "data" / "clean" / "results_senate.csv"]
P_FLOOR, Q_CLIP, CONTESTED = 0.01, 0.001, 0.95
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def last_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    parts = [p for p in re.split(r"[^a-z]+", text) if p and p not in SUFFIXES]
    return parts[-1] if parts else ""


def full_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(p for p in re.split(r"[^a-z]+", text) if p and p not in SUFFIXES)


def candidate_key(df: pd.DataFrame) -> pd.Series:
    """party, or party:full name where the race has two candidates of that party."""
    dup = df.duplicated(["race_id", "party"], keep=False)
    return np.where(dup, df["party"] + ":" + df["candidate"].map(full_name), df["party"])


def winners(year: int) -> pd.DataFrame:
    """Actual winners, frozen in data/benchmark/winners_<year>.csv (built once from MIT results by --freeze)."""
    path = Path(str(WINNERS).format(year=year))
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(ROOT)}; run --freeze once where data/clean/results_*.csv exist")
    return pd.read_csv(path)


def freeze_winners(years: list[int]) -> None:
    res = pd.concat([pd.read_csv(p) for p in RESULTS], ignore_index=True)
    for year in years:
        out = res[res["year"] == year][["year", "race_id", "winner_party", "winner_name"]].drop_duplicates("race_id")
        # MEDSL codes independents (Sanders, King) as OTHER; take the party label 538 used for the same person.
        bench = pd.read_csv(str(BENCH).format(year=year))
        bench_party = {(r, last_name(c)): p for r, c, p in zip(bench["race_id"], bench["candidate"], bench["party"])}
        out["winner_party"] = [
            bench_party.get((r, last_name(n)), p) if p == "OTHER" else p
            for r, n, p in zip(out["race_id"], out["winner_name"], out["winner_party"])
        ]
        out.to_csv(str(WINNERS).format(year=year), index=False)
        print(f"froze {len(out)} winners for {year}")


def actual_winner_flags(bench: pd.DataFrame, res: pd.DataFrame) -> pd.Series:
    """1 for the benchmark candidate who won. Match order per race: full name, then a unique last name, then a
    unique party. Races where none of these singles out one candidate get no winner (and are dropped)."""
    m = bench.merge(res, on="race_id", how="left")
    tests = [
        m["candidate"].map(full_name) == m["winner_name"].map(full_name),
        m["candidate"].map(last_name) == m["winner_name"].map(last_name),
        m["party"] == m["winner_party"],
    ]
    won = pd.Series(0, index=bench.index)
    decided = pd.Series(False, index=m["race_id"].unique())
    for test in tests:
        hits = test.groupby(m["race_id"]).transform("sum")
        pick = test & (hits == 1) & ~m["race_id"].map(decided).to_numpy()
        won[pick.to_numpy()] = 1
        decided[m.loc[pick, "race_id"].unique()] = True
    return won


def score_year(year: int, pred_path: Path) -> dict:
    bench = pd.read_csv(str(BENCH).format(year=year))
    bench["key"] = candidate_key(bench)
    bench["won"] = actual_winner_flags(bench, winners(year))
    bad = bench.groupby("race_id")["won"].sum()
    races_ok = bad[bad == 1].index  # races where exactly one benchmark candidate is the known winner
    dropped = sorted(set(bench["race_id"]) - set(races_ok))
    bench = bench[bench["race_id"].isin(races_ok)].copy()

    pred = pd.read_csv(pred_path)
    pred = pred[pred["year"] == year] if "year" in pred else pred
    pred["key"] = candidate_key(pred)
    m = bench.merge(pred[["race_id", "key", "q"]], on=["race_id", "key"], how="left")
    covered = m.groupby("race_id")["q"].transform(lambda s: s.notna().any())
    n_cand = m.groupby("race_id")["race_id"].transform("size")
    m["q"] = np.where(covered, m["q"].fillna(0.0), 1.0 / n_cand)
    m["q"] = m["q"] / m.groupby("race_id")["q"].transform("sum").replace(0, 1)
    m["pts"] = -np.log2(m["p_fte"].clip(lower=P_FLOOR))

    def metrics(df: pd.DataFrame) -> dict:
        w = df[df["won"] == 1]
        top = df.loc[df.groupby("race_id")["q"].idxmax()]
        ev = df.loc[(df["q"] * df["pts"]).groupby(df["race_id"]).idxmax()]
        chalk = df.loc[df.groupby("race_id")["p_fte"].idxmax()]
        return {
            "n_races": int(df["race_id"].nunique()),
            "log_loss": float(-np.log(w["q"].clip(Q_CLIP, 1 - Q_CLIP)).mean()),
            "brier": float(((df["q"] - df["won"]) ** 2).groupby(df["race_id"]).sum().mean()),
            "accuracy_top": float(top["won"].mean()),
            "score_top": float((top["pts"] * top["won"]).sum()),
            "score_ev": float((ev["pts"] * ev["won"]).sum()),
            "n_against_538_top": int((top.index != chalk.index).sum()),
            "fte_log_loss": float(-np.log(w["p_fte"].clip(Q_CLIP, 1 - Q_CLIP)).mean()),
            "fte_brier": float(((df["p_fte"] - df["won"]) ** 2).groupby(df["race_id"]).sum().mean()),
            "chalk_score": float((chalk["pts"] * chalk["won"]).sum()),
            "chalk_accuracy": float(chalk["won"].mean()),
        }

    out = {"year": year, "races_uncovered_by_model": int((~covered).groupby(m["race_id"]).first().sum()),
           "races_dropped_no_winner_match": dropped, "all": metrics(m)}
    # Safe seats dominate the all-race log loss; "contested" = races where 538's favourite was below 0.95.
    fav = m.groupby("race_id")["p_fte"].transform("max")
    out["contested"] = metrics(m[fav < CONTESTED])
    for chamber in ("House", "Senate"):
        out[chamber] = metrics(m[m["chamber"] == chamber])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[], help="YEAR=path.csv (repeatable)")
    ap.add_argument("--freeze", action="store_true", help="write data/benchmark/winners_<year>.csv from MIT results")
    ap.add_argument("--name", default="model")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.freeze:
        freeze_winners([2018, 2022])
    report = {"name": args.name, "years": []}
    for item in args.pred:
        year, path = item.split("=", 1)
        report["years"].append(score_year(int(year), Path(path)))
    for y in report["years"]:
        a = y["all"]
        print(f"{args.name} {y['year']}: n={a['n_races']} logloss {a['log_loss']:.4f} (538 {a['fte_log_loss']:.4f}) "
              f"brier {a['brier']:.4f} (538 {a['fte_brier']:.4f}) acc {a['accuracy_top']:.3f} "
              f"score top {a['score_top']:.2f} ev {a['score_ev']:.2f} (chalk {a['chalk_score']:.2f}) "
              f"against-538 {a['n_against_538_top']}; dropped {len(y['races_dropped_no_winner_match'])}")
        c = y["contested"]
        print(f"{'':>{len(args.name)}} {y['year']} contested (538 fav<0.95): n={c['n_races']} logloss {c['log_loss']:.4f} "
              f"(538 {c['fte_log_loss']:.4f}) brier {c['brier']:.4f} (538 {c['fte_brier']:.4f}) acc {c['accuracy_top']:.3f} "
              f"(538 {c['chalk_accuracy']:.3f})")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
