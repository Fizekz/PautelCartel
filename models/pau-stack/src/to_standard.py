"""Convert the stack's 2026 per-race output into the standard per-race probability format.

Input : results/pk/pk_2026_<TAG>.csv, written by pbase/src/pk_2026.py. One row per in-model race (452: every
        D-vs-R, R-vs-independent and same-party-jungle race; the 18 California top-two and uncontested seats are
        outside the model). Column p_<MODEL> is the model's probability that the D-side candidate wins; the D-side
        is the Democrat, or the independent in a Republican-vs-independent race (NE-SEN).
Output: year,race_id,chamber,q_dem,q_rep,q_other,k_pD_nov1, where q_dem = P(D-side wins), q_rep = 1 - q_dem,
        q_other = 0, and k_pD_nov1 = the Kalshi Nov 1 07:48Z price of the D-side (for reference only; it is not a
        model input).

Usage: python src/to_standard.py [--model noprim] [--tag refreshed] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("noprim", "stack", "v6nl")


def convert(src: Path, model: str) -> pd.DataFrame:
    d = pd.read_csv(src)
    col = f"p_{model}"
    missing = {"race_id", "senate", "k_pD", col} - set(d.columns)
    if missing:
        raise ValueError(f"{src}: missing columns {sorted(missing)}")
    q = d[col].astype(float)
    if not q.between(0, 1).all():
        raise ValueError(f"{src}: {col} outside [0, 1]")
    return pd.DataFrame({
        "year": 2026,
        "race_id": d.race_id,
        "chamber": np.where(d.senate == 1, "Senate", "House"),
        "q_dem": q,
        "q_rep": 1.0 - q,
        "q_other": 0.0,
        "k_pD_nov1": d.k_pD,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=MODELS, default="noprim")
    ap.add_argument("--tag", default="refreshed")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    src = ROOT / "results" / "pk" / f"pk_2026_{a.tag}.csv"
    name = "stack_noprim" if a.model == "noprim" else a.model
    out = a.out or ROOT / "results" / "joint2026" / f"{name}_2026_{a.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    convert(src, a.model).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
