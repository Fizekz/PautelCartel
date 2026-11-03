"""Convert a Pau-side 2026 party-level forecast (year,race_id,chamber,q_dem,q_rep,q_other[,k_pD_nov1]) to our
candidate-level pool format (race_id,candidate,party,chamber,p_kalshi,q), using a Kavi pool as the candidate frame.

Sides: DEM rows and independents marked "independent on the D side" (e.g. Osborn, NE-SEN) take q_dem; REP rows and
"independent on the R side" take q_rep; everyone else shares q_other. Within a side the mass is split in proportion
to the frame's own q. Races missing from the Pau file are left out (the pool then uses the other models alone).
Checks: the frame's Kalshi D-side price must match Pau's k_pD_nov1 (same Nov 1 snapshot) within --tol.

Usage: python src/final/pau_to_candidates.py FRAME_POOL.csv PAU_2026.csv --out OUT.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIDE_COL = {"D": "q_dem", "R": "q_rep", "O": "q_other"}


def side(party: str, note: str) -> str:
    note = str(note)
    if party == "DEM" or "independent on the D side" in note:
        return "D"
    if party == "REP" or "independent on the R side" in note:
        return "R"
    return "O"


def convert(frame: pd.DataFrame, pau: pd.DataFrame, tol: float) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    rows = []
    pau = pau.set_index("race_id")
    for race, g in frame.groupby("race_id", sort=True):
        if race not in pau.index:
            continue
        r = pau.loc[race]
        g = g.assign(side=[side(p, n) for p, n in zip(g["party"], g.get("note", ""))])
        if "k_pD_nov1" in r and not np.isnan(r["k_pD_nov1"]):
            pd_frame = g.loc[g.side == "D", "p_kalshi"].sum()
            if abs(pd_frame - r["k_pD_nov1"]) > tol:
                issues.append(f"{race}: Kalshi D-side {pd_frame:.3f} in frame vs {r['k_pD_nov1']:.3f} in Pau file")
        out = []
        for s, col in SIDE_COL.items():
            mass = float(r[col])
            gs = g[g.side == s]
            if gs.empty:
                if mass > 1e-3:
                    issues.append(f"{race}: Pau puts {mass:.3f} on side {s}, which has no candidate here; dropped")
                continue
            w = gs["q"].clip(lower=1e-6)
            out.append(gs.assign(q=mass * w / w.sum()))
        o = pd.concat(out)
        rows.append(o.assign(q=o["q"] / o["q"].sum()))
    res = pd.concat(rows)[["race_id", "candidate", "party", "chamber", "p_kalshi", "q"]]
    return res, issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame", type=Path)
    ap.add_argument("pau", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args()
    res, issues = convert(pd.read_csv(args.frame), pd.read_csv(args.pau), args.tol)
    res.to_csv(args.out, index=False)
    print(f"{res.race_id.nunique()} races, {len(res)} candidate rows -> {args.out}")
    for i in issues:
        print("  check:", i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
