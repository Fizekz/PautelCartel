"""Training table: every 2010-2024 House/Senate general election with a D-or-independent vs R contest."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import CACHE, CLEAN, POLL_WINDOW_SPARSE_DAYS, TRAIN_YEARS
from features import build_hist_features

MAJOR_IND = re.compile(r"major non-D/R candidate (.+?) \(([\d.]+)%\)")


def results() -> pd.DataFrame:
    h = pd.read_csv(CLEAN / "results_house.csv").assign(chamber="House")
    s = pd.read_csv(CLEAN / "results_senate.csv").assign(chamber="Senate")
    r = pd.concat([h, s], ignore_index=True)
    return r[r["year"].isin(TRAIN_YEARS)]


def train_struct() -> pd.DataFrame:
    """Anti-R side = the Democrat, or the major independent when there is no Democrat, the independent won, or he
    out-polled the Democrat (King, Sanders, Orman, McMullin, Osborn). Same-party finals and one-sided races dropped."""
    r = results()
    r = r[~((r["year"] == 2010) & (r["race_id"] == "MA-SEN"))]  # January 2010 special
    rows = []
    for x in r.itertuples():
        notes = x.notes if isinstance(x.notes, str) else ""
        ind = MAJOR_IND.search(notes)
        dem_pct = 100 * x.dem_votes / x.total_votes if x.total_votes and pd.notna(x.dem_votes) else 0.0
        use_ind = ind is not None and (not isinstance(x.dem_cand, str) or x.winner_party == "OTHER"
                                       or float(ind.group(2)) > dem_pct)
        if pd.isna(x.dem_share_2p):
            continue  # same-party top-two final
        if x.uncontested == 1 and not use_ind:
            continue
        if not isinstance(x.rep_cand, str):
            continue
        d_name = ind.group(1) if use_ind else x.dem_cand
        anti_margin = np.nan if use_ind else 100 * (2 * x.dem_share_2p - 1)
        rows.append({"year": x.year, "race_id": x.race_id, "chamber": x.chamber, "state": x.state,
                     "special": x.special, "d_name": d_name, "r_name": x.rep_cand, "d_is_ind": bool(use_ind),
                     "y": int(x.winner_party != "REP"), "margin": anti_margin})
    return pd.DataFrame(rows)


def training_table(refresh: bool = False) -> pd.DataFrame:
    path = CACHE / f"train_features_w{POLL_WINDOW_SPARSE_DAYS}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    t = build_hist_features(train_struct())
    t.to_csv(path, index=False)
    return t


if __name__ == "__main__":
    t = training_table(refresh=True)
    print(t.shape)
    print(t.groupby(["year", "chamber"]).agg(n=("y", "size"), d=("y", "mean"), polls=("poll_n", lambda s: (s > 0).sum()),
                                              fec=("fec_share", lambda s: s.notna().sum()),
                                              prev=("prev_rel", lambda s: s.notna().sum()),
                                              inc=("inc", lambda s: (s != 0).sum())))
    print(t[["r_mean", "lean_m", "base_m", "fec_share", "poll_m", "prev_rel"]].describe().T)
