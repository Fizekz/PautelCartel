"""2026 forecast: fit on 2010-2024, predict every race, write per-candidate probabilities.

Outputs
  out/predictions_2026.csv : race_id, chamber, party, candidate_name (Kalshi naming), q  (sums to 1 per race)
  out/submission_2026.csv  : race_id, chamber, predicted_winner_party, candidate_name, win_probability
  work/pred_detail_2026.csv: model internals per race
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import KALSHI, WORK, OUT, surname, norm_name
from model import WinModel, FINAL
import samparty

MINOR_TOTAL = 0.004      # total probability left for minor candidates Kalshi lists
UNOPPOSED_Q = 0.999
VS_MINOR_Q = 0.995
WITHDRAWN_D = 0.002      # Democrats who withdrew but remain listed (ID, NE, SD)
SPLIT_D_ON_BALLOT = 0.006  # MT: Democrat still on the ballot behind the independent; AK-AL Hafner


def parse_kalshi(s):
    out = []
    for part in s.split("|"):
        bits = part.strip().split(":")
        party, price = bits[0], float(bits[-1])
        name = ":".join(bits[1:-1])
        out.append((party, name, price))
    return out


def match(cands, name, party=None):
    """Index of the Kalshi candidate matching `name` (surname) or, failing that, `party`."""
    if name:
        for i, (p, n, _) in enumerate(cands):
            if surname(n) and surname(n) == surname(name):
                return i
    if party:
        idx = [i for i, (p, n, _) in enumerate(cands) if p == party]
        if len(idx) == 1:
            return idx[0]
        if idx:
            gen = [i for i in idx if "party" in cands[i][1].lower()]
            return gen[0] if gen else idx[0]
    return None


def assign(cands, major):
    """major: dict index -> prob for the main candidates; the rest share what is left."""
    q = np.zeros(len(cands))
    for i, v in major.items():
        q[i] += v
    rest = [i for i in range(len(cands)) if i not in major]
    left = 1 - q.sum()
    if rest:
        q[rest] = max(left, 0) / len(rest)
    else:
        q = q / q.sum()
    return q / q.sum()


def main():
    hist = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
    f26 = pd.read_csv(WORK / "features_2026.csv", low_memory=False)
    wm = WinModel(list(range(2010, 2026, 2)), FINAL).fit(hist)
    print("2026 fit:", {k: v for k, v in wm.summary().items() if k != "coefs"})
    print("coefs:", wm.summary()["coefs"])

    std = f26[f26.kind.isin(["D_vs_R", "R_vs_IND", "jungle_party"])].copy()
    pred = wm.predict(std)
    f26 = f26.merge(pred[["race_id", "yhat", "mu", "sd", "sd_total", "p_dside"]], on="race_id", how="left")
    sp = samparty.predict_2026(f26[f26.kind == "same_party"])
    f26 = f26.merge(sp, on="race_id", how="left")
    f26.to_csv(WORK / "pred_detail_2026.csv", index=False)

    k = pd.read_csv(KALSHI)
    rows, sub, sim = [], [], []
    for x in f26.itertuples():
        kr = k[k.race_id == x.race_id].iloc[0]
        cands = parse_kalshi(kr.candidates)
        major = {}
        if x.kind in ("D_vs_R", "jungle_party"):
            di = match(cands, x.dside_name if x.kind == "D_vs_R" else None, "DEM")
            rparty = "IND" if x.race_id == "CA-06" else "REP"
            ri = match(cands, x.rside_name if x.kind == "D_vs_R" else None, rparty)
            n_minor = len(cands) - 2
            pm = MINOR_TOTAL if n_minor > 0 else 0.0
            major = {di: x.p_dside * (1 - pm), ri: (1 - x.p_dside) * (1 - pm)}
        elif x.kind == "R_vs_IND":
            ii = match(cands, x.dside_name, "IND")
            ri = match(cands, x.rside_name, "REP")
            di = match(cands, None, "DEM")
            dq = SPLIT_D_ON_BALLOT if x.race_id in ("MT-SEN", "AK-AL") else WITHDRAWN_D
            major = {ii: x.p_dside * (1 - dq), ri: (1 - x.p_dside) * (1 - dq)}
            if di is not None:
                major[di] = dq
        elif x.kind == "same_party":
            i1 = match(cands, x.sp_name1)
            i2 = match(cands, x.sp_name2)
            major = {i1: x.sp_p1, i2: 1 - x.sp_p1}
        elif x.kind in ("unopposed", "vs_minor"):
            qv = UNOPPOSED_Q if x.kind == "unopposed" else VS_MINOR_Q
            name = x.dside_name or x.rside_name
            party = "DEM" if x.dside_name else "REP"
            major = {match(cands, name, party): qv}
        if None in major:
            raise ValueError(f"could not match candidates for {x.race_id}: {cands} / {x.dside_name} / {x.rside_name}")
        q = assign(cands, major)
        keys = list(major.keys())
        if x.kind in ("D_vs_R", "jungle_party", "R_vs_IND"):
            sim.append(dict(race_id=x.race_id, chamber=x.chamber, state=x.state, kind=x.kind,
                            dside_cand=cands[keys[0]][1], rside_cand=cands[keys[1]][1], p_dside=x.p_dside,
                            mu=x.mu, sd_total=x.sd_total))
        else:
            sim.append(dict(race_id=x.race_id, chamber=x.chamber, state=x.state, kind=x.kind))
        for (p, n, _), qq in zip(cands, q):
            rows.append(dict(race_id=x.race_id, chamber=x.chamber, party=p, candidate_name=n, q=qq))
        b = int(np.argmax(q))
        sub.append(dict(race_id=x.race_id, chamber=x.chamber, predicted_winner_party=cands[b][0],
                        candidate_name=cands[b][1], win_probability=q[b]))
    pd.DataFrame(sim).to_csv(WORK / "sim_inputs_2026.csv", index=False)
    pr = pd.DataFrame(rows)
    pr.to_csv(OUT / "predictions_2026.csv", index=False)
    sb = pd.DataFrame(sub)
    sb.to_csv(OUT / "submission_2026.csv", index=False)
    assert pr.groupby("race_id").q.sum().sub(1).abs().max() < 1e-9
    assert len(sb) == 470

    # compare with Kalshi favourites
    k["k_fav"] = k.fav_candidate
    cmp = sb.merge(k[["race_id", "fav_candidate", "fav_party", "fav_norm"]], on="race_id")
    diff = cmp[cmp.candidate_name != cmp.fav_candidate]
    diff.to_csv(WORK / "fav_differs_from_kalshi.csv", index=False)
    print(f"favourite differs from Kalshi in {len(diff)} races")
    print(diff.to_string())
    seats = f26.assign(pd_=np.where(f26.kind.isin(["D_vs_R", "jungle_party"]), f26.p_dside, np.nan))
    dd = f26.loc[(f26.kind == "same_party") & f26.dem_candidate.notna(), "race_id"]   # CA D-vs-D seats
    print("expected D House seats (model, incl. D-vs-D seats):",
          round(pr[(pr.chamber == "House") & (pr.party == "DEM")].q.sum() + len(dd), 1))
    print("expected D Senate seats up (model):", round(pr[(pr.chamber == "Senate") & (pr.party == "DEM")].q.sum(), 1))


if __name__ == "__main__":
    main()
