"""10,000 joint draws of the 2026 winners (out/sims_2026.csv.gz) and seat ranges.

Latent D-side margin of race i in draw k (v3: applied as a Gaussian copula on the published
probability, so the Student-t marginals are kept exactly):
    m_ik = mu_i + N_k + C_k * senate_i + G_{k, state(i)} + L_ik
  N_k  national (year-effect) error         ~ N(0, VAR_NAT)
  C_k  extra Senate-wide error               ~ N(0, VAR_SEN)
  G    state error, shared by every race in a state (House and Senate) ~ N(0, VAR_STATE)
  L    race error, variance S_i^2 - (the above), S_i = sd_i / a (the model's total predictive SD),
       so every race keeps exactly its published marginal probability.
The three variances are method-of-moments estimates from the final model's leave-one-year-out
residuals in rated races, 2010-2024 (see LOG.md). Minor candidates win a draw with their published
q (independently); same-party, unopposed and no-major-opponent races are drawn independently.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import FEAT, OUT, WORK
from model import WinModel, FINAL

N_DRAWS = 10_000
SEED = 20261103
D_SENATE_TOTAL_BEFORE = 47   # Democratic caucus seats in the outgoing Senate (incl. King, Sanders)


def variance_components():
    h = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
    wm = WinModel(list(range(2010, 2026, 2)), FINAL).fit(h)
    lo = wm.loyo_.copy()
    lo["e"] = lo.y - lo.mu
    r = lo[lo.rating.abs() < 2.5]
    ym = r.groupby("year").e.agg(["mean", "var", "size"])
    var_nat = float(ym["mean"].var() - np.mean(ym["var"] / ym["size"]))
    cm = r.groupby(["year", "senate"]).e.agg(["mean", "var", "size"]).unstack()
    diff = cm["mean"][1] - cm["mean"][0]
    noise = float((cm["var"][1] / cm["size"][1] + cm["var"][0] / cm["size"][0]).mean())
    var_sen = max(float(diff.var() - noise), 0.0)
    hh = r[r.senate == 0].copy()
    hh["e2"] = hh.e - hh.groupby("year").e.transform("mean")
    sm = hh.groupby(["year", "state"]).e2.agg(["mean", "size"])
    sm = sm[sm["size"] >= 2]
    var_state = max(float((sm["mean"] ** 2).mean() - np.mean(hh.e2.var() / sm["size"])), 0.0)
    return dict(var_nat=var_nat, var_sen=var_sen, var_state=var_state)


def main():
    vc = variance_components()
    si = pd.read_csv(WORK / "sim_inputs_2026.csv")
    pr = pd.read_csv(OUT / "predictions_2026.csv")
    ri = pd.read_csv(FEAT / "race_info_2026.csv").set_index("race_id")
    rng = np.random.default_rng(SEED)
    states = sorted(si.state.unique())
    sidx = {s: i for i, s in enumerate(states)}
    Nz = rng.normal(0, 1, N_DRAWS)
    Cz = rng.normal(0, 1, N_DRAWS)
    Gz = rng.normal(0, 1, (N_DRAWS, len(states)))

    out = {}
    floor_hits = 0
    for x in si.itertuples():
        c = pr[pr.race_id == x.race_id]
        names, qs = c.candidate_name.values, c.q.values
        u = rng.random(N_DRAWS)
        if isinstance(x.dside_cand, str):
            # Gaussian copula (v3): shared factors keep their variance share of the race's predictive
            # SD; the race wins with exactly its published probability (the v3 link is a Student-t)
            S2 = x.sd_total ** 2
            r_nat, r_st = vc["var_nat"] / S2, vc["var_state"] / S2
            r_sen = vc["var_sen"] / S2 if x.chamber == "Senate" else 0.0
            tot = r_nat + r_st + r_sen
            if tot > 0.9:
                floor_hits += 1
                r_nat, r_st, r_sen = (v * 0.9 / tot for v in (r_nat, r_st, r_sen))
                tot = 0.9
            u_lat = (np.sqrt(r_nat) * Nz + np.sqrt(r_sen) * Cz * (x.chamber == "Senate")
                     + np.sqrt(r_st) * Gz[:, sidx[x.state]] + np.sqrt(1 - tot) * rng.normal(0, 1, N_DRAWS))
            m = u_lat - norm.ppf(1 - x.p_dside)
            win = np.where(m > 0, x.dside_cand, x.rside_cand).astype(object)
            minor = [(n, q) for n, q in zip(names, qs) if n not in (x.dside_cand, x.rside_cand)]
            if minor:
                cum = np.cumsum([q for _, q in minor])
                for j, (n, _) in enumerate(minor):
                    lo = cum[j - 1] if j else 0.0
                    win[(u >= lo) & (u < cum[j])] = n
        else:   # independent categorical draw with the published q
            cum = np.cumsum(qs)
            win = np.array(names, dtype=object)[np.minimum(np.searchsorted(cum, u * cum[-1]), len(names) - 1)]
        out[x.race_id] = win
    sims = pd.DataFrame(out)
    sims.to_csv(OUT / "sims_2026.csv.gz", index=False, compression="gzip")

    # seat tallies: party of each winner (same-party CA races: the party of both finalists)
    party = {(r.race_id, r.candidate_name): r.party for r in pr.itertuples()}
    def seat_party(rid, n):
        p = party[(rid, n)]
        if p == "UNK":
            return "DEM" if isinstance(ri.loc[rid, "dem_candidate"], str) else "REP"
        return p
    house = [r for r in sims.columns if ri.loc[r, "chamber"] == "House"]
    sen = [r for r in sims.columns if ri.loc[r, "chamber"] == "Senate"]
    P = pd.DataFrame({r: [seat_party(r, n) for n in sims[r]] for r in sims.columns})
    hd = (P[house] == "DEM").sum(1)
    sd_ = (P[sen] == "DEM").sum(1)
    si_ = (P[sen] == "IND").sum(1)
    d_up = int((ri.loc[sen, "incumbent_party"] == "DEM").sum())
    holdover = D_SENATE_TOTAL_BEFORE - d_up
    tot = holdover + sd_
    q = lambda s, a: float(np.quantile(s, a))
    res = dict(var_components=vc, floor_hits=floor_hits,
               house_d=dict(mean=float(hd.mean()), p10=q(hd, .1), p50=q(hd, .5), p90=q(hd, .9),
                            p_majority=float((hd >= 218).mean())),
               senate_d_won_of_35=dict(mean=float(sd_.mean()), p10=q(sd_, .1), p90=q(sd_, .9)),
               senate_ind_won=dict(mean=float(si_.mean())),
               senate_d_total=dict(holdover=holdover, mean=float(tot.mean()), p10=q(tot, .1), p50=q(tot, .5),
                                   p90=q(tot, .9), p_d_control_51=float((tot >= 51).mean()),
                                   p_d_50_or_more=float((tot >= 50).mean())))
    pd.Series(res).to_json(WORK / "sim_summary.json", indent=1)
    print(res)
    # check: simulated marginals vs published q
    chk = []
    for r in sims.columns:
        vc_ = sims[r].value_counts(normalize=True)
        for n, qq in zip(pr[pr.race_id == r].candidate_name, pr[pr.race_id == r].q):
            chk.append(abs(vc_.get(n, 0) - qq))
    print("max |sim share - q|:", round(max(chk), 4))


if __name__ == "__main__":
    main()
