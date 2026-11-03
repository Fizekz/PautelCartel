"""pau-g: race directions from the as-of swing geometry (House: district = population-weighted county mix on the lines
in force; Senate: the state's vote-weighted county mean). Writes work/pg_features_<tag>.csv: year, race_id, senate,
g1..gK (z-scored within year over House races; Senate rows on the same scale).
Lines in force (as race/features.csv, so the ACS and learned features cover the same races):
  2012, 2014: 116th lines except PA, NC, FL, VA; 2016: except PA; 2018: 116th; 2020: except NC; 2022: 118th;
  2024: 119th; 2026: 119th except the 9 states redrawn for 2026; 2010: none (2002 lines).
No leakage: cycle t's geometry is built only from elections strictly before t."""
import sys

import numpy as np
import pandas as pd

import pg_geom as G

WORK = G.WORK
LINES = {2012: 116, 2014: 116, 2016: 116, 2018: 116, 2020: 116, 2022: 118, 2024: 119, 2026: 119}
MISS = {2012: {"PA", "NC", "FL", "VA"}, 2014: {"PA", "NC", "FL", "VA"}, 2016: {"PA"}, 2020: {"NC"},
        2026: set("AL CA FL LA NC OH TN TX UT".split())}
YEARS = list(range(2010, 2028, 2))


def weights():
    w = pd.read_csv(WORK / "pg_cd_county.csv")
    return w


def directions(H, E, t, W):
    """House district and Senate (state) directions for cycle t from county embedding E."""
    rows = []
    idx = H.idx
    if t in LINES:
        w = W[W.lines == LINES[t]]
        w = w[~w.state.isin(MISS.get(t, set())) & w.fips.isin(idx)]
        for rid, g in w.groupby("race_id"):
            ii = g.fips.map(idx).values
            p = g["pop"].values
            rows.append(dict(year=t, race_id=rid, senate=0, **{f"g{a+1}": float(np.average(E[ii, a], weights=p)) for a in range(E.shape[1])}))
    rw = H.votes(t)
    for st in np.unique(H.state):
        m = H.state == st
        rows.append(dict(year=t, race_id=f"{st}-SEN", senate=1, **{f"g{a+1}": float(np.average(E[m, a], weights=rw[m])) for a in range(E.shape[1])}))
    return pd.DataFrame(rows)


def build(tag, k, sw_weight=1.0, pres=True, statewide=True, method="als", lam=1.0, first_pres=2000, extra=None):
    H = G.Hist()
    W = weights()
    out = []
    for t in YEARS:
        M, cw, names, yrs = H.matrix(t, sw_weight=sw_weight, pres=pres, statewide=statewide, first_pres=first_pres)
        rw = H.votes(t)
        if k == "log":          # d = c log N, c = 1
            kk = max(1, int(round(np.log(max(M.shape[1], 1)))))
        elif k == "sqrt":       # d = c sqrt N, c = 1
            kk = max(1, int(round(np.sqrt(M.shape[1]))))
        else:
            kk = k
        kk = min(kk, M.shape[1])
        if method == "als":
            E = G.embed(M, rw, cw, kk, lam=lam)
        elif method == "raw":       # the sources themselves (no factorisation), missing -> 0
            E = np.where(np.isfinite(M), M, 0.0)[:, -kk:] if kk else np.zeros((len(H.fips), 0))
        elif callable(method):
            E = method(M, rw, cw, kk, H, t)
        else:
            raise ValueError(method)
        d = directions(H, E, t, W)
        h = d.senate == 0
        for a in range(E.shape[1]):
            c = f"g{a+1}"
            mu, sd = d.loc[h, c].mean(), d.loc[h, c].std()
            if not np.isfinite(sd) or sd == 0:
                mu, sd = d[c].mean(), d[c].std() or 1.0
            d[c] = (d[c] - mu) / sd
        out.append(d)
        print(tag, t, "sources", M.shape[1], "k", kk, flush=True)
    f = pd.concat(out, ignore_index=True)
    f.to_csv(WORK / f"pg_features_{tag}.csv", index=False)
    return f


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]))
