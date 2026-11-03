"""pau-g: history-informed demographic directions ("which demographic mixes have moved together").

The county swing history is regressed on a fixed non-linear demographic basis Phi (college, Hispanic, Black, white
shares: linear, squares and pairwise products, 14 terms; ACS 5-year 2022 for counties, the 1-year district files for
districts), and the fitted swings are reduced to their top-k principal directions (reduced-rank regression):
    S (counties x sources) ~ Phi B,   SVD of the weighted fit Phi B -> k directions a_1..a_k in Phi-space.
A district's direction is Phi(district demographics) a_j, so it uses the district's own demographics (no county
blur inside big counties) while history chooses which demographic mixes matter. Statewide office swings are
within-state quantities, so they are fitted on within-state-demeaned Phi. As of cycle t: only elections before t."""
import numpy as np
import pandas as pd

import pg_geom as G
from pg_features import LINES, MISS, YEARS

WORK = G.WORK
BASE = ["college", "hispanic", "black", "white"]
# fixed scaling (district means / SDs, race/features.csv) so counties and districts share one basis
CEN = dict(college=0.33, hispanic=0.18, black=0.12, white=0.60)
SCL = dict(college=0.10, hispanic=0.17, black=0.13, white=0.20)


def phi(df):
    z = {c: (df[c].values - CEN[c]) / SCL[c] for c in BASE}
    cols = dict(z)
    for i, a in enumerate(BASE):
        for b in BASE[i:]:
            cols[f"{a}*{b}"] = z[a] * z[b]
    return pd.DataFrame(cols, index=df.index)


def county_phi(H):
    a = pd.read_csv(WORK / "pg_county_acs.csv").set_index("fips").reindex(H.fips)
    st = pd.Series(H.state, index=a.index)
    for c in BASE:
        a[c] = a[c].fillna(a[c].groupby(st).transform("mean")).fillna(a[c].mean())
    return phi(a).values


def rrr_directions(H, P, t, k, sw_weight=1.0, lam=1.0, pres=True, statewide=True, shuffle=None):
    """Phi-space directions (p x k) from the sources before t. shuffle=seed: control, county rows of the swing
    matrix permuted (history no longer linked to demographics)."""
    M, cw, names, yrs = H.matrix(t, sw_weight=sw_weight, pres=pres, statewide=statewide)
    if shuffle is not None:
        M = M[np.random.default_rng(shuffle).permutation(M.shape[0])]
    rw = H.votes(t)
    n, p = P.shape
    fits = []
    is_pres = np.array([nm.startswith("pres_") for nm in names])
    for j in range(M.shape[1]):
        z = M[:, j]
        o = np.isfinite(z)
        if o.sum() < p + 5:
            continue
        X = P[o]
        if not is_pres[j]:
            # within-state: demean Phi by state (vote-weighted)
            st = H.state[o]
            Xd = X.copy()
            for s in np.unique(st):
                m = st == s
                Xd[m] -= np.average(X[m], axis=0, weights=rw[o][m])
            X = Xd
        A = np.column_stack([np.ones(o.sum()), X])
        w = rw[o]
        Pen = lam * np.eye(p + 1) * w.sum() / 1e4
        Pen[0, 0] = 0
        b = np.linalg.solve(A.T @ (A * w[:, None]) + Pen, A.T @ (w * z[o]))
        fits.append((b[1:], cw[j]))
    if not fits:
        return np.zeros((p, 0)), np.array([])
    B = np.column_stack([f[0] for f in fits]) * np.sqrt([f[1] for f in fits])[None, :]
    # principal directions of the fitted swings Phi B under county vote weights
    w = rw / rw.sum()
    Pc = (P - w @ P) * np.sqrt(w)[:, None]
    F = Pc @ B
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    # direction a_j in Phi-space: Phi a_j = fitted-swing principal score
    kk = min(k, len(s))
    Afull = B @ Vt[:kk].T          # p x k  (Phi @ Afull = projection of fitted swings on the principal axes)
    return Afull, s[:kk] ** 2 / np.sum(s ** 2)


def district_phi():
    F = pd.read_csv(G.ROOT.parent / "race" / "features.csv")
    F = F[F.college.notna()].copy()
    return F, phi(F).values


def state_phi(H):
    a = pd.read_csv(WORK / "pg_county_acs.csv")
    from pg_data import FIPS2ST
    a["state"] = (a.fips // 1000).map(lambda s: FIPS2ST.get(f"{s:02d}"))
    s = a.groupby("state").apply(lambda g: pd.Series({c: np.average(g[c], weights=g["pop"]) for c in BASE}))
    return s, phi(s).values


def pca_directions(PD, k):
    """Control without history: the top-k principal directions of the districts' own Phi (unweighted)."""
    X = PD - PD.mean(0)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k].T, s[:k] ** 2 / np.sum(s ** 2)


def build(tag, k, sw_weight=1.0, statewide=True, lam=1.0, control=None, shuffle=None):
    H = G.Hist()
    P = county_phi(H)
    F, PD = district_phi()
    S, PS = state_phi(H)
    out = []
    for t in YEARS:
        if control == "pca":
            A, ev = pca_directions(PD[(F.year == t).values] if (F.year == t).any() else PD, k)
        else:
            A, ev = rrr_directions(H, P, t, k, sw_weight=sw_weight, lam=lam, statewide=statewide, shuffle=shuffle)
        if A.shape[1] == 0:
            continue
        m = (F.year == t).values
        d = pd.DataFrame(PD[m] @ A, columns=[f"g{a+1}" for a in range(A.shape[1])])
        d.insert(0, "senate", 0)
        d.insert(0, "race_id", F.race_id.values[m])
        d.insert(0, "year", t)
        s = pd.DataFrame(PS @ A, columns=d.columns[3:])
        s.insert(0, "senate", 1)
        s.insert(0, "race_id", [f"{x}-SEN" for x in S.index])
        s.insert(0, "year", t)
        d = pd.concat([d, s], ignore_index=True)
        h = d.senate == 0
        for c in d.columns[3:]:
            mu, sd = d.loc[h, c].mean(), d.loc[h, c].std()
            if not h.any() or not np.isfinite(sd) or sd == 0:
                mu, sd = d[c].mean(), d[c].std() or 1.0
            d[c] = (d[c] - mu) / sd
        out.append(d)
        print(tag, t, "k", A.shape[1], "var shares", np.round(ev, 2), flush=True)
    f = pd.concat(out, ignore_index=True)
    f.to_csv(WORK / f"pg_features_{tag}.csv", index=False)
    return f
