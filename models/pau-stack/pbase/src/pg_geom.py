"""pau-g: a geometry of swing learned from county election history.

Data (pg_data.py): county two-party results for president 2000-2024 and statewide races (Senate, governor) by county.
Sources (columns of the county x source matrix), built "as of" a cycle t from elections strictly before t:
  - pres swing (y0 -> y1): logit D share change per county, minus its vote-weighted national mean;
  - statewide office swing (office, year, state): the race's county logit relative to its state mean, minus the
    same relative logit in the latest presidential election at or before that year (the race's geography beyond the
    presidential one). Defined only inside its state (missing elsewhere).
Factorisation: weighted low-rank matrix completion (ALS with ridge; rows weighted by votes, missing entries
skipped), then principal axes of the completed matrix -> county embedding E (counties x k), in logit units.
District direction: population-weighted mean of its counties' embeddings (ACS 5-year county-part populations);
Senate (state) direction: vote-weighted mean over the state's counties. Same space, different impact.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
PRES = [2000, 2004, 2008, 2012, 2016, 2020, 2024]
# Alaska (one unit, 02000): 2020 and 2024 statewide presidential totals (Wikipedia has no borough table)
AK = {2020: (153778, 189951), 2024: (140026, 184458)}


class Hist:
    def __init__(self, votes=None):
        v = pd.read_csv(WORK / "pg_county_votes.csv") if votes is None else votes
        v = v[(v.dem > 0) & (v.rep > 0)].copy()
        add = [dict(year=y, office="pres", state="AK", fips=2000, dem=d, rep=r, source=f"pres_{y}") for y, (d, r) in AK.items()
               if not ((v.fips == 2000) & (v.year == y) & (v.office == "pres")).any()]
        v = pd.concat([v, pd.DataFrame(add)], ignore_index=True)
        v["lg"] = np.log(v.dem / v.rep)
        v["tot"] = v.dem + v.rep
        p = v[v.office == "pres"]
        self.fips = np.array(sorted(p.fips.unique()))
        self.idx = {f: i for i, f in enumerate(self.fips)}
        self.state = p.groupby("fips").state.first().reindex(self.fips).values
        n = len(self.fips)
        self.L = np.full((n, len(PRES)), np.nan)
        self.T = np.full((n, len(PRES)), np.nan)
        for y, g in p.groupby("year"):
            j = PRES.index(y)
            ii = g.fips.map(self.idx).values
            self.L[ii, j] = g.lg.values
            self.T[ii, j] = g.tot.values
        # statewide sources
        self.sw = []
        for src, g in v[v.office != "pres"].groupby("source"):
            g = g[g.fips.isin(self.idx)]
            if len(g) < 3 and g.state.iloc[0] != "AK":
                continue
            ii = g.fips.map(self.idx).values
            y = int(g.year.iloc[0])
            self.sw.append(dict(source=src, year=y, office=g.office.iloc[0], state=g.state.iloc[0], ii=ii,
                                lg=g.lg.values, tot=g.tot.values))

    # ----- sources -----
    def votes(self, t):
        """Row weights: votes in the latest presidential election before t."""
        js = [j for j, y in enumerate(PRES) if y < t]
        w = self.T[:, js[-1]].copy()
        for j in reversed(js[:-1]):
            w = np.where(np.isfinite(w), w, self.T[:, j])
        return np.nan_to_num(w, nan=np.nanmedian(w))

    def pres_swing(self, y0, y1):
        a, b = self.L[:, PRES.index(y0)], self.L[:, PRES.index(y1)]
        s = b - a
        w = np.nan_to_num(self.T[:, PRES.index(y1)])
        ok = np.isfinite(s)
        s[ok] -= np.average(s[ok], weights=w[ok])
        return s

    def office_swing(self, src):
        """County logit of a statewide race relative to its state, minus the same for the latest pres <= year."""
        n = len(self.fips)
        out = np.full(n, np.nan)
        yp = max(y for y in PRES if y <= src["year"])
        ii, lg, w = src["ii"], src["lg"], src["tot"]
        rel = lg - np.average(lg, weights=w)
        P = self.L[ii, PRES.index(yp)]
        ok = np.isfinite(P)
        if ok.sum() < 1:
            return out
        prel = P - np.average(P[ok], weights=w[ok])
        out[ii[ok]] = (rel - prel)[ok]
        return out

    def matrix(self, t, sw_weight=1.0, pres=True, statewide=True, first_pres=2000, last_year=None):
        """County x source matrix of everything before t (or up to last_year), column weights, names."""
        cols, cw, names, yrs = [], [], [], []
        hi = t if last_year is None else last_year + 1
        if pres:
            for y0, y1 in zip(PRES[:-1], PRES[1:]):
                if y1 < hi and y0 >= first_pres:
                    cols.append(self.pres_swing(y0, y1)); cw.append(1.0); names.append(f"pres_{y0}_{y1}"); yrs.append(y1)
        if statewide:
            for s in self.sw:
                if first_pres <= s["year"] < hi:
                    c = self.office_swing(s)
                    if np.isfinite(c).sum() == 0:
                        continue
                    cols.append(c); cw.append(sw_weight); names.append(s["source"]); yrs.append(s["year"])
        M = np.column_stack(cols) if cols else np.zeros((len(self.fips), 0))
        return M, np.array(cw), names, np.array(yrs)


def als(M, rw, cw, k, lam=1.0, iters=30, seed=0):
    """Weighted low-rank completion of M (nan = missing): min sum rw_i cw_j (M_ij - U_i V_j)^2 + lam(|U|^2 + |V|^2)."""
    n, m = M.shape
    obs = np.isfinite(M)
    X = np.where(obs, M, 0.0)
    rng = np.random.default_rng(seed)
    V = rng.normal(scale=0.1, size=(m, k))
    U = np.zeros((n, k))
    # group rows by observation pattern (a state's counties share one)
    keys = {}
    for i in range(n):
        keys.setdefault(obs[i].tobytes(), []).append(i)
    groups = [(np.array(ix), obs[ix[0]]) for ix in keys.values()]
    I = np.eye(k)
    for _ in range(iters):
        for ix, o in groups:
            Vo = V[o] * cw[o][:, None]
            A = V[o].T @ Vo + lam * I
            B = X[np.ix_(ix, o)] @ Vo
            U[ix] = np.linalg.solve(A, B.T).T
        for j in range(m):
            o = obs[:, j]
            Uo = U[o] * rw[o][:, None]
            A = U[o].T @ Uo + lam * I
            V[j] = np.linalg.solve(A, Uo.T @ X[o, j])
    return U, V


def embed(M, rw, cw, k, lam=1.0, iters=30):
    """County embedding E (n x k) on principal axes of the completed matrix (column weights applied), logit units."""
    if k == 0 or M.shape[1] == 0:
        return np.zeros((M.shape[0], 0))
    k = min(k, M.shape[1])
    U, V = als(M, rw, cw, k, lam=lam, iters=iters)
    Xh = U @ V.T * np.sqrt(cw)[None, :]
    w = rw / rw.sum()
    mu = w @ Xh
    Xc = (Xh - mu) * np.sqrt(w)[:, None]
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    E = (Xh - mu) @ Vt[:k].T
    # sign convention: each axis positively correlated with the vote-weighted county D lean is not meaningful;
    # orient so that the largest-|loading| source has a positive loading
    for a in range(E.shape[1]):
        j = np.argmax(np.abs(Vt[a]))
        if Vt[a, j] < 0:
            E[:, a] *= -1
    return E


def pca_complete(M, rw, cw, k):
    """Fast alternative: impute missing with 0 (column means are ~0 by construction), weighted PCA."""
    X = np.where(np.isfinite(M), M, 0.0) * np.sqrt(cw)[None, :]
    w = rw / rw.sum()
    mu = w @ X
    Xc = (X - mu) * np.sqrt(w)[:, None]
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (X - mu) @ Vt[:k].T


def ridge_fit_predict(F, z, w, obs, lam=1e-3):
    A = np.column_stack([np.ones(len(F)), F])
    Ao, zo, wo = A[obs], z[obs], w[obs]
    P = lam * np.eye(A.shape[1]) * wo.sum()
    P[0, 0] = 0
    b = np.linalg.solve(Ao.T @ (Ao * wo[:, None]) + P, Ao.T @ (wo * zo))
    return A @ b


def heldout_r2(F, z, w, frac=0.1, ndraw=30, lam=1e-3, seed=0, groups=None):
    """Observe a vote-weighted random subset of counties (or of states, if groups is given), predict the rest.
    Returns the mean out-of-sample weighted R^2 (vs the observed-subset mean)."""
    ok = np.isfinite(z) & np.all(np.isfinite(F), axis=1) if F.shape[1] else np.isfinite(z)
    rng = np.random.default_rng(seed)
    r2 = []
    idx = np.where(ok)[0]
    for _ in range(ndraw):
        if groups is None:
            p = w[idx] / w[idx].sum()
            m = max(int(frac * len(idx)), F.shape[1] + 5)
            pick = rng.choice(idx, size=min(m, len(idx) - 5), replace=False, p=p)
        else:
            g = np.unique(groups[idx])
            gs = rng.choice(g, size=max(1, int(round(frac * len(g)))), replace=False)
            pick = idx[np.isin(groups[idx], gs)]
        obs = np.zeros(len(z), bool)
        obs[pick] = True
        rest = ok & ~obs
        if rest.sum() < 5 or obs.sum() < F.shape[1] + 3:
            continue
        zh = ridge_fit_predict(F[:, :] if F.shape[1] else np.zeros((len(z), 0)), np.nan_to_num(z), w, obs, lam)
        zbar = np.average(z[obs], weights=w[obs])
        sse = np.sum(w[rest] * (z[rest] - zh[rest]) ** 2)
        sst = np.sum(w[rest] * (z[rest] - zbar) ** 2)
        r2.append(1 - sse / sst)
    return float(np.mean(r2)) if r2 else np.nan
