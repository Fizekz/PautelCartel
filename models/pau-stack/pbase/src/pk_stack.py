"""pau-k: do the small gains of the 1102 research studies stack? One harness with every component as a cfg switch.

Base = v6+nl: PE.all_(v5.1, pr_pat_lam=1, pr_pat_set="nl", pr_pat_unp=True), nl columns as pr_theta_nl2.py 8-11.
Hook chain on WinModel._mu (inner -> outer), each a copy of the study's own hook:
  shrink (v5.1) <- PATTERN (this file: v6+nl's per-cycle ridge, generalised) <- PX (pau-x add-on columns before the
  poll shrinkage) <- pt_prop (pau-t state field, unpolled House) <- pt_senate (pau-t Senate state node)
Stage-1 design: pr_extra = ["rchg_unp"] (pau-e late rating move, unpolled rated races).
Post-hoc: logit pool with pau-w's Kalman filter (c2_tau30, cached by pk_kf.py), weight w on the filter.

Components (cfg keys):
  node   pau-t Senate state node        sen_cfg pt_senh=True (shrink 2)
  field  pau-t state-graph field        main + pool: pt_graphs=["state"], beta 5, snr 0.3, kernel A ("mean")
  rchg   pau-e late rating move         pr_extra ["rchg_unp"] (all components)
  geo    pau-g history-chosen dirs      pk_geo=3: g1-g3 of pg_features_rrrp.csv in the pattern, own ridge 10
  cells  pau-v voter weighting (CES-free) pk_cells="flat"|"eco": the pattern on citizen electorate shares (8 ACS
         cells, race4 x BA+), share + squares, ridge 10, centred over the year's races (pv_hook 'share' + pv_sq).
         "flat" = flat turnout; "eco" = pau-v's ecological turnout profile from past district totals, with a FLAT
         fallback where pau-v fell back to CES turnout (2012). No CES file is read.
  prim   pau-x House primaries          px_feats ["prim_dev"] (or prim_dev_s), px_where house, ridge 10
  pool   pau-w Kalman pool              pk_pool_w = 0.3
  clip   safe-seat clip                 pk_clip = c: |pattern shift| <= c points where |rating| >= 2.5
Placebos (pk_plc = seed): every added signal shuffled within year (see placebo())."""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate4 as E
import pr_eval as PE
import pr_pattern as PP
import model as M
from common import WORK
from evaluate import metrics

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "pk"
RES.mkdir(parents=True, exist_ok=True)
SD = WORK / "stack"
SD.mkdir(exist_ok=True)

# ---------------- v6+nl columns (pr_theta_nl2.py lines 8-11) ----------------
Z = PP.Z
for a, b in (("college", "college"), ("hispanic", "hispanic"), ("black", "black"), ("college", "hispanic")):
    v = Z[a] * Z[b]
    Z[f"{a}_x_{b}"] = (v - v.mean()) / v.std()
PP.SETS["nl"] = ["college", "hispanic", "black", "college_x_college", "hispanic_x_hispanic", "black_x_black",
                 "college_x_hispanic"]
# ---------------- pau-g: history-chosen directions (pg_demo 'rrrp', House rows) as extra PP.Z columns -------------
_G = pd.read_csv(WORK / "pg_features_rrrp.csv")
_G = _G[_G.senate == 0].set_index(["year", "race_id"])[["g1", "g2", "g3"]].add_prefix("geo_")
PP.Z = Z.join(_G, how="outer").fillna(0.0)          # as pg_hook.install (products computed before the join)
GEO = ["geo_g1", "geo_g2", "geo_g3"]
Z_REAL = PP.Z.copy()

# ---------------- pau-v CES-free cells ----------------
CELLS = [f"{r}_{e}" for r in ("wh", "bl", "hi", "ot") for e in ("nc", "co")]
_C = pd.read_csv(ROOT / "race" / "cells_v.csv")
SH = {(int(y), r): v for y, r, v in zip(_C.year, _C.race_id, _C[[f"c_{c}" for c in CELLS]].values)}
SRC = {(int(y), r): s for y, r, s in zip(_C.year, _C.race_id, _C.src)}
SH_REAL = dict(SH)
_L = np.column_stack([np.array([c.endswith("co") for c in CELLS], float),
                      np.array([c.startswith("bl") for c in CELLS], float),
                      np.array([c.startswith("hi") for c in CELLS], float)])


def _eco_turnout():
    """pau-v's ecological profile (pv_struct._eco_turnout): district total House votes / citizens 25+ on the citizen
    cell shares (NNLS), contested House races with matching-lines ACS; year y's profile is used only after y."""
    from scipy.optimize import nnls
    h = pd.read_csv(WORK / "features_hist.csv", low_memory=False, usecols=["year", "race_id", "senate", "contest_type", "total_votes"])
    d = h[(h.senate == 0) & (h.contest_type == "D_vs_R") & (h.total_votes > 0)].merge(
        _C[_C.src == "cd"][["year", "race_id", "cit25"] + [f"c_{c}" for c in CELLS]], on=["year", "race_id"])
    out = {}
    for y, g in d.groupby("year"):
        b, _ = nnls(g[[f"c_{c}" for c in CELLS]].values, (g.total_votes / g.cit25).values)
        out[int(y)] = np.maximum(b, 0.02)
    return out


ECO = _eco_turnout()


def turnout(y, mode):
    if mode == "eco":
        c = [t for t in ECO if t <= y - 4 and (t % 4) == (y % 4)] or [t for t in ECO if t <= y - 2]
        if c:
            return ECO[max(c)]
    return np.ones(8)                                   # flat (also the CES-free fallback of 'eco')


def cell_X(df, mode):
    """n x 7: 50 x electorate shares of (college, Black, Hispanic), then 100 x their squares and college x Hispanic
    (pv_hook.fit_shift / sq_feats); ok = district has matching-lines ACS cells."""
    n = len(df)
    W = np.zeros((n, 8)); ok = np.zeros(n, bool)
    for i, (y, r) in enumerate(zip(df.year.values, df.race_id.values)):
        s = SH.get((int(y), r))
        if s is None or SRC.get((int(y), r)) != "cd":
            continue
        w = s * turnout(int(y), mode)
        W[i] = w / w.sum(); ok[i] = True
    sh = W @ _L
    sq = np.column_stack([sh[:, 0] ** 2, sh[:, 1] ** 2, sh[:, 2] ** 2, sh[:, 0] * sh[:, 2]]) * 100.0
    return np.column_stack([50.0 * sh, sq]), ok


# ---------------- PATTERN hook (replaces pr_pattern's _mu; identical to it for the base) ----------------
PLOG = []


def pattern_k(df, cfg):
    add = np.zeros(len(df))
    house = df.senate.values == 0
    pol = house & (df.poll_pollsters.fillna(0).values > 0) & np.isfinite(df.poll_margin.values.astype(float))
    yv = df.year.values
    idx = pd.MultiIndex.from_arrays([df.year.values, df.race_id.values])
    cells = cfg.get("pk_cells")
    if cells:
        X, ok = cell_X(df, cells)
        pen = np.full(X.shape[1], cfg.get("pk_cells_lam", 10.0))
    else:
        X = PP.Z.reindex(idx)[PP.SETS[cfg.get("pr_pat_set", "nl")]].fillna(0.0).values
        ok = np.ones(len(df), bool)
        pen = np.full(X.shape[1], cfg.get("pr_pat_lam", 1.0))
    if cfg.get("pk_geo"):
        Xg = PP.Z.reindex(idx)[GEO[:cfg["pk_geo"]]].fillna(0.0).values
        X = np.column_stack([X, Xg])
        pen = np.r_[pen, np.full(Xg.shape[1], cfg.get("pk_geo_lam", 10.0))]
    for y in np.unique(yv):
        my = house & (yv == y) & ok
        m = pol & my
        if m.sum() < 10:
            continue
        g = np.clip(df.poll_margin.values[m].astype(float) - df.yhat.values[m], -25, 25)
        n = df.poll_pollsters.values[m].astype(float)
        w = n / (n + 1.2)
        A = np.column_stack([np.ones(m.sum()), X[m]])
        b = np.linalg.solve(A.T @ (A * w[:, None]) + np.diag(np.r_[0.0, pen]), A.T @ (w * g))
        f = X[my] @ b[1:]
        if cells:
            f = f - f.mean()                            # pau-v: centred over the year's races
        full = np.zeros(len(df)); full[my] = f
        sel = my & ~pol if cfg.get("pr_pat_unp", False) else my
        add[sel] = full[sel]
    c = cfg.get("pk_clip")
    if c is not None:
        safe = np.abs(df.rating.values.astype(float)) >= 2.5
        add[safe] = np.clip(add[safe], -c, c)
    return add


def _mu_pat(self, df):
    c = self.cfg
    if c.get("pr_pat_lam") is None and not c.get("pk_cells"):
        return PP._orig(self, df)
    d = df.copy()
    d["yhat"] = d.yhat.values + pattern_k(d, c)
    return PP._orig(self, d)


M.WinModel._mu = _mu_pat

# ---------------- PX hook (copy of px_harness._mu) ----------------
FEATS = {}
FEATS_REAL = {}
_prev_px = M.WinModel._mu


def _px_mask(df, where):
    sen = df.senate.values == 1
    pol = (df.poll_pollsters.fillna(0).values > 0) & np.isfinite(df.poll_margin.values.astype(float))
    return {"all": np.ones(len(df), bool), "house": ~sen, "house_unp": (~sen) & (~pol)}[where]


def _mu_px(self, df):
    names = self.cfg.get("px_feats")
    if not names:
        return _prev_px(self, df)
    mu0 = _prev_px(self, df)
    idx = pd.MultiIndex.from_arrays([df.year.values.astype(int), df.race_id.astype(str).values])
    X = np.column_stack([FEATS[n].reindex(idx).fillna(0.0).values for n in names])
    X = X * _px_mask(df, self.cfg.get("px_where", "house"))[:, None]
    D = np.zeros_like(X)
    for j in range(X.shape[1]):
        d = df.copy()
        d["yhat"] = d.yhat.values + X[:, j]
        D[:, j] = _prev_px(self, d) - mu0
    if not hasattr(self, "pxk_"):
        m = (np.abs(df.rating.values) < 2.5) & (np.abs(D).sum(1) > 0)
        if "y" in df and m.sum() >= 8:
            r = df.y.values[m] - mu0[m]
            A = D[m]
            self.pxk_ = np.linalg.solve(A.T @ A + self.cfg.get("px_lam", 10.0) * np.eye(A.shape[1]), A.T @ r)
        else:
            self.pxk_ = np.zeros(X.shape[1])
    return mu0 + D @ self.pxk_


M.WinModel._mu = _mu_px


def _centre(v, key):
    v = v.copy()
    has = v.notna()
    k = pd.Series(key, index=v.index)
    v[has] = v[has] - v[has].groupby(k[has]).transform("mean")
    return v.fillna(0.0)


def _prim():
    """prim_dev / prim_dev_s exactly as px_primaries.build (from its saved output work/px/primaries.csv) + 2026
    (px_2026.py: 2026 primary margin - 2 lean, clipped +-40, centred over 2026's districts)."""
    p = pd.read_csv(WORK / "px" / "primaries.csv")
    idx = pd.MultiIndex.from_arrays([p.year.values, p.race_id.values])
    raw = pd.Series(p.raw.values, index=idx)
    out = {"prim_dev": _centre(raw, p.year.values),
           "prim_dev_s": _centre(raw, (p.year.astype(str) + p.state).values)}
    try:
        g = pd.read_csv(WORK / "px" / "primaries_2026.csv")
        f26 = pd.read_csv(WORK / "features_2026.csv", low_memory=False)
        std = f26[f26.kind.isin(["D_vs_R", "R_vs_IND", "jungle_party"]) & (f26.senate == 0)]
        ids = set(std.race_id)
        g["race_id"] = [f"{s}-AL" if f"{s}-AL" in ids else f"{s}-{int(d):02d}" for s, d in zip(g.state, g.dist)]
        g = g.merge(std[["race_id", "lean"]], on="race_id", how="inner")
        g = g[(g.dv > 0) & (g.rv > 0) & ~g.state.isin(["LA"])].copy()
        g["raw"] = np.clip(100 * (g.dv - g.rv) / (g.dv + g.rv) - 2 * g.lean, -40, 40)
        i26 = pd.MultiIndex.from_arrays([[2026] * len(g), g.race_id.values])
        out["prim_dev"] = pd.concat([out["prim_dev"], pd.Series((g.raw - g.raw.mean()).values, index=i26)])
        r26 = g.raw - g.raw.groupby(g.state).transform("mean")
        out["prim_dev_s"] = pd.concat([out["prim_dev_s"], pd.Series(r26.values, index=i26)])
    except FileNotFoundError:
        pass
    for k, s in out.items():
        s.index = pd.MultiIndex.from_tuples([(int(a), str(b)) for a, b in s.index])
        FEATS[k] = FEATS_REAL[k] = s.astype(float)


_prim()

# ---------------- pau-t hooks (imported last: they wrap PX) ----------------
import pt_prop as PT      # noqa: E402
import pt_senate as PS    # noqa: E402

# ---------------- pau-w pool (post-hoc, on the top-level hybrid model's predictions) ----------------
_pred0 = M.WinModel.predict
KF_PERM = [None]


def _kf(T, tr):
    import pk_kf
    d = pk_kf.get(int(T), tuple(int(y) for y in tr))
    return dict(zip(d.race_id, d.p))


def _predict(self, df):
    out = _pred0(self, df)
    w = self.cfg.get("pk_pool_w")
    if not w or not self.cfg.get("sen_cfg") or getattr(self, "_in_hybrid", False):
        return out
    out = out.copy()
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    for T in np.unique(out.year.values):
        m = out.year.values == T
        kf = _kf(T, self.train_years)
        q = out.race_id[m].map(kf).values.astype(float)
        if KF_PERM[0] is not None:
            q = q[np.random.default_rng(KF_PERM[0] * 100 + int(T)).permutation(len(q))]
        p = out.p_dside.values[m]
        pooled = 1 / (1 + np.exp(-((1 - w) * lg(p) + w * lg(np.where(np.isfinite(q), q, p)))))
        out.loc[m, "p_dside"] = pooled
    return out


M.WinModel.predict = _predict

# ---------------- data ----------------
HIST0 = pd.read_csv(WORK / "features_hist.csv", low_memory=False)


def add_rchg(h, f=None):
    """pau-e add_cols (rchg part only): rchg_unp = rchg_res x unpolled (no 21-day and no 120-day poll) x rated."""
    f = pd.read_csv(WORK / "pe_rchg_features.csv") if f is None else f
    h = h.merge(f[["year", "race_id", "rchg_res"]], on=["year", "race_id"], how="left")
    h["rchg_res"] = h.rchg_res.fillna(0.0)
    unp = ((h.poll_pollsters.fillna(0) == 0) & (h.pn4.fillna(0) == 0)).astype(float)
    h["rchg_unp"] = h.rchg_res * unp * (h.rating.abs() < 2.5)
    return h


HIST = add_rchg(HIST0)
BASE = PE.all_(PE.V["v51"], pr_pat_lam=1.0, pr_pat_set="nl", pr_pat_unp=True)
FIELD_KW = dict(pt_graphs=["state"], pt_gw=[1], pt_beta=5, pt_snr=0.3, pt_knorm="mean")


def build(comps, clip=None, prim="prim_dev", cells_mode="flat", pool_w=0.3, plc=None):
    """cfg for a set of components (names as in the module docstring)."""
    c = dict(BASE)
    kw = {}
    if "rchg" in comps:
        kw["pr_extra"] = ["rchg_unp"]
    if "geo" in comps:
        kw.update(pk_geo=3, pk_geo_lam=10.0)
    if "cells" in comps:
        kw.update(pk_cells=cells_mode, pk_cells_lam=10.0)
    if "prim" in comps:
        kw.update(px_feats=[prim], px_where="house", px_lam=10.0)
    if clip is not None or "clip" in comps:
        kw["pk_clip"] = clip
    c = PE.all_(c, **kw)
    if "field" in comps:
        c = dict(c, **FIELD_KW)
        c["pool_cfg"] = dict(c["pool_cfg"], **FIELD_KW)
        if plc is not None:
            c = dict(c, pt_rewire=plc)
            c["pool_cfg"] = dict(c["pool_cfg"], pt_rewire=plc)
    if "node" in comps:
        c["sen_cfg"] = dict(c["sen_cfg"], pt_senh=True, **({"pt_senh_shuffle": plc} if plc is not None else {}))
    if "pool" in comps:
        c["pk_pool_w"] = pool_w
    return c


def placebo(seed, comps):
    """Shuffle every added signal within year (the studies' own placebos): rchg_unp among rated races (pau-e),
    geo directions and cell shares across districts (pau-g, pau-v), prim features across districts (pau-x),
    h_S across states (pau-t pt_senh_shuffle) and a degree-preserving rewire of the state graph (pt_rewire), the
    filter's p across races (pau-w). Returns the shuffled hist; module state is shuffled in place (call restore())."""
    rng = np.random.default_rng(seed)
    h = HIST.copy()
    if "rchg" in comps:
        for y in h.year.unique():
            idx = np.where(((h.year == y) & (h.rating.abs() < 2.5)).values)[0]
            h.loc[h.index[idx], "rchg_unp"] = h.rchg_unp.values[idx[rng.permutation(len(idx))]]
    if "geo" in comps:
        Zp = Z_REAL.copy()
        yr = Zp.index.get_level_values(0)
        for y in np.unique(yr):
            m = np.where(yr == y)[0]
            Zp.iloc[m, [Zp.columns.get_loc(c) for c in GEO]] = Zp.iloc[m][GEO].values[rng.permutation(len(m))]
        PP.Z = Zp
    if "cells" in comps:
        keys = [k for k in SH_REAL if SRC.get(k) == "cd"]
        for y in sorted({k[0] for k in keys}):
            ks = [k for k in keys if k[0] == y]
            p = rng.permutation(len(ks))
            for k, j in zip(ks, p):
                SH[k] = SH_REAL[ks[j]]
    if "prim" in comps:
        for k, s in FEATS_REAL.items():
            o = s.copy(); yrs = s.index.get_level_values(0)
            for y in np.unique(yrs):
                m = yrs == y
                o[m] = s[m].values[rng.permutation(m.sum())]
            FEATS[k] = o
    if "pool" in comps:
        KF_PERM[0] = seed
    return h


def restore():
    PP.Z = Z_REAL
    SH.update(SH_REAL)
    FEATS.update(FEATS_REAL)
    KF_PERM[0] = None


def subsets(pr):
    pol = pr.poll_pollsters.fillna(0) > 0
    rated = pr.rating.abs() < 2.5
    return {"all": pr, "rated": pr[rated], "unp_rated": pr[(~pol) & rated & (pr.senate == 0)], "senate": pr[pr.senate == 1]}


def ll(pr):
    return round(float(metrics(pr.p_dside, pr.dwin)["ll"]), 5) if len(pr) else None


def run(tag, cfg, hist=None, schemes=("six", "pre18", "bt", "loyo8"), save=True):
    hist = HIST if hist is None else hist
    out, t0 = {"tag": tag}, time.time()
    for sch in schemes:
        if sch == "ins":
            Y = list(range(2010, 2026, 2))
            sp = [(T, Y) for T in Y]
        else:
            sp = E.splits(sch)
        pr = pd.concat([E.one(hist, cfg, T, tr) for T, tr in sp])
        out[sch] = {k: ll(v) for k, v in subsets(pr).items()}
        out[sch]["per_year"] = {int(y): ll(g) for y, g in pr.groupby("year")}
        if save:
            pr.to_csv(SD / f"{tag}_{sch}.csv", index=False)
    out["secs"] = round(time.time() - t0)
    json.dump(out, open(RES / f"{tag}.json", "w"), indent=1)
    print(fmt(out, schemes), flush=True)
    return out


def fmt(out, schemes=("six", "pre18", "bt", "loyo8")):
    s = " | ".join(f"{sch} {out[sch]['all']:.4f} (r {out[sch]['rated']:.3f} u {out[sch]['unp_rated']:.3f} "
                   f"s {out[sch]['senate']:.3f})" for sch in schemes if sch in out)
    return f"{out['tag']:28s} {s}  [{out.get('secs')}s]"
