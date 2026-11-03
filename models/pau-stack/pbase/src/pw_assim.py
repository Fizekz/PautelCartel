"""pau-w: weather-style data assimilation of the opinion space through each campaign (exact Kalman filter core).

State of a cycle (margin points, D-side minus R):
  z = [n, v, b, g_1..g_K, s_1..s_S, l_1..l_R]
  n   national year effect (v5.1's year-effect units)              random walk, daily SD q_n
  v   generic-ballot bias (GB-implied year effect minus n)          constant in the cycle (bias state, VarBC-style)
  b   shared race-poll bias (poll minus outcome, D-favourable)      constant (bias state; prior from training cycles)
  g   district pattern (college, Hispanic, Black [+ nl products])   random walk, daily SD q_g
  s   state effects (a state's House and Senate races load 1)       AR(1), time constant tau
  l   race-local opinion relative to the background                 AR(1) toward 0 (= drift back to fundamentals)
Background (first guess) per race: rho_r = v5.1 stage-1 race part (fundamentals + final expert ratings, no year
effect), from a MarginModel fitted on the training cycles only.
Observation operators:
  GB poll (midterms): a + beta * gb_i = n + v + e          (a, beta: v5.1's year mapping fe = a + beta gb mid)
  race poll i of r:   m_i - rho_r = n + b + F_r g + s_j + l_r + e_i,  Var e_i = 1e4 deff / n_i + k_ir sigma_h^2
                      (k_ir = number of that pollster's polls in the race: repeated polls share a house effect)
  Kalshi (2026 only): 7 Phi^-1(p_D) - rho_r = n + F_r g + s_j + l_r + c_k + e   (own bias state c_k)
Outcome (verification): y_r - rho_r = n + F_r g + s_j + l_r + eps_r.
Quality control: an observation whose innovation exceeds QC sigmas of its predicted spread is rejected
(the background check of operational DA).
Everything a cycle's run sees is dated before its election day; the test cycle's outcomes never enter.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import model as M
import polls_v4 as P4
from common import WORK, ROOT, ELECTION_DATES
from model import FINAL

H = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
_F = pd.read_csv(ROOT.parent / "race" / "features.csv")
FEAT3 = ["college", "hispanic", "black"]
_Z = _F[["year", "race_id"] + FEAT3].copy()
for c in FEAT3:
    _Z[c] = (_Z[c] - _F[c].mean()) / _F[c].std()
for a, c in (("college", "college"), ("hispanic", "hispanic"), ("black", "black"), ("college", "hispanic")):
    v = _Z[a] * _Z[c]
    _Z[f"{a}_x_{c}"] = (v - v.mean()) / v.std()
NL = FEAT3 + ["college_x_college", "hispanic_x_hispanic", "black_x_black", "college_x_hispanic"]
_Z = _Z.fillna(0.0).set_index(["year", "race_id"])
LEADS = [120, 90, 60, 45, 30, 21, 14, 7, 3, 0]


@lru_cache(None)
def poll_rows():
    indep = set(zip(H.year[H.indep_dside == 1], H.race_id[H.indep_dside == 1]))
    r = P4.hist_rows(indep)
    r26 = pd.read_csv(WORK / "polls_v4_rows_2026.csv")
    r = pd.concat([r, r26[[c for c in r.columns if c in r26.columns]]], ignore_index=True)
    r["m_adj"] = r.margin - r["shift"] * r.part
    r["nn"] = r.n.fillna(500).clip(100, 1500)
    r = r[(r.days >= 0) & (r.days <= 400)].copy()
    r["k"] = r.groupby(["year", "race_id", "pollster"]).margin.transform("size")
    return r


@lru_cache(None)
def gb_rows():
    o = pd.read_csv(WORK / "pr_latent_obs.csv")
    g = o[(o.ind == "gb") & (o.days >= 0)].copy()
    g["k"] = g.groupby(["cycle", "pollster"]).value.transform("size")
    return g.rename(columns={"cycle": "year"})


def frame(year):
    if year == 2026:
        f = pd.read_csv(WORK / "features_2026.csv", low_memory=False)
        f = f[f.kind.isin(["D_vs_R", "R_vs_IND", "jungle_party"])].copy()
        f["year"] = 2026
        f["train_ok"] = 1
        if "y" not in f:
            f["y"], f["dwin"] = np.nan, np.nan
        return f.reset_index(drop=True)
    return H[(H.year == year) & (H.train_ok == 1)].reset_index(drop=True)


@lru_cache(None)
def background(year, fit_years):
    """rho (race part, no year effect) for `year` from MarginModels fitted on fit_years; also a, beta."""
    tr = H[(H.train_ok == 1) & H.year.isin(fit_years)]
    mh = M.MarginModel(FINAL).fit(tr)
    ms = M.MarginModel(FINAL["sen_cfg"]).fit(tr)
    f = frame(year)
    a, beta = mh.yl_[0][1]
    ye_h = float(mh.year_effect(f).iloc[0])
    ph, ps = mh.predict(f), ms.predict(f)
    sen = f.senate.values == 1
    rho = np.where(sen, ps, ph) - ye_h
    out = f[["year", "race_id", "state", "senate", "rating", "y", "dwin"]].copy()
    out["rho"] = rho
    out["ye"] = ye_h
    out["a"], out["beta"] = a, beta
    out["mid"] = int(year % 4 == 2)
    return out


def _wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


@lru_cache(None)
def params(fit_years, b_mean=False):
    """Moment estimates of the error variances from the fit cycles, each with an honest background
    (MarginModel fitted on the other fit cycles)."""
    P = poll_rows()
    rows, nat, bias, hres, eps = [], [], [], [], []
    for f in fit_years:
        others = tuple(y for y in fit_years if y != f)
        bg = background(f, others).copy()
        bg["e"] = np.clip(bg.y - bg.rho, -60, 60)
        rated = bg.rating.abs() < 2.5
        w = np.where(rated, 8.0, 1.0)
        n_f = _wmean(bg.e.values, w)
        base = bg.a.iloc[0] + bg.beta.iloc[0] * H[H.year == f].gb.iloc[0] * bg.mid.iloc[0]
        nat.append(n_f - base)
        bg["e2"] = bg.e - n_f
        rows.append(bg)
        p = P[(P.year == f) & (P.days <= 30)].merge(bg[["race_id", "y", "rho"]], on="race_id")
        if len(p):
            vs = 1e4 / p.nn.values
            bias.append(_wmean(np.clip(p.m_adj - p.y, -25, 25).values, 1 / (vs + 9.0 * p.k.values)))
            g = p.groupby("race_id")
            mm = g.m_adj.transform("mean")
            cnt = g.m_adj.transform("size")
            m3 = cnt >= 3
            hres.append(((p.m_adj - mm)[m3] ** 2 * cnt[m3] / (cnt[m3] - 1) - vs[m3]).values)
            avg = g.apply(lambda d: pd.Series(dict(m=np.average(d.m_adj, weights=1 / (1e4 / d.nn + 9)),
                                                   v=1 / np.sum(1 / (1e4 / d.nn + 9 * d.k)))))
            avg = avg.merge(bg[["race_id", "rating", "senate", "e2"]], left_index=True, right_on="race_id")
            avg = avg[avg.rating.abs() < 2.5]
            rh = dict(zip(bg.race_id, bg.rho))
            avg["pg"] = avg.m - avg.race_id.map(rh) - n_f - bias[-1]
            eps.append(avg[["senate", "pg", "e2", "v"]])
    R = pd.concat(rows)
    rr = R[R.rating.abs() < 2.5]
    # state covariance: mean cross-product of rated residuals in the same state and cycle
    num = den = 0.0
    for (y, s), g in rr.groupby(["year", "state"]):
        e = g.e2.values
        if len(e) > 1:
            num += (e.sum() ** 2 - (e ** 2).sum())
            den += len(e) * (len(e) - 1)
    sig_s2 = max(num / den, 0.5) if den else 4.0

    def rvar(e):
        e = np.asarray(e)
        mad = 1.4826 * np.median(np.abs(e - np.median(e)))
        return 0.5 * mad ** 2 + 0.5 * min(e.std(), 3 * mad) ** 2

    # poll-visible local variance: covariance of the poll gap and the outcome gap in polled rated races
    EP = pd.concat(eps)
    EP = EP[(EP.pg.abs() < 40) & (EP.e2.abs() < 40)]
    cov = float(np.mean((EP.pg - EP.pg.mean()) * (EP.e2 - EP.e2.mean())))
    sig_eps2 = float(max(rvar(EP.e2.values) - cov, 2.0))
    loc, tot = {}, {}
    for sen in (0, 1):
        for rated in (True, False):
            g = R[(R.senate == sen) & ((R.rating.abs() < 2.5) == rated)]
            tot[(sen, rated)] = max(rvar(g.e2.values) - sig_s2, 6.0)
            loc[(sen, rated)] = max(tot[(sen, rated)] - sig_eps2, 2.0)
    b = np.array(bias)
    return dict(sig_v=float(np.sqrt(np.mean(np.square(nat)))), nat=list(np.round(nat, 2)),
                mu_b=float(b.mean()) if b_mean else 0.0, sig_b=float(max(b.std(ddof=1), 2.0)), b=list(np.round(b, 2)),
                sig_h2=float(max(np.mean(np.concatenate(hres)), 1.0)), sig_eps2=sig_eps2, cov=cov,
                sig_s2=float(sig_s2), loc=loc, tot=tot)


DEF = dict(K="nl", tau=45.0, q_n=0.1, q_g=0.02, sig_g=3.0, qc=4.0, D0=120, deff=1.0, use_gb=True,
           pattern="all", state=True, b_scale=1.0, kalshi=None, sig_k=4.0, sig_ck=2.0)


class Cycle:
    """Index layout and observation lists of one cycle."""

    def __init__(self, bg, opts):
        self.bg = bg.reset_index(drop=True)
        cols = NL if opts["K"] == "nl" else FEAT3
        idx = pd.MultiIndex.from_arrays([self.bg.year.values, self.bg.race_id.values])
        Fm = _Z.reindex(idx)[cols].fillna(0.0).values
        Fm[self.bg.senate.values == 1] = 0.0
        if opts["pattern"] == "none":
            Fm[:] = 0.0
        self.F = Fm
        self.K = len(cols)
        self.states = sorted(self.bg.state.unique())
        self.si = {s: i for i, s in enumerate(self.states)}
        self.ri = {r: i for i, r in enumerate(self.bg.race_id)}
        self.o_n, self.o_v, self.o_b = 0, 1, 2
        self.o_g = 3
        self.o_c = 3 + self.K          # Kalshi bias
        self.o_s = self.o_c + 1
        self.o_l = self.o_s + len(self.states)
        self.D = self.o_l + len(self.bg)
        self.R = len(self.bg)

    def h_race(self, r, poll=True, kalshi=False):
        """sparse observation operator for race index r: (indices, coefficients)."""
        j = self.si[self.bg.state.iloc[r]]
        idx = [self.o_n] + list(range(self.o_g, self.o_g + self.K)) + [self.o_s + j, self.o_l + r]
        cf = [1.0] + list(self.F[r]) + [1.0, 1.0]
        if poll:
            idx.append(self.o_b); cf.append(1.0)
        if kalshi:
            idx.append(self.o_c); cf.append(1.0)
        return np.array(idx), np.array(cf)

    def Hy(self):
        Hm = np.zeros((self.R, self.D))
        Hm[:, self.o_n] = 1.0
        Hm[:, self.o_g:self.o_g + self.K] = self.F
        for r in range(self.R):
            Hm[r, self.o_s + self.si[self.bg.state.iloc[r]]] = 1.0
            Hm[r, self.o_l + r] = 1.0
        return Hm


def gb_on(bg0, opts):
    """GB polls are assimilated in midterms when the fold's year mapping identifies a GB slope."""
    return bool(opts["use_gb"] and bg0.mid == 1 and bg0.beta > 0.1)


def obs_list(cy, year, prm, opts, max_day=None):
    """Observations as (day, kind, race index or -1, value, variance), sorted from early to late."""
    out = []
    P = poll_rows()
    p = P[(P.year == year) & (P.days <= opts["D0"])]
    rho = dict(zip(cy.bg.race_id, cy.bg.rho))
    for r in p.itertuples():
        if r.race_id not in cy.ri:
            continue
        var = opts["deff"] * 1e4 / r.nn + r.k * prm["sig_h2"]
        out.append((int(r.days), "poll", cy.ri[r.race_id], float(r.m_adj - rho[r.race_id]), float(var)))
    bg0 = cy.bg.iloc[0]
    if gb_on(bg0, opts):
        g = gb_rows()
        g = g[(g.year == year) & (g.days <= opts["D0"])]
        beta = bg0.beta
        for r in g.itertuples():
            nn = 1000.0 if not np.isfinite(r.n) else min(max(r.n, 300), 3000)
            var = beta ** 2 * (1e4 / nn + r.k * 2.0 ** 2)
            out.append((int(r.days), "gb", -1, float(bg0.a + beta * r.value), float(var)))
    if opts.get("kalshi") is not None:
        for rid, day, val in opts["kalshi"]:
            if rid in cy.ri:
                out.append((int(day), "kalshi", cy.ri[rid], float(val - rho[rid]), opts["sig_k"] ** 2))
    out.sort(key=lambda x: -x[0])
    return out


def prior(cy, prm, opts):
    D = cy.D
    x = np.zeros(D)
    P = np.zeros((D, D))
    bg0 = cy.bg.iloc[0]
    x[cy.o_n] = bg0.a
    P[cy.o_n, cy.o_n] = (8.0 ** 2) if gb_on(bg0, opts) else prm["sig_v"] ** 2
    P[cy.o_v, cy.o_v] = prm["sig_v"] ** 2
    x[cy.o_b] = prm["mu_b"]
    P[cy.o_b, cy.o_b] = (opts["b_scale"] * prm["sig_b"]) ** 2
    for k in range(cy.K):
        P[cy.o_g + k, cy.o_g + k] = opts["sig_g"] ** 2
    P[cy.o_c, cy.o_c] = opts["sig_ck"] ** 2
    ss = prm["sig_s2"] if opts["state"] else 1e-6
    for j in range(len(cy.states)):
        P[cy.o_s + j, cy.o_s + j] = ss
    lv = np.array([prm["loc"][(int(s), abs(r) < 2.5)] for s, r in zip(cy.bg.senate, cy.bg.rating)])
    P[cy.o_l:, cy.o_l:][np.diag_indices(cy.R)] = lv
    return x, P, lv, ss


def dyn(cy, opts, lv, ss, t):
    """t-day propagation: diagonal A and Q (AR(1) for s and l, random walks for n and g)."""
    tau = opts["tau"]
    phi = 1.0 if not np.isfinite(tau) else np.exp(-t / tau)
    a = np.ones(cy.D)
    q = np.zeros(cy.D)
    q[cy.o_n] = t * opts["q_n"] ** 2
    q[cy.o_g:cy.o_g + cy.K] = t * opts["q_g"] ** 2
    a[cy.o_s:cy.o_l] = phi
    q[cy.o_s:cy.o_l] = ss * (1 - phi ** 2)
    a[cy.o_l:] = phi
    q[cy.o_l:] = lv * (1 - phi ** 2)
    return a, q


def run(year, fit_years, opts=None, prm=None, leads=LEADS, bg=None, return_state=False):
    """Exact Kalman filter over the campaign. Returns per-lead day-0 forecasts (mean, var) for every race."""
    opts = dict(DEF, **(opts or {}))
    prm = prm or params(tuple(fit_years))
    bg = background(year, tuple(fit_years)) if bg is None else bg
    cy = Cycle(bg, opts)
    x, P, lv, ss = prior(cy, prm, opts)
    obs = obs_list(cy, year, prm, opts)
    Hy = cy.Hy()
    se2 = prm["sig_eps2"]
    res, diag = {}, dict(n_obs=0, n_rej=0, chi2=[], chi2_poll=[])
    day = opts["D0"]
    oi = 0
    todo = sorted(set(leads), reverse=True)
    for t in range(opts["D0"], -1, -1):
        if t < day:
            a, q = dyn(cy, opts, lv, ss, day - t)
            x = a * x
            P = P * np.outer(a, a)
            P[np.diag_indices(cy.D)] += q
            day = t
        while oi < len(obs) and obs[oi][0] == t:
            _, kind, r, val, var = obs[oi]
            oi += 1
            if kind == "gb":
                idx, cf = np.array([cy.o_n, cy.o_v]), np.array([1.0, 1.0])
            else:
                idx, cf = cy.h_race(r, poll=(kind == "poll"), kalshi=(kind == "kalshi"))
            Ph = P[:, idx] @ cf
            S = float(cf @ Ph[idx]) + var
            d = val - float(cf @ x[idx])
            if S <= 1e-9 or abs(d) > opts["qc"] * np.sqrt(S):
                diag["n_rej"] += 1
                continue
            Kg = Ph / S
            x = x + Kg * d
            P = P - np.outer(Kg, Ph)
            diag["n_obs"] += 1
            diag["chi2"].append(d * d / S)
            if kind == "poll":
                diag["chi2_poll"].append(d * d / S)
        if t in todo:
            a, q = dyn(cy, opts, lv, ss, t)
            x0 = a * x
            P0 = P * np.outer(a, a)
            P0[np.diag_indices(cy.D)] += q
            m = cy.bg.rho.values + Hy @ x0
            v = np.einsum("ij,ij->i", Hy @ P0, Hy) + se2
            res[t] = (m, v)
    polled = np.zeros(cy.R, bool)
    for o in obs:
        if o[1] == "poll":
            polled[o[2]] = True
    diag.update(n=x[cy.o_n], v=x[cy.o_v], b=x[cy.o_b], sd_b=np.sqrt(P[cy.o_b, cy.o_b]), sd_n=np.sqrt(P[cy.o_n, cy.o_n]),
                g=x[cy.o_g:cy.o_g + cy.K].copy(), c_k=x[cy.o_c])
    if return_state:
        diag.update(x=x, P=P, cy=cy)
    return cy.bg.assign(polled=polled), res, diag


NU, EPS = 10, 0.001


def grp(df):
    return np.where(df.senate.values == 1, "S", "H").astype(object) + np.where(df.polled.values, "P", "N").astype(object)


def p_of(m, v, g, scale):
    c = np.array([scale.get(k, 1.0) for k in g])
    sig = c * np.sqrt(v) * np.sqrt((NU - 2) / NU)
    return (1 - EPS) * student_t.cdf(m / sig, NU) + EPS / 2


CAL = dict(mode="comp")


def fit_scale(frames):
    """ML scale multiplier per group (HP, HN, SP, SN) on the training cycles' day-0 outputs
    (mode "comp": only races with |mean| < 15 inform the fit, as v5.1's robust part; "all": every race)."""
    from scipy.optimize import minimize_scalar
    d = pd.concat(frames)
    if CAL["mode"] == "comp":
        d = d[np.abs(d.m) < 15]
    g = grp(d)
    out = {}
    for k in ("HP", "HN", "SP", "SN"):
        mk = g == k
        if mk.sum() < 20:
            mk = np.char.startswith(g.astype(str), k[0])
        m, v, y = d.m.values[mk], d.v.values[mk], d.dwin.values[mk]

        def nll(ls):
            p = np.clip(p_of(m, v, [k] * len(m), {k: np.exp(ls)}), 1e-6, 1 - 1e-6)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        out[k] = float(np.exp(minimize_scalar(nll, bounds=(np.log(0.4), np.log(3.0)), method="bounded").x))
    return out


def b_prior(prm, tr, T, opts, drop=None):
    """Prior mean of the shared poll bias: 0 ("zero"), the training mean ("mean"), or the mean over training cycles
    of the same type as T ("type": midterm vs presidential, the campaign-level analog). `drop` leaves a training
    cycle's own bias out of its own prior."""
    mode = (opts or {}).get("b_prior", "zero")
    if mode == "zero":
        return prm
    bs = [(y, b) for y, b in zip(tr, prm["b"]) if y != drop]
    if mode == "type":
        bs = [(y, b) for y, b in bs if (y % 4 == 2) == (T % 4 == 2)] or bs
    return dict(prm, mu_b=float(np.mean([b for _, b in bs])))


def fold(T, tr, opts=None, leads=LEADS, prm_kw=None):
    """Held-out forecasts of cycle T at every lead; calibration scales from the training cycles' day-0 runs
    (each training cycle with a background fitted without it; the variance parameters use all of tr)."""
    tr = tuple(tr)
    prm = b_prior(params(tr, **(prm_kw or {})), tr, T, opts)
    cal = []
    for f in tr:
        others = tuple(y for y in tr if y != f)
        b, res, _ = run(f, others, opts, prm=b_prior(prm, tr, f, opts, drop=f), leads=[0], bg=background(f, others))
        cal.append(b.assign(m=res[0][0], v=res[0][1]))
    scale = fit_scale(cal)
    cb = fit_cal2(cal, scale) if (opts or {}).get("cal2") else None
    b, res, diag = run(T, tr, opts, prm=prm, leads=leads)
    out = []
    for t, (m, v) in res.items():
        p = p_of(m, v, grp(b), scale)
        if cb is not None:
            p = apply_cal2(p, b.rating.values, cb)
        out.append(b.assign(lead=t, m=m, v=v, p=p, sc=[scale.get(k, 1.0) for k in grp(b)],
                            chi2_poll=float(np.mean(diag["chi2_poll"])) if diag["chi2_poll"] else np.nan))
    return pd.concat(out), diag, scale, prm


def _c2X(p, rating):
    z = np.clip(np.log(p / (1 - p)), -7, 7)
    return np.column_stack([z, np.clip(rating.astype(float), -2.5, 2.5)])


def fit_cal2(cal, scale):
    """v5.1's calibration layer (pau-a's form) on the training cycles' outputs: logit p' = b1 logit p + b2 rating,
    ridge 1 toward (1, 0), mixed 50/50 with the raw logit."""
    from scipy.optimize import minimize
    d = pd.concat(cal)
    p = np.clip(p_of(d.m.values, d.v.values, grp(d), scale), 1e-4, 1 - 1e-4)
    X = _c2X(p, d.rating.values)
    y = d.dwin.values.astype(float)
    b0 = np.array([1.0, 0.0])
    f = lambda b: np.sum(np.logaddexp(0, X @ b) - y * (X @ b)) + np.sum((b - b0) ** 2)
    return minimize(f, b0, method="BFGS").x


def apply_cal2(p, rating, cb):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    eta = 0.5 * (_c2X(p, rating) @ cb) + 0.5 * np.log(p / (1 - p))
    return 1 / (1 + np.exp(-eta))
