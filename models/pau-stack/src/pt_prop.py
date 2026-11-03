"""pau-t step 3-5: carry this cycle's poll information over a graph (pre-election inputs only).

For every cycle (training cycles inside v5.1's inner LOYO, and the test cycle):
  gap_j  = clip(poll_j - yhat_j, -25, 25) for polled House races (yhat = stage-1 margin with the year effect);
  e_j    = gap_j - (a + X_j gamma): the residual of v6-nl's own per-cycle pattern ridge (same weights, same
           ridge), so the shared poll level and the demographic pattern are removed: e is what is left to carry.
  Senate state nodes (optional, cfg pt_sen): e_S = clip(poll_S - yhat_S) - (mean House gap level a), observing the
           MEAN of the state's district field with extra noise.
Field model (linear analogue = Gaussian Markov random field on the graph):
  u ~ N(0, tau^2 K),  K = (I + beta L)^-1 rescaled to unit diagonal (correlation form), L = D - W (graph Laplacian, W from
       pt_edges.csv, union of the graphs in cfg pt_graphs with weights cfg pt_gw); a pure state block can be added
       (cfg pt_state = tau_s^2 / tau^2); cfg pt_noid drops the graph part (pure state block).
  e_j = u_j + noise, noise var = s_e^2 / w_j, w_j = n_j / (n_j + 1.2) (pattern weights);
  posterior mean at every House race; polled races get their leave-own-out value.
  tau^2 / s_e^2 = cfg pt_snr (fixed, chosen on six / pre18), beta = cfg pt_beta.
Readout: mu_i += kappa * u_i for House races (unpolled only, or all with cfg pt_scope = "all"), kappa fitted on the
training cycles' inner-LOYO predictions (rated House races in scope; least squares, clipped to [0, 1.5]) before
v5.1's scales and calibration, as pau-r's pr_sgap. cfg pt_kappa fixes it instead.
Placebo: cfg pt_rewire = seed -> every cycle's graph is degree-preservingly rewired before use."""
import numpy as np, pandas as pd
import model as M
import pr_pattern as PP
from common import WORK

EDG = pd.read_csv(WORK / "pt_edges.csv")
_orig = M.WinModel._mu
_CACHE = {}
LOG = []


def rewire(a, b, rng):
    a, b = a.copy(), b.copy()
    m = len(a)
    S = set(zip(a, b)) | set(zip(b, a))
    for _ in range(5 * m):
        x, y = rng.integers(m, size=2)
        if x == y:
            continue
        p, q, r, s = a[x], b[x], a[y], b[y]
        if rng.random() < 0.5:
            r, s = s, r
        if len({p, q, r, s}) < 4 or (p, s) in S or (r, q) in S:
            continue
        S -= {(p, q), (q, p), (r, s), (s, r)}
        S |= {(p, s), (s, p), (r, q), (q, r)}
        a[x], b[x], a[y], b[y] = p, s, r, q
    return a, b


def kernel(ids, y, graphs, gw, beta, rw, knorm="corr"):
    key = (tuple(ids), y, tuple(graphs), tuple(gw), beta, rw, knorm)
    if key in _CACHE:
        return _CACHE[key]
    n = len(ids)
    ix = {r: k for k, r in enumerate(ids)}
    W = np.zeros((n, n))
    for g, wg in zip(graphs, gw):
        e = EDG[(EDG.year == y) & (EDG.graph == g.rstrip("x"))]
        e = e[e.i.isin(ix) & e.j.isin(ix)]
        if g.endswith("x"):
            e = e[e.i.str[:2] != e.j.str[:2]]
        a, b = e.i.map(ix).values, e.j.map(ix).values
        if rw is not None and len(a) > 4:
            a, b = rewire(a, b, np.random.default_rng(rw * 1000 + y + len(g)))
        np.add.at(W, (a, b), wg * e.w.values)
        np.add.at(W, (b, a), wg * e.w.values)
    L = np.diag(W.sum(1)) - W
    K = np.linalg.inv(np.eye(n) + beta * L)
    if knorm == "mean":                           # kernel A (first runs, prop_*): unit mean diagonal
        K = K / np.mean(np.diag(K))
    else:                                         # kernel B (propc_*): correlation form, equal prior variance
        dk = np.sqrt(np.diag(K))
        K = K / np.outer(dk, dk)
    _CACHE[key] = K
    return K


def field(df, cfg):
    """Posterior mean of the graph field at every House row of df (0 elsewhere), leave-own-out at polled rows."""
    out = np.zeros(len(df))
    cols = PP.SETS[cfg.get("pr_pat_set", "nl")]
    lam = cfg.get("pr_pat_lam", 1.0) or 1.0
    idx = pd.MultiIndex.from_arrays([df.year.values, df.race_id.values])
    X = PP.Z.reindex(idx)[cols].fillna(0.0).values
    house = df.senate.values == 0
    pm = df.poll_margin.values.astype(float)
    pol = (df.poll_pollsters.fillna(0).values > 0) & np.isfinite(pm)
    yv = df.year.values
    for y in np.unique(yv):
        m = house & (yv == y)
        o = m & pol
        if o.sum() < 10:
            continue
        g = np.clip(pm[o] - df.yhat.values[o], -25, 25)
        n = df.poll_pollsters.values[o].astype(float)
        w = n / (n + 1.2)
        A = np.column_stack([np.ones(o.sum()), X[o]])
        P = np.diag(np.r_[0.0, np.full(len(cols), lam)])
        b = np.linalg.solve(A.T @ (A * w[:, None]) + P, A.T @ (w * g))
        e = g - A @ b
        hid = np.where(m)[0]
        ids = df.race_id.values[hid]
        K = kernel(ids, int(y), cfg["pt_graphs"], cfg.get("pt_gw", [1.0] * len(cfg["pt_graphs"])), cfg.get("pt_beta", 1.0),
                   cfg.get("pt_rewire"), cfg.get("pt_knorm", "corr"))
        if cfg.get("pt_noid"):
            K = np.zeros_like(K)
        if cfg.get("pt_state"):
            st = np.array([r[:2] for r in ids])
            if cfg.get("pt_state_perm") is not None:      # placebo: state labels permuted (block sizes kept)
                st = np.random.default_rng(cfg["pt_state_perm"] * 100 + int(y)).permutation(st)
            K = K + cfg["pt_state"] * (st[:, None] == st[None, :])
        snr = cfg.get("pt_snr", 0.3)
        oi = np.where(pol[hid])[0]
        Hn = np.diag(1.0 / w)                       # noise / s_e^2
        obs_K = snr * K[np.ix_(oi, oi)] + Hn
        ev = e.copy()
        rows_K = snr * K[:, oi]
        if cfg.get("pt_sen"):
            # Senate race(s) of the cycle as state nodes: observe the mean of the state's district field
            sm = (yv == y) & (df.senate.values == 1) & pol
            st = np.array([r[:2] for r in ids])
            add_rows, add_e = [], []
            for k in np.where(sm)[0]:
                hs = st == df.race_id.values[k][:2]
                if hs.sum() == 0:
                    continue
                h = hs / hs.sum()
                add_rows.append(h)
                add_e.append(np.clip(pm[k] - df.yhat.values[k], -25, 25) - b[0])
            if add_rows:
                Hs = np.array(add_rows)
                c_so = snr * Hs @ K[:, oi]
                c_ss = snr * Hs @ K @ Hs.T + np.eye(len(add_rows)) * cfg["pt_sen"]
                obs_K = np.block([[obs_K, c_so.T], [c_so, c_ss]])
                rows_K = np.column_stack([rows_K, snr * K @ Hs.T])
                ev = np.r_[ev, add_e]
        Ki = np.linalg.inv(obs_K)
        alpha = Ki @ ev
        u = rows_K @ alpha
        # leave-own-out at the polled House rows: e_i - alpha_i / Ki_ii, minus the noise part
        loo_obs = ev[:len(oi)] - alpha[:len(oi)] / np.diag(Ki)[:len(oi)]
        u[oi] = loo_obs
        out[hid] = u
        LOG.append(dict(year=int(y), n_obs=len(ev), sd_u=float(np.std(u)), sd_e=float(np.std(e))))
    return out


def _fit_k(u, r, lo, hi):
    return float(np.clip((u * r).sum() / ((u * u).sum() + 1e-9), lo, hi)) if len(u) >= 10 else 0.0


def _mu(self, df):
    mu = _orig(self, df)
    c = self.cfg
    if not c.get("pt_graphs"):
        return mu
    u = field(df, c)
    pol = (df.poll_pollsters.fillna(0).values > 0) & np.isfinite(df.poll_margin.values.astype(float))
    house = df.senate.values == 0
    split = c.get("pt_scope") == "split"       # separate readouts: unpolled (opinion carry) / polled (sign free)
    scope = house & ((~pol) | (c.get("pt_scope") in ("all", "split")))
    if not hasattr(self, "kap_"):
        rated = np.abs(df.rating.values) < 2.5
        if c.get("pt_kappa") is not None:
            self.kap_ = (float(c["pt_kappa"]),) * 2
        elif "y" in df:
            r = np.clip(df.y.values, -60, 60) - mu
            if split:
                mu_ = scope & rated & ~pol
                mp_ = scope & rated & pol
                self.kap_ = (_fit_k(u[mu_], r[mu_], 0, 1.5), _fit_k(u[mp_], r[mp_], -1.5, 1.5))
            else:
                m = scope & rated
                k = _fit_k(u[m], r[m], 0, 1.5)
                self.kap_ = (k, k)
        else:
            self.kap_ = (0.0, 0.0)
        LOG.append(dict(kappa=self.kap_[0], kappa_pol=self.kap_[1]))
    k = np.where(pol, self.kap_[1], self.kap_[0])
    return mu + np.where(scope, k * u, 0.0)


M.WinModel._mu = _mu
