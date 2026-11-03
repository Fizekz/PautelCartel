"""pau-b win-probability model: margin regression + poll shrinkage + honest calibration.

Stage 1 (fundamentals + experts): robust (Huber) regression of the D-side two-party margin on
  lean, incumbency, expert rating (piecewise), generic ballot, money, previous result, chamber,
  plus optional terms (cfg["extra"]); optional extra weight on rated-competitive races.
Stage 2 (polls): mu = yhat + k (poll - yhat), k = n / (n + lambda_chamber), n = number of
  pollsters in the final-3-week average; lambda fitted on leave-one-year-out (LOYO) residuals.
Stage 3 (uncertainty): residual SD by group (chamber x competitive/safe x polled), estimated on
  LOYO residuals, so it contains each held-out year's national (correlated) error.
Stage 4 (calibration): P(D-side wins) = F(a * mu / sd), F = normal or Student-t CDF, a fitted by
  maximum likelihood on the LOYO predictions of the training years.

Every stage only sees the training years passed in, so a backtest for year T uses years < T.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, t as student_t
from sklearn.linear_model import HuberRegressor, Ridge

V1 = dict(extra=[], comp_weight=1.0, link="normal", nu=None, epsilon=1.5, rating_src="mean")
# v2 (chosen on held-out years 2010-2024, see LOG.md): race-level coefficients within years,
# year effect = a + b * generic ballot * midterm, rated-competitive races weighted 3x, Huber eps 2
V2 = dict(V1, two_level=True, year_vars=[["gbmid"]], comp_weight=3.0, epsilon=2.0)
# v3 (round 2): pau-a's stage-A design (piecewise rating, PVI x Safe, Senate x rating) fitted by WLS
# with Safe seats at 1/4 weight, my two-level year effects (year FE + GB x midterm), no money (it
# carried no held-out skill and leaks post-election receipts), poll-weight ceiling, Student-t(10)
# calibration per chamber x polled group (both borrowed from pau-a). See LOG.md.
V3 = dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t",
          two_level=True, year_vars=[["gbmid"]], no_money=True)
# v4 (round 3): chosen on held-out years that are not the backtest years (six-year LOYO over 2010-2016, 2020, 2024
# and pre-2018 LOYO; src/evaluate4.py, LOG.md). Linear rating; Student-t(10) as in v3. House: no PVI x Safe, rated races
# weight 8, kavi-e-style polls (120 days, half-life 14 d, Kish n). Senate: v3's design (PVI x Safe, rated weight 4,
# 21-day pollster average). Each chamber's predictions come from its own config (stage 1 still fitted on both).
V4_SENATE = dict(V3, lin_rating=True, nu=10, polls="v4", v4_house_only=True, last_all=True,
                 last_miss_mode="house_sen")
V4 = dict(V3, lin_rating=True, nu=10, polls="v4", v4_house_only=True, pa_no_pvisafe=True, comp_weight=8.0,
             last_all=True, last_miss_mode="house_sen", sen_cfg=V4_SENATE)
# v5 (round 4, after within-side disclosure 2; chosen on 2010-2016/2020/2024 only, src/evaluate4.py, LOG.md):
#  - calibration layer on both chambers (pau-a v4's form: logit p' = b1 logit p + b2 clip(rating), no intercept,
#    fitted on the nested LOYO predictions, mixed 50/50 with the raw logit)
#  - Senate: Safe-seat lean term only where the lean agrees with the rating (pau-a v4), lean trend in all races
#  - House: 50/50 logit pool with a simple uniform-swing component (kavi-e / pau-a design + my previous-result terms)
CAL5 = dict(cal2="sr0", cal2_w=0.5)
# the simple component uses v3's 21-day pollster average (different polls from the main component: better on both
# selection schemes than giving both components the 120-day average)
SIMPLE5 = dict(V1, design="simple", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t", nu=10,
               two_level=False, no_money=True, last_all=True, **CAL5)
V5_SENATE = dict(V4_SENATE, safe_fix=True, ltrend="all", **CAL5)
V5 = dict(V4, sen_cfg=V5_SENATE, pool_cfg=SIMPLE5, pool_w=0.5, **CAL5)
# v5.1: the previous-House-result terms are removed from every component. The missing-history dummy acted as a year
# indicator (2010/2012/2022 have no usable previous result); demeaned within years (_last_terms_demeaned) it no
# longer beats the version without it on both selection schemes (six 0.0707 vs 0.0712, pre18 0.0721 vs 0.0719).
# Parked; see LOG.md.
FINAL = dict(V5, last_all=False, sen_cfg=dict(V5_SENATE, last_all=False), pool_cfg=dict(SIMPLE5, last_all=False))


def _rating(df, cfg):
    if cfg.get("rating_src") == "cook":
        return df.r_cook.values.astype(float)
    return df.rating.values.astype(float)


def _last_terms_demeaned(df, sen):
    """v5.1: previous-result terms demeaned within each year over House rows, so they cannot shift a cycle's level.
    last_rel_c = has x (last_rel - mean of last_rel over the year's House seats with history);
    last_miss_c = miss - share of the year's House seats without history. Senate rows get 0 for both.
    In a year where no seat has history (2010, 2012, 2022) both are exactly 0."""
    yrs = df.year.values
    house = (1 - sen).astype(bool)
    has = (1 - df.last_miss.values).astype(float) * house
    miss = df.last_miss.values.astype(float) * house
    lr = df.last_rel.values.astype(float)
    t = pd.DataFrame({"y": yrs, "h": house, "has": has, "miss": miss, "lr": lr * has})
    g = t[t.h].groupby("y")
    share_miss = t.y.map(g.miss.mean()).fillna(0).values
    mean_lr = t.y.map(g.lr.sum() / g.has.sum().replace(0, np.nan)).fillna(0).values
    return has * (lr - mean_lr), (miss - share_miss) * house


def design(df, cfg=V1):
    r = _rating(df, cfg)
    sen = df.senate.values.astype(float)
    X = {
        "lean": df.lean.values,
        "lean_sen": df.lean.values * sen,
        "inc": df.inc.values,
        "inc_sen": df.inc.values * sen,
        "rating_c": np.clip(r, -2, 2),
        "rating_c_sen": np.clip(r, -2, 2) * sen,
        "rating_tail": np.sign(r) * np.clip(np.abs(r) - 2, 0, 1),
        "rimp_sign": df.rimp.values * np.sign(r),
        "gb": df.gb.values,
        "money": df.money.values,
        "money_sen": df.money.values * sen,
        "last_rel": df.last_rel.values,
        "last_miss_lean": df.last_miss.values * df.lean.values,
        "senate": sen,
        "indep": df.indep_dside.values,
    }
    comp = (np.abs(r) < 2.5).astype(float)
    if cfg.get("design") == "simple":
        # v5: kavi-e / pau-a "simple" uniform-swing component: one linear rating slope, PVI, incumbency, Senate
        # interactions, generic ballot x midterm in every race (one-level fit), lean trend
        ln = df.plean.values / 2.0 if cfg.get("simple_plean") else df.lean.values
        X = {"r": r, "pvi": ln, "pvi_sen": ln * sen, "r_sen": r * sen,
             "inc": df.inc.values, "gb_mid": df.gb.values * df.midterm.values, "indep": df.indep_dside.values,
             "senate": sen}
        if cfg.get("ltrend"):
            X["ltrend"] = df.ltrend.values * comp
        for c in cfg.get("pr_extra", []):
            X[c] = df[c].fillna(0).values if c in df else 0.0
        if cfg.get("last_all"):
            if cfg.get("last_miss_mode") in ("demean", "demean_nomiss"):
                a, b = _last_terms_demeaned(df, sen)
                X["last_rel"] = a
                if cfg["last_miss_mode"] == "demean":
                    X["last_miss"] = b
            else:
                X["last_rel"] = df.last_rel.values
                X["last_miss"] = df.last_miss.values * (1 - sen)
        return pd.DataFrame(X)
    if cfg.get("design") == "pa":
        # pau-a's stage-A design (borrowed): piecewise rating, PVI with a steeper slope in Safe seats,
        # incumbency, money and Senate x rating only outside Safe seats; no previous result
        safe = 1 - comp
        X = {
            "r_mid": np.clip(r, -2, 2), "r_hi": np.maximum(r - 2, 0), "r_lo": np.minimum(r + 2, 0),
            "pvi": df.lean.values, "pvi_safe": df.lean.values * safe, "inc": df.inc.values,
            "money": df.money_pa.values * comp, "sen_r": sen * np.clip(r, -2, 2),
            "gb": df.gb.values, "indep": df.indep_dside.values,
        }
        if not cfg.get("two_level"):
            X["gb_mid"] = df.gb.values * df.midterm.values * comp
            del X["gb"]
        if cfg.get("no_money"):
            del X["money"]
        if cfg.get("money_cap") is not None:
            X["money"] = np.clip(df.money_pa.values, -cfg["money_cap"], cfg["money_cap"]) * comp
        if cfg.get("money_inc_only"):     # money only where an incumbent runs (their money is mostly pre-election)
            X["money"] = df.money_pa.values * comp * (df.inc.values != 0)
        if cfg.get("pa_no_pvisafe"):
            del X["pvi_safe"]
        if cfg.get("pvisafe_sen"):     # v4: PVI x Safe kept for the Senate only
            X["pvi_safe"] = X["pvi_safe"] * sen
        if cfg.get("prim"):            # v4: CA/WA top-two primary margin relative to 2 x PVI
            X["prim_dev"] = df.prim_dev.values
            X["has_prim"] = df.has_prim.values
        if cfg.get("prim_comp"):
            X["prim_dev"] = df.prim_dev.values * comp
            X["has_prim"] = df.has_prim.values
        if cfg.get("ltrend"):          # v5: presidential-lean trend (pau-a v4), non-Safe races
            X["ltrend"] = df.ltrend.values * (comp if cfg["ltrend"] == "comp" else 1.0)
        if cfg.get("safe_fix") and "pvi_safe" in X:   # v5: Safe-seat lean term only where lean agrees with rating
            sg = np.sign(r)
            X["pvi_safe"] = safe * sg * np.maximum(sg * df.lean.values, 0)
        if cfg.get("r_unpolled"):      # v5 research Q3: separate rating slope where no poll exists (rated races)
            pol = df["pn4"].notna().values if cfg.get("polls") == "v4" and "pn4" in df else df.poll_pollsters.fillna(0).values > 0
            X["r_unpolled"] = np.clip(r, -2.5, 2.5) * comp * (~pol if pol.dtype == bool else (1 - pol))
        if cfg.get("r_mid_x"):         # rating slope specific to midterms
            X["r_midterm"] = X["r_mid"] * df.midterm.values
        if cfg.get("inc_pres_mid"):    # president's-party incumbent in a midterm
            X["inc_pres_mid"] = df.midterm.values * df.inc.values * df.pres.values
        if cfg.get("pvi_sen"):         # v4: Senate-specific PVI slope
            X["pvi_sen"] = df.lean.values * sen
        if cfg.get("rsafe_sen"):
            X["r_safe_sen"] = np.clip(r, -3, 3) * safe * sen
        if cfg.get("pa_money_all"):
            X["money"] = df.money.values
        if cfg.get("pa_extra_sen"):
            X["senate"] = sen
        if cfg.get("pa_last"):
            X["last_rel"] = df.last_rel.values * comp
        if cfg.get("last_all"):        # previous House result vs nation, all races, with a missing dummy
            X["last_rel"] = df.last_rel.values
            lm = cfg.get("last_miss_mode", "dummy")
            if lm == "dummy":
                X["last_miss"] = df.last_miss.values
            elif lm == "house_sen":    # missing dummy for House seats only, plus a Senate dummy
                X["last_miss"] = df.last_miss.values * (1 - sen)
                X["senate"] = sen
            elif lm == "senate":       # only a Senate dummy
                X["senate"] = sen
            elif lm in ("dev", "dev_d"):   # centred: previous result minus 2 x PVI, 0 when missing
                has = 1 - df.last_miss.values
                X["last_rel"] = has * np.clip(df.last_rel.values - 2 * df.lean.values, -40, 40)
                X["senate"] = sen
                if lm == "dev_d":
                    X["last_miss"] = df.last_miss.values * (1 - sen)
            elif lm == "demean":       # v5.1: within-year demeaned terms, Senate dummy
                X["last_rel"], X["last_miss"] = _last_terms_demeaned(df, sen)
                X["senate"] = sen
            elif lm == "demean_nomiss":
                X["last_rel"], _ = _last_terms_demeaned(df, sen)
                X["senate"] = sen
            elif lm == "lean":         # missing x PVI (v2's last_miss_lean), Senate dummy
                X["last_miss_lean"] = df.last_miss.values * df.lean.values * (1 - sen)
                X["senate"] = sen
        if cfg.get("last_inc"):        # only where the same incumbent runs again
            X["last_rel_inc"] = df.last_rel.values * (df.inc.values != 0)
        # v4 options: presidential lean (kavi-e / kavi-a) in place of, or next to, Cook PVI
        ls = cfg.get("lean_src", "pvi")
        if ls != "pvi":
            pl = df.plean.values / 2.0          # back on the PVI (share-point) scale
            pl2 = df.plean2.values / 2.0
            if ls == "plean":
                X["pvi"], X["pvi_safe"] = pl, pl * safe
            elif ls == "plean12":
                X["pvi"], X["pvi_safe"] = pl, pl * safe
                X["pl2"], X["pl2_safe"] = pl2, pl2 * safe
            elif ls == "both":
                X["pl_dev"] = pl - df.lean.values
            elif ls == "both_comp":
                X["pl_dev"] = (pl - df.lean.values) * comp
        if cfg.get("lin_rating"):
            X["r_mid"] = r
            del X["r_hi"], X["r_lo"]
        if cfg.get("pres_lean"):     # lean slope in presidential years (kavi-e's L_pres)
            X["pvi_pres"] = X["pvi"] * (1 - df.midterm.values)
        for c in cfg.get("pr_extra", []):   # pau-r: extra race-level columns (0 where missing)
            X[c] = df[c].fillna(0).values if c in df else 0.0
        return pd.DataFrame(X)
    ex = cfg.get("extra", [])
    if "pvi_safe" in ex:
        X["pvi_safe"] = df.lean.values * (1 - comp)
    # out-party bonus in midterms: +1 when the D-side is the out-party in a midterm (2018, 2026)
    outmid = df.midterm.values * (-df.pres.values)
    if "outparty_mid" in ex:
        X["outparty_mid"] = outmid
    if "outparty_mid_comp" in ex:
        X["outparty_mid_comp"] = outmid * comp
    if "inc_pres_mid" in ex:      # incumbent of the president's party in a midterm
        X["inc_pres_mid"] = df.midterm.values * df.inc.values * df.pres.values
    if "gb_comp" in ex:           # generic-ballot slope specific to rated-competitive races
        X["gb_comp"] = df.gb.values * comp
    if "gb_inc" in ex:
        X["gb_inc"] = df.gb.values * df.inc.values
    if "approval" in ex:          # president's net approval, signed toward the D-side
        X["approval"] = df.approval_net.values * df.pres.values
    if "rating_raters" in ex:     # disagreement among the three raters
        rr = np.vstack([df.r_cook.values, df.r_ie.values, df.r_sab.values]).astype(float)
        X["rating_cook_dev"] = np.clip(df.r_cook.values, -2, 2) - np.clip(r, -2, 2)
        X["rating_spread"] = rr.std(0) * np.sign(r)
    if "rating_fine" in ex:       # one dummy per rounded rating step inside the competitive band
        for v in (-2, -1, 0, 1, 2):
            X[f"rat_{v}"] = (np.round(np.clip(r, -2, 2)) == v).astype(float)
    if "lean_comp" in ex:
        X["lean_comp"] = df.lean.values * comp
    if "inc_comp" in ex:
        X["inc_comp"] = df.inc.values * comp
    if "money_comp" in ex:
        X["money_comp"] = df.money.values * comp
    return pd.DataFrame(X)


YEAR_LEVEL = ("gb", "outparty_mid", "approval")


def _regressor(cfg, alpha):
    if cfg.get("fitter") == "wls":      # weighted least squares, tiny ridge (pau-a's stage A)
        return Ridge(alpha=1e-3)
    return HuberRegressor(epsilon=cfg.get("epsilon", 1.5), alpha=alpha, max_iter=3000)


def _pen_wls(Z, y, w, pen, tgt):
    """Weighted least squares with an unpenalised intercept and a ridge penalty pen_j (b_j - tgt_j)^2."""
    A = np.column_stack([np.ones(len(Z)), Z])
    W = w[:, None]
    P = np.diag(np.r_[0.0, pen])
    t = np.r_[0.0, tgt]
    sol = np.linalg.solve(A.T @ (W * A) + P, A.T @ (w * y) + P @ t)
    return sol[0], sol[1:]


def year_vars(df, names):
    Z = {}
    for n in names:
        if n == "gb":
            Z[n] = df.gb.values.astype(float)
        elif n == "outmid":
            Z[n] = (df.midterm.values * (-df.pres.values)).astype(float)
        elif n == "gbmid":     # generic ballot trusted in midterms only
            Z[n] = (df.gb.values * df.midterm.values).astype(float)
        elif n == "approval":
            Z[n] = (df.approval_net.values * df.pres.values).astype(float)
    return pd.DataFrame(Z, index=df.index)


class TwoLevelMarginModel:
    """Race-level Huber regression with year fixed effects (so race-level coefficients are
    identified within years), plus a year-level regression of the fixed effects on year-level
    variables (cfg["year_vars"]; "avg" variants average one-variable fits)."""

    def __init__(self, cfg, alpha=1e-3):
        self.cfg = cfg
        self.alpha = alpha

    def fit(self, df):
        X = design(df, self.cfg).drop(columns=[c for c in YEAR_LEVEL if c in design(df, self.cfg)])
        self.years_ = sorted(df.year.unique())
        for y in self.years_[1:]:
            X[f"fe{y}"] = (df.year.values == y).astype(float)
        self.mu_, self.sd_ = X.mean(), X.std().replace(0, 1)
        Z = (X - self.mu_) / self.sd_
        self.reg_ = _regressor(self.cfg, self.alpha)
        w = np.where(np.abs(_rating(df, self.cfg)) < 2.5, self.cfg.get("comp_weight", 1.0), 1.0)
        if self.cfg.get("mid_weight"):     # extra weight on midterm training years
            w = w * np.where(df.midterm.values == 1, self.cfg["mid_weight"], 1.0)
        if self.cfg.get("teacher_years"):
            # v5 research Q4: pretrain on 538's projected margins of earlier cycles, then refit on outcomes with
            # a ridge penalty pulling the race-level coefficients toward the 538-distilled ones
            race_cols = [c for c in X.columns if not c.startswith("fe")]
            bT = self._teacher_beta(race_cols)
            tgt = np.zeros(Z.shape[1])
            pen = np.zeros(Z.shape[1])
            for j, c in enumerate(X.columns):
                if c in race_cols:
                    tgt[j] = bT[c] * self.sd_[c]
                    pen[j] = self.cfg.get("teacher_lam", 10.0)
                else:
                    pen[j] = 1e-3
            b0, b = _pen_wls(Z.values, np.clip(df.y.values, -60, 60), w, pen, tgt)
            self.reg_ = type("R", (), {})()
            self.reg_.coef_, self.reg_.intercept_ = b, b0
        else:
            self.reg_.fit(Z.values, np.clip(df.y.values, -60, 60), sample_weight=w)
        coef = pd.Series(self.reg_.coef_ / self.sd_.values, index=X.columns)
        self.beta_ = coef.drop([f"fe{y}" for y in self.years_[1:]])
        icpt = self.reg_.intercept_ - float((self.reg_.coef_ * self.mu_.values / self.sd_.values).sum())
        fe = {self.years_[0]: 0.0}
        fe.update({y: coef[f"fe{y}"] for y in self.years_[1:]})
        self.icpt_ = icpt
        yr = df.groupby("year").first().loc[self.years_]
        self.fe_ = pd.Series([icpt + fe[y] for y in self.years_], index=self.years_)
        if self.cfg.get("year_model") == "viaN":
            self._fit_viaN(df)
        specs = self.cfg.get("year_vars", ["gb"])
        specs = specs if isinstance(specs[0], list) else [specs]   # list of specs -> average
        self.yl_ = []
        for spec in specs:
            Zy = year_vars(yr, spec)
            A = np.column_stack([np.ones(len(yr))] + [Zy[c].values for c in spec])
            lam = max(self.cfg.get("year_ridge", 0.0), 1e-3)   # tiny ridge keeps 1-2 year fits defined
            P = lam * np.eye(A.shape[1])
            P[0, 0] = 0
            b = np.linalg.lstsq(A.T @ A + P, A.T @ self.fe_.values, rcond=None)[0]
            self.yl_.append((spec, b))
        return self

    def _teacher_beta(self, race_cols):
        """Fit the race-level design on 538's final projected margins of the teacher years (year FE, weights as
        the outcome fit, plus a poll term the teacher may use) and return its race-level coefficients."""
        from common import WORK
        key = (tuple(self.cfg["teacher_years"]), tuple(race_cols), self.cfg.get("comp_weight"), self.cfg.get("pr_tfile"))
        cache = TwoLevelMarginModel.__dict__.get("_tcache", {})
        if key in cache:
            return cache[key]
        h = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
        t = pd.read_csv(WORK / self.cfg.get("pr_tfile", "teacher_final.csv"))
        T = h[(h.train_ok == 1) & h.year.isin(self.cfg["teacher_years"])].merge(t[["year", "race_id", "m538"]],
                                                                               on=["year", "race_id"])
        cfg = {k: v for k, v in self.cfg.items() if k not in ("teacher_years",)}
        XT = design(T, cfg)
        XT = XT[race_cols].copy()
        pol = T.pm4.notna().values if "pm4" in T else np.zeros(len(T), bool)
        XT["t_poll"] = np.where(pol, np.clip(T.pm4.fillna(0).values, -40, 40), 0.0)
        XT["t_polled"] = pol.astype(float)
        ys = sorted(T.year.unique())
        for y in ys[1:]:
            XT[f"tfe{y}"] = (T.year.values == y).astype(float)
        mu, sd = XT.mean(), XT.std().replace(0, 1)
        ZT = (XT - mu) / sd
        wT = np.where(np.abs(T.rating.values) < 2.5, cfg.get("comp_weight", 1.0), 1.0)
        b0, b = _pen_wls(ZT.values, np.clip(T.m538.values, -60, 60), wT, np.full(ZT.shape[1], 1e-3), np.zeros(ZT.shape[1]))
        coef = pd.Series(b / sd.values, index=XT.columns)
        out = coef[race_cols]
        cache[key] = out
        TwoLevelMarginModel._tcache = cache
        return out

    def _fit_viaN(self, df):
        """v5 research (RESEARCH.md Q1c): year effect = c0 + c1 N (training years, N = actual national House
        two-party margin, known after each training election); N predicted from the generic ballot by a separate
        line for midterms and presidential years, fitted on 1998-2008 (no ratings, so outside the race-level fit)
        plus the training years. The test year is never used."""
        from common import WORK
        ext = pd.read_csv(WORK / "national_ext.csv")
        yr = df.groupby("year").first().loc[self.years_]
        N = yr.house_actual.values.astype(float)
        A = np.column_stack([np.ones(len(N)), N])
        self.cN_ = np.linalg.lstsq(A, self.fe_.values, rcond=None)[0]
        ext = ext[(ext.year < 2010) | ext.year.isin(self.years_)].copy()
        hn = dict(zip(yr.index, N))
        ext["N"] = [hn.get(y, n) for y, n in zip(ext.year, ext.N)]
        self.nline_ = {}
        for mid in (0, 1):
            e = ext[ext.midterm == mid]
            if self.cfg.get("viaN_shrink"):      # slope shrunk toward 1 (GB measures N), ridge on (b - 1)
                lam = self.cfg["viaN_shrink"]
                X = np.column_stack([np.ones(len(e)), e.gb.values])
                P = np.diag([0.0, lam])
                b = np.linalg.solve(X.T @ X + P, X.T @ e.N.values + P @ np.array([0.0, 1.0]))
            else:
                b = np.polyfit(e.gb.values, e.N.values, 1)[::-1]
            self.nline_[mid] = b

    def year_effect(self, df):
        if self.cfg.get("year_model") == "viaN":
            yr = df.groupby("year").first()
            nh = np.array([self.nline_[int(m)][0] + self.nline_[int(m)][1] * g for g, m in zip(yr.gb, yr.midterm)])
            via = pd.Series(self.cN_[0] + self.cN_[1] * nh, index=yr.index)
            if self.cfg.get("viaN_mid_only"):
                base = self._ye_specs(df)
                w = self.cfg.get("viaN_w", 1.0)
                mix = w * via.values + (1 - w) * base.values
                return pd.Series(np.where(yr.midterm.values == 1, mix, base.values), index=yr.index)
            return via
        return self._ye_specs(df)

    def _ye_specs(self, df):
        yr = df.groupby("year").first()
        preds = []
        for spec, b in self.yl_:
            Zy = year_vars(yr, spec)
            preds.append(b[0] + sum(b[j + 1] * Zy[c].values for j, c in enumerate(spec)))
        return pd.Series(np.mean(preds, axis=0), index=yr.index)

    def predict(self, df):
        X = design(df, self.cfg)
        X = X.drop(columns=[c for c in YEAR_LEVEL if c in X])
        race = X[self.beta_.index].values @ self.beta_.values
        ye = self.year_effect(df)
        hook = TwoLevelMarginModel.__dict__.get("_ye_hook")
        if hook is not None:
            ye = hook(self, df, race, ye)
        if self.cfg.get("oracle_ye") and "y" in df:   # pau-r headroom: the true year effect (leaks the outcome)
            w = np.where(np.abs(_rating(df, self.cfg)) < 2.5, self.cfg.get("comp_weight", 1.0), 1.0)
            r = np.clip(df.y.values, -60, 60) - race
            t = pd.DataFrame({"year": df.year.values, "wr": w * r, "w": w})
            g = t.groupby("year").sum()
            ye = (g.wr / g.w)
            frac = self.cfg["oracle_ye"]
            if frac != 1:
                base = self.year_effect(df)
                ye = base + frac * (ye - base)
        return race + df.year.map(ye).values

    def coefs(self):
        out = self.beta_.copy()
        if hasattr(self, "nline_"):
            out["fe_c0"], out["fe_cN"] = self.cN_
            for k, b in self.nline_.items():
                out[f"N_icpt_mid{k}"], out[f"N_gb_mid{k}"] = b
        for spec, b in self.yl_:
            out[f"yl_icpt[{'+'.join(spec) or 'none'}]"] = b[0]
            for j, c in enumerate(spec):
                out[f"yl_{c}[{'+'.join(spec)}]"] = b[j + 1]
        return out


class MarginModel:
    def __new__(cls, cfg=V1, alpha=1e-3):
        if cfg.get("two_level"):
            return TwoLevelMarginModel(cfg, alpha)
        return super().__new__(cls)

    def __init__(self, cfg=V1, alpha=1e-3):
        self.cfg = cfg
        self.alpha = alpha

    def fit(self, df):
        X = design(df, self.cfg)
        self.mu_, self.sd_ = X.mean(), X.std().replace(0, 1)
        Z = (X - self.mu_) / self.sd_
        self.reg_ = _regressor(self.cfg, self.alpha)
        w = np.where(np.abs(_rating(df, self.cfg)) < 2.5, self.cfg.get("comp_weight", 1.0), 1.0)
        # cap |y| so that 90-point blowouts don't dominate the fit
        self.reg_.fit(Z.values, np.clip(df.y.values, -60, 60), sample_weight=w)
        return self

    def predict(self, df):
        Z = (design(df, self.cfg) - self.mu_) / self.sd_
        return self.reg_.predict(Z.values)

    def coefs(self):
        return pd.Series(self.reg_.coef_ / self.sd_.values, index=self.sd_.index)


def group_of(df, cfg=V1):
    ar = np.abs(_rating(df, cfg))
    comp = (ar < 2.5).astype(int)
    polled = df.poll_pollsters.fillna(0).values > 0
    a = pd.Series(np.where(df.senate.values == 1, "S", "H"))
    if cfg.get("sd_bins"):
        b = pd.Series(np.where(ar <= 1.0, "c", np.where(ar < 2.5, "l", "s")))
    else:
        b = pd.Series(np.where(comp == 1, "c", "s"))
    c = pd.Series(np.where(polled, "p", "n"))
    return (a + b + c).values


def shrink(yhat, poll, n, lam, lmax=1.0):
    n = np.nan_to_num(n)
    k = np.where(n > 0, lmax * n / (n + lam), 0.0)
    innov = np.clip(np.nan_to_num(poll - yhat), -25, 25)
    return yhat + k * innov


def link_cdf(z, cfg):
    if cfg.get("link") == "t":
        return student_t.cdf(z, cfg.get("nu", 5))
    return norm.cdf(z)


class WinModel:
    """Fits stages 1-4 on the training years of `hist` (rows with train_ok == 1)."""

    def __init__(self, train_years, cfg=V1):
        self.train_years = list(train_years)
        self.cfg = cfg

    def _loyo(self, tr):
        out = []
        for t in self.train_years:
            a, b = tr[tr.year != t], tr[tr.year == t]
            m = MarginModel(self.cfg).fit(a)
            out.append(b.assign(yhat=m.predict(b)))
        return pd.concat(out)

    def _polls(self, df):
        """v3: swap in the pau-a-style poll averages of the right training regime; v4: kavi-e-style averages."""
        if self.cfg.get("polls") == "v4":
            df = df.copy()
            sfx = self.cfg.get("v4_sfx", "")
            n = df[self.cfg.get("v4_n", "pn4") + sfx].astype(float)
            if self.cfg.get("v4_age"):          # discount stale averages: n x 0.5 ** (days since last poll / age)
                n = n * 0.5 ** (df["pd4" + sfx].astype(float) / self.cfg["v4_age"])
            if self.cfg.get("v4_sen_sfx") is not None:   # Senate: v4 average with other parameters
                s = df.senate.values == 1
                ss = self.cfg["v4_sen_sfx"]
                df["poll_margin"] = np.where(s, df["pm4" + ss], df["pm4" + sfx])
                df["poll_pollsters"] = np.where(s, df["pn4" + ss].astype(float), n)
            elif self.cfg.get("v4_house_only"):  # Senate keeps the v3 21-day pollster average
                s = df.senate.values == 1
                df["poll_margin"] = np.where(s, df.poll_margin, df["pm4" + sfx])
                df["poll_pollsters"] = np.where(s, df.poll_pollsters, n)
            else:
                df["poll_margin"], df["poll_pollsters"] = df["pm4" + sfx], n
            return df
        if self.cfg.get("polls") != "pa":
            return df
        reg = self.cfg.get("poll_regime", "loyo")
        col = f"pm_{reg}" if f"pm_{reg}" in df else "pm_all"
        ncol = col.replace("pm_", "pn_")
        df = df.copy()
        df["poll_margin"], df["poll_pollsters"] = df[col], df[ncol]
        return df

    def fit(self, hist):
        cfg = self.cfg
        if cfg.get("sen_cfg"):        # v4 hybrid: the Senate rows come from a WinModel with another config
            self.sen_ = WinModel(self.train_years, cfg["sen_cfg"]).fit(hist)
        if cfg.get("pool_cfg"):       # v5: House logit pool with a second component
            self.pool_ = WinModel(self.train_years, cfg["pool_cfg"]).fit(hist)
        if cfg.get("pool3_cfg"):
            self.pool3_ = WinModel(self.train_years, cfg["pool3_cfg"]).fit(hist)
        hist = self._polls(hist)
        tr = hist[(hist.train_ok == 1) & hist.year.isin(self.train_years)].copy()
        if cfg.get("fit_chamber") is not None:   # v4: stage 1 fitted on one chamber's rows only
            tr = tr[tr.senate == cfg["fit_chamber"]]
        self.stage1_ = MarginModel(cfg).fit(tr)
        lo = self._loyo(tr)

        # stage 2: lambda per chamber, robust loss on polled races
        self.lam_ = {}
        self.lmax_ = {0: 1.0, 1: 1.0}
        for s in (0, 1):
            g = lo[(lo.senate == s) & (lo.poll_pollsters.fillna(0) > 0)]
            if len(g) < 15:
                self.lam_[s] = 4.0  # prior: a single poll gets ~20% weight
                continue

            def loss_k(lam, lmax):
                mu = shrink(g.yhat.values, g.poll_margin.values, g.poll_pollsters.values, lam, lmax)
                e = g.y.values - mu
                return np.sum(np.where(np.abs(e) < 10, e ** 2, 20 * np.abs(e) - 100))

            if cfg.get("lam_max"):
                # ceiling on the poll weight (borrowed from pau-a): k = lmax * n / (n + lam)
                res = minimize(lambda th: loss_k(np.exp(th[0]), 1 / (1 + np.exp(-th[1]))), [0.0, 2.0],
                               method="Nelder-Mead")
                self.lam_[s] = float(np.exp(res.x[0]))
                self.lmax_[s] = float(1 / (1 + np.exp(-res.x[1])))
            else:
                res = minimize_scalar(lambda ll: loss_k(np.exp(ll), 1.0), bounds=(np.log(0.05), np.log(50)),
                                      method="bounded")
                self.lam_[s] = float(np.exp(res.x))
        lo["mu"] = self._mu(lo)

        # stage 3: residual SD by group (robust: 1.4826 * MAD blended with SD)
        lo["grp"] = group_of(lo, cfg)
        self.sd_ = {}
        for gname, g in lo.groupby("grp"):
            e = g.y - g.mu
            mad = 1.4826 * np.median(np.abs(e - np.median(e)))
            self.sd_[gname] = float(np.sqrt(0.5 * mad ** 2 + 0.5 * min(e.std(), 3 * mad) ** 2)) if len(g) > 10 else np.nan
        fallback = {"H": 8.0, "S": 7.0}
        for c in ("H", "S"):
            for comp in ("c", "l", "s"):
                for p in ("p", "n"):
                    k = c + comp + p
                    if k not in self.sd_ or not np.isfinite(self.sd_[k]):
                        alt = c + comp + ("n" if p == "p" else "p")
                        self.sd_[k] = self.sd_.get(alt, fallback[c]) if np.isfinite(self.sd_.get(alt, np.nan)) else fallback[c]
        lo["sd"] = lo.grp.map(self.sd_)

        if cfg.get("cal") == "group_t":
            self._fit_group_t(lo)
            if cfg.get("cal2"):
                self._fit_cal2(lo)
            self.loyo_ = lo
            if cfg.get("sen_cfg"):
                self.loyo_ = pd.concat([lo[lo.senate == 0], self.sen_.loyo_[self.sen_.loyo_.senate == 1]])
            return self

        # stage 4: calibration slope
        z = (lo.mu / lo.sd).values
        yy = lo.dwin.values

        def nll(a):
            p = np.clip(link_cdf(a * z, cfg), 1e-6, 1 - 1e-6)
            return -np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p))

        self.a_ = float(minimize_scalar(nll, bounds=(0.3, 3), method="bounded").x)
        self.loyo_ = lo
        return self

    def _mu(self, df):
        s = df.senate.values == 1
        return shrink(df.yhat.values, df.poll_margin.values, df.poll_pollsters.values,
                      np.where(s, self.lam_[1], self.lam_[0]), np.where(s, self.lmax_[1], self.lmax_[0]))

    def _g4(self, df):
        g = (pd.Series(np.where(df.senate.values == 1, "S", "H")) +
             pd.Series(np.where(df.poll_pollsters.fillna(0).values > 0, "P", "N"))).values
        if self.cfg.get("sig_cluster"):    # v5 research Q3: type-aware scale, rated races split by k-means type
            from common import WORK
            if not hasattr(WinModel, "_clu"):
                c = pd.read_csv(WORK / "race_cluster.csv")
                WinModel._clu = dict(zip(zip(c.year, c.race_id), c.clu))
            lab = np.array([WinModel._clu.get((y, r), 0) for y, r in zip(df.year, df.race_id)])
            g = np.array([f"{a}{b}" if b > 0 else a for a, b in zip(g, lab)], dtype=object)
        return g

    def _fit_group_t(self, lo):
        """Student-t calibration with a scale per chamber x polled group (borrowed from pau-a):
        sigma = average of the likelihood optimum on the LOYO predictions and a robust MAD spread of
        the competitive (|mu| < 15) residuals put on the t scale; tail floor eps."""
        df_t, eps = (self.cfg.get("nu") or 10), self.cfg.get("eps", 0.001)
        g = self._g4(lo)
        yy = lo.dwin.values.astype(float)
        mu = lo.mu.values
        comp = np.abs(mu) < 15
        self.sig_, self.sig_parts_ = {}, {}
        keys = sorted(set(g)) if self.cfg.get("sig_cluster") else ("HN", "HP", "SN", "SP")
        for k in keys:
            m = g == k
            if (m & comp).sum() < 12:
                m = np.char.startswith(g.astype(str), k[:2]) if len(k) > 2 else np.char.startswith(g.astype(str), k[0])
            if (m & comp).sum() < 12:
                m = np.char.startswith(g.astype(str), k[0])

            def nll(ls):
                p = (1 - eps) * student_t.cdf(mu[m] / np.exp(ls), df_t) + eps / 2
                return -np.sum(yy[m] * np.log(p) + (1 - yy[m]) * np.log(1 - p))

            s_ml = float(np.exp(minimize_scalar(nll, bounds=(0.5, 3.5), method="bounded").x))
            e = (lo.y.values - mu)[m & comp]
            s_rob = 1.4826 * np.median(np.abs(e - np.median(e))) / np.sqrt(df_t / (df_t - 2))
            mode = self.cfg.get("sigma_mode", "avg")
            self.sig_[k] = {"mle": s_ml, "resid": s_rob, "avg": 0.5 * (s_ml + s_rob)}[mode]
            self.sig_parts_[k] = (round(s_ml, 2), round(s_rob, 2))

    def _p_t(self, df):
        df_t, eps = (self.cfg.get("nu") or 10), self.cfg.get("eps", 0.001)
        sig = pd.Series(self._g4(df)).map(self.sig_).values
        return (1 - eps) * student_t.cdf(df.mu.values / sig, df_t) + eps / 2

    def _cal2_X(self, df, p):
        """Calibration-layer design (kavi-a's idea): logit of the t probability, clipped rating, Senate terms."""
        z = np.clip(np.log(p / (1 - p)), -7, 7)
        r = np.clip(df.rating.values.astype(float), -1.5, 1.5)
        s = df.senate.values.astype(float)
        kind = self.cfg["cal2"]
        if kind == "sr0":              # v5: pau-a's form, slope + rating, no intercept
            return np.column_stack([z, np.clip(df.rating.values.astype(float), -2.5, 2.5)])
        cols = [np.ones(len(z)), z]
        if kind in ("slope_r", "slope_r_sen"):
            cols.append(r)
        if kind == "slope_r_sen":
            cols += [s * z, s * r]
        if kind == "slope_comp":       # separate slope for rated races
            cols.append(z * (np.abs(df.rating.values) < 2.5))
        return np.column_stack(cols)

    def _fit_cal2(self, lo):
        p = np.clip(self._p_t(lo), 1e-4, 1 - 1e-4)
        X = self._cal2_X(lo, p)
        yy = lo.dwin.values.astype(float)
        ridge = self.cfg.get("cal2_ridge", 1.0)

        def nll(b):
            eta = X @ b
            return np.sum(np.logaddexp(0, eta) - yy * eta) + ridge * np.sum((b - self._cal2_b0(X.shape[1])) ** 2)

        b0 = self._cal2_b0(X.shape[1])
        self.cal2_b_ = minimize(nll, b0, method="BFGS").x

    def _cal2_b0(self, k):
        b = np.zeros(k)
        b[0 if self.cfg.get("cal2") == "sr0" else 1] = 1.0
        return b

    def predict(self, df):
        if self.cfg.get("sen_cfg") and not getattr(self, "_in_hybrid", False):
            self._in_hybrid = True
            try:
                h = self.predict(df)
            finally:
                self._in_hybrid = False
            s = self.sen_.predict(df)
            sen = (df.senate.values == 1)
            h = h.reset_index(drop=True)
            s = s.reset_index(drop=True)
            if self.cfg.get("pool_cfg"):
                q = self.pool_.predict(df).reset_index(drop=True)
                w = self.cfg.get("pool_w", 0.5)
                lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
                h["p_main"], h["p_pool2"] = h.p_dside.values, q.p_dside.values
                eta = (1 - w) * lg(h.p_dside.values) + w * lg(q.p_dside.values)
                if self.cfg.get("pool3_cfg"):
                    q3 = self.pool3_.predict(df).reset_index(drop=True)
                    eta = (lg(h.p_dside.values) + lg(q.p_dside.values) + lg(q3.p_dside.values)) / 3
                pooled = 1 / (1 + np.exp(-eta))
                h.loc[~sen, "p_dside"] = pooled[~sen]
                if self.cfg.get("pool_sen"):
                    ws = self.cfg["pool_sen"]
                    ps = 1 / (1 + np.exp(-((1 - ws) * lg(s.p_dside.values) + ws * lg(q.p_dside.values))))
                    s = s.copy()
                    s["p_dside"] = ps
            for c in s.columns:
                if c in h.columns:
                    h.loc[sen, c] = s.loc[sen, c].values
            return h
        out = self._polls(df).copy()
        out["yhat"] = self.stage1_.predict(out)
        out["mu"] = self._mu(out)
        if self.cfg.get("cal") == "group_t":
            df_t, eps = (self.cfg.get("nu") or 10), self.cfg.get("eps", 0.001)
            out["grp"] = self._g4(out)
            out["sig"] = out.grp.map(self.sig_)
            out["sd"] = out.sig * np.sqrt(df_t / (df_t - 2))     # SD of the t error (for simulations)
            out["p_dside"] = (1 - eps) * student_t.cdf(out.mu / out.sig, df_t) + eps / 2
            out["sd_total"] = out.sd
            if self.cfg.get("cal2"):
                p = np.clip(out.p_dside.values, 1e-4, 1 - 1e-4)
                eta = self._cal2_X(out, p) @ self.cal2_b_
                w = self.cfg.get("cal2_w", 1.0)
                lg = w * eta + (1 - w) * np.log(p / (1 - p))
                out["p_raw"] = out.p_dside
                out["p_dside"] = 1 / (1 + np.exp(-lg))
            return out
        out["grp"] = group_of(out, self.cfg)
        out["sd"] = out.grp.map(self.sd_)
        out["p_dside"] = link_cdf(self.a_ * out.mu / out.sd, self.cfg)
        out["sd_total"] = out.sd / self.a_
        return out

    def summary(self):
        if self.cfg.get("sen_cfg"):
            hs = dict(self._summary())
            return dict(house=hs, senate=self.sen_._summary(), train_years=self.train_years, cfg=self.cfg,
                        lam=hs["lam"], lmax=hs["lmax"], a=None, sig={**{k: v for k, v in hs["sig"].items() if k[0] == "H"},
                        **{k: v for k, v in self.sen_.sig_.items() if k[0] == "S"}}, sig_parts=None, coefs=hs["coefs"])
        return self._summary()

    def _summary(self):
        return dict(train_years=self.train_years, cfg=self.cfg, lam=self.lam_, lmax=self.lmax_,
                    sd=getattr(self, "sd_", None), a=getattr(self, "a_", None), sig=getattr(self, "sig_", None),
                    sig_parts=getattr(self, "sig_parts_", None),
                    coefs=self.stage1_.coefs().round(3).to_dict())
