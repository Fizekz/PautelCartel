"""v4 model selection on held-out years that are NOT the 2018/2022 backtest years.

Schemes (nested: the WinModel's inner LOYO for poll weights and calibration only sees the training years):
  six   : the six non-test cycles 2010, 2012, 2014, 2016, 2020, 2024; each held out in turn, trained on the
          other five (2018 and 2022 are never used for training or scoring)
  pre18 : 2010-2016 only; each held out, trained on the other three
  bt    : report only (not used to choose): the official backtests, 2018 trained on 2010-2016, 2022 on 2010-2020
Scored on contested D-side vs R races (train_ok == 1); 1% floor like the contest.
"""
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import WORK
from model import WinModel, FINAL
from evaluate import metrics

SIX = [2010, 2012, 2014, 2016, 2020, 2024]
PRE18 = [2010, 2012, 2014, 2016]


def splits(scheme):
    if scheme == "six":
        return [(T, [y for y in SIX if y != T]) for T in SIX]
    if scheme == "pre18":
        return [(T, [y for y in PRE18 if y != T]) for T in PRE18]
    if scheme == "bt":
        return [(2018, [2010, 2012, 2014, 2016]), (2022, [2010, 2012, 2014, 2016, 2018, 2020])]
    if scheme == "q4":     # teacher-compatible tests: the teacher (538) must predate the test year
        return [(2020, [2010, 2012, 2014, 2016, 2024]), (2024, [2010, 2012, 2014, 2016, 2020])]
    if scheme == "q4bt":
        return [(2022, [2010, 2012, 2014, 2016, 2018, 2020])]
    if scheme == "loyo8":
        Y = list(range(2010, 2026, 2))
        return [(T, [y for y in Y if y != T]) for T in Y]
    raise ValueError(scheme)


def _teach(cfg, T):
    """Teacher years allowed for test year T: 538 forecasts made before T (2018 for 2020/2022, 2018+2022 for 2024)."""
    if not cfg.get("q4"):
        return cfg
    ty = [y for y in (2018, 2022) if y < T]
    kw = dict(teacher_years=ty, teacher_lam=cfg["q4"])
    c = dict(cfg, **kw)
    if cfg.get("q4_all"):
        if "pool_cfg" in c:
            c["pool_cfg"] = dict(c["pool_cfg"], **kw)
        if "sen_cfg" in c:
            c["sen_cfg"] = dict(c["sen_cfg"], **kw)
    return c


def one(hist, cfg, T, train):
    wm = WinModel(train, _teach(cfg, T)).fit(hist)
    te = hist[(hist.year == T) & (hist.train_ok == 1)]
    return wm.predict(te)[["year", "race_id", "senate", "rating", "y", "dwin", "yhat", "mu", "p_dside",
                           "poll_pollsters"]]


def run(name, cfg, hist, schemes=("six", "pre18", "bt")):
    if cfg.get("q4") is not None or name.startswith("q4"):
        schemes = ("q4", "q4bt")
    res = {}
    for sch in schemes:
        pr = pd.concat([one(hist, cfg, T, tr) for T, tr in splits(sch)])
        pr.to_csv(WORK / f"ev4_{name}_{sch}.csv", index=False)
        rated = pr[pr.rating.abs() < 2.5]
        pw = np.where(pr.dwin == 1, pr.p_dside, 1 - pr.p_dside)
        res[sch] = dict(c3=float(-np.mean(np.log(np.clip(pw, 0.001, 0.999)))),   # common scorecard's clip
                        all=metrics(pr.p_dside, pr.dwin)["ll"], rated=metrics(rated.p_dside, rated.dwin)["ll"],
                        sen=metrics(pr[pr.senate == 1].p_dside, pr[pr.senate == 1].dwin)["ll"],
                        per_year={int(y): round(metrics(g.p_dside, g.dwin)["ll"], 4) for y, g in pr.groupby("year")})
    return name, res


from model import V1, V3, V4, V5, FINAL as V51
P4 = dict(polls="v4")
SIMPLE = dict(V1, design="simple", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t", nu=10,
              two_level=False, no_money=True, polls="v4", v4_house_only=True)


def _sen(cfg, **kw):
    return dict(cfg, sen_cfg=dict(cfg["sen_cfg"], **kw))


VARIANTS = {
    "v51": V51,
    "v5": V5,
    "v4": V4,
    "v5_lt": dict(_sen(V4, ltrend="comp"), ltrend="comp"),
    "v5_lt_all": dict(_sen(V4, ltrend="all"), ltrend="all"),
    "v5_lt_h": dict(V4, ltrend="comp"),
    "v5_cal": dict(_sen(V4, cal2="sr0", cal2_w=0.5), cal2="sr0", cal2_w=0.5),
    "v5_cal_h": dict(V4, cal2="sr0", cal2_w=0.5),
    "v5_cal_s": _sen(V4, cal2="sr0", cal2_w=0.5),
    "v5_sfix": _sen(V4, safe_fix=True),
    "v5_pool": dict(V4, pool_cfg=SIMPLE),
    "v5_pool_lt": dict(V4, pool_cfg=dict(SIMPLE, ltrend="comp")),
    "v5_pool_last": dict(V4, pool_cfg=dict(SIMPLE, last_all=True)),
    "v5_pool_v3": dict(V4, pool_cfg=V3),
}
CAL = dict(cal2="sr0", cal2_w=0.5)
C1 = dict(_sen(V4, safe_fix=True, **CAL), **CAL)
SL = dict(SIMPLE, last_all=True)
VARIANTS.update({
    "c1": C1,
    "v51_dm": dict(V5, last_miss_mode="demean", pool_cfg=dict(V5["pool_cfg"], last_miss_mode="demean"),
                   sen_cfg=dict(V5["sen_cfg"], last_miss_mode="demean")),
    "v51_dmr": dict(V5, last_miss_mode="demean_nomiss", pool_cfg=dict(V5["pool_cfg"], last_miss_mode="demean_nomiss"),
                    sen_cfg=dict(V5["sen_cfg"], last_miss_mode="demean_nomiss")),
    "v51_nopool": {k: v for k, v in dict(V5, last_all=False, sen_cfg=dict(V5["sen_cfg"], last_all=False)).items() if k != "pool_cfg"},
    "v51_none_nocal": dict(V5, last_all=False, cal2=None, pool_cfg=dict(V5["pool_cfg"], last_all=False, cal2=None),
                           sen_cfg=dict(V5["sen_cfg"], last_all=False, cal2=None)),
    "v51_none": dict(V5, last_all=False, pool_cfg=dict(V5["pool_cfg"], last_all=False),
                     sen_cfg=dict(V5["sen_cfg"], last_all=False)),
    "q4_base": dict(V5, q4_dummy=True),
    **{f"q4_l{l}": dict(V5, q4=float(l)) for l in (1, 10, 100, 1000, 100000)},
    **{f"q4a_l{l}": dict(V5, q4=float(l), q4_all=True) for l in (10, 100, 1000)},
    "q3_runp": dict(V5, r_unpolled=True),
    "q3_runp_all": dict(V5, r_unpolled=True, pool_cfg=dict(V5["pool_cfg"], r_unpolled=True),
                        sen_cfg=dict(V5["sen_cfg"], r_unpolled=True)),
    "q3_sigclu": dict(V5, sig_cluster=True, pool_cfg=dict(V5["pool_cfg"], sig_cluster=True),
                      sen_cfg=dict(V5["sen_cfg"], sig_cluster=True)),
    "q3_sigclu_h": dict(V5, sig_cluster=True, pool_cfg=dict(V5["pool_cfg"], sig_cluster=True)),
    "vN": dict(V5, year_model="viaN", sen_cfg=dict(V5["sen_cfg"], year_model="viaN"),
               pool_cfg=dict(V5["pool_cfg"])),
    "vN_s1": dict(V5, year_model="viaN", viaN_shrink=1.0, sen_cfg=dict(V5["sen_cfg"], year_model="viaN", viaN_shrink=1.0)),
    "vN_s10": dict(V5, year_model="viaN", viaN_shrink=10.0, sen_cfg=dict(V5["sen_cfg"], year_model="viaN", viaN_shrink=10.0)),
    "vN_h": dict(V5, year_model="viaN"),
    "vNm50": dict(V5, year_model="viaN", viaN_mid_only=True, viaN_w=0.5,
                  sen_cfg=dict(V5["sen_cfg"], year_model="viaN", viaN_mid_only=True, viaN_w=0.5)),
    "vNm": dict(V5, year_model="viaN", viaN_mid_only=True, sen_cfg=dict(V5["sen_cfg"], year_model="viaN", viaN_mid_only=True)),
    "v5_splean": dict(V5, pool_cfg=dict(V5["pool_cfg"], simple_plean=True)),
    "v5_sp21": dict(V5, pool_cfg=dict(V5["pool_cfg"], polls=None, v4_house_only=False)),
    "v5_p3_v4": dict(V5, pool3_cfg=dict(V5["pool_cfg"], simple_plean=True)),
    "v5_psen25": dict(V5, pool_sen=0.25),
    "v5_psen50": dict(V5, pool_sen=0.5),
    "s21_w25": dict(V5, pool_cfg=dict(V5["pool_cfg"], polls=None, v4_house_only=False), pool_w=0.25),
    "s21_w75": dict(V5, pool_cfg=dict(V5["pool_cfg"], polls=None, v4_house_only=False), pool_w=0.75),
    "s21_3way": dict(V5, pool_cfg=dict(V5["pool_cfg"], polls=None, v4_house_only=False), pool3_cfg=V5["pool_cfg"]),
    "s21_main21": dict(V5, polls=None, v4_house_only=False, pool_cfg=dict(V5["pool_cfg"], polls=None, v4_house_only=False)),
    "s21_main21_s120": dict(V5, polls=None, v4_house_only=False),
    "s21_psen": dict(V5, pool_cfg=dict(V5["pool_cfg"], polls=None, v4_house_only=False), pool_sen=0.25),
    "v5_slt": dict(V5, pool_cfg=dict(V5["pool_cfg"], ltrend="comp")),
    "c1_slt": _sen(C1, ltrend="all"),
    "c1_pool": dict(C1, pool_cfg=SL),
    "c1_poolc": dict(C1, pool_cfg=dict(SL, **CAL)),
    "c1_poolc_w25": dict(C1, pool_cfg=dict(SL, **CAL), pool_w=0.25),
    "c1_poolc_w75": dict(C1, pool_cfg=dict(SL, **CAL), pool_w=0.75),
    "c1_poolc_slt": dict(_sen(C1, ltrend="all"), pool_cfg=dict(SL, **CAL)),
    "c1_poolc_lt": dict(C1, pool_cfg=dict(SL, ltrend="comp", **CAL)),
    "c1_cal75": dict(_sen(V4, safe_fix=True, cal2="sr0", cal2_w=0.75), cal2="sr0", cal2_w=0.75),
    "c1_cal100": dict(_sen(V4, safe_fix=True, cal2="sr0", cal2_w=1.0), cal2="sr0", cal2_w=1.0),
    "v4_nolast": dict(V4, last_all=False),
    "v4_wz8": dict(V4, v4_sfx="_wz8"),
    "v4_mw2": dict(V4, mid_weight=2.0, sen_cfg=dict(V4["sen_cfg"], mid_weight=2.0)),
    "v4_rmx": dict(V4, r_mid_x=True, sen_cfg=dict(V4["sen_cfg"], r_mid_x=True)),
    "v4_ipm": dict(V4, inc_pres_mid=True, sen_cfg=dict(V4["sen_cfg"], inc_pres_mid=True)),
    "v4_wz12": dict(V4, v4_sfx="_wz12"),
    "lm_none": dict(V4, last_miss_mode="none", sen_cfg=dict(V4["sen_cfg"], last_miss_mode="none")),
    "lm_hs": dict(V4, last_miss_mode="house_sen", sen_cfg=dict(V4["sen_cfg"], last_miss_mode="house_sen")),
    "lm_sen": dict(V4, last_miss_mode="senate", sen_cfg=dict(V4["sen_cfg"], last_miss_mode="senate")),
    "lm_dev": dict(V4, last_miss_mode="dev", sen_cfg=dict(V4["sen_cfg"], last_miss_mode="dev")),
    "lm_devd": dict(V4, last_miss_mode="dev_d", sen_cfg=dict(V4["sen_cfg"], last_miss_mode="dev_d")),
    "lm_lean": dict(V4, last_miss_mode="lean", sen_cfg=dict(V4["sen_cfg"], last_miss_mode="lean")),
    "v4_plean": dict(V4, lean_src="plean"),
    "v4_cal": dict(V4, cal2="slope_r", cal2_w=0.75),
    "v4_pvisafe": dict(V4, pa_no_pvisafe=False),
    "v4_lastinc2": dict(V4, last_inc=True),
    "v4_cw16": dict(V4, comp_weight=16.0),
    "v4_sen_last": dict(V4, sen_cfg=dict(V4["sen_cfg"], last_all=True)),
    "v4_prim2": dict(V4, prim=True),
    "v4_nu30": dict(V4, nu=30, sen_cfg=dict(V4["sen_cfg"], nu=30)),
    "v4_prim": dict(V4, prim=True),
    "v4_primc": dict(V4, prim_comp=True),
    "v4_last": dict(V4, pa_last=True),
    "v4_lastall": dict(V4, last_all=True),
    "v4_lastinc": dict(V4, last_inc=True),
    "v4_cw12": dict(V4, comp_weight=12.0),
    "v4_cw6": dict(V4, comp_weight=6.0),
    "v4_nu10": dict(V4, nu=10, sen_cfg=dict(V4["sen_cfg"], nu=10)),
    "v4_nu6": dict(V4, nu=6, sen_cfg=dict(V4["sen_cfg"], nu=6)),
    "v4_nu4": dict(V4, nu=4, sen_cfg=dict(V4["sen_cfg"], nu=4)),
    "v4_eps01": dict(V4, eps=0.01, sen_cfg=dict(V4["sen_cfg"], eps=0.01)),
    "v4_eps004": dict(V4, eps=0.004, sen_cfg=dict(V4["sen_cfg"], eps=0.004)),
    "v4_nu10_eps004": dict(V4, nu=10, eps=0.004, sen_cfg=dict(V4["sen_cfg"], nu=10, eps=0.004)),
    "v4_nu6_eps004": dict(V4, nu=6, eps=0.004, sen_cfg=dict(V4["sen_cfg"], nu=6, eps=0.004)),
    "v3": V3,
    "v3_plean": dict(V3, lean_src="plean"),
    "v3_plean12": dict(V3, lean_src="plean12"),
    "v3_both": dict(V3, lean_src="both"),
    "v3_lin": dict(V3, lin_rating=True),
    "v3_p4": dict(V3, **P4),
    "v3_p4k": dict(V3, **P4, v4_n="pk4"),
    "v3_p4_age30": dict(V3, **P4, v4_age=30),
    "v3_p4_plean": dict(V3, **P4, lean_src="plean"),
    "v3_p4_plean12": dict(V3, **P4, lean_src="plean12"),
    "lin_plean": dict(V3, lin_rating=True, lean_src="plean"),
    "lin_p4": dict(V3, lin_rating=True, **P4),
    "lin_p4_plean": dict(V3, lin_rating=True, **P4, lean_src="plean"),
    "lin_p4_plean12": dict(V3, lin_rating=True, **P4, lean_src="plean12"),
    "lin_p4h": dict(V3, lin_rating=True, **P4, v4_house_only=True),
    "lin_p4h_plean": dict(V3, lin_rating=True, **P4, v4_house_only=True, lean_src="plean"),
    "lin_p4_w60": dict(V3, lin_rating=True, **P4, v4_sfx="_w60"),
    "lin_p4_w30": dict(V3, lin_rating=True, **P4, v4_sfx="_w30"),
    "lin_p4_h7": dict(V3, lin_rating=True, **P4, v4_sfx="_h7"),
    "lin_p4_h28": dict(V3, lin_rating=True, **P4, v4_sfx="_h28"),
    "lin_p4_pw": dict(V3, lin_rating=True, **P4, v4_sfx="_pw"),
    "lin_p4k": dict(V3, lin_rating=True, **P4, v4_n="pk4"),
})
B = dict(V3, lin_rating=True, **P4, v4_house_only=True)       # current best (lin_p4h)
VARIANTS.update({
    "b": B,
    "b_sen_h7": dict(B, v4_sen_sfx="_h7"),
    "b_sen_w30": dict(B, v4_sen_sfx="_w30"),
    "b_h_w60": dict(B, v4_sfx="_w60"),
    "b_h_h7": dict(B, v4_sfx="_h7"),
    "b_h_h28": dict(B, v4_sfx="_h28"),
    "b_nopvisafe": dict(B, pa_no_pvisafe=True),
    "b_cal_s": dict(B, cal2="slope"),
    "b_cal_sr": dict(B, cal2="slope_r"),
    "b_cal_srs": dict(B, cal2="slope_r_sen"),
    "b_cal_sr75": dict(B, cal2="slope_r", cal2_w=0.75),
    "b_cal_sc": dict(B, cal2="slope_comp"),
    "b_nu6": dict(B, nu=6),
    "b_nu30": dict(B, nu=30),
    "b_mle": dict(B, sigma_mode="mle"),
    "b_pss": dict(B, pvisafe_sen=True),
    "b_pss_nu30": dict(B, pvisafe_sen=True, nu=30),
    "b_pss_cw1": dict(B, pvisafe_sen=True, comp_weight=1.0),
    "b_pss_cw2": dict(B, pvisafe_sen=True, comp_weight=2.0),
    "b_pss_cw8": dict(B, pvisafe_sen=True, comp_weight=8.0),
    "b_pss_plean": dict(B, pvisafe_sen=True, lean_src="plean"),
    "b_pss_p4sen": dict(B, pvisafe_sen=True, v4_house_only=False),
    "b_pss_h28": dict(B, pvisafe_sen=True, v4_sfx="_h28"),
    "b_pss_cal": dict(B, pvisafe_sen=True, cal2="slope_r", cal2_w=0.75),
    "b_nps_psen": dict(B, pa_no_pvisafe=True, pvi_sen=True),
    "b_pss_psen": dict(B, pvisafe_sen=True, pvi_sen=True),
    "hyb": dict(B, pa_no_pvisafe=True, sen_cfg=B),
    "hyb_nu30": dict(B, pa_no_pvisafe=True, nu=30, sen_cfg=dict(B, nu=30)),
    "hyb_cw8": dict(B, pa_no_pvisafe=True, comp_weight=8.0, sen_cfg=B),
})
H = dict(B, pa_no_pvisafe=True, nu=30)
S = dict(B, nu=30)
VARIANTS.update({
    "h_cw4": dict(H, sen_cfg=S),
    "h_cw8": dict(H, comp_weight=8.0, sen_cfg=S),
    "h_cw16": dict(H, comp_weight=16.0, sen_cfg=S),
    "h_cw8_s8": dict(H, comp_weight=8.0, sen_cfg=dict(S, comp_weight=8.0)),
    "h_cw8_hfit": dict(H, comp_weight=8.0, fit_chamber=0, sen_cfg=S),
    "h_cw8_hfit_sfit": dict(H, comp_weight=8.0, fit_chamber=0, sen_cfg=dict(S, fit_chamber=1)),
    "h_cw8_plean": dict(H, comp_weight=8.0, lean_src="plean", sen_cfg=S),
    "h_cw8_h28": dict(H, comp_weight=8.0, v4_sfx="_h28", sen_cfg=S),
    "h_cw8_cal": dict(H, comp_weight=8.0, cal2="slope_r", cal2_w=0.75, sen_cfg=S),
    "h_cw8_scal": dict(H, comp_weight=8.0, sen_cfg=dict(S, cal2="slope_r", cal2_w=0.75)),
    "h_cw8_pres": dict(H, comp_weight=8.0, pres_lean=True, sen_cfg=S),
})


def main(names, show_only=False):
    """One variant per process (the sandbox blocks joblib's pool): `evaluate4.py NAME` runs and saves
    work/ev4/NAME.json; `evaluate4.py --show [NAMES]` prints the saved table."""
    (WORK / "ev4").mkdir(exist_ok=True)
    out = []
    for n in names:
        f = WORK / "ev4" / f"{n}.json"
        if show_only:
            if f.exists():
                out.append((n, json.load(open(f))))
            continue
        hist = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
        n, r = run(n, VARIANTS[n], hist)
        json.dump(r, open(f, "w"), indent=1)
        out.append((n, r))
    rows = []
    for n, r in out:
        rows.append(dict(variant=n, **{f"{s}_{k}": r[s][k] for s in r for k in ("c3", "all", "rated", "sen") if k in r[s]}))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    for n, r in out:
        print(n, {s: r[s]["per_year"] for s in r})


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--show":
        main(a[1:] or list(VARIANTS), show_only=True)
    else:
        main(a or list(VARIANTS))
