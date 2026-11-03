"""Held-out evaluation of model variants over 2010-2024 (model selection never looks at 538).

Two schemes, both nested (the WinModel's own inner LOYO only sees the training years):
  loyo    : for each year T in 2010..2024, train on the other 7 years, predict T
  forward : for each year T in 2014..2024, train on the years before T only
Scored on every contested D-side vs R race with a result (train_ok == 1). "rated" = races where
the mean rating is not Safe (|rating| < 2.5), "close" = Tossup/Lean (|rating| <= 1.5); both are
pre-election definitions. Log loss uses the contest's 1% floor.
"""
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import WORK
from model import WinModel, V1

YEARS = list(range(2010, 2026, 2))


def metrics(p, y):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    pw = np.clip(np.where(y == 1, p, 1 - p), 0.01, 1)
    return dict(n=len(p), ll=float(-np.mean(np.log(pw))), brier=float(np.mean((p - y) ** 2)),
                acc=float(np.mean((p > 0.5) == (y == 1))))


def run_scheme(hist, cfg, scheme):
    preds = []
    tests = YEARS if scheme == "loyo" else [y for y in YEARS if y >= 2014]
    for T in tests:
        train = [y for y in YEARS if y != T] if scheme == "loyo" else [y for y in YEARS if y < T]
        c = dict(cfg, poll_regime="loyo" if scheme == "loyo" else "fwd")
        wm = WinModel(train, c).fit(hist)
        te = hist[(hist.year == T) & (hist.train_ok == 1)]
        preds.append(wm.predict(te)[["year", "race_id", "chamber", "senate", "rating", "y", "dwin",
                                     "yhat", "mu", "sd", "p_dside", "poll_pollsters"]])
    return pd.concat(preds)


def summarise(pr):
    out = {}
    for name, sub in [("all", pr), ("rated", pr[pr.rating.abs() < 2.5]), ("close", pr[pr.rating.abs() <= 1.5]),
                      ("house_rated", pr[(pr.rating.abs() < 2.5) & (pr.senate == 0)]),
                      ("senate", pr[pr.senate == 1])]:
        per_year = {int(y): metrics(g.p_dside, g.dwin) for y, g in sub.groupby("year")}
        out[name] = dict(pooled=metrics(sub.p_dside, sub.dwin),
                         mean_year_ll=float(np.mean([v["ll"] for v in per_year.values()])),
                         per_year_ll={k: round(v["ll"], 4) for k, v in per_year.items()})
    comp = pr[pr.rating.abs() < 2.5]
    out["year_bias_rated"] = {int(y): round(float((g.y - g.mu).mean()), 2) for y, g in comp.groupby("year")}
    return out


def evaluate(name, cfg, hist):
    res = {}
    for scheme in ("loyo", "forward"):
        pr = run_scheme(hist, cfg, scheme)
        pr.to_csv(WORK / f"eval_{name}_{scheme}.csv", index=False)
        res[scheme] = summarise(pr)
    return name, res


V2 = dict(V1, two_level=True, year_vars=[["gbmid"]], comp_weight=3.0, epsilon=2.0)
VARIANTS = {
    "v2": V2,
    "v2_lmax": dict(V2, lam_max=True),
    "v2_gt": dict(V2, cal="group_t"),
    "v2_gt_mle": dict(V2, cal="group_t", sigma_mode="mle"),
    "v2_lmax_gt": dict(V2, lam_max=True, cal="group_t"),
    "v2_lmax_gt_mle": dict(V2, lam_max=True, cal="group_t", sigma_mode="mle"),
    # pau-a's stage A inside my pipeline (my polls)
    "pa1": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t"),
    "pa_tl": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t",
                  two_level=True, year_vars=[["gbmid"]]),
    "pa_tl_huber": dict(V1, design="pa", comp_weight=4.0, epsilon=2.0, lam_max=True, cal="group_t",
                        two_level=True, year_vars=[["gbmid"]]),
    "pa_tl_last": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t",
                       two_level=True, year_vars=[["gbmid"]], pa_last=True),
    "v2_gt_fixonly": dict(V2, lam_max=True, cal="group_t"),
    "pa_tl_polls": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t",
                        two_level=True, year_vars=[["gbmid"]], polls="pa"),
    "pa_tl_probit": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True,
                         two_level=True, year_vars=[["gbmid"]]),
    "pa_tl_nolmax": dict(V1, design="pa", fitter="wls", comp_weight=4.0, cal="group_t",
                         two_level=True, year_vars=[["gbmid"]]),
    "pa_tl_mle": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t",
                      sigma_mode="mle", two_level=True, year_vars=[["gbmid"]]),
    "pa_tl_nomoney": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t",
                          two_level=True, year_vars=[["gbmid"]], no_money=True),
    "pa_tl_moneycap2": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t",
                            two_level=True, year_vars=[["gbmid"]], money_cap=2.0),
    "pa_tl_moneyinc": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t",
                           two_level=True, year_vars=[["gbmid"]], money_inc_only=True),
    "pa1_polls": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t", polls="pa"),
    "pa_tl_nopvisafe": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t",
                            two_level=True, year_vars=[["gbmid"]], pa_no_pvisafe=True),
    "pa_tl_moneyall": dict(V1, design="pa", fitter="wls", comp_weight=4.0, lam_max=True, cal="group_t",
                           two_level=True, year_vars=[["gbmid"]], pa_money_all=True),
    "v2_pvisafe": dict(V2, lam_max=True, cal="group_t", extra=["pvi_safe"]),
    "v1": V1,
    "t5": dict(V1, link="t", nu=5),
    "outmid": dict(V1, extra=["outparty_mid"]),
    "outmid_comp": dict(V1, extra=["outparty_mid_comp"]),
    "incpres": dict(V1, extra=["inc_pres_mid"]),
    "gbcomp": dict(V1, extra=["gb_comp"]),
    "gbinc": dict(V1, extra=["gb_inc"]),
    "approval": dict(V1, extra=["approval"]),
    "raters": dict(V1, extra=["rating_raters"]),
    "ratfine": dict(V1, extra=["rating_fine"]),
    "compint": dict(V1, extra=["lean_comp", "inc_comp", "money_comp"]),
    "cw3": dict(V1, comp_weight=3.0),
    "cook": dict(V1, rating_src="cook"),
    "eps1": dict(V1, epsilon=1.2),
    "eps2": dict(V1, epsilon=2.0),
    # two-level: race-level coefficients within years + year-level model of the year effects
    "tl_none": dict(V1, two_level=True, year_vars=[[]]),
    "tl_gb": dict(V1, two_level=True, year_vars=[["gb"]]),
    "tl_om": dict(V1, two_level=True, year_vars=[["outmid"]]),
    "tl_gbom": dict(V1, two_level=True, year_vars=[["gb", "outmid"]]),
    "tl_gbom_r": dict(V1, two_level=True, year_vars=[["gb", "outmid"]], year_ridge=3.0),
    "tl_avg": dict(V1, two_level=True, year_vars=[["gb"], ["outmid"]]),
    "tl_avg_incpres": dict(V1, two_level=True, year_vars=[["gb"], ["outmid"]], extra=["inc_pres_mid"]),
    "tl_avg_gbinc": dict(V1, two_level=True, year_vars=[["gb"], ["outmid"]], extra=["gb_inc"]),
    "tl_om_incpres": dict(V1, two_level=True, year_vars=[["outmid"]], extra=["inc_pres_mid"]),
    "v1_incpres_eps2": dict(V1, extra=["inc_pres_mid"], epsilon=2.0),
    "tl_omgbmid_incpres": dict(V1, two_level=True, year_vars=[["outmid", "gbmid"]], extra=["inc_pres_mid"]),
    "tl_gbmid_incpres": dict(V1, two_level=True, year_vars=[["gbmid"]], extra=["inc_pres_mid"]),
    "tl_om_incpres_cw3": dict(V1, two_level=True, year_vars=[["outmid"]], extra=["inc_pres_mid"], comp_weight=3.0),
    "tl_om_incpres_eps2": dict(V1, two_level=True, year_vars=[["outmid"]], extra=["inc_pres_mid"], epsilon=2.0),
    "tl_om_incpres_bins": dict(V1, two_level=True, year_vars=[["outmid"]], extra=["inc_pres_mid"], sd_bins=True),
    "tl_om_incpres_t": dict(V1, two_level=True, year_vars=[["outmid"]], extra=["inc_pres_mid"], link="t", nu=6),
    "tl_om_incpres_raters": dict(V1, two_level=True, year_vars=[["outmid"]], extra=["inc_pres_mid", "rating_raters"]),
    "tl_gbmid_incpres_cw3": dict(V1, two_level=True, year_vars=[["gbmid"]], extra=["inc_pres_mid"], comp_weight=3.0),
    "tl_gbmid_incpres_cw2": dict(V1, two_level=True, year_vars=[["gbmid"]], extra=["inc_pres_mid"], comp_weight=2.0),
    "tl_gbmid_incpres_cw3_eps2": dict(V1, two_level=True, year_vars=[["gbmid"]], extra=["inc_pres_mid"], comp_weight=3.0, epsilon=2.0),
    "tl_gbmid": dict(V1, two_level=True, year_vars=[["gbmid"]]),
    "tl_gbmid_cw3_eps2": dict(V1, two_level=True, year_vars=[["gbmid"]], comp_weight=3.0, epsilon=2.0),
    "tl_gb_cw3_eps2": dict(V1, two_level=True, year_vars=[["gb"]], comp_weight=3.0, epsilon=2.0),
    "tl_om_cw3_eps2": dict(V1, two_level=True, year_vars=[["outmid"]], comp_weight=3.0, epsilon=2.0),
    "tl_none_cw3_eps2": dict(V1, two_level=True, year_vars=[[]], comp_weight=3.0, epsilon=2.0),
    "tl_om_incpres_gbinc": dict(V1, two_level=True, year_vars=[["outmid"]], extra=["inc_pres_mid", "gb_inc"]),
}


def table(results):
    rows = []
    for name, res in results.items():
        for scheme in ("loyo", "forward"):
            r = res[scheme]
            rows.append(dict(variant=name, scheme=scheme,
                             ll_all=r["all"]["pooled"]["ll"], ll_rated=r["rated"]["pooled"]["ll"],
                             ll_close=r["close"]["pooled"]["ll"], brier_rated=r["rated"]["pooled"]["brier"],
                             acc_rated=r["rated"]["pooled"]["acc"], ll_sen=r["senate"]["pooled"]["ll"],
                             myll_rated=r["rated"]["mean_year_ll"]))
    return pd.DataFrame(rows)


def main(names=None):
    hist = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
    names = names or list(VARIANTS)
    out = Parallel(n_jobs=-1)(delayed(evaluate)(n, VARIANTS[n], hist) for n in names)
    results = dict(out)
    prev = {}
    try:
        prev = json.load(open(WORK / "eval_results.json"))
    except FileNotFoundError:
        pass
    prev.update(results)
    json.dump(prev, open(WORK / "eval_results.json", "w"), indent=1)
    t = table(results)
    pd.set_option("display.width", 200)
    for scheme in ("loyo", "forward"):
        print(f"--- {scheme}")
        print(t[t.scheme == scheme].drop(columns="scheme").round(4).to_string(index=False))
    for n in names:
        print(n, "year bias (rated, loyo):", results[n]["loyo"]["year_bias_rated"])
    return results


if __name__ == "__main__":
    main(sys.argv[1:] or None)
