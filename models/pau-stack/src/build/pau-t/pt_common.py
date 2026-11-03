"""pau-t: shared setup. v6+nl configuration (pr_theta_nl2.py lines 8-11), scheme runner and metrics."""
import json, time
import numpy as np, pandas as pd
import evaluate4 as E
import pr_eval as PE
import pr_pattern as PP
from common import WORK
from evaluate import metrics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "pt"
RES.mkdir(parents=True, exist_ok=True)
Z = PP.Z
for a, b in (("college", "college"), ("hispanic", "hispanic"), ("black", "black"), ("college", "hispanic")):
    v = Z[a] * Z[b]; Z[f"{a}_x_{b}"] = (v - v.mean()) / v.std()
PP.SETS["nl"] = ["college", "hispanic", "black", "college_x_college", "hispanic_x_hispanic", "black_x_black",
                 "college_x_hispanic"]
HIST = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
BASE = PE.all_(PE.V["v51"], pr_pat_lam=1.0, pr_pat_set="nl", pr_pat_unp=True)


def cfg(**kw):
    return PE.all_(BASE, **kw)


def subsets(pr):
    pol = pr.poll_pollsters.fillna(0) > 0
    rated = pr.rating.abs() < 2.5
    return {"all": pr, "rated": pr[rated], "unp_rated": pr[(~pol) & rated & (pr.senate == 0)],
            "pol_rated": pr[pol & rated], "senate": pr[pr.senate == 1]}


def ll(pr):
    return round(float(metrics(pr.p_dside, pr.dwin)["ll"]), 4) if len(pr) else None


def run(tag, c, schemes=("six", "pre18", "bt", "loyo8"), save=True):
    out, t0 = {}, time.time()
    for sch in schemes:
        pr = pd.concat([E.one(HIST, c, T, tr) for T, tr in E.splits(sch)])
        out[sch] = {k: ll(v) for k, v in subsets(pr).items()}
        out[sch]["n_unp_rated"] = int(len(subsets(pr)["unp_rated"]))
        out[sch]["per_year"] = {int(y): ll(g) for y, g in pr.groupby("year")}
        if save:
            pr.to_csv(WORK / f"pt_{tag}_{sch}.csv", index=False)
    out["secs"] = round(time.time() - t0)
    json.dump(out, open(RES / f"{tag}.json", "w"), indent=1)
    print(tag, {s: (out[s]["all"], out[s]["rated"], out[s]["unp_rated"]) for s in schemes}, out["secs"], flush=True)
    return out
