"""pau-w: harness runs of the assimilation filter. Usage: pw_run.py NAME (configs in CONFIGS)."""
import json
import sys

import numpy as np
import pandas as pd

import evaluate4 as E
import pw_assim as A
from common import WORK, ROOT
from evaluate import metrics

OUTD = ROOT.parent / "results"
OUTD.mkdir(exist_ok=True)


def evaluate(opts, schemes=("six", "pre18", "bt", "loyo8"), leads=(0,), tag=None, prm_kw=None, keep=False):
    res, frames = {}, {}
    for sch in schemes:
        fr, diags = [], []
        for T, tr in E.splits(sch):
            out, diag, scale, prm = A.fold(T, tr, opts, leads=list(leads), prm_kw=prm_kw)
            fr.append(out)
            diags.append(dict(year=T, n=round(float(diag["n"]), 2), b=round(float(diag["b"]), 2),
                              v=round(float(diag["v"]), 2), sd_b=round(float(diag["sd_b"]), 2),
                              chi2_poll=round(float(np.mean(diag["chi2_poll"])) if diag["chi2_poll"] else np.nan, 3),
                              scale={k: round(v, 2) for k, v in scale.items()}))
        d = pd.concat(fr)
        frames[sch] = d
        r = {}
        for L, g in d.groupby("lead"):
            rated = g.rating.abs() < 2.5
            unp = rated & ~g.polled & (g.senate == 0)
            r[int(L)] = dict(all=round(metrics(g.p, g.dwin)["ll"], 4), rated=round(metrics(g[rated].p, g[rated].dwin)["ll"], 4),
                             unp=round(metrics(g[unp].p, g[unp].dwin)["ll"], 4), n_unp=int(unp.sum()),
                             per_year={int(y): round(metrics(x.p, x.dwin)["ll"], 4) for y, x in g.groupby("year")})
        res[sch] = dict(leads=r, diag=diags)
    if tag:
        json.dump(res, open(OUTD / f"pw_{tag}.json", "w"), indent=1, default=float)
    return (res, frames) if keep else res


def show(res, lead=0):
    s = {sch: (r["leads"][lead]["all"], r["leads"][lead]["rated"], r["leads"][lead]["unp"]) for sch, r in res.items()}
    return " | ".join(f"{k} {v[0]:.4f} ({v[1]:.3f}/{v[2]:.3f})" for k, v in s.items())


CONFIGS = {
    "base": {},
    "k3": dict(K="k3"),
    "nopat": dict(pattern="none"),
    "bfree": dict(b_scale=3.0),
    "static": dict(tau=np.inf),
    "tau15": dict(tau=15.0),
    "tau120": dict(tau=120.0),
    "nostate": dict(state=False),
    "nogb": dict(use_gb=False),
    "nob": dict(b_scale=0.01),
    "cal2": dict(cal2=True),
    "bmean": dict(b_prior="mean"),
    "btype": dict(b_prior="type"),
    "qc3": dict(qc=3.0),
    "qn03": dict(q_n=0.3),
}
C2 = dict(cal2=True)
CONFIGS.update({
    "c2_static": dict(C2, tau=np.inf), "c2_tau15": dict(C2, tau=15.0), "c2_tau30": dict(C2, tau=30.0),
    "c2_nostate": dict(C2, state=False), "c2_nob": dict(C2, b_scale=0.01), "c2_btype": dict(C2, b_prior="type"),
    "c2_bmean": dict(C2, b_prior="mean"), "c2_k3": dict(C2, K="k3"), "c2_nopat": dict(C2, pattern="none"),
    "c2_nogb": dict(C2, use_gb=False), "c2_bfree": dict(C2, b_scale=3.0),
})

if __name__ == "__main__":
    if sys.argv[1] == "--leads":
        name = sys.argv[2]
        r, fr = evaluate(CONFIGS[name], leads=A.LEADS, tag=f"leads_{name}", keep=True)
        for sch, d in fr.items():
            d.to_csv(OUTD / f"pw_leads_{name}_{sch}.csv", index=False)
        print(name, show(r), flush=True)
        sys.exit()
    if sys.argv[1] == "--calall":
        A.CAL["mode"] = "all"
        sys.argv = sys.argv[1:]
    for name in sys.argv[1:]:
        r = evaluate(CONFIGS[name], schemes=("six", "pre18"), tag=name + ("" if A.CAL["mode"] == "comp" else "_calall"))
        print(name, show(r), flush=True)
