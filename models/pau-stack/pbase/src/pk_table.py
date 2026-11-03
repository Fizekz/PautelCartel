"""pau-k: markdown tables for RESEARCH_STACK.md from results/pk/*.json and the saved six predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2] / "results" / "pk"
W = Path(__file__).resolve().parents[1] / "work" / "stack"
S4 = ("six", "pre18", "bt", "loyo8")


def j(t):
    return json.load(open(R / f"{t}.json"))


def row(label, t, ref=None, decision=""):
    d = j(t)
    cells = [f"{d[s]['all']:.4f}" for s in S4]
    if ref:
        r = j(ref)
        cells += [f"{d['six']['all'] - r['six']['all']:+.4f} / {d['pre18']['all'] - r['pre18']['all']:+.4f}"]
    else:
        cells += [""]
    return f"| {label} | " + " | ".join(cells) + f" | {decision} |"


def sub(label, t):
    d = j(t)
    return f"| {label} | " + " | ".join(f"{d[s]['rated']:.3f} / {d[s]['unp_rated']:.3f} / {d[s]['senate']:.3f}" for s in S4) + " |"


def nats(t):
    p = pd.read_csv(W / f"{t}_six.csv")
    pw = np.clip(np.where(p.dwin == 1, p.p_dside, 1 - p.p_dside), 0.01, 1)
    p["nll"] = -np.log(pw)
    pol = p.poll_pollsters.fillna(0) > 0
    rated = p.rating.abs() < 2.5
    h = p.senate == 0
    g = {"House rated, polled": h & rated & pol, "House rated, unpolled": h & rated & ~pol, "House safe": h & ~rated,
         "Senate": p.senate == 1}
    return {k: (int(m.sum()), round(float(p.nll[m].sum()), 1)) for k, m in g.items()}


if __name__ == "__main__":
    print(nats("base"), nats("base__clip2"), nats("prim+node+geo+field+rchg__clip2"))
