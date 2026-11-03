"""pau-x: 2026 race probabilities with the primary add-on (v6+nl + prim_dev, House races), trained on all 8 cycles,
vs v6+nl and Kalshi (k_pD from pau-r's pr_nl_2026.csv: Kalshi race markets pulled 2026-11-01, before the cut-off).
prim_dev 2026 = 2026 primary margin (px_prim2026.py) - 2 * lean, clipped +-40, centred over 2026's districts."""
import numpy as np, pandas as pd
import px_harness as H
import px_primaries as P
from model import WinModel
from common import WORK

_, f = P.build(H.HIST)
g = pd.read_csv(WORK / "px" / "primaries_2026.csv")
f26 = pd.read_csv(WORK / "features_2026.csv", low_memory=False)
std = f26[f26.kind.isin(["D_vs_R", "R_vs_IND", "jungle_party"])].copy()
std["year"] = 2026
house = std[std.senate == 0] if "senate" in std else std[~std.race_id.str.contains("SEN")]
def rid(s, d):
    al = f"{s}-AL"
    return al if (std.race_id == al).any() else f"{s}-{int(d):02d}"
g["race_id"] = [rid(s, d) for s, d in zip(g.state, g.dist)]
g = g.merge(house[["race_id", "lean"]], on="race_id", how="inner")
both = (g.dv > 0) & (g.rv > 0)
g = g[both & ~g.state.isin(["LA"])].copy()
g["raw"] = np.clip(100 * (g.dv - g.rv) / (g.dv + g.rv) - 2 * g.lean, -40, 40)
print("2026 districts with prim_dev:", len(g), "| corr(prim margin, lean):",
      round(np.corrcoef(100 * (g.dv - g.rv) / (g.dv + g.rv), g.lean)[0, 1], 3))
v26 = pd.Series((g.raw - g.raw.mean()).values, index=pd.MultiIndex.from_arrays([[2026] * len(g), g.race_id.values]))
s = pd.concat([f["prim_dev"], v26])
H.register("prim_dev", s)
Y = list(range(2010, 2026, 2))
cfg = H.PE.all_(H.BASE, px_feats=["prim_dev"], px_where="house", px_lam=10.0)
p_add = WinModel(Y, cfg).fit(H.HIST).predict(std)
p_nl = WinModel(Y, H.BASE).fit(H.HIST).predict(std)
d = pd.DataFrame(dict(race_id=std.race_id.values, p_nl=p_nl.p_dside.values, p_prim=p_add.p_dside.values))
k = pd.read_csv(WORK / "pr_nl_2026.csv")[["race_id", "chamber", "rating", "k_pD"]]
d = d.merge(k, on="race_id", how="left")
d["prim_dev"] = d.race_id.map(dict(zip(g.race_id, g.raw - g.raw.mean())))
d["dp"] = d.p_prim - d.p_nl
d.to_csv(WORK / "px" / "px_prim_2026.csv", index=False)
d.round(3).to_csv(H.WORK.parent.parent / "results" / "px" / "px_prim_2026.csv", index=False)
print(d.reindex(d.dp.abs().sort_values(ascending=False).index).head(14).round(3).to_string(index=False))
fav = lambda p: (p >= 0.5).astype(int)
ok = d.k_pD.notna()
h = d.chamber == "House"
print("favourite differs from Kalshi: v6+nl", int((fav(d.p_nl[ok]) != fav(d.k_pD[ok])).sum()),
      "| + prim_dev", int((fav(d.p_prim[ok]) != fav(d.k_pD[ok])).sum()),
      "| expected D-side House wins: v6+nl", round(d.p_nl[h].sum(), 1), "+ prim_dev", round(d.p_prim[h].sum(), 1))
print("mean |p - Kalshi| (House, matched): v6+nl", round((d.p_nl - d.k_pD)[h & ok].abs().mean(), 4),
      "+ prim_dev", round((d.p_prim - d.k_pD)[h & ok].abs().mean(), 4))
print("k:", [x for x in H.LOGK if "prim_dev" in x][-3:])
