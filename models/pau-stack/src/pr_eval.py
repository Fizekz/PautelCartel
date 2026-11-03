"""pau-r experiment runner on top of pau-b v5.1's harness (evaluate4.run). Usage: pr_eval.py NAME [NAME...]"""
import json, sys
import numpy as np, pandas as pd
import evaluate4 as E
import pr_year  # noqa: F401 (installs the year-level hook)
import pr_sgap  # noqa: F401
import pr_lev  # noqa: F401
from model import FINAL
from common import WORK


def all_(cfg, **kw):
    c = dict(cfg, **kw)
    for k in ("sen_cfg", "pool_cfg"):
        if k in c:
            c[k] = dict(c[k], **kw)
    return c


V = {
    "v51": FINAL,
    "oracle_ye": all_(FINAL, oracle_ye=1.0),
    "oracle_ye50": all_(FINAL, oracle_ye=0.5),
    "ye_base": all_(FINAL, pr_ye="base"),
    "ye_gbsep": all_(FINAL, pr_ye="gbsep"),
    "ye_gbsep_l10": all_(FINAL, pr_ye="gbsep", pr_lam=10.0),
    "ye_gbsep_l100": all_(FINAL, pr_ye="gbsep", pr_lam=100.0),
    "ye_hfe": all_(FINAL, pr_ye="base", pr_house_fe=True),
    "ye_old0608": all_(FINAL, pr_ye="old", pr_old_years=[2006, 2008]),
    "ye_old06": all_(FINAL, pr_ye="old", pr_old_years=[2006]),
    "ye_old0608_w05": all_(FINAL, pr_ye="old", pr_old_years=[2006, 2008], pr_old_w=0.5),
    "ye_old0208": all_(FINAL, pr_ye="old", pr_old_years=[2002, 2004, 2006, 2008]),
    "ye_old0208_hfe": all_(FINAL, pr_ye="old", pr_old_years=[2002, 2004, 2006, 2008], pr_house_fe=True),
    "ye_old0608_gbsep10": all_(FINAL, pr_ye="old", pr_old_years=[2006, 2008], pr_gbsep=True, pr_lam=10.0),
    "ye_old0208_gbsep10": all_(FINAL, pr_ye="old", pr_old_years=[2002, 2004, 2006, 2008], pr_gbsep=True, pr_lam=10.0),
    "ye_old0208_gbsep": all_(FINAL, pr_ye="old", pr_old_years=[2002, 2004, 2006, 2008], pr_gbsep=True),
    "ye_old0206": all_(FINAL, pr_ye="old", pr_old_years=[2002, 2006]),
    "ye_old0208_midicpt": all_(FINAL, pr_ye="old", pr_old_years=[2002, 2004, 2006, 2008], pr_mid_icpt=True),
    "ye_midicpt": all_(FINAL, pr_ye="base", pr_mid_icpt=True),
    "sgap": all_(FINAL, pr_sgap=True),
    "sgap_l100": all_(FINAL, pr_sgap=True, pr_sgap_lam=100.0),
    "sgap_l1000": all_(FINAL, pr_sgap=True, pr_sgap_lam=1000.0),
    "lev1": all_(FINAL, pr_lev=1.0),
    "lev05": all_(FINAL, pr_lev=0.5),
    "lev2": all_(FINAL, pr_lev=2.0),
    "ye_old0608_hfe": all_(FINAL, pr_ye="old", pr_old_years=[2006, 2008], pr_house_fe=True),
}
EXTRA = {}
for _f in ("", "shift", "slope", "blend50", "outcome"):
    for _l in (10, 100, 1000, 100000):
        _k = f"q4{_f or 'raw'}_l{_l}"
        _kw = dict(pr_tfile=f"teacher_final_{_f}.csv") if _f else {}
        EXTRA[_k] = all_(dict(FINAL, q4=float(_l), q4_all=True), **_kw)
EXTRA["q4v51_base"] = FINAL


def main(names, schemes=("six", "pre18", "bt", "loyo8")):
    if names and names[0].startswith("q4"):
        schemes = ("q4", "q4bt")
    rows = []
    hist = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
    for n in names:
        cfg = V.get(n) or EXTRA[n]
        _, r = E.run(n, cfg, hist, schemes=schemes)
        (WORK / "pr").mkdir(exist_ok=True)
        json.dump(r, open(WORK / "pr" / f"{n}.json", "w"), indent=1)
        rows.append(dict(variant=n, **{f"{s}": round(r[s]["all"], 4) for s in r},
                         **{f"{s}_rated": round(r[s]["rated"], 4) for s in r}))
        print(n, {s: r[s]["per_year"] for s in r})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
