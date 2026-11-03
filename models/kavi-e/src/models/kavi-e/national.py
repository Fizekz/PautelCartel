"""National environment estimate (House popular-vote margin, D-R points) for a target year.

Four signals, each turned into an implied national House margin, bias-corrected with errors from years BEFORE the
target only, and combined by inverse-variance weights:

  gb   final generic-ballot average (national_hist for <=2024; the 2026-10-30 aggregator mean for 2026)
  sp   D-vs-R special elections since Jan 1 of the year before (Downballot sheets): national presidential margin of
       the baseline election + 10%-trimmed mean overperformance vs that presidential margin
  gov  VA + NJ governor races of the previous year: national pres margin + mean overperformance vs pres in-state
  fund president's-party midterm penalty, adjusted for net approval (midterms only)

Errors of a signal for an earlier year e = implied - actual House margin. bias = mean(e) * n / (n + K_SHRINK);
sd^2 = (K_SHRINK * prior_sd^2 + sum((e - bias)^2)) / (K_SHRINK + n).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from common import CLEAN, ELECTION_DAY, MIDTERMS, OUT, ROOT, ensure_dirs

K_SHRINK = 2.0
PRIOR_SD = {"gb": 3.0, "sp": 4.0, "gov": 6.0, "fund": 4.5}
SIGMA_N_FLOOR = 2.5  # signals' errors are correlated, so the combined sd is floored
FUND_SLOPE = 0.15  # pres-party margin per point of net approval (prior, not fitted: 4 midterms give no signal)
TRIM = 0.10
NAT_PRES_RAW_MARGIN = {2016: 2.1, 2020: 4.5, 2024: -1.5}
SPECIALS = ROOT / "data" / "specials" / "specials_clean.csv"
GOVS = ROOT / "data" / "specials" / "offyear_governors.csv"


@dataclass(frozen=True)
class Signal:
    name: str
    raw: float
    bias: float
    sd: float
    n_hist: int

    @property
    def adjusted(self) -> float:
        return self.raw - self.bias


def national_hist() -> pd.DataFrame:
    return pd.read_csv(CLEAN / "national_hist.csv").set_index("year")


def actual_margin(year: int) -> float:
    return float(national_hist().loc[year, "house_margin_dr"])


def gb_raw(year: int) -> float:
    if year == 2026:
        d = pd.read_csv(CLEAN / "national_2026.csv")
        avg = d[d["metric"] == "generic_ballot_avg"]
        return float((avg["value_dem"] - avg["value_rep"]).mean())
    return float(national_hist().loc[year, "generic_margin_dr"])


def approval_net(year: int) -> float:
    if year == 2026:
        d = pd.read_csv(CLEAN / "national_2026.csv")
        app = d.loc[d["metric"] == "approve_avg", "value"].mean()
        dis = d.loc[d["metric"] == "disapprove_avg", "value"].mean()
        return float(app - dis)
    return float(national_hist().loc[year, "net_approval"])


def pres_party(year: int) -> str:
    return {2010: "D", 2012: "D", 2014: "D", 2016: "D", 2018: "R", 2020: "R", 2022: "D", 2024: "D",
            2026: "R"}[year]


def sp_raw(year: int) -> float | None:
    sp = pd.read_csv(SPECIALS)
    sp = sp[sp["included"]]
    start, end = f"{year - 1}-01-01", ELECTION_DAY[year]
    window = sp[(sp["date"] >= start) & (sp["date"] < end)]
    if len(window) < 10:
        return None
    base = int(window["pres_base_year"].mode().iloc[0])
    return NAT_PRES_RAW_MARGIN[base] + float(trim_mean(window["overperf"], TRIM))


def gov_raw(year: int) -> float | None:
    g = pd.read_csv(GOVS)
    g = g[g["target_year"] == year]
    if g.empty:
        return None
    over = (g["dem_pct"] - g["rep_pct"]) - (g["pres_dem_pct"] - g["pres_rep_pct"])
    return float(g["nat_pres_margin"].iloc[0] + over.mean())


def fund_raw(year: int) -> float | None:
    """D-R margin implied by the prior midterms' average pres-party margin, shifted by relative net approval."""
    if year not in MIDTERMS:
        return None
    nh = national_hist()
    prior = [y for y in MIDTERMS if y < year and y in nh.index]
    if not prior:
        return None
    sign = {y: (1 if pres_party(y) == "D" else -1) for y in prior + [year]}
    pp_margin = np.mean([nh.loc[y, "house_margin_dr"] * sign[y] for y in prior])
    net_mean = np.mean([nh.loc[y, "net_approval"] for y in prior])
    pp_pred = pp_margin + FUND_SLOPE * (approval_net(year) - net_mean)
    return float(pp_pred * sign[year])


RAW_FUNCS = {"gb": gb_raw, "sp": sp_raw, "gov": gov_raw, "fund": fund_raw}


def history_years(name: str, year: int) -> list[int]:
    """Earlier years usable to calibrate a signal; midterm targets use midterm history for sp/gov/fund."""
    nh = national_hist()
    years = [y for y in nh.index if y < year]
    if name in ("gov", "fund", "sp") and year in MIDTERMS:
        years = [y for y in years if y in MIDTERMS]
    return years


def signal(name: str, year: int) -> Signal | None:
    raw = RAW_FUNCS[name](year)
    if raw is None:
        return None
    errs = []
    for y in history_years(name, year):
        r = RAW_FUNCS[name](y)
        if r is not None:
            errs.append(r - actual_margin(y))
    n = len(errs)
    bias = float(np.mean(errs) * n / (n + K_SHRINK)) if n else 0.0
    ss = sum((e - bias) ** 2 for e in errs)
    sd = float(np.sqrt((K_SHRINK * PRIOR_SD[name] ** 2 + ss) / (K_SHRINK + n)))
    return Signal(name, raw, bias, sd, n)


def estimate(year: int, use: tuple[str, ...] = ("gb", "sp", "gov", "fund")) -> dict:
    sigs = [s for s in (signal(n, year) for n in use) if s is not None]
    w = np.array([1 / s.sd ** 2 for s in sigs])
    mean = float(np.sum(w * np.array([s.adjusted for s in sigs])) / w.sum())
    sd = float(max(np.sqrt(1 / w.sum()), SIGMA_N_FLOOR))
    return {
        "year": year, "n_hat": mean, "sigma_n": sd,
        "signals": {s.name: {"raw": round(s.raw, 2), "bias": round(s.bias, 2), "adjusted": round(s.adjusted, 2),
                             "sd": round(s.sd, 2), "n_hist": s.n_hist, "weight": round(float(1 / s.sd ** 2 / w.sum()), 3)}
                    for s in sigs},
        "actual": actual_margin(year) if year <= 2024 else None,
    }


def main() -> int:
    ensure_dirs()
    out = {str(y): estimate(y) for y in (2010, 2014, 2018, 2022, 2026)}
    (OUT / "national_env.json").write_text(json.dumps(out, indent=2))
    for y, e in out.items():
        sig = "; ".join(f"{k} {v['raw']:+.1f}->{v['adjusted']:+.1f} (sd {v['sd']:.1f}, w {v['weight']:.2f})"
                        for k, v in e["signals"].items())
        act = "" if e["actual"] is None else f" actual {e['actual']:+.2f}"
        print(f"{y}: N_hat {e['n_hat']:+.2f} sd {e['sigma_n']:.2f}{act} | {sig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
