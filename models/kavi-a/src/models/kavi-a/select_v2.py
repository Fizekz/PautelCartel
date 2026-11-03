"""v2 design selection, on contested D-v-R rows of the historical table (not the official scorecard).

A. Calibration spec x CAL_WEIGHT: two-level leave-one-year-out on 2010-2016 only (first level: share model fit
   on the other years of the window; second level: calibration fit on the other years' out-of-sample rows).
   The same table over 2010-2024 is printed for information only (it sees 2018/2022).
B. Poll window mode: forward tests 2020 (fit 2010-2018) and 2024 (fit 2010-2022); 60-day polls are all that
   exists before 2018, so pre-2018 LOYO cannot tell the modes apart. 2018/2022 forward tests printed, info only.
C. Quantitative-outlet rating blend r = core + lam (quant - core) at prediction time: forward tests 2020/2024
   (quant = DDHQ/Economist/Split Ticket/538 categories where present).
Usage: python select_v2.py [A] [B] [C]     (writes out/select_v2.json)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import calib
import model
from common import OUT
from features import POLL_MODE, POLL_MODES, trainable

WEIGHTS = (0.5, 0.75, 1.0)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def logloss(q: np.ndarray, won: np.ndarray) -> float:
    q = np.clip(np.asarray(q, dtype=float), 1e-3, 1 - 1e-3)
    return float(-np.mean(np.where(won == 1, np.log(q), np.log(1 - q))))


def second_level(frame: pd.DataFrame, spec: str, weight: float) -> dict:
    out = {}
    for held in sorted(frame["year"].unique()):
        tr, te = frame[frame["year"] != held], frame[frame["year"] == held]
        m = calib.fit_on_frame(tr, spec)
        cal = m.predict_proba(calib.design(te, spec))[:, 1]
        q = 1 / (1 + np.exp(-(weight * calib.logit(cal) + (1 - weight) * calib.logit(te["q"]))))
        out[int(held)] = logloss(q, te["won"].to_numpy())
    out["mean"] = float(np.mean(list(out.values())))
    return out


def part_a(hist: pd.DataFrame) -> dict:
    res = {}
    for label, years in (("pre2018", list(range(2010, 2017, 2))), ("all_info", list(range(2010, 2025, 2)))):
        frame = calib.loyo_frame(hist, years, True, True)
        res[label] = {"uncalibrated": {int(y): logloss(g["q"], g["won"].to_numpy())
                                       for y, g in frame.groupby("year")}}
        for spec in calib.CALIB_SPECS:
            for w in WEIGHTS:
                res[label][f"{spec}_w{w}"] = second_level(frame, spec, w)
        coef = calib.fit_on_frame(frame, "mo").coef_[0]
        res[label]["mo_coef_full_window"] = [round(float(c), 3) for c in coef]
        print(label)
        print(pd.DataFrame({k: v for k, v in res[label].items() if isinstance(v, dict)}).T.round(4).to_string())
    return res


def with_poll_mode(hist: pd.DataFrame, mode: str) -> pd.DataFrame:
    h = hist.copy()
    for c in ("poll_share", "n_polls", "poll_weight"):
        h[c] = h[f"{c}_{mode}"]
    return h


def forward(hist: pd.DataFrame, test_year: int, lam: float = 0.0) -> dict:
    fits = model.fit_all(hist, list(range(2010, test_year, 2)), True, True, calibrate=True)
    test = hist[trainable(hist) & (hist["year"] == test_year)].copy()
    has_q = test["rating_quant"].notna()
    test.loc[has_q, "rating"] = test.loc[has_q, "rating"] + lam * (test.loc[has_q, "rating_quant"]
                                                                   - test.loc[has_q, "rating"])
    qs, won, quant = [], [], []
    for ch, fit in fits.items():
        rows = test[test["chamber"] == ch]
        pr = model.predict_chamber(fit, rows, rows["g"].iloc[0])
        qs.append(model.win_prob(pr["mu"], pr["sd_nat"], pr["sd_state"], pr["sd_race"]))
        won.append((rows["dem_share_2p"] > 0.5).astype(int).to_numpy())
        quant.append(rows["rating_quant"].notna().to_numpy())
    q, w, hq = np.concatenate(qs), np.concatenate(won), np.concatenate(quant)
    return {"all": logloss(q, w), "quant_rows": logloss(q[hq], w[hq]), "n": int(len(q)), "n_quant": int(hq.sum())}


def part_b(hist: pd.DataFrame) -> dict:
    res = {}
    for mode in POLL_MODES:
        h = with_poll_mode(hist, mode)
        res[mode] = {y: forward(h, y)["all"] for y in (2020, 2024, 2018, 2022)}
        res[mode]["select_2020_2024"] = res[mode][2020] + res[mode][2024]
    print(pd.DataFrame(res).T.round(4).to_string())
    return res


def part_c(hist: pd.DataFrame) -> dict:
    res = {}
    for lam in LAMBDAS:
        res[lam] = {}
        for y in (2020, 2024):
            r = forward(hist, y, lam)
            res[lam][f"{y}_all"], res[lam][f"{y}_quant_rows"] = r["all"], r["quant_rows"]
        res[lam]["select_sum_all"] = res[lam]["2020_all"] + res[lam]["2024_all"]
    print(pd.DataFrame(res).T.round(4).to_string())
    return res


def main() -> None:
    parts = set(sys.argv[1:]) or {"A", "B", "C"}
    hist = pd.read_csv(OUT / "hist_features.csv")
    path = OUT / "select_v2.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if "A" in parts:
        res["A_calibration"] = part_a(hist)
    if "B" in parts:
        res["B_poll_mode"] = part_b(hist)
    if "C" in parts:
        res["C_quant_blend"] = {str(k): v for k, v in part_c(with_poll_mode(hist, POLL_MODE)).items()}
    res["settings"] = {"CAL_SPEC": calib.CAL_SPEC, "CAL_WEIGHT": model.CAL_WEIGHT, "POLL_MODE": POLL_MODE}
    path.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
