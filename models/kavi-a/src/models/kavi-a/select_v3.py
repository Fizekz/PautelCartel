"""v3 design selection (contested D-v-R rows of the historical table; never the official scorecard).

Options (model.py switches):
  PREV_MODE  : previous-House-result features "full" (v2) / "inc_only" (only the same-incumbent interaction) /
               "none" (kavi-e's v1 finding that prev features hurt; re-tested here in my own model)
  SPONSOR_SCALE (sponsor.py): 0 (v2) / 0.5 / 1: shift D- and R-sponsored polls by their house effect estimated on
               the training years (pau-a v3's sponsor de-biasing, adapted)
  STAGE2_WLS : 0 (v2, OLS) or > 0: stage 2 weighted toward races the fundamentals call close (pau-a uses a
               weighted least squares fit; my calibration layer exists because OLS on shares is dominated by safe
               seats, so this attacks the same problem one step earlier)
  calibration spec (calib.V3_SPECS): "inc" adds incumbency (+1 D, -1 R) to the logistic layer, "poll" a clipped
               poll lead x poll-weight saturation, "inc_poll" both (--calib; prev full, no WLS, no sponsor shift)
Criteria, in order:
  1. pre-2018: two-level leave-one-year-out on 2010-2016 (share model on 3 years, calibration LOYO on the
     4 held-out frames), base spec, CAL_WEIGHT 0.75: mean held-out contested log loss.
  2. forward 2020 (fit 2010-2018) + 2024 (fit 2010-2022), calibrated: tie-breaker, sees post-2018 data.
2018/2022 are not computed here.
Usage: python select_v3.py [--quick] [--sponsor]   (writes out/select_v3.json; --sponsor: base + sponsor rows only)
"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

import calib
import model
import sponsor
from common import OUT
from features import trainable
from select_v2 import forward, logloss, second_level

PREV_MODES = ("full", "inc_only", "none")
WLS_GRID = (0.0, 2.0, 5.0)
SPONSOR_GRID = (0.5, 1.0)
PRE_YEARS = list(range(2010, 2017, 2))
CONFIGS = ([(prev, wls, 0.0) for prev, wls in itertools.product(PREV_MODES, WLS_GRID)]
           + [("full", 0.0, s) for s in SPONSOR_GRID])


def set_config(prev: str, wls: float, scale: float) -> None:
    model.PREV_MODE = prev
    model.STAGE2_WLS = wls
    sponsor.SPONSOR_SCALE = scale


def evaluate(hist: pd.DataFrame, with_forward: bool) -> dict:
    frame = calib.loyo_frame(hist, PRE_YEARS, True, True)
    res = {"pre2018_uncal": {int(y): logloss(g["q"], g["won"].to_numpy()) for y, g in frame.groupby("year")}}
    res["pre2018_uncal"]["mean"] = float(np.mean(list(res["pre2018_uncal"].values())))
    res["pre2018_cal"] = second_level(frame, "base", model.CAL_WEIGHT)
    if with_forward:
        res["fwd"] = {y: forward(hist, y)["all"] for y in (2020, 2024)}
        res["fwd"]["sum"] = res["fwd"][2020] + res["fwd"][2024]
    return res


def calib_specs(hist: pd.DataFrame) -> dict:
    set_config("full", 0.0, 0.0)
    frame = calib.loyo_frame(hist, PRE_YEARS, True, True)
    out = {}
    for spec in ("base",) + calib.V3_SPECS:
        calib.CAL_SPEC = spec
        r = {f"pre2018_cal_w{w}": second_level(frame, spec, w) for w in (0.75, 1.0)}
        r["fwd"] = {y: forward(hist, y)["all"] for y in (2020, 2024)}
        coef = calib.fit_on_frame(frame, spec).coef_[0]
        r["coef_pre2018"] = [round(float(c), 3) for c in coef]
        out[spec] = r
        print(f"{spec:10s} pre-cal w.75 {r['pre2018_cal_w0.75']['mean']:.5f} w1 {r['pre2018_cal_w1.0']['mean']:.5f}"
              f" fwd2020 {r['fwd'][2020]:.5f} fwd2024 {r['fwd'][2024]:.5f} coef {r['coef_pre2018']}", flush=True)
    calib.CAL_SPEC = "base"
    return out


def gclose_test(hist: pd.DataFrame) -> dict:
    set_config("full", 0.0, 0.0)
    out = {}
    for flag, scale in ((False, 0.1), (True, 0.05), (True, 0.1), (True, 0.2)):
        model.GCLOSE, model.GCLOSE_SCALE = flag, scale
        r = evaluate(hist, True)
        out[f"gclose={flag},scale={scale}"] = r
        print(f"gclose={flag} scale={scale}: pre-uncal {r['pre2018_uncal']['mean']:.5f} pre-cal "
              f"{r['pre2018_cal']['mean']:.5f} fwd2020 {r['fwd'][2020]:.5f} fwd2024 {r['fwd'][2024]:.5f}", flush=True)
    model.GCLOSE, model.GCLOSE_SCALE = False, 0.1
    return out


def pollster_test() -> dict:
    import features
    set_config("full", 0.0, 0.0)
    out = {}
    for flag in (False, True):
        features.POLLSTER_DAMP = flag
        r = evaluate(features.build_history(), True)
        out[f"pollster_damp={flag}"] = r
        print(f"pollster_damp={flag}: pre-uncal {r['pre2018_uncal']['mean']:.5f} pre-cal "
              f"{r['pre2018_cal']['mean']:.5f} fwd2020 {r['fwd'][2020]:.5f} fwd2024 {r['fwd'][2024]:.5f}", flush=True)
    features.POLLSTER_DAMP = False
    return out


def dedup_test() -> dict:
    """Forward 2020/2024 only: before 2018 the 538 files hold one question per poll, so LOYO cannot see this."""
    import features
    set_config("full", 0.0, 0.0)
    out = {}
    for mode in ("none", "lv", "avg"):
        features.POLL_DEDUP = mode
        h = features.build_history()
        out[mode] = {y: forward(h, y)["all"] for y in (2020, 2024)}
        print(f"poll_dedup={mode}: fwd2020 {out[mode][2020]:.5f} fwd2024 {out[mode][2024]:.5f}", flush=True)
    features.POLL_DEDUP = "none"
    return out


def rclip_test(hist: pd.DataFrame) -> dict:
    """Calibration-layer rating clip R_CLIP (v2 1.5; ratings run -3..3, so 3 = no clip) x CAL_WEIGHT."""
    set_config("full", 0.0, 0.0)
    frame = calib.loyo_frame(hist, PRE_YEARS, True, True)  # uncalibrated frame does not depend on the clip
    out, keep_w = {}, model.CAL_WEIGHT
    for rc in (1.0, 1.5, 2.0, 3.0):
        calib.R_CLIP = rc
        for w in (0.5, 0.75, 1.0):
            model.CAL_WEIGHT = w
            r = {"pre2018_cal": second_level(frame, "base", w)["mean"]}
            if w == 0.75:
                r["fwd"] = {y: forward(hist, y)["all"] for y in (2020, 2024)}
            out[f"rclip={rc},w={w}"] = r
            print(f"rclip={rc} w={w}: pre-cal {r['pre2018_cal']:.5f} fwd {r.get('fwd', '')}", flush=True)
    calib.R_CLIP, model.CAL_WEIGHT = 1.5, keep_w
    return out


def main() -> None:
    hist = pd.read_csv(OUT / "hist_features.csv")
    if "--rclip" in sys.argv:
        (OUT / "select_v3_rclip.json").write_text(json.dumps(rclip_test(hist), indent=1, default=str))
        return
    if "--dedup" in sys.argv:
        (OUT / "select_v3_dedup.json").write_text(json.dumps(dedup_test(), indent=1, default=str))
        return
    if "--pollster" in sys.argv:
        (OUT / "select_v3_pollster.json").write_text(json.dumps(pollster_test(), indent=1, default=str))
        return
    if "--gclose" in sys.argv:
        (OUT / "select_v3_gclose.json").write_text(json.dumps(gclose_test(hist), indent=1, default=str))
        return
    if "--calib" in sys.argv:
        (OUT / "select_v3_calib.json").write_text(json.dumps(calib_specs(hist), indent=1, default=str))
        return
    with_forward = "--quick" not in sys.argv
    out = {}
    only_sponsor = "--sponsor" in sys.argv
    for prev, wls, scale in CONFIGS:
        if only_sponsor and scale == 0 and not (prev == "full" and wls == 0):
            continue
        t0 = time.time()
        set_config(prev, wls, scale)
        r = evaluate(hist, with_forward)
        key = f"prev={prev},wls={wls},sponsor={scale}"
        out[key] = r
        fwd = r.get("fwd", {})
        print(f"{key:40s} pre-uncal {r['pre2018_uncal']['mean']:.5f} pre-cal {r['pre2018_cal']['mean']:.5f}"
              f" fwd2020 {fwd.get(2020, np.nan):.5f} fwd2024 {fwd.get(2024, np.nan):.5f}"
              f" ({time.time() - t0:.0f}s)", flush=True)
    set_config("full", 0.0, 0.0)
    (OUT / ("select_v3_sponsor.json" if only_sponsor else "select_v3.json")).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
