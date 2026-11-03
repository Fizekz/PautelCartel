"""v3 selection harness: leave-one-year-out on NON-test years only.

Two criteria, neither touches 2018 or 2022:
  pre2018: years 2010, 2012, 2014, 2016 (each held out, fit on the other three)
  six:     years 2010-2016, 2020, 2024 (each held out, fit on the other five non-test years)
Variants are v3 options (model.V3 flags) and extra features. Scores = log loss of P(D side wins), contested
D-v-R races; 'close' = |mean rating| < 2.5.
Run: .venv/bin/python src/models/kavi-e/eval_v3.py
"""
from __future__ import annotations

import sys

import model
from backtest import load_frames
from eval_loyo import loyo, score

CRITERIA = {"pre2018": (2010, 2012, 2014, 2016), "six": (2010, 2012, 2014, 2016, 2020, 2024)}
VARIANTS = {
    "base": (set(), (), ()),
    "hetero": ({"hetero"}, (), ()),
    "wls": ({"wls"}, (), ()),
    "hetero+wls": ({"hetero", "wls"}, (), ()),
    "r_hinge": (set(), ("r_hinge",), ("r_hinge",)),
    "L_safe": (set(), ("L_safe",), ("L_safe",)),
    "hetero+r_hinge": ({"hetero"}, ("r_hinge",), ("r_hinge",)),
    "hetero+L_safe": ({"hetero"}, ("L_safe",), ("L_safe",)),
    "pool": ({"pool"}, (), ()),  # adopted: House a 0.4; Senate a 0 (a 0.2 tested, reverted)
    "pool+hetero": ({"pool", "hetero"}, (), ()),
}


def main(argv: list[str]) -> int:
    names = argv or list(VARIANTS)
    house, senate = load_frames()
    for crit, years in CRITERIA.items():
        for name in names:
            flags, eh, es = VARIANTS[name]
            model.V3 = set(flags)
            ph = loyo(house, model.HOUSE_FEATURES + eh, years)
            ps = loyo(senate, model.SENATE_FEATURES + es, years, clf_extra=house)
            print(f"{crit:8s} {name:16s} House {score(ph)}  Senate {score(ps)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
