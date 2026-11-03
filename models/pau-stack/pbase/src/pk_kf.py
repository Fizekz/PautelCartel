"""pau-k: cache pau-w's Kalman-filter day-0 forecasts (config c2_tau30, the one pooled at w = 0.3 in
RESEARCH_ASSIM.md section 5) for every fold of the four schemes, plus the 2018/2022 backtests and 2026.
The filter does not depend on v6+nl or on any add-on, so one run per (test year, training years) serves every stack.
Usage: pk_kf.py [six pre18 bt loyo8 2026]  -> work/stack/kf/kf_<T>_<train years>.csv (race_id, p)"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time

import pandas as pd

import evaluate4 as E
import pw_assim as A
import pw_run as R
from common import WORK

KD = WORK / "stack" / "kf"
KD.mkdir(parents=True, exist_ok=True)
OPTS = R.CONFIGS["c2_tau30"]


def path(T, tr):
    return KD / f"kf_{T}_{'-'.join(str(y) for y in tr)}.csv"


def get(T, tr):
    f = path(T, tuple(tr))
    if not f.exists():
        t0 = time.time()
        kf, _, _, _ = A.fold(int(T), tuple(int(y) for y in tr), OPTS, leads=[0])
        kf[["year", "race_id", "senate", "polled", "p"]].to_csv(f, index=False)
        print(f"kf {T} {tr} {time.time() - t0:.0f}s", flush=True)
    return pd.read_csv(f)


if __name__ == "__main__":
    for s in sys.argv[1:] or ("six", "pre18", "bt", "loyo8"):
        if s == "2026":
            get(2026, tuple(range(2010, 2026, 2)))
            continue
        for T, tr in E.splits(s):
            get(T, tr)
