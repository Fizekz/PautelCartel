"""v3 design selection: generic-ballot cycle designs against the v2 designs.

Same criteria as select_v2.py (side-model log loss on non-Safe contested races); features fixed at v2's
r + poll + lean, no floor. Also prints the cycle prediction (mean, sd) each design gives for 2018, 2022 and 2026.

Usage: python src/models/kavi-b/select_v3.py   -> out/report/model_selection_v3.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402
from cycle import CYCLE_DESIGNS  # noqa: E402
from features import hist_table  # noqa: E402
from model import Config, SideModel  # noqa: E402
from select_v2 import evaluate  # noqa: E402

warnings.filterwarnings("ignore")
FEATS = ("r", "poll", "lean")


def cycle_pred(hist: pd.DataFrame, cfg: Config, year: int) -> str:
    m = SideModel(cfg).fit(hist[hist["year"] < year])
    s = m.cycle_summary(hist[hist["year"] == year])
    return f"{s['pred_mean']:+.2f} ({s['pred_sd']:.2f})"


def main() -> None:
    hist = hist_table(pre2018_fec=True)
    rows = []
    for cyc in CYCLE_DESIGNS:
        cfg = Config(f"v3|{cyc}", FEATS, ridge=0.005, cycle=cyc, floor=False)
        r = {"cycle": cyc, **evaluate(hist, cfg),
             "cyc_2018": cycle_pred(hist, cfg, 2018), "cyc_2022": cycle_pred(hist, cfg, 2022)}
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    out = pd.DataFrame(rows)
    (OUT / "report").mkdir(exist_ok=True, parents=True)
    out.round(5).to_csv(OUT / "report" / "model_selection_v3.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
