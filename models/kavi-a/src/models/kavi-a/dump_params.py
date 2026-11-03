"""Write out/model_params.json: fitted coefficients for the 2018, 2022 and 2026 training windows."""
from __future__ import annotations

import json

import pandas as pd

from backtest import VARIANTS
from common import OUT
from model import fit_all

WINDOWS = {"2018": list(range(2010, 2017, 2)), "2022": list(range(2010, 2021, 2)), "2026": list(range(2010, 2025, 2))}


def main() -> None:
    hist = pd.read_csv(OUT / "hist_features.csv")
    out = {}
    for name, years in WINDOWS.items():
        fits = fit_all(hist, years, *VARIANTS["F+P+R+C"])
        out[name] = {}
        for ch, f in fits.items():
            out[name][ch] = {"train_years": years, "beta": f.beta.round(5).to_dict(), "a": round(f.a, 5),
                             "b": round(f.b, 4), "nat_sd": round(f.nat_sd, 5), "sd_state": round(f.sd_state, 5),
                             "sd_race": {k: round(v, 5) for k, v in f.sd_race.items()},
                             "poll_k": f.poll_k, "stage2": {k: round(v, 5) for k, v in f.stage2.items()},
                             "year_effects": {int(k): round(v, 5) for k, v in f.alpha.items()},
                             "n_train": int(len(f.frame))}
            if f.calib is not None:
                out[name]["calibration"] = {"intercept": round(float(f.calib.intercept_[0]), 4),
                                            "coef_z_r_sz_sr": [round(float(c), 4) for c in f.calib.coef_[0]]}
    (OUT / "model_params.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
