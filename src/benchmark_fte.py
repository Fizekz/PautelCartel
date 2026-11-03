"""Build the 2018 and 2022 stand-in market: FiveThirtyEight's final forecast, one row per candidate.

538's sites are gone (projects.fivethirtyeight.com redirects to ABC News), so the raw files come from the
Wayback Machine. Headline models: "classic" in 2018, "_deluxe" in 2022 (what 538 showed by default).

Usage: python src/benchmark_fte.py [--download]
Output: data/benchmark/fte_final_2018.csv, data/benchmark/fte_final_2022.csv with columns
    year,race_id,chamber,special,candidate,party,p_fte,forecastdate,model
race_id: House AL-01 / AK-AL; Senate OH-SEN, and <ST>-SEN-S for a special held alongside a regular race.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "fte"
OUT = ROOT / "data" / "benchmark"
WAYBACK = "https://web.archive.org/web/{stamp}id_/https://projects.fivethirtyeight.com/{path}"
FILES = {
    "house_district_forecast_2018.csv": ("20181106", "congress-model-2018/house_district_forecast.csv"),
    "senate_seat_forecast_2018.csv": ("20181106", "congress-model-2018/senate_seat_forecast.csv"),
    "house_district_toplines_2022.csv": ("20230427025551", "2022-general-election-forecast-data/house_district_toplines_2022.csv"),
    "senate_state_toplines_2022.csv": ("20230427025614", "2022-general-election-forecast-data/senate_state_toplines_2022.csv"),
}
AT_LARGE = {"AK", "DE", "ND", "SD", "VT", "WY"}  # plus MT before 2022 (it gained a 2nd seat)
PARTY = {"D": "DEM", "R": "REP", "I": "IND", "O": "OTHER", "L": "LIB", "G": "GRN"}
SLOTS = ["D1", "D2", "D3", "D4", "R1", "R2", "R3", "R4", "I1", "O1"]


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (stamp, path) in FILES.items():
        with urllib.request.urlopen(WAYBACK.format(stamp=stamp, path=path), timeout=300) as resp:
            (RAW / name).write_bytes(resp.read())


def house_id(state: str, district: int, year: int) -> str:
    at_large = state in AT_LARGE or (state == "MT" and year < 2022) or district == 0
    return f"{state}-AL" if at_large else f"{state}-{int(district):02d}"


def build_2018() -> pd.DataFrame:
    frames = []
    for name, chamber in (("house_district_forecast_2018.csv", "House"), ("senate_seat_forecast_2018.csv", "Senate")):
        df = pd.read_csv(RAW / name)
        df = df[df["model"] == "classic"].copy()
        df["forecastdate"] = pd.to_datetime(df["forecastdate"])
        df = df[df["forecastdate"] == df["forecastdate"].max()]
        if chamber == "House":
            df["special"] = 0
            df["race_id"] = [house_id(s, d, 2018) for s, d in zip(df["state"], df["district"])]
        else:
            df["special"] = df["special"].astype(str).str.lower().eq("true").astype(int)
            df["race_id"] = df["state"] + "-SEN" + df["special"].map({0: "", 1: "-S"})
        frames.append(pd.DataFrame({
            "year": 2018, "race_id": df["race_id"], "chamber": chamber, "special": df["special"],
            "candidate": df["candidate"], "party": df["party"].map(PARTY).fillna("OTHER"),
            "p_fte": df["win_probability"], "forecastdate": df["forecastdate"].dt.date, "model": "classic",
        }))
    return pd.concat(frames, ignore_index=True)


def build_2022() -> pd.DataFrame:
    frames = []
    for name, chamber in (("house_district_toplines_2022.csv", "House"), ("senate_state_toplines_2022.csv", "Senate")):
        df = pd.read_csv(RAW / name, low_memory=False)
        df = df[df["expression"] == "_deluxe"].copy()
        df["forecastdate"] = pd.to_datetime(df["forecastdate"], format="%m/%d/%y")
        df = df[df["forecastdate"] == df["forecastdate"].max()]
        rows = []
        for _, r in df.iterrows():
            state, seat = r["district"].split("-")
            if chamber == "House":
                race_id, special = house_id(state, 0 if seat in ("AL", "0") else int(seat), 2022), 0
            else:
                # 2022 had one special Senate race alongside a regular one: Oklahoma (class 2 special).
                special = int(state == "OK" and seat == "S2")
                race_id = f"{state}-SEN" + ("-S" if special else "")
            for slot in SLOTS:
                cand, prob = r.get(f"name_{slot}"), r.get(f"winner_{slot}")
                if isinstance(cand, str) and cand != "all others" and pd.notna(prob) and float(prob) > 0:
                    rows.append({"year": 2022, "race_id": race_id, "chamber": chamber, "special": special,
                                 "candidate": cand, "party": PARTY[slot[0]], "p_fte": float(prob),
                                 "forecastdate": r["forecastdate"].date(), "model": "deluxe"})
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="re-download the raw files from the Wayback Machine")
    if ap.parse_args().download:
        download()
    OUT.mkdir(parents=True, exist_ok=True)
    for year, build in ((2018, build_2018), (2022, build_2022)):
        df = build()
        dupes = df.duplicated(["race_id", "candidate"]).sum()
        df.to_csv(OUT / f"fte_final_{year}.csv", index=False)
        sums = df.groupby("race_id")["p_fte"].sum()
        print(f"{year}: {df['race_id'].nunique()} races ({(df.chamber == 'House').groupby(df.race_id).first().sum()} House), "
              f"{len(df)} candidate rows, forecastdate {df['forecastdate'].max()}, dup rows {dupes}, "
              f"p sums {sums.min():.3f}-{sums.max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
