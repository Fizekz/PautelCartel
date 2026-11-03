"""Pre-election FEC candidate totals for the backtest years (removes the post-election look-ahead).

The shared fec_hist.csv holds two-year totals through Dec 31 of the election year, so a backtest that reads them
sees post-election money (debt retirement, the MS-SEN-S 2018 and GA-SEN 2022 runoffs). Wayback copies of the
FEC bulk files give earlier cuts:
  2018: weball18.zip (all offices) captured 2018-11-07 08:07 UTC, file dated election day 2018-11-06 (URL found
        by kavi-b, borrowed). Coverage ends mostly Oct 17 (pre-general reports).
  2022: webl22.zip ("current campaigns", House and Senate) captured 2022-08-03; coverage ends mostly
        2022-06-30 (Q2 reports), i.e. four months before election day. No weball22/webl22/webs22 capture exists
        between Aug 3 and Dec 28 2022 (Wayback CDX checked 2026-11-01), so this is the latest pre-election cut.
Writes data/kavi-a/fec_preelection.csv: year, office, race_id, cand_id, cand_name, receipts_pre, cvg_end.

Usage: python fetch_fec_preelection.py [--offline]
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile

import pandas as pd

from common import ROOT

OWN = ROOT / "data" / "kavi-a"
RAW = OWN / "raw"
OUT_CSV = OWN / "fec_preelection.csv"
SOURCES = {
    2018: ("https://web.archive.org/web/20181107080700id_/https://cg-519a459a-0ea3-42c2-b7bc-fa1143481f74."
           "s3-us-gov-west-1.amazonaws.com/bulk-downloads/2018/weball18.zip", RAW / "fec18" / "weball18_20181107.zip"),
    2022: ("https://web.archive.org/web/20220803192625id_/https://cg-519a459a-0ea3-42c2-b7bc-fa1143481f74."
           "s3-us-gov-west-1.amazonaws.com/bulk-downloads/2022/webl22.zip", RAW / "fec22" / "webl22_20220803.zip"),
}
COLS = ["CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION", "TTL_RECEIPTS", "TRANS_FROM_AUTH",
        "TTL_DISB", "TRANS_TO_AUTH", "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS",
        "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "CAND_OFFICE_ST",
        "CAND_OFFICE_DISTRICT", "SPEC_ELECTION", "PRIM_ELECTION", "RUN_ELECTION", "GEN_ELECTION",
        "GEN_ELECTION_PRECENT", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT", "INDIV_REFUNDS",
        "CMTE_REFUNDS"]


def parse(year: int, path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        info = zf.infolist()[0]
        raw = zf.read(info.filename).decode("latin-1")
    print(f"{year}: zip member {info.filename} dated {info.date_time}")
    df = pd.read_csv(io.StringIO(raw), sep="|", header=None, names=COLS, dtype=str, keep_default_na=False,
                     quoting=3)
    office = df["CAND_ID"].str[0]
    dist = pd.to_numeric(df["CAND_OFFICE_DISTRICT"], errors="coerce").fillna(0).astype(int)
    race = [f"{st}-SEN" if o == "S" else f"{st}-{d:02d}" for o, st, d in zip(office, df["CAND_OFFICE_ST"], dist)]
    out = pd.DataFrame({"year": year, "office": office, "race_id": race, "cand_id": df["CAND_ID"],
                        "cand_name": df["CAND_NAME"],
                        "receipts_pre": pd.to_numeric(df["TTL_RECEIPTS"], errors="coerce"),
                        "cvg_end": pd.to_datetime(df["CVG_END_DT"], format="%m/%d/%Y", errors="coerce").dt.date})
    return out[out["office"].isin(["H", "S"])].drop_duplicates("cand_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="only parse the saved zips")
    args = ap.parse_args()
    frames = []
    for year, (url, path) in SOURCES.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if args.offline:
                raise SystemExit(f"missing {path}; run without --offline")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                path.write_bytes(resp.read())
        frames.append(parse(year, path))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    for (y, o), g in out.groupby(["year", "office"]):
        cv = pd.Series(pd.to_datetime(g["cvg_end"])).dropna()
        print(y, o, len(g), "coverage end median", cv.median().date(), "max", cv.max().date())

if __name__ == "__main__":
    main()
