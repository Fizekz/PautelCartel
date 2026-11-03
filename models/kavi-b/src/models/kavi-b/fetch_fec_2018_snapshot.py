"""Pre-election FEC totals for 2018 from a Wayback Machine copy of weball18.zip.

The shared fec_hist.csv holds full-cycle totals (reports through 2018-12-31), which include post-election money.
The Internet Archive captured the FEC's S3 copy of weball18.zip at 2018-11-07 08:07 UTC; the .txt inside is
dated 2018-11-06 21:22 (election day, before any post-general report existed), so its totals are pre-election.
No pre-election capture exists for 2022 (first capture after Nov 8 is 2022-12-28), see data/kavi-b/README.md.

Usage: python src/models/kavi-b/fetch_fec_2018_snapshot.py [--refresh]
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile

import pandas as pd

from common import OWN

URL = ("https://web.archive.org/web/20181107080700id_/https://cg-519a459a-0ea3-42c2-b7bc-fa1143481f74."
       "s3-us-gov-west-1.amazonaws.com/bulk-downloads/2018/weball18.zip")
RAW = OWN / "raw" / "weball18_20181107.zip"
OUT_CSV = OWN / "fec_2018_preelection.csv"
COLS = [
    "CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION", "TTL_RECEIPTS",
    "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH", "COH_BOP", "COH_COP", "CAND_CONTRIB",
    "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY",
    "TTL_INDIV_CONTRIB", "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "SPEC_ELECTION",
    "PRIM_ELECTION", "RUN_ELECTION", "GEN_ELECTION", "GEN_ELECTION_PRECENT",
    "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT", "INDIV_REFUNDS", "CMTE_REFUNDS",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if args.refresh or not RAW.exists():
        with urllib.request.urlopen(URL, timeout=120) as resp:
            RAW.write_bytes(resp.read())
    with zipfile.ZipFile(RAW) as zf:
        info = zf.infolist()[0]
        raw = zf.read(info.filename).decode("latin-1")
    df = pd.read_csv(io.StringIO(raw), sep="|", header=None, names=COLS, dtype=str, keep_default_na=False, quoting=3)
    office = df["CAND_ID"].str[0]
    dist = pd.to_numeric(df["CAND_OFFICE_DISTRICT"], errors="coerce").fillna(0).astype(int)
    race = [f"{st}-SEN" if o == "S" else f"{st}-{d:02d}" for o, st, d in zip(office, df["CAND_OFFICE_ST"], dist)]
    out = pd.DataFrame({
        "cand_id": df["CAND_ID"], "cand_name": df["CAND_NAME"], "office": office, "race_id_raw": race,
        "receipts_pre": pd.to_numeric(df["TTL_RECEIPTS"], errors="coerce"),
        "cvg_end_pre": pd.to_datetime(df["CVG_END_DT"], format="%m/%d/%Y", errors="coerce").dt.date,
    }).drop_duplicates("cand_id")
    out.to_csv(OUT_CSV, index=False)
    print(f"zip member {info.filename} dated {info.date_time}; {len(out)} candidates -> {OUT_CSV}")
    print("latest coverage end:", out["cvg_end_pre"].max())


if __name__ == "__main__":
    main()
