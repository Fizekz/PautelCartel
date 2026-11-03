# fec_hist.csv: FEC fundraising, House and Senate candidates, 2010-2024

Built by `src/data/fetch_fec.py`. Pulled 2026-11-01.

## Sources
- `https://www.fec.gov/files/bulk-downloads/20YY/weballYY.zip`, `.../cnYY.zip` and the committee master
  `.../cmYY.zip` for YY = 10, 12, ..., 24. Raw copies are in `data/raw/fec/` (cm header:
  https://www.fec.gov/files/bulk-downloads/data_dictionaries/cm_header_file.csv).

## Rows (27,186)
| year | 2010 | 2012 | 2014 | 2016 | 2018 | 2020 | 2022 | 2024 |
|---|---|---|---|---|---|---|---|---|
| rows | 3437 | 3012 | 2621 | 2854 | 3710 | 3847 | 4229 | 3476 |
| House race_ids | 435 | 435 | 435 | 435 | 435 | 435 | 435 | 435 |
| Senate race_ids | 37 | 33 | 34 | 34 | 35 | 34 | 34 | 33 |

The file has one row per FEC House or Senate candidate whose `CAND_ELECTION_YR` equals the cycle year. This
includes **primary losers and candidates who raised nothing**. FEC bulk files have no general-election flag (only
2010 fills the GEN_ELECTION field), so join to MIT/MEDSL results by name to get general-election candidates.
`party_rank` (rank within year × race × DEM/REP/IND bucket, by receipts) is a proxy: rank 1 is usually the nominee,
but not always.

Columns: `year, race_id, cand_name, party, ici, receipts, disbursements` (as specified), plus `cand_id`,
`party_bucket` (DEM incl. DFL / REP / IND = everything else), `party_rank`, `cash_on_hand`, `coverage_end_date`.
Receipts are blank (about 35-40% of rows) when the candidate filed no financial reports. Those are almost always
non-viable candidates.

## race_id conventions
- The race_id uses **that year's district numbering**. 2010 uses the 2000-census apportionment, 2012-2020 the
  2010-census apportionment and 2022-2024 the 2020-census apportionment. Mid-decade redraws (PA 2018, NC 2016/2020,
  FL 2016, VA 2016, OH/NY 2024, AL/LA 2024, etc.) change district boundaries under the same number, so do not
  assume a district is the same place across years.
- At-large seats are `ST-AL`: AK, DE, ND, SD, VT, WY, plus MT through 2020.
- I dropped candidates whose FEC district is impossible for that year, usually typos with state-legislative
  numbers such as OH-43 or TN-92: 28/48/38/37/27/39/132/64 rows for 2010-2024. I also dropped DC and the territories.
- Senate is `ST-SEN`. **The same id covers both races when a state held two in one year** (special plus regular,
  e.g. GA 2020, MN/MS 2018, OK/SC 2014, OK 2022, NE 2024). Candidates from the two races are mixed together.
- A Senate state is kept for a year only if at least one candidate there is I or O, which drops early filers for
  later cycles (64 rows). A few extra states may remain, for example 2018 shows 35 states.

## Duplicate candidate ids (fixed 2026-11-01 by the verifier)
When one person holds several FEC candidate ids that share one principal campaign committee (PCC) in the same
election year, weball repeats that committee's totals under every id. The first build therefore put House members
who were running for Senate into their old House district with the full Senate money (O'Rourke 2018 in TX-16 with
$80.3M, Baldwin 2012 in WI-02, Gardner 2014 in CO-04, Sinema 2018 in AZ-09, Rosen 2018 in NV-03, ...). That
inflated House receipts and made the wrong person `party_rank` 1 (CO-04 2014 ranked Gardner above the nominee Buck).
`drop_shared_pcc_duplicates` now keeps, within each (PCC, election year) group, the id that the committee
master (`cmYY.zip`, field CAND_ID) links the PCC to, or else the id whose office matches the committee type (H/S).
This dropped 225 rows (9/15/15/11/22/20/39/94 for 2010-2024). Every dropped id was checked: each is a second id of a
person who is still in the file under the kept id (or a same-PCC twin filed in DC/another state). Rows with identical
totals but different committees remain (38 rows, all under $75k, e.g. two $100 filers): these are coincidences.

## Caveats
- weball totals cover the **two-year cycle only**. Senate fundraising from the earlier four years of a six-year
  term is not included.
- `ici` is FEC's own I/C/O flag and is sometimes wrong in the bulk file. 32 rows have it blank.
- Special elections held in odd years are not included.
