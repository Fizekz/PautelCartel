# data/kavi-b: data collected by the kavi-b model

## fec_2018_preelection.csv (+ raw/weball18_20181107.zip)
- Source: Internet Archive capture (2018-11-07 08:07 UTC) of the FEC's S3 copy of the 2017-2018 "all candidates"
  summary, https://web.archive.org/web/20181107080700id_/https://cg-519a459a-0ea3-42c2-b7bc-fa1143481f74.s3-us-gov-west-1.amazonaws.com/bulk-downloads/2018/weball18.zip
- The .txt inside is dated 2018-11-06 21:22 (election day, before results; no post-general report existed yet).
  Coverage ends: 1,260 rows at 2018-09-30 (Q3), 1,011 at 2018-10-17 (pre-general), a handful later (termination
  reports, all filed before the capture).
- Built by `src/models/kavi-b/fetch_fec_2018_snapshot.py`. Columns: cand_id, cand_name, office, race_id_raw,
  receipts_pre, cvg_end_pre.
- Why: data/clean/fec_hist.csv holds full-cycle totals (through 2018-12-31), which include post-election money,
  so the 2018 backtest would not be strictly out of sample. features.py swaps in these totals for 2018 rows,
  matching by cand_id, else by surname + office + state + district (fec_hist keeps one id per committee; the
  snapshot sometimes files a person under a sibling id, e.g. Kirkpatrick AZ-02 H8AZ01104 vs H8AZ02151).
- **2022 gap:** the Archive has no capture of weball22.zip between 2022-04-11 and 2022-12-28, so the 2022
  backtest uses full-cycle totals (a mild leak). The final side model does not use FEC money at all; money only
  enters the within-party split of same-party races (CA/WA top-two, LA jungle, AK 2022). On 2018 the
  pre-election vs full-cycle log ratios differ by 0.07 on average.
