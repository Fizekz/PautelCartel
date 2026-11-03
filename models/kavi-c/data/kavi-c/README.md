# data/kavi-c: files used by the kavi-c model that are not in data/clean

## fec_2018_preelection.csv (borrowed from kavi-b, byte-identical copy)
- Source: Internet Archive capture (2018-11-07 08:07 UTC) of the FEC's S3 copy of weball18.zip; the .txt inside is
  dated 2018-11-06 21:22 (election day, before any post-general report). Built by kavi-b's
  `src/models/kavi-b/fetch_fec_2018_snapshot.py` (see kavi-b `data/kavi-b/README.md`).
- Columns: cand_id, cand_name, office, race_id_raw, receipts_pre, cvg_end_pre. 3,656 candidates.
- Use here: `hist_features.load_fec_hist()` replaces every 2018 receipts value in fec_hist.csv by this snapshot
  (cand_id, else race + surname, else 0). 2018 is then leak-free as a test year and as a training year.
- 2022: the Archive has no capture of weball22.zip between 2022-04-11 and 2022-12-28, so the 2022 backtest drops
  money entirely (backtest.NO_MONEY_BACKTEST).

## dossiers_summary_2026-11-01.csv
- Frozen copy of `repo/data/dossiers/summary.csv` (85 contested 2026 races, researcher + skeptic probabilities for
  the Kalshi underdog, as of 2026-11-01). The per-race .md dossiers stay in `repo/data/dossiers/`.
- Use here: the late-information blend in `dossiers.py` (2026 only).

## fec_2022_preelection.csv (v3, borrowed from kavi-a)
- The 2022 rows of kavi-a's `data/kavi-a/fec_preelection.csv` (branch kavi/kavi-a-1101), built by kavi-a's
  `src/models/kavi-a/fetch_fec_preelection.py` from the Internet Archive capture (2022-08-03) of the FEC's
  `webl22.zip` (current House and Senate campaigns). Coverage ends mostly 2022-06-30 (Q2 reports): four months before
  election day, but strictly pre-election. 2,670 candidates.
- Columns: year, office, race_id (AK-00 style), cand_id, cand_name, receipts_pre, cvg_end.
- Use here: only while backtesting 2022 (`backtest.PRE_CUT_2022`), `hist_features.load_fec_hist()` replaces every
  2022 receipts value by this cut (cand_id, else race + surname, else 0). 2022 stays a full-cycle training year for
  the 2024 and 2026 fits.
