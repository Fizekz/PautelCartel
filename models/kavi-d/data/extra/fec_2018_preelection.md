# fec_2018_preelection.csv (borrowed from kavi-b, unchanged copy)

Source: kavi-b `data/kavi-b/fec_2018_preelection.csv`, built by kavi-b's `fetch_fec_2018_snapshot.py` from an
Internet Archive capture (2018-11-07 08:07 UTC) of the FEC bulk `weball18.zip` whose .txt is dated 2018-11-06
21:22, i.e. pre-election totals (coverage mostly to 2018-09-30 or the 2018-10-17 pre-general report).
Columns: cand_id, cand_name (FEC "LAST, FIRST"), office (H/S), race_id_raw (e.g. AK-00, state-district),
receipts_pre, cvg_end_pre.

kavi-d use: replaces the full-cycle 2018 receipts (which run to 2018-12-31) in the 2018 backtest for the within-side
split, the side principal and the rated-independent tie-break. Joined by cand_id, else by surname within the race.
No pre-election capture exists for 2022 (kavi-b searched the Archive), so the 2022 backtest uses no money at all.
