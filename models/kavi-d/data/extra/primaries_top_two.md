# primaries_top_two.csv (borrowed from kavi-b, unchanged copy)

Source: kavi-b `data/kavi-b/primaries_top_two.csv`, built by kavi-b's `fetch_primaries.py` from the Wikipedia parse
API (election boxes of the CA/WA House pages, parsed by district heading; cache in kavi-b's data/kavi-b/raw/wiki/).
CA 2012-2026 and WA 2012-2024, 2,418 rows: year, state, race_id, candidate, party_label, votes, pct, page, revid,
party. June/August top-two primaries are pre-general information in every year.

kavi-d use: the primary share (pct) of each finalist of the 52 same-party finals 2012-2024 (98 of 102 matched by
name) fits the primary weight of the same-party model, and gives the 2018/2022 backtest same-party splits their
primary shares. 2026 shares come from kavi-d's own data/extra/ca_primary_2026.csv.
