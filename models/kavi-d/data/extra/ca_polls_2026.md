# ca_polls_2026.csv (borrowed from kavi-b, unchanged copy)

Source: kavi-b `data/kavi-b/ca_polls_2026.csv`, built by kavi-b's `fetch_ca_polls_2026.py` from the Wikipedia
"2026 United States House of Representatives elections in California (districts 1-26 / 27-52)" sub-pages.
37 head-to-head polls in 10 districts, columns race_id, pollster, dates, sample, cand_a, party_a, pct_a, cand_b,
party_b, pct_b, page, end_date.

kavi-d use: same-party contests only (CA D-vs-D and CA-40 R-vs-R), polls ending on or after 2026-06-03 (after the
top-two primary), merged with the shared polls_2026 cand_detail shares (a poll present in both within 2 days is
counted once). They feed the hand-set same-party poll term (0.08 logit per point above the pair mean).
