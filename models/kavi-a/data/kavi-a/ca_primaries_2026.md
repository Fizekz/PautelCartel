# ca_primaries_2026.csv (kavi-a only)

June 2 2026 California top-two primary results for the 9 same-party general-election races
(CA-04/07/11/12/14/29/34/37/40), used by `src/models/kavi-a/same_party.py`. Pre-election information.

- Source: Ballotpedia district pages `https://ballotpedia.org/California%27s_<N>th_Congressional_District`,
  fetched 2026-11-01 with `src/models/kavi-a/fetch_ca_primaries.py` (urllib, generic user agent). Raw HTML is in
  `data/kavi-a/raw/ballotpedia_CA-XX.html`; `--offline` re-parses it.
- Parsed: the first "Nonpartisan primary election" results table mentioning 2026. Columns: race_id, candidate,
  party_label, primary_pct, votes.
- Caveats: names of candidates without a photo are sometimes only a surname (the page's markup differs); only the
  two general-election finalists are used and they are matched by surname. **CA-12 and CA-29 did not parse**
  (different table layout); those two races fall back to the incumbency/receipts heuristic in races.py (both are
  lopsided: Simon, Rivas). Ballotpedia is crowd-maintained; the percentages were not checked against the CA SOS
  Statement of Vote.
