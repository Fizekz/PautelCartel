# national_2026.csv: 2026-cycle generic ballot and Trump job approval

Built by `src/data/fetch_polls.py` (pulled 2026-11-01). The file is long format, with one row per poll per metric.

## Columns
`date` (poll end date, or the update date for averages), `start_date`, `metric`, `value_dem`, `value_rep`
(generic ballot), `value` (approval), `pollster`, `sample`, `population` (lv/rv/a/v), `source`, `source_url`.

## Metrics and rows (2,691 rows in total)
| metric | source | n | dates |
|---|---|---|---|
| generic_ballot | VoteHub API | 544 | 2024-12-11 to 2026-10-04 |
| approve / disapprove | VoteHub API | 1,022 each | 2025-01-20 to 2026-10-06 |
| approve / disapprove | Wikipedia "Opinion polling on the second Trump presidency" | 40 each | 2026-07-27 to 2026-10-27 |
| approve_avg / disapprove_avg | Wikipedia aggregator table (Ballotpedia, CNN, DDHQ, FiftyPlusOne, NBC, RCP, Silver Bulletin, Economist, NYT) | 9 each | 2026-10-30 |
| generic_ballot_avg | Wikipedia "2026 United States House of Representatives elections", aggregator table (DDHQ, FiftyPlusOne, NBC, RCP, Silver Bulletin) | 5 | 2026-10-30 |

The latest aggregates (2026-10-30), copied as published and not recomputed:
- Generic ballot: D 51.3 to 52.2, R 42.4 to 44.0, roughly D+8 to D+9.
- Trump approval: 34 to 39 approve, 59 to 64 disapprove.

## Sources
- VoteHub public API: https://api.votehub.com/polls (full dump in `data/raw/polls/votehub/polls_all.json`).
  Attribute VoteHub if this data is published.
- Wikipedia via the MediaWiki parse API. The revision id is in `source_url`, and the cache is in
  `data/raw/polls/wiki/`.

## Gaps and caveats
- **VoteHub's national series stop in early October:** generic ballot on 10-04, approval on 10-06. Only 6 generic
  ballot polls end after 2026-09-01.
  - October approval is filled from Wikipedia's monthly tables. A Wikipedia poll is dropped when VoteHub has a poll
    from the same pollster (a shared distinctive word in the name) within 2 days with approval within 1 point.
  - **There are no individual October generic-ballot polls.** Wikipedia keeps only the aggregates, RealClearPolling
    returns 403, and FiftyPlusOne is paywalled. Use `generic_ballot_avg` (2026-10-30) as the late-cycle anchor.
- Wikipedia's monthly approval tables for 2025 and 2026 are short; they keep only selected polls. VoteHub is the
  bulk source.
- Population mixes A, RV and LV. Filter or adjust for it.
- VoteHub lists a few polls twice with identical fields (6 approval, 3 generic ballot); those exact duplicates are
  dropped.
- Verification (2026-11-01): the 2026-10-30 generic-ballot aggregates (D+8.0 to D+9.1) agree with PollCheck's
  independent average (D 51.1, R 42.4, D+8.7, updated 10-27). The approval aggregates are in line with Silver
  Bulletin and DDHQ figures reported for late October (about 36-38 approve, 60 disapprove). All values are 0-100,
  and no date falls after 2026-11-01.
- Some VoteHub tracking polls (Rasmussen, Morning Consult and others) appear many times. Weight them by pollster.
