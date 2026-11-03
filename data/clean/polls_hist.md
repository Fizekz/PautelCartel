# polls_hist.csv: historical Senate and House race polls, 2010-2024 (for backtests)

Built by `src/data/fetch_polls.py` from the FiveThirtyEight archives (pulled 2026-11-01).
The schema is the polls_2026 schema plus `year`, with these extras:
- `stage`: general, runoff, or jungle primary (the LA and CA/WA-style top-two generals).
- `special`: Senate special election.
- `seat_name`
- `is_hypothetical`
- `ranked_choice_reallocated`
- `numeric_grade`, `pollster_rating_id`: 538's pollster rating.
- `internal`
- `question_id`, `poll_id`
- `source`
- `margin_actual`: present only in raw_polls rows (2010-2016); it is the actual result margin, cand1 minus cand2.
  cand1 is the Democrat in 3,197 of those rows, so it equals D-R there. In the D-vs-D (26) and R-vs-I (25) rows it
  is not D-R. It is blank for 2018-2024; join those years to a results file.

## Rows (8,767 in total)
| year | House | Senate | source |
|---|---|---|---|
| 2010 | 481 | 507 | 538 raw_polls.csv (final ~60 days only) |
| 2012 | 295 | 489 | 538 raw_polls.csv (final ~60 days only) |
| 2014 | 215 | 445 | 538 raw_polls.csv (final ~60 days only) |
| 2016 | 155 | 445 | 538 raw_polls.csv (final ~60 days only) |
| 2018 | 799 | 878 | 538 house/senate_polls_historical.csv (full cycle) |
| 2020 | 385 | 1,028 | same |
| 2022 | 345 | 878 | same |
| 2024 | 280 | 1,142 | same |

## Sources
- https://projects.fivethirtyeight.com/polls/data/senate_polls_historical.csv and `house_polls_historical.csv`.
  That host is dead (it redirects to ABC News), so the files come from Wayback Machine snapshots of 2025-03-05
  (URLs in `SOURCES_538` in the script). Raw copies are in `data/raw/polls/538/`.
- https://github.com/fivethirtyeight/data/blob/master/pollster-ratings/raw_polls.csv covers 1998-2023. It is used
  only for cycles the full archives do not cover (2010-2016).

## Caveats
- **2010-2016 are not comparable to 2018-2024.** raw_polls keeps only polls from the final stretch (538's
  `time_to_election` is at most 59-61 days; the median is about 25) and only the top two candidates, with no start
  date or population. For a like-for-like backtest, restrict 2018+ to polls that end within 60 days of election
  day. The 2010 MA Senate special (Jan 2010) has older polls dated relative to its own election day.
- **Fixed 2026-11-01:** 219 rows of 2010-2016 national generic-ballot polls (`<year>_House-G_US`, `race_id` "US") were
  removed. They are not race polls.
- **Shape of a row:** one row per poll question. Multi-population polls have one row per population. The 538 files
  keep hypothetical matchups, so filter on `is_hypothetical == False`.
- **Ranked-choice polls** (AK, ME): only the final reallocation round is kept.
- **District numbers follow that cycle's map.** Lines changed in 2012 (decennial), 2016 (FL, VA, NC), 2018 (PA),
  2020 (NC), 2022 (decennial) and 2024 (AL, GA, LA, NC, NY). Join on (year, race_id) to that year's results only.
- **At-large districts are written ST-AL:** AK, DE, ND, SD, VT, WY, and MT before 2022.
- **Senate specials:** `special` is True when 538's seat class differs from that year's regular class (538's
  historical files) or the race code is `Sen-GS` (raw_polls). Two races in one state-year therefore share
  `race_id` ST-SEN, so join on (year, race_id, special).
- **Runoffs:** Georgia's runoffs (`stage == runoff`) are included and need to be filtered.
- **Partisan flags:** `partisan_sponsor` uses 538's partisan field (DEM/REP/...). 2026 uses Wikipedia's D/R tags,
  which mark many more polls.

## Licence
FiveThirtyEight released its poll data under the Creative Commons Attribution 4.0 International licence (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/). Attribution: FiveThirtyEight. Changes: rows filtered and reshaped to
our schema and race identifiers, as described above. The 2018-2024 files were retrieved from the Internet Archive's
copies because the original site is offline.
