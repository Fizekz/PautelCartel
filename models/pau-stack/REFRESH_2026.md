# REFRESH_2026: the 2026 poll file refreshed on Nov 2, and what it does to v6+nl (pau-f)

Branch `pau/f-1102` (local only, nothing pushed). Data: Wikipedia via the MediaWiki API only, current revisions
of Nov 2 (every revision before the 2026-11-03T23:00Z cutoff); Kalshi snapshots from `wt/market/data/kalshi/`
(read-only). Everything below is 2026 and out of sample: no outcome exists yet.

## Results (in short)

**The refresh.**
- **Polls.** 964 polls against 916 in the Nov 1 file:
  - 49 are new, 45 of them ending Oct 25 or later (44 Senate, 1 House);
  - 1 is renamed, 1 was removed and 1 changed.
- **Page dates.** Pages fetched 2026-11-02 23:10-23:25Z. Revision timestamps run from Oct 24 to Nov 2 23:07Z, so
  none is after the cutoff.
- **Generic ballot.** D+8.6 (aggregate as of Oct 30) → **D+8.7** (as of Nov 1).

**What changes in v6+nl (452 races):**
- 13 races move by ≥ 1 point of P(D); 3 EV picks against the Nov 1 price change.
- All the large moves are Senate races with new polls:
  - NE +0.109, ME +0.071, IA −0.053, TX +0.039, OH −0.018, AK +0.015, KS +0.013.
- The House moves (at most 0.024) come only from the two House-page changes (AZ-01 poll removed, MI-13 poll added),
  through the re-estimated 2026 district pattern. Check: refreshing only the Senate polls leaves every House P(D)
  exactly unchanged.
- The generic-ballot update alone moves P(D) by 0.0003 on average.

**EV pick changes against the Nov 1 price:**

| race | before | after | gain under the new q |
|---|---|---|---|
| TX-SEN | Paxton (R) | **Talarico (D)** | +0.056 |
| OH-SEN | Brown (D) | **R** | +0.006, knife edge: EV 0.388 vs 0.394 |
| GA-08 | D | **R** | +0.006, noise: P(D) 0.004 → 0.003 against a 1.6% price |

Total expected gain under the new q: **+0.07 points**.

**Against the market:**
- The refresh does not bring v6+nl closer to Kalshi on average. Mean |P(D) − Kalshi latest| over the Senate goes
  0.049 → 0.052.
  - Moves toward the market: IA, and NE, which overshoots it.
  - Moves further from it: ME (0.886 vs 0.708) and TX (0.782 vs 0.715).
- KS, the biggest Kalshi mover (0.505 → 0.625), barely moves here (0.785 → 0.797): v6+nl was already well above
  the market there.

## Races whose P(D) or EV pick changes
- P(D) is v6+nl's probability for the D-side; the D-side is the independent in NE.
- Kalshi D: the normalised price of the D-side candidate.
  - Nov 1: `kalshi_races_2026-11-01T074812Z.csv`.
  - Latest: `races_2026-11-02T230947Z.csv`.
- EV pick: argmax q·(−log2 p) against the Nov 1 price, using old and then new polls.
- gain: the new pick's expected points minus the old pick's, both valued under the new q.
- pm4: the v4 poll average (D-side margin). The model's poll input is mostly `poll_margin` (21 days, LV/RV/V only):
  NE −3.7 → −1.9, ME +3.1 → +5.0, IA +1.4 → +0.7, TX +2.6 → +3.3, OH +3.6 → +3.2, KS +4.1 → +4.3,
  AK +2.0 → +2.3.
- Senate mu moves about 0.77 × the change in `poll_margin`.

| race | P(D) old → new | Δ | Kalshi D Nov 1 → latest | EV pick old → new | gain (new q) | poll avg pm4 old → new | new polls (D-side margin) |
|---|---|---|---|---|---|---|---|
| NE-SEN | 0.262 → 0.371 | +0.109 | 0.254 → 0.304 | D | +0.000 | -1.6 → -1.4 | NPA (R) 10-29 LV -3; YouGov 10-29 LV +0 |
| ME-SEN | 0.815 → 0.886 | +0.071 | 0.685 → 0.708 | D | +0.000 | +2.7 → +3.7 | YouGov 10-29 LV +5; UNH 10-31 LV +9; NYT/Siena 11-01 LV +6; Survey 160 11-01 LV +3 |
| IA-SEN | 0.658 → 0.605 | -0.053 | 0.439 → 0.510 | D | +0.000 | +0.4 → +0.2 | YouGov 10-29 LV +1; GrayHouse 10-31 LV -2; Rasmussen (R) 10-31 LV -1; NYT/Siena 11-01 LV +1 |
| TX-SEN | 0.743 → 0.782 | +0.039 | 0.688 → 0.715 | **R → D** | +0.056 | +3.1 → +3.5 | YouGov 10-29 LV +4; AtlasIntel 11-01 A +6; NYT/Siena 11-01 LV +7; Trafalgar (R) 11-01 LV +0; Research Co. 11-02 LV +3 |
| AZ-06 | 0.821 → 0.845 | +0.024 | 0.895 → 0.884 | R | +0.000 |  |  |
| IA-03 | 0.796 → 0.819 | +0.023 | 0.847 → 0.837 | R | +0.000 |  |  |
| OH-SEN | 0.796 → 0.778 | -0.018 | 0.708 → 0.695 | **D → R** | +0.006 | +3.5 → +3.1 | Concord (D) 10-24 LV +4; YouGov 10-29 LV +2; InsiderAdvantage (R) 10-30 LV +2; ActiVote 10-31 LV +0; NYT/Siena 11-01 LV +5; Research Co. 11-01 LV +1; Trafalgar (R) 11-01 LV +2 |
| WI-03 | 0.797 → 0.815 | +0.018 | 0.885 → 0.880 | R | +0.000 |  |  |
| GA-12 | 0.041 → 0.024 | -0.017 | 0.145 → 0.126 | R | +0.000 |  |  |
| AK-SEN | 0.727 → 0.742 | +0.015 | 0.708 → 0.705 | R | +0.000 | +2.7 → +2.1 | NYT/Siena 10-30 LV +2; InsiderAdvantage (R) 11-01 LV +0 |
| AZ-01 | 0.930 → 0.916 | -0.014 | 0.813 → 0.813 | D | +0.000 | +3.9 → +2.3 |  |
| NJ-07 | 0.919 → 0.933 | +0.014 | 0.933 → 0.947 | R | +0.000 | +5.6 → +5.6 |  |
| KS-SEN | 0.785 → 0.797 | +0.013 | 0.505 → 0.625 | D | +0.000 | +2.7 → +3.4 | YouGov 10-29 LV +1; NYT/Siena 11-01 LV +5; Survey 160 11-01 LV +6 |
| GA-08 | 0.004 → 0.003 | -0.001 | 0.016 → 0.020 | **D → R** | +0.006 |  |  |

**The House rows have no new polls.** They move only through the 2026 district pattern, which is re-estimated from the
2026 House polls after AZ-01 lost its Keating D+4 poll (pm4 +3.9 → +2.3) and MI-13 gained a D-sponsored poll at
D+50 (its own P(D) moves by less than 0.01). AZ-06, IA-03, WI-03 and NJ-07 are R picks although P(D) is above 0.8:
Kalshi's Nov 1 D price is higher still (0.85-0.93), so the underdog side has the higher EV, both before and after
the refresh.

The full table for all 452 races, including pD old/new, mu, the poll features, both Kalshi prices, EV and picks, is
`pbase/data/refresh_1102/model_1102/v6nl_2026_old_vs_new.csv`. That folder also holds the `features_2026_*.csv`
rebuilds, which are not committed. Columns `pD_newpolls_oldgb` give the race-poll effect alone.

## The 49 new polls
- Every one is in `pbase/data/refresh_1102/live_1102/new_polls.csv`, with candidates, sponsor and source URL.
- D-side margin = D-side % − R %, as Wikipedia lists it (top-line, before any two-way or partisan adjustment).

| race | pollster | field dates | n | pop | D-side | R | D-side margin | partisan flag |
|---|---|---|---|---|---|---|---|---|
| AK-SEN | NYT/Siena | 10-23 to 10-30 | 610 | LV | Peltola 48 | 46 | +2 |  |
| AK-SEN | InsiderAdvantage | 10-31 to 11-01 | 1200 | LV | Peltola 48 | 48 | +0 | R |
| CO-SEN | Lake Research Partners | 09-16 to 09-30 (ends before Oct 25) | 600 | RV | Hickenlooper 52 | 38 | +14 | D |
| CO-SEN | CU Boulder / YouGov | 10-09 to 10-20 (ends before Oct 25) | 710 | LV | Hickenlooper 57 | 39 | +18 |  |
| CO-SEN | Research Co. | 10-31 to 11-02 | 450 | LV | Hickenlooper 57 | 35 | +22 |  |
| FL-SEN | Research Co. | 10-31 to 11-02 | 450 | LV | Nixon 44 | 51 | -7 |  |
| GA-SEN | InsiderAdvantage | 10-31 to 11-02 | 2500 | LV | Ossoff 53 | 46 | +7 | R |
| GA-SEN | Research Co. | 10-31 to 11-02 | 450 | LV | Ossoff 52 | 42 | +10 |  |
| IA-SEN | YouGov | 10-26 to 10-29 | 2020 | LV | Turek 50 | 49 | +1 |  |
| IA-SEN | GrayHouse | 10-28 to 10-31 | 601 | LV | Turek 46 | 48 | -2 |  |
| IA-SEN | Rasmussen Reports | 10-27 to 10-31 | 901 | LV | Turek 46 | 47 | -1 | R |
| IA-SEN | NYT/Siena | 10-26 to 11-01 | 705 | LV | Turek 48 | 47 | +1 |  |
| IL-SEN | Research Co. | 10-31 to 11-02 | 450 | LV | Stratton 53 | 41 | +12 |  |
| KS-SEN | YouGov | 10-26 to 10-29 | 2092 | LV | Hamilton 50 | 49 | +1 |  |
| KS-SEN | NYT/Siena | 10-27 to 11-01 | 736 | LV | Hamilton 49 | 44 | +5 |  |
| KS-SEN | Survey 160 | 10-27 to 11-01 | 1267 | LV | Hamilton 50 | 44 | +6 |  |
| MA-SEN | UNH | 10-27 to 10-31 | 905 | LV | Markey 53 | 32 | +21 |  |
| ME-SEN | YouGov | 10-26 to 10-29 | 1503 | LV | Jackson 52 | 47 | +5 |  |
| ME-SEN | UNH | 10-27 to 10-31 | 1654 | LV | Jackson 54 | 45 | +9 |  |
| ME-SEN | NYT/Siena | 10-27 to 11-01 | 714 | LV | Jackson 51 | 45 | +6 |  |
| ME-SEN | Survey 160 | 10-27 to 11-01 | 1323 | LV | Jackson 50 | 47 | +3 |  |
| MI-13 | Center for Strategic Politics | 10-28 to 10-30 | 310 | LV | McKinney 69 | 19 | +50 | D |
| MI-SEN | YouGov | 10-26 to 10-29 | 2093 | LV | El-Sayed 51 | 47 | +4 |  |
| MI-SEN | NYT/Siena | 10-23 to 11-01 | 1011 | LV | El-Sayed 53 | 42 | +11 |  |
| MI-SEN | Research Co. | 10-30 to 11-01 | 450 | LV | El-Sayed 48 | 46 | +2 |  |
| MN-SEN | InsiderAdvantage | 10-29 to 10-30 | 1200 | LV | Flanagan 49 | 46 | +3 | R |
| MN-SEN | Research Co. | 10-30 to 11-01 | 450 | LV | Flanagan 48 | 43 | +5 |  |
| MN-SEN | Survey 160 | 10-27 to 11-01 | 1365 | LV | Flanagan 52 | 43 | +9 |  |
| NC-SEN | RedBox Research | 10-20 to 10-23 (ends before Oct 25) | 3292 | LV | Cooper 50 | 43 | +7 | D |
| NC-SEN | East Carolina University | 10-26 to 10-29 | 1041 | LV | Cooper 54 | 45 | +9 |  |
| NC-SEN | YouGov | 10-26 to 10-29 | 2130 | LV | Cooper 55 | 44 | +11 |  |
| NE-SEN | National Public Affairs | 10-25 to 10-29 | 800 | LV | Osborn 45 | 48 | -3 | R |
| NE-SEN | YouGov | 10-26 to 10-29 | 1979 | LV | Osborn 49 | 49 | +0 |  |
| NH-SEN | National Public Affairs | 10-25 to 10-29 | 800 | LV | Pappas 51 | 45 | +6 | R |
| NH-SEN | UNH | 10-27 to 10-31 | 2040 | LV | Pappas 54 | 42 | +12 |  |
| NM-SEN | Research Co. | 10-31 to 11-02 | 450 | LV | Luján 56 | 37 | +19 |  |
| OH-SEN | Concord Public Opinion Partners | 10-22 to 10-24 (ends before Oct 25) | 496 | LV | Brown 46 | 42 | +4 | D |
| OH-SEN | YouGov | 10-26 to 10-29 | 2090 | LV | Brown 50 | 48 | +2 |  |
| OH-SEN | InsiderAdvantage | 10-29 to 10-30 | 1200 | LV | Brown 49 | 47 | +2 | R |
| OH-SEN | ActiVote | 10-18 to 10-31 | 400 | LV | Brown 50 | 50 | +0 |  |
| OH-SEN | NYT/Siena | 10-26 to 11-01 | 717 | LV | Brown 49 | 44 | +5 |  |
| OH-SEN | Research Co. | 10-30 to 11-01 | 450 | LV | Brown 45 | 44 | +1 |  |
| OH-SEN | Trafalgar Group | 10-31 to 11-01 | 1092 | LV | Brown 47 | 45 | +2 | R |
| RI-SEN | UNH | 10-27 to 10-31 | 753 | LV | Reed 63 | 32 | +31 |  |
| TX-SEN | YouGov | 10-26 to 10-29 | 2141 | LV | Talarico 51 | 47 | +4 |  |
| TX-SEN | AtlasIntel | 10-28 to 11-01 | 1678 | A | Talarico 51 | 45 | +6 |  |
| TX-SEN | NYT/Siena | 10-27 to 11-01 | 1215 | LV | Talarico 50 | 43 | +7 |  |
| TX-SEN | Trafalgar Group | 10-31 to 11-01 | 1089 | LV | Talarico 48 | 48 | +0 | R |
| TX-SEN | Research Co. | 10-31 to 11-02 | 450 | LV | Talarico 49 | 46 | +3 |  |

**Not new:**
- **Renamed:** GA-SEN "University of Georgia School of Public and International Affairs" Oct 7-15 is now listed as
  "University of Georgia". Same poll, same numbers, so it counts once.
- **Removed:** AZ-01 Keating Research, Oct 26-28, D 49-45.
  - Wikipedia deleted it on Nov 2 00:33Z in revision 1383300075, with the edit summary "W:RS" (not a reliable
    source).
  - The refreshed file follows Wikipedia and drops it.
  - Two other AZ-01 polls added at 05:20 and 05:30Z were reverted within minutes and are not in the file.
- **Changed:** MN-SEN Big Data Poll, Sep 30. The kept version is now 46-44, not 40-38: the page now lists a
  different version of the same poll first.

## Sanity checks
- **Duplicates.**
  - No duplicate (race, pollster, end date) rows.
  - No poll appears under two pollster names: same race, end date and D/R/other shares.
  - Two same-pollster pairs have overlapping field dates. Both are distinct releases, not duplicates:
    - ActiVote OH-SEN, Sep 26-Oct 18 and Oct 18-31: consecutive rolling waves sharing one day;
    - Impact Research SC-SEN, Aug 14-22 and Aug 18-24: already in the Nov 1 file.
  - The near-match of NE-SEN NYT/Siena (Oct 23-29) and YouGov (Oct 26-29) is two different polls: Osborn 43 vs 49.
- **Dates.**
  - No poll ends after the cutoff, or after the fetch day; the latest end date is Nov 2 (Research Co.).
  - No start date is after its end date.
  - Every page revision used predates 2026-11-03T23:00Z.
- **Partisan flags.** They come from the same code as the existing pipeline: Wikipedia's per-poll "(D)/(R)" tag, plus
  the RedBox → D override in `polls_v4.py`.
  - **Flagged R (9):** InsiderAdvantage ×4, Rasmussen, National Public Affairs ×2, Trafalgar ×2.
  - **Flagged D (4):** Lake Research, RedBox, Concord, Center for Strategic Politics.
  - **One inconsistency to decide on:** the IA-SEN GrayHouse poll (Oct 28-31, R+2) carries no tag on Wikipedia. The
    pipeline therefore treats it as non-partisan, although GrayHouse's three earlier polls in the file (AR ×2, MT)
    are tagged R.
    - Flagged R, the v4 average would shift it 1 point toward the D-side.
    - `pm_all` would apply the fitted 2026 R-sponsor bias (−4.6) instead.
    - I left it as Wikipedia has it.
- **Population.** AtlasIntel TX-SEN (Oct 28-Nov 1, D+6) is listed on Wikipedia as "1,678 (A)", meaning adults.
  - `poll_margin` and `pm_all` keep only LV/RV/V polls, so they drop it.
  - `pm4` keeps it at the lowest population rank.
- **Generic ballot.** Wikipedia still has no per-poll list for the 2026 generic ballot. As in the shared pipeline, the
  input is the aggregator table's "Average" row: daily snapshots from the House-elections article's revisions,
  Oct 30 to Nov 2.
- **Rate limit.**
  - One HTTP 429 (retry-after 3 s) at the 11th request of the first pass, at 5-s gaps; the retry succeeded.
  - I then stopped and resumed at 10-s gaps, reusing the cached pages, and got no further 429s.
  - Total about 100 API requests, plus one more to read the AZ page's edit summaries.

## Files
- **The refreshed input, same format as the shared file:**
  - `pbase/data/refresh_1102/polls_2026_wiki_2026-11-02T2324Z.csv`;
  - `national_2026_wiki_2026-11-02T2324Z.csv`.
  - These are identical copies of `live_1102/polls_2026.csv` and `live_1102/national_2026.csv`.
  - The shared `wt/model/data/features/polls_2026.csv` and `work/features_2026.csv` are untouched.
- **Alongside them:**
  - `live_1102/pages.csv`: title, revision id and timestamp, fetch time and poll rows per page;
  - `live_1102/meta.json`;
  - `live_1102/{new,changed,removed}_polls.csv`;
  - `live_1102/polls_2026_senate_only_refresh.csv`: the Senate refreshed, House as on Nov 1, for the House check.
- **Code (`pbase/src/`):**
  - `refresh_polls_2026.py`: the fetch;
  - `fetch_polls_vendored.py`;
  - `refresh_diff_2026.py`: new polls and checks;
  - `refresh_v6nl_2026.py`: model old vs new;
  - `refresh_report_2026.py`: the tables above.
- **Rerun:**
  - `python3 src/refresh_v6nl_2026.py data/refresh_1102/model_1102 data/refresh_1102/live_1102/polls_2026.csv data/refresh_1102/live_1102/national_2026.csv`
  - Run it from `pbase/`; it takes about 10 s.


## How it was built

**The current input.**
- `features_2026.csv` takes its poll columns from three averagers:
  - `polls.py`: `poll_margin`, 21 days, LV/RV/V only;
  - `polls_pa.py`: `pm_all`;
  - `polls_v4.py`: `pm4` and its variants, 120 days, 14-day half-life, partisan polls shifted 1 point against the
    sponsor.
- All three read the shared Wikipedia file `wt/model/data/features/polls_2026.csv`: 916 polls, pages fetched
  2026-11-01 08:05Z. The fetcher is `wt/model/src/fetch_polls.py`.
- The 2026 generic ballot (`gb`, the only year-level input v5.1/v6 uses) is the latest "Average" row of the
  aggregator table in `national_2026.csv`: 8.6 as of Oct 30.

**The refresh** (`pbase/src/refresh_polls_2026.py`):
- **The same parser.** It re-runs the shared fetcher's code, vendored as `pbase/src/fetch_polls_vendored.py`.
  - One change: the HTML parser is `html.parser` instead of `lxml`, which is not installed on this Mac.
  - Check: re-parsing the Nov 1 page cache with the vendored copy reproduces the shared file exactly, 916 of 916
    polls with zero cell differences. So every difference below comes from Wikipedia, not from the code.
- **Dating.**
  - A batched `prop=revisions` query records each page's latest revision id and timestamp. Every one is earlier
    than the cutoff, and the script refuses any later revision.
  - Each page is then parsed by `oldid`.
  - Per-page revision ids and timestamps are in `pages.csv`.
- **Politeness.**
  - Gaps between requests: 5 s at first, then 10 s.
  - Generic User-Agent, no e-mail; `maxlag=5`; exponential backoff on 429/503; a hard stop at the second rate-limit
    response.
- **Name matching.** Matchups are matched with the same Nov 1 07:48Z Kalshi candidate file the Nov 1 build used.

**The model run** (`pbase/src/refresh_v6nl_2026.py`):
- **Inputs.** `features.build_2026()` is unchanged. Only its reads are redirected:
  - `features/polls_2026.csv` goes to the file under test;
  - the 2026 row of `work/national.csv` goes to the refreshed generic ballot, for the "new" run.
- **Writes.** Its one side-effect write goes into the output folder. Nothing shared is written.
- **Checks.**
  - The old rebuild equals `work/features_2026.csv`, all 126 columns to 1e-9.
  - Old v6+nl P(D) equals `work/pr_nl_2026.csv` `p_nl`, max difference 5e-8.
- **Model.** v6+nl is `PE.all_(PE.V["v51"], pr_pat_lam=1.0, pr_pat_set="nl", pr_pat_unp=True)`, with the nl products
  of `pr_theta_nl2.py` lines 8-11, fitted on 2010-2024 and predicting the 452 D-vs-R / R-vs-independent /
  Louisiana races.
- **EV pick.** argmax of q·(−log2 p) between the D-side and the R-side candidate:
  - q is the model's P(D-side);
  - p is the Nov 1 07:48Z Kalshi price (the earliest Nov 1 snapshot), normalised within the race and floored at 1%;
  - the candidates are matched by surname, so CA-06 uses Kiley's price and NE-SEN uses Osborn's.
