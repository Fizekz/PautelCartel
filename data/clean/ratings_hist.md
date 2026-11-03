# ratings_hist.csv: final pre-election expert ratings, 2010-2024

Built by `src/data/fetch_ratings.py` (same code path as `ratings_2026.csv`). Pulled 2026-11-01 from the MediaWiki
parse API. Raw page JSON is cached in `data/raw/ratings/` (27 pages, 28 MB). `source_revid` on every row pins the
exact Wikipedia revision.

## Sources (one set per year Y in 2010, 2012, ..., 2024)

- `https://en.wikipedia.org/wiki/Y_United_States_House_of_Representatives_election_ratings`: per-district ratings
  table. The column headers carry each forecaster's final pre-election rating date.
- `https://en.wikipedia.org/wiki/Y_United_States_Senate_elections`: the Predictions table, which covers every race
  that cycle, specials included.
- `https://en.wikipedia.org/wiki/Y_United_States_House_of_Representatives_elections`: per-state tables, giving PVI,
  incumbent party and listed candidates for each district, used for the House seats that need imputation.

## Rows: 3760

| year | House rows (listed / imputed safe) | Senate rows | core dates (Cook / Sabato / Inside, House) |
|---|---|---|---|
| 2010 | 435 (152 / 283) | 38 | 11-01 / 11-01 / 11-01 (Senate table has no dates) |
| 2012 | 435 (111 / 324) | 33 | 11-05 / 11-05 / 11-02 |
| 2014 | 435 (77 / 358) | 36 | 11-03 / 10-30 / 10-29 |
| 2016 | 435 (75 / 360) | 34 | 11-07 / 11-07 / 11-03 |
| 2018 | 435 (136 / 299) | 35 | 11-05 / 11-05 / 11-05 |
| 2020 | 435 (118 / 317) | 35 | 11-02 / 11-02 / 10-28 |
| 2022 | 435 (143 / 292) | 35 | 11-07 / 11-07 / 11-03 |
| 2024 | 435 (93 / 342) | 34 | 11-01 / 11-04 / 10-31 |

Cook, Sabato and Inside are non-null on every row (`n_raters` = 3). Inside Elections is the renamed Rothenberg
Political Report (later Rothenberg & Gonzales). Wikipedia's "Rothenberg" / "Roth." columns (2010-2016) are mapped
to `inside`.

`other_*` (numeric only, never imputed, blank where that outlet did not publish): cq (2010), nyt (2010-2018),
fte (538, 2010-2024), rcp (all years), rollcall and thehill (2012), princeton (2012 Senate), dailykos (2014-2020),
desart (2014 Senate), tpm (2016 Senate), cnn and politico (2018 Senate; politico 2020 Senate and 2022), fox
(2018 Senate, 2022, 2024), ddhq (2020-2024), economist (2020 Senate, 2022), elections_daily (2020-2024), cbs
(2022 Senate), split_ticket (2024).

## Keys and conventions

- `race_id` uses the district numbering in force that year: 2010 = 2002-2010 lines; 2012-2020 = 2010-census lines
  (plus court redraws: FL 2016, VA 2016, NC 2016/2020, PA 2018); 2022 = 2020-census lines; 2024 = the 2022 lines with
  the 2024 redraws (AL, GA, LA, NC, NY). The same id in two years is not necessarily the same territory. `pvi_wiki` is
  the Cook PVI as printed on that year's page, a different PVI vintage each cycle.
- Senate: `XX-SEN` when the state had one race that year; when a state had a regular race and a special the same
  year, the special is `XX-SEN-S` (NY 2010, OK and SC 2014, MN and MS 2018, GA 2020, OK 2022, NE 2024). `special=1`
  marks every special, including the ones without a suffix (DE, WV and MA 2010, HI 2014, AZ 2020).
- Same scale and `imputed_safe` / `imputed_basis` logic as `ratings_2026.md`. For a seat missing from the
  competitive table, the side is the party of both finalists when `same_party_final` is set (55 rows, basis
  `same_party_final`); otherwise the incumbent's party, unless the seat is new or vacant, or its PVI points the
  other way by 5 or more points. Then the PVI sign is used (34 rows).
- `same_party_final`: House only. D or R when every candidate on that year's results table belongs to one party
  (top-two CA/WA lockouts, Louisiana jungle races). 2012: 9, 2014: 8, 2016: 11, 2018: 5, 2020: 9, 2022: 6, 2024: 7.
  In past years the candidate cell lists general-election results, so `n_cands_wiki`=1 means uncontested
  (13-34 seats per year).
- Independents: "Safe I" / "Likely I" (King ME 2012/2018/2024, Sanders VT 2012/2018/2024) are scored on the
  non-Republican side (negative). Check `*_raw` if you model independents separately. Nebraska 2024 (Osborn, I) was
  rated Lean R, so it needed no I handling.

## Validation

After imputation I recomputed each forecaster's House D/R/Toss-up seat totals and compared them with Wikipedia's
"Overall" row for that column. They match exactly for 2016, 2020, 2024 (all three) and 2026, for Inside in 2010,
2014 and 2022, and for Sabato in 2012 (after the CA-31 fix below), 2014 and 2022. The mismatches are:

| forecaster | year | Wikipedia Overall row | our totals (D / R / Toss-up) |
|---|---|---|---|
| Cook | 2010 | D 181, R 204, 50 toss-ups | 182 / 204 / 49 |
| Cook | 2012 | D 174, R 217, 44 toss-ups | 179 / 227 / 29 |
| Cook | 2014 | D 182, R 228, 25 toss-ups | 183 / 231 / 21 |
| Cook | 2018 | D 210, R 194, 30 toss-ups | 210 / 195 / 30 |
| Cook | 2022 | D 191, R 211, 33 toss-ups | 187 / 212 / 36 |
| Inside | 2012 | D 185, R 233, 18 toss-ups | 182 / 236 / 17 |
| Inside | 2018 | D 214, R 202, 19 toss-ups | 214 / 201 / 20 |
| Sabato | 2010 | D 198, R 237 | 201 / 234 |
| Sabato | 2018 | D 228, R 207 | 229 / 206 |

Every rating cell parsed and none was blank, so these gaps look like the Overall row and the per-district column
being edited at different dates on Wikipedia. Cook 2012 is the largest; treat Cook 2012 House with some suspicion.

## Caveats

- Wikipedia is a secondary, crowd-edited source. It is the only free source that has every forecaster's final
  ratings for every year. For a high-stakes check, compare a sample with the archived forecaster pages on the
  Wayback Machine.
- MA-SEN 2010 is the January 19, 2010 special (Scott Brown), and its ratings are presumably from before that January vote; the 2010 Senate table has no dates. Drop it if you want
  November races only. GA 2020 (both seats) went to January runoffs, and their ratings are the pre-November ones.
- The imputation assumes "absent from the table" means Safe from all three forecasters. The Overall-total check
  supports this for every year listed above.
- Ratings only. Winners and results are not in this file. Join a results dataset on (year, race_id), using the same
  year's district numbering.

## Independent verification (2026-11-01, second agent)

- Re-parsed the cached Wikipedia tables with separate code: every cook/sabato/inside cell matches.
- Second source: 270toWin's per-forecaster tables (Crystal Ball as of 2022-11-07 and 2024-11-04; Inside Elections
  as of 2022-11-03 and 2024-10-31), all 435 House seats. Sabato 2022 and 2024: 0 differences, imputed seats included.
  Inside 2022: 2 differences. 270toWin has FL-05 as Likely R where we have FL-04 Likely R (flip); the Wikipedia row
  is the more plausible one, since the new FL-4 was the 2022 flip seat, so it is kept. Inside 2024: OH-13 Tilt D
  here vs Lean D on 270toWin. 270toWin's Cook archive pages were not reachable (404).
- Cross-checked against `data/clean/results_house.csv`: no non-imputed seat rated Safe by all three forecasters was
  won by the other party. One imputed seat was: **2012 CA-31**, an R-vs-R top-two lockout (Gary Miller vs Bob
  Dutton) that had been imputed Safe D from the PVI sign. **Fixed**: `same_party_final` now takes precedence in the
  imputation (`safe_side` in `src/data/fetch_ratings.py`), so CA-31 2012 is Safe R. After the fix, no imputed seat
  was won by the other party in any year.
- Spot check: Cook moving NJ-SEN 2018 to Toss-up (2018-10-26), with Inside at Likely D, is confirmed by
  contemporary press (njbiz, insidernj).
- **Gap:** the 2022 and 2024 California Senate partial-term specials (held the same day as the regular race,
  with the same candidates) are not in Wikipedia's Predictions tables and are not rows here.
