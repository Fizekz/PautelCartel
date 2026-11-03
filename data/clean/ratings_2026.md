# ratings_2026.csv: expert race ratings for the 470 races of 2026

Built by `src/data/fetch_ratings.py` (helpers `src/data/ratings_parse.py`, `src/data/html_tables.py`,
`src/data/wiki_fetch.py`). Pulled 2026-11-01 from the MediaWiki parse API. Raw page JSON (rendered HTML + revid) is
cached in `data/raw/ratings/`. Rerun `.venv/bin/python src/data/fetch_ratings.py` to rebuild from the cache, or add
`--refresh` to download again.

## Sources

| Page | Used for | revid |
|---|---|---|
| https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_election_ratings | per-district ratings table (147 districts) | 1383158409 |
| https://en.wikipedia.org/wiki/2026_United_States_Senate_elections (Predictions table) | all 35 Senate races | 1383102895 |
| https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections | PVI (2026 lines), incumbent party, candidate list for the 288 unlisted districts | 1383103767 |

Rating dates in the column headers: Cook 2026-10-30 (House and Senate), Sabato 2026-10-27 (House) and 2026-10-29
(Senate), Inside Elections 2026-10-29. `as_of` is the latest of the three (2026-10-30).

I also tried the forecasters' own sites (centerforpolitics.org/crystalball, cookpolitical.com, insideelections.com).
All three returned HTTP 403 to a scripted request, so I used no data from them.

## Rows and coverage

- 470 rows, one per template `race_id` (435 House + 35 Senate). Every template id matched, and there are no extras.
- 182 rows come straight from a Wikipedia table: 147 House, 35 Senate.
- 288 House rows are `imputed_safe=1`. They are absent from the competitive table, which lists every district that
  at least one forecaster rates below Safe/Solid. They get +/-3 for cook, sabato and inside, with raw text
  `Safe X (imputed)`. The side comes from the incumbent's party, or from the sign of the 2026 PVI if the seat is
  new or vacant (`imputed_basis` = `incumbent_party` / `pvi_sign`; 7 seats used PVI). When both finalists
  are from one party (`same_party_final`), that party is used first (`imputed_basis` = `same_party_final`, 9 seats; the
  values are the same as before).
  - Check: after imputation, the D/R/Toss-up seat totals for Cook (D 222, R 195, 18 toss-ups), Inside (224/198/13)
    and Sabato (225/193/17) equal the "Overall" row Wikipedia gives for each forecaster, exactly.
- `other_*` columns give the numeric rating from the other forecasters in the same tables: ddhq, economist,
  fiftyplusone (FPO), fox, rcp, silver_bulletin, split_ticket, plus nbc (Senate only). They are not imputed, so they
  are blank for the 288 imputed House rows. Raw text and dates are kept only for the core three.

## Columns

`race_id, chamber` · `cook, cook_raw, sabato, sabato_raw, inside, inside_raw` (numeric + Wikipedia text) ·
`other_*` · `mean_rating` (mean of cook/sabato/inside, imputed values included) · `n_raters` (core raters with a value;
3 everywhere) · `imputed_safe` · `as_of` · `imputed_basis` · `pvi_wiki` (Cook PVI as Wikipedia prints it, R positive,
EVEN = 0) · `incumbent_party` (House only, D/R/I, blank for new or vacant seats) · `same_party_final` (House only, D or
R when every listed general-election candidate belongs to that party) · `n_cands_wiki` (candidates Wikipedia lists) ·
`label, source_page, source_revid` · `cook_date, sabato_date, inside_date`.

Scale: Safe/Solid D = -3, Likely D = -2, Lean D = -1, Tilt D = -0.5, Toss-up = 0, Tilt R = 0.5, Lean R = 1,
Likely R = 2, Safe/Solid R = 3. "(flip)" is dropped for the numeric value but kept in `*_raw`. An independent rating
("Safe I") would count on the non-Republican side (negative). No 2026 row has one.

## Caveats

- **Same-party general elections.** In nine House races both finalists are from one party: CA-04, CA-07, CA-11,
  CA-12, CA-14, CA-29, CA-34, CA-37 (all D vs D) and CA-40 (Calvert vs Kim, R vs R). The party rating is Safe and
  says nothing about which candidate wins. These are the same nine races Kalshi lists with party `UNK`. Use
  `same_party_final` to send them to a candidate-level model.
- **Redistricting.** Wikipedia's ratings page uses the 2026 district lines, so `race_id` numbers follow the new maps
  (states with new maps for 2026, per `data/clean/redistricting_2026.csv`: AL, CA (Prop 50), FL, LA, NC, OH, TN, TX,
  UT). `pvi_wiki` is the 2026 PVI on those lines. Don't join these ids to 2024 results without a crosswalk.
- **Top-two and RCV.** Ratings are party ratings. AK and ME use ranked-choice voting. CA and WA use top-two, which is
  why the same-party races above exist.
- **Louisiana.** On the 2026 map, LA's House races on Nov 3 are an all-candidate (jungle) primary, with a Dec 12
  runoff if nobody gets 50% (see `redistricting_2026.md`). The ratings are for the eventual winner's party. Georgia
  also holds a runoff if no candidate gets 50%.
- **Maps not redrawn.** MO (2025 map frozen by referendum) and VA (amendment voided) use 2024 lines. That is why MO-05
  (Cleaver) is imputed Safe D, which agrees with Kalshi's D 0.97.
- **Senate specials.** FL-SEN (Moody, appointed) and OH-SEN (Husted, appointed) are the 2026 specials. Wikipedia
  labels them "(special)". The template ids have no `-S` suffix because neither state has a regular race in 2026.
- **Wikipedia is crowd-edited.** Some incumbent names in the Senate table look odd (e.g. "Darline Graham" for SC,
  "Alan Armstrong" for OK). The ratings are cited per column and match the forecasters' Overall totals. Treat the
  name text as unverified.
- **Cross-check with Kalshi (diagnostic only, not a model input).** Excluding toss-ups and the nine same-party
  races, the rating side agrees with Kalshi's favourite in 446 of 449 races. The three exceptions are cases where
  the experts lean R and Kalshi favours D: MI-10 (mean +0.17), VA-01 (+0.17) and WI-01 (Lean R by all three, Kalshi
  50/50). 12 more races are rated an exact Toss-up on average.
- Ratings in `other_*` are as published. Some (Economist, Silver Bulletin, Split Ticket, DDHQ) are model outputs
  binned into categories.

## Independent verification (2026-11-01, second agent)

- **Parse.** I re-parsed the cached House and Senate tables with separate code. All 147 House rows and 35 Senate
  rows match on every core and `other_*` column. The schema matches the spec. There are 470 unique ids equal to the
  template, every chamber matches, values are in [-3, 3], and `mean_rating` = mean(cook, sabato, inside).
- **Second source: 270toWin's per-forecaster House tables** (Crystal Ball as of 2026-10-30, Cook as of 2026-10-23,
  Inside as of 2026-10-15), all 435 seats, imputed ones included:
  - Sabato: 434 of 435 identical. The exception is **LA-05**: 270toWin lists the open, redrawn LA-5 as *Likely
    R*, while Wikipedia omits it and we had imputed Safe R. **Fixed** with a documented override (`RATING_OVERRIDES`
    in `src/data/fetch_ratings.py`; `sabato_date` 2026-10-30; the source URL is in `imputed_basis`). `imputed_safe`
    stays 1 because cook/inside are still imputed. The Overall-row totals are unchanged.
  - Cook: 12 differences and Inside: 44. All of them move toward D in our (later-dated) Wikipedia values, e.g. Cook
    CO-05, PA-01 and SC-01 Lean R to Toss-up, and NC-05, NC-09, NJ-02 and WA-05 Solid to Likely R. That fits the
    late-October rating changes reported in the press (e.g. Cook's 2026-10-09 move of 18 races toward D), not
    parse errors. They are kept as is.
- **Senate.** 270toWin's Cook (10-20), Crystal Ball (10-27) and Inside (10-15) maps agree with ours apart from
  moves toward D that fit our later dates (e.g. Cook NE Likely to Lean R, Inside TX/OH Toss-up to Tilt D). Sabato
  LA-SEN Likely R and MN-SEN Likely D are as printed in Wikipedia's 10-29 column. 270toWin's 10-27 map still has
  Safe R and Lean D for them.
- **Kalshi.** No race with mean |rating| >= 1.5 has Kalshi favouring the other side, and no imputed-safe seat has a
  Kalshi favourite below 0.85. The largest soft disagreements: CA-22 (Cook/IE Toss-up, Kalshi D 0.91), NC-14
  (Solid R, Kalshi R 0.77) and NC-03 (Likely R, Kalshi R 0.56).
- **Not verified directly.** cookpolitical.com, centerforpolitics.org and insideelections.com stay blocked (403 /
  fetch denied). The second source is a mirror (270toWin), not the forecasters themselves.
