# polls_2026.csv: 2026 race-level Senate and House general-election polls

Built by `src/data/fetch_polls.py` (pulled 2026-11-01; `--offline` rebuilds from `data/raw/polls/`).
There is one row per poll question. A pollster that reports LV and RV, or several matchups, gets several rows.

## Rows and coverage
- **1,513 rows across 152 race_ids**: Senate 1,035 rows, House 478 rows (Wikipedia 1,433, VoteHub 80).
- **Usable rows** (the head-to-head between the actual nominees): `nominee_match == True`, or `nominee_match` blank
  and `is_hypothetical == False`, and `rcv_round` not `2`/`3` (later ranked-choice rounds). That gives **1,116 rows
  in 147 races: 29 of 35 Senate races (733 rows) and 118 of 435 House districts (383 rows).**
- **Senate races with no general-election poll anywhere:** CO, DE, NJ, OR, WV, WY. All are safe seats.
- Usable polls in the priority Senate races: TX 86, MI 80, NC 71, OH 53, IA 45, AK 42, MT 39, ME 36, FL 34,
  NH 33, GA 30, NE 25, KS 24, MN 24, SC 17, MS 7, LA 3. (Several pollsters report two versions of one survey, e.g.
  LV and RV, or with and without leaners; these are separate rows. Collapse per pollster + end date if needed.)
- **Most House districts (317 of 435) have no usable poll.** The polled districts are the competitive ones plus a few
  oddities.
- Dates run from 2023-04 to 2026-10-30. The model should weight by recency.

## Columns
Required schema: `race_id, chamber, pollster, start_date, end_date, sample_size, population (lv/rv/a/v),
partisan_sponsor, dem_cand, dem_pct, rep_cand, rep_pct, other_cand, other_pct, source_url`.

Extra columns:
- `undecided_pct`
- `top_other_cand` / `top_other_pct`: the highest-polling candidate who is not the top D or top R. **Use this for
  independents** (NE-SEN Osborn, MT-SEN Bodnar, ID-SEN Achilles, SD-SEN Bengs, AK-AL Bill Hill, CA-06 Kiley):
  `other_pct` lumps that candidate together with every minor candidate and the generic "Other" column.
- `cand_detail`: every named candidate with party and share, sorted by share, e.g.
  `Pete Ricketts (R): 49; Dan Osborn (I): 43`. `(?)` means no party could be resolved (VoteHub answers in
  same-party top-two races: CA-11, CA-34, CA-40). For those races `dem_*`/`rep_*` are blank and `other_pct` is the
  sum of both finalists, so read the two shares from `cand_detail`.
- `rcv_round`: Alaska's ranked-choice tables. `BA` (first choices of all respondents, before allocation) and `1`
  are first-choice readings; `2`/`3` are later elimination rounds and should be dropped (or used only as an RCV
  final-round signal). Blank everywhere else. The same AK poll often also appears as a blank-round row from the
  first-past-the-post table, with the same first-choice numbers.
- `is_hypothetical`: True means the table is a Wikipedia "Hypothetical polling" or collapsed table. It is blank for
  VoteHub rows, because VoteHub does not flag hypotheticals.
- `nominee_match`: True when the surnames of the top D and top R candidates both appear in Kalshi's 2026-11-01
  candidate list for that race. When there is no D or no R, the top two named candidates are used instead. It is
  blank where Kalshi lists only "Democratic party"/"Republican party" (AK-SEN, for example). Kalshi is used here
  **only to cross-check names**, never as a model input.
- `sponsor_note`: Wikipedia's sponsor footnote, e.g. "Poll sponsored by Red Eagle Politics, which supports Paxton".
- `source`: wikipedia or votehub.
- `source_page`

How the main columns are derived:
- `dem_*` / `rep_*`: the highest-polling D (DFL counts as D) and the highest-polling R in the question.
- `other_cand` / `other_pct`: every other named candidate, plus the generic "Other" column.
- `partisan_sponsor`: D, R or D/R (bipartisan team). It follows Wikipedia's "(D)"/"(R)" tag on the pollster cell,
  which marks partisan pollsters or sponsors. Wikipedia tags far more polls than 538's "partisan" field would: about
  half the rows carry a tag.

## Sources
1. **Wikipedia race pages**, rendered HTML via the MediaWiki parse API. Each row's `source_url` has `?oldid=` set
   to the exact revision that was parsed.
   - The 35 pages `2026 United States Senate election in <State>`. FL and OH use `... special election in ...`.
   - The 50 statewide `2026 United States House of Representatives election(s) in <State>` pages. Polls are taken
     from `District N` sections.
   - The two linked district pages that exist: CO-01 and NY-13.
   - Tables are dropped when their heading path contains primary, runoff, caucus or convention.
   - Cached in `data/raw/polls/wiki/` (62 MB of JSON in total; no single file is over 50 MB).
2. **VoteHub public polls API**, https://api.votehub.com/polls. The full dump is in
   `data/raw/polls/votehub/polls_all.json`. Only `us-senator` and `us-representative` polls with 2026 general
   subjects are used.
   - VoteHub answers have no party. Party comes from Wikipedia poll headers in the same state, then from Kalshi
     names in the same state.
   - VoteHub "Dem"/"Rep" answers are district generic-ballot questions, written as "Generic Democrat" / "Generic
     Republican".
   - A VoteHub row is dropped as a duplicate when any Wikipedia row (hypothetical tables included) has the same
     race, an end date within 2 days, D and R values each within 1.5 points, and the same D and R surnames.
   - Party for VoteHub answers comes from the candidate's label in the latest Wikipedia poll in that state, then
     from Kalshi's list.
   - Attribute VoteHub if this data is published.

## Not obtained
- **FiftyPlusOne** (the source the organizers suggested): its CSV downloads (`/api/csv`) need a paid API key
  ($100/month standard account). The poll lists on its race pages are loaded with that key, so we could not get its
  polls. Only its public average series is embedded in the page HTML, and we did not use it. This is a cost decision
  for the humans.
- **RealClearPolling** returns HTTP 403 to scripted requests.

## Caveats
- **Redistricting:** these are 2026 polls of 2026 districts, so district numbers match the template. Do not join
  them to 2024 results by district number in states that redrew their maps (TX, CA Prop 50, MO, NC, OH, UT and
  others).
- **Top-two and same-party races** (CA, WA): some general elections are D vs D or R vs R.
  - In those races the second same-party candidate goes into `other_cand`. Kalshi labels these races UNK, for
    example CA-40 Calvert vs Kim (both R) and CA-11 Wiener vs Chan.
  - A few WA and CA rows are summer top-two primary polls on the district pages (WA-03 Emerson 2026-07, for
    example). They show up as `nominee_match == False` or many-candidate rows.
- **Independents** (read `top_other_pct`, not `other_pct`):
  - NE-SEN: Dan Osborn (I) vs Ricketts. There is no Democrat; Osborn is in `other_cand`.
  - MT-SEN: Kurt Alme (R) vs Seth Bodnar (I) with a weak Democrat (Alani Bankhead) also on the ballot, so
    `dem_pct` there is Bankhead's share, not the main challenger's. ID-SEN (Todd Achilles, I) and SD-SEN
    (Brian Bengs, I) are similar.
  - AK-AL: Begich vs Bill Hill (I), with Democrat Eric Hafner far behind.
  - AK-SEN has two Dan Sullivans: the incumbent Dan S. Sullivan (R) and Dan J. Sullivan (I), a minor candidate.
  - CA-06: Kiley is listed as I.
- **Maine:** the Democratic nominee is Troy Jackson. The 26 Platner and 14 Mills rows are pre-primary matchups and
  are flagged `nominee_match == False`.
- **Ranked-choice voting** (AK, ME): ME tables and the AK first-past-the-post tables give first-choice shares. AK
  ranked-choice tables add one row per round; see `rcv_round`.
- **Georgia:** Ossoff vs Mike Collins. A runoff is possible if nobody passes 50%.
- **Duplicates:** the same poll can appear in both sources when the numbers differ by more than the tolerance or
  the names are spelled differently (one known case: NM-01 Research & Polling, 2026-10-09, "Okpareke" vs
  "Okapareke"). Wikipedia itself sometimes lists a poll twice (e.g. TX-09 and TX-28 DCCC); those identical rows are
  dropped.
- **Source quirks kept as published:** a few Wikipedia rows have shares that sum to more than 100, e.g. SC-01 GQR
  (Other 7%, Undecided 7%) and FL-25 National Research (42/49 with 24% undecided). They are left unchanged.
- **Parsing:** some rows have no sample size (27) or no population (9).

## Verification (independent check, 2026-11-01)
- All 152 race_ids are in the template, chamber labels match, no exact duplicate rows. Usable-poll leaders agree
  with Kalshi favourites everywhere: no race has a poll average above 3 points one way while Kalshi gives the
  other party more than 0.7.
- Spot-checked against PollCheck (pollcheck.co.uk), pollster releases and news reports: all 24 October FL-SEN rows,
  5 OH-SEN rows, 8 NE-SEN rows, IA-SEN Suffolk (48-44), TX-SEN Emerson (47-46-2), ME-SEN CNN/SSRS (48-45), NC-SEN
  Harper (49-34), SC-SEN NYT/Siena (48-48), UT-02 Lighthouse (39-34) and CA-40 (Kim 43 / Calvert 34; 32/33). All
  shares matched. The population label differs from some outlets for ME SSRS and UT-02.
- Fixes made in this pass:
  - `rcv_round` added. AK ranked-choice rounds had been mixed in as unlabeled separate polls.
  - `top_other_*` and `cand_detail` added. Independents' and same-party finalists' shares had been lumped into
    `other_pct`.
  - VoteHub duplicates of Wikipedia hypothetical-table rows are now removed (OH-SEN had 31). The duplicate check
    now also requires the same candidates.
  - VoteHub party hints now use each candidate's latest Wikipedia label. MT's Seth Bodnar had been coded D, and
    the AK-SEN incumbent had been confused with the independent Dan J. Sullivan.
  - The VoteHub `source_page` had been blank.
  - Accented surnames are now folded to ASCII for name matching ("Giménez" had become "nez").
