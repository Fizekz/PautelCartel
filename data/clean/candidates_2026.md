# candidates_2026.csv: provenance

General-election candidates, incumbency and seat status for all 470 Full Congress race_ids (435 House + 35 Senate).
Rebuild with `.venv/bin/python src/data/fetch_candidates.py` (it uses the cached raw pages; add `--refresh` to re-download).

## Sources (pulled 2026-11-01 ~07:53 UTC)

| Page | URL | revid |
|---|---|---|
| 2026 United States House of Representatives elections | https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections | 1383103767 |
| 2026 United States Senate elections | https://en.wikipedia.org/wiki/2026_United_States_Senate_elections | 1383102895 |
| 2026 US House elections in Louisiana / Tennessee / Alabama (redistricting and LA election format only) | https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections_in_Louisiana (same pattern for the others) | see the cached JSON |

Each page was fetched through the MediaWiki parse API (`action=parse`, rendered HTML) and cached as JSON in
`data/raw/candidates/`. Parsing uses stdlib `html.parser`. The House page's per-state tables (District, PVI,
Member, Party, First elected, Status, Candidates) and the Senate page's "Special elections during the preceding
Congress" and "Elections leading to the next Congress" tables are the inputs.

Cross-check only: `data/kalshi/races_2026-11-01T074812Z.csv` (column `candidates`). It sets `kalshi_name_match` and
adds a race note. It is never used for any other column.

## Counts

- 1,282 rows (one per general-election candidate on Wikipedia's list; write-ins dropped). All 470 race_ids present.
- Parties: DEM 482, REP 470, IND 149, LIB 93, GRN 32, OTHER 56 (the original label is kept in `notes`, e.g. Constitution, Working Class, Forward).
- Open seats: 75 House, 13 Senate (AL, IA, IL, KY, LA, MI, MN, MT, NC, NH, OK, TX, WY).
- Incumbents on the ballot: 361 House, 22 Senate. 11 House incumbents are running in a different district number than the one they hold now
  (CA-03 Bera, CA-06 Kiley, CA-40 Calvert, CA-41 Sanchez, FL-20 Wasserman Schultz, FL-23 Frankel, FL-25 Moskowitz, TX-37 Casar,
  UT-02 Moore, UT-03 Maloy, UT-04 Kennedy).
- 928 is_major rows. 98.8% match a Kalshi name. The 11 that don't are races where Kalshi shows only parties
  (AK-SEN, LA-01..04) or omits a minor second contender (AZ-03, MA-01, PA-03).
- Kalshi names missing from Wikipedia's ballot list (probably withdrawn or write-in): IL-04 Mayra Macias, NE-01 Austin Ahlman,
  ID-SEN David Roth (D), NE-SEN Cindy Burbank (D), SD-SEN Julian Beaudion.

## Column definitions / rules

- `party`: Democratic/DFL to DEM; Republican to REP; Independent / No party preference / Nonpartisan to IND; Libertarian to LIB;
  Green to GRN; anything else to OTHER. Kevin Kiley (CA-06) is listed by Wikipedia as **Independent**, so he is IND.
- `is_incumbent`: the candidate is a sitting member of **that chamber** listed in this race's incumbent rows (so House
  members running for Senate, such as Collins, Hinson, Barr, Moore, Hern and Hageman, are 0 in their Senate race). Appointed
  senators running count as incumbents: FL Ashley Moody, OH Jon Husted, SC Darline Graham.
  AK-SEN has two "Dan Sullivan"s on the ballot. The incumbent is Dan S. Sullivan, set by a hand override.
- `incumbent_party_of_seat`: party of the member who now holds that district **number** (old lines), for every House race,
  including when that member runs in another district on the new lines, retires or lost renomination. The script finds
  this holder across the whole state from Wikipedia's "Redistricted from the Nth district" notes. House totals: REP 219,
  DEM 215, IND 1 (CA-03, Kiley). Exceptions and details:
  - Vacant seats use the departed member's party: FL-20 (D, Cherfilus-McCormick resigned) and TX-23 (R, Gonzales resigned). Both stay vacant through election day, with no special election.
  - In redrawn states the old-number holder often runs elsewhere. The row's note names them, for example "district number now held by Al Green (DEM), listed by Wikipedia under TX-18".
    This applies to AL-01, CA-03, CA-06, CA-38, CA-41, FL-22, FL-23, FL-25, TX-09, TX-30, TX-32, TX-33, TX-35, UT-01, UT-02 and UT-03.
  - In the 9 redrawn states this column describes the old seat, not the new district's lean. Use lean data for that.
  - Senate: the current senator's party.
- `open_seat`: 1 if no sitting member of the chamber is on the general ballot. A House member running in a renumbered district
  makes it not open, for example FL-20 with Wasserman Schultz.
- `is_major`: the D and R nominees, with these exceptions:
  - In top-two races (CA, WA), both candidates.
  - Forced independents: NE-SEN Osborn, MT-SEN Bodnar, AK-AL Bill Hill, ID-SEN Todd Achilles, SD-SEN Brian Bengs. In NE-SEN, MT-SEN, ID-SEN and AK-AL the Democrat is set to 0. NE-SEN has no Democrat on Wikipedia's list.
  - With only one major-party nominee, the single strongest-label minor candidate (IND, then LIB, GRN, OTHER) fills the second slot, if one is unique: AZ-03 Redkey (GRN), MA-01 Milleron, PA-03 Mahoney.
  - With several same-party candidates (LA jungle), the incumbent is picked if running. Otherwise **no one is flagged**, with a note.
  - Races without exactly two majors: the 5 uncontested races, NJ-08 (two IND opponents, ambiguous), LA-03 and LA-04 (incumbent only), LA-05 and LA-06 (none).
- `special`: 1 only for FL-SEN and OH-SEN (class 3 specials). There are no House specials on Nov 3. The 2025-26 House specials (TX-18, GA-14, NJ-11, CA-01, CA-14, GA-13) are already decided, and their winners appear as incumbents.
- `system`:
  - `top-two`: CA, WA.
  - `rcv`: AK (top-four primary, RCV general), ME.
  - `runoff-possible`: GA (majority rule, Dec 1 runoff) and **all six LA House races**. After the May 2026 map, LA's party primaries were cancelled. Nov 3 is an all-party jungle election with a Dec 12 runoff, so several candidates of the same party are on one ballot.
  - LA-SEN kept its closed primaries, so it is `plurality`.
  - Everything else is `plurality`.
- `uncontested`: 1 if only one candidate is listed: FL-10 (Frost, declared elected without appearing on the ballot), MA-02, MA-05, MA-07, WI-02.
  Races with no D or no R nominee are noted ("no REP nominee") but not flagged uncontested: the 9 same-party CA races, CA-06, AZ-03, MA-01, NJ-08, PA-03, ID/NE/SD-SEN.
- `notes`: non-standard party label, redistricting or old-district info, Wikipedia's incumbent status text ("Incumbent
  retiring", "lost renomination", ...), system notes and Kalshi discrepancies.

## Caveats

- **Redistricting:** AL, CA, FL, LA, NC, OH, TN, TX and UT use new lines in 2026 (see `redistricting_2026.csv`). District numbers there
  do **not** line up with 2024 results. Only the members on each race's incumbent rows are linked; vote-share history must be
  re-mapped separately.
- Wikipedia is crowd-edited. Names were cross-checked against Kalshi (above) but not against each state's certified ballot list.
  Ballotpedia was not scraped for this file (kavi-a separately parsed nine Ballotpedia pages for California primary
  results; see `models/kavi-a/data/kavi-a/ca_primaries_2026.md`).
- Minor / write-in completeness varies by state. Some states' tables list only ballot-qualified candidates.
- "Major" for the LA jungle races and NJ-08 could not be set without polling data. These are left unassigned rather than guessed.
- CA-11 (Pelosi retiring) is an open same-party race: Wiener vs Chan, both DEM.

## Verifier pass (2026-11-01)

An independent check loaded the CSV with pandas and rebuilt it with the script.
- **Checks that passed:** schema, 470/470 race_ids, no duplicate candidates, and race-level fields consistent within each race.
  Every Kalshi-named candidate maps to a row, apart from the 5 names listed above and two spelling variants (KY-03 Rodriquez/Rodriguez, NY-09 Azumah/Anabilah-Azumah).
  No party disagreed with Kalshi.
- **Fix:** `incumbent_party_of_seat` was blank for 8 House races (AL-01, CA-38, FL-22, TX-09, TX-30, TX-32, TX-35, UT-01), which had been
  labelled "brand-new seats". Every district number has a current holder, so these were filled. CA-06 changed from IND (Kiley, redistricted
  in) to DEM (Bera holds CA-06). CA-41 changed from DEM (Sánchez) to REP (Calvert holds CA-41). The rule above is now the same for every House race.
- **Second-source spot checks (news searches; Ballotpedia blocked this verifier's fetch):** all agree with the data.
  - Random rows: WA-09 Smith/Basler, MS-01 Kelly/C. Johnson (+ Baucom LIB), CT-01 Bronin beat Larson / open, AZ-08 Hamadeh/Greene-Placentia, OK-04 Cole/Jacob/Bonacci, MI-13 McKinney beat Thanedar.
  - Tricky races: CA-06 Kiley left the GOP and runs as no party preference against Pan, CA-40 Calvert vs Kim (both R),
    NJ-08 no Republican filed, UT-01 McAdams/Owen/West/Montgomery on the court map, OK-SEN Armstrong appointed and pledged not to run (Hern R),
    LA-05 jungle field (Cathey, Firment, Echols, Cordell) with Nov 3 open primary and Dec 12 runoff.
- **Still open:** is_major is unassigned in LA-05 and LA-06, is incumbent-only in LA-03 and LA-04, and is ambiguous in NJ-08. Minor-candidate completeness was not checked against state ballot lists.
