# results_senate.csv: U.S. Senate general-election results, 2000-2024

Built by `src/data/fetch_results.py` (same script as `results_house.csv`). Pulled 2026-11-01.

## Sources

- MIT Election Data + Science Lab, "U.S. Senate statewide 1976–2024", Harvard Dataverse, doi:10.7910/DVN/PEJ5QU,
  version 8.0 (released 2026-05-11). File `1976-2024-senate-state.tab`, saved as
  `data/raw/mit/1976-2024-senate-state.csv`, Dataverse file id 13887039, MD5 `0f8b51cf0f77a2ef0bff992f7a64d1b4`
  (verified). URL: https://dataverse.harvard.edu/api/access/datafile/13887039?format=original (no guestbook).
- Class of each special election: Wikipedia "List of special elections to the United States Senate" (wikitext via
  the API, saved to `data/raw/mit/wikipedia/senate_specials.json`). All 18 specials matched (IL 2010 is now coded as the regular race).
- **CT 2022 (missing from MEDSL v8):** official canvass of the CT Secretary of the State
  (https://electionhistory.ct.gov/eng/contests/view/34715): Blumenthal 701,175 D + 22,689 Working Families =
  723,864, Levy (R) 535,943, write-ins 80, total 1,259,887. The counts are hard-coded in `SENATE_SUPPLEMENT`
  and the build asserts each one appears in the saved Wikipedia wikitext
  (`data/raw/mit/wikipedia/2022_United_States_Senate_election_in_Connecticut.json`, revid 1383127738).
- LA runoff vote checks: the Wikipedia pages for the 2002, 2014 and 2016 Louisiana Senate elections (saved in
  `data/raw/mit/wikipedia/`). They were used only to confirm values, not as input.

## Shape

451 rows, 2000-2024: every regular Class I/II/III race (the build checks 33/33/34 regular races per class
cycle) plus 18 November-cycle specials. The GA January 2021 runoffs are folded
into the 2020 cycle. Regular-race class comes from the cycle (2000/06/12/18/24 = I, 2002/08/14/20/26 = II,
2004/10/16/22 = III). race_id is `OH-SEN`; when a state has two races in one year, the special is `XX-SEN-S`
(MS/WY 2008, NY 2010, OK/SC 2014, MN/MS 2018, GA 2020, CA/OK 2022, CA/NE 2024). A special that is the state's only
race keeps `XX-SEN`, matching the 2026 template's FL-SEN and OH-SEN. Columns are the same as `results_house.csv`
plus `class` (district is always 0). Incumbency is not in MEDSL.

Winner counts by year agree with history (e.g. 2024 D18/R15/I2 counting both CA races; 2020 D15/R20 after the GA
runoffs; 2022 D16/R20 counting both CA races).

## Verifier fixes (2026-11-01, independent check against Wikipedia / state canvasses)

- **CT 2022 was missing** (MEDSL has no CT rows for 2022), so 2022 had 33 of 34 Class III races and the 2022 tally
  read D15/R20. Added from the CT canvass (see Sources); 2022 now has 36 rows.
- **IL 2010 was mislabeled special.** MEDSL has one IL 2010 race with special=True, but its counts (Kirk
  1,778,698 / Giannoulias 1,719,478) are the regular full-term election (Wikipedia: special was Kirk 1,677,729 /
  Giannoulias 1,641,486). Now special=0, class 3, noted; the same-day special itself is not in the file. 2010
  specials are DE, NY-S and WV.
- **OR 2022 names** were `LAST FIRST` with no comma (`WYDEN RON`); now `RON WYDEN`, `JO RAE PERKINS`.
- Spot-checked against Wikipedia (dem/rep/total votes all match): AK 2008, IN 2010, IL 2010, MO 2012, MS 2014,
  LA 2014 (runoff, after the -1,000,000 fix), LA 2016 runoff, CT 2018, MS 2018 special runoff, UT 2024, NE 2024
  regular and special.

## Traps in the MEDSL Senate file (handled)

1. **LA 2014: the runoff votes carry an extra 1,000,000.** MEDSL has Cassidy 1,712,379 and Landrieu 1,561,210; the
   official December runoff is 712,379 and 561,210 (confirmed on Wikipedia). The script subtracts 1,000,000. The
   other LA 2014 rows (Maness etc.) are November values, and the finalists' November rows are missing.
2. **LA 2002 and 2016 mix November and December rows under stage "gen".** In 2002 each finalist appears twice
   (November and December). In 2016 the finalists (Kennedy, Campbell) carry December runoff votes and everyone else
   carries November votes. The script uses the finalists' December votes (runoff=1).
3. **GA 2008:** the November round is coded stage `pre` and the December runoff `gen`. GA 2020: the runoffs are
   year **2021** rows (stage `runoff`). GA 2022: stage `GEN RUNOFF`. All four GA runoffs use the runoff votes, with
   the November share in `round1_dem_share_2p`. Note that GA 2020 regular was led by Perdue (R) in November but won
   by Ossoff (D) in the runoff.
4. **MS 2018 special:** MEDSL has only the November 27 runoff (Hyde-Smith vs Espy), with no jungle round.
5. **party_detailed is blank for many 2020-2024 rows** (NV, MS, WY, AK...). The script falls back to
   `party_simplified`. Where both are blank or uninformative it overrides: WY 2020 (Lummis R, Ben David D), AK 2022
   (Murkowski R, Tshibaka R, Chesbro D, Kelley R), AK 2010 Murkowski write-in (coded R), MS 2008 special (nonpartisan
   ballot; Musgrove coded D). Each override is noted in the row.
6. AK/HI/NV 2022 names are `LAST, FIRST`; the script reorders them to `FIRST LAST`.

## Caveats

- **Independents:** Sanders (VT) and King (ME) are winner_party OTHER, with the note "caucuses with Democrats".
  Lieberman 2006 (CT for Lieberman) is OTHER. When no Democrat ran (VT 2006-24, UT 2022 McMullin, NE 2024 Osborn,
  KS 2014 Orman), dem_share_2p is 0 or 1 with uncontested=1 even though the race was competitive; the note
  `major non-D/R candidate X (y%)` flags these. Treat them separately in a model.
- **Top-two:** CA 2016 and 2018 were D vs D, so dem_share_2p is blank. CA 2022 and 2024 each have a regular race
  and a special for the same seat (same class).
- **Louisiana jungle:** in non-runoff years the November jungle votes are summed by party. LA 2014 and 2016 have no
  usable November first-round share (the finalists' November rows are missing).
- **RCV:** AK 2022 and 2024 counts are first round. ME 2018 and 2024 King won a first-round majority.
- Off-cycle specials (MA Jan 2010, MA Jun 2013, NJ Oct 2013, AL Dec 2017) are **not** in MEDSL and are not here.
  Neither is the IL 2010 same-day special (only the regular race is in MEDSL).
- MS 2018 special: no November jungle round in MEDSL, so `round1_dem_share_2p` is blank (Wikipedia: Hyde-Smith
  389,995, Espy 386,742, McDaniel 154,878).
- Some names keep MEDSL quirks (e.g. LA 2020 `"BILL" CASSIDY`, NM 2020 `BEN R. LUJAN` without the accent).
- 9 rows contain counts MEDSL marks unofficial (WV 2018, GA 2020 both runoffs, SD 2020, MO 2022, AZ/CA-S/NE-S/WA 2024).
- Only 2000-2024 is output (raw file goes back to 1976).
