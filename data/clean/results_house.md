# results_house.csv: U.S. House general-election results, 2000-2024

Built by `src/data/fetch_results.py` (run `.venv/bin/python src/data/fetch_results.py`, or `--offline` to rebuild
from `data/raw/mit/`). Pulled 2026-11-01.

## Source

- MIT Election Data + Science Lab, "U.S. House 1976–2024", Harvard Dataverse, doi:10.7910/DVN/IG0UN2, version 15.0
  (released 2026-03-09). File `1976-2024-house.tab` (comma-separated despite the extension), Dataverse file id
  13592823, MD5 `a9566ac393913a1af343284e837c17bb`, MEDSL row version stamp 20250910.
  - Direct URL: https://dataverse.harvard.edu/api/access/datafile/13592823?format=original. This URL needs a
    guestbook response (name, email, institution, position), so a scripted download without one fails with HTTP 400.
  - So the script falls back to public GitHub mirrors of the same file and accepts one only if its MD5 equals the
    checksum Dataverse publishes for v15. The file in `data/raw/mit/` is therefore byte-identical to the official
    one. Mirror used:
    https://raw.githubusercontent.com/sta199-f26/sta199-f26.github.io/main/_data/gerrymander/1976-2024-house.tab
    (second mirror: Hackquantumcpp/snoutcounter-midterms-2026-model, same MD5).
  - Underlying data come from the Clerk of the House, "Statistics of the Congressional Election" (see
    `data/raw/mit/codebook-us-house.md`; the codebook copy comes from the mirror and is not checksum-verified).
- Wikipedia was used only to confirm a few runoff facts (see the Senate note). No Wikipedia vote counts enter this
  file.

## Shape

5,656 rows: 13 cycles × 435 regular races (checked: exactly 435 per year), plus 1 special (TX-22 2006, the
unexpired DeLay term won by Shelley Sekula-Gibbs). DC's non-voting delegate (in MEDSL for 2020 and 2024) is dropped.
Seat totals match the historical record every year, e.g. 2018 D235/R200 (NC-09 counted for Harris, see below),
2022 R222/D213, 2024 R220/D215. Other winners: Sanders (VT-AL 2000-04) and Goode (VA-05 2000), both independents.

| column | meaning |
|---|---|
| year, state, district | district 0 = at-large |
| race_id | `AL-01`, `WY-AL` (template style) |
| special | 1 = election for an unexpired term (only TX-22 2006 exists in MEDSL; see gaps) |
| dem_cand / rep_cand | leading on-ballot D / R candidate (blank if none) |
| dem_votes / rep_votes | **sum** over all on-ballot D (R) candidates, all fusion lines included |
| total_votes | all valid votes including minor parties, write-ins and scattering; blank/over/under/void/exhausted rows excluded |
| dem_share_2p | dem_votes / (dem_votes + rep_votes). Equals 1.0 or 0.0 when only one major party ran; blank for top-two same-party races |
| winner_party | DEM / REP / OTHER, from the winner's ballot line (fusion: the major-party line) |
| winner_name | MEDSL spelling, upper case |
| uncontested | 1 if only one major party had an on-ballot candidate (676 rows; includes the 50 same-party top-two races) |
| notes | `;`-separated flags (see below) |
| runoff | 1 if the race was decided in a runoff that MEDSL contains (LA-05 2002, TX-23 2006); votes are then the runoff's |
| round1_dem_share_2p | November first-round share for those runoff races |
| winner_share | winner votes / total_votes |

Incumbency is **not** in MEDSL, so it is left to the candidates dataset.

## Cleaning rules and caveats

- **Fusion (NY, CT, and some SC/VT rows):** each candidate's rows are summed across party lines and assigned to the
  candidate's major-party line (395 rows say `fusion:`). In CT 2022-24, MEDSL lists the Working Families line as
  a "candidate" named `WORKING FAMILIES` with no party; those votes go to the D nominee (noted). CT's
  `INDEPENDENT` party-line rows (2-5k votes) are unattributed and count as OTHER.
- **Top-two (CA from 2012, WA from 2008):** flagged `top-two general`. For the 50 same-party generals (D vs D or
  R vs R), dem_share_2p is blank, dem_votes/rep_votes hold the sum for that party, and uncontested=1.
- **Louisiana:** the November ballot is a jungle election (except the closed-primary years 2008-2010). MEDSL has
  only the November round, apart from LA-05 2002, where it has the December runoff (flagged runoff=1). Seven races
  had no November majority and went to a December runoff that is **not** in MEDSL: LA-02 2006, LA-03 2012, LA-05
  2014, LA-06 2014, LA-03 2016, LA-04 2016, LA-05 2020. For these, votes are the November round (party sums over
  multiple same-party candidates), and the script asserts that the November leader is the known runoff winner (it
  is in all seven).
- **TX 2006 (LULAC v. Perry):** districts 15, 21, 23, 25 and 28 held court-ordered open November elections. MEDSL
  flags them special, but they elected the full term, so here special=0 with a note. TX-23 went to a December
  runoff (Rodriguez D beat Bonilla R), which MEDSL stores as non-special rows; the row uses the runoff (runoff=1).
- **Ranked-choice (ME from 2018, AK from 2022):** MEDSL is inconsistent. ME-02 2018 and 2024 look like final-round
  counts, ME-02 2022 has an odd total with an "EXHAUSTED BALLOT" row, and AK 2022 and 2024 are first-round counts.
  Winners are right in every case. AK-AL 2022 has dem_share_2p 0.498 even though Peltola (D) won: R votes are
  Palin + Begich summed and she won on RCV.
- **Unopposed, not on the ballot (FL, OK, AR, LA):** MEDSL puts a vote count of 1 (or 0). Votes are blanked (46
  rows), and the note says so. dem_share_2p keeps the uncontested convention (1.0/0.0); before the verifier fix
  FL-24 2016 and FL-25 2020 (MEDSL 0 votes) had it blank. dem_share_2p is now blank only for the 50 same-party
  top-two races.
- **Party label fixes:** 2002 VA-09 Katzen (party blank, flagged write-in) is coded REP. 2006 TX-22 Sekula-Gibbs
  (GOP write-in) is coded REP and counted as on-ballot.
- **NC-09 2018:** MEDSL lists Harris (R) as the winner, but the result was never certified; the 2019 special was won
  by Bishop (R). This is noted.
- Names: MEDSL's escaped quotes (`\"`) and UTF-8 mojibake (`RAÃ\x9aL`) are repaired, and names are upper-cased.
  Names are **not** harmonized across years (e.g. `JARED F GOLDEN` vs `JARED F. GOLDEN`).
- 16 rows contain counts MEDSL marks unofficial.

## Verification (2026-11-01, independent verifier)

Checked: schema, 435 regular races per cycle, no duplicate keys, race_id = state + district, dem_share_2p =
dem/(dem+rep) everywhere, D+R <= total, winner = leading D/R candidate, seat totals per cycle (all match the
historical record). 2024 race_ids equal the 435 template House ids. Spot-checked against Wikipedia
(dem/rep/total votes): CT-05 2000 (margin), AZ-04 2000, FL-05 2008, TX-11 2010, VA-07 2010, DE-AL 2012, AZ-02 2014,
MI-05 2014, OH-01 2016, OH-07 2018, NC-09 2018, IL-10/NY-10/NY-03/CO-08/TX-28/UT-03 2022, CA-22/NE-02/AK-AL 2024,
TX-22 (regular and special) and TX-23 runoff 2006: all match. NY-16 2020 and NY-01 2024 totals differ from
Wikipedia only by write-ins/blank handling. ME-02 2018 (139,231 / 136,326) matches the Maine SOS RCV final-round
release; Wikipedia shows 142,440 / 138,931, apparently a later certified count. 2024 AK-AL is round 1 (Begich
159,550; Peltola 152,828 + Hafner 3,417 as D sum). Only fix needed in the House file: the two blank shares above.

## Coverage gaps

- **Special elections:** MEDSL keeps only November-cycle results. Same-day specials for unexpired terms (other than
  TX-22 2006) and off-cycle specials are missing. They are not needed for 2026 regular races.
- **District lines:** race_id uses the lines in force that year. Lines changed in 2002, 2012 and 2022 (national
  redistricting); for 2026, states including TX, CA (Prop 50), NC, OH, UT, FL, TN, LA and AL redrew mid-decade (see
  `data/clean/redistricting_2026.csv` from the redistricting dataset). Other states also redrew mid-decade earlier
  (e.g. PA 2018, NC 2016/2020, VA 2016, FL 2016, AL 2024, LA 2024, GA 2024, NY 2024). **Do not join 2024 results to
  2026 race_ids as "same district" in redrawn states**; use a lean/PVI recomputed on 2026 lines instead. The 2024
  race_id set happens to equal the 2026 template's 435 ids, but in redrawn states the same id is a different
  district.
- Only 2000-2024 is output (raw file goes back to 1976; change `FIRST_YEAR` to extend).
