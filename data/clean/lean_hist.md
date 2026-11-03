# lean_hist.csv: prior presidential lean for every House and Senate general election, 2010-2024

Built by `src/data/fetch_lean.py`. Pulled 2026-11-01. 3,751 rows: 8 cycles x 435 House, plus 271 Senate state-years.
Key: (year, race_id). It is unique.

## Columns

| column | meaning |
|---|---|
| year | general-election year (2010, 2012, ..., 2024) |
| race_id, chamber | House `ST-NN` / `ST-AL` on **that year's lines**. Senate `ST-SEN` |
| pres_prev_dem_2p | Dem two-party share in the most recent presidential election **before** `year`, on that year's lines: 2010<-2008, 2012<-2008, 2014<-2012, 2016<-2012, 2018<-2016, 2020<-2016, 2022<-2020, 2024<-2020 |
| pres_prev2_dem_2p | the presidential election before that (optional; see gaps) |
| national_pres_dem_2p | national Dem two-party share in the `pres_prev_year` election (MIT, includes DC) |
| pres_prev_year, pres_prev2_year, national_pres_prev2_dem_2p | *extra.* Which elections those were, and the national share for prev2 |
| source | sheet or file used |

National Dem two-party share (MIT): 2004 0.48767, 2008 0.53771, 2012 0.52010, 2016 0.51163, 2020 0.52268, 2024 0.49250.
A PVI-style relative lean is `pres_prev_dem_2p - national_pres_dem_2p`.

## Sources: one Downballot / Daily Kos Elections sheet per set of lines

These are listed in The Downballot's data guide (https://www.the-downballot.com/p/data) and its historical archive
(https://www.the-downballot.com/p/the-downballots-calculations-of-presidential). Each is exported as CSV from the
public Google Sheet, and the raw copies are in `data/raw/lean/`.

| House year | lines | sheet id (gid) | raw file | prev / prev2 |
|---|---|---|---|---|
| 2010 | 2006-2010 lines | 1l7W130tPRF6dQ4JoqlQJSJexnEg4rZct-7kwDXqgoLE (429358610) | db_pres08_2006-2010lines.csv | 2008 / 2004 |
| 2012 | 2012-2014 lines | 1xn6nCNM97oFDZ4M-HQgoUT3X4paOiSDsRMSuxbaOBdg (0) | db_pres08-12_2012-2014lines.csv | 2008 / - |
| 2014 | 2012-2014 lines | same | same | 2012 / 2008 |
| 2016 | 2016 lines (FL, VA, NC redrawn) | 1VfkHtzBTP5gf4jAu8tcVQgsBJ1IDvXEHjuMqYlOgYbA (0) | db_pres08-16_2016lines.csv | 2012 / 2008 |
| 2018 | 2018 lines (PA redrawn) | 1zLNAuRqPauss00HDz4XbTH2HqsCzMe0pR8QmD1K8jk8 (0) | db_pres08-16_2018lines.csv | 2016 / 2012 |
| 2020 | 2020 lines (NC redrawn) | 1XbUXnI9OyfAuhP5P3vWtMuGc5UJlrhXbzZo3AwMuHtk (0) | db_pres08-20_2020lines.csv | 2016 / 2012 |
| 2022 | 2022 lines (post-census) | 1CKngqOp8fzU22JOlypoxNsxL6KSAH920Whc-rd7ebuM (1871835782) | db_pres20_2022lines.csv | 2020 / - |
| 2024 | 2024 lines (AL, GA, LA, NC, NY redrawn) | 1ng1i_Dm_RMDnEvauH44pgE6JCUsapcuu8F2pCfeLWFo (1491069057) | db_pres24_2024lines.csv | 2020 / - |

Senate: statewide results from MIT U.S. President 1976-2024 (doi:10.7910/DVN/42MVDX). The list of states with a Senate
general election each year comes from `data/raw/mit/1976-2024-senate-state.csv` (stage `gen`, years 2010-2024).
That MIT file omits **Connecticut 2022** (Blumenthal v. Levy), so the script adds it by hand
(`SENATE_MISSING_FROM_MIT`; added by the verification pass on 2026-11-01). The script now fails unless the Senate
state-year counts are 36/33/34/34/33/34/34/33 for 2010-2024, which match the class 1/2/3 seat lists plus specials.
Note: `data/clean/results_senate.csv` (built from the same MIT file) has no CT 2022 row either, so a join on it will drop CT 2022.

## Coverage and gaps

| year | House rows | House prev blank | House prev2 blank | Senate rows |
|---|---|---|---|---|
| 2010 | 435 | 0 | 0 | 36 |
| 2012 | 435 | **53 (all CA)** | 435 (not published) | 33 |
| 2014 | 435 | 0 | 53 (CA) | 34 |
| 2016 | 435 | **27 (all FL)** | 91 (CA 53, FL 27, VA 11) | 34 |
| 2018 | 435 | 0 | 27 (FL) | 33 |
| 2020 | 435 | 0 | 27 (FL) | 34 |
| 2022 | 435 | 0 | 435 (not published) | 34 |
| 2024 | 435 | 0 | 435 (not published) | 33 |

- The blanks are blank in the Downballot/DKE sheets themselves. DKE never computed 2008 on California's 2012 lines, or
  2012/2008 on Florida's court-drawn 2016 lines. Possible fill for 2012 CA: use the 2014 row's `pres_prev2` (also
  blank), or the 2012 result itself, which is concurrent and so not "prior". We left them blank rather than guess.
- **2010 values come from integer percentages** (the 2006-2010 sheet rounds to whole points), so their two-party shares
  carry about ±0.5 pt of rounding noise. All other years have one decimal, or exact vote counts for 2024 lines.
- **Senate specials:** one row per state-year. When a state held a regular and a special Senate race in the same year
  (2010 DE/IL/NY/WV, 2014 HI/OK/SC, 2018 MN/MS, 2020 AZ/GA, 2022 CA/OK, 2024 CA/NE), both races share this row.
  Results tables that key specials separately must join on state + year. The Jan 2021 GA runoffs belong to year 2020.
- **Redistricting:** race_id is on each year's own lines, so `NC-13` in 2018 and `NC-13` in 2022 are different places.
  Join each year's results to the same year here, and never across years by district number. Mid-decade redraws
  covered: 2016 FL/VA/NC, 2018 PA, 2020 NC, 2024 AL/GA/LA/NC/NY. MT is `MT-AL` through 2020 and MT-01/02 from 2022.
- Like lean_2026, this says nothing about uncontested, top-two (CA/WA same-party), or RCV races. Filter those using
  the results tables.

## Independent verification (second agent, 2026-11-01)

- Fixed: CT-SEN 2022 was missing (inherited from the MIT senate file). Now 271 Senate state-years.
- House race_ids for every year 2010-2024 match `data/clean/results_house.csv` exactly (435/435, no strays).
  Seat counts per state follow apportionment (e.g. TX 32 -> 36 -> 38, MT 1 -> 2 in 2022).
- Spot checks vs memory of DKE/press figures: 2018 PA-01 0.510, PA-17 0.486, 2018 NY-11 0.450 (2012: 0.522),
  2022 TX-15 0.485, 2020 GA-07 0.467 (2012: 0.389), 2018 MN-SEN 0.508, 2022 CT-SEN 0.602: all plausible/match.
- Source quirk left as is: DKE gives OK-05 identical 2012 and 2008 shares (40.8/59.2), so 2014 and 2016 OK-05 have
  pres_prev == pres_prev2.
