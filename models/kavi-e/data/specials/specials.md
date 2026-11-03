# data/specials: special elections and off-year governor races (kavi-e)

Collected 2026-11-01 for the national-environment signal of the kavi-e model.

## specials_clean.csv (built by src/models/kavi-e/specials.py from raw/)
- Source: The Downballot (formerly Daily Kos Elections) "D-vs.-R special elections vs. presidential results"
  spreadsheets, one per cycle, linked from https://www.the-downballot.com/p/data#special-elections:
  2017-18 `1C2MVeM2K7WgqmJw5RCQbWyTo2u73CX1pI8zw_G-7BJo`, 2019-20 `1dXQhlzCAw05fC7P21eCivdfLOB-F3nykPVRPzG49U9g`,
  2021-22 `1G1SeLtlfDY9xqqVfFTsoh0jwr6h7aQzOOWpJ29aez2w`, 2023-24 `1ajyphWQru9TgDDiBe8kvEmApBEXND2wl9MVaxi1dndk`,
  2025-26 `1JGk1r1VXnxBrAIVHz1C5HTB5jxCO6Zw4QNPivdhyWHw`, plus the Special Elections Index 1989-2022
  `1mP8z1TWsrJbWGXk_pz3QgOZQpAmQxLY6sZjPp4qzY6c` (downloaded, not used: different method).
  Every tab exported as CSV with `src/models/kavi-e/fetch_specials.py` into `raw/<cycle>_<gid>.csv`.
- Cite as: The Downballot (Daniel Donner, James Lambert, David Nir), the-downballot.com.
- Content: every congressional and state-legislative special with exactly one D and one R, the special's D% / R%,
  and the district's most recent presidential D% / R% (2016 for 2017-20, 2020 for 2021-24, 2024 for 2025-26).
- Columns: date, state, district, excluded_star, dem_pct, rep_pct, special_margin, pres_dem_pct, pres_rep_pct,
  pres_margin, pres_base_year, overperf (= special margin - presidential margin, points), is_federal, source_file,
  included.
- `included` follows The Downballot's averaging rule: not starred, and D% + R% >= 90.
- Percentages in the export are rounded to whole points, so each overperf carries about +-1 point of rounding.
- Means by year (included rows, our parse; The Downballot's printed averages in brackets): 2017 10.9 [11.2],
  2018 11.0 [9.9, ours includes a few post-election December rows that the model's date window drops],
  2019 4.5 [4.7], 2020 6.2 [5.0], 2021 -4.8 [-4.4], 2022 -2.6 [-2.5], 2023 5.3 [5.2], 2024 0.7 [0.8],
  2025 13.9 [13.9], 2026 10.4 [10.4] (43 races through 2026-08-18).
- Caveat: low-turnout contests; a few extreme outliers (NY SD-22 2025 +90). The model uses a 10% trimmed mean.

## offyear_governors.csv (hand-entered)
- VA and NJ governor results of the odd year before each midterm (2009, 2013, 2017, 2021, 2025) and the state's
  previous presidential result, plus the national presidential D-R margin of that election.
- 2009-2021 values are the certified results as printed on the Wikipedia pages for each race (entered from those
  pages' infobox percentages); 2025 values are the near-final counts reported by MasonVotes / NBC / 270toWin
  (Spanberger 57.4-42.4, Sherrill 56.2-43.2). Presidential shares: Wikipedia state results pages.
- These are low-weight signals (sd ~4-6 points); a 0.5-point error in any entry moves the 2026 national estimate by
  less than 0.03 points.
