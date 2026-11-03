# data/benchmark: FiveThirtyEight's final 2018 and 2022 forecasts, and the winners

Used only to score the 2018 and 2022 backtests and to list those years' candidates. No 2026 forecast reads these files.

| File | Source |
|---|---|
| `fte_final_2018.csv` | FiveThirtyEight, 2018 House and Senate forecasts ("classic" model, final run of 2018-11-05): `house_district_forecast.csv` and `senate_seat_forecast.csv` from https://projects.fivethirtyeight.com/congress-model-2018/, retrieved from Internet Archive (Wayback Machine) snapshots of 2018-11-06. |
| `fte_final_2022.csv` | FiveThirtyEight, 2022 House and Senate forecasts ("deluxe" model, final run): `house_district_toplines_2022.csv` and `senate_state_toplines_2022.csv` from https://projects.fivethirtyeight.com/2022-general-election-forecast-data/, retrieved from Wayback Machine snapshots of 2023-04-27 (stamps 20230427025551 and 20230427025614). |
| `winners_2018.csv`, `winners_2022.csv` | Winners of each race, from the MIT Election Data and Science Lab returns (see `data/raw/mit/README.md`). |

**Licence.** FiveThirtyEight released this data under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). Attribution: FiveThirtyEight. Changes: one row per
candidate, a subset of the columns, renamed, with our race identifiers; built by `src/benchmark_fte.py`. The original
site (projects.fivethirtyeight.com) is offline, so the files were retrieved from the Internet Archive's copies.
