# national_hist.csv: national environment per election year, 2010-2024

Built by `src/data/fetch_polls.py` (pulled 2026-11-01). There is one row per election year.

| year | type | generic D-R (final avg) | president | approve / disapprove (net) | actual House vote D-R |
|---|---|---|---|---|---|
| 2010 | midterm | -6.72 | Obama (D) | 44.5 / 48.0 (-3.5) | -6.76 |
| 2012 | pres | +2.45 | Obama (D) | 51.0 / 45.0 (+6.0) | +1.02 |
| 2014 | midterm | -1.89 | Obama (D) | 42.0 / 53.0 (-11.0) | -5.95 |
| 2016 | pres | +2.36 | Obama (D) | 53.5 / 44.5 (+9.0) | -1.08 |
| 2018 | midterm | +8.61 | Trump (R) | 42.0 / 53.0 (-10.9) | +8.57 |
| 2020 | pres | +7.26 | Trump (R) | 44.1 / 52.9 (-8.8) | +3.10 |
| 2022 | midterm | -1.23 | Biden (D) | 42.3 / 52.4 (-10.1) | -2.83 |
| 2024 | pres | +0.67 | Biden (D) | 38.7 / 56.1 (-17.5) | -2.55 |

## Columns
- `generic_dem`, `generic_rep`, `generic_margin_dr`, `generic_source`
- `president`, `president_party`, `approve`, `disapprove`, `net_approval`, `approval_source`
- `house_dem_share`, `house_rep_share`, `house_margin_dr`, `house_total_votes`, `house_vote_source`

## Sources and method
- **Generic ballot**: the final 538 average on or before election day.
  - 2010-2016: `congress-generic-ballot/generic_topline_historical.csv` (GitHub fivethirtyeight/data).
  - 2018-2024: `polls/data/generic_ballot_averages.csv` (Wayback snapshot of 2024-12-03).
- **Approval**: the mean of the daily average over the 7 days before election day.
  - 2018/2020: 538 Trump `approval_topline.csv` ("All polls"; Wayback snapshot of 2024-12-08).
  - 2022/2024: 538 Biden `approval_topline.csv` (Wayback snapshot of 2024-11-02).
  - 2010-2016: **Gallup weekly** readings, from the UCSB American Presidency Project
    (https://www.presidency.ucsb.edu/statistics/data/barack-obama-public-approval), averaging Gallup readings that
    end in the 14 days before election day. This is a single pollster, not an average, so it differs in method
    from 2018 onward.
  - 2024 lists Biden as the sitting president; Harris was the nominee.
- **House popular vote**: computed from MIT Election Lab `1976-2024-house` (`data/raw/mit/`; that file is
  comma-separated despite its `.tab` extension).
  - Only regular general-election rows are counted; specials and runoffs are excluded.
  - Fusion lines (NY, CT) count toward the candidate's major party.
  - DFL (MN) and Democratic-NPL (ND) count as D.
  - Shares use total votes, including third parties and write-ins.
  - Check against official Clerk figures: 2010 R+6.6, 2012 D+1.2, 2014 R+5.7, 2016 R+1.1, 2018 D+8.6, 2020 D+3.1,
    2022 R+2.7, 2024 R+2.6. Ours agree within about 0.3 points. Uncontested races that list no votes (some FL
    seats) make the national vote slightly noisy.

## Caveats
- **Generic-ballot polling error (final average minus actual)**: 2012 +1.4 D, 2014 +4.1 D, 2016 +3.4 D, 2020 +4.2 D, 2022
  +1.6 D, 2024 +3.2 D. In 2010 and 2018 the average was close to the result.
- **The 2026 comparison point** is in national_2026.csv: the aggregates of 2026-10-30 put it at about D+8 to D+9.
- **Only 8 rows**, so any regression on these has very few degrees of freedom.
