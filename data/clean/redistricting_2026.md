# redistricting_2026.csv: provenance

One row for each of the 44 states with more than one House district: `state, state_name, n_districts, new_map_for_2026, how,
wiki_status, date, notes`. Built by `src/data/fetch_candidates.py`.

## Sources (pulled 2026-11-01)

- Summary table "Summary of mid-decade changes to congressional districts in advance of the 2026 election cycle" on
  https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections (revid 1383103767). This gives `wiki_status`.
  The script warns if the hand-coded list of new maps disagrees with it. Today it agrees.
- The per-state pages for Louisiana (SB 121 signed 2026-05-29), Tennessee (HB 7003 signed 2026-05-07) and Alabama
  (legislature, May 2026), cached in `data/raw/candidates/`.
- `how`, `date` and `notes` are hand-coded from those texts (the `NEW_MAPS` / `ATTEMPTED_NOT_USED` dicts in the script).

## States whose 2026 lines differ from 2024 (new_map_for_2026 = 1)

| State | How | Date | Main effect |
|---|---|---|---|
| TX | legislature | 2025-08-29 | +5 R target; SCOTUS let it stand 2025-12-04 |
| CA | ballot measure (Prop 50) | 2025-11-04 | +5 D target, replaces the commission map |
| NC | legislature | 2025-10-22 | NC-01 made more R |
| OH | commission | 2025-10-31 | OH-01 more competitive, OH-09 more R, OH-13 more D |
| UT | court | 2025-11-10 | court-adopted Map 1 creates a Salt Lake D seat |
| FL | legislature | 2026-05-04 | wide redraw, +3 to 5 R target |
| TN | legislature | 2026-05-07 | Memphis split (TN-09) |
| AL | legislature | 2026-05 | post-Callais; ends the 2024 court remedial map |
| LA | legislature | 2026-05-29 | removes one of two Black-plurality D seats; Nov 3 becomes a jungle election |

Attempted but **not** in effect for 2026 (2024 lines used): MO (2025 map frozen by referendum; SCOTUS on 2026-09-25 left the 2022 map in place),
VA (referendum invalidated 2026-05-08), GA (June 2026 special session dropped redistricting), IN, MD, KS, SC, NY (NY-11), WI, AR.

## Caveats

- `date` is the enactment or approval date, not the court-clearance date. AL's map was cleared by SCOTUS on 2026-06-02, and TX's on 2025-12-04.
- "Same lines as 2024" means no mid-decade change. GA, NY, NC, AL and LA already had new lines in 2024 compared with 2022.
- At-large states (AK, DE, ND, SD, VT, WY) are excluded by design.

## Verifier pass (2026-11-01)

These facts were checked against news reports:
- LA SB 121 signed 2026-05-29/30 (Fox 8), with Nov 3 open primary and Dec 12 runoff (KALB, WWNO).
- FL map signed on a Monday after Callais (WPTV; matches 2026-05-04).
- TN map passed and signed 2026-05-07, and Cohen quit 2026-05-15 (Tennessee Lookout).
- AL: SCOTUS allowed the map on 2026-06-02 (NPR). NPR calls it the legislature's 2023 plan, while Wikipedia calls it the May 2026 lines. This is noted in the row.
- MO: Missouri Supreme Court 2026-09-03 and SCOTUS around 2026-09-25/26 kept the 2022 map (AP via ADN).
- VA: amendment voided 2026-05-08 (WHRO).
- GA: no 2026 redraw. A note was added.

No state's new_map_for_2026 flag changed. `how` = "commission" for OH is outside the suggested legislature/court/ballot-measure values, and was kept because it is accurate.
