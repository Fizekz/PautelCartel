# kavi-a v3 (secret round 2): what changed from v2, and the full model in brief

Branch `kavi/kavi-a-1101`, code `src/models/kavi-a/`, outputs `out/` (v2 copies in `out/v2/`, v1 in `out/v1/`).
The v2 description below (sections 1-10) still holds for everything not listed in this section; the v3 changes
are confined to same-party top-two races and one candidate-matching bug. Running log: `LOG.md`, "v3" section.

## v3.1 The model end to end (v3 state)

1. **Data.** Shared Kavi data layer (`data/clean`): MEDSL results 2008-2024, presidential lean (lean, lean2),
   FEC (training: two-year totals and I/C/O flags; test years: pre-election FEC snapshots, 2018 Nov 7 2018 file,
   2022 Aug 3 2022 file; 2026: Oct 14 cut), 538 poll archive (final 60 days) and 2026 Wikipedia/VoteHub polls, core
   ratings (mean of Cook / Sabato / Inside), generic ballot (2026: D+8.56, mean of 5 aggregators, Oct 30), the 85
   Nov 1 research dossiers (2026 only), Kalshi Nov 1 07:48Z slots (names only; `p_kalshi` is never an input).
   **New in v3:** kavi-e's CA/WA top-two primary table 2012-2026 (`data/kavi-a/borrowed/primaries_top_two_kavi-e.csv`,
   copied from origin/kavi/kavi-e-1101 9efb6c7; vote counts from revision-pinned Wikipedia results tables, not
   article/link existence, so no look-ahead leak from current Wikipedia pages).
2. **Two-party races (unchanged from v2).** Stage 1, per chamber, OLS with year fixed effects on D two-party share
   - 0.5: lean, lean2, incumbency, log receipts ratio, previous House result's deviation (+ x same incumbent,
   has-prev). National: year effect = a + b g (b shrunk toward 0.8, prior sd 0.25), residual sd floored at 0.015.
   Stage 2: OLS of the share on the forecast-time fundamentals mean, the weighted poll gap w (poll - 0.5 - mu_F)
   with w = W / (W + k), and a piecewise rating (knot +-1). Error: national + state (0.0155) + Student-t6 race
   (House 0.022 polled / 0.031 unpolled). Calibration: logistic on nested leave-one-year-out predictions,
   logit P = c0 + c1 probit(q) + c2 clip(rating, +-1.5) + Senate interactions, mixed 75/25 with the raw logit
   (CAL_WEIGHT 0.75). 2026 coefficients and fits: section 3 below and `out/model_params.json` (unchanged).
3. **Same-party top-two races (v3, `same_party_fit.py`).** For the two finalists A, B:
   logit P(A wins) = b_share * 10 (s_A - 0.5) + b_inc (inc_A - inc_B), s_A = A's share of the two finalists'
   primary votes, inc = FEC incumbent flag; no intercept (A/B labels arbitrary: both orientations at half weight),
   L2 with C = 1. Fitted on same-party House generals of the training years only, winners from MEDSL:
   | used for | training years | n | b_share (per 10 pts) | b_inc | primary leader won |
   |---|---|---|---|---|---|
   | 2018 backtest | 2012-2016 | 25 | 0.99 | 0.29 | 76% |
   | 2022 backtest | 2012-2020 | 37 | 1.21 | 0.30 | 84% |
   | 2026 | 2012-2024 | 48 | 1.19 | 0.71 | 85% |
   Backtests use P(A) directly (2018: CA-06, CA-08, CA-27, CA-44, WA-09; 2022: CA-15, CA-16, CA-29, CA-34, CA-37);
   races not in the table (CA-SEN 2018) keep the v2 softmax (1.6 logit per incumbency, 0.7 per clipped log
   receipts). 2026: the fitted P(A) is turned into an equivalent share m = 0.5 + 0.09 probit(P), each two-way
   poll of the last 60 days is averaged in at weight 1.5 against the prior's weight 1, q = Phi((m - 0.5) / 0.09),
   then the dossier step (logit weight 0.25) as in v2. All 9 CA races now have a primary share (v2 lacked CA-12,
   CA-29). v2's hand-set prior (shrink 0.6, +0.02 incumbency, +0.01 log money) is retired.
4. **Candidate matching (v3 fix).** Two candidates of one race that match the same FEC row (same surname: WA-09
   2018 Adam Smith, incumbent, vs Sarah Smith) are re-matched by first name; only those clashes are re-matched (a
   global first-name rule broke Mike Kelly / Tom Graves, whose FEC names are George J. / John Thomas).
   Compound surnames (Kamlager vs Kamlager-Dove) match by token in the primary table.
5. **Other special races (unchanged):** D-role independents (NE/MT/ID/SD-SEN, AK-AL), CA-06 Kiley on the R side,
   LA jungle and multi-candidate ballots via the softmax split, minor candidates 0.001, uncontested seats.
6. **Simulation (unchanged):** 20,000 draws, seed 2026: one national shock per chamber, one state shock shared by
   a state's House and Senate races, t6 race noise; same-party and other non-two-party races drawn from q.
7. **Late information (unchanged):** dossier pull, logit weight 0.25, 85 races.

## v3.2 Validation and selection

Pre-registered criterion: two-level LOYO on 2010-2016 (calibrated, contested log loss, v2 = 0.07249), with forward
tests 2020 (fit 2010-2018) and 2024 (fit 2010-2022) as tie-breakers; the official scorecard last, with the round's
rule "revert anything that makes it worse". `select_v3.py` (`--sponsor`, `--calib`, `--gclose`, `--pollster`,
`--dedup`, `--rclip`) reproduces every row; results in `out/select_v3*.json`.

| tested (source of the idea) | pre-2018 | fwd 2020 / 2024 | result |
|---|---|---|---|
| v2 base | 0.07249 | 0.13183 / 0.04885 | |
| no previous-result features (kavi-e) | 0.07459 | 0.13796 / 0.04940 | rejected |
| prev x same-incumbent only | 0.07274 | 0.13231 / 0.04911 | rejected |
| stage-2 WLS toward close races (pau-a), 2 / 5 | 0.07260 / 0.07269 | 0.13234 / 0.04938; 0.13090 / 0.04932 | rejected |
| partisan-sponsor poll shift (pau-a), scale 0.5 / 1 | 0.07304 / 0.07376 | 0.12995 / 0.04931; 0.12675 / 0.04909 | rejected |
| calibration + incumbency / + poll lead / both | 0.07443 / 0.07338 / 0.07530 | 0.1329 / 0.0488; 0.1318 / 0.0484; 0.1326 / 0.0483 | rejected |
| g x closeness in House stage 1 (pau-a, symmetric) | best 0.07218 | 0.13777 / 0.04936 | rejected |
| per-pollster damping (kavi-e) | 0.07260 | 0.13117 / 0.04890 | rejected |
| rating clip 3 (none), CAL_WEIGHT 0.5 | 0.07121 | 0.12374 / 0.05096 | passed, then **worse on the scorecard -> reverted** |
| one row per poll (538 LV/RV duplicates) | n/a (no duplicates pre-2018) | tie | **worse on the scorecard -> reverted** |
| fitted same-party model + WA-09 fix | not testable by LOYO (no same-party rows in the share model) | | adopted, scorecard better |

Scorecard (src/scorecard.py, all 470 races per year; contested = 538 favourite < 0.95):

```
kavi-a 2018: n=470 logloss 0.0814 (538 0.1002) brier 0.0465 (538 0.0578) acc 0.970 score top 45.87 ev 46.53 (chalk 42.13) against-538 11; dropped 0
       2018 contested (538 fav<0.95): n=120 logloss 0.3069 (538 0.3778) brier 0.1809 (538 0.2257) acc 0.883 (538 0.875)
kavi-a 2022: n=470 logloss 0.0806 (538 0.0952) brier 0.0487 (538 0.0576) acc 0.964 score top 33.57 ev 47.33 (chalk 30.43) against-538 12; dropped 0
       2022 contested (538 fav<0.95): n=90 logloss 0.3555 (538 0.4375) brier 0.2319 (538 0.2787) acc 0.822 (538 0.822)
```
v2: 2018 0.0823 / contested 0.3069; 2022 0.0821 / 0.3576. By chamber v3: House 2018 0.0801, Senate 0.0968, House
2022 0.0796, Senate 0.0934 (Senate unchanged).

**Choices that saw 2018/2022 (selection optimism):** (1) the list of ideas came partly from a race-level
comparison with kavi-e on the 2018/2022 backtests (which is how WA-09's 0.50 was spotted); (2) two changes that
passed or tied the pre-2018 / 2020-2024 criteria were reverted because the 2018/2022 scorecard got worse (rating
clip 3 + weight 0.5; poll dedup); (3) the Kamlager-Dove token match was found by reading the 2022 backtest output;
(4) all v1/v2 choices flagged in sections 4 and 9 remain. The same-party coefficients themselves are fitted only on
years before each test year.

## v3.3 2026 forecast (out/pred_2026.csv, sims_summary.json)

Unchanged outside the 9 CA same-party races: House D mean 232.8 (median 233, 80% 221-245), P(D majority) 0.948;
Senate D (incl. King/Sanders) median 52 (80% 49-54), P(D control) 0.772. Same 6 picks against the Kalshi
favourite as v2 (WI-01, MI-04, FL-22, MI-10, PA-01, VA-01). Same-party q (v2 -> v3, Kalshi) for the named
finalist: CA-04 Jones 0.21 -> 0.14 (0.47), CA-07 Matsui 0.61 -> 0.62 (0.64), CA-11 Chan 0.26 -> 0.27 (0.21), CA-12
Joyce 0.03 -> 0.01 (0.02), CA-14 Wahab 0.92 -> 0.95 (0.92), CA-29 Duenas 0.04 -> 0.06 (0.04), CA-34
Gonzales-Torres 0.52 -> 0.51 (0.82), CA-37 Mota 0.003 -> 0.010 (0.025), CA-40 Calvert 0.48 -> 0.49 (0.48).

## v3.4 Known weaknesses (added to section 9)

- The same-party model sees only the two finalists' primary votes and incumbency. It cannot see where the
  eliminated candidates' voters go (CA-04: 36% of the June vote went to Republicans, whom Jones may consolidate;
  Kalshi 0.47 vs our 0.14). 48 training races, 2 coefficients.
- Every two-party improvement I tried failed the held-out criteria; the R lean in R-held toss-ups in wave years
  (2018 CA-21, TX-07, GA-06) is still there.
- The 538 poll history double-counts LV/RV versions in 2018-2024 (known, deliberately left in, see above); 2026
  polls are not deduplicated either, for the same consistency.
- Pre-2018 LOYO (3-year share fits, 4 held-out years) proved a weak guide: the rating-clip change won it and the
  2020/2024 forward tests and still lost on 2018/2022.

## v3.5 Reproduce
As section 10, plus `$PY same_party_fit.py` (prints the three fits) and `$PY select_v3.py [--sponsor|--calib|--gclose|
--pollster|--dedup|--rclip]` for the selection tables. `$PY -m pytest -q test_outputs.py` runs 10 tests.

---

# kavi-a v2 (still valid where v3 does not say otherwise): fundamentals vote-share model with polls, ratings, a calibration layer and a late-information step

Branch `kavi/kavi-a-1101`, code in `src/models/kavi-a/`, outputs in `out/` (v1 copies in `out/v1/`, sensitivity
runs in `out/sens/`). Running log: `LOG.md` (v2 section at the bottom).
One line: a per-chamber least-squares model of the D-side two-party share from presidential lean, incumbency,
receipts and the previous result, with the national swing set by the generic ballot; race polls and expert ratings
update it; a logistic layer fitted on nested leave-one-year-out predictions recalibrates the win probability; for
2026 only, the 85 research dossiers nudge the probability (weight 0.25 in logit); win probabilities and seat totals
come from a three-level error (national, state, Student-t race).

## 1. Data used

All from the shared data layer (`data/clean/*.csv`, read with their `.md` notes) unless marked kavi-a.

| file | years | n used | how it is used / cleaning done here |
|---|---|---|---|
| results_house.csv, results_senate.csv (MIT MEDSL) | 2008-2024 | 3,761 race-years 2010-2024 (+2008 for prior results) | Target `dem_share_2p`; previous House result; `uncontested`; notes flag major independents. Trainable rows (3,278: House 3,018, Senate 260) = contested D v R, share in [0.03, 0.97], lean present, no major non-D/R candidate, excluding AK 2022+ (RCV first-round party sums) and LA House (jungle party sums). |
| lean_hist.csv (Downballot/DKE, MIT president) | 2010-2024 | 3,751 | `lean = pres_prev_dem_2p - national_pres_dem_2p` on that year's lines; `lean2` = the election before (filled with lean when blank). Senate specials share the state row. |
| lean_2026.csv | 2026 lines | 470 | `lean = pres24 - 0.49250`, `lean2 = pres20 - 0.52268` (blank for 145 redrawn districts, filled with lean). |
| fec_hist.csv (FEC weball) | 2010-2024 | 27,186 rows | **Training years**: incumbency (FEC I/C/O flag) and two-year receipts per nominee, surname match inside the race (party bucket first), then reverse-token / compound surnames, fuzzy surname (difflib >= 0.75), and for House a unique statewide surname match. Unmatched nominee (5.5% D, 6% R) = non-incumbent with $0. One correction: Hyde-Smith (MS-SEN-S 2018, appointed) is FEC 'O' while every other appointee 2010-2024 is 'I'; set to incumbent. |
| **kavi-a v2: data/kavi-a/fec_preelection.csv** (`fetch_fec_preelection.py`) | 2018, 2022 | 3,550 / 2,670 candidates | **Test-year receipts.** 2018: Wayback copy of the FEC S3 `weball18.zip` captured 2018-11-07 08:07 UTC, file dated election night (coverage median 2018-09-30; URL found by kavi-b). 2022: Wayback `webl22.zip` captured 2022-08-03 (House and Senate current campaigns; coverage median 2022-06-30). No capture of weball22/webl22/webs22 exists between Aug 3 and Dec 28 2022 (CDX, checked 2026-11-01), so 2022 is the latest pre-election cut available. Join on cand_id, else a unique surname in the race; absent = $0 (5 of 1,411 2022 candidates with >$100k). |
| fec_2026.csv, fec_race_2026.csv | 2025-26 (to 2026-10-14) | 951 slots | Receipts per Kalshi slot; Louisiana party slots use the all-filer party sums of fec_race_2026. |
| candidates_2026.csv | 2026 | 1,282 | Incumbency (appointees Moody/Husted = incumbents), `is_major` (who is the real challenger), system. |
| polls_hist.csv (538 archive) | 2010-2024 | 8,767 rows | General-stage, non-hypothetical polls ending 0-60 days before election day; share = D/(D+R), D replaced by `other_pct` when no D. Weight sqrt(n/600) (n clipped 200-3000, missing 500) x exp(-days/21) x 0.5 if partisan. 2010-2016 files hold only the final ~60 days. Jungle-round polls (MS-SEN-S 2018, GA-SEN-S 2020) are left out. |
| polls_2026.csv (Wikipedia + VoteHub) | 2026 | 1,513 rows | Usable rows per the note; Alaska final RCV round, others first choice; same window and weights. Independent-led races use the IND's `top_other_pct`; VoteHub rows dropped in independent-led races (swapped columns, MT-SEN). **v2:** RedBox Research polls (A Fight Worth Having, a D super PAC) all tagged D; the file had MI-04 and MI-10 blank. |
| ratings_hist.csv / ratings_2026.csv | 2010-2026 | 3,760 / 470 | Core rating = mean of Cook/Sabato/Inside, sign flipped so + = D. **v2:** the quantitative outlets' mean (`other_ddhq/economist/split_ticket` + `other_fte` in history; `ddhq/economist/fiftyplusone/silver_bulletin/split_ticket` in 2026) is built and tested as a blend (section 4); not used in the final q. |
| national_hist.csv / national_2026.csv | 2010-2026 | 8 years + 5 aggregates | Final generic-ballot margin `g = margin/200`; 2026 = mean of the 5 aggregator averages of 2026-10-30 = D+8.56. Midterm flag and president's party (v2 calibration candidates). |
| **repo data/dossiers/summary.csv** (85 race dossiers, Nov 1, skeptic-reviewed) | 2026 | 85 races | `q_dog_mean` (mean of researcher and skeptic probability for the Kalshi underdog). 2026 late-information step only (section 3). |
| benchmark/fte_final_{2018,2022}.csv, winners_{2018,2022}.csv | | 1,130 / 856 rows | Backtest ballot (names, parties); `p_fte` never read by the model. winners_2018 updated to the repo copy (King/Sanders labelled IND; scores unchanged). |
| kalshi/races_2026-11-01T074812Z.csv | | 470 races | Candidate slot names and the `p_kalshi` column (mid / sum of mids). Never a model input. |
| kavi-a: data/kavi-a/ca_primaries_2026.csv (Ballotpedia) | June 2026 | 7 of 9 races | Top-two primary shares for same-party CA races, parsed from the 9 Ballotpedia district pages fetched 2026-11-01 by `fetch_ca_primaries.py` (raw HTML not redistributed). CA-12, CA-29 did not parse. |

## 2. Features and why

- **lean, lean2**: the dominant predictor of district partisanship; two elections smooth candidate swings.
- **inc** (+1 D incumbent, -1 R incumbent, 0 open): personal vote, ~1.8 points of share per side.
- **money** = ln((D receipts + 25k) / (R receipts + 25k)) clipped to +-3.5: candidate quality and competitiveness.
  Training uses two-year totals; backtest test years use the pre-election cut; 2026 stops at Oct 14. The log ratio
  makes the cut date matter less than the level, but the 2022 June cut is a weaker signal (section 5).
- **prev_dev** (House, unchanged lines): last result minus (national House D share + lean), plus its interaction
  with "same incumbent running" and a has-prev dummy. Missing in the 9 states redrawn for 2026.
- **national environment**: year fixed effects in the fit; at forecast time alpha = a + b g with b shrunk toward
  0.8 (prior sd 0.25). The midterm/out-party effect is carried by the generic ballot; an extra midterm term was
  tested again in v2 at the calibration stage and rejected on held-out years (section 4).
- **polls** (stage 2): poll gap = w (poll share - 0.5 - mu_F), w = W/(W+k), W = summed poll weights.
- **ratings** (stage 2 and calibration): piecewise linear, knot at +-1.

## 3. Model and parameters

Stage 1 (per chamber, OLS with year fixed effects, y = share - 0.5). 2026 fit (2010-2024):
House lean 0.549, lean2 0.230, inc 0.0177, money 0.0146, prev_dev 0.205, prev x same-inc 0.054, has-prev -0.006;
Senate lean 0.552, lean2 0.054, inc 0.0194, money 0.0226 (`out/model_params.json` has all three fits).
National mapping: House a = -0.0006, b = 0.690; Senate a = -0.0063, b = 0.621. With g = 0.0428 (D+8.56):
alpha_hat House +0.0290, Senate +0.0203.

Stage 2 (OLS on forecast-time mu_F): House mu_F 0.958, poll gap 0.742 (k 0.25), rating 0.0017/pt inside +-1 and
0.0038/pt outside; Senate mu_F 0.932, poll gap 0.671 (k 0.1), rating 0.0140/pt inside, 0.0010 outside.

Error: national sd = residual sd of the alpha mapping, floor 0.015 (House 0.015, Senate 0.0151); state sd 0.0155
(same-state same-year residual pairs, both chambers); race sd House 0.0221 polled / 0.0305 unpolled, Senate
0.0210 / 0.0384; race error Student t with 6 df.

Calibration layer (calib.py): logistic on nested LOYO predictions within the training years:
logit P = c0 + c1 z + c2 r + c3 s z + c4 s r (z = probit of model q clipped +-3.5, r = rating clipped +-1.5,
s = Senate). 2026 fit: c0 -0.075, c1 1.296, c2 1.481, c3 0.061, c4 0.123. **v2: the final q mixes calibrated and
raw logits 75/25 (CAL_WEIGHT 0.75, was 0.5)**, then is turned back into an equivalent mean share by bisection.

**Late-information step (v2, 2026 only, late_info.py):** for each of the 85 dossier races,
logit p = 0.75 logit p_model + 0.25 logit p_dossier, on P(D side) in two-party races (80) and on the finalists' q
in same-party races (5); the share mean is re-solved so the simulation reproduces the new p. Weight 0.25 is a prior
choice, not fitted: there is no history of such dossiers, and their writers saw the Kalshi prices (their numbers
sit close to them), so a small weight keeps the model's own evidence dominant. Effect: mean P(D) +0.019 over the 85
races; +1.6 expected House D seats (231.2 -> 232.8); largest moves OH-07 +0.083, MI-10 +0.066, MI-04 +0.060, MT-01
-0.033; CA-34 Gomez 0.57 -> 0.48. Log: `out/dossier_adjustments_2026.csv`.

Race types and candidate allocation (races.py), same-party CA (same_party.py) and the independent rules are
unchanged from v1: D-role = Democrat unless a true independent leads the non-R side (2026: `is_major` IND where no
major D: NE/MT/ID/SD-SEN, AK-AL); CA-06 Kiley (IND) on the R side; minor candidates 0.001; several candidates of one
party split by softmax(1.6 incumbency + 0.7 clipped log receipts); same-party CA 2026: primary two-finalist share
shrunk by 0.6, +0.02 incumbency, +0.01 log money, each recent poll averaged at weight 1.5, sd 0.09. Parties in
`pred_2026.csv` are the candidates' real parties (8 D-v-D CA races DEM, CA-40 REP; tested).

Simulation (predict_2026.py, 20,000 draws, seed 2026): one national normal shock per chamber's sd shared by every
race, one state shock shared by all House and Senate races in a state, t6 race noise; non-two-party races drawn
from q. Senate totals add the seats not up (D 34 incl. King/Sanders, R 31); D control needs 51.

## 4. Training and validation

- Backtest 2018: fit on 2010-2016. Backtest 2022: fit on 2010-2020. 2026: 2010-2024. Test-year inputs are all
  pre-election: lean, prior result, 60-day polls, final ratings, generic ballot, ballot list, and (v2) receipts
  from the pre-election FEC snapshot. The test year's results are used only by the scorecard.
- **v2 selection discipline** (`select_v2.py`, `out/select_v2.json`):
  - Calibration spec and weight: two-level leave-one-year-out on 2010-2016 only (mean held-out contested log
    loss): base w0.5 0.07250, **base w0.75 0.07248**, base w1.0 0.0738; midterm x out-party intercept (M = midterm
    x [+1 if D is the out-party, -1 if D holds the White House]) 0.0772-0.0843; M + midterm x z 0.0775; M + M x r
    0.0773; midterm x r 0.0728; midterm x z 0.0729; midterm intercept 0.0772. Every midterm-conditioned variant
    loses. The weight was a tie pre-2018; 0.75 also wins the 2010-2024 table (0.0802 vs 0.0810), which includes
    2018/2022: recorded in LOG as a tie-break that saw the test years.
  - Poll window: 2010-2016 files have only ~60 days of polls, so the 2020 + 2024 forward tests decide: v1 0.18068,
    extension for thin races (61-150 days, tau 30, where < 2 polls in 60 days) 0.18255, 150-day window 0.18070.
    Kept v1.
  - Quantitative-outlet blend r = core + lambda (quant - core), forward 2020 + 2024: lambda 0 0.1807, 0.25 0.1817,
    0.5 0.1834, 1 0.1881. Kept lambda 0 (2024 alone liked 0.25-0.5; 2020, a shared polling miss, did not).
  - Dossier weight: set a priori (no history).
- v1 caveats remain: polls + ratings, the piecewise rating and the calibration layer were chosen after seeing
  forward tests that include 2018/2022; the audit re-derived the calibration motivation from 2010-2016 alone and
  measured the selection effect at <= 0.002 log loss.

## 5. Scorecard (src/scorecard.py, all 470 races per year, out/scorecard.json)

```
kavi-a 2018: n=470 logloss 0.0823 (538 0.1002) brier 0.0474 (538 0.0578) acc 0.970 score top 45.87 ev 46.53 (chalk 42.13) against-538 11; dropped 0
       2018 contested (538 fav<0.95): n=120 logloss 0.3069 (538 0.3778) brier 0.1809 (538 0.2257) acc 0.883 (538 0.875)
kavi-a 2022: n=470 logloss 0.0821 (538 0.0952) brier 0.0500 (538 0.0576) acc 0.964 score top 33.57 ev 47.66 (chalk 30.43) against-538 12; dropped 0
       2022 contested (538 fav<0.95): n=90 logloss 0.3576 (538 0.4375) brier 0.2348 (538 0.2787) acc 0.822 (538 0.822)
```

By chamber: House 2018 0.0811 (538 0.0938), Senate 2018 0.0968 (0.1799); House 2022 0.0812 (0.0914), Senate 2022
0.0934 (0.1414). Pure fundamentals (variant F, de-leaked, `out/scorecard_F.json`): 2018 0.1358, 2022 0.0994.
The dossier step has no backtest (2026 only).

## 6. 2026 forecast and the picks against the Kalshi favourite

Simulation: House D seats mean 232.8, median 233, 80% interval 221-245 (95% 215-251), P(D majority) 0.948; sum of
q over D candidates 232.6 vs Kalshi 246.0. Senate D seats (incl. King/Sanders) median 52, 80% 49-54, P(D control)
0.772; new independents win 0.11 seats on average. Sensitivity (`out/sens/`, 10k sims): no dossiers 231.2; quant
blend lambda 0.5 234.7; the rejected midterm calibration specs 234.3-234.9. Expected contest points under our q:
top-q picks 54.2 (46.5 under Kalshi's prices), max-EV picks 68.4 (54.4), chalk 50.4.

By core rating (House, expected D seats, ours vs Kalshi): Safe D 185.8 vs 182.5 (186 seats), Likely D 15.8 vs
15.2 (16), Lean D 18.4 vs 19.3 (22), Tilt D 2.8 vs 3.2 (4), Toss-up 4.7 vs 5.6 (9), Tilt R 1.6 vs 2.3 (4), Lean R
2.0 vs 5.1 (14), Likely R 1.0 vs 4.3 (15), Safe R 0.7 vs 8.5 (164). Most of the 13-seat gap is Kalshi's thin-book
longshot prices in Safe/Likely R seats.

Our top candidate differs from Kalshi's favourite in 6 races (70 flagged in `out/disagreements_2026.csv`):

| race | Kalshi favourite (p) | our q for them | our pick (q, Kalshi p) | reason |
|---|---|---|---|---|
| WI-01 | Berman D (0.50, a tie) | 0.17 | Steil R (0.83, 0.50) | R incumbent with an R money edge, fundamentals 47.1% D, polls 50.2% (2), Lean R by all three; model 0.11, dossier 0.47; Kalshi looks unpriced |
| MI-04 | McCann D (0.615) | 0.37 | Huizenga R (0.63, 0.385) | R incumbent, fundamentals 48.1%, polls 48.6% (2, incl. RedBox now D-tagged), Toss-up; model 0.31, dossier 0.57 |
| FL-22 | Dandiya D (0.715) | 0.41 | Askar R (0.59, 0.285) | new seat, lean R+4.4, open, R money edge, no usable polls, Toss-up; model 0.37, dossier 0.56 |
| MI-10 | Hines D (0.757) | 0.42 | Bouchard R (0.58, 0.243) | open, fundamentals 51.0% D but the one in-window poll (RedBox, now D-tagged) 47.2%; model 0.35, dossier 0.63 |
| PA-01 | Harvie D (0.535) | 0.45 | Fitzpatrick R (0.55, 0.465) | R incumbent with a big money edge, polls 50.1% (5), Toss-up; model 0.42, dossier 0.53 |
| VA-01 | Taylor D (0.629) | 0.45 | Wittman R (0.55, 0.371) | R incumbent, fundamentals 48.2%, polls 50.8% (4), Toss-up; model 0.40, dossier 0.59 |

CA-34, TX-35 and OH-07 (v1 picks against Kalshi) are now on Kalshi's side, mostly through the dossiers
(OH-07 0.50 -> 0.58, TX-35 0.48 -> 0.53, CA-34 Gonzales-Torres 0.43 -> 0.52).

## 7. What I borrowed from whom

- **kavi-b**: the Wayback URL of the pre-election `weball18.zip` (2018-11-07 capture) and the warning that
  fec_hist's cand_id can differ from the snapshot's (sibling committee ids), hence my surname fallback. I wrote my
  own fetcher and extended it to the 2022 `webl22.zip` capture of 2022-08-03 (kavi-b had looked only for weball22).
- **kavi-c**: the idea of conditioning on midterms (their env x midterm). I tested the analogue at my calibration
  stage (midterm intercept, midterm x z, midterm x r, and the investigation's midterm x out-party term); all lose
  on 2010-2016 held-out years in my model, so none is used. kavi-c's independent fade (past independents ran
  behind their polls) was checked against my own stage-2 residuals for independents (about 0 with polls), not
  adopted.
- **Cross-model investigation**: the RedBox sponsor fix, the quantitative outlets' ratings and the thin-race poll
  window (both tested, both rejected on 2020/2024 forward tests), the CA party-label check (already right).
- **Referee audit**: the FEC look-ahead (fixed with pre-election cuts) and the MS-SEN-S 2018 incumbency flag.
- Kept my own: least-squares share model + nested-LOYO calibration + three-level simulation. No sibling's model
  form, market offset or rating recipe was copied.

## 8. v1 -> v2 changes and their measured effect

| change | 2018 all / contested | 2022 all / contested | 2026 House D (mean) |
|---|---|---|---|
| v1 as submitted (new scorecard, winners_2018 fixed) | 0.0825 / 0.3065 | 0.0805 / 0.3662 | 231.6 |
| pre-election FEC in test years + Hyde-Smith incumbent | 0.0826 / 0.3071 | 0.0829 / 0.3624 | (training unchanged) |
| CAL_WEIGHT 0.5 -> 0.75 | 0.0823 / 0.3069 | 0.0821 / 0.3576 | |
| RedBox tagged D (2026) | n/a | n/a | 231.2 (with the weight change) |
| dossier late-information step, weight 0.25 (2026) | n/a | n/a | 232.8 |
| tested and rejected: midterm/out-party calibration, thin/wide poll window, quant blend | see section 4 | | 234-235 if adopted |

## 9. Known weaknesses

- Still more Republican than Kalshi (246) and than the investigation's fair value (~238) in toss-ups and Lean
  seats: Toss-up seats 4.7 of 9 D vs the out-party's ~2/3 historical rate. My held-out tests do not support a
  midterm term, but 2026 could be a 2018-type year, the one year where my model under-called the out-party by
  ~12 seats in LOYO.
- The 2022 backtest money is a June cut (no later pre-election snapshot exists); 2026 money is Oct 14; training
  is full-cycle. Pure fundamentals in 2022 lost 0.006 from this.
- National error estimated from 4-8 elections with a floor of 0.015 share; a 2020-sized polling miss would hurt.
- Same-party CA, LA jungle and AK RCV are heuristics; the dossier weight is a judgement with no backtest.
- No district demographics, outside spending, early vote or primary results (except CA); prev-result feature
  missing in all 9 redrawn states.
- Design choices partly saw the test years (section 4).

## 10. How to reproduce (from the worktree root; Python = python)

```
cd src/models/kavi-a
$PY features.py                          # out/hist_features.csv (all poll modes, quant ratings)
$PY fetch_ca_primaries.py --offline      # data/kavi-a/ca_primaries_2026.csv from the saved HTML
$PY fetch_fec_preelection.py --offline   # data/kavi-a/fec_preelection.csv from the saved Wayback zips
$PY select_v2.py                         # out/select_v2.json (design selection tables, ~2.5 minutes)
$PY backtest.py                          # out/backtest_{2018,2022}.csv (+ backtest_races_*)
$PY backtest.py --variant F --suffix _F
$PY predict_2026.py --sims 20000         # out/pred_2026.csv, disagreements_2026.csv, races_2026.csv,
                                         # sims_summary.json, dossier_adjustments_2026.csv
$PY predict_2026.py --sims 10000 --dossier-weight 0 --suffix _nodossier   # sensitivity -> out/sens/
$PY dump_params.py                       # out/model_params.json
$PY -m pytest -q test_outputs.py         # 8 invariant tests
cd ../../..
$PY src/scorecard.py --pred 2018=out/backtest_2018.csv --pred 2022=out/backtest_2022.csv --name kavi-a --json out/scorecard.json
```
Needs statsmodels, scikit-learn and pytest in the shared venv. The dossier summary is read from
`data/dossiers/summary.csv`.
