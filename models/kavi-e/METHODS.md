# kavi-e METHODS (v3): national environment + uniform swing model, pooled with a winner classifier

Branch `kavi/kavi-e-1101`, worktree `kavi-models/kavi-e`. Code: `src/models/kavi-e/`. Outputs: `out/` (v2 copies
in `out/v2/`, v1 in `out/v1/`). Log: `LOG.md` (v1, v2, then the v3 section). This v3 section is complete on its
own for what changed and summarises what did not; the v2 text below (sections 1-13) remains the detailed
reference for the unchanged parts and its numbers are v2's.

## v3.1 What v3 is, in one paragraph
For each race, the v2 model gives P_margin: a margin regression (lean, national environment, incumbency, expert
rating), blended with a recency-weighted poll average, turned into a win probability by a calibrated Student-t(6)
error with a shared national term. v3 adds a second, structurally different model of the same pre-election inputs:
a D/R mirror-symmetric logistic classifier of the winner, P_clf. In the House the two are pooled in logit space,
`logit P = 0.6 logit P_margin + 0.4 logit P_clf`. The Senate keeps P_margin alone. The pooled probability is mapped
back to an implied margin `mu = sigma x Finv_t6(P)` so that the dossier adjustment and the correlated simulation work
on it unchanged. Everything else is v2.

## v3.2 Data (unchanged from v2; full table in section 1)
- Training: MIT MEDSL House 2010-2024 (3,058 contested D-v-R rows) and Senate (261 rows); presidential lean on the
  lines in force (lean_hist / lean_2026); core-3 mean ratings (Cook, Sabato, Inside; -3 Safe D .. +3 Safe R);
  538 poll archives (one row per poll_id, RCV final round first; 120-day window from 2018, 60 days before);
  actual national House margin as training N.
- 2026: Kalshi candidate list (07:48Z Nov 1 snapshot, never an input), candidates_2026, ratings_2026, polls_2026
  (Wikipedia + VoteHub, sponsor-tagged), national_2026 (GB, approval), FEC 2026 to Oct 14 (same-party split only),
  June top-two primaries (kavi-b's file), the 85 Nov 1 dossiers.
- Leakage rules kept: no FEC in the race models (full-cycle totals run past election day); for a backtest year
  only earlier years train anything; no current-Wikipedia features for history (they would leak post-election information).

## v3.3 Features
- Margin model (stage A, OLS): `margin = b0 + b_L L + b_Lp L x pres_year + b_N N + b_inc inc + b_r rating`.
- Poll layer: `mu = muA + w (poll - muA)`, `w = n_eff / (n_eff + k)`, k by LOYO least squares (House 2, Senate 1).
- **Classifier (v3, `classifier.py`)**: x = (rating, L/10, L x pres_year/10, N/10, inc, w x poll/10), with N the
  actual national margin in training and N_hat when predicting (as in the margin model), the poll margin capped at
  +-30 and 0 for unpolled races, and w the margin model's poll weight. Target: D side won.

## v3.4 Model and calibration
1. National environment N_hat (unchanged, `national.py`): GB, special elections, VA/NJ governors and an approval
   fundamental, each de-biased with earlier years' errors and combined by inverse variance; sd floor 2.5.
   2018 +8.32, 2022 -3.21, **2026 D+8.08 (sd 2.5)**.
2. Margin model and error (unchanged): `P_margin = F_t6(mu / sigma)`, `sigma^2 = ((b_N(1-w) + w) sigma_N)^2 +
   tau^2 (s_e^2 (1-w)^2 + s_p^2 w^2)`; tau = geometric mean of the likelihood-optimal and the margin-calibrated tau on
   LOYO stage-A predictions of the training years. 2026 House: b 1.30 / L 0.739 / L_pres 0.010 / N 0.682 / inc 2.13 /
   rating -3.31, s_e 7.73, s_p 7.67, k 2, tau 0.87. Senate: -0.15 / 0.455 / -0.001 / 0.388 / 2.30 / -5.35, s_e 8.76,
   s_p 6.13, k 1, tau 0.64.
3. **Classifier (v3)**: sklearn LogisticRegression, L2, C = 1, no intercept; every training row enters twice, as is
   and mirrored (x -> -x, y -> 1-y), so the model treats D and R symmetrically. Fitted on the same training years
   as the margin model (2010-16 for 2018, 2010-20 for 2022, 2010-24 for 2026). Coefficients (rating, L/10,
   L_pres/10, N/10, inc, w*poll/10): for 2018 -1.71, 0.67, -0.01, 1.64, 0.43, 1.64; for 2022 -1.57, 0.73, 0.37,
   1.10, 0.07, 1.55; for 2026 -1.58, 0.72, 0.43, 1.12, 0.30, 1.52.
4. **Pool (v3)**: `logit P = (1-a) logit P_margin + a logit P_clf` (each clipped to [1e-4, 1-1e-4]); a = 0.4 House,
   0 Senate; independents (rows with an inflated sigma) are not pooled because the classifier never saw such races.
   Then `mu := sigma x Finv_t6(P)`, keeping sigma.
5. Calibration: no separate recalibration layer. The pooled model's calibration is checked by the non-test LOYO
   scores (v3.7); the t df and tau mode were re-validated there.

## v3.5 Special races (unchanged from v2; sections 4.3-4.6)
- Independents on the D side (Osborn NE, Achilles, Bengs, Bodnar MT, AK-AL Hill): polls shifted by the shrunk
  independent-specific miss of earlier cases (-2.9 for 2026), idiosyncratic sd x1.25; withdrawn Democrats get the
  minimum q. Incumbent independents (King, Sanders) are not shifted.
- Same-party CA/WA top-two races: fitted within-party logit on the June primary lead, incumbency and (2018 / 2026
  only, where a pre-election FEC cut exists) ln receipts ratio; 2026 adds 0.15 logit per point of late poll lead.
- Louisiana: party-level model, candidates on a side split by softmax(2.5 x incumbent). One-sided races 0.998.
- 2026 dossiers: margin-space pool `mu' = (1-lam) mu + lam sigma Finv(q_dossier)`, lam 0.15 / 0.25 / 0.35 by
  dossier confidence; logit pool on the pair for the 5 same-party races. In v3 this applies to the pooled mu.

## v3.6 Simulation (unchanged, `simulate.py`)
20,000 draws, seed 20261103: shared national error (sd 2.5, loading b_N(1-w)+w), a state error shared by a state's
House and Senate races (ICC 0.30 from same-state House residuals), t6 race errors; same-party races drawn from q.
Uses the (pooled, dossier-adjusted) mu and sigma per race.

## v3.7 Validation
- **Selection without the test years.** Every v3 option was scored by leave-one-year-out on non-test years only
  (`eval_v3.py` -> `out/work/selection_v3.txt`): `pre2018` (2010/12/14/16) and `six` (2010-16, 2020, 2024).
  Rule: minimise the sum of all + close (|rating| < 2.5) log loss over both criteria without worsening close
  midterm races by more than ~0.003.
- House contested log loss, base -> pool a 0.4: pre2018 0.0710 -> 0.0655 (close 0.3172 -> 0.3119, midterms 0.0810
  -> 0.0743); six 0.0816 -> 0.0761 (close 0.3792 -> 0.3695). Senate a 0.2 would be 0.1212 -> 0.1214 and 0.1385 ->
  0.1350 but midterms 0.0949 -> 0.1026 and 0.0983 -> 0.1040, so the Senate is not pooled.
- Rejected on the same criteria: group-specific residual sd, WLS down-weighting Safe seats, a piecewise rating
  slope, a Safe-seat lean slope (LOG.md v3 table).
- **Backtests (out of sample; 2018 trained on 2010-16, 2022 on 2010-20), `src/scorecard.py`:**
```
kavi-e 2018: n=470 logloss 0.0694 (538 0.1002) brier 0.0367 (538 0.0578) acc 0.977 score top 48.18 ev 53.43 (chalk 42.13) against-538 10; dropped 0
       2018 contested (538 fav<0.95): n=120 logloss 0.2582 (538 0.3778) brier 0.1407 (538 0.2257) acc 0.908 (538 0.875)
kavi-e 2022: n=470 logloss 0.0812 (538 0.0952) brier 0.0509 (538 0.0576) acc 0.960 score top 31.01 ev 43.14 (chalk 30.43) against-538 12; dropped 0
       2022 contested (538 fav<0.95): n=90 logloss 0.3768 (538 0.4375) brier 0.2450 (538 0.2787) acc 0.800 (538 0.822)
```
  | | 2018 all | 2018 contested | 2022 all | 2022 contested | 2018 House / Senate | 2022 House / Senate |
  |---|---|---|---|---|---|---|
  | v2 | 0.0711 | 0.2632 | 0.0820 | 0.3813 | 0.0678 / 0.1126 | 0.0807 / 0.0974 |
  | v3 | 0.0694 | 0.2582 | 0.0812 | 0.3768 | 0.0659 / 0.1126 | 0.0799 / 0.0974 |
- Paired state-cluster bootstrap of v3 - v2 (`paired_bootstrap.py`, 90%): 2018 all -0.0017 (-0.0048..+0.0012),
  contested -0.0050 (-0.0162..+0.0072); 2022 all -0.0008 (-0.0055..+0.0045), contested -0.0045 (-0.0294..+0.0232).
  Four of four point estimates improve; none is individually distinguishable from noise.
- Senate pool with a classifier trained on House + Senate rows (a 0.2) passed the non-test rule (Senate pre2018
  0.1212 -> 0.1206, six 0.1385 -> 0.1344) but worsened both Senate backtests (0.1126 -> 0.1155, 0.0974 -> 0.0990);
  reverted to a = 0.
- **Choices made by looking at 2018/2022:** the Senate revert above; classifier C (non-test LOYO preferred C = 0.1, the backtests were worse
  in total with it, reverted to C = 1 under the round's revert rule); the v1 choices (drop prev features, t6, no poll
  shift) remain, though t6 / blend tau are now re-confirmed on non-test years.

## v3.8 2026 forecast
- E[D House] from q 235.9 (v2 236.7); sims mean 235.8, median 236, 80% 226-246, 90% 223-249, P(D majority) 0.992.
  Senate (unchanged): D caucus median 52, 80% 49-55, P(D >= 51) 0.770.
- Against the Kalshi favourite (7): IA-SEN (Turek 0.50), CA-34 (Gomez 0.60), CA-40 (Calvert 0.67), CO-05 (Killin
  0.51, new), MI-04 (Huizenga 0.50), MI-10 (Bouchard 0.52), WI-01 (Steil 0.63). ME-02 (LePage) is now with Kalshi.
- The pool moves Lean-R seats further R (MO-02 0.36 -> 0.24, ME-02 0.57 -> 0.46, WI-01 0.46 -> 0.38) and Lean-D
  seats further D (TX-15 0.57 -> 0.62, WI-03 0.69 -> 0.75, PA-08 0.74 -> 0.79).

## v3.9 Known weaknesses (v3-specific first; the v2 list in section 10 still applies)
- The gain rests on two backtest years whose bootstrap intervals cover zero; the non-test LOYO evidence is
  stronger but includes years that also trained the 2026 model.
- The classifier leans harder on ratings than the margin model; if the core-3 ratings lag the late-October
  environment, v3 is more exposed than v2 (it is ~0.8 seats more R than v2).
- Backtest seat totals were too R in both years (2018 232.9 vs 235, 2022 205.9 vs 213); LOYO shows this bias is
  year-specific, so it is not corrected.
- Senate unpooled: Senate remains the weaker chamber (2018 0.1126).
- The pool weight (0.4) and C were chosen on 6 historical years; C was finally set with the test years in view.

## v3.10 Borrowed in v3
| from | what | how used |
|---|---|---|
| kavi-d (disclosed v2) | mirror-symmetric L2 logistic classifier on ratings, lean, incumbency, polls | my own features and fit on my frames; used as a pool component, not alone |
| REPORT 1.3 pools (kavi audit ensembles) | two different models pooled beat either alone | logit pool inside one model, weight chosen on non-test LOYO |
| pau-a / pau-b (disclosed v3) | group-specific error scales, WLS, piecewise rating, Safe-seat PVI slope | tested, rejected |
| REPORT 8.4 item 17 | missing uncertainty intervals | paired state-cluster bootstrap |

## v3.11 Reproduce
`bash src/models/kavi-e/run_all.sh` from the worktree root (~60 s). It now also writes `out/work/selection_v3.txt`
(the v2 harness `eval_loyo.py` runs with `KAVI_E_V3=""`, i.e. the v2 model). `KAVI_E_V3=""` on any script switches
the pool off (reproduces v2 exactly); `KAVI_E_V3=pool,hetero` etc. for the rejected options. Bootstrap:
`.venv/bin/python src/models/kavi-e/paired_bootstrap.py out/v2 out`.

---------------------------------------------------------------------------------------------------------------

# kavi-e METHODS (v2): national environment + uniform swing model

Branch `kavi/kavi-e-1101`, worktree `kavi-models/kavi-e`. Code: `src/models/kavi-e/`. Outputs: `out/` (v1 copies
in `out/v1/`). Running log: `LOG.md` (v1 sections, then the v2 section).

Angle: estimate the 2026 national House environment as precisely as possible from several independent signals,
then apply a uniform swing from each race's presidential baseline (on the lines in force), with incumbency,
expert-rating and poll adjustments and a calibrated, correlated error model. The same recipe is run out of sample
for 2018 and 2022. v2 fixes every bug the referee audit found, fits the same-party (CA top-two) split on history
instead of judgment, estimates the independent-candidate poll shift instead of hand-setting it, widens the poll
window where the archive allows, and adds a small, documented late-information adjustment from the Nov 1 dossiers.
The midterm / out-party calibration the cross-model investigation recommended was tested in five forms and NOT
adopted: in this model the wave is carried by the national-environment term, and none of the forms validated.

## 1. Data used

| file | source | years | n used | cleaning / notes |
|---|---|---|---|---|
| data/clean/results_house.csv | MIT MEDSL House 1976-2024 (v15) | 2008-2024 | 3,915 regular races; 3,058 contested D-v-R training rows 2010-2024 | target = 2-party D-R margin; uncontested and same-party top-two rows excluded from the swing model; specials dropped; same-party top-two CA/WA generals 2012-2024 (50 with both finalists in the primary file) train the within-party model |
| data/clean/results_senate.csv | MIT MEDSL Senate | 2010-2024 | 261 contested D-v-R training rows | rows where a non-D/R got >15% or won are excluded from training; runoffs use the runoff result |
| data/clean/lean_hist.csv | Downballot / DKE presidential results by CD; MIT statewide | 2010-2024 | 3,751 | L = 200 x (district D 2-party pres share - national D 2-party share), latest presidential election BEFORE the year, on that year's lines |
| data/clean/lean_2026.csv | Downballot pres 2024 on 2026 lines | 2026 | 470 | L = 200 x (pres24_dem_2p - 0.49250); handles the 9 redrawn states and the new seats |
| data/clean/ratings_hist.csv, ratings_2026.csv | Wikipedia ratings pages (Cook, Sabato, Inside) | 2010-2026 | all races | `mean_rating` (-3 Safe D .. +3 Safe R; imputed Safe for unlisted seats). The `other_*` outlets were tested as a feature (section 9) and not used |
| data/clean/polls_hist.csv | 538 archives | 2010-2024 | general-stage, non-hypothetical; final 60 days for 2010-2016 (archive limit), final 120 days for 2018-2024 | **v2: one row per poll_id** (RCV-reallocated final-round question first, then population lv > v > rv > a, then the mean of what is left). Major-independent races use the independent's share (other_pct) |
| data/clean/polls_2026.csv | Wikipedia + VoteHub | 2026 | final 120 days | **v2:** in RCV races each poll's final-round (2-candidate) question is used when it has one (AK-AL); missing sponsor tags filled from the same pollster / the sponsor note (RedBox / A Fight Worth Having -> D in MI-04 and MI-10; "progressive"/"Democratic-aligned" -> D; "conservative"/"Republican-aligned" -> R); independents read from top_other_*; same-party CA races from cand_detail |
| data/clean/national_hist.csv | 538 generic ballot / approval averages, MIT House vote | 2010-2024 | 8 rows | actual national House margin = training N |
| data/clean/national_2026.csv | aggregator rows of 2026-10-30 | 2026 | 5 GB, 9 approval aggregators | GB D+8.56 (mean of 5), approval 36.8 / 60.5 (net -23.7) |
| data/clean/candidates_2026.csv | Wikipedia | 2026 | 1,282 | incumbency, party of UNK Kalshi slots (all 951 output rows carry a real party; the 8 CA D-v-D races are DEM, CA-40 REP) |
| data/clean/fec_2026.csv | FEC weball26 (to 2026-10-14) | 2026 | 951 | used ONLY in the within-party split of same-party races |
| data/clean/fec_hist.csv | FEC weball10-24 | 2010-2024 | | incumbent flag (FEC `ici`); receipts only as TRAINING data of the within-party money term (they run to Dec 31) |
| **data/kavi-e/primaries_top_two.csv (v2, borrowed)** | June CA/WA top-two primaries, Wikipedia, parsed by kavi-b (`kavi-b/src/models/kavi-b/fetch_primaries.py`) | 2012-2026 | 2,418 rows | copied unchanged; pre-general information in every year |
| **data/kavi-e/fec_2018_preelection.csv (v2, borrowed)** | Wayback capture (2018-11-07 08:07 UTC) of FEC weball18.zip, found by kavi-b | 2018 | 3,656 | the only pre-election FEC cut for a backtest year; used for the 2018 within-party money term |
| repo/data/dossiers/summary.csv (v2) | Nov 1 research dossiers, each reviewed by a skeptic | 2026 | 85 races | late-information adjustment only (section 4.6); no historical analogue |
| data/specials/specials_clean.csv | The Downballot D-v-R special-election sheets, 2017-18 .. 2025-26 | 2016-12 .. 2026-08 | 525 included specials | collected for this model (`fetch_specials.py` / `specials.py`) |
| data/specials/offyear_governors.csv | VA + NJ governor results 2009-2025, hand-entered | 10 rows | | |
| data/benchmark/fte_final_{2018,2022}.csv | 538 final | | 470 races per year | candidate LISTS of the backtest and the comparison, never an input |
| data/kalshi/*_074812Z.csv | Kalshi snapshot 2026-11-01 07:48 UTC | | 470 races, 951 slots | candidate list and p_kalshi (mid normalised within each race's primary event); never a model input |

Data traps handled: redrawn 2026 maps (lean from lean_2026; no 2024 House result joined by district number);
Louisiana jungle races (party-level model); withdrawn Democrats in ID/NE/SD-SEN and MT-SEN (independent is the D
side, the withdrawn Democrat gets the minimum q); AK-AL Hill as D side; CA-06 Pan (D) vs Kiley (I, incumbent) with
Kiley on the R side; unopposed seats; the 9 CA same-party races; the two AK-SEN Dan Sullivans; Kalshi ticker !=
state (joins on race_id only). **v2 incumbency overrides** (`common.INCUMBENT_OVERRIDES`, ballot facts known
before each election): Hyde-Smith MS-SEN-S 2018 and McSally AZ-SEN 2020 (appointed senators FEC files as open-seat
candidates), Letlow LA-05, Carter LA-02 and Stansbury NM-01 2022 (House special-election winners; results_house
holds no specials and FEC codes Letlow 'C').

## 2. Features and why

- **L (lean)**: the presidential baseline the swing is applied to. `L_pres` = L x pres_year lets the slope differ
  in presidential years (2026 is a midterm, L_pres = 0).
- **N (national environment)**: actual national House margin in training years; pre-election N_hat in the
  forecast year (section 3). This term is why the midterm wave does not need a separate calibration here: given N,
  ratings and lean, the wave midterms resolve as predicted (section 5).
- **inc**: +1 D incumbent running, -1 R incumbent running, 0 open.
- **rating**: the experts' mean rating; carries candidate quality, scandal and district information a lean
  cannot. The strongest single feature.
- **polls**: recency-weighted two-party D-R margin (half-life 14 days, sqrt(n) capped at 1,500, a pollster's
  repeated polls down-weighted by 1/sqrt(count), partisan polls shifted 2 points against the sponsor (1 point in
  2026, where Wikipedia tags about half of all polls)); n_eff = Kish effective size. v2 window: 120 days where the
  archive has it (2018+, and 2026), 60 days for 2010-2016.
- **Same-party split (v2)**: June top-two primary lead, incumbency and (2018 / 2026 only) the receipts ratio, in a
  logit fitted on past same-party generals (section 4.4).
- Tried and NOT used: previous House over-performance, proportional swing, FEC receipts in the race model,
  historical poll-bias shift (v1); midterm x out-party terms, other raters' gap, midterm-only tau (v2, section 9).

## 3. National environment model (national.py, unchanged in v2)

Each signal gives an implied national House margin (D-R points):
- **gb**: final generic-ballot average (2026: mean of DDHQ, FiftyPlusOne, NBC, RCP, Silver Bulletin, 2026-10-30).
- **sp**: special elections from Jan 1 of the year before to election day: national raw presidential margin of
  the baseline election + 10%-trimmed mean overperformance (2018 window 118 specials; 2022 76; 2026 110).
- **gov**: VA + NJ governor races of the previous year: national pres margin + mean in-state overperformance.
- **fund**: midterm penalty: mean president's-party margin of earlier midterms + 0.15 x relative net approval.

For each signal, errors e = implied - actual of EARLIER years only (midterm targets: earlier midterms for sp, gov,
fund; all years for gb). bias = mean(e) n/(n+2); sd^2 = (2 prior_sd^2 + sum (e - bias)^2)/(2 + n), prior sds 3 / 4
/ 6 / 4.5. Inverse-variance combination; combined sd floored at 2.5.

| target | N_hat | sd | actual | gb (raw -> adj, w) | sp | gov | fund |
|---|---|---|---|---|---|---|---|
| 2018 | +8.32 | 2.5 | +8.57 | +8.6 -> +7.1, 0.54 | +12.9 -> +12.9, 0.17 | +3.9 -> +8.4, 0.09 | +6.9 -> +7.5, 0.20 |
| 2022 | -3.21 | 2.5 | -2.83 | -1.2 -> -2.9, 0.49 | +1.6 -> +0.1, 0.18 | -7.9 -> -3.3, 0.10 | -7.3 -> -6.4, 0.23 |
| 2026 | **+8.08** | 2.5 | ? | +8.6 -> +6.8, 0.50 | +10.1 -> +7.9, 0.19 | +6.7 -> +11.3, 0.10 | +8.3 -> +9.9, 0.20 |

v2 considered an out-party-signed generic-ballot bias (GB understates the out-party in midterms). The four
midterms in the data fit it, but it is identified only by 2018 (the only D-out midterm) and 2006 (GB D+11.5 vs
actual D+7.9, out-party overstated) contradicts it. Not adopted. The audit showed N_hat barely moves the backtest
scores (a 1-sd miss costs at most 0.005); for 2026, using the raw GB D+8.56 instead of N_hat adds ~0.7 House seats.

## 4. Race model and parameters

### 4.1 Stage A (OLS on contested D-v-R races of the training years, N = actual national margin)
`margin = b0 + b_L L + b_Lp L_pres + b_N N + b_inc inc + b_r rating + e`

| fit | b0 | L | L_pres | N | inc | rating | s_e | k | tau |
|---|---|---|---|---|---|---|---|---|---|
| House 2010-16 (for 2018) | 3.23 | 0.685 | 0.045 | 1.019 | 3.24 | -4.30 | | 2 | 0.79 |
| House 2010-20 (for 2022) | 1.34 | 0.702 | 0.026 | 0.665 | 2.00 | -3.91 | | 2 | 0.86 |
| House 2010-24 (for 2026, n 3,058) | 1.30 | 0.739 | 0.010 | 0.682 | 2.13 | -3.31 | 7.73 | 2 | 0.87 |
| Senate 2010-16 (for 2018) | 1.12 | 0.404 | 0.019 | 0.537 | 2.24 | -6.69 | | | 0.52 |
| Senate 2010-20 (for 2022) | -0.19 | 0.399 | 0.032 | 0.308 | 1.59 | -6.21 | | | 0.71 |
| Senate 2010-24 (for 2026, n 261) | -0.15 | 0.455 | -0.001 | 0.388 | 2.30 | -5.35 | 8.76 | 1 | 0.64 |

### 4.2 Stage B (polls) and error model
`mu = muA + w (poll - muA)`, `w = n_eff / (n_eff + k)`; muA uses N_hat; k chosen on a grid by least squares on
leave-one-year-out stage-A predictions of the training years. Error: `y - mu = (b_N (1 - w) + w) eps_N + u`,
eps_N ~ N(0, sigma_N = 2.5), sd(u) = tau sqrt(s_e^2 (1-w)^2 + s_p^2 w^2), s_p = sd of poll misses after removing
each year's mean (2026 fit: House 7.67, Senate 6.13). P(D side wins) = F_t6(mu / sigma), Student-t (6 df) scaled
to sd sigma. tau = geometric mean of the likelihood-optimal tau and the tau that makes close races' LOYO margin
z-scores have sd 1 (2026: House 0.85 / 0.89 -> 0.87; Senate 0.52 / 0.78 -> 0.64).

### 4.3 Independent D-side candidates (v2: estimated; `independents.py`)
Challenger independents ran behind their polls. The independent-specific part = miss minus that year's mean
Senate poll miss: Orman KS 2014 -8.9, McMullin UT 2022 -4.4, Osborn NE 2024 -1.3. For a target year only earlier
cases are used, shrunk n/(n+2): shift -3.0 (2018, 2022 backtests), **-2.9 for 2026** (v1 hand-set -5). Their
idiosyncratic sd is x1.25 (v1 x1.5: the net misses have sd ~4, close to an ordinary poll miss; the extra 25%
covers the uncertainty of the shift itself). Incumbent independents (King, Sanders) are not shifted. In the
backtests the independents' own polls are now used (v1 dropped them; McMullin got 0.0009 from fundamentals).

### 4.4 Same-party top-two split (v2: fitted; `within_party.py`)
`logit P(A beats B) = b_pct (primary% A - primary% B)/10 + b_inc (inc_A - inc_B) [+ b_lr ln(receipts ratio)]`,
no intercept, both orderings of each pair, ridge 0.1, years < target only. The June primary leader won 86% of 50
same-party generals 2012-2024.

| fit | n | b_pct | b_inc | money variant: b_pct / b_inc / b_lr |
|---|---|---|---|---|
| for 2018 | 25 | 1.03 | -0.61 | 0.73 / -0.96 / 0.79 |
| for 2022 | 38 | 1.24 | -0.61 | 1.19 / -0.73 / 0.21 |
| for 2026 | 50 | 1.14 | +0.10 | 1.11 / -0.14 / 0.32 |

The money variant is applied only where a pre-election FEC cut of the target year exists: 2018 (Wayback snapshot)
and 2026 (fec_2026 to Oct 14). 2022 uses the no-money model. Training receipts are full-cycle (a small coefficient
leak; kavi-b measured a 0.07 mean log-ratio gap in 2018). For 2026 the late same-party poll lead adds 0.15 logit per
point (judgment, no same-party poll history). Races without primary shares (LA jungle) keep v1's rule.

### 4.5 Allocation to candidates (allocate.py)
D side (Democrat or designated independent) gets P(D), R side 1 - P(D); several candidates on a side share it by
the within-party model (same-party CA/WA) or softmax(2.5 x incumbent) (LA); minor candidates 0.001; one-sided
races 0.998; renormalised so every race sums to 1.

### 4.6 Late-information adjustment from the dossiers (v2, 2026 only; `dossier_adjust.py`)
- What the dossiers are: researched P(underdog) after a skeptic review (q_dog_mean) for 85 contested races. They
  include polls, news and money up to Nov 1 the model's inputs miss. They have no track record and are anchored on
  the Kalshi price (60 of 85 say "market about right").
- How they are used: in the model's own margin space,
  `mu' = (1 - lam) mu + lam sigma Finv(q_D_dossier)`, so the correlated simulation stays coherent. For the 5
  same-party races the pool is in logit space on the pair.
- How much: lam = 0.15 / 0.25 / 0.35 for low / medium / high dossier confidence (36 / 46 / 3 races). This is a
  judgment: 0.25 is about the weight of one more rater next to the core three.
- Effect: +0.66 expected D House seats, -0.09 Senate. Over the 80 two-sided races the sum of P(D) moved 27.35 ->
  27.92, while the dossiers' own sum is 33.34. Largest moves: NC-03 0.05 -> 0.08, OH-07 0.52 -> 0.57, MI-10 0.44 ->
  0.47, NE-SEN 0.12 -> 0.15, AK-AL 0.10 -> 0.13; CA-34 Gomez 0.72 -> 0.60.

## 5. Training and validation

- Backtest 2018: stage A, k, s_p and tau fitted on 2010-2016 only; N_hat from pre-election signals with bias/sd
  from years before 2018; ratings and polls dated before election day; ballot facts (nominees, incumbency, June
  primary shares); pre-election FEC (Wayback) only in the same-party split. Backtest 2022: same with 2010-2020, no
  FEC at all. 538's forecast is never an input (its candidate list only).
- **Selection discipline (v2).** Design options were chosen by leave-one-year-out over the pre-2018 years
  (`eval_loyo.py`, output `out/work/selection_loyo_pre2018.txt`). Where pre-2018 data could not test an option
  (polls older than 60 days do not exist before 2018), the six non-test years 2010-16, 2020 and 2024 were used. The
  2018/2022 backtests were then run and recorded for every option, including those already decided:
  - midterm / out-party terms: rejected on pre-2018 LOYO (all 5 forms worse in the House);
  - other raters' gap (rgap): rejected on pre-2018 LOYO (0.0710 -> 0.0723); the 2018/2022 backtests agree;
  - 120-day window: adopted on the 2020/2024 evidence (poll-miss sd flat with age; LOYO House 0.0835 -> 0.0821);
    the backtests then showed a small cost (+0.0013 / +0.0008), disclosed and kept;
  - within-party model: adopted on structure (a fitted model replaces judgment weights; years < target only);
  - money in the within-party model where a pre-election cut exists: pre-registered from the audit; costs 0.0004
    in 2018.
  - v1 choices made on the backtests (drop prev features, t6 errors, no poll shift) are unchanged; the audit put
    their selection effect at ~0.004 (2018) and ~0.0015 (2022) log loss.
- **Wave-midterm calibration of this model** (House, toward the out-party, 2010/2014 pre-2018 LOYO + 2018
  backtest): toss-ups predicted 0.69 / won 0.73 (n 84); out-party Lean 0.955 / 1.00 (47); out-party Likely 0.987 /
  1.00 (39); in-party Lean 0.218 / 0.167 (60); in-party Likely 0.087 / 0.102 (59). There is no one-directional
  out-party miss in competitive seats, which is why the investigation's recommended interaction does not transfer.

## 6. Scorecard numbers (src/scorecard.py = repo version with the contested block; out/scorecard.json)

```
kavi-e 2018: n=470 logloss 0.0711 (538 0.1002) brier 0.0378 (538 0.0578) acc 0.977 score top 49.03 ev 52.98 (chalk 42.13) against-538 10; dropped 0
       2018 contested (538 fav<0.95): n=120 logloss 0.2632 (538 0.3778) brier 0.1450 (538 0.2257) acc 0.908 (538 0.875)
kavi-e 2022: n=470 logloss 0.0820 (538 0.0952) brier 0.0513 (538 0.0576) acc 0.957 score top 30.36 ev 42.08 (chalk 30.43) against-538 11; dropped 0
       2022 contested (538 fav<0.95): n=90 logloss 0.3813 (538 0.4375) brier 0.2472 (538 0.2787) acc 0.789 (538 0.822)
```

| | 2018 House | 2018 Senate | 2022 House | 2022 Senate |
|---|---|---|---|---|
| kavi-e v2 log loss | 0.0678 | 0.1126 | 0.0807 | 0.0974 |
| kavi-e v1 log loss | 0.0660 | 0.1360 | 0.0843 | 0.0928 |
| 538 log loss | 0.0938 | 0.1799 | 0.0914 | 0.1414 |

Variants (same v2 settings): no_rating 0.0948 / 0.0850 (contested 0.3434 / 0.3930); with_prev 0.0744 / 0.0839.

## 7. 2026 forecast summary

- Correlated simulation (20,000 draws, seed 20261103; national + state (ICC 0.30 from same-state House residuals)
  + t6 race errors; same-party races drawn from q): House D seats mean 236.6, median 237, 80% interval 226-247,
  90% 223-250, P(D majority) 0.991. Senate D caucus (34 not up) median 52, 80% 49-55, P(D >= 51) 0.770, P(R >= 50
  incl. VP tie) 0.221; mean new-independent wins 0.18.
- E[D House seats] from q: 236.7 (v1 236.1); Kalshi 246.0. The cross-model investigation's fair value is ~238. I
  did not tune toward it. Remaining gap by seat type: R-incumbent toss-up / Lean D seats where incumbency and thin
  polls pull down (TX-15 0.57, OH-07 0.57, TX-35 0.55, FL-22 0.54), and the other-raters signal I rejected
  (it would add ~1.5 seats).

## 8. 2026 picks against the Kalshi favourite (out/disagreements_2026.csv, 7 of 470; 59 more with |q-p| > 0.10)

| race | Kalshi favourite (p) | our top (q) | reason |
|---|---|---|---|
| IA-SEN | Hinson R 0.56 | Turek D 0.50 | 23 polls average D+0.4 (n_eff 17, w 0.95); triple toss-up; the dossier (Turek 0.44) pulls a 0.52 model to 0.50. A coin flip. |
| CA-34 | Gonzales-Torres D 0.82 | Gomez D 0.60 | same-party: Gomez led the June primary 46.0-30.7 (fitted model: primary leaders win 86%), against one USC poll with the challenger +6 and the dossier (0.775 challenger, weight 0.25). Low confidence. |
| CA-40 | Kim R 0.52 | Calvert R 0.67 | same-party (two R incumbents): Calvert led the primary 34.9-20.6. The model cannot see D crossover voters, who may favour Kim; the dossier (Kim 0.525) is at low weight. |
| ME-02 | LePage R 0.56 | Dunlap D 0.57 | open seat; polls D+2.7 (w 0.65) outweigh the Lean R ratings. |
| MI-04 | McCann D 0.61 | Huizenga R 0.52 | R incumbent; polls R+3.8 (w 0.38) incl. the RedBox D-sponsored poll now tagged D; toss-up x3. |
| MI-10 | Hines D 0.76 | Bouchard R 0.53 | open R seat; polls R+6.5 (w 0.34; the latest RedBox poll had Bouchard +5); dossier (Hines 0.62) at weight 0.15. |
| WI-01 | Berman D 0.50 | Steil R 0.54 | Kalshi is at 0.50; Lean R x3 and R incumbency. |

All except CA-34 and CA-40 sit within 0.07 of 50/50. CO-05 and OH-07 (v1 against Kalshi) are now with Kalshi's
favourite. Largest same-pick gaps: Likely-R seats where the model is more confident in the R favourite than
Kalshi (NC-03 R 0.92 vs 0.56, NC-13 0.97 vs 0.60, TX-23 0.84 vs 0.54), and D favourites where it is less confident
(TX-15 0.57 vs 0.89, OH-07 0.57 vs 0.85, WI-03 0.69 vs 0.89).

## 9. Tried and failed

v1:
- Prior House over-performance features (prev_resid): worse in both backtests.
- Historical poll-bias shift: worse in both years (2022 Senate 0.088 -> 0.123).
- Proportional swing (N x |L|): worse in LOYO.
- Pure likelihood tau: replaced by the blend to avoid an overconfident Senate.
- FEC log ratio in the race model: leaky history (receipts run to Dec 31).

v2 (pre-2018 LOYO House / Senate log loss; base 0.0710 / 0.1212):
- out_mid (midterm x out-party direction) 0.0719 / 0.1229; x close rating 0.0719 / 0.1230; x (1 - |r|/3) 0.0714 /
  0.1227; out_mid + its close version 0.0733 / 0.1236; out_mid + rating x midterm 0.0733 / 0.1252; incumbency x
  out_mid 0.0729 / 0.1211.
- rgap (other raters minus core-3) 0.0723 / 0.1219; backtests 2018 0.0693 -> 0.0701, 2022 0.0834 -> 0.0879.
- Midterm-only tau: House 0.0711, Senate 0.1204 (midterms 0.0949 -> 0.0934); not adopted.
- Longer poll half-lives (30, 45 days, six-year LOYO): House 0.0829 / 0.0826, Senate 0.1439 / 0.1463, vs 0.0821 /
  0.1380 at 14 days.
- Out-party-signed generic-ballot bias: not identifiable before 2018, contradicted by 2006.

## 10. Known weaknesses

- The national-environment rule was designed knowing roughly how 2018 and 2022 ended; expect 2.5-3 points of
  error, which the model carries (the audit found the backtests barely depend on it).
- Ratings carry much of the signal and absorb the environment (b_N 0.68 House). If the late-October GB surge is not
  fully in the core-3 ratings, the model is too R; the faster quantitative outlets point that way (+1.5 seats), but
  they did not validate historically.
- Senate idiosyncratic error is small (tau 0.64): well-polled Senate races get sd ~4.4 points; t6 tails soften it.
- Same-party races: the fitted model knows primary shares, incumbency and money, not crossover voting by the party
  without a candidate (CA-40, CA-34, CA-04, where the dossiers and Kalshi disagree with it).
- Independents: the shift rests on 3 cases; the q for Osborn / Hill (0.15 / 0.13) is still below Kalshi (0.25 /
  0.21).
- The dossier weight is a judgment with no track record.
- The 120-day window helped in 2020/2024 but cost a little in 2018/2022.
- Louisiana's Nov 3 jungle and Georgia's possible runoff are modelled as the eventual party winner.
- Results-side quirks: MEDSL RCV rows are mixed rounds (AK, ME); AK-AL 2022's training margin is a first round.

## 11. What I borrowed from whom

| from | what | how used here |
|---|---|---|
| kavi-b | June top-two primary results 2012-2026 (`primaries_top_two.csv`, its `fetch_primaries.py`) and the idea of a fitted within-party model on primary lead, money and incumbency | data copied unchanged; my own fit (ridge logit, years < target, mirrored pairs), no-money model for 2022 |
| kavi-b | Wayback pre-election FEC 2018 snapshot (`fec_2018_preelection.csv`) | money term of the 2018 within-party split only |
| referee audit | the five bugs (missed incumbents, duplicate poll questions, RCV rounds, minor-candidate split, McMullin) | all fixed |
| cross-model investigation | the midterm / out-party hypothesis, other_* outlets, longer poll window, RedBox tag, UNK parties | tested; window + RedBox adopted, the rest tested and rejected for this model (sections 5, 9) |
| kavi-b (dossiers.py), as a reference | pooling the dossier as one extra opinion | I pool in margin space with confidence-dependent weights instead of a fixed logit weight on the pair |

Kept my own: the national-environment model with specials and governors, uniform swing on a margin regression,
the t6 margin error model with a shared national term, and the state-correlated simulation.

## 12. v1 -> v2 changes and their measured effect (out-of-sample log loss, all races / contested)

| step | 2018 all | 2018 contested | 2022 all | 2022 contested |
|---|---|---|---|---|
| v1 as submitted | 0.0713 | 0.2654 | 0.0849 | 0.3857 |
| + audit bug fixes (incumbency, poll dedupe, independents' polls) | 0.0693 | 0.2578 | 0.0834 | 0.3773 |
| + 120-day poll window (2018+) | 0.0706 | 0.2632 | 0.0842 | 0.3846 |
| + fitted within-party model (no money) | 0.0707 | 0.2632 | 0.0820 | 0.3813 |
| + money in the 2018 within-party split (pre-election snapshot) = **v2** | **0.0711** | **0.2632** | **0.0820** | **0.3813** |

Audit's single-fix numbers for reference: incumbency fix 0.0700 / 0.0829; dedupe 0.0694 / 0.0850; RCV final round
0.0713 / 0.0852. 2026 effects: within-party model (CA-07 Matsui 0.97 -> 0.54, CA-04 Thompson 0.86 -> 0.79),
independents (Osborn 0.09 -> 0.15, Hill 0.05 -> 0.13), dossiers (+0.66 House seats), total E[D House] 236.1 ->
236.7; Senate P(D >= 51) 0.806 -> 0.770 (older polls in the 120-day window, mainly AK/OH/MI-SEN).

## 13. How to reproduce

From the worktree root (`models/kavi-e`):
```
bash src/models/kavi-e/run_all.sh            # everything from the committed data, ~90 s
FETCH=1 bash src/models/kavi-e/run_all.sh    # also re-download The Downballot special-election sheets
```
Order (Python = python): specials.py, national.py, house_data.py,
senate_data.py, independents.py (-> out/work/independents.txt), within_party.py (-> out/work/within_party.txt),
eval_loyo.py (-> out/work/selection_loyo_pre2018.txt), backtest.py, src/scorecard.py (-> out/scorecard.json),
the no_rating / with_prev variants, predict_2026.py (-> pred_2026.csv, disagreements_2026.csv, fit_2026.json,
out/work/races_2026.csv with mu_model / mu / dossier columns), simulate.py (-> sims_summary.json).
Ablations of section 12: `KAVI_E_ABLATE=no_within` or `no_money` for backtest.py; `LONG_WINDOW` in polls.py.
