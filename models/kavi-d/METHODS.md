# kavi-d: METHODS (v3) — symmetric classifier with a generic-ballot x midterm term

Branch `kavi/kavi-d-1101`, code in `src/models/kavi-d/`, outputs in `out/` (v1 outputs in `out/v1/`, v2 outputs in
`out/v2/`, design variants in `out/variants/`). Everything is reproduced by `bash src/models/kavi-d/run_all.sh`
(about 6 minutes). The v2 METHODS follows below this section unchanged; v3 changes one feature of the party model
and keeps every other component, which is restated here so this section can be read on its own.

One-line summary (v3): a D<->R **mirror-symmetric L2 logistic regression** on expert ratings, partisan lean +
generic ballot, incumbency, previous margin, race polls and a **generic ballot x midterm** term (replacing v2's
White-House out-party sign), trained on 3,450 contested House/Senate races 2010-2024, slope-only Platt calibration
on rolling out-of-sample predictions, a 1% floor for races not rated Safe by all three raters; the v2 same-party
model (fitted primary, incumbency and money weights); for 2026 only the v2 late-information pool with the 85
research dossiers; a 20,000-run Gaussian-copula simulation.

## v3.1 What changed and why

- **Removed** `op` (+1 midterm under an R president, -1 under a D president). It says "the out-party beats its
  ratings by a fixed amount in every midterm". Fitted on 2010/2014/2018 it pushed every 2022 toss-up toward the
  Republicans (model 0.69 out-party, actual 7 of 27) and cost 0.019 log loss in 2022. REPORT.md section 1.4 found
  the same pattern in all four Kavi models with an out-party term, while the models that tie the midterm
  environment to the generic ballot (pau-a, pau-b, kavi-e, kavi-a) were stable in both years.
- **Added** `gb_mid` = final generic-ballot D-R margin x [midterm] / 10. It is D/R symmetric (its mirror image is
  its negative), known before election day, and its size and sign follow the measured environment: 2010 -0.67,
  2014 -0.19, 2018 +0.86, 2022 -0.12, 2026 +0.86, 0 in presidential years. The lean+generic feature `base_m`
  already carries the generic ballot in all years; gb_mid lets midterms move further than the ratings and lean
  imply (the experts' midterm lag the cross-model investigation documented), in proportion to the generic ballot
  instead of by a fixed out-party amount. Form borrowed from pau-a ("GB x midterm in non-safe races") and pau-b
  ("year effect = a + b x GB x midterm"); here it is one term of the classifier.
- Tested and **not adopted** (section v3.6): kavi-e's true-national-margin training with an integrated N_hat
  (reverted: worse 2022 scorecard), a bias-corrected generic in base_m, N x midterm, a Senate-specific base slope,
  kavi-c's competitive-seat concentration of the wave, a poll-count interaction, the GBM+logit blend, other C.

## v3.2 Data

Unchanged from v2 (table in section 1 below): MIT House/Senate results 2010-2024 (training targets, previous
margins), Wikipedia Cook/Sabato/Inside ratings (2010-2026), Downballot/DKE presidential lean, 538 generic-ballot
finals 2010-2024 and the mean of five 2026 aggregators (D+8.56, 2026-10-30), 538 / Wikipedia / VoteHub polls (final
60 days), FEC (2026 receipts to 2026-10-14 for same-party splits only; I/C/O flag for incumbency in training; 2018
backtest uses the pre-election Wayback snapshot borrowed from kavi-b; 2022 backtest uses no money), kavi-b's top-two
primary history and CA 2026 polls, my CA 2026 primary file, the 85 dossiers (2026 only), Kalshi's Nov 1 07:48Z
snapshot (slot list and p_kalshi column only, never an input), 538's 2018/2022 files (ballot list only).
New in v3: `data/extra/kavi-e_national_env.json` (+ `.md`), kavi-e's N_hat table, used only by the reverted env
variants. `national_hist.csv` `cycle_type` defines the midterm flag (2026 = midterm). No current Wikipedia page is
used for any historical feature (board P016).

## v3.3 Features (all signed + = anti-R side; mirror = negate signed features, flip y)

| feature | definition | scale in the logit |
|---|---|---|
| r_mean | -1 x mean of Cook/Sabato/Inside numeric ratings (-3..3) | /1.5 |
| base_m | lean_m + final generic-ballot D-R margin; lean_m = 200 x (prior presidential D two-party share - national) | /20 |
| inc | +1 anti-R incumbent, -1 R incumbent, 0 open | 1 |
| poll_m | race poll D-R margin, final 60 days, weights exp(-days/21) x sqrt(n/600), pollster-shared, nominee-matched; NaN -> 0 + missing flag | /10 |
| prev_rel | previous House D margin minus that year's national House margin (unchanged lines only); NaN -> 0 + missing flag | /20 |
| **gb_mid** (v3) | generic-ballot D-R margin x 1[midterm] | /10 |

## v3.4 Model, calibration, floor

1. Training table: 3,450 contested D(or anti-R independent)-vs-R generals 2010-2024 (House 3,173, Senate 277),
   each row plus its mirror image, so the model has no partisan base rate (intercept 0).
2. `LogisticRegression(C=0.3)` on the six features + two missing flags. Coefficients (2026 model, cycles 2010-2024):
   r_mean 2.24, base_m 1.39, inc 0.49, poll_m 0.84, prev_rel 0.49, gb_mid 1.07 (missing flags 0). The gb_mid
   coefficient is 0.81 when fitted on cycles < 2018 and 1.04 on cycles < 2022: stable, where v2's op went
   0.75 -> 0.80 -> 0.34. For 2026: 1.07 x 0.856 = +0.92 logit toward D in every race (v2's op: +0.34).
3. Slope-only Platt: logit q = b x logit p_raw, b fitted on rolling OOF predictions of earlier cycles (each fitted
   on cycles before it), clipped to [0.5, 1.25]: b = 1.25 for 2018, 0.99 for 2022, 1.06 for 2026.
4. q clipped to [0.01, 0.99] unless all three raters say Safe.

## v3.5 Special races, assembly, late information, simulation (unchanged from v2)

- Sides: R vs anti-R (Democrat or a viable independent: NE/MT/ID/SD-SEN, AK-AL, rated-independent rules in v2
  section 3.3); CA-06 Kiley is the R side against Pan; unopposed races get 0.998+ for the only viable side.
- Same-party contests (9 CA top-two generals, LA jungle, AK, CA-SEN history): softmax of a x incumbent +
  b x log10(receipts) + c x primary-share deviation (+ 0.08 x poll points, hand-set, 2026 only), ridge MLE on the
  52 same-party finals before the predicted year (2026: c 0.085/pt, a 0.62, b 0.44; no-primary fit a 1.17,
  b 1.15). Minor candidates 0.0005 each. Details: section 3.2.
- 2026 late-information pool (85 dossiers): logit pool of the Kalshi underdog's share with the dossier mean,
  w = 0.25 (medium/high confidence), 0.15 (low / refuted), 0.5 (same-party). Moves House E[D] 238.48 -> 238.77.
  The w values are not validated (no history); `q_model` keeps the pre-pool value.
- Simulation: Gaussian copula with national factor rho_n = 0.228 (from the v3 OOF per-year probit shifts 2014
  -0.33, 2016 -0.37, 2018 +0.24, 2020 -1.14, 2022 +0.40, 2024 -0.14) and state factor 0.05 (hand-set); marginals
  equal q; 20,000 runs, seed 20261103. Senate: 34 D-caucus + 31 R seats not up, D control at 51; House 218.

## v3.6 Validation and selection

Criteria (`experiments_v3.py` -> `out/experiments_v3.csv`): (1) LOYO inside 2010-2016 (v2's pre-registered
criterion, never sees 2018+), (2) **LOYO over all eight cycles 2010-2024 (new; includes 2018 and 2022, so it is a
test-year-informed criterion)**, (3) rolling 2018-2024 for transparency. Log loss, all races (contested in brackets):

| config | LOYO pre-2018 | LOYO all 8 | roll 2018 | roll 2020 | roll 2022 | roll 2024 |
|---|---|---|---|---|---|---|
| logit_v1 (no midterm term) | 0.0704 (0.2748) | 0.0770 (0.2972) | 0.0853 | 0.1191 | 0.0812 | 0.0473 |
| logit_op = v2 | 0.0666 (0.2601) | 0.0783 (0.3022) | 0.0812 | 0.1156 | 0.1014 | 0.0478 |
| **logit_gbmid = v3** | 0.0682 (0.2670) | **0.0744 (0.2878)** | 0.0781 | 0.1157 | 0.0802 | 0.0468 |
| logit_op_gbmid | 0.0668 (0.2610) | 0.0769 (0.2981) | 0.0810 | 0.1150 | 0.0947 | 0.0474 |
| env_gb (true N, own N_hat) | 0.0698 (0.2719) | 0.0767 (0.2954) | 0.0881 | 0.1142 | 0.0837 | 0.0468 |
| env_e (true N, kavi-e N_hat) | 0.0689 (0.2686) | 0.0760 (0.2932) | 0.0856 | 0.1150 | 0.0842 | 0.0467 |
| env_e_op | **0.0664 (0.2587)** | 0.0779 (0.3004) | 0.0814 | 0.1125 | 0.1022 | 0.0471 |
| env_e_gbmid (reverted) | 0.0674 (0.2633) | 0.0740 (0.2859) | 0.0782 | 0.1122 | 0.0831 | 0.0462 |
| env_e_mid | 0.0676 (0.2637) | 0.0752 (0.2899) | 0.0807 | 0.1133 | 0.0867 | 0.0464 |
| env_e_sen | 0.0696 (0.2713) | 0.0760 (0.2934) | 0.0842 | 0.1152 | 0.0839 | 0.0471 |
| env_e_gbclose | 0.0683 (0.2663) | 0.0752 (0.2897) | 0.0814 | 0.1125 | 0.0836 | 0.0465 |
| env_e_gbmid_polls | 0.0674 (0.2631) | 0.0736 (0.2842) | 0.0771 | 0.1120 | 0.0827 | 0.0458 |

The "env" variants (kavi-e's idea) train base_m on lean + the ACTUAL national House margin and predict with a
pre-election N_hat (my GB-bias-corrected estimate, or kavi-e's combined estimate for midterms) integrated over
N ~ Normal(N_hat, sd >= 2.5) with 9-node Gauss-Hermite (`environment.py`).

How the choice was made (and where it used the test years):
- The pre-2018 LOYO has only two midterms, both D-president R waves, so it cannot distinguish "out-party shift"
  from "generic-ballot shift"; it prefers op by 0.0016 (2014's generic ballot, R+1.9, understated the wave).
- I chose gb_mid on the structural argument above plus the all-8 LOYO (0.0744 vs 0.0783 for op). Both the report's
  diagnosis and the all-8 LOYO use 2018/2022 outcomes: **test-year-informed**.
- env_e_gbmid was my first v3 candidate (best on both LOYO criteria among the gb_mid family) but scored worse than
  logit_gbmid on the scorecard (2022 0.0838 vs 0.0810), so it was **reverted** per the round rule: a second
  test-year choice.

Scorecard (`src/scorecard.py`, all 470 races, out of sample; 2018 trained on 2010-2016, 2022 on 2010-2020):

```
kavi-d 2018: n=470 logloss 0.0742 (538 0.1002) brier 0.0407 (538 0.0578) acc 0.977 score top 48.84 ev 48.57 (chalk 42.13) against-538 10; dropped 0
       2018 contested (538 fav<0.95): n=120 logloss 0.2764 (538 0.3778) brier 0.1554 (538 0.2257) acc 0.908 (538 0.875)
kavi-d 2022: n=470 logloss 0.0810 (538 0.0952) brier 0.0497 (538 0.0576) acc 0.966 score top 34.43 ev 42.84 (chalk 30.43) against-538 11; dropped 0
       2022 contested (538 fav<0.95): n=90 logloss 0.3755 (538 0.4375) brier 0.2390 (538 0.2787) acc 0.833 (538 0.822)
```

| version | 2018 all / contested | 2022 all / contested | 2018 House / Senate | 2022 House / Senate |
|---|---|---|---|---|
| v2 (final_v2) | 0.0771 / 0.2877 | 0.1008 / 0.4701 | 0.0693 / 0.1741 | 0.0963 / 0.1572 |
| env_e_gbmid (tried, reverted) | 0.0743 / 0.2768 | 0.0838 / 0.3887 | | |
| **v3 (final_v3 = logit_gbmid)** | **0.0742 / 0.2764** | **0.0810 / 0.3755** | 0.0668 / 0.1661 | 0.0782 / 0.1153 |
| 538 | 0.1002 / 0.3778 | 0.0952 / 0.4375 | 0.0938 / 0.1799 | 0.0914 / 0.1414 |

OOF midterm check (`out/analysis_midterm.csv`): toss-ups, model vs actual out-party share: 2014 0.60 vs 0.69,
2018 0.66 vs 0.67, 2022 0.55 vs 0.26 (v2 0.69). OOF calibration (`analysis_calibration.csv`): bins 0.15-0.30 and
0.70-0.95 still over-predict the anti-R side (0.22 vs 0.11; 0.78 vs 0.71; 0.90 vs 0.86): the symmetric model gives
Democrats a little too much, consistent with the generic ballot's average D bias (-2.2 pts 2010-2024); a bias
correction was tested (env_gb, hat-only) and lost on the rolling years.

## v3.7 2026 output

- House: E[D] 238.8 (model-only 238.5; v2 234.9; Kalshi sum of normalised mids 246.0; investigation ~238); median
  239, 80% 228-249, P(D control) 0.992. Senate D-caucus median 53, 80% 50-55, P(D control) 0.845 (v2 0.717).
- Picks against the Kalshi favourite (6, `out/disagreements_2026.csv`): IA-SEN Turek D 0.62 (Kalshi 0.44),
  AL-02 Figures D 0.52 (0.43), CO-05 Killin D 0.56 (0.45), WI-01 Steil R 0.68 (0.50), CA-34 Gomez 0.55 (0.18),
  CA-40 Calvert 0.61 (0.48). v2's KS-SEN, MI-04, MI-10 fades are gone (the larger D environment term now agrees
  with Kalshi there). Largest q moves vs v2: ME-02 Dunlap 0.34 -> 0.46, AL-02 Figures 0.40 -> 0.52, TX-35 Garcia
  0.61 -> 0.72, MI-10 Hines 0.49 -> 0.61, KS-SEN Hamilton 0.45 -> 0.56, IA-SEN Turek 0.51 -> 0.62.

## v3.8 Known weaknesses (v3)

- gb_mid is identified by four midterms; 2026's D+8.56 is at the top of that range (2018 +8.6), so the 2026 push
  (+0.92 logit everywhere) is an extrapolation of a 2018-like year. If the 2026 generic ballot overstates Democrats
  as it did on average 2010-2024, toss-up and Lean-R D prices are too high.
- Selection used 2018/2022 (gb_mid over op, revert of env): the v3 scorecard is optimistic by an unknown amount.
- Ratings still dominate (permutation importance 0.94 of ~1.04); polls add little.
- Same-party contests and the dossier weights are unvalidated (as v2). Safe-by-all-three tails 0.998-0.9995.
- Simulation rho_n from six cycles, rho_s hand-set, runoffs not modelled.

## v3.9 Borrowed in v3

| from | what | status | effect |
|---|---|---|---|
| pau-a, pau-b | generic ballot x midterm as the midterm environment term | adopted (one symmetric logit term) | scorecard 2022 0.1008 -> 0.0810, 2018 0.0771 -> 0.0742 (vs v2) |
| kavi-e | train on actual national House margin, predict with N_hat + integrated uncertainty; its N_hat table | tested, reverted | 2022 0.0810 -> 0.0838 |
| kavi-c | wave concentrated in competitive seats | tested (gb_mid_close) | LOYO worse, not used |
| REPORT.md 1.4 | diagnosis of the op failure | motivated the change | - |

---------------------------------------------------------------------------------------------------------------------

# kavi-d: METHODS (v2, previous version, kept for reference) — symmetric classifier with a midterm out-party term, upset analysis

Branch `kavi/kavi-d-1101`, code in `src/models/kavi-d/`, outputs in `out/` (v1 outputs kept in `out/v1/`, design
variants in `out/variants/`). Everything below is reproduced by `src/models/kavi-d/run_all.sh` from `data/`.

One-line summary (v2): a D<->R **mirror-symmetric L2 logistic regression** on expert ratings, partisan lean +
generic ballot, incumbency, previous margin, race polls and a **midterm out-party indicator** (+1 when the anti-R
side is the out-party in a midterm, -1 when it is the president's party), trained on 3,450 contested House/Senate
races 2010-2024, slope-only Platt calibration on rolling out-of-sample predictions, a 1% floor for races not rated
Safe by all three raters; a same-party candidate model with **fitted** top-two-primary, incumbency and money
weights; for 2026 only, a small, disclosed **late-information pool** with the 85 research dossiers; a 20,000-run
Gaussian-copula simulation for seat totals.

---------------------------------------------------------------------------------------------------------------------

## 1. Data used

All from the side's data layer (`data/clean/*.csv`; each `.md` note was read before use) unless marked. Files in
`data/extra/` were collected by me or copied from a sibling (marked "borrowed"; each has a `.md` note).

| file | source | years | n used | how it is used / cleaning |
|---|---|---|---|---|
| results_house.csv | MIT MEDSL House 1976-2024 v15 | 2000-2024 | 5,656 rows | training targets (winner party, D two-party share), previous-cycle margin, national House margin per year |
| results_senate.csv | MIT MEDSL Senate v8 (+CT 2022 canvass) | 2010-2024 | 281 rows | training targets; independents (King, Sanders, Orman, McMullin, Osborn) parsed from the `major non-D/R candidate` note |
| raw MIT house/senate files (`repo/data/raw/mit/`) | same | 2012-2024 | 52 same-party finals (102 finalists) | both finalists of same-party top-two generals, for the same-party model |
| ratings_hist.csv / ratings_2026.csv | Wikipedia ratings tables | 2010-2026 | 3,760 + 470 | Cook, Sabato, Inside numeric ratings (sign flipped, + = anti-R), their mean/spread, imputed-safe flag, raw text ("Safe I"), `incumbent_party`, `pvi_wiki`. `other_*` columns tested as a feature (oth_dev, rcp_dev; rejected, section 8) |
| lean_hist.csv / lean_2026.csv | Downballot/DKE, MIT president | 2010-2026 | 3,751 + 470 | lean_m = 200 x (prior pres. D 2-party share - national); 2026 = pres24 on the 2026 lines minus 0.4925; blank CA 2012 / FL 2016 filled from -2 x Cook PVI |
| national_hist.csv / national_2026.csv | 538 generic-ballot finals; 2026 aggregators | 2010-2026 | 8 + 5 | final generic D-R margin (2026: mean of five 10-30 aggregates, D+8.56); `cycle_type` and `president_party` for the out-party term |
| polls_hist.csv / polls_2026.csv | 538 archives; Wikipedia + VoteHub | 2010-2026 | 8,767 / 1,513 rows | race poll average, final 60 days, nominee-matched (section 2). Sponsor tags are not used (so the RedBox tag inconsistency does not matter here) |
| fec_hist.csv / fec_2026.csv / fec_race_2026.csv | FEC bulk | 2010-2026 | 27,186 / 951 / 470 | training: only the I/C/O flag (incumbency) and same-party training receipts; 2026: receipts to 2026-10-14 (pre-election) for the same-party split and side principal. Not in the party model |
| **borrowed (kavi-b)** data/extra/fec_2018_preelection.csv | Wayback capture 2018-11-07 of FEC `weball18.zip` (file dated 2018-11-06) | 2018 | 3,550 H/S candidates | pre-election 2018 receipts for the 2018 backtest (cand_id join, else surname + given name in the race) |
| **borrowed (kavi-b)** data/extra/primaries_top_two.csv | Wikipedia CA/WA top-two primary boxes | 2012-2026 | 2,418 rows; 98 of 102 finalists matched | primary share of each finalist: fits the primary weight of the same-party model and feeds 2018/2022 backtest splits |
| **borrowed (kavi-b)** data/extra/ca_polls_2026.csv | Wikipedia CA district sub-pages | 2026 | 37 polls, 10 districts | extra post-primary polls for the CA same-party contests (de-duplicated against polls_2026) |
| data/extra/ca_primary_2026.csv (mine) | Wikipedia CA district pages (revids in its .md) | 2026 | 297 rows, 52 districts | June 2 2026 top-two primary shares for the 9 CA same-party generals |
| repo/data/dossiers/summary.csv + <race>.md | research + skeptic review per race, as of Nov 1 | 2026 | 85 races | late-information pool for 2026 only (section 3.6). No historical analogue |
| data/kalshi/races_2026-11-01T074812Z.csv | Kalshi snapshot | 2026 | 470 races, 951 slots | ONLY the slot list and the p_kalshi column (mid / sum of mids). Never a model input |
| data/benchmark/fte_final_{2018,2022}.csv | 538 final forecasts | 2018, 2022 | 1,130 / 856 rows | ONLY the ballot list for the backtests; 538's probabilities and its `other_fte` ratings are never read |

Training table (`history.py`, `out/cache/train_features_w60.csv`): every 2010-2024 general between an anti-R side
and a Republican: **3,450 races** (House 3,173, Senate 277). Dropped: same-party finals (to the same-party model),
one-sided races, the January 2010 MA-SEN special. `y` = 1 if the winner is not the Republican.

Data traps handled: redistricting (prev margin and seat party blanked where lines changed; 2026: the 9 redrawn
states except the districts flagged unchanged); Senate specials keyed `-SEN-S`; LA jungle = eventual winner's
party; withdrawn Democrats and weak Democrats next to forced independents are minor; NE/MT/ID/SD-SEN and AK-AL use
the independent as the anti-R side; CA-06 Kiley (ex-R incumbent, no Republican on the ballot) is the R side against
Pan; unopposed races 0.998+ to the only viable side; Kalshi party-only slots mapped to the party's candidate;
pred_2026 `party` is the candidates' real party (DEM for both finalists in the 8 CA D-vs-D races, REP for both in
CA-40, IND for Kiley and the independents), never UNK. v2 matching fixes: FEC rows disambiguated by given name
(WA-09 2018 Adam vs Sarah Smith), ignoring money-less stub committees (GA-14 2018 "GRAVES, TOM"); one-edit surname
spelling variants with the same first name (KY-03 2026 Rodriguez/Rodriquez).

## 2. Features and why

All signed so that + favours the anti-R side, so that the mirror image of a row is the row with every signed
feature negated and the outcome flipped.

| feature | definition | why | in final_v2 |
|---|---|---|---|
| r_mean | -1 x mean of Cook/Sabato/Inside (-3..3) | experts aggregate late and non-public information | yes (/1.5) |
| base_m | lean_m + final generic-ballot margin | expected margin before candidate effects; the generic enters only through it (8 distinct values would be a year id) | yes (/20) |
| inc | +1 anti-R incumbent, -1 R incumbent (FEC I flag; candidates_2026 in 2026) | incumbency | yes |
| poll_m | race poll D-R margin, final 60 days, weights exp(-days/21) x sqrt(n/600) (n clipped 100-3000), each pollster's weights divided by its poll count, nominee-matched | direct measurement | yes (/10, NaN -> 0 + missing flag) |
| prev_rel | previous House D margin - that year's national House margin (unchanged lines) | candidate over/under-performance | yes (/20, NaN -> 0 + missing flag) |
| **op** (new) | +1 if midterm and president R (2018, 2026), -1 if midterm and president D (2010, 2014, 2022), 0 otherwise | in midterms the out-party beat the ratings in 2010, 2014 and 2018 (the investigation's main finding); the term is D/R symmetric: it says "out-party", not "Democrat" | yes |
| op_close (new) | op x (1 - |r_mean|/3) | lets the out-party shift concentrate in competitive seats | tested, no |
| oth_dev, rcp_dev (new) | mean of the non-core outlets (538 excluded) / RCP alone, minus the core mean | faster-moving second opinion | tested, no |
| sparse poll window (new) | if < 2 matched polls in 60 days, use 150 days | summer polls in under-polled races (OH-07, TX-15) | tested, no |
| r_spread, seat, fec_*, poll_n, senate, lean_m | as in v1 | | GBM only (not in final_v2) |

## 3. Model and parameters

### 3.1 Party model (`model.py`, config `final_v2` = `logit_op` in `experiments_v2.py`)

1. **Symmetric training**: every training row is appended with its D<->R mirror image (signed features incl. op
   negated, y flipped). The model has no partisan base rate; it can only learn "the out-party in a midterm does
   better than its inputs say", never "Democrats do".
2. **Logistic regression** `LogisticRegression(C=0.3)` on r_mean/1.5, base_m/20, inc, poll_m/10, prev_rel/20, op
   (+ missing flags for poll_m and prev_rel). Coefficients for the 2026 model (2010-2024, mirrored, so the
   intercept is 0): r_mean 2.27, base_m 1.42, inc 0.43, poll_m 0.85, poll-missing 0, prev_rel 0.28,
   prev-missing 0, **op 0.34**. The op coefficient is 0.75 when fitted on cycles < 2018 and 0.80 on cycles < 2022
   (all earlier midterms had the out-party beating the ratings); adding 2022 (out-party under-performed) halves it.
   Per held-out midterm out-party shifts (base model fitted on the other cycles): 2010 +0.47, 2014 +0.74, 2018
   +1.29, 2022 -0.73 logit (mean +0.44, sd 0.85), so 0.34 is in line with the cross-midterm mean.
3. **Calibration**: slope-only Platt, logit(q) = b x logit(p_raw), b clipped to [0.5, 1.25], fitted on rolling
   OOF predictions of the cycles before the predicted year (each fitted on the cycles before it). b = 1.25 for
   2018 (cycles 2014-16), 0.98 for 2022 (2014-20), 1.01 for 2026 (2014-24). No intercept (v1 finding: an
   intercept learns the calibration cycles' partisan error).
4. **Floor**: q clipped to [0.01, 0.99] unless all three raters say Safe.

### 3.2 Same-party / within-party model (`sameparty.py`)

Strength s_i = a x incumbent_i + b x log10(receipts_i + 1e4) + c x (primary share_i - pair mean, points)
(+ 0.08 x poll points above the pair mean, 2026 only, hand-set); P(i) = softmax(s). (a, b, c) by ridge MLE
(penalty 0.5 on a, b and 10c) on the 52 same-party finals before the predicted year. **v2: c is fitted** (v1 hand-set
0.04): c = 0.062 / 0.080 / 0.085 per point (for 2018 / 2022 / 2026); a = 0.00 / 0.37 / 0.62; b = 0.73 / 0.35 /
0.44. Contests without a top-two primary (LA jungle, AK, CA-SEN) use a fit without primary (2026: a = 1.17,
b = 1.15). The 2022 backtest uses fits with b = 0 (no pre-election money): a = 0.49, c = 0.084 with primary;
a = 1.81 without. Minor candidates get 0.0005 each.

### 3.3 Assembly (`assemble.py`, `backtest.py`, `predict_2026.py`)

Sides: R = Republicans; anti-R = Democrats + viable independents. Backtests (v2, money-free viability): an
independent is viable if he is the race's leading independent and either the race has no Democrat and is not Safe
R for all three raters (UT-SEN 2022 McMullin), or the raters rate it for an independent ("Safe I": King,
Sanders); the nominal Democrat is then minor. 2026: candidates_2026's `is_major`. P(anti-R) is split within each
side by the same-party softmax.

### 3.4 Backtest money policy (audit fix)

2018: pre-election receipts from the Wayback snapshot (borrowed from kavi-b). 2022: no pre-election cut exists, so
**no money is used anywhere in the 2022 backtest** (same-party splits by incumbency + primary only). Training rows
use FEC only for the I/C/O status flag. 2026 FEC ends 2026-10-14 (pre-election).

### 3.5 Simulation (`simulate.py`)

Gaussian copula, X_i = sqrt(rho_n) Z_nat + sqrt(rho_s) Z_state + sqrt(1-rho_n-rho_s) e_i, anti-R wins when
X_i < Phi^-1(q_anti-R,i), so marginals equal q. rho_n = s^2/(1+s^2), s = RMS of the per-year probit shifts that
maximise the likelihood of the final_v2 rolling OOF predictions (2014 -0.15, 2016 -0.35, 2018 +0.26, 2020 -1.14,
2022 +0.73, 2024 -0.11): rho_n = 0.254. rho_s = 0.05 (hand-set). 20,000 runs, seed 20261103. Senate: 34 D-caucus
and 31 R seats not up; D control needs 51 (VP Vance). House control: 218.

### 3.6 Late-information pool (2026 only, `late_info.py`)

For each of the 85 dossier races: s = underdog's share of (favourite + underdog) q, where favourite/underdog are
Kalshi's; logit s' = (1-w) logit s + w logit(q_dog_mean) with q_dog_mean = mean of the researcher's and the
skeptic's underdog probability. **w = 0.25** for medium/high confidence, **0.15** for low confidence or a skeptic
"refuted" verdict, **0.5** for same-party contests (CA-04, CA-07, CA-11, CA-34, CA-40), where the model's own split
rests on unvalidated weights. The general w was set before looking at any total; the same-party exception was set
after seeing CA-34 (model Gomez 0.83, dossier 0.22; an August ethics investigation is in no model input). There is
no historical analogue to fit w: the dossiers are partly anchored on Kalshi's price, and the model keeps >= 75%
of the logit weight outside same-party races. Effect: House E[D] 234.19 -> 234.87; largest moves CA-34 Gomez
0.833 -> 0.546, CA-40 Calvert 0.723 -> 0.606, CA-07 Vang 0.16 -> 0.26, NC-03 Smith 0.031 -> 0.062, FL-27
0.114 -> 0.156, FL-07 0.073 -> 0.108, OH-07 Miller 0.395 -> 0.363. `q_model` (before the pool) is kept in
pred_2026.csv and `out/late_info_2026.csv` logs every adjustment.

## 4. Training and validation

- **Selection criterion (v2, new): leave-one-year-out inside 2010-2016** — train on the other three pre-2018
  cycles, predict the held-out one, no calibration step, 1% floor. It never sees 2018+. All design options in
  section 8 were ranked on it. Contested = |r_mean| < 3.
- **Rolling origin 2018-2024** (fit + calibration on cycles < Y) is reported for transparency only.
- **Backtests**: 2018 trained on 2010-2016, 2022 on 2010-2020; inputs for Y are Y's final pre-election ratings,
  prior presidential lean, Y-2 results, Y's final generic average, polls ending in the final 60 days, top-two
  primary shares, pre-election money (2018) or none (2022). 538's file supplies only the ballot list.
- **2026**: trained on 2010-2024, calibration slope from OOF 2014-2024.

`out/experiments_v2.csv` (log loss; LOYO = mean of 2010/12/14/16 folds; window 60 days unless noted):

| config | LOYO all | LOYO contested | 2010 | 2012 | 2014 | 2016 | roll 2018 | roll 2020 | roll 2022 | roll 2024 |
|---|---|---|---|---|---|---|---|---|---|---|
| oth | 0.0751 | 0.2948 | 0.1121 | 0.0798 | 0.0649 | 0.0436 | 0.0856 | 0.1190 | 0.0857 | 0.0489 |
| v1_final | 0.0734 | 0.2892 | 0.1054 | 0.0802 | 0.0661 | 0.0419 | 0.0830 | 0.1215 | 0.0848 | 0.0494 |
| rcp | 0.0732 | 0.2889 | 0.1042 | 0.0795 | 0.0675 | 0.0415 | 0.0877 | 0.1174 | 0.0897 | 0.0494 |
| op_oth | 0.0719 | 0.2815 | 0.1076 | 0.0780 | 0.0610 | 0.0408 | 0.0812 | 0.1183 | 0.1002 | 0.0493 |
| op_close | 0.0718 | 0.2818 | 0.1046 | 0.0791 | 0.0633 | 0.0401 | 0.0793 | 0.1206 | 0.0966 | 0.0496 |
| logit_oth | 0.0713 | 0.2786 | 0.1035 | 0.0773 | 0.0612 | 0.0432 | 0.0875 | 0.1176 | 0.0833 | 0.0474 |
| op_both | 0.0713 | 0.2795 | 0.1044 | 0.0788 | 0.0624 | 0.0395 | 0.0790 | 0.1216 | 0.0981 | 0.0494 |
| op | 0.0711 | 0.2789 | 0.1039 | 0.0789 | 0.0622 | 0.0395 | 0.0792 | 0.1207 | 0.0987 | 0.0496 |
| op_rcp | 0.0710 | 0.2790 | 0.1030 | 0.0782 | 0.0631 | 0.0396 | 0.0821 | 0.1168 | 0.1051 | 0.0498 |
| logit_v1 | 0.0704 | 0.2748 | 0.1009 | 0.0775 | 0.0620 | 0.0413 | 0.0853 | 0.1191 | 0.0812 | 0.0473 |
| logit_op_oth | 0.0672 | 0.2628 | 0.0965 | 0.0756 | 0.0567 | 0.0400 | 0.0828 | 0.1146 | 0.1042 | 0.0479 |
| **logit_op = final_v2** | 0.0666 | 0.2601 | 0.0948 | 0.0758 | 0.0574 | 0.0385 | 0.0812 | 0.1156 | 0.1014 | 0.0478 |
| logit_op_rcp | 0.0664 | 0.2598 | 0.0942 | 0.0751 | 0.0574 | 0.0388 | 0.0837 | 0.1145 | 0.1106 | 0.0483 |
| v1_final (150 d) | 0.0734 | 0.2892 | 0.1054 | 0.0802 | 0.0661 | 0.0419 | 0.0837 | 0.1203 | 0.0847 | 0.0490 |
| op (150 d) | 0.0711 | 0.2789 | 0.1039 | 0.0789 | 0.0622 | 0.0395 | 0.0794 | 0.1192 | 0.0978 | 0.0490 |
| logit_v1 (150 d) | 0.0704 | 0.2748 | 0.1009 | 0.0775 | 0.0620 | 0.0413 | 0.0871 | 0.1179 | 0.0807 | 0.0470 |
| logit_op (150 d) | 0.0666 | 0.2601 | 0.0948 | 0.0758 | 0.0574 | 0.0385 | 0.0829 | 0.1144 | 0.1001 | 0.0475 |

## 5. Scorecard numbers (src/scorecard.py, all 470 benchmark races, out of sample)

```
kavi-d 2018: n=470 logloss 0.0771 (538 0.1002) brier 0.0426 (538 0.0578) acc 0.974 score top 47.01 ev 47.51 (chalk 42.13) against-538 9; dropped 0
       2018 contested (538 fav<0.95): n=120 logloss 0.2877 (538 0.3778) brier 0.1629 (538 0.2257) acc 0.900 (538 0.875)
kavi-d 2022: n=470 logloss 0.1008 (538 0.0952) brier 0.0653 (538 0.0576) acc 0.953 score top 30.56 ev 37.65 (chalk 30.43) against-538 15; dropped 0
       2022 contested (538 fav<0.95): n=90 logloss 0.4701 (538 0.4375) brier 0.3195 (538 0.2787) acc 0.767 (538 0.822)
```

| | 2018 House | 2018 Senate | 2022 House | 2022 Senate |
|---|---|---|---|---|
| log loss (538) | 0.0693 (0.0938) | 0.1741 (0.1799) | 0.0963 (0.0914) | 0.1572 (0.1414) |
| accuracy top | .977 | .943 | .956 | .914 |

Reading: v2 beats 538 clearly in 2018 (all 0.0771 vs 0.1002, contested 0.288 vs 0.378) and loses to it in 2022
(0.1008 vs 0.0952, contested 0.470 vs 0.438). The 2022 loss is the midterm term: with op fitted on 2010/2014/2018
(three out-party over-performances), 2022 toss-ups were pushed to the Republicans (model 0.69 out-party, actual 7
of 27) and D-held Lean seats too (WA-03, NM-02, CO-08, PA-SEN, OH-13, OH-01, NC-13 are the largest misses).

Largest 2018 misses: OK-05 Horn 0.01 (floor), FL-SEN Scott 0.035 (v1 0.13), CA-21 Cox 0.055, NY-11 Rose 0.077,
SC-01 Cunningham 0.080 (v1 0.033); GA-06 McBath and TX-07 Fletcher left the bottom list. 2022: WA-03 Gluesenkamp Perez 0.025, NM-02 Vasquez 0.14, CO-08 Caraveo 0.15, PA-SEN Fetterman 0.19.

## 6. 2026 output and the picks that go against the Kalshi favourite

Seat simulation (`out/sims_summary.json`, 20,000 correlated runs): **House D seats mean 234.9, median 235, 80%
interval 223-246 (90%: 220-250), P(D control) 0.968**; Kalshi's sum of normalised D prices is 246.0 (inflated by
9.2 expected D seats in the 166 Safe-R-by-mean seats, 0.6 for us). **Senate D-caucus median 52, 80% 49-54,
P(D control) 0.717** (0.724 counting the new independents as anti-R).

Expected D seats by mean core rating (House, all 435 seats; v2 final q vs Kalshi): Safe R 0.6 (Kalshi 9.2), Likely R 1.0 (4.7), Lean R
2.9 (5.2), Tilt R 1.0 (1.4), Toss-up 5.3 of 9 (5.6), Tilt D 3.0 of 4 (3.2), Lean D 16.5 of 19 (16.5), Likely D 12.8
(12.2), Safe D 174.8 (171.4). Versus v1 (model-only q) v2 moved +0.7 in Lean R, +0.4 in Toss-up, +0.4 in Tilt D,
+0.6 in Lean D. The remaining gap to Kalshi is almost all in R-rated longshots (favourite-longshot compression).

Against the Kalshi favourite (top q != Kalshi favourite), 7 races (`out/disagreements_2026.csv`):

| race | Kalshi fav (p) | our pick: q (Kalshi p) | reason |
|---|---|---|---|
| WI-01 | Berman D 0.50 | Steil R 0.75 (0.50) | Lean R by all three raters, R incumbent who ran 7.7 pts ahead of the national environment in 2024; two polls (D +0.6) too few to outweigh it. Dossier (Steil 0.535, low conf) pulls it from 0.78 to 0.75 |
| CA-34 | Gonzales-Torres D 0.82 | Gomez D 0.55 (0.18) | Same-party. Incumbent Gomez led the primary 46.0-30.7 (the fitted primary weight is large) and has 2.6x the money; post-primary polls have Gonzales-Torres +6. Dossier (ethics investigation since August; Gomez 0.22) at w = 0.5 takes the model's 0.83 to 0.55. Least trustworthy pick |
| CA-40 | Kim R 0.52 | Calvert R 0.61 (0.48) | Same-party R vs R, both incumbents. Calvert led the primary 34.9-20.6; Kim leads the polls 36.3-34.0. Dossier (Calvert 0.475) at w = 0.5 |
| KS-SEN | Hamilton D 0.505 | Marshall R 0.55 (0.495) | Toss-up by all three, 19 polls D+2.8, but R+15 state (lean+generic -6.4), R incumbent; the out-party term (+0.35) is not enough. Dossier 0.43 for Marshall moves it 0.58 -> 0.55 |
| IA-SEN | Hinson R 0.56 | Turek D 0.51 (0.44) | Toss-up, 31 polls D+0.2, open seat; the out-party term tips a coin flip. Dossier 0.435 brings it back from 0.53 to 0.51: a coin flip either way |
| MI-04 | McCann D 0.615 | Huizenga R 0.52 (0.385) | Toss-up, R incumbent with +9 personal over-performance, 2 polls R+2.5 |
| MI-10 | Hines D 0.76 | Bouchard R 0.51 (0.24) | Open, Toss-up/Tilt R (mean -0.17), the only poll (RedBox Oct 20-23, nonpartisan) R+5; lean+generic +3.4 D. The quantitative outlets have it Lean D, which the model does not use |

Under our own q, picking our top candidate is worth 54.1 expected points vs 50.5 for Kalshi chalk; the
q x -log2(p) rule picks against Kalshi in 52 races (63.2 expected points under our q), mostly R-rated D longshots
that Kalshi prices at 10-25% and we at 1-10% (i.e. picking the R favourite is cheap) and toss-ups. That is only as
good as our tail calibration.

## 7. Upset analysis (what predicts the rating favourite losing)

`analysis.py`, final_v2 rolling OOF 2014-2024.

- Upset rates: Lean 0.18 (n = 198), Likely 0.035 (n = 154), Safe 0.002 (n = 2,102); model means per bucket match
  (Lean/polls fav < +5: 0.18 vs 0.175; Likely/no polls 0.043 vs 0.042).
- OOF calibration bins: 0.05-0.15 0.093 -> 0.098, 0.15-0.30 0.226 -> 0.204, 0.30-0.50 0.392 -> 0.500,
  0.50-0.70 0.616 -> 0.608, 0.70-0.85 0.781 -> 0.712, 0.85-0.95 0.905 -> 0.875. Better than v1 in the tails
  (v1 0.05-0.30 over-predicted D by 2x).
- **Midterms, by who the ratings favour** (`out/analysis_midterm.csv`, OOF): out-party Lean seats won 11/11 (2014),
  22/23 (2018), 14/15 (2022), model 0.88-0.89: still too low; toss-ups went 69% (2014) and 67% (2018) to the
  out-party, model 0.67 / 0.64, but only 26% in 2022 (model 0.69). In-party Lean: out-party won 29% (2014), 13%
  (2018), 5% (2022) vs model 0.25 / 0.12 / 0.34.
- Standardised logistic of upset on favourite-signed features (Lean/Likely, n = 352, 10.8% upsets): rating strength
  -1.09, no previous result -0.55, favourite is a Democrat +0.54, lean+generic toward the favourite -0.44, rater
  spread +0.21, poll lead -0.17, environment +0.15.
- Permutation importance (final_v2, log-loss increase when shuffled, mean over 2014-2024): r_mean 0.96, base_m
  0.077, poll_m 0.008, inc 0.003, prev_rel 0.000 (op is constant within a cycle, so it has no within-cycle
  permutation importance; its value shows in the LOYO table).

## 8. What was tried and failed (v1 and v2)

- v1: free-intercept Platt (2018 0.205), plain 17-feature GBM (overfits), margin regression, FEC money in the party
  model (no gain, leak risk), ratings-free model (0.088 pre-2018).
- v2, on LOYO pre-2018: logit_op_rcp was 0.0002 lower than logit_op (0.0664 vs 0.0666) but not chosen (one more
  feature for a negligible gain; it is worse in 3 of the 4 rolling years). Other raters' deviation oth_dev (0.0751 vs 0.0734 blend; 0.0713 vs 0.0704 logit) and
  rcp_dev (neutral) — the historical other_* columns are media raters (RCP, Daily Kos, NYT, CQ), so this says
  nothing definitive about the 2026 quantitative outlets, but there is no history to back their use; op_close
  (worse than plain op); op_both (equal to op); the GBM+logit blend with op (0.0711 vs 0.0666 for logit_op).
- Sparse 150-day poll window: a wash on 2018-2024 rolling (cannot be tested earlier). Not used.
- A year-level "rater lag" index (mean residual of ratings on lean+generic) to predict which midterms the
  out-party beats the ratings: it does not separate 2022 from 2010/2014/2018. Dropped.
- Same-party model with incumbency fitted jointly with the primary, applied to contests that have no primary: LA
  jungle incumbents fell to 0.33 in 2022. Fixed by the no-primary fit.

## 9. Known weaknesses

- **The midterm term is a bet on 2026 resembling 2010/2014/2018 more than 2022.** It is fitted (0.34 logit,
  halved by 2022) not hand-set, but it rests on four midterms, one of which went the other way; in 2022 it cost
  0.019 log loss. If 2026 is another "wave that wasn't", our toss-up and Lean-D prices are too high.
- The ratings dominate (permutation importance 0.96 of ~1.05); the model cannot see late news the core raters have
  not priced in, except through the 85-race dossier pool.
- The quantitative forecasters' 2026 ratings (which have MI-10, OH-07, TX-15 more D) are not used.
- Tails: Safe-by-all-three races get 0.998-0.9995; 52 slots have q < 0.01 where Kalshi > 0.05.
- Same-party contests: poll weight hand-set; scandal-type information only through the dossier pool. CA-34 is the
  clearest case where Kalshi probably knows more.
- The dossier weights cannot be validated; the dossiers lean on Kalshi's price.
- 2022 backtest has no money; 2018 uses the Wayback snapshot (coverage to 9/30 or 10/17).
- Simulation: rho_n from 6 cycles; rho_s hand-set; LA/GA runoffs not modelled.

## 10. What I borrowed from whom

| from | what | where it enters | measured effect |
|---|---|---|---|
| kavi-b | pre-election FEC 2018 Wayback snapshot (`fec_2018_preelection.csv`) | 2018 backtest same-party split, principal, rated-independent tie-break | removes the 2018 look-ahead; together with the other pipeline fixes 2018 0.0796 -> 0.0787 (v1 config) |
| kavi-b | top-two primary history 2012-2026 (`primaries_top_two.csv`) | fitted primary weight of the same-party model; backtest splits | replaces a hand-set 0.04/pt with a fitted 0.085/pt; with the no-primary fit, recovers the 2022 money loss (v1 config 2022 0.0883 -> 0.0852) |
| kavi-b | CA 2026 sub-page polls (`ca_polls_2026.csv`) | same-party poll term | CA-04 Thompson/Jones, CA-07, CA-34 poll shares now include sub-page polls |
| cross investigation | out-party midterm conditioning; dossier use; fix list (UNK parties, RedBox tag, 60-day window) | op feature; late_info.py; window tested | op: LOYO 0.0704 -> 0.0666; 2026 House +2.4 D seats |
| kavi-c (idea) | conditioning on midterm (kavi-c uses env x midterm in a vote-share regression) | I used a different, symmetric out-party indicator in a classifier | - |
| research dossiers | 85 race estimates (q_dog_mean) | late-information pool, 2026 only | House E[D] +0.7 |

Not borrowed (to keep the side diverse): kavi-f's market-as-prior, kavi-e's national-environment swing, kavi-a's
vote-share regression, kavi-b's cycle random effect.

## 11. v1 -> v2 changes and their measured effect

Scorecard log loss, all / contested (538: 2018 0.1002/0.3778, 2022 0.0952/0.4375):

| step | 2018 | 2022 | 2026 House E[D] |
|---|---|---|---|
| v1 committed | 0.0796 / 0.2931 | 0.0849 / 0.3867 | 231.6 |
| + pipeline fixes (pre-election money 2018, no money 2022, WA-09/GA-14 FEC matching, money-free viability, fitted same-party primary weight, KY-03 match), v1 config | 0.0787 / 0.2937 | 0.0852 / 0.3961 | 231.6 |
| + logistic only (LOYO-preferred), no op | 0.0808 / 0.3016 | 0.0820 / 0.3809 | 231.8 |
| + midterm out-party term = **final_v2** | **0.0771 / 0.2877** | **0.1008 / 0.4701** | 234.2 |
| (blend + op, not chosen) | 0.0752 / 0.2803 | 0.0978 / 0.4553 | - |
| + dossier late-information pool (2026 only) | - | - | **234.9** |

Other v2 changes: midterm OOF table and logit permutation importance in analysis.py; disagreement reasons cite the
dossier adjustment; `q_model` column; variants in `out/variants/`.

## 12. How to reproduce

```
cd models/kavi-d
bash src/models/kavi-d/run_all.sh
```

Python = `python`, cwd `src/models/kavi-d` unless noted:

1. `fetch_ca_primary.py` -> `data/extra/ca_primary_2026.csv`
2. `history.py` -> `out/cache/train_features_w60.csv` (`KAVID_POLL_SPARSE_DAYS=150` builds the sparse-window table)
3. `sameparty.py` -> `out/cache/sameparty_train.csv` and printed fits
4. `backtest.py [--config C --outdir D]` -> `out/backtest_{2018,2022}.csv`; variants in `out/variants/<C>/`
5. (repo root) `src/scorecard.py --pred 2018=... --pred 2022=... --name kavi-d --json out/scorecard.json`
6. `predict_2026.py [--no-late-info]` -> `out/pred_2026.csv`, `out/disagreements_2026.csv`, `out/late_info_2026.csv`
7. `simulate.py --n 20000` -> `out/sims_summary.json`
8. `analysis.py` -> `out/analysis_{calibration,upsets,upset_logit,importance,midterm}.csv`
9. `experiments_v2.py <configs>` -> `out/experiments_v2.csv` (LOYO pre-2018 + rolling)
10. `check_outputs.py` -> invariants (470 races, 951 Kalshi slots, q sums to 1, no UNK party, >= 10k sims)

Deterministic: GBM `random_state=0`, simulation seed 20261103.
