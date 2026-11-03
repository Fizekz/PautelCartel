# kavi-c v3: polls-first Bayesian aggregator (METHODS)

Branch `kavi/kavi-c-1101`, code in `src/models/kavi-c/`, outputs in `out/` (v2 outputs in `out/v2/`, v1 in
`out/v1/`). This v3 section describes the whole model; the v2 METHODS below it is kept as disclosed, and section
numbers like "v2 §3" point there for tables that did not change. Every number is out of sample unless labelled
otherwise, and the code reproduces it (`bash src/models/kavi-c/run_all.sh`, about 15 s).

One line: a fundamentals prior on side A's two-party share (lean, ratings, incumbency, money, previous result,
generic ballot, an out-party midterm shift in competitive seats), updated by a house-effect-adjusted poll average with
a learned weight, turned into a win probability by a fitted national + local Student-t error model, then **(v3)**
log-linearly pooled with a ratings-only logistic (weight 0.25). Same-party top-two finals use **(v3)** a fitted June-
primary model. For 2026 only, a small fixed blend with the Nov 1 research dossiers.

## v3.1 What changed from v2

| # | change | source | how chosen | 2018 all / contested | 2022 all / contested |
|---|---|---|---|---|---|
| | v2 | | | 0.0810 / 0.3078 | 0.0956 / 0.4465 |
| 1 | 2022 backtest receipts from the June 30 2022 pre-election FEC cut (v2: no money in 2022) | kavi-a (`fetch_fec_preelection.py`, Wayback `webl22.zip` of 2022-08-03) | data availability, not scores | 0.0810 / 0.3078 | 0.0912 / 0.4051 |
| 2 | log-linear pool with a ratings-only logistic, w = 0.25 | v2 `baselines.py`; pooling as in the audit ensembles (K016) | pre-2018 LOYO | 0.0813 / 0.3091 | 0.0884 / 0.3950 |
| 3 | same-party finals from the June top-two primary (fitted logit); 2026 same-party polls from Pau's poll layer | kavi-b / kavi-e (primary data and model idea), pau/model-1101 (polls) | fitted on training years; no score-based choice | **0.0818 / 0.3091** | **0.0878 / 0.3924** |

Rejected in v3: `env x midterm x comp` (pre-2018 LOYO 0.0755 replacing mid_out x comp, 0.0745 = no change when
added); ratings logistic on [rating, env] (tie 0.0729, env coefficient ~0 in share units; with env in margin points
0.0733); epsilon tail mix (LOYO prefers 0).

## v3.2 Data

Everything in v2 §1, plus three new files in `data/kavi-c/` (see its README.md):

| file | source | used for |
|---|---|---|
| fec_2022_preelection.csv | 2022 rows of kavi-a's `data/kavi-a/fec_preelection.csv`: Wayback capture 2022-08-03 of FEC `webl22.zip` (current House and Senate campaigns), coverage end mostly 2022-06-30; 2,670 candidates | 2022 **test-year** receipts only (`backtest.PRE_CUT_2022`). Matched by cand_id, else (race, surname), else 0. |
| primaries_top_two.csv | kavi-b's parse of the June CA/WA top-two primaries (Wikipedia election pages), copied unchanged from kavi-e's branch; 2012-2026, 2,418 rows | same-party model (training pairs 2012-2024; 2026 primary shares). Primary vote shares are public in June, before every general. Write-in rows dropped. Not affected by the P016 trap: no article-existence feature. |
| pau_polls_2026_same_party.csv | 14 rows of Pau's disclosed `data/features/polls_2026.csv` (origin/pau/model-1101) for the 9 same-party finals | 2026 same-party poll averages, added to my own rows when (pollster key, end date) is new |

FEC timing, as used: training years 2010-16 and 2020-24 full-cycle totals (through Dec 31; mild look-ahead in
training only); 2018 the election-day snapshot (kavi-b) in training and test; 2022 test year the June 30 cut; 2026
receipts through 2026-10-14.

## v3.3 Features

Unchanged from v2 §2. Side A is the Democrat, or the independent in NE/MT/ID/SD-SEN and AK-AL; CA-06 has Kiley (I) as
side B. Prior design (13 columns): const, senate, lean, lean x senate, clip(rating, ±2), safe, rating x senate, inc,
money = clip(ln((A+25k)/(B+25k)), ±3), prev_rel, env (generic D two-party share - 0.5), env x midterm, mid_out x comp
(midterm x out_d x clip(1 - |rating|/2, 0, 1)).

Pool feature: the mean core-3 rating (Cook, Inside, Sabato; -3 Safe D .. +3 Safe R), unclipped.

Same-party features: primary share gap (A - B)/10 points, incumbency difference (FEC I flag in backtests,
`candidates_2026.is_incumbent` in 2026).

## v3.4 Model

1. **Fundamentals prior.** WLS of (share - 0.5) on the 13 columns; Safe races weight 0.3; ridge 1e-3 n. 2026 fit
   (2010-2024): const 0.0025, senate -0.0034, lean 0.406, lean x sen -0.013, rating -0.0158, safe -0.0374,
   rating x sen -0.0036, inc 0.0002, money 0.0158, prev_rel 0.047, env 0.055, env x midterm 0.059,
   mid_out x comp 0.0102.
2. **Polls.** Weight = recency (half-life 21 d, window 75 d) x sqrt(n/600) (n clipped 100-3000) x population
   (LV 1, V 0.9, RV 0.8, A 0.5) x 0.6 if partisan x pollster track record (earlier years only) x 1/k for a
   pollster's k-th poll. Within-cycle house effects (3 pseudo-polls) subtracted; no sponsor offset. RCV races keep a
   poll's final two-candidate round.
3. **Poll update.** mu = prior + w (poll - prior), w = neff / (neff + k): 2026 k_house 0.71, k_sen 0.61 (LOYO).
4. **Error model and model q.** National sd 0.0141 share; local scale sqrt(0.0168^2 + exp(gamma . [1, senate,
   log1p(neff), |rating|, safe])^2), gamma [-4.30, 0.10, -0.96, 0.18, 0.47]; q_model = T_20((mu - 0.5)/sd) with
   sd = scale x sqrt(local^2 + nat^2), scale House 0.906, Senate 0.803.
5. **(v3) Ratings pool** (`pool.py`). q_rating = logistic(a + b x rating), sklearn L2 C = 10, fitted on the same
   training races as the model (2026: a = -0.271, b = -2.252, so a toss-up gets 0.433 for side A). Pooled
   logit q = 0.75 logit q_model + 0.25 logit q_rating (q clipped to [1e-4, 1 - 1e-4] first). Applied to every rated
   race in the D-vs-R path (backtests: all non-special races; 2026: not the five independent-side races, CA-06,
   the 9 same-party finals or the 9 no-Republican seats). mu is then reset to 0.5 + sd t_20^-1(q) so the simulation
   reproduces the pooled q. 2026: 446 races pooled, mean |dq| 0.007; largest moves ME-02 (D 0.57 -> 0.41), MO-02
   (0.35 -> 0.22), SC-01 (0.64 -> 0.51), KY-06, MT-01, MN-01, CO-05, KS-SEN (0.77 -> 0.67) toward the raters, and
   TX-23, AZ-02, NV-02 up from tiny D q's.
6. **(v3) Same-party finals** (`same_party.py`). logit P(A beats B) = b_pct gap + b_inc inc_diff, no intercept,
   each training pair entered in both orders, L2 C = 10. Training pairs: CA/WA same-party generals in the training
   years whose top two primary finishers are the two finalists (2018 test 25 pairs: b_pct 0.89, b_inc 0.01; 2022
   test 37 pairs: 1.05, 0.00; 2026 48 pairs: 0.97, 0.80; leader won 85%). No money term (training receipts are
   full-cycle; the primary already measures strength).
   - Backtests: used when both finalists have primary shares; a head-to-head poll average (as in v2) overrides it.
     Same-surname finalists (WA-09 2018) are told apart by first name.
   - 2026: P0 from the fitted logit sets the conjugate-normal prior mean, prior share = 0.5 + 0.07 t_20^-1(P0), sd
     0.07; polls (mine plus Pau's rows) enter with sd^2 = 0.04^2 + 0.035^2/neff. Results: CA-04 Thompson 0.52
     (primary 41.0-22.2, two Jones-campaign polls Jones +10/+13), CA-07 Matsui 0.58, CA-11 Wiener 0.84, CA-12 Simon
     1.00, CA-14 Wahab 0.95, CA-29 Rivas 0.97, CA-34 Gomez 0.51 after the dossier (primary 46.0-30.7 vs three
     Gonzales-Torres-leading polls), CA-37 Kamlager-Dove 0.99, CA-40 Calvert 0.63 (primary 34.9-20.6; Kim's
     sponsored polls).
7. **2026-only structure (unchanged from v2 §3.5-3.6).** Independent fade (-2 pts on the poll part, +2.5 pts sd) in
   the 5 independent races; no-Republican seats D 0.998; minor/withdrawn candidates 0.001; Louisiana and AK-SEN by
   party; dossier blend logit q_fav' = 0.75 logit q + 0.25 logit q_dossier (0.15 if the skeptic weakened it), 85 races,
   applied after the pool.

## v3.5 Calibration

The error model is fitted, not recalibrated afterwards; the pool is the only layer on top. Pooled q, held-out
contested trainable races (favourite's q vs hit rate):

| q bin | pre-2018 LOYO n | q | hit | forward 2018-24 n | q | hit |
|---|---|---|---|---|---|---|
| 0.5-0.6 | 30 | 0.550 | 0.733 | 49 | 0.551 | 0.571 |
| 0.6-0.7 | 39 | 0.656 | 0.641 | 42 | 0.644 | 0.643 |
| 0.7-0.8 | 44 | 0.750 | 0.750 | 43 | 0.748 | 0.698 |
| 0.8-0.9 | 52 | 0.850 | 0.865 | 50 | 0.857 | 0.840 |
| 0.9-0.95 | 39 | 0.925 | 0.949 | 48 | 0.930 | 0.917 |
| 0.95-0.99 | 99 | 0.974 | 0.960 | 117 | 0.974 | 0.983 |
| 0.99-0.999 | 375 | 0.997 | 0.997 | 329 | 0.997 | 0.994 |
| >0.999 | 976 | 0.9997 | 1.000 | 1040 | 0.9997 | 1.000 |

(forward rows use full-cycle 2022 money; diagnostic only.)

## v3.6 Simulation

Unchanged (`simulate.py`, 20,000 draws, seed 20261103): side-A share = mu + sd_nat t_20 + sqrt(0.33) sd_loc N(0,1)
shared within a state + sqrt(0.67) sd_loc t_20; same-party and one-party races independent with their q. mu already
carries the pool and the dossier blend, so simulated marginals match q within 0.009.

2026: House D mean 234.9, median 235, 80% 221-248, 90% 217-252, P(D control) 0.944. Senate D caucus median 52
(80% 49-55), P(D >= 51) 0.743 (0.752 counting the new independents), P(both) 0.742.

## v3.7 Validation and selection discipline

- Backtests: 2018 fitted on 2010-2016, 2022 on 2010-2020; the pool's ratings logistic and the same-party model are
  fitted on the same training years. Test-year inputs are pre-election only: final ratings, polls 0-75 days out,
  final generic average, previous-election lean, the 2018 election-day or the 2022 June 30 FEC cut, June primaries.
- Selection metric (pre-registered in v2): mean held-out contested log loss over 2010/12/14/16, each fitted on the
  other three pre-2018 years. The pool weight (0.25) and the rejections were chosen on it (`pool_experiment.py`,
  `experiments.py`, `out/pool_experiment_v3_rating.csv`, `out/experiments_v3.csv`).
- **Choices made with 2018/2022 scores in view (selection optimism):**
  - The [rating, env] pool variant was scored on the backtests before it was rejected (it tied on pre-2018 LOYO);
    its margin-points version was rejected on pre-2018 LOYO, with the forward 2018-24 numbers printed in the same run.
  - The same-party model was kept although it made 2018 worse by 0.0005 and 2022 better by 0.0006; I had seen both.
  - Borrowing kavi-a's 2022 FEC cut and the pool idea was prompted by REPORT.md, whose leaderboard shows kavi-a's 2022
    and the pools' scores; the data fix itself is a leak-free input, not a tuned parameter.
  - Inherited from v1/v2: env x midterm (v1 test-set choice) and the out-party family of mid_out x comp (soft).
- Scorecard (`src/scorecard.py` on shared/scorecard-1101, verbatim):
```
kavi-c 2018: n=470 logloss 0.0818 (538 0.1002) brier 0.0440 (538 0.0578) acc 0.972 score top 46.42 ev 45.71 (chalk 42.13) against-538 10; dropped 0
       2018 contested (538 fav<0.95): n=120 logloss 0.3091 (538 0.3778) brier 0.1712 (538 0.2257) acc 0.892 (538 0.875)
kavi-c 2022: n=470 logloss 0.0878 (538 0.0952) brier 0.0561 (538 0.0576) acc 0.955 score top 30.47 ev 43.18 (chalk 30.43) against-538 14; dropped 0
       2022 contested (538 fav<0.95): n=90 logloss 0.3924 (538 0.4375) brier 0.2666 (538 0.2787) acc 0.778 (538 0.822)
```
- v2 -> v3: 2018 0.0810 -> 0.0818 (contested 0.3078 -> 0.3091), 2022 0.0956 -> 0.0878 (0.4465 -> 0.3924). v3 is
  better overall and is the final; the small 2018 loss is the pool (0.0003) and the primary split (0.0005).

## v3.8 The 2026 picks that go against the Kalshi favourite (top-q pick)

| race | our pick (q, Kalshi p) | Kalshi favourite (p) | why |
|---|---|---|---|
| WI-01 | Steil R (0.78, 0.50) | Berman D (0.50) | Lean R x3, R incumbent, R money edge; thin polls |
| CA-40 | Calvert R (0.63, 0.48) | Kim R (0.52) | primary 34.9-20.6; Kim's polls are sponsored and thin (neff 0.15) |
| CO-05 | Killin D (0.60, 0.445) | Crank R (0.555) | toss-up, D money, polls 51.5% D, out-party shift; pool and dossier pull it toward 0.5 |
| FL-22 | Askar R (0.54, 0.285) | Dandiya D (0.715) | new seat, R+4.4 lean, R money; one weak poll |
| MI-10 | Bouchard R (0.54, 0.24) | Hines D (0.76) | Lean R +2.6; the only polls lean R |
| SC-01 | Lacore D (0.51, 0.49) | Honeycutt R (0.51) | coin flip; the pool took it from 0.61 |
| CA-34 | Gomez D (0.51, 0.18) | Gonzales-Torres D (0.82) | incumbent led the primary 46.0-30.7 (P0 0.09 for the challenger) against three challenger-leading polls (two sponsored); dossier pulls toward the challenger |

## v3.9 Known weaknesses

- **The pool leans R in toss-ups.** The ratings logistic gives a toss-up 0.43 for the Democrat (intercept -0.27), fitted on
  2010-2024, a mix of R-wave and neutral years. In a D-leaning 2026 it pulls toss-ups toward R and costs about 2.0
  House seats (sum of D q 237.0 without the pool vs 235.0 with it, dossier included). The cross-model investigation found the Kavi
  models already ~5 seats too R in out-party waves; this change goes the other way. The weight 0.25 is the LOYO
  choice. An env-aware ratings logistic (env in margin points) did worse on pre-2018 LOYO (0.0733 vs 0.0728), so
  the bias is known and left in.
- 2018 is slightly worse than v2. Within noise for 120 contested races, but it is the wave-year test.
- 2022 money is a June cut while training money is full-cycle: the ratio is scale-free to first order, but early
  money is noisier, so the money coefficient may overstate its weight in the 2022 test.
- Same-party model: 48 pairs, no crossover-voting or self-funding term (CA-04 Jones's $15M self-funding enters only
  through polls). The 2026 prior sd (0.07) is a v1 judgement.
- Same-party polls from Pau's file are mostly sponsored internals; I down-weight them (0.6) but do not shift them.
- Everything in v2 §8 still holds (ratings dependence, out-party symmetry assumption, longshot tails, poll layer can
  hurt, dossier weight is a judgement, heuristics for independents / LA / AK RCV).

## v3.10 Reproduce and files

`bash src/models/kavi-c/run_all.sh` writes out/backtest_{2018,2022}.csv, out/scorecard.json, out/pred_2026.csv
(+ disagreements, race detail with q_unpooled / q_rating, dossier log, model diag), out/sims_summary.json and
out/baselines.csv. v3 diagnostics: `pool_experiment.py rating` (pool weight), `ablation_v3.py` (v2 -> v3 steps,
`out/ablation_v3.csv`), `experiments.py --forward --out experiments_v3.csv`. New files: `pool.py`, `same_party.py`,
`pool_experiment.py`, `ablation_v3.py`; changed: `hist_features.py` (2022 cut), `backtest.py`, `predict_2026.py`,
`special.py`, `model.py` / `experiments.py` (env x mid x comp option, off).

## v3.11 Borrowed in v3

- **kavi-a**: the 2022 pre-election FEC cut (Wayback webl22.zip, 2022-08-03) and its file.
- **kavi-b / kavi-e**: the CA/WA top-two primary file (kavi-b's parse, via kavi-e's branch) and the idea of a fitted
  within-party model on primary share and incumbency; my fit and form (no money, symmetric, training years only).
- **pau-a / pau-b**: the primary-margin same-party idea (also theirs), the GB x midterm idea behind the rejected
  `env x mid x comp`, and Pau's 2026 same-party poll rows (origin/pau/model-1101).
- **Kavi audit ensembles (K016)** and my own v2 `baselines.py`: the case for pooling with a ratings view.

---

# (v2 METHODS, as disclosed) kavi-c v2: polls-first Bayesian aggregator

Branch `kavi/kavi-c-1101`, code in `src/models/kavi-c/`, outputs in `out/` (v1 outputs kept in `out/v1/`).
Every number below comes from these files, and the code reproduces it from `data/clean/*.csv`,
`data/benchmark/*.csv`, `data/kalshi/races_2026-11-01T074812Z.csv` and the two files in `data/kavi-c/`.
Unless a number is labelled otherwise, it is out of sample.

One line: a fundamentals prior on side A's two-party share (lean, ratings, incumbency, money, previous result,
generic ballot, and an out-party midterm shift in competitive seats), updated by a house-effect-adjusted poll
average with a learned weight, turned into win probabilities by a fitted national + local Student-t error model;
for 2026 a small, fixed late-information blend with the Nov 1 research dossiers.

## 1. Data used

| file | source (see the matching .md) | years | n used | cleaning / how used |
|---|---|---|---|---|
| results_house.csv, results_senate.csv | MIT MEDSL (v15 House, v8 Senate) | 2010-2024 | 3,761 race-years; 3,372 trainable | Target = `dem_share_2p`. Trainable = both majors on the ballot, not same-party, no major independent, not ranked-choice (MEDSL's AK RCV counts are first-round party sums). In a target year, results supply only who was on the ballot (nominee names, one-party seats), public before election day. |
| lean_hist.csv / lean_2026.csv | Downballot / DKE sheets, MIT president | 2010-2026 | all races | lean = previous presidential D two-party share minus the national share (2026: pres24 on the **2026 lines**). CA 2012 and FL 2016 gaps filled with -PVI/100 (80 rows). |
| ratings_hist.csv / ratings_2026.csv | Wikipedia (Cook, Sabato, Inside) | 2010-2026 | all races | mean of the three numeric ratings (-3 Safe D .. +3 Safe R). The `other_*` outlets were tested as a feature and rejected (section 7). |
| fec_hist.csv / fec_race_2026.csv / fec_2026.csv | FEC bulk weball/cn/cm | 2010-2026 | nominees matched by surname (first name breaks same-surname ties), fallback to the party's top fundraiser | money = ln((A+25k)/(B+25k)), clipped to +/-3. **2018 receipts are replaced by the election-day snapshot below.** 2010-2016 and 2020-2024 remain full-cycle totals (through Dec 31). 2026 runs to 2026-10-14. Incumbency: FEC I flag OR "nominee = last cycle's winner in an unchanged district"; 2026 uses `candidates_2026.is_incumbent`. |
| **data/kavi-c/fec_2018_preelection.csv** | Wayback capture 2018-11-07 08:07 UTC of FEC weball18.zip (file dated election day); **borrowed from kavi-b** | 2018 | 3,656 candidates | matched by cand_id, else race + surname, else 0. See `data/kavi-c/README.md`. |
| polls_hist.csv | 538 archive | 2010-2024 | general stage, not hypothetical, both surnames match the MIT nominees, 0-75 days before election day | Specials re-keyed to `-SEN-S`. **v2: in RCV races (AK 2022+, ME 2018+) only a poll's final two-candidate round is kept.** |
| polls_2026.csv | Wikipedia + VoteHub | 2026 | 1,116 usable rows -> 126 races with an average | "usable" rule from polls_2026.md. Independent races use `top_other_pct`, same-party finals `cand_detail`. **v2: all RedBox Research rows tagged D-sponsored** (16 of 18 were; MI-04 and MI-10 were blank). |
| national_hist.csv / national_2026.csv | 538 generic-ballot averages; Wikipedia aggregator table | 2010-2026 | 8 years + 2026 | env = generic D/(D+R) - 0.5; 2026 = mean of the five 10-30 aggregates (D 51.68 / R 43.12, env +0.0451). President's party gives `out_d` (+1 when D is the out-party). |
| candidates_2026.csv | Wikipedia | 2026 | 470 races | incumbency, independents, uncontested seats, same-party finals |
| **data/kavi-c/dossiers_summary_2026-11-01.csv** | frozen copy of `repo/data/dossiers/summary.csv` | 2026 | 85 races | researcher/skeptic probability for the Kalshi underdog; 2026-only late-information blend (section 3.6). |
| fte_final_{2018,2022}.csv | 538 | 2018, 2022 | 470 races each | **benchmark only, never an input** (candidate list for the backtest files) |
| Kalshi races snapshot 07:48Z | Kalshi | 2026 | 470 races, 951 candidate prices | p_kalshi = mid / sum of mids within the race. **Never a model input**, joined at the end. (v1 METHODS said 987: wrong, it is 951.) |

## 2. Features and why

Every race is reduced to **side A's share of the A+B two-way vote**. A is the Democrat, except where the real
challenger to the Republican is an independent (NE/MT/ID/SD-SEN, AK-AL: A = the independent). In CA-06, B = Kiley (I).

| feature | definition | why |
|---|---|---|
| lean | previous pres D 2-party share minus national (statewide for Senate) | base partisan anchor (coef 0.41; ratings carry the rest) |
| lean x senate | | statewide races swing less with lean |
| rating | clip(mean rating, -2, 2) | strongest prior signal for unpolled House seats |
| safe | sign(rating) if \|rating\| > 2.5 | Safe seats are far from 50% |
| rating x senate | | |
| inc | +1 A incumbent, -1 B incumbent, 0 open | candidate quality; ~0 once ratings and money are in |
| money | clipped log receipts ratio | 0.016 share per log unit |
| prev_rel | last House D share minus that year's national House share, unchanged districts only | district-specific residual |
| env | generic-ballot two-party D share - 0.5 | national environment beyond the ratings |
| env x midterm | | generic ballot carries more beyond-ratings signal in midterms (v1 test-set choice; pre-2018 LOYO is neutral on it) |
| **mid_out x comp (v2)** | midterm x out_d x comp, comp = clip(1 - \|rating\|/2, 0, 1) (1 at a toss-up, 0.5 at Lean, 0 at Likely) | in midterms the party out of the White House over-performs the ratings in competitive seats. Fitted: +0.0102 share (about 2 points of margin) toward the out-party in a toss-up. |

Polls: per poll the A two-way share; one row per (race, pollster, end date), best population.
- **Weight** = recency (half-life 21 days, window 75 days) x sqrt(n/600) (n clipped 100-3000) x population (LV 1, V 0.9, RV 0.8, A 0.5) x partisan sponsor 0.6 x pollster track record x 1/k for a pollster's k-th most recent poll in that race.
- **Track record**: each pollster's mean squared error vs results in *earlier* years only, the year's average error removed, shrunk with 8 pseudo-polls; multiplier clipped 0.5-1.6.
- **House effects** estimated *within the cycle* from deviations from the race average, shrunk with 3 pseudo-polls, subtracted. **v2: no D/R sponsor offset** (partisan polls are down-weighted to 0.6 but not shifted); chosen on pre-2018 LOYO (section 4).
- `poll_neff` = sum of weights (one fresh 600-person LV poll = 1).

## 3. Model and parameters

1. **Fundamentals prior.** WLS of (share - 0.5) on the 13 columns above; Safe races weight 0.3; ridge 1e-3 n on the slopes. 2026 fit (2010-2024): const 0.0025, senate -0.0034, lean 0.406, lean x sen -0.013, rating -0.0158, safe -0.0374, rating x sen -0.0036, inc 0.0002, money 0.0158, prev_rel 0.047, env 0.055, env x midterm 0.059, **mid_out x comp 0.0102**.
2. **Poll update.** mu = prior + w (poll - prior), w = neff / (neff + k_chamber); k minimises squared error with leave-one-year-out (LOYO) priors on the training years. 2026: k_house 0.71, k_sen 0.61.
3. **Error model.** LOYO residuals r = share - mu.
   - National sd = RMS of the year means of r over competitive races (\|rating\| <= 2), floor 1 point: 0.0141 share (~2.8 points of margin).
   - Local scale = sqrt(floor^2 + exp(gamma . [1, senate, log1p(neff), \|rating\|, safe])^2), Student-t (df 5) MLE on r minus its year mean: floor 0.0168, gamma = [-4.30, 0.10, -0.96, 0.18, 0.47].
   - State share of the local variance = 0.33 (simulation only).
4. **Win probability.** q_A = T_nu((mu - 0.5) / sd), sd = scale_chamber x sqrt(local^2 + nat^2); nu and per-chamber scales tuned on LOYO win log loss (nu 20; raw House 0.90, Senate 0.65), shrunk toward 1 as 1 + (s - 1) n / (n + 200): House 0.906, Senate 0.803.
5. **2026-only structure** (`predict_2026.py`):
   - *Independent fade*: past independent challengers ran behind their final two-way poll average (KS-SEN 2014 -6.5, UT-SEN 2022 -2.7, NE-SEN 2024 -2.6 points); half (2 points) is taken off the poll part of mu in the 5 independent races, +2.5 points sd in quadrature.
   - *Same-party finals* (8 CA D-v-D, CA-40 R-v-R): conjugate-normal on A's share; prior 0.5 + 0.04 (inc_A - inc_B) + 0.025 x clipped log receipts ratio, sd 0.07; polls sd^2 = 0.04^2 + 0.035^2/neff. **v2: these rows carry the candidates' real party (DEM; CA-40 REP), not Kalshi's UNK.** Backtests use `special.split_within_party` (incumbency/money softmax).
   - *Uncontested / no Republican*: Democrat 0.998. Minor and withdrawn candidates 0.001 before renormalising.
   - *Louisiana House* (jungle) and *AK-SEN* priced by party, as Kalshi does.
6. **Late-information blend (2026 only, `dossiers.py`).** For each of the 85 dossier races:
   logit q_fav' = (1 - W) logit q_model + W logit (1 - q_dog_mean), W = 0.25, or 0.15 when the skeptic "weakened"/"refuted" the dossier's view; other candidates keep their relative shares. The partisan race's mean is then moved so the simulation reproduces q' (`sync_side_a`). How much it moves things: House E[D] 236.6 -> 237.0; mean |q' - q| on the 85 races 0.019; largest moves CA-34 (Gonzales-Torres 0.43 -> 0.52), TX-35 (Garcia 0.44 -> 0.50), CO-05 (Crank 0.28 -> 0.34), KY-06, MT-01, AZ-02, NV-02, KS-SEN (0.04-0.05). W is hand-set, not fitted: dossiers have no historical analogue and they saw Kalshi's price, so a larger W would mostly import the market. Model-only q is kept in the `q_model` column of `pred_2026.csv`; the per-race log is `out/dossier_adjustments_2026.csv`.
7. **Simulation** (`simulate.py`, 20,000 draws, seed 20261103). Side-A share = mu + sd_nat t_20 (all races) + sqrt(0.33) sd_loc N(0,1) (shared within a state, House and Senate) + sqrt(0.67) sd_loc t_20. Same-party and one-party races are independent draws with their q. Simulated marginals match q within 0.009. Senate holdovers 34 D-caucus and 31 R; D control needs 51 (VP is a Republican).

## 4. Training and validation

- **2018 backtest**: fitted only on 2010-2016 (1,654 trainable races). **2022 backtest**: fitted only on 2010-2020. The target year's inputs are its own pre-election data: final ratings, polls 0-75 days out, final generic average, lean from the previous presidential election, nominee receipts, previous House result. Pollster track records use only earlier years. k, error scales and nu are nested LOYO inside the training years.
- **FEC look-ahead (audit L1), v2 treatment.** 2018 receipts come from the election-day Wayback snapshot, in the 2018 test and in every later training set. No pre-election cut of 2022 exists, so **the 2022 backtest uses no receipts at all**: money is removed from the training and test design (refit without it), and same-party splits use incumbency only. The leaky full-cycle 2022 variant is reported but not used (0.0904).
- **Design selection (v2).** Pre-registered metric: mean over held-out 2010/2012/2014/2016 of the contested-race win log loss, each predicted from the other three pre-2018 years (`experiments.py`, `out/experiments_v2.csv`). Forward tests 2018-2024 are printed for information. Choices made: drop the sponsor offset (0.0765 -> 0.0755), add mid_out x comp (-> 0.0745). Rejected: symmetric mid_out (0.0773 alone), other_dev (0.0771 alone, 0.0746 with v2), dropping house effects (0.0808). Kept env x midterm (tie, 0.0745 vs 0.0746 without).
- **Selection caveats, recorded in LOG.md:** (a) env x midterm was added in v1 after a 2018/2022 check. (b) Pre-2018 midterms both had a Democratic president, so pre-2018 LOYO identifies mid_out x comp only as "midterm competitive seats shift toward R". Its sign for 2018/2026 (toward D) comes from the out-party symmetry assumption. That assumption was prompted by the cross-model investigation, whose evidence includes 2018, so the choice of this term family is soft test-set selection; its coefficient is not.
- `baselines.py`: every year 2014-2024 from earlier years only. `calibration.py`: pooled calibration of the favourite, 2014-2024 forward. Both use full-cycle money outside 2018 (diagnostic only).
- 538's forecast is never an input.

## 5. Scorecard numbers

`src/scorecard.py` output (verbatim):
```
kavi-c 2018: n=470 logloss 0.0810 (538 0.1002) brier 0.0442 (538 0.0578) acc 0.970 score top 44.56 ev 48.87 (chalk 42.13) against-538 9; dropped 0
       2018 contested (538 fav<0.95): n=120 logloss 0.3078 (538 0.3778) brier 0.1731 (538 0.2257) acc 0.883 (538 0.875)
kavi-c 2022: n=470 logloss 0.0956 (538 0.0952) brier 0.0625 (538 0.0576) acc 0.953 score top 32.33 ev 42.52 (chalk 30.43) against-538 17; dropped 0
       2022 contested (538 fav<0.95): n=90 logloss 0.4465 (538 0.4375) brier 0.3005 (538 0.2787) acc 0.767 (538 0.822)
```

| | 2018 ll (538) | 2018 brier (538) | 2022 ll (538) | 2022 brier (538) |
|---|---|---|---|---|
| House (435) | 0.0773 (0.0938) | 0.0411 (0.0533) | 0.0935 (0.0914) | 0.0614 (0.0551) |
| Senate (35) | 0.1277 (0.1799) | 0.0828 (0.1137) | 0.1220 (0.1414) | 0.0766 (0.0877) |
| contested | 0.3078 (0.3778) | 0.1731 (0.2257) | 0.4465 (0.4375) | 0.3005 (0.2787) |

- 2018 beats 538 everywhere. 2022, leak-free and without money, ties 538 on all-race log loss and loses on the contested subset and on Brier. The Senate beats 538 in both years.
- Calibration of the favourite, forward 2014-2024 pooled, contested trainable races (`out/calibration_v2.csv`):

| q bin | n | mean q | hit rate |
|---|---|---|---|
| 0.5-0.6 | 65 | 0.549 | 0.615 |
| 0.6-0.7 | 69 | 0.644 | 0.667 |
| 0.7-0.8 | 54 | 0.742 | 0.704 |
| 0.8-0.9 | 64 | 0.854 | 0.844 |
| 0.9-0.95 | 76 | 0.929 | 0.921 |
| 0.95-0.99 | 165 | 0.976 | 0.982 |
| 0.99-0.999 | 471 | 0.997 | 0.998 |
| >0.999 | 1548 | 0.9998 | 0.9994 |

- Baselines (`out/baselines.csv`): mean 2014-2024 contested log loss model 0.0793, prior-only 0.0801, ratings-only logistic 0.0775. The model wins 2018 and 2022 vs the ratings logistic; it loses in presidential years.

## 6. The 2026 picks that go against the Kalshi favourite (top-q pick)

Six races out of 470 (`out/disagreements_2026.csv` lists these plus 54 same-favourite races with |q - p| > 0.10):

| race | our pick (q, Kalshi p) | Kalshi favourite (p) | reason |
|---|---|---|---|
| WI-01 | Steil R (0.73, 0.50) | Berman D (0.50, tie) | All three raters Lean R, R incumbent, R money edge 1.17 log. Polls 51.0% D, but neff only 0.36. The dossier (Berman 0.465) pulls it from 0.76 to 0.73. |
| CO-05 | Killin D (0.66, 0.445) | Crank R (0.555) | Toss-up rating, D money edge +0.66 log, polls 51.5% D (neff 0.87), out-party toss-up shift. The dossier (0.445) pulls it from 0.72. |
| SC-01 | Lacore D (0.61, 0.49) | Honeycutt R (0.51) | Open seat, D money +1.31 log, 4 polls averaging 51.2% D (neff 1.1). Rating leans R (+0.5). |
| ME-02 | Dunlap D (0.56, 0.44) | LePage R (0.56) | Polls 52.5% D (neff 0.79) against all-Lean-R ratings and R+4 lean. The dossier (0.475) keeps it close. |
| FL-22 | Askar R (0.54, 0.285) | Dandiya D (0.715) | New seat, R+4.4 lean, R money edge, toss-up; one weak poll (neff 0.11). The out-party shift moved it toward D from v1's 0.73 R, but not past even. |
| MI-10 | Bouchard R (0.51, 0.24) | Hines D (0.76) | Lean R+2.6, the only polls average 48.3% D (latest nonpartisan R+5), D money edge. A coin flip for us. |

- Dropped from v1's list: IA-SEN (now Hinson 0.53, same as Kalshi; the v1 sponsor offset had shifted R polls toward D), CA-34 (dossier blend 0.43 -> 0.52 Gonzales-Torres), MI-04, TX-35 (out-party shift).
- Large same-favourite gaps remain, all where Kalshi prices a D longshot in a Likely/Safe R seat (NC-03 Murphy q 0.99 vs p 0.555; TX-23 Herrera 0.92 vs 0.54; NC-13, NC-09, NJ-02). The investigation attributes most of these to thin-book longshot compression. We still price the Republican higher than its fair value of ~0.75 there: an underdog floor is not modelled.
- Where Kalshi is more D and history supports it (TX-15 Pulido 0.78 vs 0.89, WI-03 Cooke 0.78 vs 0.885, OH-07 Poindexter 0.76 vs 0.85), v2 moved toward the market (v1 0.74 / 0.74 / 0.65).

Seats: House D mean 237.0, median 237 (80% 223-250, 90% 218-254), P(D control) 0.956. Senate D-caucus median 53 (80% 49-55), P(D >= 51) 0.788, P(both chambers) 0.787. Kalshi's sum of race prices is 246 (238 excluding the 8 D-v-D seats' UNK labels). The investigation's fair value is ~238.

## 7. What was tried and failed / was rejected

- v1 items: Gaussian MLE for local error (replaced by t(5)); free Senate scale (now shrunk); raw-poll same-party probabilities (replaced); stacked logistic calibration on [z, rating] (no gain over 2016-2024).
- v2, on pre-2018 LOYO:
  - Symmetric out-party midterm shift `mid_out` applied to all seats: 0.0773 vs v1 0.0765. It hurts presidential years, because it moves the intercept.
  - Other outlets' ratings (`other_dev` = mean of the non-538 `other_*` columns minus the core-3 mean): 0.0771 alone, 0.0746 with v2. Rejected. The historical "other" set (RCP, Daily Kos, NYT, CQ...) is not the 2026 quantitative set (DDHQ, Silver Bulletin, Split Ticket, Economist, FiftyPlusOne), so the investigation's point about faster quantitative raters cannot be backtested here.
  - Dropping pollster house effects: 0.0808. It would help 2022 (forward 0.0890) but hurts pre-2018.
  - Dropping env x midterm with v2: 0.0746 (tie). Kept.
- Longer poll window / summer-poll term for thin races: cannot be tested pre-2018 (the 2010-16 archive has ~60 days). With a 21-day half-life, August polls get <5% weight anyway. Not done.
- Ablation of the v1 -> v2 steps on the test years: section 11.

## 8. Known weaknesses

- **2022 without money is only a tie with 538.** Money is a real signal (dropping it costs 0.005 in 2022). The strict number is the honest one.
- **Ratings dependence.** The prior leans on the core-3 raters. The out-party term recovers part of their wave-year lag (+2 seats in 2026). The faster quantitative outlets are not used.
- **Out-party symmetry assumption.** The 2026 sign of the competitive-seat shift rests on 2010/2014 (R out-party) mirrored to a Republican president. Only 2018 is a historical test of the D-out direction.
- **Likely/Safe R longshots** get ~0.5-3%. History (Likely R seats lost 7-18% in out-party waves) says a few points more, so a 2010-size wave would beat us in those seats in a correlated way.
- **The poll layer can hurt.** Polls hurt in 2020 and 2022 (prior-only was better). A shared 2026 poll bias moves all Senate q's together. National sd covers a typical year, not a 2020-sized miss.
- **Dossier blend weight is a judgement**, with no backtest. Its effect is small by construction (0.4 seats).
- **Heuristics with little data:**
  - Same-party finals (9 races).
  - Independents (5; the fade is from 3 past cases).
  - Louisiana jungle (party level).
  - AK RCV (Hafner's first choices go to Hill).
- **Training-year FEC**: 2010-16 and 2020-24 receipts are still full-cycle in training (the money coefficient is learned on them). That is a mild mismatch with the pre-election 2026 receipts. kavi-b measured a mean |difference| of 0.07 log units in 2018.

## 9. How to reproduce

From the worktree root (`models/kavi-c`):
```
bash src/models/kavi-c/run_all.sh
# = backtest.py -> out/backtest_{2018,2022}.csv, out/backtest_diag.json
#   src/scorecard.py --pred 2018=... --pred 2022=... --name kavi-c --json out/scorecard.json
#   predict_2026.py -> out/pred_2026.csv, disagreements_2026.csv, race_detail_2026.csv, model_2026_diag.json,
#                      dossier_adjustments_2026.csv
#   simulate.py --n 20000 -> out/sims_summary.json
#   baselines.py -> out/baselines.csv
# diagnostics: experiments.py --forward (out/experiments_v2.csv), ablation_v2.py (out/ablation_v2.csv),
#              calibration.py (out/calibration_v2.csv)
```
Python: `python` (pandas, numpy, scipy, scikit-learn). The run takes ~12 s; experiments take ~1 min. Everything is deterministic (fixed seed).

Files:
- `common.py`
- `hist_features.py` (features, FEC loader with the 2018 snapshot)
- `polls.py` (averaging, RCV final round)
- `dataset.py`
- `model.py`
- `special.py`
- `backtest.py`
- `features_2026.py`
- `predict_2026.py`
- `dossiers.py`
- `simulate.py`
- `baselines.py`
- `experiments.py`
- `ablation_v2.py`
- `calibration.py`

## 10. What I borrowed from whom

- **kavi-b**: the pre-election 2018 FEC snapshot (`data/kavi-c/fec_2018_preelection.csv`, byte-identical to kavi-b's), and the matching rule (cand_id, else race + surname). Also kavi-b's finding that weball22 has no capture between Apr 11 and Dec 28, 2022.
- **Referee audit (kavi-audit/kavi-c)**:
  - The first-name tie-break for same-surname FEC rows.
  - The final-round RCV poll filter, adapted from `rcv_fix.py`.
  - The L2 fix.
- **Cross-model investigation**:
  - The hypothesis of an out-party midterm effect in competitive seats. I implemented it as a fitted share term and selected it on pre-2018 LOYO; it was not hand-set.
  - The RedBox tag inconsistency and the UNK labels.
  - The suggestion to test the `other_*` ratings (tested, rejected).
- **Considered, not adopted**:
  - kavi-a's 50/50 logistic calibration layer: v1 tried a similar stacked layer, with no gain.
  - kavi-b's year random effect: my national t-error plays that role.
  - kavi-e's specials-based environment: my env stays the generic average, to keep the model distinct.
  - kavi-f's market-as-prior: Kalshi is never an input here.

## 11. v1 -> v2 changes and their measured effect

Cumulative steps, scored with the new scorecard (`out/ablation_v2.csv`). These numbers use the test years: they are reported, not used to choose.

| step | 2018 ll | 2018 contested | 2022 ll | 2022 contested |
|---|---|---|---|---|
| v1 as committed (661fcbd) | 0.0826 | 0.3112 | 0.0937 | 0.4356 |
| bug fixes (surname tie-break, RCV final round, L2) | 0.0818 | 0.3112 | 0.0941 | 0.4354 |
| + 2018 pre-election FEC snapshot | 0.0827 | 0.3145 | 0.0942 | 0.4361 |
| + 2022 money dropped (no pre-election cut) | 0.0827 | 0.3145 | 0.0992 | 0.4569 |
| + no sponsor offset (pre-2018 LOYO choice) | 0.0838 | 0.3184 | 0.0939 | 0.4397 |
| + mid_out x comp (pre-2018 LOYO choice) = **v2** | **0.0810** | **0.3078** | **0.0956** | **0.4465** |
| (v2 with leaky full-cycle 2022 money) | 0.0810 | 0.3078 | 0.0904 | 0.4232 |

2026 effects:

| change | effect |
|---|---|
| no sponsor offset | House E[D] +1.0. IA-SEN Turek 0.63 -> 0.49 (0.47 after the dossier blend). Senate P(D >= 51) 0.88 -> 0.79. |
| mid_out x comp | House E[D] +2.1 (234.6 -> 236.7). The 15 House toss-ups (\|mean rating\| <= 1/3) moved +0.10 toward D on average from v1 (all v2 changes together). |
| RedBox tags | MI-04 and MI-10 R-leaning RedBox polls down-weighted to 0.6. |
| UNK -> real party | Summing DEM rows of pred_2026 now gives 237.0; in v1 the same sum gave 225.4 because 8 D-v-D seats were UNK. |
| dossier blend | House E[D] +0.4. Mean \|dq\| 0.019 on 85 races. |

House E[D]: v1 233.4 -> v2 237.0.
