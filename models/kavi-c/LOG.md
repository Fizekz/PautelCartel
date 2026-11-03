# kavi-c LOG (polls-first Bayesian aggregator)

Running log, newest at the bottom. All scorecard numbers are out of sample (2018 fitted on 2010-2016, 2022 on
2010-2020), from `src/scorecard.py`.

## v0 (first full pipeline)
- Fundamentals prior: WLS of D two-party share on lean (prev pres, relative to nation), mean rating (clipped
  +/-2 plus a Safe flag), incumbency (FEC I flag on the nominee), money (ln receipts ratio, clipped), previous
  House result where lines are unchanged, generic-ballot environment. Poll update mu = prior + w (poll - prior),
  w = neff/(neff+k), k fitted on leave-one-year-out (LOYO) prior residuals. Error: national sd from year means +
  local sd (t-likelihood MLE on chamber, log1p(neff), |rating|, safe). Student-t win prob, df/scale tuned on LOYO.
- First run, contested races only: AK-AL 2022 counted as a "miss" because MEDSL shares are RCV first-round party
  sums. Fixed: RCV rows excluded from training (`trainable`), and evaluation uses winners.
- Scorecard v0:
  - 2018: logloss 0.0835 (538 0.1002), brier 0.0468 (538 0.0578), acc 0.966, score top 42.57 (chalk 42.13)
  - 2022: logloss 0.0945 (538 0.0952), brier 0.0622 (538 0.0576), acc 0.949, score top 30.69 (chalk 30.43)
  - House 2022 slightly worse than 538 (0.0927 vs 0.0914); Senate better both years.

## v1 (error model, special races, 2026 pipeline)
- Error model: added an irreducible local floor (in quadrature) and per-chamber scale; t-likelihood MLE for the
  local scale (Gaussian MLE was inflated by a few 10-point misses).
- Scale tuning on LOYO win log loss wanted Senate scale 0.55 (very tight). Out-of-sample it did help a little
  (Senate ll 2018 0.1316 vs 0.1392 at scale 1), but the Senate sample is ~250 races with few close ones, so
  the tuned scale is now shrunk to 1 with n0=200 pseudo-races (Senate ends ~0.83, House ~0.9).
- Hyperparameter ablations (all on the 2018/2022 backtests, so NOT used for selection, only reported):
  | variant | 2018 ll | 2022 ll |
  |---|---|---|
  | v1 baseline (house effects, hl 21d, window 75d) | 0.0837 | 0.0939 |
  | half-life 14d | 0.0824 | 0.0976 |
  | half-life 35d | 0.0853 | 0.0902 |
  | partisan weight 0.3 / 1.0 | 0.0846 / 0.0830 | 0.0939 / 0.0942 |
  | window 45d | 0.0835 | 0.0915 |
  | no house/sponsor adjustment | 0.0849 | **0.0825** |
  | drop money | 0.0909 | 0.0951 |
  | drop prev_rel | 0.0826 | 0.0935 |
  | drop env | 0.0857 | 0.0950 |
  | drop inc | 0.0841 | 0.0926 |
- Honest note on house effects: dropping pollster house effects and the D/R sponsor offset would make 2022 much
  better (0.0825), because in 2022 D-sponsored internals were right and the within-cycle offset (+3.2 pts for D
  sponsors) "corrected" them toward R. But on 2010-2016 (the only data available before 2018) the adjusted
  average was better in every year (competitive-race RMSE 0.0325 vs 0.0352 in 2010, etc.), and across all 8
  years it is better in 5. Choosing "none" because of 2022 would be test-set selection, so adjustments stay.
- Poll window / half-life: competitive-race poll RMSE is flat across 45-75 days and 10-60 day half-lives
  (0.0302-0.0314 averaged over 2010-2024), so the pre-registered 75d/21d stays.
- Money: the strongest single feature after lean/ratings (dropping it costs 0.007 in 2018). Historical FEC totals
  run to Dec 31, so they include ~2 months of post-election receipts: a known leak. The money coefficient is
  0.016 per log unit; a +0.05 log-ratio bias toward winners moves shares by <0.1 pt, so the gain is signal, not
  the leak. The no-money ablation (2018 0.0909, 2022 0.0951) is the strict-input reference.
- Special races in backtests (hand list, pre-election facts): ME/VT-SEN 2018 independents, CA-SEN 2018 D-v-D,
  UT-SEN 2022 McMullin, AK-SEN 2022 Murkowski-v-Tshibaka: A-vs-B two-way poll average with the Senate local sd.
- 2026: side A = independent in NE/MT/ID/SD-SEN and AK-AL (AK-AL adds Hafner's first choices to Hill, RCV);
  CA-06 is Pan (D) vs Kiley (I) with Kiley as side B; nine same-party CA finals use a small conjugate-normal
  model (incumbency/money prior, polls with a 4-point floor). First version used the backtest's
  incumbency logit and gave raw polls a near-certain verdict (CA-06 Pan 1.00 from one poll): replaced.
- Independent fade: every comparable past independent challenger ran behind the final two-way poll average
  (KS-SEN 2014 Orman -6.5, UT-SEN 2022 McMullin -2.7, NE-SEN 2024 Osborn -2.6 points). 2026 applies half
  (-2 pts on the poll part) and +2.5 pts sd for the five independent-challenger races.
- Scorecard v1: 2018 ll 0.0837 (538 0.1002), 2022 ll 0.0944 (538 0.0952).

## v2 (final for this round)
- Added env x midterm (generic ballot matters more beyond ratings in midterms): 2018 0.0837 -> 0.0826, 2022
  0.0944 -> 0.0937. Mild test-set selection (it was tried on the backtests), flagged in METHODS.
- Tried a stacked logistic calibration layer on [z, rating, z*senate]: 2016-2024 mean 0.0827 vs 0.0826 without.
  No gain, dropped.
- Baselines (`baselines.py`, each year from earlier years only, contested races): a ratings-only logistic beats
  the model on average over 2014-2024 (0.0775 vs 0.0798); the model wins the midterms (0.0853 vs 0.0895). Polls
  helped in 2014, 2016 and 2018 and hurt in 2020 and 2022 (prior-only beat full in those two). An honest limit of
  a polls-first approach.
- Scorecard v2 (final):
  - kavi-c 2018: n=470 logloss 0.0826 (538 0.1002) brier 0.0465 (538 0.0578) acc 0.966 score top 42.57 ev 45.44 (chalk 42.13) against-538 9
  - kavi-c 2022: n=470 logloss 0.0937 (538 0.0952) brier 0.0614 (538 0.0576) acc 0.949 score top 30.69 ev 33.84 (chalk 30.43) against-538 17
- 2026: 126 races with a poll average, env +0.045 (generic D+8.6).
  - Top pick differs from the Kalshi favourite in 9 races: IA-SEN, CA-34, CO-05, FL-22, MI-04, MI-10, SC-01,
    TX-35, WI-01. There are 64 more with |q-p| > 0.10.
  - House: D seats median 234 (80% 220-246), P(D control) 0.93. Kalshi's implied E[D] is 246, the raters' about 232.
  - Senate: D-caucus median 53 (80% 50-55), P(D >= 51) 0.88.

## v2 round 2 (2026-11-01, after referee audit, cross-model investigation and in-side disclosure)
- Recovered state: an interrupted v2 attempt had only copied v1 outputs to `out/v1/` (identical to commit 661fcbd),
  copied the new scorecard (contested block) and kavi-b's 2018 FEC snapshot into `data/kavi-c/`. Continued from it.
- Audit bugs fixed: (1) same-surname FEC collision in `special.fec_lookup` (WA-09 2018 Adam v Sarah Smith):
  first-name tie-break; (2) RCV polls: AK 2022+/ME 2018+ keep only a poll's final two-candidate round
  (`polls.rcv_final_only`), also for the special two-way averages; (3) L2: races with a major third candidate now use
  the model's pre-election qA, not the actual result; uncontested races use who is on the ballot. Combined effect with
  v1 settings: 2018 0.0826 -> 0.0818, 2022 0.0937 -> 0.0941 (contested 0.3112 / 0.4354).
- FEC look-ahead removed: 2018 receipts (training and test) now come from the election-day Wayback snapshot (borrowed
  from kavi-b); 2018 0.0818 -> 0.0827. No pre-election cut exists for 2022 (no capture Apr 11 - Dec 28, 2022), so the
  2022 backtest drops money from every input (refit without it, same-party splits by incumbency only):
  2022 0.0942 -> 0.0992. The leaky 2022 variant of the final model is 0.0904 (reported, not used).
- Design selection, pre-registered: mean pre-2018 LOYO win log loss on contested races (`experiments.py`; fit on 3 of
  2010/12/14/16, predict the 4th). Forward 2018-2024 printed for information only.
  | config | pre-2018 LOYO | fwd 2018 | fwd 2022 |
  |---|---|---|---|
  | v1 | 0.0765 | 0.0884 | 0.0973 |
  | v1 - sponsor offset | **0.0755** | 0.0908 | 0.0938 |
  | v1 - house effects | 0.0808 | 0.0892 | 0.0890 |
  | v1 + mid_out (symmetric out-party shift) | 0.0773 | 0.0847 | 0.0995 |
  | v1 + mid_out x comp | 0.0767 | 0.0863 | 0.0984 |
  | v1 + other_dev (other outlets' ratings) | 0.0771 | 0.0892 | 0.0991 |
  | no sponsor + mid_out x comp (**v2**) | **0.0745** | 0.0883 | 0.0930 |
  | no sponsor + mid_out x comp + other_dev | 0.0746 | 0.0890 | 0.0948 |
  | no sponsor + mid_out x comp - env x mid | 0.0746 | 0.0888 | 0.0939 |
  - Chosen on pre-2018 LOYO: drop the sponsor offset (keep the 0.6 partisan weight and pollster house effects), add
    mid_out x comp. other_dev and the symmetric mid_out shift rejected. env x midterm stays (tie, 0.0745 vs 0.0746);
    it was a v1 test-set choice, still flagged.
  - Caveat: pre-2018 midterms (2010, 2014) both have a Democratic president, so pre-2018 LOYO identifies the
    mid_out x comp term only as a midterm-competitive-seat shift toward R. Its sign flip for a Republican president
    (2018, 2026) is the out-party symmetry assumption, motivated by the cross-model investigation, whose evidence
    includes 2018. That motivation used test-year information, so count it as soft selection.
- 2026-only changes (no backtest possible): Kalshi `UNK` labels on the 9 CA same-party rows -> real party (DEM, CA-40
  REP); RedBox Research rows all tagged D (MI-04, MI-10 were blank); dossier late-information blend (`dossiers.py`):
  logit q_fav' = 0.75 logit q_model + 0.25 logit q_dossier (0.15 when the skeptic weakened the view), 85 races.
- 2026 E[D House] (model before dossiers): v1 sims mean 233.4; v1 flags with the v2 fixes 233.6 -> - sponsor offset
  234.6 -> + mid_out x comp 236.7 (mid_out x comp alone, with the sponsor offset kept: 235.6); after the dossier blend 237.0. Not tuned to any target.
- Scorecard v2 (final):
  - kavi-c 2018: logloss 0.0810 (538 0.1002) brier 0.0442; contested 0.3078 (538 0.3778)
  - kavi-c 2022: logloss 0.0956 (538 0.0952) brier 0.0625; contested 0.4465 (538 0.4375). 2022 is now leak-free
    and no longer beats 538; with full-cycle money it would be 0.0904.
- 2026: House D mean 237.0, median 237 (80% 223-250), P(D control) 0.956. Senate D-caucus median 53 (80% 49-55),
  P(D >= 51) 0.788 (v1 0.88: IA-SEN moved from Turek 0.63 to Hinson 0.53 once R-sponsored polls stopped being
  shifted toward D). Top pick differs from Kalshi in 6 races: CO-05, FL-22, ME-02, MI-10, SC-01, WI-01.

## v3 (secret round 2, 2026-11-02)
Started from the disclosed v2 (91e3993). v2 outputs copied to `out/v2/` (v1 stays in `out/v1/`). Read first: REPORT.md
(origin/report/strategies-1102), audit/kavi-c/AUDIT.md, audit/cross/INVESTIGATION.md, pau-a/pau-b v3 and the Kavi v2
METHODS. Three changes kept, two tried and rejected.

1. **2022 backtest money from a pre-election cut (borrowed from kavi-a).** v2 dropped money from the 2022 backtest
   because kavi-b found no Wayback capture of weball22 between Apr 11 and Dec 28 2022. kavi-a found another file:
   `webl22.zip` (current House+Senate campaigns) captured 2022-08-03, coverage to 2022-06-30. I copied the 2022 rows of
   kavi-a's `data/kavi-a/fec_preelection.csv` to `data/kavi-c/fec_2022_preelection.csv` and replace 2022 receipts
   with it only while backtesting 2022 (`backtest.PRE_CUT_2022`, `hist_features.USE_FEC_2022_PRE`); 2022 stays a
   full-cycle training year for 2024/2026, like every other training year. Strictly pre-election, four months early.
   Effect: 2022 0.0956 -> 0.0912, contested 0.4465 -> 0.4051; 2018 unchanged. Not a selection on scores: v2 already
   said money would be used if a pre-election cut existed; I did know from REPORT.md that kavi-a's 2022 was good.
2. **Log-linear pool with a ratings-only logistic, w = 0.25 (pre-2018 LOYO choice).** v2's `baselines.py` showed
   the ratings-only logistic beating the model on average 2014-2024; the audit's ensembles (K016) showed pooling
   helps. `pool.py`: logit q = 0.75 logit q_model + 0.25 logit q_rating, where q_rating = logistic(mean core-3
   rating), fitted on the same training years as the model. `pool_experiment.py` picks w on the v2 pre-registered
   metric (mean held-out log loss over 2010/12/14/16, each fitted on the other three pre-2018 years):
   | w | pre-2018 LOYO | fwd 2018 | fwd 2020 | fwd 2022 | fwd 2024 |
   |---|---|---|---|---|---|
   | 0 (v2) | 0.0745 | 0.0883 | 0.1171 | 0.0930 | 0.0543 |
   | 0.15 | 0.0732 | 0.0879 | 0.1137 | 0.0910 | 0.0515 |
   | **0.25** | **0.0728** | 0.0881 | 0.1119 | 0.0903 | 0.0499 |
   | 0.30 | 0.0728 | 0.0884 | 0.1111 | 0.0901 | 0.0493 |
   | 0.40 | 0.0731 | 0.0893 | 0.1097 | 0.0900 | 0.0480 |
   (forward columns use full-cycle 2022 money; information only). Applied in the backtests and in 2026 to every
   rated D-vs-R race (not the independent-side, same-party or uncontested races) before the dossier blend; mu is
   moved so the simulation reproduces the pooled q. Scorecard effect (after change 1): 2018 0.0810 -> 0.0813
   (contested 0.3078 -> 0.3091, slightly worse), 2022 0.0912 -> 0.0884 (contested 0.4051 -> 0.3950). Kept: the
   choice came from pre-2018 LOYO, and the sum over the two years improves.
   - Tried and rejected: ratings logistic on [rating, env]. It tied on pre-2018 LOYO (0.0729 vs 0.0728) and its env
     coefficient is ~0 (env is in share units and C = 10 shrinks it), so it is the same model. I scored it on the
     backtests before rejecting it (2018 0.0813 / 0.3072, 2022 0.0880 / 0.3934): flagged, but the decision followed
     the pre-registered metric.
   - Follow-up, after the commit 67c12c0: the same variant with env in margin points (env x 100, so the L2 penalty
     does not erase it; fitted env coefficient 0.01-0.49 per point). Pre-2018 LOYO best 0.0733 (w 0.20) vs 0.0728 for
     rating-only: rejected. Forward 2018-24 mean 0.0868 vs 0.0851 (information only, computed in the same run).
   - Tried: an epsilon tail mix q' = (1 - e) q + e/2 on top of the pool. Pre-2018 LOYO prefers e = 0 (0.07285 vs
     0.07295 at e = 0.001). Not adopted.
3. **Same-party (CA/WA top-two) finals from the June primary (idea from kavi-b / kavi-e; data copied from kavi-e's
   branch).** `same_party.py`: logit P(A beats B) = b_pct (primary% A - primary% B)/10 + b_inc (inc_A - inc_B), no
   intercept, mirrored pairs, L2 (C = 10), fitted on same-party generals of the training years only (2018 test:
   25 pairs, b_pct 0.89; 2022: 37 pairs, 1.05; 2026: 48 pairs, b_pct 0.97, b_inc 0.80; the primary leader won 85%).
   Backtests: replaces v2's hand-set incumbency + money softmax when both finalists are in the primary file (head-to-
   head polls still override). 2026: the fitted probability becomes the prior mean of the conjugate-normal model
   (prior share 0.5 + 0.07 t^-1(P0)); v2's money term is dropped there. First run had a same-surname bug (WA-09 2018
   Adam v Sarah Smith both got 48.4%): fixed with a first-name tie-break. Scorecard: 2018 0.0813 -> 0.0818 (WA-09
   Adam Smith 0.975 -> 0.87, CA-08 Cook 0.95 -> 0.83: v2's incumbency rule was more confident and both incumbents
   won), 2022 0.0884 -> 0.0878, contested 0.3950 -> 0.3924. Net about neutral; kept because it is fitted rather than
   hand-set and it decides 9 races in 2026.
   - 2026 same-party polls: my poll file lacked several same-party polls (CA-04's two Jones-campaign polls, CA-07,
     CA-11, two CA-34 polls). Added Pau's disclosed 2026 poll rows for the 9 finals (origin/pau/model-1101
     `data/features/polls_2026.csv`, frozen at `data/kavi-c/pau_polls_2026_same_party.csv`), deduplicated on
     (pollster key, end date); any named sponsor counts as partisan (weight 0.6). CA-04 Thompson 0.90 -> 0.52 once the
     Jones internals are in.
4. **Tried, rejected:** `env x midterm x comp` (symmetric GB form of the out-party term, after pau-a's gb_mid) in
   place of or next to mid_out x comp. Pre-2018 LOYO: replacing 0.0755 (vs v2 0.0745), adding 0.0745 (no change; its
   coefficient is ~0). `out/experiments_v3.csv`. v2's mid_out x comp stays.

- Scorecard (`src/scorecard.py`, shared/scorecard-1101), v2 -> v3:
  | | 2018 all | 2018 contested | 2022 all | 2022 contested |
  |---|---|---|---|---|
  | v2 | 0.0810 | 0.3078 | 0.0956 | 0.4465 |
  | + 2022 June FEC cut | 0.0810 | 0.3078 | 0.0912 | 0.4051 |
  | + ratings pool (w 0.25) | 0.0813 | 0.3091 | 0.0884 | 0.3950 |
  | + primary same-party = **v3** | **0.0818** | **0.3091** | **0.0878** | **0.3924** |
  | v3 without the pool | 0.0815 | 0.3078 | 0.0907 | 0.4025 |
  | v3 without the 2022 cut | 0.0818 | 0.3091 | 0.0922 | 0.4302 |
  | 538 | 0.1002 | 0.3778 | 0.0952 | 0.4375 |
  (`out/ablation_v3.csv`). v3 is worse than v2 in 2018 by 0.0008 (all) / 0.0013 (contested) and better in 2022 by
  0.0078 / 0.054; 2022 now beats 538 on every metric. v3 kept as the final.
  - kavi-c 2018: n=470 logloss 0.0818 (538 0.1002) brier 0.0440 (538 0.0578) acc 0.972 score top 46.42 ev 45.71 (chalk 42.13) against-538 10
  - kavi-c 2018 contested: n=120 logloss 0.3091 (538 0.3778) brier 0.1712 acc 0.892
  - kavi-c 2022: n=470 logloss 0.0878 (538 0.0952) brier 0.0561 (538 0.0576) acc 0.955 score top 30.47 ev 43.18 (chalk 30.43) against-538 14
  - kavi-c 2022 contested: n=90 logloss 0.3924 (538 0.4375) brier 0.2666 acc 0.778
- 2026 (v3): House D mean 234.9, median 235 (80% 221-248), P(D control) 0.944 (v2 237.0, 0.956). The ratings pool
  costs about 2.0 seats (237.0 without it): the ratings logistic gives a toss-up 0.43 for the Democrat, so it pulls
  toss-ups toward R. Senate D-caucus median 52 (80% 49-55), P(D >= 51) 0.743 (v2 0.788). Top pick differs from Kalshi
  in 7 races: CA-34 (Gomez), CA-40 (Calvert), CO-05 (Killin), FL-22 (Askar), MI-10 (Bouchard), SC-01 (Lacore),
  WI-01 (Steil). ME-02 dropped off the list (LePage now 0.59, after the pool).
