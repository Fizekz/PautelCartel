# kavi-d LOG (ML classifier / upset finder)

Running log, newest last. All loss numbers: natural-log loss; "rolling" = contested D/R races of the training table,
year Y predicted by a model fitted only on cycles < Y; "scorecard" = src/scorecard.py on all 470 benchmark races.

## 2026-11-01 session 1

- Setup: worktree kavi/kavi-d-1101 from kavi/data-1101. Read every data/clean/*.md.
- Built training table (src/models/kavi-d/history.py): 3,450 contested races 2010-2024 (House ~370-415/yr,
  Senate 33-36/yr). Anti-R side = Democrat, or the major independent (King, Sanders, Orman, McMullin, Osborn).
  Dropped: same-party top-two finals (50 House + CA-SEN 2016/18), one-sided races, MA-SEN Jan 2010.
  Features (D-positive): ratings (cook/sabato/inside/mean/spread/imputed flag), lean_m = 200*(prior pres share -
  national), base_m = lean_m + final generic-ballot margin, inc (FEC I/C/O of the matched nominees), seat party
  (House, unchanged lines only), prev_rel (previous House margin minus national House margin, unchanged lines only),
  FEC share / log ratio / log total, poll_m (recency+sample weighted, pollster-shared weights, final 60 days), poll_n.
- Rolling-origin run 1 (experiments.py), free-intercept Platt calibration on OOF cycles:
  gbm 2014 .087 2016 .056 2018 .205 (!) 2020 .130 2022 .085 2024 .054. Calibration HURT 2018 (uncalibrated .136).
  Cause: Platt intercept a=-1.31 learned that 2014/2016 were R-overperforming years and pushed every 2018 race to R.
  That is a year effect, not a calibration property.
- Fix 1: slope-only Platt (a=0, b clipped to [0.5,1.25]). gbm 2018 .129.
- Fix 2: "symmetric" training: append the D<->R mirror image of every training row (signed features negated,
  outcome flipped), so the model cannot learn a partisan base rate from the composition of the training cycles.
  gbm_sym 2018 .094, 2022 .090 (worse than asym .078 in 2022), pre-2018 mean .055 vs .072.
- Logistic regression (6 features + missing flags) is as good as or better than GBM everywhere:
  logit_sym pre-2018 mean .0508, all-years mean .0747; gbm_core_sym .0556/.0780; blend_core_sym (GBM+logit
  average in logit space) .0514/.0753. GBM alone with all 17 features overfits (.0554/.0787). Fewer features win.
- Chose blend_core_sym (ensemble of GBM-core and logit, symmetric, slope Platt). Selection caveat: I saw the
  2018-2024 rolling numbers when choosing; on the strict pre-2018 criterion logit_sym is marginally best (.0508 vs
  .0514), a negligible difference.
- Backtest on 538 race lists + scorecard (blend_core_sym):
  2018: logloss 0.0878 (538 0.1002), brier .0480 (.0578), acc .970, score top 48.27 (chalk 42.13)
  2022: logloss 0.0851 (538 0.0952), brier .0530 (.0576), acc .960, score top 31.76 (chalk 30.43)
- Leak check on FEC: weball totals run to Dec 31 of the election year (post-election money). Dropping all FEC
  features: 2018 0.0859, 2022 0.0848, i.e. FEC adds nothing (slightly negative). Switched the party model to the
  no-FEC variant (blend_core_sym_nofec) so the possible leak cannot help it. FEC is still used to split probability
  between same-party candidates (CA top-two, LA jungle, AK) and to pick each side's principal.
- Rescored after switching to no-FEC: 2018 0.0859, 2022 0.0848 (538 0.1002 / 0.0952).
- Inspected the worst 2018/2022 misses. Found a structure bug: NY-14 2018 Crowley (WFP line, big receipts,
  incumbent) was a "viable independent" on the D side and took AOC's probability (q 0.32). New rule: an
  independent is viable only if there is no Democrat (and >= $1M), or the raters rate the race for an independent
  (King "Safe I") and he is the top-money independent; the nominal Democrat is then minor (fixed ME-SEN's
  Ringelstein getting 0.13 too). Found by looking at 2018 output, so this is a test-set-informed fix (logged).
- OK-05 2018 (Horn, q 0.002 vs 538 0.14) cost 6.4 nats. Tested a probability floor for races not rated Safe by all
  three on the rolling OOF: floor 0.01 -> 2014 .0621 (+.0005), 2016 .0419 (+.0003), 2018 .0830 (-.0034), 2020
  .1215 (-.0034), 2022 .0848 (+.0002), 2024 .0494 (+.0001). Pre-2018 evidence says it costs a little; adopted as
  insurance for wave years (config `final`).
- Final scorecard (config final):
  kavi-d 2018: logloss 0.0796 (538 0.1002) brier 0.0448 (0.0578) acc .979 score top 51.20 ev 44.98 (chalk 42.13)
  kavi-d 2022: logloss 0.0849 (538 0.0952) brier 0.0527 (0.0576) acc .960 score top 31.76 ev 42.92 (chalk 30.43)
- Same-party model (sameparty.py): softmax of a*incumbent + b*log10(receipts); ridge MLE on 52 same-party finals
  2012-2024 (from the raw MIT files): (a,b) = (0.39,1.43) for 2018, (0.99,0.98) for 2022, (1.15,1.14) for 2026.
- 2026 first pass: 9 picks against Kalshi incl. CA-34 (Gomez 0.76 vs Kalshi 0.18). Collected the June 2026 CA
  top-two primary results from Wikipedia (fetch_ca_primary.py, data/extra/ca_primary_2026.csv + .md): Gomez led
  46.0-30.7, Calvert led Kim 34.9-20.6, Vang led Matsui 31.2-29.1. Added hand-set weights (0.04 logit / primary
  point, 0.08 / poll point) for the 9 same-party races only. CA-34 Gomez now 0.85 (Kalshi 0.18): the model and the
  market disagree hard; I trust the market's extra information more than my hand-set weights here.
- 2026 final: 470 races, 951 Kalshi slots; 74 disagreement rows, 9 against the Kalshi favourite (WI-01, MI-10,
  KS-SEN, VA-01, MI-04, KY-06, FL-22, CA-34, CA-40). Model is 50/50-ish in Toss-ups where Kalshi has D at 0.69-0.89
  (OH-SEN, TX-SEN, AK-SEN, TX-15, OH-07) and much more confident than Kalshi in Likely/Safe-R seats.
- Simulation (simulate.py): Gaussian copula, rho_national 0.255 from the per-year probit shifts of the OOF
  predictions (2014 -.42, 2016 -.48, 2018 +.49, 2020 -1.11, 2022 +.38, 2024 -.14), rho_state 0.05 hand-set.
  House D median 232 (80% 221-242), P(D House) 0.945. Senate D-caucus median 51 (80% 48-54), P(D Senate) 0.62.
- Upset analysis (analysis.py): upset rates Lean .18 / Likely .035 / Safe .002, model matches each bucket. Beyond
  the rating, upsets come with lean+environment against the favourite (-0.44 sd coef), rater spread (+0.21), and a
  Democratic favourite (+0.54): R beat the symmetric model in 4 of 6 validation years. OOF calibration bins
  0.05-0.30 and 0.85-0.95 over-predict D. Not corrected (the sign flipped in 2018 and 2022).
- GBM permutation importance: r_mean 1.75 nats, base_m 0.04, poll_m 0.003, rest ~0. Honest conclusion: the ML part
  adds little over a 5-feature logistic; the gains vs 538 come from leaning on the expert ratings, a symmetric
  (no partisan prior) fit and slope-only calibration.

## 2026-11-01 session 2 (v2)

- Recovered context from LOG/METHODS. The interrupted v2 attempt had only copied the new scorecard (with the
  contested block) into src/scorecard.py and copied v1 outputs to out/v1/. Kept both; everything else is new.
- Read: referee audit (kavi-audit/kavi-d/AUDIT.md), cross-model investigation, siblings' METHODS/LOG/out, dossiers.
- Audit fixes:
  - FEC post-election look-ahead in the backtests: 2018 now uses the pre-election Wayback snapshot of weball18
    (borrowed from kavi-b, data/extra/fec_2018_preelection.csv; joined by cand_id, else surname + given name in the
    race). 2022 has no pre-election cut, so the 2022 backtest uses NO money at all (receipts NaN; the same-party
    model is refitted with b = 0). The $1M independent-viability threshold is replaced by a money-free rule (no
    Democrat and not Safe R for all three raters, or rated "I"). Training rows still use FEC only for the I/C/O flag.
  - WA-09 2018 surname collision: FEC rows are now disambiguated by given name (ignoring money-less stub committees,
    which otherwise broke GA-14 2018 Tom Graves). Adam Smith 0.50 -> 0.68 (he won). Same fix changes inc for
    MO-08 2016 (Jason Smith) in the training table.
  - KY-03 2026 Rodriguez/Rodriquez: one-edit surname match with the same first name. Rodriguez is now the R
    nominee (q 0.0004, Safe D) instead of a minor slot.
- New features (all D-positive, mirror-symmetric): op = +1 midterm under an R president / -1 under a D president /
  0 presidential; op_close = op x (1 - |r_mean|/3); oth_dev = mean of the non-core outlets (538 excluded) minus the
  core-3 mean; rcp_dev = RCP minus core mean; sparse poll window (if < 2 matched polls in 60 days, use 150 days).
- Design selection: new criterion = leave-one-year-out inside 2010-2016 (train on the other three cycles, no
  calibration step, 1% floor), never sees 2018+. Results (log loss, mean of 4 folds; contested = |r_mean| < 3):
  v1_final 0.0734 / 0.2892, logit_v1 0.0704 / 0.2748, op (blend) 0.0711 / 0.2789, logit_op 0.0666 / 0.2601,
  logit_op_rcp 0.0664 / 0.2598, oth 0.0751, rcp 0.0732, logit_oth 0.0713, logit_op_oth 0.0672.
  logit_op improves every one of the four folds over logit_v1 (2010 .1009->.0948, 2012 .0775->.0758, 2014
  .0620->.0574, 2016 .0413->.0385). Chosen: final_v2 = logit_op (symmetric L2 logistic on r_mean, base_m, inc,
  poll_m, prev_rel, op + missing flags; slope Platt; 1% non-Safe floor). rcp_dev's extra 0.0002 not worth a feature.
  The other raters (oth_dev) hurt in LOYO, so the quant forecasters' 2026 ratings are NOT a model input (history only
  has media raters in those columns; there is no backtest for the 2026 quant mix).
- Choices that used 2018/2022 (test-year selection, logged):
  - Poll window: 60 vs 150 days could not be tested before 2018 (2010-16 538 files keep ~60 days). Rolling
    2018-24 was a wash (logit_op 60: .0812/.1156/.1014/.0478; 150: .0829/.1144/.1001/.0475). Kept 60.
  - The blend (GBM + logit) with op scored better than logit_op on both backtests (2018 .0752 vs .0771, 2022 .0978
    vs .1008). Not switched: the pre-specified criterion picked logit_op.
- op coefficient (logit, C = 0.3): fitted on cycles < 2018: 0.75; < 2022: 0.80; 2010-2024 (2026 model): 0.34. Per
  held-out midterm out-party shifts (base logit fitted on the other cycles): 2010 +0.47, 2014 +0.74, 2018 +1.29,
  2022 -0.73 (mean +0.44, sd 0.85): 2022 is the one midterm where the out-party underperformed the ratings.
- Same-party model: borrowed kavi-b's top-two primary history (data/extra/primaries_top_two.csv) and FITTED the
  primary weight (was hand-set 0.04/pt): c = 0.062 (for 2018), 0.080 (2022), 0.085 (2026) per point above the pair
  mean; incumbency drops to 0.0/0.37/0.62, money (log10) 0.73/0.35/0.44. A no-primary fit (LA jungle, AK, CA-SEN)
  keeps incumbency+money (2026: 1.17, 1.15), and a no-money, no-primary fit for 2022 (a = 1.81). Before this fix the
  2022 LA jungle incumbents (Letlow, Higgins) fell to 0.33 because the incumbency weight from the primary fit was
  too small: found on the 2022 output (test-year-informed, logged).
- Borrowed kavi-b's CA sub-page polls (data/extra/ca_polls_2026.csv) for the same-party poll term.
- Scorecard steps (all / contested 2018, 2022):
  v1 committed 0.0796/0.2931, 0.0849/0.3867 -> pipeline fixes (v1 config) 0.0787/0.2937, 0.0852/0.3961 ->
  logit_v1 0.0808/0.3016, 0.0820/0.3809 -> final_v2 0.0771/0.2877, 0.1008/0.4701 (538 0.1002/0.3778,
  0.0952/0.4375). The midterm term buys 0.0037 in 2018 and costs 0.0188 in 2022 (2022 toss-ups: the out-party
  won 7 of 27, the model said 0.69). v2 loses to 538 in 2022.
- 2026 late-information adjustment from the 85 dossiers (late_info.py): logit pool of the underdog's share, w = 0.25
  (medium/high confidence) / 0.15 (low, or skeptic refuted) / 0.5 for same-party contests (set after seeing CA-34:
  model Gomez 0.85, dossier 0.22, an August ethics scandal no input carries). House E[D] 234.19 -> 234.87.
- 2026 House E[D] by step: v1 231.6, logit_v1 231.8, final_v2 model-only 234.2, + dossiers 234.9 (Kalshi 246.0,
  investigation fair value ~238). Not tuned toward 238.
- Simulation (20,000 runs): rho_n 0.254 (OOF probit shifts 2014 -.15, 2016 -.35, 2018 +.26, 2020 -1.14, 2022 +.73,
  2024 -.11). House D median 235 (80% 223-246), P(D) 0.968; Senate D median 52 (80% 49-54), P(D) 0.717.
- Picks against the Kalshi favourite: 7 (IA-SEN, KS-SEN, CA-34, CA-40, MI-04, MI-10, WI-01).
- Added check_outputs.py (invariants: coverage, 951 slots, q sums, no UNK, sims >= 10k): OK. Committed locally.

## 2026-11-02 session 3 (v3, secret round 2)

- Read: own LOG/METHODS, REPORT.md (strategies-1102, sections 1.3-1.4, 8.4), kavi-e METHODS section 3 and
  out/national_env.json, the leaderboard. Diagnosis shared by the report and my own v2 log: the out-party-midterm
  sign `op` (+1/-1 by White-House party) is the single design choice that made kavi-b/c/d/f good in 2018 and bad in
  2022; the models that tie the environment to the generic ballot (pau-a, pau-b, kavi-e, kavi-a) are stable.
- v2 outputs copied to out/v2/ (backtests, pred_2026, scorecard, sims, disagreements, late_info). out/v1/ untouched.
- New code: `environment.py` (N_true / N_hat with uncertainty), `experiments_v3.py` (v3 configs; the v2 criteria
  plus a LOYO over all eight cycles), new logit design terms in `model.py` (gb_mid, env_mid, base_sen,
  gb_mid_close, poll_strong) and an `env` path in `raw_fit_predict`.
- Ideas tested (all mirror-symmetric L2 logistic, slope Platt, 1% non-Safe floor; numbers = LOYO pre-2018 all /
  contested | LOYO all-8 all / contested | rolling 2018, 2020, 2022, 2024):
  - logit_op (v2)        0.0666/0.2601 | 0.0783/0.3022 | .0812 .1156 .1014 .0478
  - logit_v1 (no env)    0.0704/0.2748 | 0.0770/0.2972 | .0853 .1191 .0812 .0473
  - **logit_gbmid**      0.0682/0.2670 | 0.0744/0.2878 | .0781 .1157 .0802 .0468   <- chosen
    (gb_mid = final generic ballot x midterm / 10, D-positive; idea from pau-a "GB x midterm in non-safe races"
    and pau-b "year effect = a + b x GB x midterm"; my version is one symmetric logit term)
  - logit_op_gbmid       0.0668/0.2610 | 0.0769/0.2981 | .0810 .1150 .0947 .0474
  - env_gb / env_e       0.0698 / 0.0689 | 0.0767 / 0.0760 | 2022 .0837 / .0842
    (kavi-e idea: train base_m = lean + ACTUAL national House margin, predict with a pre-election N_hat integrated
    over N ~ Normal(N_hat, sd>=2.5) by 9-node Gauss-Hermite; "gb" = my GB-bias-corrected N_hat, "e" = kavi-e's
    combined N_hat with specials/governors/fundamentals for midterms)
  - env_e_op             0.0664/0.2587 | 0.0779/0.3004 | .0814 .1125 .1022 .0471   (best pre-2018 LOYO)
  - env_e_gbmid          0.0674/0.2633 | 0.0740/0.2859 | .0782 .1122 .0831 .0462   (best all-8 LOYO)
  - env_e_mid (N x midterm) 0.0676 | 0.0752; env_e_sen (Senate base slope) 0.0696 | 0.0760; env_e_gbclose
    (kavi-c idea, wave concentrated in competitive seats) 0.0683 | 0.0752; env_e_gbmid_close 0.0674 | 0.0742;
    env_e_gbmid_polls (poll margin x poll count) 0.0674 | 0.0736; env_e_gbmid_op 0.0664 | 0.0757.
  - Not in the table: GB-bias-corrected base_m without true-N training (pre 0.0673-0.0678, all-8 0.0744, rolling
    2018 .0791-.0800, 2022 .0828: worse rolling, not used); GBM+logit blend with gb_mid (pre 0.0724, all-8 0.0778:
    worse); C grid 0.1/0.3/1/3 on env_e_gbmid (pre 0.0696/0.0674/0.0679/0.0715): kept C = 0.3.
- Selection. The pre-registered v2 criterion (LOYO inside 2010-2016) cannot tell `op` from `gb_mid` apart on the
  question that matters: its only midterms (2010, 2014) are both D-president, R-wave years, so `op` = -1 is just
  a "midterm R shift" there; it prefers op by 0.0016 because 2014's generic ballot (R+1.9) understated the wave.
  I therefore switched to gb_mid on (a) the structural argument (two Pau models, kavi-e and the report all tie the
  midterm environment to the generic ballot rather than to the president's party) and (b) the LOYO over all
  eight cycles (0.0744 vs 0.0783), which includes 2018 and 2022. **Test-year-informed choice, flagged.**
- Scorecard (src/scorecard.py, all / contested):
  - v2 final_v2          2018 0.0771 / 0.2877   2022 0.1008 / 0.4701
  - env_e_gbmid (tried)  2018 0.0743 / 0.2768   2022 0.0838 / 0.3887
  - **v3 = logit_gbmid** 2018 0.0742 / 0.2764   2022 0.0810 / 0.3755   (538 0.1002/0.3778, 0.0952/0.4375)
  - decomposition: env_e alone 0.0810/0.3026, 0.0848/0.3945; env_e_op 0.0772/0.2880, 0.1016/0.4742;
    env_e_gbmid_polls 0.0733/0.2730, 0.0835/0.3878.
- **Reverted: the env (true-N + N_hat integration) layer.** It was my first v3 candidate (env_e_gbmid, committed
  as the intermediate 84bdb0d) and is preferred by both LOYO criteria, but on top of gb_mid it made the scorecard
  worse (2018 0.0742 -> 0.0743, 2022 0.0810 -> 0.0838), so per the round rules it is reverted; the code stays as a
  variant (`out/variants/env_e_gbmid/`). Choosing logit_gbmid over env_e_gbmid is itself a test-year choice.
  poll_strong was not added (neutral on pre-2018 LOYO; its scorecard gain was seen only after the decision).
- Per chamber v3 (v2): 2018 House 0.0668 (0.0693), Senate 0.1661 (0.1741); 2022 House 0.0782 (0.0963), Senate
  0.1153 (0.1572). v3 beats 538 in both years on all races and on contested races, and beats v2 everywhere.
- Fitted gb_mid coefficient (per 10 GB points in a midterm, logit): 0.81 (cycles < 2018), 1.04 (< 2022), 1.07
  (2010-2024, 2026 model). Unlike op it did not halve when 2022 was added (2022's GB R+1.2 asked for only a small
  R shift, which is what happened). 2026: GB D+8.56 x 1.07 = +0.92 logit toward D in every race (v2 op: +0.34).
  Calibration slope b: 2018 1.25 (clipped), 2022 0.99, 2026 1.06.
- OOF midterm table (analysis_midterm.csv): 2022 toss-ups model 0.55 for the out-party (v2 0.69; actual 0.26);
  2018 toss-ups 0.66 (actual 0.67); 2014 0.60 (actual 0.69).
- 2026: House E[D] model-only 238.5, after the unchanged dossier pool 238.8 (v2 234.9; Kalshi sum 246.0;
  investigation ~238; kavi-e 236.6, pau-b 238). Simulation (20,000, rho_n 0.228): House D median 239, 80% 228-249,
  P(D) 0.992; Senate D median 53, 80% 50-55, P(D) 0.845 (v2 0.717).
- Picks against the Kalshi favourite: 6 (IA-SEN Turek 0.62, AL-02 Figures 0.52, CA-34 Gomez 0.55, CA-40 Calvert
  0.61, CO-05 Killin 0.56, WI-01 Steil 0.68). Dropped from v2's list: KS-SEN, MI-04, MI-10 (now with Kalshi);
  new: AL-02, CO-05. Largest q moves vs v2: ME-02 Dunlap 0.34 -> 0.46, AL-02 Figures 0.40 -> 0.52, TX-35 Garcia
  0.61 -> 0.72, MI-10 Hines 0.49 -> 0.61, KS-SEN Hamilton 0.45 -> 0.56, IA-SEN Turek 0.51 -> 0.62.
- Borrowed and credited: pau-a / pau-b (GB x midterm form, adopted), kavi-e (true-N training + N_hat with
  uncertainty, data/extra/kavi-e_national_env.json; tested, reverted), kavi-c (competitive-seat concentration;
  tested, rejected), REPORT.md 1.4 (diagnosis).
- run_all.sh reproduces everything (5:49 wall), check_outputs.py OK.
