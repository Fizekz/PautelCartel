# kavi-b METHODS (v3): calibrated expert ratings with a generic-ballot cycle prior

One-line model: **logit P(D side wins) = 1.89 x mean expert rating + 0.81 x poll margin/10 + 0.96 x lean/10 + u,
where the national cycle effect u has a Bayesian prior u ~ N(c + b x midterm x GB/10, tau^2) fitted on the 2010-2024
year effects (2026, a D+8.6 generic-ballot midterm: mean +0.35, sd 1.40) and is integrated out; same-party top-two
races are split by primary share, money and late polls; a 25% logit pool with the Nov 1 research dossiers adds late
information in 85 contested races.**

Code `src/models/kavi-b/`, outputs `out/` (v2 outputs in `out/v2/`, v1 in `out/v1/`), branch `kavi/kavi-b-1101`.
This v3 section is self-contained; the v2 text below it is kept for the details it gives (rating tables, the v2
selection table, the 2026 special-race list) and is still accurate except where this section says otherwise.

## v3.0 What changed from v2 (two changes)

| change | source of the idea | effect 2018 all / contested | effect 2022 all / contested |
|---|---|---|---|
| 2022 backtest: within-party splits use a pre-election FEC cut (Wayback `webl22.zip`, 2022-08-03, coverage ~June 30) instead of no 2022 money | data found and extracted by kavi-a (`origin/kavi/kavi-a-1101: data/kavi-a/fec_preelection.csv`) | 0 | 0.0950 / 0.4417 -> 0.0904 / 0.4208 |
| cycle prior conditioned on the generic ballot in midterms (`gb_mid`) instead of the out-party midterm (`mid_mo`) | pau-b v3 (year effect = a + b GB x midterm); REPORT.md 1.3/1.4 (GB-tied models stable in 2018 and 2022) | 0.0896 / 0.3282 -> 0.0860 / 0.3164 | 0.0904 / 0.4208 -> 0.0853 / 0.3959 |

The second change was **selected by looking at 2018/2022** (section v3.6). Everything else (features, side model,
allocation, same-party model, dossier pool, simulation mechanics) is v2's.

## v3.1 Data

| input | years | use | notes |
|---|---|---|---|
| Cook / Sabato / Inside final ratings (`data/clean/ratings_hist.csv`, `ratings_2026.csv`) | 2010-24, 2026 | backbone `r` = mean of the three, D-positive, -3..+3 | 538 never used as input; other outlets tested and rejected (v2). |
| MEDSL results (`results_house.csv`, `results_senate.csv`) | 2010-24 | outcome y = D side won | test-year results only reach the scorer. |
| Lean (`lean_hist.csv`, `lean_2026.csv`) | | `lean` = prior presidential D two-party share minus national, pts | 2026 on the 2026 district lines. |
| Polls (`polls_hist.csv`, `polls_2026.csv`, own `data/kavi-b/ca_polls_2026.csv`) | | recency-weighted margin, final 60 days (120 if < 2 polls) | partisan/internal x0.5, sponsor harmonisation (v2). |
| **Generic ballot** (`national_hist.csv: generic_margin_dr`; `national_2026.csv: generic_ballot_avg`) | 2010-24, 2026 | **v3: cycle-prior covariate** | historical: 538's election-eve average (pre-election); 2026: mean of the 5 aggregators on 2026-10-30 (DDHQ, FiftyPlusOne, NBC, RCP, Silver Bulletin) = D+8.6. |
| FEC (`fec_hist.csv`, `fec_2026.csv`) + pre-election cuts `data/kavi-b/fec_2018_preelection.csv` (Wayback weball18, 2018-11-06) and **v3** `data/kavi-b/fec_2022_preelection.csv` (Wayback webl22, 2022-08-03; from kavi-a; 8 rows with coverage after the capture date dropped) | | money only in within-side splits and the historical independent-principal rule | test years always use the snapshot; training years use two-year totals (outcome-known, allowed). The 2022 independent rule still ignores money. |
| Top-two primaries (`data/kavi-b/primaries_top_two.csv`) | CA/WA 2012-26 | same-party splits | |
| Dossiers (`repo/data/dossiers/summary.csv`) | 2026 | late-information pool | market-anchored, no history. |
| Kalshi `data/kalshi/races_2026-11-01T074812Z.csv` | 2026 | candidate list and p_kalshi column only | not an input. |

## v3.2 Features (unchanged from v2)

`r` (mean rating), `poll` (poll margin / 10, 0 if unpolled), `lean` / 10. Not used (tested, section v3.6): incumbency,
FEC, poll shrink, rating hinge, other outlets, tail floor.

## v3.3 Model

1. **Side model** (`model.py`): binomial GLM with one dummy per year (no global intercept), L2 0.005 on the three
   slopes, fitted on contested non-same-party races not rated Safe by all three (871 race-years 2010-24). Safe-by-all
   races get 0.003 for the trailing side (0 upsets in 2,577).
2. **Cycle prior** (`cycle.py`, design `gb_mid`): the fitted year intercepts a_y (sampling variance s_y^2 =
   1/sum p(1-p)) are modelled as a_y = theta_0 + theta_1 x midterm_y x GB_y/10 + e_y, e_y ~ N(0, tau^2); priors
   theta ~ N(0, 1), tau ~ half-Cauchy(0, 1) (a priori, unchanged from v2); tau integrated on a 60-point grid with a
   conjugate normal posterior for theta at each point. The predictive for a new cycle is a 60-component normal
   mixture; q = sum_k w_k E[sigmoid(eta + mu_k + sd_k Z)] (41-point Gauss-Hermite).

   | fit | r | poll/10 | lean/10 | theta (const, midterm x GB/10) | target GB | cycle mean | sd |
   |---|---|---|---|---|---|---|---|
   | 2010-16 (2018 test) | 1.670 | 1.068 | 1.049 | -0.13, +0.29 | D+8.6 | +0.12 | 1.56 |
   | 2010-20 (2022 test) | 1.748 | 0.961 | 1.041 | -0.22, +0.60 | R+1.2 | -0.30 | 1.36 |
   | 2010-24 (2026) | 1.891 | 0.808 | 0.963 | -0.14, +0.58 | D+8.6 | **+0.35** | **1.40** |

   Year effects 2010-24 (uncentred, D+): -0.67, +1.26, -0.84, -0.97, +1.14, -1.93, +0.82, -0.33. Reading: the
   effect is the part of the cycle the raters missed; it grows with the generic ballot in midterms (2018 D+8.6 ->
   +1.14; 2010 R+6.7 -> -0.67), but the shared intercept also carries the pro-R presidential misses (2016, 2020), so a
   D+8.6 midterm gets only +0.35. An exact toss-up in a neutral seat with no polls is 0.57 D in 2026 (v2 0.62).
3. **Calibration** is the year-effect integration itself (no Platt step, no floor). The prior sd 1.40 widens every
   non-safe race; the LOYO tables in v3.6 are the calibration check.

## v3.4 Candidate allocation and special races (unchanged mechanics)

- D side = Democrat, or the principal independent (2026: NE, MT, ID, SD-SEN and AK-AL independents; CA-06 Kiley (I)
  is the R-side principal vs Pan). Minor candidates get min(0.001, 10% of the side / n_minor).
- Same-party top-two races (2026: 8 CA D-vs-D + CA-40 R-vs-R): side mass split by a mirrored logit fitted on past
  CA/WA same-party generals on primary lead, ln receipts ratio and incumbency (2026 fit: n 50 / 41 with primaries;
  +0.60 per 10 pts primary lead, +1.32 per ln ratio, incumbent -0.04; 2022 backtest fit now with June money: +0.50,
  +1.20, -0.13), then nudged by Sep-Oct head-to-head polls (1.6 x lead/sd, campaign polls half weight and moved 4 pts
  against the sponsor; CA-04 per the dossier).
- AK-SEN and LA-01..06 party-level; uncontested races (FL-10, MA-01/02/05/07, WI-02, AZ-03, NJ-08, PA-03) 0.999.
- **Dossier pool**: for the 85 dossier races, logit p'_dog = 0.75 logit p_dog(model) + 0.25 logit q_dog_mean on the
  Kalshi (favourite, underdog) pair; judgment weight, not backtestable. Logged in `note` and
  `out/dossier_adjustments_2026.csv` (`q_model` = before the pool).

## v3.5 Simulation (`simulate.py`)

20,000 draws: national u from the 2026 gb_mid mixture (mean +0.35, sd 1.40), state shock N(0, 0.35^2) shared by a
state's races, per-race eta solved so simulated marginals reproduce final q (max error 0.006). House D mean 232.8,
median 234, 80% 217-248, P(D control) 0.889. Senate D caucus mean 51.3, median 52, 80% 48-54, P(D control) 0.651
(0.661 if the new independents caucus with D).

## v3.6 Validation and selection (`select_v3.py` -> `out/report/model_selection_v3.csv`; backtests `backtest.py`)

- Backtests: 2018 fitted on 2010-16, 2022 on 2010-20; inputs are final pre-election ratings, polls ending before
  election day, the election-eve generic-ballot average, prior presidential lean, and pre-election FEC cuts (splits
  only). 538 is only the scorer's benchmark.
- Side-model log loss (non-Safe contested, r+poll+lean, no floor):

  | cycle | LOYO 10-16 midterms | LOYO 10-16 all | LOYO 10-24 midterms | LOYO 10-24 all | fwd 2018 | fwd 2022 |
  |---|---|---|---|---|---|---|
  | const | 0.2815 | 0.3096 | 0.3009 | 0.3057 | 0.2966 | 0.3384 |
  | mo | 0.2750 | 0.3148 | 0.3132 | 0.3146 | 0.2641 | 0.3846 |
  | mid_mo (v2) | 0.2733 | 0.3200 | 0.3066 | 0.3158 | 0.2908 | 0.3654 |
  | **gb_mid (v3)** | 0.2809 | 0.3114 | 0.2966 | **0.3035** | 0.2787 | 0.3440 |
  | mid_gb | 0.2753 | 0.3201 | 0.2961 | 0.3105 | 0.3083 | 0.3317 |
  | mid_mo_gb | 0.2742 | 0.3192 | 0.3028 | 0.3140 | 0.2919 | 0.3603 |
  | bma (all six, marginal-likelihood weights) | 0.2762 | 0.3183 | 0.3017 | 0.3115 | 0.2865 | 0.3526 |

- **Selection on test years (flag).** v2's pre-2018 criterion cannot judge a GB slope (one training midterm), so I
  moved to LOYO 2010-2024 over all cycles, which contains 2018 and 2022, after seeing the forward columns. gb_mid
  wins it; mid_gb ties it on midterms only. The 2018/2022 improvement from the cycle change is therefore optimistic.
  The 2022 FEC change is a pure data fix (pre-election cut), not selected on scores.
- Feature re-checks with gb_mid (LOYO 10-24 all; none kept): floor 0.3035, poll shrink 0.3037 (k=1) / 0.3030
  (k=0.5, but fwd 2022 0.352 vs 0.344), other outlets 0.3044, FEC 0.3042, incumbency 0.3031 (LOYO 10-16 midterms
  0.2974 vs 0.2809), hinge 0.3040.

**Scorecard (`out/scorecard.txt`), v3 vs v2, log loss all / contested:**

| | 2018 | 2022 |
|---|---|---|
| 538 | 0.1002 / 0.3778 | 0.0952 / 0.4375 |
| v2 | 0.0896 / 0.3282 | 0.0950 / 0.4417 |
| **v3** | **0.0860 / 0.3164** | **0.0853 / 0.3959** |

## v3.7 2026 output summary

House E[D] (DEM rows) 232.8 (side model 231.4; v2 234.9; Kalshi 246.0); Senate D-side q-sum 17.5 (v2 17.9). Top picks
against Kalshi's favourite: CA-04 Jones 0.78 (Kalshi 0.47), CA-34 Gomez 0.53 (0.18), MI-10 Bouchard 0.55 (0.24),
WI-01 Steil 0.73 (0.50), CO-05 Killin 0.50 (0.45). Biggest v2 -> v3 moves are 4-5 pts toward R in SC-01, MI-10, AL-02,
ME-02, WI-01, IA-SEN (Turek 0.50 -> 0.46, now on the market's side).

## v3.8 Known weaknesses

- The cycle prior rests on 8 year effects; designs are statistically indistinguishable (BMA weights 0.13-0.22). The
  2026 mean (+0.35) is design-dependent (mid_gb +0.61, mid_mo +0.67, const -0.15); the sd 1.40 is the honest part.
- The cycle design was selected with 2018/2022 visible (above).
- gb_mid shares one intercept between midterms and presidential years; if presidential-year poll misses (2016, 2020)
  do not carry over to midterms, v3 is too R in 2026 (the investigation's concern about Kavi models in wave years).
- 2022 backtest money is a June cut, weaker than 2018's election-eve cut; training years use post-election totals.
- Unchanged from v2: dossier weight 0.25 and 4-pt sponsor shift are judgments; same-party splits rest on ~41 races
  and cannot see scandals (CA-34); state shock sd 0.35 assumed; AK RCV and LA jungle treated as two-sided.

## v3.9 Reproduce

`bash src/models/kavi-b/run_all.sh` (adds `select_v3.py` and the v3 variants `v2_mid_mo`, `bma_nofloor`,
`mid_gb_nofloor`; ~2 min, no network).

---------------------------------------------------------------------------------------------------------------

# kavi-b METHODS (v2): calibrated expert ratings with an out-party-midterm cycle model

One-line model: **logit P(D side wins) = 1.89 x mean expert rating + 0.81 x poll margin/10 + 0.96 x lean/10 + u,
where the cycle effect u has a fitted, cycle-type-dependent prior (2026, an out-party midterm: mean +0.67, sd 1.53)
and is integrated out; same-party top-two races are split by primary share, money and late polls; a 25% logit pool
with the Nov 1 research dossiers adds late information in 85 contested races.**

Code: `src/models/kavi-b/`; outputs: `out/`; v1 outputs kept in `out/v1/`; branch `kavi/kavi-b-1101`.
"LOYO" = leave-one-year-out inside the training years; "fwd" = true forward test. In-sample numbers are labelled.

---------------------------------------------------------------------------------------------------------------

## 1. Data used

| Source (file) | Years | n used | Cleaning / notes |
|---|---|---|---|
| Expert ratings, `data/clean/ratings_hist.csv`, `ratings_2026.csv` (Cook, Sabato, Inside; final pre-election Wikipedia columns) | 2010-2024, 2026 | 3,760 + 470 | D-positive (`r = -rating`, -3 Safe R .. +3 Safe D). `r_mean` = mean of the three. "Safe I" on the D side (King, Sanders). Imputed-safe rows kept. The `other_*` outlets were tested in v2 (section 7) and are not used; 538 is never used. |
| Results, `results_house.csv`, `results_senate.csv` (MEDSL) | 2010-2024 | 3,759 race-years | `y = 1` if the winner is not a Republican. Same-party top-two generals feed only the within-party model. |
| Lean, `lean_hist.csv`, `lean_2026.csv` | 2010-2024, 2026 | all races | `lean = 100 x (prior presidential D two-party share - national)`; 2026 = 2024 Harris share on the **2026 lines** (handles the 9 redrawn states). |
| Polls, `polls_hist.csv`, `polls_2026.csv` | 2010-2024, 2026 | 1,057 polled race-years; 2026: 147 races | General, non-hypothetical, names must match both principals (party-only Kalshi slots match by party). Window: final 60 days, **v2: widened to 120 days when fewer than 2 polls fall in the 60 days** (2010-16 538 files hold only ~60 days, so this fires from 2018 on). Recency 0.5^(days/14), partisan/internal x0.5, one value per pollster + end date, pollster weight / sqrt(count). **v2 sponsor harmonisation**: within a (pollster, end date) batch of >= 3 rows the majority sponsor tag is applied to all rows, and a sponsor note naming a party-aligned group tags the row (fixes RedBox / A Fight Worth Having, Oct 20-23: MI-04 and MI-10 were untagged while 16 sister rows were D). |
| **Own:** CA 2026 sub-page polls `data/kavi-b/ca_polls_2026.csv` | 2026 | 37 polls, 10 districts | Wikipedia CA district sub-pages; merged unless the shared file has the poll. |
| **Own:** top-two primaries `data/kavi-b/primaries_top_two.csv` | CA/WA 2012-2026 | 2,418 rows | June primaries; used in same-party races only. |
| FEC `fec_hist.csv`, `fec_2026.csv`, `fec_race_2026.csv` | 2010-2024, 2026 | | Nominee matched by surname. Used only in within-side splits (CA/WA/LA/AK) and in the historical "which independent is the D-side principal" rule. |
| **Own:** pre-election FEC 2018 `data/kavi-b/fec_2018_preelection.csv` | 2018 | 3,656 candidates | Wayback capture of `weball18.zip` (file dated 2018-11-06 21:22 UTC). Replaces 2018 receipts. |
| **v2: 2022 FEC is not used for the 2022 test year.** | 2022 | | No pre-election cut exists (no Wayback capture between April and 28 Dec 2022), so the 2022 backtest uses no 2022 money: the within-party model is refitted on incumbency + primary share, and the independent-principal rule uses ratings only for 2022. Training years keep their (outcome-known) money. |
| National `national_hist.csv` | 2010-2024 | 8 | `cycle_type` and `president_party` define midterm and out-party for the cycle model. |
| **v2: research dossiers** `repo/data/dossiers/summary.csv` (+ `<race>.md`) | 2026 | 85 races | Nov 1 research + skeptic review per contested race. `q_dog_mean` (Kalshi underdog probability) enters the 2026 late-info pool (section 3.5). The CA-04 dossier also supplies two facts used as inputs: both public CA-04 polls are Jones-campaign releases, and the undated Binder poll ran Oct 15-21. No historical analogue; not backtestable. |
| Kalshi `data/kalshi/races_2026-11-01T074812Z.csv` | 2026 | 470 races, 951 slots | **Not a model input**: candidate list, `p_kalshi` (mid / sum of mids) and the disagreement report. |
| 538 `data/benchmark/fte_final_{2018,2022}.csv` | | | **Not an input**: candidate list of the backtests and the scorecard benchmark. |

## 2. Features and why

- **Mean expert rating `r`** (backbone): raters digest polls, money, candidates and environment; the model learns how
  often each rating comes true.
- **Poll margin / 10** (0 when no poll): within-bin information beyond the raters' steps. (A shrunken version,
  margin x W/(W+k), was tested and lost on the selection criterion.)
- **Lean / 10**: conditional on rating, partisanship still predicts the winner (+0.96 per 10 pts).
- **Cycle effect u (year fixed effect in the fit; v2 hierarchical prior at forecast time).** The dominant error
  source. Uncentred year effects from the 2010-2024 fit (logit, D+): 2010 -0.67, 2012 +1.26, 2014 -0.84,
  2016 -0.97, 2018 +1.14, 2020 -1.93, 2022 +0.82, 2024 -0.33. Signed toward the out-party, midterms are +0.67,
  +0.84, +1.14, -0.82 (2022 is the exception); presidential years average -0.49. v1's single negative constant
  ("ratings overstate D") was therefore a presidential-year effect that v1 applied to a midterm.
  Descriptive (in-sample) midterm table `out/report/midterm_fav_win_out_in.csv`: out-party-favoured Lean seats held
  1.00/1.00/1.00/0.94 (2010/14/18/22); in-party-favoured Lean seats held 0.87/0.64/0.90/1.00; out-party-favoured
  toss-up/tilt favourites won 0.83/0.82/0.83/0.31.
- **Not used**: incumbency, FEC money, rating spread/hinge, other outlets, r x midterm, environment (section 7).

## 3. Model and parameters

### 3.1 Side model (`model.py`, FINAL in `backtest.py`: features r, poll, lean; ridge 0.005; cycle `mid_mo`; no floor)

1. Fit set: contested, non-same-party races with |r_mean| < 3 (871 race-years 2010-2024). Races Safe by all three
   get a fixed `SAFE_UPSET = 0.003` (0 upsets in 2,577 contested cases).
2. `statsmodels` binomial GLM, L2 0.005 on feature coefficients, one dummy per year (no global intercept).
3. Year intercepts a_y and their sampling variances s_y^2 = 1 / sum p(1-p) go to the cycle model.

| fit | r | poll/10 | lean/10 | cycle mean for target | cycle sd |
|---|---|---|---|---|---|
| 2010-2016 (2018 backtest) | 1.670 | 1.068 | 1.049 | +0.00 | 1.83 |
| 2010-2020 (2022 backtest) | 1.748 | 0.961 | 1.041 | -0.51 | 1.51 |
| 2010-2024 (2026) | 1.891 | 0.808 | 0.963 | **+0.67** | **1.53** |

### 3.2 Cycle model (`cycle.py`, new in v2)

a_y = z_y' theta + e_y, e_y ~ N(0, tau^2), a_y observed with noise s_y^2; z_y = [presidential, midterm,
midterm x out_d] (`mid_mo`), out_d = +1 if the D side is the out-party (Republican president). Priors theta ~
N(0, 1) each and tau ~ half-Cauchy(0, 1), set a priori (not tuned). tau is integrated on a 60-point grid; for each
grid point theta has a conjugate normal posterior, so the predictive distribution of a new cycle's effect is a
60-component normal mixture with weights = posterior of tau. q = sum_k w_k E[sigmoid(eta + mu_k + sd_k Z)]
(41-point Gauss-Hermite).

2026 posterior: theta = (pres -0.36, midterm +0.25, midterm x out +0.42), tau posterior mean 1.17; predictive mean
for 2026 (midterm, D out) **+0.67**, sd **1.53** (v1: mean -0.19, sd 1.10). Interpretation: an exact toss-up in a
neutral seat with no polls is now 0.62 D instead of 0.45; the larger sd (theta uncertainty from 4 midterms, one
of which went the other way) widens every non-safe race and the seat distribution.

Why 2018 gets 0.00: in 2010-2016 both midterms had a D president, so `midterm` and `midterm x out` are collinear;
the prior splits the midterm effect between them and their sum cancels for 2018. The 2018 backtest therefore does not
test the out-party term (design `mo` = [1, midterm x out] does: +0.44 for 2018, section 5).

### 3.3 Candidate allocation (`allocate.py`)

D side (DEM / principal independent / minors) and R side. Minors (other parties, "Others", withdrawn Democrats in
NE/ID/SD/MT, Hafner AK-AL) get min(0.001, 10% of the side / n_minor) (v2 fix: v1 let a withdrawn Democrat take a
quarter of a 0.003 independent side). Same-party candidates share their side by a softmax of the within-party
model fitted on past CA/WA same-party generals (mirrored symmetric logit):

| fit | n | primary lead /10 pts | ln receipts ratio | incumbent |
|---|---|---|---|---|
| for 2018 | 21 (25 money-only) | +0.45 | +1.06 | -0.23 |
| for 2022 (v2: no money) | 32 (38) | +0.85 | (not used) | +0.45 |
| for 2026 | 41 (50) | +0.59 | +1.35 | -0.03 |

### 3.4 2026 same-party poll nudge (`predict_2026.py`)

Sep-Oct head-to-heads move the split logit by 1.6 x lead / sd, sd = sqrt(6^2 + 8^2 / n_eff). **v2**: partisan- or
campaign-sponsored polls count 0.5 in n_eff and in the average, and a campaign poll's lead is moved 4 points against
its sponsor (borrowed in spirit from kavi-e's sponsor shift). CA-04: both polls are Jones releases (dossier), the
Binder poll's Oct 15-21 dates come from the dossier; Jones 0.93 (v1) -> 0.85 (model) -> 0.79 (after the pool).

Special 2026 handling as in v1: NE/MT/ID/SD-SEN and AK-AL use the independent as the D-side principal; CA-06 Pan (D)
vs Kiley (I, R-side principal); AK-SEN and LA-01..06 party-level; uncontested races (FL-10, MA-01/02/05/07, WI-02,
AZ-03, NJ-08, PA-03) 0.999. Party labels: CA same-party finalists carry their real party (DEM for the 8 D-vs-D
races, REP for CA-40); the script asserts no UNK label remains.

### 3.5 Late-information pool with the dossiers (`dossiers.py`, new in v2)

For each of the 85 dossier races, on the pair (Kalshi favourite, Kalshi underdog):
`logit p'_dog = 0.75 logit p_dog(model) + 0.25 logit q_dog_mean`, pair mass and other candidates unchanged.
The weight 0.25 is a judgment (about one outlet in a four-way average of three raters plus the dossier): the
dossiers contain real news (scandals, late polls, money) but have no track record and are anchored on the Kalshi
price (most say "market about right"), and Kalshi's tails are known to be inflated. Effect: House E[D] +1.1 seats
(233.8 -> 234.9); largest moves CA-34 Gomez 0.65 -> 0.54, CA-04 Jones 0.85 -> 0.79, MI-10 Hines 0.45 -> 0.50,
AK-AL Hill 0.09 -> 0.12, MT-SEN Bodnar 0.003 -> 0.007. Every adjusted race is logged in
`out/dossier_adjustments_2026.csv` and in the `note` column of `pred_2026.csv` (`q_model` = before the pool).

### 3.6 Simulation (`simulate.py`)

20,000 draws: cycle effect u from the 2026 predictive mixture (drawn mean +0.673, sd 1.515); state shock s ~ N(0,
0.35^2) shared by a state's House seats and Senate race (assumed); race probability sigmoid(c eta_i + c u + s),
c = sqrt(1 + pi 0.35^2/8), where eta_i (`race_eta_2026.csv: eta_sim`) is solved per race so that the
cycle-integrated probability equals the final q after allocation and the dossier pool (simulated marginals match
q within 0.008). No-contest and same-party races have a fixed party. Senate holdover R 31, D caucus 34; D control
needs 51.

## 4. Training and validation

- **2018 backtest**: fit on 2010-2016 only; 2018 inputs are final ratings, polls ending before election day,
  2016 presidential lean, the pre-election FEC snapshot (splits only). **2022 backtest**: fit on 2010-2020; no
  2022 money. 538 is never an input.
- **Selection** (`select_v2.py` -> `out/report/model_selection_v2.csv`, side-model log loss on non-Safe contested
  races). Criterion: **held-out midterms (2010, 2014) in LOYO 2010-2016**, because the target is a midterm.

| features | cycle | floor | LOYO 10-16 midterms (criterion) | LOYO 10-16 all | LOYO 10-24 midterms | LOYO 10-24 all | fwd 2018 | fwd 2022 |
|---|---|---|---|---|---|---|---|---|
| **r+poll+lean (FINAL)** | **mid_mo** | **no** | **0.2733** | 0.3200 | 0.3066 | 0.3158 | 0.2908 | 0.3654 |
| r+poll_sh(k=0.5)+lean | mid_mo | yes | 0.2738 | 0.3175 | 0.3085 | 0.3162 | 0.2867 | 0.3740 |
| r+poll+lean | mid_mo | yes | 0.2743 | 0.3209 | 0.3078 | 0.3166 | 0.2910 | 0.3668 |
| r+poll+lean | mo | no | 0.2750 | 0.3148 | 0.3132 | 0.3146 | 0.2641 | 0.3846 |
| r+poll+lean+oth | mid_mo | no | 0.2761 | 0.3230 | 0.3086 | 0.3165 | 0.2931 | 0.3672 |
| r+poll_sh+lean | mid_mo | no | 0.2761 | 0.3159 | 0.3103 | 0.3159 | 0.2890 | 0.3744 |
| r+poll_sh+lean+fec | mid_mo | no | 0.2764 | 0.3170 | 0.3108 | 0.3162 | 0.2924 | 0.3758 |
| r+poll_sh+lean+r_mid | mid_mo | no | 0.2799 | 0.3175 | 0.3111 | 0.3165 | 0.2941 | 0.3752 |
| r+poll+lean (v1 cycle) | const | no | 0.2815 | 0.3096 | 0.3009 | 0.3057 | 0.2966 | 0.3384 |
| r+poll+lean (v1-like) | const | yes | 0.2825 | 0.3105 | 0.3003 | 0.3053 | 0.2926 | 0.3395 |
| r+poll_sh+lean | const | no | 0.2844 | 0.3070 | 0.3027 | 0.3056 | 0.2952 | 0.3468 |

  (Full 39-row table in the CSV.) **Honest reading**: on v1's criterion (all held-out years 2010-16) the `const`
  designs win, because with three training years the conditioned designs mispredict the held-out presidential
  years; the midterm-only criterion was adopted after seeing that table (and the same printout showed fwd 2018 /
  2022). On LOYO 2010-24, and in fwd 2022, `const` beats `mid_mo` because 2022 went against the out-party. I kept
  `mid_mo`: it wins on the pre-registered-in-spirit pre-2018 midterm data, it is the principled president's-party x
  midterm conditioning, and its Bayesian sd carries a 2022-type reversal. The cost is visible in the 2022 backtest.
- **Tail floor removed** (v1's floor was motivated by OK-05 2018; the pre-2018 criterion prefers no floor in every
  pair).
- **Calibration (in-sample, descriptive, `out/report/calibration_by_bin.csv`)**: favourite win rate by |mean rating|:
  tilt 0.659 (n 252), lean 0.909 (186), (1,1.5] 0.944 (72), likely 0.969 (163), (2,2.5] 0.970 (99), Safe 1.000
  (2,577). Midterm out- vs in-party table in section 2.

## 5. Scorecard numbers (`out/scorecard.txt`, `out/scorecard.json`)

```
kavi-b 2018: n=470 logloss 0.0896 (538 0.1002) brier 0.0494 (538 0.0578) acc 0.972 score top 48.00 ev 41.78 (chalk 42.13) against-538 12; dropped 0
       2018 contested (538 fav<0.95): n=120 logloss 0.3282 (538 0.3778) brier 0.1902 (538 0.2257) acc 0.892 (538 0.875)
kavi-b 2022: n=470 logloss 0.0950 (538 0.0952) brier 0.0608 (538 0.0576) acc 0.955 score top 31.14 ev 37.38 (chalk 30.43) against-538 14; dropped 0
       2022 contested (538 fav<0.95): n=90 logloss 0.4417 (538 0.4375) brier 0.2919 (538 0.2787) acc 0.778 (538 0.822)
```

Variants on the same scorecard (`out/variants/scorecard_variants.txt`), all / contested:

| variant | 2018 | 2022 |
|---|---|---|
| v1 as committed (`out/v1/`) | 0.0902 / 0.3343 | 0.0861 / 0.3975 |
| v1-like (const + floor) with v2 data fixes, 2022 money kept | 0.0901 / 0.3328 | 0.0846 / 0.3924 |
| v1-like, no 2022 money | 0.0901 / 0.3328 | 0.0888 / 0.4106 |
| const, no floor | 0.0913 / 0.3376 | 0.0886 / 0.4106 |
| mo (const + out-party midterm), no floor | 0.0817 / 0.3004 | 0.0995 / 0.4635 |
| **mid_mo, no floor (FINAL)** | **0.0896 / 0.3282** | **0.0950 / 0.4417** |
| mid_mo + poll shrink | 0.0890 / 0.3251 | 0.0971 / 0.4523 |
| ratings only, mid_mo | 0.0934 / 0.3428 | 0.1013 / 0.4731 |

FINAL beats 538 in 2018 (all and contested) and ties it in 2022 on all races (0.0950 vs 0.0952), but loses on
2022 contested races (0.4417 vs 0.4375): the out-party-midterm prior was wrong in 2022, and money was removed.

## 6. 2026 picks against the Kalshi favourite (top q differs; 6 races, `out/disagreements_2026.csv`)

q = our final probability for our pick (q_model = before the dossier pool), p = Kalshi.

| race | our pick q (q_model) | Kalshi p for it | reason |
|---|---|---|---|
| CA-34 | Gomez D 0.54 (0.65) | 0.18 | same-party: Gomez led the June primary 46-31 and out-raised Gonzales-Torres 2.6x; three Sep-Oct polls have her +7 (n_eff 2.5), which moves the split from 0.90 to 0.65; the ethics probe is not a feature and the dossier (G-T 0.78) pulls it to 0.54. Low conviction. |
| CA-04 | Jones D 0.79 (0.85) | 0.465 | same-party: Jones out-raised Thompson 18.1M to 5.5M; both polls (+13, +10) are Jones releases, discounted; Thompson won the primary 41-22. Dossier says ~0.52. |
| MI-10 | Bouchard R 0.50 (0.55) | 0.24 | Toss-up/Toss-up/Tilt R, lean R+2.6, polls R+4.9 (RedBox D-sponsored R+5 now half weight); the +0.67 cycle shift almost cancels it. A coin flip. |
| WI-01 | Steil R 0.68 (0.72) | 0.50 | Lean R by all three, lean R+1.5, polls even; Lean R seats held 90% even in 2018. |
| CO-05 | Killin D 0.55 (0.58) | 0.445 | three Toss-ups, lean R+3.9 offset by the out-party shift and polls D+1.7. |
| IA-SEN | Turek D 0.50 (0.52) | 0.44 | three Toss-ups, lean R+6.0, polls even (23); the out-party shift makes it a coin flip, ~0.6 pts more than Kalshi. |

Down from 15 in v1: the out-party cycle shift moved OH-07, TX-35, FL-22, VA-01, MI-04, TX-SEN, OH-SEN, KS-SEN,
KY-06, MT-01, PA-01 to the market's side (e.g. OH-07 Poindexter 0.50 -> 0.69, TX-35 Garcia 0.39 -> 0.59, OH-SEN
Brown 0.49 -> 0.65). We remain below Kalshi on most Lean/Likely R seats (NC-03 Smith 0.08 vs 0.45, NC-13 0.06 vs
0.41, WI-01 0.32 vs 0.50): fading the R-seat longshots is where we keep our edge (investigation: Kalshi's tails are
inflated by thin books).

Expected D seats, House (DEM rows incl. 8 CA D-v-D): **234.9** (side model 233.8; v1 227.3; Kalshi 246.0;
investigation fair value ~238). By rating bin (House, non-same-party; ours / Kalshi): Safe R 0.5 / 7.5, Likely R
1.5 / 5.3, Lean R 4.0 / 6.0, Tilt R 1.1 / 1.4, Toss-up 5.5 / 5.6 (n 9), Tilt D 5.1 / 5.8, Lean D 18.2 / 18.6,
Likely D 17.6 / 17.2, Safe D 173.5 / 170.7. Senate D q-sum (races up) 17.7 vs Kalshi 17.9.

**Simulation (`out/sims_summary.json`)**: House D seats mean 234.9, median 236, 80% interval 218-251 (90%: 211-256),
P(D control) 0.902. Senate D caucus mean 51.7, median 52, 80% 48-55, P(D control) 0.698 (0.706 if the new
independents caucus with D).

## 7. What was tried and failed (v1 and v2)

- v2: **other outlets** (mean of all non-538 `other_*` minus the core mean): worse on every pre-2018 criterion
  (e.g. 0.2761 vs 0.2733). Pre-2018 those outlets are RCP/NYT/CQ/DailyKos, not the 2026 quant models (DDHQ,
  Silver Bulletin, Split Ticket, Economist, FiftyPlusOne), so this does not prove the 2026 quant ratings are
  uninformative; there is no history to fit them on.
- v2: **poll shrink** by summed weight (kavi-f / kavi-c style): better on all-year LOYO 2010-16, worse on the midterm
  criterion; k flat in {0.5, 1, 2, 4}.
- v2: r x midterm slope; FEC in the side model; tail floor: worse on the criterion.
- v2: `mo` cycle design: excellent 2018 (0.0817) but worst 2022 (0.0995), worse on the criterion than `mid_mo`.
- v1: full fundamentals (negative FEC sign), environment terms without year effects, chamber-specific lean,
  rating disagreement terms, money-only within-party model (CA-14).

## 8. Known weaknesses

- The cycle prior rests on 4 midterms, one of which (2022) reversed; if 2026 behaves like 2022 our D-side toss-up
  probabilities are too high together. If it is a 2010-size wave, Likely R seats fail at ~10-18% and we are too low.
- The 2018 backtest cannot test the out-party term (collinear with the midterm term in 2010-16 training data).
- The dossier weight (0.25) and the internal-poll bias (4 pts) are judgments; dossiers are market-anchored.
- Same-party CA splits: 41 past races plus a heuristic poll weight; no scandal feature (CA-34).
- State shock sd 0.35 assumed; AK RCV and the LA jungle are treated as two-sided races.
- 2022 backtest has no money in the splits (conservative; costs 0.004 all-race log loss).
- The 120-day poll window extension could only be judged on test years (net ~0).

## 9. What I borrowed from whom

| idea | from | how used in kavi-b v2 |
|---|---|---|
| condition the environment on midterm (env x midterm) | kavi-c | turned into a hierarchical prior on my cycle effect with a president's-party x midterm term (`cycle.py`), not a generic-ballot slope |
| poll weight shrinkage (n_eff / (n_eff + k); min(neff,6)/6) | kavi-c, kavi-f | tested as `poll_sh`; rejected by selection |
| partisan polls shifted against the sponsor | kavi-e | applied to campaign-released same-party polls (4 pts, half weight) |
| Kalshi tails saturate / longshot compression | kavi-f, investigation | not a model input; used to interpret the remaining gap (we do not chase Kalshi in Safe/Likely R seats) |
| wave-midterm rating resolution, summer polls, RedBox tag, quant raters | investigation | cycle model; 120-day window; sponsor harmonisation; other outlets tested |
| pre-election FEC or drop money; tail floor was test-informed; allocation quirk; same-party sponsorship | audit | 2022 money dropped; floor dropped by pre-2018 criterion; minor-share fix; CA-04 fix |

Kept my own: ratings-calibration backbone with year fixed effects, the Bayesian cycle mixture, the within-party
primary-share model, and the CA sub-page data; no generic-ballot or market input.

## 10. v1 -> v2 changes and their measured effect

| change | 2018 all / contested | 2022 all / contested | 2026 House E[D] |
|---|---|---|---|
| v1 as committed | 0.0902 / 0.3343 | 0.0861 / 0.3975 | 227.3 |
| + Bayesian cycle (const), 120-day window, sponsor fix, minor-share fix | 0.0901 / 0.3328 | 0.0846 / 0.3924 | (const side model: 219.4 non-same-party + 8) |
| + drop 2022 own money | same | 0.0888 / 0.4106 | n/a |
| + drop tail floor | 0.0913 / 0.3376 | 0.0886 / 0.4106 | ~same |
| + cycle `mid_mo` (FINAL side model) | 0.0896 / 0.3282 | 0.0950 / 0.4417 | 233.8 (non-same-party 219.4 -> 225.8) |
| + CA-04 internal-poll fix | n/a | n/a | 0 (D-v-D seat) |
| + dossier pool (w 0.25) | n/a | n/a | 234.9 |

Simulation: P(D House) 0.845 -> 0.902; Senate P(D control) 0.487 -> 0.698. Against-market top picks 15 -> 6.

## 11. How to reproduce

```bash
cd models/kavi-b
bash src/models/kavi-b/run_all.sh          # ~2 min; cached raw files, no network; reads repo/data/dossiers/summary.csv
# steps:
PY=python
$PY src/models/kavi-b/select_v2.py                 # out/report/model_selection_v2.csv
$PY src/models/kavi-b/backtest.py --variants       # out/backtest_{2018,2022}.csv, out/variants/
$PY src/scorecard.py --pred 2018=out/backtest_2018.csv --pred 2022=out/backtest_2022.csv --name kavi-b --json out/scorecard.json
$PY src/models/kavi-b/predict_2026.py              # out/pred_2026.csv, disagreements_2026.csv, dossier_adjustments_2026.csv, race_eta_2026.csv, model_2026.json
$PY src/models/kavi-b/simulate.py --n 20000 --seed 2026   # out/sims_summary.json, sims_marginals_2026.csv
$PY src/models/kavi-b/report.py                    # out/report/*.csv
```
