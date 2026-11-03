# Model Citizens 2026: Full Congress forecast

This repository holds our entry to the Stanford Democracy Day "Model Citizens" forecasting contest, **Full Congress
track**. We pick a winner in each of the 470 races of the 2026 US midterm elections (435 House seats and 35 Senate
seats, election day Tuesday, November 3, 2026). The repository also holds every piece of code and data needed to
rebuild those picks.

- **The submission:** [`predictions_full_congress.csv`](predictions_full_congress.csv), in the organizers' template
  (`race_id, chamber, predicted_winner_party, candidate_name, win_probability`), 470 rows.
  SHA-256 `b0cba991a4be0871a176463ffbe49cb039acbe12e0a76ad53ba23e6e56bd5cb4`.
- **Scoring.** A correct pick earns `-log2(max(p, 0.01))` points, where `p` is the Kalshi price of the picked
  candidate on November 1. A wrong pick earns 0. Picking a 50-cent candidate who wins earns 1 point; picking a
  10-cent candidate who wins earns 3.3; picking a 95-cent favourite earns 0.07.
- **The one-paragraph write-up** is in [`WRITEUP.md`](WRITEUP.md).

## The result

Our picks give Democrats 225 House seats and Republicans 210, and Democrats 20 Senate races to the Republicans' 15.
The forecast behind them expects about 235.5 Democratic House seats and about 18.6 Democratic Senate wins (18.8
counting the independents on the Democratic side).

In 453 races we pick the market favourite. In **17 races we pick the underdog**, because our forecast gives that
candidate at least the chance the market does and the payoff is larger:

| Party picked | Races |
|---|---|
| Democrat (1 + 1 Senate) | CO-05 (Jessica Killin), **IA-SEN (Josh Turek)** |
| Republican (15) | FL-22, MI-04, MI-10, NC-01, NJ-07, OH-07, PA-08, PA-10, TX-15, TX-35, VA-01, VA-02, WA-03, WI-01, WI-03 |

(WI-01 is a coin flip on Kalshi, both candidates at 0.495; we count the listed favourite, so Bryan Steil appears
here as an underdog.)

Under our own probabilities the picks are worth **58.3 expected points**, against 51.3 for simply picking every
market favourite. Both figures use our probabilities, which favours our picks by construction; if the Kalshi prices
are taken as the truth instead, the picks are worth 48.5 expected points against 45.9 for the favourites. Every pick, with our probability, the Kalshi price, its points and its expected value, is listed
in [`out/picks_detail.csv`](out/picks_detail.csv).

## Method

The forecast is an equal blend of two independently built forecasts. Each gives a win probability `q` for every
candidate in every race. A pick rule then turns those probabilities into one pick per race.

### 1. Five "Kavi" models, pooled (pool5)

Five models, each fitted on the House and Senate races of 2010-2024 and each producing a candidate-level win
probability for all 470 races of 2026. All five use the shared data layer: expert ratings (Cook Political Report,
Sabato's Crystal Ball, Inside Elections), district and state partisan lean, polls, FEC fundraising, incumbency and
past results.

| Model | Approach |
|---|---|
| `kavi-a` | Least-squares model of the Democratic two-party vote share (lean, incumbency, money, previous result, national swing from the generic ballot), updated by race polls and ratings, recalibrated on nested leave-one-year-out predictions, and turned into a win probability with a three-level Student-t error (national, state, race). |
| `kavi-b` | Calibrated logistic model of the expert ratings: logit P(D) = a·rating + b·poll margin + c·lean + u, where u is a cycle-level effect with a fitted prior for the cycle type (2026 is a midterm with a Republican president) that is integrated out. |
| `kavi-c` | Polls-first Bayesian aggregator: a fundamentals prior on the vote share, updated by a house-effect-adjusted poll average, with a fitted national-plus-local Student-t error, then pooled (weight 0.25) with a ratings-only logistic model. |
| `kavi-d` | Party-symmetric L2-regularised logistic classifier on ratings, lean plus generic ballot, incumbency, previous margin, polls and a generic-ballot × midterm term, Platt-calibrated on rolling out-of-sample predictions. |
| `kavi-e` | Estimates the national environment from several signals (generic ballot, special elections, off-year governor races), applies a uniform swing from each race's presidential baseline, adds incumbency, rating and poll adjustments, and uses a Student-t(6) error with a shared national term. In the House it is pooled with a symmetric winner classifier. |

All five share two extras. California and Washington races between two candidates of the same party use fitted
models of the top-two primary vote shares. For 2026 only, each model also blends in, with a small fixed weight,
research dossiers we wrote on November 1 for 85 races with late developments the data might miss; see
`data/dossiers/`.

**pool5** is the equal log-linear pool of the five: for each candidate, the geometric mean of the five
probabilities, renormalised so each race sums to 1 (`src/final/ensemble.py`). Each model is documented in its own
`models/kavi-X/METHODS.md`.

### 2. The "Pau" no-primaries stack

A two-stage margin model with five add-ons, trained on 2010-2024 (about 3,750 races), built by the other half of
the team. Stage 1 regresses each race's two-party margin on lean, incumbency, the mean expert rating and a
generic-ballot year effect. Stage 2 shrinks that margin toward the race's poll average. A Student-t calibration
turns margins into probabilities, and a "district pattern" fitted on demographics carries polled races' poll errors
over to unpolled House races. The five add-ons (a safe-seat clip, a Senate state node, history-chosen demographic
directions, a same-state field and the late expert-rating move) were each kept only if they improved held-out log
loss under two selection schemes. It was run on poll tables refreshed on November 2 (49 new polls, generic ballot
about D+8.7). "No-primaries" means the version without a sixth add-on, House primary vote shares, which
over-read 2018's uneven primary turnout and hurt that backtest. It models 452 races; the other 18 (California top-two and uncontested seats) use pool5 alone. Details:
[`models/pau-stack/README.md`](models/pau-stack/README.md).

### 3. The joint forecast

The final probability `q` is the equal log-linear pool of pool5 and the stack. The stack reports a probability for
each side (Democratic side, Republican side, other); `src/final/pau_to_candidates.py` spreads each side's
probability over that side's candidates in proportion to the Kavi pool's own split. The Democratic side includes the
independents Dan Osborn (Nebraska), Todd Achilles (Idaho) and Brian Bengs (South Dakota). We chose an equal blend because, on the 2018 and 2022 backtests,
the blend beat pool5 and tied the stack (see Evaluation).

### 4. The pick rule (C′)

The score rewards correct long shots far more than correct favourites, so the best pick is not always the most
likely winner. If our probability that candidate *i* wins is `q_i` and the market price is `p_i`, picking *i* is
worth, in expectation,

    E_i = q_i × −log2(max(p_i, 0.01))

and expected points add across races. Rule **C** picks the candidate with the largest `E_i` in every race.

We use **C′**: the same, but an underdog (anyone who is not the Kalshi favourite) may be picked only if our `q` is
at least its market price. The reason is long-shot calibration. In our 2018 and 2022 backtests, candidates our models
rated 5-35% won about half as often as we said (Republican long shots in that range won 3 of 46). And when we
replayed both rules with FiveThirtyEight's final forecast as the model and real market prices for scoring
(PredictIt 2018, Kalshi and Polymarket 2024), the long shots that C took but C′ did not (positive expected value,
but rated below the market) won **0 of 37**, where the markets expected about 8, and C′ beat C in all three. (With
our own five-model pool as the model and 538's probabilities standing in for prices, C beat C′ in 2018, 47.5 to
46.0, and lost in 2022, 42.8 to 45.2.) The summary and printed outputs are in
[`research/pick_rule/`](research/pick_rule/README.md).

Two further rules:
- **The four same-party California races where C′ would take the underdog (CA-07, CA-14, CA-34, CA-40) are forced
  to the market favourite.** Only the five Kavi-side models cover these races (the stack models D-vs-R contests), so
  these picks rested on one forecast. In the 11 same-party California and Washington finals of 2018 and 2022, the
  underdog lost every time, while pool5 had given the underdogs 0.95 expected wins in total, so pool5 seems to
  overrate same-party underdogs (weak evidence: 11 races). The edges were thin or contradicted: CA-07 0.359 against
  0.3586; CA-40 0.540 against 0.481, with our own dossier below the price; CA-34 0.537 for Jimmy Gomez against 0.179,
  where our dossier says 0.22 and no poll has him ahead; CA-14 0.106 against 0.085, with the market moving away.
  Where to draw this line was a human judgement call, not a model output.
- Candidates who have withdrawn but are still listed on Kalshi (Democrats in ID-SEN, NE-SEN and SD-SEN) are never
  picked, and where Kalshi lists only a party ("Republican party"), we name that party's nominee.

The `win_probability` column is our joint `q` for the picked candidate.

## Data

All inputs are frozen as of November 1, 2026 (the Pau stack's polls and expert-rating pages as of November 2), well before the
23:00 UTC November 3 cutoff. Each file in `data/clean/` has a provenance note beside it (`<name>.md`) giving its
sources, revision IDs, row counts and known gaps.

| Data | Source | How it was fetched | Files |
|---|---|---|---|
| Election results 1976-2024 | MIT Election Data and Science Lab (MEDSL), U.S. House and Senate returns (CC0) | House file from a public GitHub mirror, MD5-checked against the Harvard Dataverse release; Senate file from the Dataverse file API. One missing CT-2022 race added from the official state canvass | `data/clean/results_*.csv`, `data/raw/mit/` |
| Benchmark forecasts and winners, 2018 and 2022 | FiveThirtyEight's final House and Senate forecasts (CC BY 4.0); winners from MEDSL | 538's files from Internet Archive (Wayback Machine) snapshots, since the original site is offline | `data/benchmark/` (backtest scoring and 2018/2022 candidate lists only) |
| Historical polls and generic ballot, 2010-2024 | FiveThirtyEight poll archives (CC BY 4.0); Gallup presidential approval via the UCSB American Presidency Project | 538's GitHub repository for 2010-2016; Internet Archive snapshots of projects.fivethirtyeight.com for 2018-2024 | `data/clean/polls_hist.csv`, `national_hist.csv` |
| 2026 polls and generic ballot | Wikipedia race pages; the VoteHub public API | MediaWiki parse API at pinned revision IDs; VoteHub public endpoint, attributed | `data/clean/polls_2026.csv`, `national_2026.csv` |
| Expert ratings, 2010-2026 | Cook Political Report, Sabato's Crystal Ball, Inside Elections, as compiled on Wikipedia's ratings pages | MediaWiki parse API, pinned revisions | `data/clean/ratings_2026.csv`, `ratings_hist.csv` |
| Candidates and redistricting, 2026 | Wikipedia race pages | MediaWiki parse API, pinned revisions | `data/clean/candidates_2026.csv`, `redistricting_2026.csv` |
| Campaign finance | Federal Election Commission bulk files | Bulk downloads, no API key. Pre-election 2018/2022 cuts from Internet Archive copies of the FEC bulk files | `data/clean/fec_*.csv`, model folders |
| Partisan lean | The Downballot's public Google Sheets (presidential results by district); MEDSL presidential returns; Census ACS demographics (Pau stack) | Public CSV exports; MEDSL via Dataverse; Census bulk files | `data/clean/lean_*.csv`, `models/pau-stack/race/` |
| Special elections | The Downballot's public special-election sheets | Public CSV exports | `models/kavi-e/data/specials/` |
| Market prices | Kalshi, snapshot of 2026-11-01 07:48 UTC | Kalshi's public market endpoints, no account or key (`src/kalshi_snapshot.py`); raw JSON kept | `data/kalshi/` |
| Race dossiers | Our own research notes on 85 races, citing news and filings | Written November 1 | `data/dossiers/` |

**Data rules we followed.** The fetchers respected robots.txt and each site's terms, and used only public pages,
public endpoints and bulk files. No API keys or credentials are stored in this repository. No market price after
the November 3 23:00 UTC cutoff was used. The Kalshi snapshot lists each race's candidates and drives the pick rule.
Market prices are not a model input, with one exception: the 85 race dossiers were written with the November 1
Kalshi prices in view, and their probabilities are partly anchored on them. They enter each Kavi model at about 25%
logit weight in those races, which pulls our `q` toward the market there and makes the `q ≥ price` guard slightly
more conservative. We did not use survey microdata such as the Cooperative Election
Study.

**Frozen inputs.** The cleaned CSVs are the frozen inputs. The fetchers can rebuild them, but they write to a
`data/raw/` download cache that is not redistributed here (about 230 MB). Without it they download again, and live
sources such as Wikipedia and VoteHub will have changed since November 1.

## Evaluation

We scored the models on the 2018 and 2022 elections with natural-log loss per race (lower is better), counting only
races between a Democrat and a Republican and flooring probabilities at 1%. "Contested" races are those where
FiveThirtyEight's favourite was below 95%.

| Forecast | 2018 all (426) | 2018 contested (120) | 2022 all (437) | 2022 contested (88) |
|---|---|---|---|---|
| pool5 (Kavi models) | 0.0792 | 0.2757 | 0.0833 | 0.3637 |
| Pau no-primaries stack | 0.0827 | 0.2901 | 0.0710 | 0.3157 |
| **Equal blend (our final forecast)** | **0.0799** | **0.2796** | **0.0750** | **0.3302** |

To compare with FiveThirtyEight's final forecasts we use a second scorer, `src/scorecard.py`, which counts all 470
races of each year (contested: 120 in 2018, 90 in 2022) and clips probabilities to [0.001, 0.999]. The two tables
are not comparable with each other.

| Forecast (all races, 0.001 clip) | 2018 all (470) | 2018 contested (120) | 2022 all (470) | 2022 contested (90) |
|---|---|---|---|---|
| pool5 (Kavi models) | 0.0746 | 0.2812 | 0.0796 | 0.3628 |
| **Equal blend (our final forecast)** | **0.0757** | **0.2865** | **0.0719** | **0.3301** |
| FiveThirtyEight's final forecast | 0.1002 | 0.378 | 0.0952 | 0.437 |

- Across both years the blend beats pool5 (difference −0.0038 per race, 95% CI −0.0071 to −0.0008) and ties the
  stack.
- The stack's leave-one-year-out log loss over 2010-2024 is 0.0619 per race (3,756 races), which we reproduced
  independently.
- **Caveat.** The stack's models never trained on the test year (the 2018 model was fitted on 2010-2016 only), but
  its final version, without the House-primaries add-on, was chosen partly because it scored better on 2018 and 2022.
  The Kavi models were fitted without the test year but designed with 2018 and 2022 visible. The blend weight and
  the pick rule were also chosen after seeing these years. Treat all 2018/2022 numbers as somewhat optimistic; the
  cleanest held-out figure is the stack's leave-one-year-out score. Historical expert ratings also carry a small
  leak (see Limitations), worth about 0.0005 log loss.

The common scorer is `src/scorecard.py`; each model's backtest predictions are in its `out/` folder.

## How to run

### Environment

Python 3.12 with the packages in `requirements.txt` (pinned to the versions we used):

    python3.12 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    export PY="$PWD/.venv/bin/python"

Rebuilding the Pau stack's input tables from raw downloads (not needed for anything below) also needs
`beautifulsoup4`, `pyshp`, `shapely` and `openpyxl`.

### One command

    bash run_all.sh            # rebuild the submission from the committed model outputs (a few seconds)
    bash run_all.sh --models   # first re-run the five Kavi models from the frozen data (about 2 minutes, offline)

`run_all.sh` rebuilds pool5, the joint forecast and the picks, writes `out/predictions_full_congress.csv`, and
checks that it is byte-identical to `predictions_full_congress.csv`. With `--models` it also re-runs kavi-a..e and
reports whether their outputs match the committed ones.

### Step by step

    # 0. tests (pooling and the pick rule)
    $PY -m pytest -q tests

    # 1. the five Kavi models (offline; writes models/kavi-X/out/pred_2026.csv and backtest_2018/2022.csv)
    PY=$PY bash scripts/run_kavi_models.sh

    # 2. the Pau stack: standard per-race file from the model's 2026 output, checked against the committed one
    (cd models/pau-stack && PYTHON=$PY bash run.sh)

    # 3. pool5
    $PY src/final/ensemble.py --out out/joint/pool5_2026.csv models/kavi-{a,b,c,d,e}/out/pred_2026.csv

    # 4. map the stack onto candidate rows, then pool it with pool5
    $PY src/final/pau_to_candidates.py out/joint/pool5_2026.csv \
        models/pau-stack/results/joint2026/stack_noprim_2026_refreshed.csv --out out/joint/pau_cand.csv
    $PY src/final/ensemble.py --out out/joint/joint_2026.csv out/joint/pool5_2026.csv out/joint/pau_cand.csv

    # 5. picks
    $PY src/final/make_submission.py out/joint/joint_2026.csv --rule Cprime --favourite CA-07,CA-14,CA-34,CA-40 \
        --out out/predictions_full_congress.csv --detail out/picks_detail.csv

    # optional: backtest scorecard for one model
    $PY src/scorecard.py --pred 2018=models/kavi-a/out/backtest_2018.csv \
        --pred 2022=models/kavi-a/out/backtest_2022.csv --name kavi-a

    # optional: rebuild the Kalshi CSVs from the saved raw JSON (copy the JSON elsewhere first)
    $PY src/kalshi_snapshot.py --raw /path/to/raw_2026-11-01T074812Z.json

**About the candidate frame.** Step 4 needs a list of candidate rows to spread the stack's side-level probabilities
over; it uses pool5's rows. The frame only sets how a side's probability is split between several candidates of that
side (it matters only for Nebraska's independents).

**Re-running the Pau stack from scratch** needs intermediate tables that are rebuilt from raw downloads not
redistributed here; `models/pau-stack/run.sh` lists the order. We did not repeat that full chain on a second machine
before the deadline; the committed outputs are the ones the model produced.

## Repository layout

    predictions_full_congress.csv   the submission
    WRITEUP.md                      one-paragraph write-up
    run_all.sh                      one-shot rebuild and check
    requirements.txt
    data/
      clean/        frozen Nov 1 inputs, each with a provenance .md
      kalshi/       Kalshi Nov 1 07:48 UTC snapshot (raw JSON, per-market and per-race CSVs)
      benchmark/    538 final forecasts and winners, 2018 and 2022 (backtests only)
      templates/    the organizers' templates
      dossiers/     85 race research dossiers and their summary table
      raw/mit/      MEDSL House and Senate returns (read by kavi-d)
    src/
      data/         fetchers that built data/clean
      final/        ensemble.py, pau_to_candidates.py, make_submission.py
      scorecard.py  common 2018/2022 backtest scorer
      kalshi_snapshot.py, benchmark_fte.py, pick_policy.py
    models/
      kavi-a … kavi-e/   code (src/models/kavi-X), own inputs (data/), METHODS.md, LOG.md, out/
      pau-stack/         code (pbase/src), inputs, outputs, README.md, study notes
    scripts/        run_kavi_models.sh, build_pool5.sh
    out/
      joint/        pool5, the candidate frame, the stack on candidate rows, the joint forecast
      picks_detail.csv
    research/pick_rule/   the analysis behind rule C′ (summary and outputs)
    tests/          unit tests for pooling and the pick rule
    docs/PROVENANCE.md    where each part came from

The `METHODS.md` and `LOG.md` files in each model folder, and the study notes in `models/pau-stack/`, are the
working notes written while building each model. They contain internal labels and are included as technical
appendices.

## Limitations

- **An unusual year.** The 2026 national environment (generic ballot about D+8.6) is outside the range of the
  training cycles, so the models extrapolate.
- **Long shots.** Both forecasts overrated candidates they gave 5-35% in 2018 and 2022. The q ≥ price guard limits
  the damage, but our 17 underdog picks still depend on our probabilities being right where the market disagrees.
- **Correlated errors.** A single national polling miss moves most races together, so the 58.3 expected points come
  with a wide spread: a strong Democratic night would sink most of our 15 Republican underdog picks at once.
- **Partly in-sample backtests** for the Kavi models, and design choices made with 2018 and 2022 in view (see
  Evaluation).
- **Dossiers anchored on the market.** Our 85 race dossiers were written with the Kalshi prices in view, so in those
  races our `q` is not fully independent of the price it is compared with in the pick rule.
- **A small leak in historical ratings.** Past expert ratings were parsed from current revisions of Wikipedia's
  ratings pages, which may include post-election edits; we estimate the effect at about 0.0005 log loss.
- **Sparse polling.** Most House races have no public polls; there the models lean on ratings, lean and money.
- **Market tails.** Kalshi prices are compressed at the extremes (the favourite-longshot bias), so a market
  "favourite" at 0.9 may be a better bet than its price suggests, and our q ≥ p guard is only a rough correction.
- **Archived 538 files.** The historical poll training data for 2018-2024 and the 538 benchmark files come from
  Internet Archive snapshots of FiveThirtyEight's public (CC BY 4.0) data, because the original site is offline.

## Team and credits

Built by **Kavi Patel** and **Pau Schulz**. The two of us first built forecasts separately, competing against each
other, then pooled our best models for this submission. Much of the engineering (data fetching, model code,
backtests and these documents) was done by AI agents (Anthropic's Claude) working under our direction; the modelling
choices and the final calls, including the pick rule and the four California overrides, were ours.

Data credits: MIT Election Data and Science Lab (CC0), FiveThirtyEight (CC BY 4.0), Wikipedia contributors (CC BY-SA),
Ballotpedia (2026 California primary results), VoteHub, the Federal Election Commission, The Downballot, Gallup and the UCSB American Presidency Project, the US
Census Bureau, Kalshi, Cook Political Report, Sabato's Crystal Ball and Inside Elections.

The code is released under the MIT License (see `LICENSE`); the data files keep their own licences, listed above
and in `docs/PROVENANCE.md`.
