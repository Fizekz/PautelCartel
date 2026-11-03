# Model 2: the no-primaries stack

This folder holds the second of the two models in our final blend: a margin model with five add-ons ("the stack"),
run on the poll file refreshed on Nov 2, 2026. Its 2026 output is
`results/joint2026/stack_noprim_2026_refreshed.csv`: one row per in-model race (452 races), with the probability
that the Democratic-side candidate wins. The 18 races outside the model (California top-two contests and
uncontested seats) are handled by the other model alone in the final blend.

## Method
**Base model.** A two-stage margin model, fitted on the 2010-2024 cycles.
- Stage 1 regresses each race's two-party margin on the district lean (Cook PVI), incumbency, the mean expert
  rating (Cook, Sabato, Inside Elections) and a year effect tied to the generic ballot in midterms. Rated
  competitive races get extra weight.
- Stage 2 shrinks the stage-1 margin toward the race's poll average, more strongly when more pollsters have
  polled the race.
- A Student-t calibration, fitted on leave-one-year-out predictions, turns the margin into a win probability.
  The House prediction is a 50/50 logit pool with a simpler uniform-swing model.
- A per-cycle "district pattern" estimates, from that cycle's polled House races, how far polls run from stage 1
  as a function of district demographics (college share, Hispanic share, Black share, their squares and a
  college x Hispanic term), and applies that shift to the unpolled House races.

**Add-ons.** Each was tested on its own, then added greedily and kept only if it improved held-out log loss on
both selection schemes (six held-out cycles 2010-2016, 2020, 2024; and the cycles before 2018):
1. **Safe-seat clip:** the district-pattern shift is capped at 2 points in safe seats.
2. **Senate state node:** a state's House polls, net of the pattern, say how far its Senate polls may be off.
3. **History-chosen directions:** three demographic directions, learned from past cycles' swings, added to the
   pattern with their own ridge penalty.
4. **State field:** unpolled House races borrow the poll error of polled races in the same state.
5. **Late rating move:** the change in expert ratings over the final five weeks, net of what the polls explain,
   for unpolled competitive races.

A sixth add-on, House primary vote shares, passed the selection rule but hurt the 2018 backtest badly
(it over-read 2018's uneven Democratic primary turnout). We use the version without it, which is as good on
leave-one-year-out and clearly better on the 2018 and 2022 backtests.

**Skill on years the model was not trained on** (natural-log loss per race, 1% floor):
- Leave-one-year-out over 2010-2024: 0.0627 on 3,703 races (same-party races excluded), against 0.0654 for the
  base model alone; 0.0619 on all 3,756 races won by a Democrat or a Republican.
- 2018 backtest (trained on 2010-2016 only): 0.0752; 2022 (trained on 2010-2020): 0.0663. These are scored on
  this model's own set of contested Democrat-vs-Republican races; the main README's Evaluation table uses all
  Democrat-vs-Republican races present in FiveThirtyEight's benchmark (426 and 437), so its numbers differ.
- The variant without the primaries add-on was chosen partly on its 2018 and 2022 scores, so those two backtests
  are not fully held out.

## What is in this folder
| path | what |
|---|---|
| `pbase/src/` | model code (`pk_2026.py` writes the 2026 file, `pk_proto.py` the backtests, `pk_stack.py` defines the add-ons) and the scripts that rebuild its input tables from raw sources |
| `src/to_standard.py` | converts the model's 2026 output into the standard per-race format |
| `ratings/` | expert-rating parser and the election-eve rating pages |
| `race/` | district demographics (Census ACS) used by the district pattern |
| `data/features/` | shared input tables: race history, polls, FEC money, generic ballot, 2026 race list |
| `pbase/data/refresh_1102/live_1102/` | the Nov 2 refreshed 2026 poll file (Wikipedia), generic ballot, page revisions and the 49 new polls |
| `pbase/data/borrowed/kavi-e/` | lean and historical poll tables shared with the other half of the team |
| `data/benchmark/` | the Kalshi Nov 1 07:48 UTC race file (reference column only) |
| `results/pk/pk_2026_refreshed.csv` | the model's raw 2026 output (D-side probabilities, three model variants) |
| `results/joint2026/stack_noprim_2026_refreshed.csv` | the same in the standard per-race format (input to the final blend) |
| `proto/pk_stack_noprim/out/` | backtest predictions: 2018, 2022 and leave-one-year-out 2010-2024 |
| `RESEARCH_STACK.md`, `REFRESH_2026.md` | the full study notes: greedy table, checks, placebo tests, 2026 vs the market |

## How to run
The last step runs from this folder with only pandas and numpy:

    PYTHON=/path/to/venv/bin/python bash run.sh     # PYTHON defaults to python3

It rebuilds the standard per-race file from `results/pk/pk_2026_refreshed.csv` and checks it against the
committed `results/joint2026/stack_noprim_2026_refreshed.csv` (they agree to within 1e-16; the last digit of some
values differs because of float formatting). `bash run.sh --full` re-runs the model itself.

The full model run is

    cd pbase/src
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python pk_2026.py ../work/stack/features_2026_refresh1102.csv refreshed     # 2026
    python pk_proto.py node+geo+field+rchg:clip2 pk_stack_noprim                    # backtests

It reads a working folder (`pbase/work/`) of intermediate tables: the historical and 2026 feature tables, the
district demographic directions, the state graph, the rating-move features, the primary-vote table and the
Wikipedia rating pages of Sep 29 and Nov 2. Those tables are rebuilt by scripts in `pbase/src/` from raw
downloads (Wikipedia via the MediaWiki API, MIT Election Lab returns, FEC files, Census ACS and boundary files,
MEDSL county returns). Apart from the MEDSL House and Senate files in the repository's `data/raw/mit/`, those raw
downloads are not redistributed here. `run.sh` lists the rebuild order. We did not re-run the
full chain on a second machine before the deadline; the outputs here are the ones the model produced.

## Data and rules
- No market price enters the model. The Kalshi Nov 1 price appears only as a reference column.
- No survey microdata (CES), no DIME or Klarner data and no 538 forecast is read on this model's path. MEDSL
  election returns (CC0) enter through the shared results and lean tables (`data/features/history_races.csv` and
  `pbase/data/borrowed/kavi-e/lean_*.csv`); their provenance is in the repository's `data/clean/results_*.md` and
  `lean_*.md`.
- The 2026 polls are Wikipedia poll tables at revisions before the Nov 3 23:00Z cutoff.
- Historical race and generic-ballot polls (2018-2024) come from FiveThirtyEight's public poll archives
  (CC BY 4.0), retrieved from Internet Archive snapshots after 538 shut down.

## Limitations
- 2026's national environment (generic ballot about D+8.7) is outside the range of the training cycles.
- Most House races have no polls; there the model leans on ratings, lean and the district pattern.
- Historical ratings were parsed from current Wikipedia revisions of each cycle's ratings page, a small leak.
- The model overrated long shots rated 5-35% in the 2018 and 2022 backtests.
