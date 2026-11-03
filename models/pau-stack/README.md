# pau/: Pau's model (the no-primaries stack)

A self-contained snapshot of Pau's 2026 forecast. One command trains on 2010-2024 and writes the 2026
probabilities for the 452 races the model covers. Nothing in this folder reads the network unless you pass `--pull`.

## The model

Pau's model predicts each race's two-party margin with a two-level regression on partisan lean, incumbency, the
generic ballot, expert ratings and recency-weighted polls (polls shrunk toward the fundamentals), plus a per-cycle
district demographic pattern, trained on every contested House and Senate race 2010-2024. The pattern is fitted each
cycle on that cycle's polled House races and applied to the unpolled ones. It uses college, Hispanic and Black
shares, and three demographic directions chosen on past county swings. On top sit small additions that each passed
placebo and held-out tests:
- a Senate-specific state component;
- a state-level field shared across a state's races;
- late expert-rating moves in unpolled seats;
- a 2-point cap on how far the pattern can move a safe seat.

The error scale (Student-t, per chamber and polled/unpolled group) is learned leave-one-year-out and turned into a
win probability.

In code this is the model `node+geo+field+rchg:clip2`, built on the base model `base` (v6+nl) in `src/pk_stack.py`:

| part | code | input |
|---|---|---|
| base model: two-level margin regression, poll shrinkage, calibration (v5.1), per-cycle demographic pattern (nl) | `src/model.py`, `src/pr_pattern.py`, `src/pk_stack.py` (`pattern_k`) | `data/features_hist.csv`, `data/race/features.csv` |
| `geo`: three history-chosen demographic directions in the pattern, ridge 10 (pau-g) | `pk_stack.py` | `data/pg_features_rrrp.csv` |
| `node`: Senate state component (pau-t) | `src/pt_senate.py` | |
| `field`: state-level field over a state's unpolled House races (pau-t) | `src/pt_prop.py` | `data/pt_edges.csv` |
| `rchg`: late rating move in unpolled rated seats (pau-e) | `pk_stack.add_rchg`, `pk_2026.rchg_2026` | `data/pe_rchg_features.csv` (2010-2024), `data/stack/wiki/` (2026) |
| `clip2`: the pattern's shift is capped at ±2 points where \|rating\| ≥ 2.5 | `pk_stack.pattern_k` | |

The run also writes two other models for comparison: `stack` (the same plus House primaries, `prim_dev` from FEC
primary returns: `data/px/`) and `v6nl` (the base model alone).

## Run

```
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python run_2026.py --check-ref
```

- **What it does:**
  - rebuilds the 2026 feature table from the frozen poll pull `data/pulls/2026-11-03T0125Z` (Wikipedia, pages as
    of 01:33Z Nov 3);
  - recomputes the 2026 rating move;
  - fits the three models on 2010-2024 and writes `out/{stack_noprim,stack,v6nl}_2026.csv`, plus `summary.json`
    and `detail.csv`.
- **Output format:** `year,race_id,chamber,q_dem,q_rep,q_other,k_pD_nov1`.
  - `q_dem` is P(the D side wins): the Democrat, or the independent in R-vs-IND races such as NE-SEN.
  - `q_other` is 0.
  - `k_pD_nov1` is Kalshi's Nov 1 07:48Z price for the same side. It is used for comparison only, never as an input.
- **`--check-ref`** compares the outputs with `reference/<pull>/`, the committed outputs of the same pull.
- **Runtime:** 6.2 s wall on a laptop, measured from a clean `git archive` copy: 3 s for the feature rebuild, 2 s
  for the three model fits, the rest for imports. It needs no network: an audit hook that blocks every socket call
  lets it finish.
- **Other entry points:**
  - `--polls DIR`: another pull folder, for example `data/pulls/live_1102`, pau-f's 23:24Z Nov 2 pull.
  - `--features PATH`: a ready feature table, with no rebuild. `data/features_2026_refresh1102.csv` is the one built
    from `live_1102`.
  - `--pull --gap 10`: a fresh MediaWiki-API pull into `data/pulls/<UTC tag>/` (**network**; the fetcher refuses to
    run after 23:00Z Nov 3). The shipped pulls carry no page cache, so a pull downloads all of about 95 pages, which
    takes about 16 min at 10-s gaps.
- **Backtests:** `python run_backtests.py` re-runs the evidence files in `backtests/` in about 7 s. `--name base`
  runs v6+nl and `--name prim+node+geo+field+rchg:clip2` runs the stack with primaries.
- **Environment:**
  - Python 3 with numpy, scipy, pandas, scikit-learn, joblib and beautifulsoup4 (requests only for `--pull`).
  - Tested with Python 3.12.4 / numpy 1.26.4 / scipy 1.14.1 / pandas 2.2.3 / scikit-learn 1.5.2 (miniconda), and
    with Python 3.10.7 / numpy 1.23.5 / scipy 1.11.3 / pandas 2.2.2 / scikit-learn 1.5.1.

**Reproduction checks:**

| run | reference | Python 3.12.4 (numpy 1.26.4, scipy 1.14.1) | Python 3.10.7 (numpy 1.23.5, scipy 1.11.3) |
|---|---|---|---|
| default (`2026-11-03T0125Z`) | `reference/2026-11-03T0125Z` = `results/final/2026-11-03T0125Z` (pau/y-1103 69de6de, made under 3.12.4) | **exact** (max \|Δ\| 0) | ≤ 1.7e-7 (`--tol 1e-6`) |
| `--polls data/pulls/live_1102` | `reference/live_1102` = `results/joint2026/*_refreshed.csv` (0cce402, made under 3.10.7) | ≤ 1.7e-7 (`--tol 1e-6`) | **exact** |

- **Why the interpreters differ:** numpy/scipy builds round the optimizer differently, at the 1e-7 level. No pick
  depends on it.
- **The feature rebuild is byte-identical** under both interpreters. From `live_1102` it reproduces
  `data/features_2026_refresh1102.csv` byte for byte.

## Inputs (`data/`) and where they came from

| file | what | source |
|---|---|---|
| `features_hist.csv` | 2010-2024 race table: results, lean, incumbency, ratings, polls, generic ballot, approval | MIT Election Lab House/Senate returns 1976-2024 (the CC0 MEDSL files, fetched from a public GitHub copy that matches Dataverse's published MD5); Cook PVI and the House/Senate election and ratings pages on Wikipedia (MediaWiki API); FiveThirtyEight's poll archive (GitHub); FEC receipts (in the table, not used by the model); Gallup approval (UCSB). Built by `src/build/shared/` (fetch_history, fetch_538_history, fetch_fec, build_history), then `src/features.py` `build()` |
| `pulls/2026-11-03T0125Z/`, `pulls/live_1102/` | 2026 polls (`polls_2026.csv`) and generic ballot (`national_2026.csv`) | Wikipedia poll tables via the MediaWiki API (`src/refresh_polls_2026.py`, pau-f; pages listed with revision ids in `pages.csv`); the generic-ballot "Average" row of the Wikipedia generic-ballot page |
| `build/features/race_info_2026.csv`, `build/features/fec_2026.csv`, `borrowed/kavi-e/lean_2026.csv` | 2026 candidates, incumbency, ratings, lean (Cook PVI 2026), 2026 FEC receipts | shared data pipeline (`src/build/shared/fetch_race_info_2026.py`, `fetch_fec.py`); Kavi's lean table |
| `build/features/history_national.csv`, `build/raw/` | national history; 538 historical polls (the historical error of partisan-sponsored polls, used to de-bias them); Wikipedia CA/WA top-two pages | shared data pipeline, FiveThirtyEight archive, MediaWiki API |
| `national.csv` | national inputs by year (generic ballot, approval); the 2026 generic ballot is overwritten from the pull | pau-b |
| `race/features.csv` | district demographics (college, Hispanic, Black, ...) per race-year | Census ACS 1-year table-based summary files (`src/build/race/build_features.py`) |
| `pg_features_rrrp.csv` | pau-g's three history-chosen demographic directions per House race | MEDSL county presidential returns 2000-2016 (MEDSL's GitHub), Wikipedia county result tables, Census ACS 5-year (`src/build/pau-g/`) |
| `pt_edges.csv` | pau-t's state / district graphs | Census cartographic boundaries, CBSA delineation, ACS commuting flows (`src/build/pau-t/`) |
| `pe_rchg_features.csv`, `stack/wiki/*.wikitext` | rating change between 35 days out and election eve (2010-2024), and the 2026 rating pages as of 2026-09-29 and 2026-11-02 | Wikipedia rating pages, dated revisions via the MediaWiki API (`src/build/pau-e/`) |
| `px/primaries.csv`, `px/primaries_2026.csv` | House primary vote margins (only the `stack` variant uses them) | FEC "Federal Elections" primary results 2010-2022; 2026 primary results from the Wikipedia district pages (MediaWiki API) (`src/build/pau-x/`) |
| `kalshi/kalshi_races_2026-11-01T074812Z.csv` | Kalshi Nov 1 07:48Z prices, the benchmark | Kalshi public market data |
| `build/templates/full_congress_template.csv` | the organizers' race list (used by the fetcher) | organizers |

The scripts in `src/build/` are copies of the builders that made these tables. They are not run by `run_2026.py` and
may need the raw downloads they fetch. No CES, Harvard Dataverse download, DIME, Klarner or Wayback Machine file is
included or read. pau-v's ACS cell file is not shipped: only the unused `cells` component of `pk_stack.py` would read
it.

Two edits to third-party copies, neither of which changes any output: the press-office e-mail addresses in one
citation of the 2026 House rating pages (`data/stack/wiki/2026_house_*.wikitext`) and 538's contact address in
`data/build/raw/polls/538/senate_polls_historical.csv` are replaced by `[e-mail redacted]`.

## Validation (`backtests/`)

All numbers are out of sample and come from the same harness. `run_backtests.py` reproduces the files exactly under
Python 3.10.7, and to 1.1e-7 under Python 3.12.4.

**Leave-one-year-out, 2010-2024.** Each cycle is predicted by the model trained on the other seven.

| | LOYO log loss, races won by a D or an R (n = 3,756) | referee-style LOYO (n = 3,703) |
|---|---|---|
| **no-primaries stack** (`backtests/loyo_backtest.csv`) | **0.0619** | 0.0627 |
| stack with primaries (`backtests/stack/`) | 0.0618 | 0.0626 |
| v6+nl (`backtests/v6nl/`) | 0.0645 | 0.0654 |

- **Scoring:** natural log, q floored at 1%.
- **The first column** scores every race won by a Democrat or a Republican with the probability of the winner's party.
  Kavi's side reproduced it independently (board K025).
- **Referee-style** (pau-t's scorer) uses all races with a winner except same-party contests, with 3-way q.

**Honest backtests.** Trained on 2010..T-2 only; all 470 races.

| | 2018 log loss (538 favourite < 95%) | 2018 EV points vs 538 as the market | 2022 log loss (close) | 2022 EV points |
|---|---|---|---|---|
| **no-primaries stack** | **0.0752** (0.319) | **40.1** | **0.0663** (0.331) | **47.3** |
| stack with primaries | 0.0784 (0.334) | 33.0 | 0.0670 (0.331) | 47.1 |
| v6+nl | 0.0777 | 39.7 | 0.0699 | 44.5 |

## Limitations

- **The historical ratings leak slightly.** Where no election-eve Wikipedia revision lists a race, the 2010-2024
  ratings come from the data layer's retrospective value, which leans a little toward the eventual winner. Measured
  against the election-eve revisions, this flatters the backtests by about 0.0003-0.0006 log loss, roughly equally
  for every model.
- **The no-primaries stack was chosen after looking at the backtests.** The selection rule (six-year and pre-2018
  LOYO) kept the primaries. We dropped them after seeing the 2018/2022 backtests above, where the primaries over-read
  2018's uneven Democratic primary surge. So the backtest gap between the two stacks is not a clean held-out test.
- **18 races are not covered.**
  - These are the 9 California same-party top-two races (CA-04, 07, 11, 12, 14, 29, 34, 37, 40) and 9 seats
    without a major-party contest: unopposed FL-10, MA-02, MA-05, MA-07 and WI-02; minor-party opponent only in
    AZ-03, MA-01, NJ-08 and PA-03.
  - The joint submission uses Kavi's model for them.
- **Few House polls.** Most House races have no poll. There the forecast rests on the fundamentals, the ratings and
  the demographic pattern estimated from the polled races.
- **Training years.** Eight cycles carry the national error, and 2026 may not resemble them.
