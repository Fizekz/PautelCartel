# Pick rule: B vs C vs C' (Kavi side, 2026-11-02)

How to turn our win probabilities `q` into one pick per race, under the contest score: a correct pick earns
`-log2(max(p, 0.01))`, with `p` = Kalshi's Nov 1 price.

- **A (chalk):** the Kalshi favourite everywhere.
- **B:** the underdog where our `q >= 0.40`.
- **C:** maximise expected points `q * -log2(p)` in each race.
- **C' (chosen by Kavi):** C, but an underdog is eligible only where our `q >=` its Kalshi price.
  `src/final/make_submission.py --rule Cprime` (the default).

## Scripts (run with the repo venv from this directory; outputs in `outputs/`)

| Script | What it does |
|---|---|
| `compare2.py` | Plays every rule on the same 20k simulated elections (national swing, rho 0.2) under 8 "truths": our pools, Pau's v5 pair, Kalshi as priced, and Kalshi corrected for the favourite-longshot bias (FLB, slope b 1.3 / 1.5 / 1.77). |
| `analyze.py` | The 24 races where C and C' differ: break-even `q` and FLB slope per race, expected gain under each truth, the simulated distribution of C minus C', and long-shot calibration of our clean pool in 2018/2022. |
| `hist.py` | History. It uses 538 as the model and real market prices for scoring (PredictIt 2018 T-7, Kalshi 2024 T-2, Polymarket 2024 T-2), against actual outcomes. |

Inputs:
- `cvb/`: log-linear pools of the model outputs. pool6 = kavi-a..f v3, pool5 = without kavi-f, pauv5 = Pau's v5 pair, all8v5 = all eight.
- `cc/past_race_probs.csv`: copied from `pau/market-1101` (`data/market_history/`, disclosed in D1).

## Findings

**Expected gain of C over C' (sum over the 24 swaps):**

| Truth | C minus C' |
|---|---|
| Our 6 models | +2.3 |
| Pau v5 | +1.7 |
| Kalshi as priced | +4.2 |
| FLB slope 1.5 | -0.6 |
| FLB slope 1.77 | -2.0 |

- The swaps break even at an FLB slope of b = 1.41. Pau's fit on 193 past market races is b = 1.77 ± 0.31.
- **History (`hist.py`).** We call a long shot "C-only" when C takes it and C' does not: positive expected value, but the model rates it below the market.
  - C-only long shots won **0 of 37**, where the markets expected about 8.
  - C' beat C in all three markets: 40.4 vs 35.5, 24.5 vs 19.4 and 18.2 vs 14.8.
  - The underdogs C' kept won 30 of 63, against 24.3 expected by 538.
- **Calibration of our clean pool (2018/2022).** Long shots won about half as often as we said. Republican long shots rated 5-35% won 3 of 46. 19 of the 24 C-only dogs in 2026 are Republicans.
- **Caveat.** Each replay is one correlated cycle, and 2018-2024 had fewer upsets than any forecaster expected.
