# Provenance

Where each part of this repository came from in the team's working repositories, what was left out, and why.
The branch `submission/official` is an orphan branch assembled from the commits below on 2026-11-02.

## Sources

| Part | Branch | Commit | Notes |
|---|---|---|---|
| Data layer (`data/`, `src/data/`, `src/scorecard.py`, `src/kalshi_snapshot.py`, `src/benchmark_fte.py`) | `kavi/data-1101` | `cd268f0` | Identical in `kavi/final-1101`. `data/raw/mit/` copied from the local download cache (MD5-checked MEDSL v15 House file; Senate file from the Dataverse file API). |
| Joint step (`src/final/`, `tests/`, `out/`, `src/pick_policy.py`, `research/pick_rule/`) | `kavi/final-1101` | `d9507fc` | The final build adds the CA-14 and CA-34 overrides and uses pool5 as the candidate frame (see `kavi/final-1101` head). |
| kavi-a | `kavi/kavi-a-1101` | `24ea95d` | |
| kavi-b | `kavi/kavi-b-1101` | `386ff04` | |
| kavi-c | `kavi/kavi-c-1101` | `8088ed9` | |
| kavi-d | `kavi/kavi-d-1101` | `52bfe2e` | |
| kavi-e | `kavi/kavi-e-1101` | `2b17169` | |
| Pau stack (`models/pau-stack/`): code, data, poll pulls, outputs, `reference/`, backtests | `pau/sub-1103` | `5bbff94` | Copied unchanged from `pau/` (`git archive`). Its 2026 file differs from the earlier `pau/y-1103` `0cce402` file only by the 01:25Z Nov 3 poll pull: six Senate probabilities move, no pick changes. |
| Joint evaluation table | `pau/q-1103` `9126a64`, `pau/j-1102` `7ba4fb1` | | The numbers in the README's Evaluation section. |

## Changes made while assembling

- Model code: hard-coded absolute paths replaced by repository-relative ones (the dossier summary in kavi-a, b, d
  and e; the raw MIT folder in kavi-d), and the default
  interpreter in the model `run_all.sh` scripts set to `python3`. Outputs re-run in this layout are byte-identical
  to the committed ones.
- `scripts/run_kavi_models.sh`, `scripts/build_pool5.sh` and the root `run_all.sh` were written for this repository.
  `models/pau-stack/` is Pau's own self-contained package, unchanged.
- Absolute local paths in five dossiers and in the model notes were rewritten as relative paths.
- Internal coordination labels were removed from code comments and model notes. Licence notes were added for the
  FiveThirtyEight and MEDSL data (`data/benchmark/README.md`, `data/raw/mit/README.md`, `data/clean/polls_hist.md`,
  `national_hist.md`).

## Left out, and why

- **kavi-f**, a sixth Kavi-side model: its 2018/2022 backtest used FiveThirtyEight's 2022 forecast as a prior, so its
  backtest skill is not comparable. None of its code, data or forecasts is used.
- **Cooperative Election Study microdata, DIME and Klarner data**, fetched on the Pau side through the Harvard
  Dataverse API behind a bot challenge: not on the final model's path and not included.
- **Wayback copies of FiveThirtyEight forecast files** used by other Pau-side studies (2020 forecasts, 2024 poll
  files) and the Pau-side 538 forecast history: not on the final path and not included.
- The nine Ballotpedia pages behind kavi-a's `ca_primaries_2026.csv` (fetched 2026-11-01): the parsed CSV is
  included, the raw HTML is not.
- Superseded drafts (pick files without the CA-40/CA-07 override), the earlier 07:45 UTC Kalshi snapshot, model
  diagnostics and experiments, the raw download cache (`data/raw/` except `mit/`), internal coordination files and
  the organizers' slides.
- From `research/pick_rule/`, only the summary and its printed outputs are included. The scripts read a table of past
  market prices and 538 probabilities and model pools that include kavi-f.

## Archived and API-fetched data that is included

- `data/clean/polls_hist.csv` and `national_hist.csv` (2018-2024 rows), `data/benchmark/fte_final_2018/2022.csv`,
  the FiveThirtyEight files under `models/pau-stack/data/build/` and `models/pau-stack/data/borrowed/kavi-e/`: FiveThirtyEight
  data (CC BY 4.0) retrieved from Internet Archive snapshots. The historical polls are training inputs of every model;
  the 538 finals are used only for backtest scoring and 2018/2022 candidate lists.
- `models/kavi-a/data/kavi-a/raw/` and `models/kavi-b/data/kavi-b/raw/`: Internet Archive copies of FEC bulk files
  (2018/2022 pre-election cuts, backtests only). FEC bulk data is a US government work in the public domain; the
  archived copies are used only because the FEC publishes current files, not the pre-election cuts.
- `data/clean/results_senate.csv`, `data/raw/mit/1976-2024-senate-state.csv` and the MEDSL presidential file behind
  `lean_*.csv`: fetched from the Harvard Dataverse file API (CC0, no guestbook).

The team reviewed these and kept them: each licence allows redistribution with attribution, the sources are credited
here and in the data notes, and removing them would break reproduction.

## How it was built

Most of the code, data fetching, backtests and documentation were written by AI agents (Anthropic's Claude), run by
Kavi and Pau, who set the direction and made the final calls.
