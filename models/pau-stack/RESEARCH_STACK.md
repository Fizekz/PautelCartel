# pau-k: do the 1102 research gains stack? (2026-11-02)

**The question.** Twelve studies each found a small gain over v6+nl, each one placebo-checked on its own. Do they add
up when combined, or do they all read the same information? I tested this with a greedy stack: one harness, every
component as a switch, each added in order of its standalone gain and kept only if the stack improves on **both six
and pre18**. Whether any model adopts the result is Pau's call.

**Bottom line.**
- **They mostly stack.** The final stack has six components: House primaries, a safe-seat clip, the Senate state node,
  history-chosen directions, the state-graph field and the late rating move. It reaches six **0.0648**, pre18
  **0.0659**, bt **0.0785**, loyo8 **0.0678**, against v6+nl's 0.0690 / 0.0698 / 0.0797 / 0.0707. That is −0.0042 /
  −0.0039 on the selection schemes. The kept components' standalone gains add to −0.0054 / −0.0045, so the stack
  keeps 78% / 87% of them.
  - **Referee-style LOYO: 0.0654 → 0.0626** (pau-t's scorer, which gives the referee's 0.0654 for v6+nl). Every
    cycle improves except 2018 (0.0813 → 0.0823).
  - Leave-one-out inside the final stack: removing any single component hurts on both six and pre18.
- **Two components fail the stack:**
  - **pau-w's Kalman pool**, +0.0001 / +0.0002 (and +0.0004 / +0.0006 when re-tested on the final stack);
  - **pau-v's CES-free voter weighting**, flat turnout +0.0002 / +0.0004 and ecological turnout +0.0006 / +0.0020.
    It replaces the nl pattern rather than adding to it, and it loses once the directions and primaries are in.
- **The safe-seat clip is the surprise.** Capping the applied pattern shift at ±2 points in safe seats (|rating| ≥ 2.5)
  is worth −0.0012 / −0.0008 alone, nearly as much as the Senate node. It recovers the 2.9 nats v6+nl loses in safe
  seats (six: 6.7 → 3.8 nats; v5.1 had 3.9). Any cap from 0 to 5 points does about as well.
- **Overfitting checks:**
  - The in-sample vs held-out gap grows from 0.0018 to **0.0021**: loyo8 0.0678, in-sample 0.0657.
  - **Joint placebo:** every added signal shuffled within year, 3 seeds, with the clip kept. Six 0.0677-0.0687 and
    pre18 0.0682-0.0708, against 0.0678 / 0.0690 for the clip alone. The real stack (0.0648 / 0.0659) is beyond
    every seed.
- **Caveat: the primaries fail 2018 again.** With them, bt 2018 is 0.0858 vs 0.0823 without; 2018 contest-replay EV
  points are 33.0 vs 40.1 (v6+nl 39.7).
  - **The no-primaries stack** (the other five components) scores six 0.0659 / pre18 0.0672 / **bt 0.0764** / loyo8
    0.0679, with referee LOYO **0.0627**: as good as the full stack on LOYO and clearly better on bt.
  - The selection rule (six and pre18) keeps the primaries. Choosing between the two stacks is a judgment call on
    how much 2018 should count, and it is Pau's.
- **2026, current poll file:**
  - The stack barely moves the House: 218.9 expected D-side seats, against 218.3 for v6+nl and 229.4 for Kalshi on
    Nov 1.
  - Senate D-side wins: 19.2, against 19.1 for v6+nl and 18.4 for Kalshi.
  - It disagrees with Kalshi's favourite in the same 5 races as v6+nl: IA-SEN, CO-05, MI-04, MI-10 and WI-01.
  - Biggest moves: IA-03, WI-03 and AZ-06 from about 0.80 to 0.92-0.95, from the late rating move (Kalshi
    0.85-0.90). The Senate node makes AK, OH, ME and MI more D (further from Kalshi) and IA less D (toward Kalshi).
  - **With pau-f's refreshed polls** (committed 2ae49b3, 49 new polls): House 219.0 (v6+nl 218.5), Senate D-side
    19.4 (v6+nl 19.3). The same 5 disagreements remain; IA-SEN drops to 0.53 (Kalshi 0.44). See section 6.

**Conventions.**
- Log loss (LL) with the 1% floor, contested D-side vs R races (`train_ok == 1`), from v5.1's harness.
- **Selection only on six and pre18.** bt (2018/2022) and loyo8 are reported, never chosen on.
- Subsets: "rated" = |rating| < 2.5; "unp" = unpolled rated House races (n 75 / 44 / 30 / 105); "Senate" = Senate
  rows.
- Every number is held out (test cycle never in training) unless it says in-sample.
- Base v6+nl: `PE.all_(PE.V["v51"], pr_pat_lam=1.0, pr_pat_set="nl", pr_pat_unp=True)`. Re-run here: 0.0690 /
  0.0698 / 0.0797 / 0.0707, Senate LL 0.128 / 0.119 / 0.106 / 0.123. That matches every study.

**Code** (`pbase/src/`):
- `pk_stack.py`: the harness. All eight components are cfg switches (`build(comps, clip=...)`), and the joint
  placebo is `placebo(seed, comps)`.
- `pk_run.py NAME`: names like `prim+node+geo+field+rchg:clip2`.
- `pk_kf.py`: caches pau-w's filter forecasts per fold.
- `pk_proto.py`: standard outputs and the referee-style LOYO.
- `pk_2026.py`: 2026 vs Kalshi.
- `pk_table.py`: the tables below.
- Copied unchanged from the study branches: `pt_prop.py`, `pt_senate.py` (pau-t); `pw_assim.py`, `pw_run.py` (pau-w);
  `px_primaries.py` (pau-x, provenance only).
- Results: `results/pk/*.json`. Predictions: `pbase/work/stack/` (gitignored).
- Standard outputs: `proto/pk_stack/out` (final stack) and `proto/pk_stack_noprim/out`.

**Data** (all local copies, nothing downloaded):
- pau-t's `pt_edges.csv`;
- pau-e's `pe_rchg_features.csv` and its 2026 rating revisions (`work/stack/wiki`);
- pau-g's `pg_features_rrrp.csv`;
- pau-v's ACS cells (`race/cells_v.csv`, committed on pau/v-1102; ACS only);
- pau-x's `primaries.csv` and `primaries_2026.csv`.
- **No CES file is read:** `pbase/data/raw_v` and `results/pv_ces_cells.csv` are not used.
- No Kalshi price later than Nov 1 07:48Z.

## 1. The components as implemented, and their standalone reproduction
The hooks chain on `WinModel._mu` (inner → outer): shrink ← pattern ← pau-x add-on ← pau-t field ← pau-t Senate node.
That is the order each study used relative to the v6+nl pattern.
- **Pattern** (`pattern_k`): the v6+nl per-cycle ridge, generalised. It optionally adds pau-g's 3 directions with
  their own ridge 10, swaps the design to pau-v's cells, and clips the applied shift in safe seats.
- **pau-e's rating move:** a stage-1 column (`pr_extra`).
- **pau-w's pool:** post-hoc, on the top-level model's predictions.

| component (study) | form | six | pre18 | bt | loyo8 | study's own numbers |
|---|---|---|---|---|---|---|
| v6+nl | | 0.0690 | 0.0698 | 0.0797 | 0.0707 | 0.0690 / 0.0698 / 0.0797 / 0.0707 |
| Senate state node (pau-t) | h_S, shrink 2, Senate model | 0.0683 | 0.0690 | 0.0780 | 0.0701 | same |
| state-graph field (pau-t) | state graph, β 5, snr 0.3, kernel A, unpolled House | 0.0686 | 0.0694 | 0.0797 | 0.0705 | same |
| node + field (pau-t) | | 0.0678 | 0.0686 | 0.0780 | 0.0698 | 0.0678 / 0.0686 / 0.0780 / 0.0699 |
| late rating move (pau-e) | `rchg_unp`, stage-1 column | 0.0685 | 0.0696 | 0.0789 | 0.0699 | same |
| history-chosen directions (pau-g) | g1-g3 in the pattern, ridge 10 | 0.0686 | 0.0693 | 0.0791 | 0.0703 | same |
| voter weighting, CES-free (pau-v) | citizen cells × flat turnout, share + squares, ridge 10 | 0.0686 | 0.0695 | 0.0796 | 0.0704 | same (`sq_c_flat_l10`) |
| same, ecological turnout | pau-v's profile from past district totals; **flat** where pau-v fell back to CES (2012) | 0.0682 | 0.0698 | 0.0793 | 0.0700 | 0.0682 / 0.0697 / 0.0794 / 0.0700 (with the CES fallback) |
| House primaries (pau-x) | `prim_dev`, House, ridge 10 | 0.0670 | 0.0680 | 0.0813 | 0.0702 | same |
| same, within-state centring | `prim_dev_s` | 0.0676 | 0.0693 | 0.0794 | 0.0702 | 0.0676 / 0.0693 / 0.0794 / 0.0701 |
| Kalman pool (pau-w) | filter c2_tau30, logit pool w = 0.3 | 0.0683 | 0.0695 | 0.0811 | 0.0701 | same |
| **safe-seat clip** (new) | \|applied pattern shift\| ≤ c in \|rating\| ≥ 2.5 | see below | | | | |

**Safe-seat clip, alone on v6+nl:**

| c (points) | six | pre18 | bt | loyo8 |
|---|---|---|---|---|
| none (v6+nl) | 0.0690 | 0.0698 | 0.0797 | 0.0707 |
| 0 (no pattern in safe seats) | 0.0678 | 0.0691 | 0.0795 | 0.0698 |
| 1 / 2 / 5 | 0.0678 / 0.0678 / 0.0678 | 0.0690 / 0.0690 / 0.0690 | 0.0795 | 0.0698 |
| 8 | 0.0678 | 0.0692 | 0.0796 | 0.0699 |

- **pau-x had no contest adjustment** (it is listed as a next step in RESEARCH_ANGLES.md). Its robustness variant is
  the within-state centring `prim_dev_s`, which I test in the final stack (section 3).
- **The clip is a plateau.** Any c from 0 to 5 is within 0.00003 on six + pre18. In the stack, c = 2 was best on six +
  pre18 (step 2) and is kept fixed after that.

## 2. Greedy table
**Order:** by standalone six + pre18 gain. Primaries −0.0038, clip −0.0020, node −0.0016, pool −0.0010, directions
−0.0009, cells −0.0008 (eco) / −0.0007 (flat), field −0.0008 (the brief pairs it with the node, which comes earlier),
rating move −0.0007. A component is kept only if the stack improves on **both** six and pre18. The Δ column compares
each step with the stack before it.

| step | six | pre18 | bt | loyo8 | Δ six / pre18 | decision |
|---|---|---|---|---|---|---|
| v6+nl | 0.0690 | 0.0698 | 0.0797 | 0.0707 | | reference |
| 1. + primaries (`prim_dev`) | 0.0670 | 0.0680 | 0.0813 | 0.0702 | −0.0020 / −0.0018 | keep |
| 2. + safe-seat clip, c = 2 | 0.0664 | 0.0677 | 0.0814 | 0.0696 | −0.0006 / −0.0003 | keep (six + pre18: c 0 .13415, 1 .13414, **2 .13409**, 5 .13418) |
| 3. + Senate state node | 0.0657 | 0.0669 | 0.0797 | 0.0689 | −0.0007 / −0.0008 | keep |
| 4. + Kalman pool, w = 0.3 | 0.0658 | 0.0671 | 0.0810 | 0.0688 | +0.0001 / +0.0002 | **reject** |
| 5. + history-chosen directions | 0.0654 | 0.0666 | 0.0794 | 0.0687 | −0.0002 / −0.0003 | keep |
| 6a. + cells, flat turnout (replaces the nl pattern) | 0.0657 | 0.0670 | 0.0785 | 0.0686 | +0.0002 / +0.0004 | **reject** |
| 6b. + cells, ecological turnout | 0.0660 | 0.0686 | 0.0783 | 0.0687 | +0.0006 / +0.0020 | **reject** |
| 7. + state-graph field | 0.0651 | 0.0662 | 0.0796 | 0.0685 | −0.0004 / −0.0004 | keep |
| 8. + late rating move | **0.0648** | **0.0659** | **0.0785** | **0.0678** | −0.0003 / −0.0003 | keep: **final stack** |

**Subsets per step** (rated / unp / Senate LL; the full set is in `results/pk/*.json`):

| | six | pre18 | bt | loyo8 |
|---|---|---|---|---|
| v6+nl | 0.306 / 0.146 / 0.128 | 0.305 / 0.171 / 0.119 | 0.300 / 0.119 / 0.106 | 0.301 / 0.134 / 0.123 |
| 1. primaries | 0.302 / 0.144 / 0.129 | 0.301 / 0.172 / 0.119 | 0.307 / 0.106 / 0.107 | 0.301 / 0.129 / 0.123 |
| 3. + clip + node | 0.298 / 0.142 / 0.120 | 0.297 / 0.171 / 0.109 | 0.301 / 0.105 / 0.085 | 0.298 / 0.127 / 0.115 |
| 5. + directions | 0.297 / 0.140 / 0.120 | 0.296 / 0.169 / 0.109 | 0.300 / 0.100 / 0.085 | 0.297 / 0.125 / 0.115 |
| 7. + field | 0.296 / 0.139 / 0.120 | 0.295 / 0.165 / 0.109 | 0.301 / 0.102 / 0.085 | 0.296 / 0.125 / 0.115 |
| **8. final stack** | **0.294 / 0.136 / 0.116** | **0.293 / 0.153 / 0.109** | **0.296 / 0.077 / 0.085** | **0.293 / 0.115 / 0.111** |
| final, no primaries | 0.297 / 0.133 / 0.116 | 0.298 / 0.150 / 0.108 | 0.288 / 0.087 / 0.085 | 0.292 / 0.116 / 0.111 |

**Where the nats go** (six, sum of −ln p):

| group (n) | v6+nl | v6+nl + clip | final stack |
|---|---|---|---|
| House rated, polled (384) | 131.1 | 131.2 | 127.5 |
| House rated, unpolled (75) | 11.0 | 10.7 | 10.2 |
| House safe (1,874) | 6.7 | 3.8 | 2.7 |
| Senate (207) | 26.6 | 26.6 | 24.0 |

**Reading:**
- **The gains are mostly independent.** Each touches a different part of the races:
  - the primaries: rated polled House;
  - the clip: safe seats;
  - the node: Senate;
  - the field and the rating move: unpolled House;
  - the directions: the pattern.
- **The clip and the primaries overlap.** The clip's gain halves on top of the primaries (−0.0006 / −0.0003 vs
  −0.0012 / −0.0008). The primary feature moves yhat in safe seats and so changes the pattern there.
- **The pool double-counts.** The filter is a second reading of the same polls, and once the node and the
  primaries are in, it adds nothing. It is also the component that hurts bt most (+0.0013 at step 4).
- **pau-v's cells do not combine with the directions.** On their own they match the nl pattern plus citizen weighting.
  As a replacement under the stack they lose six and pre18 but gain bt (0.0785 vs 0.0794) and loyo8. I followed the
  selection rule.

## 3. The final stack: checks
**Leave-one-out and second pass** (each row is the final stack with one change; Δ against the final stack):

| variant | six | pre18 | bt | loyo8 | Δ six / pre18 |
|---|---|---|---|---|---|
| **final stack** (prim + clip 2 + node + directions + field + rating move) | **0.0648** | **0.0659** | **0.0785** | **0.0678** | |
| without primaries | 0.0659 | 0.0672 | **0.0764** | 0.0679 | +0.0012 / +0.0013 |
| without clip | 0.0652 | 0.0663 | 0.0782 | 0.0682 | +0.0005 / +0.0004 |
| without Senate node | 0.0655 | 0.0667 | 0.0803 | 0.0684 | +0.0007 / +0.0008 |
| without directions | 0.0650 | 0.0663 | 0.0790 | 0.0680 | +0.0002 / +0.0004 |
| without field | 0.0652 | 0.0664 | 0.0783 | 0.0679 | +0.0004 / +0.0005 |
| without rating move | 0.0651 | 0.0662 | 0.0796 | 0.0685 | +0.0003 / +0.0003 |
| `prim_dev_s` instead of `prim_dev` | 0.0652 | 0.0670 | 0.0764 | **0.0676** | +0.0005 / +0.0011 |
| + Kalman pool | 0.0651 | 0.0665 | 0.0800 | 0.0679 | +0.0004 / +0.0006 |
| + cells, flat | 0.0647 | 0.0663 | 0.0776 | 0.0675 | −0.0000 / +0.0004 |
| + cells, ecological | 0.0648 | 0.0679 | 0.0772 | 0.0674 | +0.0000 / +0.0020 |
| clip 1 / clip 5 | 0.0648 / 0.0648 | 0.0659 / 0.0660 | 0.0785 / 0.0785 | 0.0678 / 0.0678 | 0 / ≤ +0.0001 |

- **Every kept component still pays in the final stack.** Neither rejected component passes on a second try.
- **The bt-robust choices are the primary variants.** Dropping the primaries, or centring them within state, gives
  the best bt (0.0764) and loyo8 (0.0676-0.0679). On the selection schemes they lose 0.0005-0.0013.

**Overfitting gap** (in-sample = all 8 cycles in training, each cycle predicted):

| | in-sample | loyo8 | gap |
|---|---|---|---|
| v6+nl | 0.0689 | 0.0707 | 0.0018 |
| final stack | 0.0657 | 0.0678 | **0.0021** |

The gap grows by 0.0003. The stack adds 4 outcome-fitted coefficients per fold (the primary k, the node κ_S, the
field κ and the rating-move slope) plus the clip. The 3 direction loadings are fitted on each cycle's polls, not on
outcomes.

**Joint placebo.** Every added signal is shuffled within year, each the way its own study shuffled it:
- primaries and directions: permuted across districts;
- rating move: permuted among rated races;
- Senate node: h_S permuted across states (`pt_senh_shuffle`);
- field: state graph rewired, degree-preserving (`pt_rewire`).

The clip is kept, since it has no signal to shuffle.

| | six | pre18 | bt | loyo8 |
|---|---|---|---|---|
| v6+nl + clip 2 (what the placebo should fall back to) | 0.0678 | 0.0690 | 0.0795 | 0.0698 |
| placebo seed 1 / 2 / 3 | 0.0687 / 0.0686 / 0.0677 | 0.0708 / 0.0682 / 0.0684 | 0.0813 / 0.0807 / 0.0803 | 0.0703 / 0.0707 / 0.0698 |
| **real** | **0.0648** | **0.0659** | **0.0785** | **0.0678** |

- The real stack beats every placebo seed by 0.0029-0.0039 on six and 0.0023-0.0049 on pre18.
- Two pre18 seeds land below the clip-only level (0.0682-0.0684 vs 0.0690). This is the non-state-specific part of
  the node's pre18 gain that pau-t already saw in its shuffles.

**Per year** (loyo8 / bt):

| | 2010 | 2012 | 2014 | 2016 | 2018 | 2020 | 2022 | 2024 | bt 2018 | bt 2022 |
|---|---|---|---|---|---|---|---|---|---|---|
| v6+nl | 0.0833 | 0.0825 | 0.0663 | 0.0378 | 0.0879 | 0.0870 | 0.0750 | 0.0434 | 0.0850 | 0.0746 |
| final stack | 0.0821 | 0.0811 | 0.0567 | 0.0354 | **0.0889** | 0.0800 | 0.0723 | 0.0426 | **0.0858** | 0.0714 |
| no primaries | 0.0835 | 0.0821 | 0.0575 | 0.0359 | 0.0863 | 0.0807 | 0.0712 | 0.0425 | 0.0823 | 0.0707 |
| `prim_dev_s` | 0.0827 | 0.0814 | 0.0574 | 0.0356 | 0.0870 | 0.0802 | 0.0712 | 0.0425 | 0.0826 | 0.0704 |

The final stack is worse than v6+nl only in 2018. The primaries cause it, as in pau-x, where 2018's uneven
Democratic primary surge was over-read.

**Standard outputs and referee-style LOYO** (`proto/pk_stack/out`, `proto/pk_stack_noprim/out`).
- I found no referee scorer code; `.git/wt/referee` holds outputs only. I used pau-t's `referee_loyo`: 3-way q, all
  races with a winner except same-party contests, 1% floor. It reproduces the referee's 0.0654 for
  `proto/pattern_nl/out`.

| | LOYO all | 2010 | 2012 | 2014 | 2016 | 2018 | 2020 | 2022 | 2024 |
|---|---|---|---|---|---|---|---|---|---|
| v6+nl (`proto/pattern_nl`) | 0.0654 | 0.0783 | 0.0762 | 0.0570 | 0.0340 | 0.0813 | 0.0834 | 0.0711 | 0.0409 |
| **final stack** (`proto/pk_stack`) | **0.0626** | 0.0771 | 0.0749 | 0.0488 | 0.0318 | 0.0823 | 0.0767 | 0.0685 | 0.0401 |
| no primaries (`proto/pk_stack_noprim`) | 0.0627 | 0.0784 | 0.0758 | 0.0495 | 0.0323 | 0.0799 | 0.0774 | 0.0675 | 0.0400 |

**Backtests** (`backtest.run_year`, all 470 races; close = 538 favourite < 0.95; replay against 538 as the market):

| | 2018 LL (close) | 2018 EV points | 2022 LL (close) | 2022 EV points |
|---|---|---|---|---|
| v6+nl (pau-r's numbers) | 0.0787 | 39.7 | 0.0709 | 44.5 |
| v6+nl + node (pau-t) | 0.0755 (0.3198) | 39.5 | 0.0689 (0.3436) | 46.6 |
| final stack | 0.0784 (0.3336) | **33.0** | **0.0670** (0.3311) | 47.1 |
| no primaries | **0.0752** (0.3193) | **40.1** | **0.0663** (0.3307) | **47.3** |

## 4. 2026 (current poll file, trained on 2010-2024)
**Sources:**
- `pbase/work/features_2026.csv`: the poll inputs every study used. pau-f's refresh is in section 6.
- Kalshi Nov 1: `.git/wt/market/data/kalshi/kalshi_races_2026-11-01T074812Z.csv`, the earliest Nov 1 snapshot.
  P(D-side) = `fav_norm`, or 1 − it where the favourite is R.
- The 2026 rating move is rebuilt from pau-e's revisions. It matches `pe_rchg_2026.csv` to 1e-16.
- Output: `results/pk/pk_2026_current.{csv,json}`.

| | expected D-side House wins | expected D-side Senate wins | favourite ≠ Kalshi | mean \|p − Kalshi\|, House | EV picks against Kalshi's favourite |
|---|---|---|---|---|---|
| Kalshi Nov 1 | 229.4 | 18.4 | | | |
| v6+nl | 218.3 | 19.1 | IA-SEN, CO-05, MI-04, MI-10, WI-01 | 0.0528 | 35 |
| **final stack** | **218.9** | **19.2** | IA-SEN, CO-05, MI-04, MI-10, WI-01 | 0.0533 | 33 |
| no primaries | 218.8 | 19.2 | the same 5 | 0.0530 | 34 |

**Favourite disagreements** (P(D): v6+nl → stack (Kalshi)):
- IA-SEN 0.66 → 0.59 (0.44);
- CO-05 0.52 → 0.51 (0.45);
- MI-04 0.49 → 0.49 (0.62);
- MI-10 0.42 → 0.47 (0.76);
- WI-01 0.30 → 0.34 (0.50).

The stack moves toward Kalshi in IA-SEN, MI-10 and WI-01, but no favourite flips.

**Biggest moves, House:**

| race | rating | polled | v6+nl | final stack | no primaries | Kalshi | driver |
|---|---|---|---|---|---|---|---|
| IA-03 | +0.83 | no | 0.80 | 0.95 | 0.95 | 0.85 | rating move +0.59 |
| WI-03 | +0.83 | no | 0.80 | 0.95 | 0.93 | 0.89 | rating move +0.59; primary +18 |
| AZ-06 | +0.83 | no | 0.82 | 0.92 | 0.94 | 0.90 | rating move +0.59; primary −11 |
| NV-02 | −0.83 | no | 0.19 | 0.11 | 0.14 | 0.40 | primary −20 |
| WI-01 | −1.00 | no | 0.30 | 0.34 | 0.32 | 0.50 | primary +8 |
| MI-10 | −0.17 | yes | 0.42 | 0.47 | 0.43 | 0.76 | primary +16 |
| TX-15 | +0.50 | yes | 0.79 | 0.83 | 0.79 | 0.89 | primary +28 |
| GA-12 | −3.00 | no | 0.04 | 0.007 | 0.008 | 0.15 | clip / field |
| FL-22 | 0.00 | no | 0.58 | 0.54 | 0.59 | 0.72 | primary −16 |

**Biggest moves, Senate** (all from the node; the primaries do not touch the Senate):

| race | v6+nl | final stack | Kalshi |
|---|---|---|---|
| AK | 0.73 | 0.84 | 0.71 |
| OH | 0.80 | 0.87 | 0.71 |
| IA | 0.66 | 0.59 | 0.44 |
| ME | 0.82 | 0.87 | 0.69 |
| MI | 0.92 | 0.97 | 0.76 |
| NE | 0.26 | 0.22 | 0.26 |
| KS | 0.79 | 0.82 | 0.51 |
| TX | 0.74 | 0.71 | 0.69 |

**Reading:**
- **The House total barely moves** (+0.6 seats), so the gap to Kalshi (about 11 seats) stays. It is the
  national-level question pau-p and pau-r describe, which none of these components addresses.
- **The moves are concentrated:** 9 House races move by more than 0.03; the mean House |Δp| is 0.003.
- **The rating move pulls the three college-heavy unpolled seats back toward Kalshi** (IA-03, WI-03, AZ-06),
  undoing the nl pattern's R pull there. In IA-03 it overshoots (0.95 vs 0.85).
- **The Senate node pushes AK, OH, ME and MI further from Kalshi** and IA toward it. It reads each state's House
  polls as a measure of its Senate poll error. pau-d found Kalshi moved after Nov 1 on new polls in IA, KS, ME and
  TX, so the refreshed poll file matters most for these Senate rows.

## 5. Caveats
- **Greedy order.** The order follows the standalone gains, and near-ties (cells / field / rating move at
  0.0007-0.0008) could have gone either way. The second pass (section 3) shows the result does not depend on it:
  every kept component still pays at the end, and neither rejected one does.
- **The rating move's unpolled restriction was chosen after a look** (pau-e). In the stack it adds −0.0003 / −0.0003
  and −0.0011 on bt.
- **The clip value was chosen on six and pre18,** but the clip itself came from a diagnosis on six (pau-r's
  safe-seat nats). It is one parameter on a flat plateau.
- **pau-v is CES-free here.** The CES-based variant (citizen × CES validated turnout) was not run, pending Pau's
  ruling. pau-v's own numbers for it (0.0685 / 0.0695) are a little better than the flat version, but within the
  margin by which the cells lose in the stack.
- **The primaries have no 2024 feature** (no FEC 2024 primary file). 2024 is scored without the feature.
- **The overfitting gap grows to 0.0021,** and the bt numbers favour the stack without primaries.

## 6. 2026 with pau-f's refreshed polls
**Inputs:**
- pau-f committed its refresh at 2ae49b3: 49 new polls, 44 of them Senate polls ending Oct 25 or later, pages dated
  before 23:25Z Nov 2, and the refreshed generic ballot.
- I used its deterministic rebuild of the features, `.git/wt/pau-f/pbase/data/refresh_1102/model_1102/features_2026_new.csv`
  (read-only, copied to `pbase/work/stack/features_2026_refresh1102.csv`). pau-f checked that the old rebuild equals
  `work/features_2026.csv` to 1e-9.
- The rating move's poll-movement regression is recomputed against it. The three unpolled seats' `rchg_unp` goes
  from 0.59 to 0.58.
- Kalshi stays at the Nov 1 07:48Z price.
- Output: `results/pk/pk_2026_refreshed.{csv,json}`.

| | expected D-side House wins | expected D-side Senate wins | favourite ≠ Kalshi | mean \|p − Kalshi\|, House | EV picks against Kalshi's favourite |
|---|---|---|---|---|---|
| Kalshi Nov 1 | 229.4 | 18.4 | | | |
| v6+nl, refreshed | 218.5 | 19.3 | IA-SEN, CO-05, MI-04, MI-10, WI-01 | 0.0526 | 34 |
| **final stack, refreshed** | **219.0** | **19.4** | the same 5 | 0.0533 | 31 |
| no primaries, refreshed | 218.9 | 19.4 | the same 5 | 0.0531 | 33 |

**What the refresh moves in the final stack** (P(D), current → refreshed; Kalshi Nov 1):

| race | v6+nl current → refreshed | final stack current → refreshed | Kalshi |
|---|---|---|---|
| NE-SEN | 0.26 → 0.37 | 0.22 → 0.34 | 0.26 |
| IA-SEN | 0.66 → 0.61 | 0.59 → **0.53** | 0.44 |
| ME-SEN | 0.82 → 0.89 | 0.87 → 0.93 | 0.69 |
| TX-SEN | 0.74 → 0.78 | 0.71 → 0.76 | 0.69 |
| OH-SEN | 0.80 → 0.78 | 0.87 → 0.85 | 0.71 |
| KS-SEN | 0.79 → 0.80 | 0.82 → 0.83 | 0.51 |
| AK-SEN | 0.73 → 0.74 | 0.84 → 0.86 | 0.71 |
| AZ-06 | 0.82 → 0.85 | 0.92 → 0.94 | 0.90 |

- **The refresh and the stack act on different races.** The refresh moves the Senate through the new polls. The
  stack's House moves (IA-03, WI-03, AZ-06, NV-02) are the same before and after it.
- **IA-SEN is the closest call:** with both the refresh and the Senate node it is 0.53, the nearest of the five
  disagreements to flipping toward Kalshi.

**Rerun:** set `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, then
`cd pbase/src && python3 pk_2026.py ../work/stack/features_2026_refresh1102.csv refreshed`.
