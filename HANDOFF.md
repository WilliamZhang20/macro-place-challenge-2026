# DREAMPlace Pipeline Placer Handoff

This document describes the current end-to-end strategy behind
`submissions/dreamplace_pipeline_placer.py` and
`submissions/_dreamplace_pipeline.py` as of this checkout.

## Entry Point

`DreamplacePipelinePlacer` is the evaluator-facing wrapper. Its `place()` method
delegates to `DreamPlacePipeline.place()`.

Configuration comes from either:

- default construction: `DreamPlacePipeline(rich_candidate_set=rich)`;
- tuner JSON: `MACRO_PLACE_DP_CONFIG` points to a JSON file with optional
  `pipeline` and `dreamplace_json_overrides` keys.

`MACRO_PLACE_DP_RICH_CANDIDATES` controls the rich DREAMPlace portfolio and
defaults to enabled. A tuner JSON value for `pipeline.rich_candidate_set`
overrides the environment flag.

The wrapper accepts tuner keys for the current DP, rescue, and polish controls:
pre-DP valid-start discovery, explicit DP output legalization, RePlAce rescue,
coordinate descent, GWTW SA, and experimental hyperband.

## Current Strategy

The pipeline is:

1. Load the `.plc` handoff and DREAMPlace install.
2. Build a legalized `.plc` reference placement as a true-proxy guardrail.
3. Build diverse DREAMPlace starts, optionally appending valid pre-DP starts
   discovered by legalization plus proxy scoring.
4. Run multi-start DREAMPlace with feature-aware start caps and rich variants.
5. Salvage low-valid-yield DREAMPlace runs with conservative template reruns.
6. Select valid candidates by the true evaluator proxy.
7. Run a short post-DREAMPlace SA polish on the top valid current candidate by
   default.
8. Run RePlAce rescue from the original seed and top true-proxy DP outputs.
9. Run post-rescue coordinate descent and multiprocessing GWTW SA.
10. Return the best valid placement by true evaluator proxy.

Every adoption step is gated by `compute_proxy_cost()` through `score_placement()`
or `select_best_true_proxy_candidates_only()`.

## Important Defaults

`DreamPlacePipeline` defaults:

- `num_starts=6`, capped by `cap_num_starts()`.
- `top_dp_for_rescue=5`.
- `jitter_sigma_um=0.115`.
- `global_iterations=240`; feature scaling is off by default.
- `num_bins=128`.
- `target_density=0.72`.
- `timeout_seconds=720.0` per full DREAMPlace run.
- `rich_candidate_set=True`.
- `pre_dp_valid_starts=44`, `pre_dp_valid_pool_size=56`,
  `pre_dp_valid_selection="diverse"`.
- `explicit_legalize_dp_outputs=True`.
- `replace_rescue=True`, `replace_rescue_trigger_proxy=0.0`,
  `replace_rescue_timeout_seconds=240.0`.
- Coordinate descent is enabled:
  `post_rescue_coord_descent_seconds=240.0`,
  `post_rescue_coord_descent_max_passes=1`,
  `post_rescue_coord_descent_node_order="descending_size"`.
- GWTW SA is enabled:
  `post_rescue_gwtw_seconds=180.0`,
  `post_rescue_gwtw_num_workers=8`,
  `post_rescue_gwtw_num_iters=120`,
  `post_rescue_gwtw_syncup_freq=0.20`,
  `post_rescue_gwtw_top_k=2`,
  `post_rescue_gwtw_t_max=5e-3`,
  `post_rescue_gwtw_t_min=5e-6`.
- Removed stages are not available as pipeline options: post-DP SA, legacy
  post-rescue SA, congestion polish, single-worker TILOS SA, and greedy refine.
- `hyperband_enabled=False`.

Base DREAMPlace JSON overrides are seeded with:

- `density_weight=2.15e-4`
- `gamma=3.3`
- `gp_noise_ratio=0.060`
- `stop_overflow=0.050`
- one global-place stage with `learning_rate=0.013`,
  `Llambda_density_weight_iteration=2`, and `Lsub_iteration=3`

Tuner-provided `dreamplace_json_overrides` are deep-merged on top.

## End-to-End Flow

1. Clone the initial handoff from `benchmark.macro_positions`.
2. Load the `.plc` through `PlcLookup`.
3. Resolve and validate the DREAMPlace install.
4. If `.plc` or DREAMPlace is unavailable, legalize the seed and return a
   fallback reason.
5. Cap requested starts with `cap_num_starts()`.
6. Scale global iterations with `scaled_global_iterations()` if enabled.
7. Legalize the `.plc` reference with the stronger reference legalizer and add
   it as the initial true-proxy guardrail candidate.
8. Build rich DREAMPlace variants.
9. Build diverse initial placements.
10. Discover valid pre-DP starts by legalizing generated starts and selecting
    diverse starts, optionally using proxy scoring.
11. Run DREAMPlace for each start/spec pair, or experimental hyperband.
12. Optionally legalize DREAMPlace outputs if `explicit_legalize_dp_outputs` is
    enabled.
13. Salvage invalid or out-of-bounds DREAMPlace slots only as needed to reach a
    healthy valid pool: target `max(8, 2 * top_dp_for_rescue)` valid initial DP
    outputs.
14. True-proxy select the best valid preliminary candidates.
15. Run RePlAce rescue from the original seed and the top-K preliminary DP
    outputs.
16. Merge cached RePlAce scores with current scores and select the best valid
    proxy without re-scoring all rescue placements.
17. Run post-rescue polish stages in order: coordinate descent, then GWTW SA.
18. Return `selection.placement` with reason `ok`.

## Start and Iteration Policy

`cap_num_starts()` keeps runtime bounded while allowing 50 DREAMPlace starts on
smaller benchmarks:

- `num_hard_macros >= 1600`: cap at 8 starts.
- `>= 1000`: cap at 14.
- `>= 700`: cap at 20.
- `>= 450`: cap at 32.
- otherwise: cap at 50.

When `scale_iterations_with_features=True`, iterations are stretched for higher
hard-area utilization and macro count, capped at 1.25x the base count.

## Initial Placement Portfolio

`make_diverse_initial_placements()` builds maximin-diverse DREAMPlace handoff
seeds from the evaluator placement:

- symmetry transforms: identity, mirror X, mirror Y, mirror XY, transpose, and
  anti-transpose;
- anchor-biased starts toward corners and edge centers;
- Gaussian jittered copies;
- deterministic `native_jit` first;
- exact `initial_plc` reserved as the final slot;
- remaining starts selected by normalized RMS distance across movable hard macro
  centers.

Sparse high-net cases use a gentler local policy with fewer disruptive
transforms and lower jitter. The detector is low hard-area utilization, at least
20k nets, and roughly 250-520 hard macros.

## Pre-DP Valid-Start Discovery

The pre-DP discovery stage is opt-in (`pre_dp_valid_starts > 0`). It:

- starts from a strongly legalized `.plc` reference;
- generates a larger diverse pool;
- clamps and legalizes generated placements using the faster spiral legalizer;
- ranks valid placements by true proxy, surrogate, diverse selection, or hybrid
  selection depending on `pre_dp_valid_selection`;
- optionally runs direct-proxy TILOS SA on the selected starts;
- appends selected starts to the normal DREAMPlace portfolio rather than
  replacing the empirically useful base start/variant pairings.

Default knobs are `pre_dp_valid_pool_size=24`, `pre_dp_valid_selection="proxy"`,
`pre_dp_proxy_eval_limit=8`, and pre-DP SA disabled.

## DREAMPlace Variant Portfolio

With `rich_candidate_set=True`, starts cycle through feature-derived variants.
The normal-case portfolio is roughly 30 orthogonal modes spanning:

- target densities from spread modes to tight modes;
- bin counts `64`, `128`, `256`, plus evaluator-grid-aligned bins;
- density weight scale;
- `gp_noise_ratio`;
- `stop_overflow`;
- `gamma`;
- learning-rate stage variants;
- two-stage cooling;
- congestion-friendly softer-density variants.

The first variants remain the empirical core: `base`, `spread`, `tight`,
`xspread`, `xtight`, `aligned_coarse`, `aligned_fine`, and
`salvaged_explore` or `net_escape`. Later variants include broader search such
as `td065_g25`, `td080_b256`, `noise_high`, `ovf_tight`, `wide_spread`,
`wide_tight`, `gamma_high`, `twostage_cool`, and `cong_friendly_*`.

Sparse high-net cases use the specialized wirelength-sensitive set: `wire`,
`wire_tight`, `wire_loose`, `wire_coarse`, `wire_fine`, `mild_spread`,
`dense_wire`, and `soft_relax`.

Each DREAMPlace spec receives a deterministic random seed based on start index
and benchmark size.

## DREAMPlace Execution and Salvage

Each DREAMPlace run calls `run_dreamplace_placement()` with the benchmark,
`.plc`, selected initial placement, per-spec target density/bin count, iteration
budget, timeout, thread count, JSON overrides, and optional GPU selection.

Hyperband/successive halving remains opt-in only. The code comments keep it
experimental because short DREAMPlace runs did not reliably predict full-budget
proxy quality.

In the normal path, failed or out-of-bounds initial runs are salvaged only if
the valid-yield is low:

- If some initial runs are valid, their specs become rotating safe templates.
  Invalid slots are retried with those templates plus heavier jitter until the
  valid pool reaches `max(8, 2 * top_dp_for_rescue)`.
- If no initial runs are valid, the fallback template uses `target_density=0.76`,
  `num_bins=128`, density weight scale `1.05`, `stop_overflow=0.075`, and
  `gp_noise_ratio=0.045`.

The legalized `.plc` reference is scored alongside DP outputs, so the pipeline
has a proxy guardrail even when DP quality is poor.

## Selection Rule

Selection uses `select_best_true_proxy_candidates_only()` in
`submissions/_candidate_select.py`.

For each candidate:

- `validate_placement()` checks bounds and zero hard-macro overlap;
- `compute_overlap_metrics()` records overlap count;
- invalid candidates receive infinite cost;
- valid candidates are scored with `compute_proxy_cost()`;
- the selected candidate is the valid placement with minimum `proxy_cost`.

Score records include proxy, wirelength, density, congestion, overlaps, and
validation violations.

## Post-DREAMPlace Refinement

After true-proxy selection, the only refinement stages are RePlAce rescue,
coordinate descent, and multiprocessing GWTW SA. Post-DP SA, legacy post-rescue
SA, congestion polish, single-worker TILOS SA, and greedy refine have been
removed from the pipeline surface.

## RePlAce Rescue

RePlAce rescue is enabled whenever the current selection is valid because
`replace_rescue_trigger_proxy=0.0`.

The rescue portfolio is a lean 3-config congestion-focused set:

- `density=0.74`, `pcofmax=1.20`, `-bin 128`, `-overflow 0.04`,
  `-pcofmin 0.90`;
- `density=0.82`, `pcofmax=1.50`, `-bin 128`, `-overflow 0.05`,
  `-pcofmin 0.85`;
- `density=0.72`, `pcofmax=1.08`, `-bin 128`, `-overflow 0.06`,
  `-racnti 5`, `-racnto 10`.

It runs those configs from:

- the original `.plc` seed as a guardrail;
- the top `top_dp_for_rescue` valid preliminary DREAMPlace outputs by true
  proxy, default 5.

With defaults, the code attempts 3 initial-seeded RePlAce runs plus up to
`3 * 5` DP-seeded runs. `ReplacePipeline` scores its own candidates; the outer
pipeline renames and reuses those cached score records, then selects the lowest
valid proxy across DREAMPlace, post-DP-SA, and RePlAce candidates.

## Post-Rescue Polish Stages

Post-rescue polish stages run only when their budgets are positive and the
current best is valid.

1. Legacy post-rescue surrogate SA: disabled by default.
2. Congestion-targeted directional polish: disabled by default.
3. Single-worker direct-proxy TILOS-style SA: disabled by default.
4. Coordinate descent: enabled by default.
5. Go-With-The-Winners SA: enabled by default.
6. Random-jitter greedy true-proxy refine: disabled by default.

Failures in these stages are caught so the current best placement remains the
fallback.

## Coordinate Descent

`submissions/_coord_descent.py` implements a TILOS-style coordinate descent
polish with this repo's direct proxy scoring.

For each movable hard macro, it:

- chooses a k-distance bounded neighborhood around the macro's current evaluator
  grid cell;
- optionally filters cells with `plc.get_node_mask()`;
- optionally subsamples cells with `cell_search_prob`;
- tries candidate cell centers, clips to canvas, legalizes lightly, validates,
  and scores with `compute_proxy_cost()`;
- accepts only strict proxy improvements.

Defaults are one pass, 240 seconds, descending macro-size order, and adaptive
`k_bound` if not provided. The adaptive `k_bound` is broader for small macro
counts and shrinks on larger benchmarks to keep scored cells bounded.

## GWTW SA

`submissions/_tilos_gwtw_sa.py` is wired into the pipeline. It is a Python
multiprocessing implementation of the TILOS Go-With-The-Winners idea, not a
direct invocation of the external C++ binary.

Key mechanics:

- `num_workers` independent worker processes start from the current best
  placement.
- Each worker loads its own `PlacementCost` through `PlcLookup`.
- Workers use the `_tilos_moves.py` five-action move set: swap, shift, mirror,
  move, shuffle.
- Every proposal is legalized, validated, and scored with direct
  `compute_proxy_cost()`.
- Cooling is geometric from `post_rescue_gwtw_t_max` to
  `post_rescue_gwtw_t_min`.
- After every `syncup_freq * num_iters` steps, workers are sorted by cost; the
  bottom workers are replaced by clones of the top `top_k` winners.
- Workers return their current Metropolis state plus the best proxy they saw;
  the master tracks the lowest reported proxy candidate, and the pipeline
  re-scores before adopting it.

Defaults are intentionally light but more exploratory than the old near-greedy
single-worker SA: 8 workers, 120 iterations per worker, 180 seconds, `top_k=2`,
equal action probabilities, `t_max=5e-3`, `t_min=5e-6`.

The pipeline scales GWTW iterations down only when a tuner raises the default:

- `>=150k` nets: at most 800 iterations.
- `>=80k` nets: at most 1500.
- `>=40k` nets: at most 2400.
- `>=15k` nets: at most 3000.
- otherwise: configured `post_rescue_gwtw_num_iters`.

With the current default of 120 iterations, these caps do not reduce normal
default runs.

## Runtime Environment Notes

`submissions/_dreamplace_cpu_smoke.py` now tries harder to preload a compatible
`libstdc++.so.6` before importing or running DREAMPlace. It searches the active
conda/venv lib directory plus common conda env/package locations, prefers a
library containing `CXXABI_1.3.15`, then updates `LD_LIBRARY_PATH`, `LD_PRELOAD`,
and `ctypes` preload state on Linux.

Set `MACRO_PLACE_DP_DEBUG_SUBPROCESS=1` to inherit DREAMPlace subprocess logs.
Set `MACRO_PLACE_TUNER_DEBUG=1` for stage-level tuner progress logs.

## Fallback Reasons

`DreamPlacePipelineResult.reason` can be:

- `ok`: normal successful selection.
- `missing_plc`: no `.plc` could be loaded; repaired initial seed returned.
- `dreamplace_install_missing`: DREAMPlace install check failed; repaired seed
  returned.
- `all_dreamplace_starts_failed`: no candidate was available after attempted
  DREAMPlace execution.
- `no_valid_dreamplace_candidate`: no candidate passed validation.
- `selection_failed`: scoring/selection threw unexpectedly.

Fallback placement is produced by `legalize_hard()` on the original seed with
1200 legalizer rounds and `overlap_gap=1e-3`.

## Current Mental Model

The pipeline is a three-layer strategy:

- DREAMPlace creates diverse continuous-placement basins.
- RePlAce performs compact congestion-focused rescue from the initial seed and
  strongest DP seeds.
- Direct-proxy local search does the final work: coordinate descent for monotone
  single-macro improvements, then GWTW SA for population-based escape moves.

The highest-leverage knobs are:

- effective DREAMPlace start count and rich-variant ordering;
- opt-in pre-DP valid-start discovery;
- `top_dp_for_rescue`;
- the 3 RePlAce rescue configs;
- coordinate descent `k_bound`, pass count, and time budget;
- GWTW worker count, iteration count, temperatures, sync frequency, and `top_k`;
- runtime caps on high-net-count benchmarks.
