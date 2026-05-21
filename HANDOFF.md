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
  `replace_rescue_timeout_seconds=150.0` (was 240.0; per-config rescue
  timeout is further capped by remaining wall budget at call time).
- Coordinate descent is **disabled by default**
  (`post_rescue_coord_descent_seconds=0.0`); dropped after a
  bang-per-minute analysis showed it found 0–1 moves vs GWTW's 7–21 on
  the same wall time. Set this > 0 to re-enable.
- GWTW SA is enabled:
  `post_rescue_gwtw_seconds=360.0` (was 180s; CD's freed budget),
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

`cap_num_starts()` keeps runtime bounded; tiers use macro count AND net
count (`macros × nets` is the runtime proxy):

- `num_hard_macros >= 1600`: cap at 8 starts.
- `>= 1000`: cap at 14.
- `>= 700`: cap at 20.
- `>= 450`: cap at 32.
- `>= 350 AND num_nets >= 10000`: cap at 24 (added after ibm10 = 9952 s
  rule violation in the 2026-05-20 sweep).
- otherwise: cap at 50.

Total DP starts also include pre-DP discovered starts (`pre_dp_valid_starts`,
nh-capped — see Pre-DP Valid-Start Discovery below).

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

Default knobs are `pre_dp_valid_pool_size=56`, `pre_dp_valid_starts=44`,
`pre_dp_valid_selection="diverse"`, `pre_dp_proxy_eval_limit=8`, and pre-DP
SA disabled. Both `pool_size` and `requested` are dynamically capped by
hard-macro count to keep pre-DP discovery time bounded:

- nh ≥ 1000 → pool 10, req 6
- nh ≥ 600 → pool 18, req 12
- nh ≥ 400 → pool 30, req 22
- otherwise → 56/44 (defaults)

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

The rescue portfolio is an **orthogonal 12-config** set spanning the
`(density, pcofmax, bin, overflow, pcofmin, racnt*)` grid. Memory note:
RePlAce's final basin is determined by `(density, pcofmax)` regardless of
seed, so seed diversity without config diversity is wasted. Slots 0-2 are
the proven congestion-attack winners; slots 3-7 are historical
`_GENERIC_CONFIGS` winners (including the `-bin 64` paths that found the
ibm01 0.9219 baseline); slots 8-11 fill in `(density × pcofmax)` corners
that were not yet covered:

| # | density | pcofmax | extra args |
|---|---------|---------|------------|
| 0 | 0.74 | 1.20 | `-bin 128 -overflow 0.04 -pcofmin 0.90` |
| 1 | 0.82 | 1.50 | `-bin 128 -overflow 0.05 -pcofmin 0.85` |
| 2 | 0.72 | 1.08 | `-bin 128 -overflow 0.06 -racnti 5 -racnto 10` |
| 3 | 0.80 | 1.03 | `-bin 128 -pcofmin 0.98` |
| 4 | 0.80 | 1.20 | `-bin 128` |
| 5 | 0.84 | 1.20 | `-bin 128` |
| 6 | 0.70 | 1.03 | `-bin 64` |
| 7 | 0.84 | 1.03 | `-bin 64` |
| 8 | 0.76 | 1.08 | `-bin 128 -overflow 0.05` |
| 9 | 0.78 | 1.20 | `-bin 128 -overflow 0.05 -pcofmin 0.92` |
| 10 | 0.86 | 1.08 | `-bin 128` |
| 11 | 0.74 | 1.50 | `-bin 128 -overflow 0.05 -racnti 8 -racnto 12` |

Seeds and configs are paired **Latin-square** style so every `(seed, config)`
combination targets a distinct basin and no config is duplicated across
seeds. Densities are matched to seed character so RePlAce does not abort
with `"no more tier to assign!"` on the dense `.plc` initial placement.
After winner analysis, DP ranks 1, 2, 3 are **dropped** entirely (never
won outright). Only three rescue groups survive:

- `.plc` initial seed (guardrail, dense): configs 3, 4, 5 (`density >= 0.80`)
- DP rank 0 (top DP output, dense): configs 1, 7
- DP rank 4 (most-spread DP): config 6 (low-density 0.70 bin=64)

Total: **6 invocations** (was 12). Reasoning per benchmark:

- `replace_initial` wins on ibm07, ibm14, ibm17
- `dp_rank0` wins on ibm10
- `dp_rank4` wins on ibm01 (only path where low-density bin=64 converges
  — needs a spread seed)

The three groups are dispatched via a `ThreadPoolExecutor` (3 workers by
default, `MACRO_PLACE_RESCUE_WORKERS`). Each task gets its own
`work_root` subdir (`<tmp>/macro_place_replace_pipeline/<bench>__<prefix>/`)
so concurrent Bookshelf exports don't collide. Thread-safe
`rescue_scores` aggregation via `threading.Lock`. Expected ~2× speedup
on rescue stage.

`ReplacePipeline` scores its own candidates; the outer pipeline renames
and reuses those cached score records, then selects the lowest valid
proxy across DREAMPlace and RePlAce candidates.

## Post-Rescue Polish Stages

Only GWTW SA is enabled by default. Coordinate descent was dropped after
bang-per-minute analysis showed it found 0–1 moves vs GWTW's 7–21 on the
same wall time; on the only benchmark where polish moved the needle
(ibm01) GWTW was ~5× more cost-effective per minute. GWTW gets the
budget freed by CD (360s, up from 180s).

Failures in GWTW are caught so the current best (rescue or DP)
placement remains the fallback.

**Critical fd-leak fix (2026-05-20):** GWTW SA's `multiprocessing.Pool`
crashed with `OSError(24, "Too many open files")` on slurm runs once
the process accumulated ~51k fds (cgroup cap 51200). Root cause was
PyTorch's default `file_descriptor` shared-memory strategy holding one
fd per shared tensor across DREAMPlace + RePlAce invocations. Both
`_dreamplace_pipeline.py` and `_tilos_gwtw_sa.py` now call
`torch.multiprocessing.set_sharing_strategy("file_system")` at module
import, switching to `/dev/shm` filenames instead of fds.

## Coordinate Descent (disabled by default)

`submissions/_coord_descent.py` implements a TILOS-style coordinate descent
polish with this repo's direct proxy scoring. Set
`post_rescue_coord_descent_seconds > 0` to re-enable; otherwise skipped.
The 0–1 moves it found across ibm01/07/10/14 didn't justify its 240s.

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

## Runtime Budget Enforcement (2026-05-20)

After observing **ibm10 = 9952s** (2.77× the 1h rule cap) and **nvdla = 4663s**
in the prior sweeps, the pipeline now enforces a per-benchmark wall budget end
to end. Default is 2700s via `MACRO_PLACE_PIPELINE_WALL_BUDGET_S` (900s margin
under the 3600s rule).

Layered guards:

| Stage | Threshold | Action |
|-------|----------:|--------|
| DP loop | elapsed > 55% (1485s) **and** ≥8 starts done | early-exit |
| Pre-DP discovery cap | nh≥1000 → pool=10 req=6; nh≥600 → 18/12; nh≥400 → 30/22 | shrink pool/req |
| `cap_num_starts` mid-tier | nh≥350 AND nets≥10000 | cap=24 |
| Rescue per-call | elapsed > 85% (2295s) | skip remaining `merge_rescue` calls |
| Rescue per-config timeout | – | `min(150s, (remaining − 360s) / n_configs)`, floor 30s |
| Polish (GWTW) | elapsed > 92% (2484s) | skip |
| Polish stage budget | – | `min(360s, remaining × 0.85)` |
| File-descriptor soft cap | first thing in `run()` | raise to `min(hard, 65536)` to prevent GWTW `OSError(24)` |

These guards target the 1h hard cap with margin. The fd-raise also resolved
the **GWTW SA `OSError(24, "Too many open files")`** that was crashing the
ibm10/nvdla GWTW stage in the 2026-05-20 sweep.

### Polish stages: coord_desc dropped

Bang-per-minute analysis across ibm01/07/10/14 showed:

- coord_desc: 0–1 accepted moves per 240s budget, max Δ ≈ -0.0002
- GWTW SA: 7–21 accepted moves per 180s, max Δ ≈ -0.0008

GWTW won ~5× on Δproxy/minute. **coord_desc was dropped** (default
`post_rescue_coord_descent_seconds=0.0`); GWTW SA was bumped from 180s to
360s with the freed budget.

### Rescue trim

After analyzing per-benchmark winners, dropped DP ranks 1, 2, 3 entirely.
Only `replace_initial` (3 configs), `dp_rank0` (2 configs), and `dp_rank4`
(1 config) survive. Six invocations down from eleven. Reasoning:

- `replace_initial` wins on ibm07, ibm14
- `dp_rank0` wins on ibm10
- `dp_rank4` wins on ibm01 (only path where low-density bin=64 converges)
- Ranks 1/2/3 never won outright on tested benchmarks

### Parallel rescue + parallel scoring

- `merge_rescue` calls now dispatched via `ThreadPoolExecutor` (3 workers
  by default, `MACRO_PLACE_RESCUE_WORKERS`). Each task gets its own
  `work_root` subdir so concurrent Bookshelf exports don't collide.
  Expected ~2× speedup on rescue stage.
- `select_best_true_proxy_candidates_only` now scores candidates via
  `multiprocessing.Pool` (4 workers by default,
  `MACRO_PLACE_PARALLEL_SCORE_WORKERS`) when ≥4 candidates. Each worker
  loads its own `PlacementCost`. Expected ~3× speedup on big benchmarks
  with many DP outputs.

### Status as of 2026-05-20 16:52 UTC (interactive about to be cancelled)

**Slurm job 1441565** — `--all` IBM sweep with fd-leak fix
(`torch.multiprocessing.set_sharing_strategy('file_system')` in both
`_dreamplace_pipeline.py` and `_tilos_gwtw_sa.py`) applied. Submitted
2026-05-20 ~17:35 UTC, queued behind 1441528. Logs:
`sweep_logs/slurm_1441565_dreamplace_pipeline.{out,err}` plus per-run
`sweep_logs/dreamplace_pipeline_1441565_<timestamp>.log`. This is the
"clean" sweep — GWTW SA should not crash mid-run anymore.

**Slurm job 1441528** — `--all` IBM sweep with current runtime-fix code on
`watgpu108`, started 15:34 UTC, 12h wall limit:

- ibm01: 0.9219 (1554s) ✓
- ibm02: 1.3059 (1743s) ✓
- ibm03+: in flight

**Old slurm 1441398** — pre-fix code, still running on same node; its
ibm10 hit 9952s (rule violation). Don't promote those numbers.

**Local validation chain** (watgpu608, GPU contended by another user's
vLLM jobs — local timings inflated):

- ibm10 v4: **1.1687** (2761s, 46 min) ✓ beats historical 1.1688
- ibm16 v4: **1.2533** (2398s, 40 min) ✓ beats historical 1.4780 by 0.225
- ibm17 v4: in progress, DP loop just done at ~32 min — will likely
  abort when interactive 1441448 cancels.

### Outstanding work for next session

1. **Wait for slurm 1441528 to finish** — gives clean per-benchmark numbers
   on all 17 IBM (current code, runtime-safe).
2. **Compare 1441528 vs historical baselines** to identify any regression
   the budget guards may have introduced (e.g., truncated rescue on
   net-heavy benchmarks).
3. **Stage uncommitted changes**: `submissions/_dreamplace_pipeline.py`
   (budget guards, rescue trim, parallel rescue, CD-drop, GWTW boost),
   `submissions/_candidate_select.py` (parallel scoring), HANDOFF.md
   updates. GPG signing still requires a TTY; commit manually.
4. **Re-validate ibm17 locally** once interactive session is back —
   chain v4 was killed mid-ibm17 by interactive cancellation.
5. **Consider**: relaunching a slurm sweep that picks up the
   parallel-rescue + parallel-scoring edits (1441528 was launched before
   those landed and won't benefit).

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

## Sweep Logs

Two SLURM jobs were submitted on 2026-05-20 against the orthogonal RePlAce
rescue portfolio (`_dreamplace_pipeline.py` Latin-square pairing). Both use
`scripts/slurm/run_dreamplace_pipeline*.slurm`.

- **IBM `--all`** — job `1441398`, script
  `scripts/slurm/run_dreamplace_pipeline.slurm`.
  - SLURM stdout/err: `sweep_logs/slurm_1441398_dreamplace_pipeline.out`,
    `sweep_logs/slurm_1441398_dreamplace_pipeline.err`.
  - Per-run tee: `sweep_logs/dreamplace_pipeline_1441398_<timestamp>.log`.
- **NG45 `--ng45`** — job `1441399`, script
  `scripts/slurm/run_dreamplace_pipeline_ng45.slurm`.
  - SLURM stdout/err: `sweep_logs/slurm_1441399_dreamplace_pipeline_ng45.out`,
    `sweep_logs/slurm_1441399_dreamplace_pipeline_ng45.err`.
  - Per-run tee: `sweep_logs/dreamplace_pipeline_1441399_<timestamp>.log`.

Track queue state with `squeue -j 1441398,1441399`.

### Density-Aware Re-Pairing Validation (2026-05-20)

After observing repeated `ValueError("no valid placement candidates")`
exceptions in the `.plc`-seeded rescue on multiple benchmarks, the pairing
was redesigned so the dense `.plc` seed only sees high-density (`>=0.80`)
configs. Validation on three IBM benchmarks:

| Benchmark | New (re-paired) | Historical | Δ | Note |
|-----------|----------------:|-----------:|---:|------|
| ibm01 | **0.9217** | 0.9219 | -0.0002 | tie within polish variance; `.plc` rescue now succeeds (3 valid candidates vs 0) |
| ibm07 | 1.2634 | 1.2368 | +0.0266 | matches a different historical state; `.plc` rescue now succeeds but does not beat the older "high-cost initial rescue config" path that the orthogonal refactor removed |
| ibm14 | **1.4011** | 1.4012 | -0.0001 | matches historical at rescue exit (`.plc` seed + config 3 = `0.80, 1.03, pcofmin=0.98`), GWTW SA polishes by 0.0001 |

Validation logs:
`sweep_logs/ibm01_repair_pair_20260520_060533.log`,
`sweep_logs/ibm07_repair_pair_20260520_062753.log`,
`sweep_logs/ibm14_repair_validation_20260520_072714.log`.

Root cause for the rescue exception: RePlAce's mixed-size global placer
aborts with `"no more tier to assign"` when a low-density target (e.g.
`density=0.74`) is applied to a dense seed, producing zero `.pl` files
that the candidate batch then rejects. Re-pairing only the `.plc` seed
to high-density configs eliminates the abort across the tested
benchmarks.

### ibm01 Validation Result

Single-benchmark dry run of the orthogonal rescue change before submitting
the full sweeps. Log: `sweep_logs/ibm01_ortho_rescue_20260520_043517.log`.
Total wall: 1357.60 s.

| Stage | Best valid proxy | Δ |
|-------|------------------|----|
| DP top-5 (post-DP true-proxy selection) | 0.9959 | — |
| RePlAce rescue (winner: DP rank 1, config 6 `den=0.70 pcof=1.03 bin=64`) | **0.9219** | −0.0740 |
| Coordinate descent (1 accepted move) | 0.9217 | −0.0002 |
| GWTW SA (21/119 accepted) | **0.9209** | −0.0008 |

Final: `proxy=0.9209 (wl=0.067, den=0.596, cong=1.112), VALID`.

For reference: historical baseline is 0.9219 (from the older
`_GENERIC_CONFIGS` 17-config sweep) and the trimmed 3-config rescue
regressed to 0.9321. The new orthogonal portfolio recovers the historical
basin (config 6, `den=0.70 pcof=1.03 bin=64`, was dropped by the trim)
and beats it by 0.0010.

Per-seed rescue breakdown (10 of 12 invocations succeeded):

| Rescue seed | Configs | Best valid proxy |
|-------------|---------|------------------|
| `.plc` initial | 0, 1, 2 | exception — `no valid placement candidates` |
| DP rank 0 | 3, 4 | 0.9790 |
| DP rank 1 | 5, 6 | **0.9219** |
| DP rank 2 | 7, 8 | 1.0043 |
| DP rank 3 | 9, 10 | 0.9737 |
| DP rank 4 | 11 | exception — `no valid placement candidates` |

Two known soft failures on this benchmark:

1. The `.plc`-seeded guardrail rescue (configs 0-2) threw `ValueError("no
   valid placement candidates")`. Under the old cross-product the same
   three configs were also run from each DP seed, masking this; under the
   Latin-square pairing they only run from the `.plc` seed and so the
   failure now surfaces. Did not affect the final result because DP rank 1
   won, but worth investigating before promoting on harder benchmarks
   where the guardrail matters.
2. DP rank 4 with config 11 (aggressive routability, `racnti=8 racnto=12`)
   threw the same error. Likely a convergence-from-this-seed problem with
   that specific config.

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
- the 12 orthogonal RePlAce rescue configs and their Latin-square pairing
  with the `.plc` seed and the top-K DP outputs;
- coordinate descent `k_bound`, pass count, and time budget;
- GWTW worker count, iteration count, temperatures, sync frequency, and `top_k`;
- runtime caps on high-net-count benchmarks.
