## How To Run

```bash
# Use the conda env on this machine (no `~/myenv` plain venv exists here):
conda activate myenv
# Or invoke the interpreter directly:
#   /u5/w223zhan/.conda/envs/myenv/bin/python -m macro_place.evaluate ...
evaluate submissions/<placer>.py -b ibm01
evaluate submissions/<placer>.py --all
evaluate submissions/<placer>.py --ng45
```

Use `scripts/overnight_sweep.sh` for long sweeps. It writes timestamped logs
under `sweep_logs/`.

## Rules And Constraints

- Optimize proxy cost: `wirelength + 0.5 * density + 0.5 * congestion`.
- Hard-macro overlap must be exactly zero; fixed macros must stay fixed.
- Do not hardcode benchmark names. Use only benchmark-derived features such as
  grid size, utilization, macro counts, net statistics, congestion, runtime, or
  legality diagnostics.

## Current Best Path

`submissions/dreamplace_pipeline_placer.py` is the current scoring branch.
Run it directly through the standard evaluator; no sidecar config is required:

```bash
conda run -n myenv python -m macro_place.evaluate submissions/dreamplace_pipeline_placer.py --all
conda run -n myenv python -m macro_place.evaluate submissions/dreamplace_pipeline_placer.py --ng45
```

- Flow: multi-start DREAMPlace -> true-proxy selection -> bounded SA polish ->
  compact RePlAce rescue -> true-proxy final selection.
- DREAMPlace build: `scripts/setup_dreamplace.sh` now defaults to CUDA fatbins
  for both L40S/Ada (`sm_89`) and H200/Hopper (`sm_90`).
- Successive halving/hyperband is supported only as an opt-in experiment and is
  off by default. The May 13 hyperband run regressed average proxy from
  `1.2748` to `1.4656`, with severe `ibm07`/`ibm15` losses, so it should not
  be used for scoring.
- Current projected public IBM average is about `1.2484`, better than the
  prior best full pushed run average `1.2748` from
  `sweep_logs/slurm_1437819_dreamplace_pipeline.out`. Projection uses the
  prior full run plus confirmed targeted improvements:
  `ibm07 1.4444 -> 1.2368`, `ibm14 1.4479 -> 1.4012`,
  `ibm17 1.6691 -> 1.5053`, and `ibm18 1.6372 -> 1.6060` from the validated
  high-cost initial RePlAce rescue config. `ibm10`, `ibm12`, `ibm15`, and
  `ibm16` were rechecked with no proxy loss.
- Key evidence logs:
  `sweep_logs/dreamplace_rebuild_sm89_sm90_20260518.log`,
  `sweep_logs/dreamplace_h200_smoke_20260518.log`,
  `sweep_logs/dreamplace_pipeline_ibm07_initial_rescue_fix_20260517_182449.log`,
  `sweep_logs/dreamplace_pipeline_ibm10_compact_rescue_h200_20260518.log`,
  `sweep_logs/dreamplace_pipeline_ibm12_compact_rescue_h200_20260518.log`,
  `sweep_logs/dreamplace_pipeline_ibm14_high_cost_initial_rescue_h200_20260518.log`,
  `sweep_logs/dreamplace_pipeline_ibm15_high_cost_initial_rescue_h200_20260518.log`,
  `sweep_logs/dreamplace_pipeline_ibm16_compact_rescue_h200_20260518.log`,
  `sweep_logs/dreamplace_pipeline_ibm17_high_cost_initial_rescue_h200_20260518.log`,
  `sweep_logs/dreamplace_pipeline_ibm18_high_cost_initial_rescue_h200_20260518.log`,
  and `sweep_logs/replace_initial_ibm18_existing_artifact_score_20260518.log`.

## Evidence From Earlier Branches

- `casadi_placer.py`: practical floor, complete IBM average about `1.454376`,
  reliable and valid, but local rather than global.
- Selector over older completed branches has low ceiling, around `1.453418`
  oracle average, so packaging old branches is not enough.
- Rudy correlates with evaluator congestion and should not be discarded; poor
  Rudy branch results likely reflect move generation/legalization issues.
- Orientation flips have small real signal but need sidecar plumbing and are a
  polish lever, not the main gap.
- `dccp_placer.py` was the initial baseline, with an average over IBM benchmarks of 1.4556. It used convex-concave procedure from a library wrapping CVXPY.
- Variations built on `casadi_placer.py` (local search and coordinate-style
  experiments) failed to move the needle meaningfully.

## Avoid

- Benchmark-name-specific tuning.
- Using CVXPY; the canonicalization overhead isn't worth it.
- More CasADi global/two-phase variants without a new measured mismatch.
- Large local-search sweeps before a stronger global candidate exists.

## DREAMPlace Pipeline Overview

**Entry point:** `submissions/dreamplace_pipeline_placer.py` wraps
`DreamPlacePipeline` from `submissions/_dreamplace_pipeline.py`. Install
DREAMPlace with `scripts/setup_dreamplace.sh`; device selection flows through
`MACRO_PLACE_DP_GPU` and `resolve_dreamplace_gpu` in
`submissions/_dreamplace_cpu_smoke.py`.

High-level flow:

1. Load the benchmark's initial `.plc` placement and matching
   `PlacementCost` object via `PlcLookup`. If either the `.plc` or DREAMPlace
   install is missing, return a legalized repair of the initial placement.
2. Compute benchmark-derived features only: hard utilization, macro counts,
   net counts, grid size, etc. Do not branch on benchmark names.
3. Choose the number of DREAMPlace starts with a feature cap. IBM-scale designs
   normally get 8 starts; very large designs are capped lower for runtime.
4. Generate diverse starting placements. The normal path builds a maximin pool
   from mirrored, transposed, edge/corner-biased, and jittered seeds. Sparse,
   high-net-count mid-size cases use a more local seed pool to preserve
   connectivity and avoid congestion explosions.
5. Build a DREAMPlace variant portfolio. Each start cycles through
   feature-derived target density, bin grid, density weight, overflow, noise,
   and global-place-stage settings. Sparse high-net cases use wire-friendlier
   variants with milder noise and density pressure.
6. For each start, export a temporary Bookshelf handoff with that initial
   placement. The export omits `.shapes` and `.route` for DREAMPlace parser
   compatibility.
7. Run DREAMPlace, import the `.gp.pl`, clamp/fix coordinates, and run hard
   legalization with a tiny gap to satisfy zero-overlap validation.
8. Score valid DREAMPlace outputs with the true proxy evaluator, then run a
   bounded post-DP simulated annealing polish on the top candidate(s). The SA
   loop uses a fast local surrogate (HPWL, coarse density, and RUDY pressure)
   for Metropolis decisions, legalizes proposals, and lets final true-proxy
   selection decide whether the polished candidate is worth keeping.
9. If the selected proxy is still high, run a narrow RePlAce rescue portfolio
   seeded from the current best placement. This is intentionally only a small
   set of high-value configs, not the full RePlAce sweep, and final true-proxy
   selection decides whether any rescue candidate is kept.
10. Final selection uses `_candidate_select.select_best_true_proxy`, so the
   original initial placement is also scored as a guardrail and wins if all
   DREAMPlace/SA/rescue candidates are worse.
11. Return the selected placement tensor. Diagnostics record the selected label
   and per-candidate proxy/validity information when `pipeline.run()` is used.

Lighter diagnostic: `submissions/dreamplace_cpu_smoke_placer.py` runs a single
short DREAMPlace pass through the same export/import plumbing.

## Next Steps

1. Run `evaluate submissions/replace_pipeline_placer.py --all` as the final
   harness check against the promoted defaults.
2. Promote only general policies that improve true-proxy aggregate selection;
   never branch on benchmark names.
