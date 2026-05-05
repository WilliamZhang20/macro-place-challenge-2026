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

`submissions/replace_pipeline_placer.py` is the current scoring branch.

- Flow: CasADi baseline -> Bookshelf export -> external RePlAce candidates ->
  `.pl` import -> hard legalization -> true-proxy selection.
- Full promoted public IBM sweep:
  `/tmp/macro_place_replace_full_promoted/summary.json`.
  Average is `1.276728` over 17 public cases; 16/17 select a RePlAce
  candidate, all selected placements are valid with zero hard overlaps.
- Winning knob families are general, not benchmark-name rules:
  lower-density/finer-bin candidates (`0.70/0.72/0.80` with `-bin 64/128`),
  compact `pcof=1.08`/`pcofmin` variants, and high-spread `pcof=1.20` with
  `-bin 64/128`.
- Biggest recent unlocks: `0.80:1.20 -bin 128` on late large cases,
  `0.84:1.20 -bin 128` on a high-density case, and fine-bin `0.72/1.03`
  candidates on mid-grid cases. `ibm15` still prefers baseline.

This RePlAce bridge is now the safe scoring path below the 1.3 target. Keep
promotions generic and validate on feature strata/full public sweeps before
submission.

## Evidence From Earlier Branches

- `casadi_placer.py`: practical floor, complete IBM average about `1.454376`,
  reliable and valid, but local rather than global.
- `hard_macro_lns_quick_placer.py`: average about `1.454071`, real but tiny
  improvement over CasADi.
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

## DREAMPlace Submission And Tuning

**Submission entry:** `submissions/dreamplace_pipeline_placer.py` — multi-start
DREAMPlace only (Bookshelf seed = loader `.plc`, plus jittered seeds), true-proxy
selection **among DREAMPlace outputs** via `_candidate_select` (no CasADi/DCCP
floor — failures surface as bad proxy or fallback to initial handoff).
Bookshelf export omits `.shapes` / `.route` for DREAMPlace parser compatibility.
Install: `scripts/setup_dreamplace.sh`; CPU/GPU: `MACRO_PLACE_DP_GPU` and
`resolve_dreamplace_gpu` in `_dreamplace_cpu_smoke.py`.

**Tuning:** `scripts/tune_dreamplace_optuna.py` (dependency: `optuna`, e.g. `uv sync --extra tuning`).
**Strategy:** Tree-structured Parzen Estimator (TPE) on a **single global knob vector**
(minimize mean or median true proxy over a calibration set). If `--benchmarks` is
omitted, the study uses an **evenly spaced subset** of ICCAD04 directories (sorted
on disk—no hand-picked benchmark identities). **Knobs** span pipeline settings
(`target_density`, `global_iterations`, `num_bins`, multi-start `num_starts`,
`jitter_sigma_um`, `num_threads`) plus DREAMPlace JSON fields merged correctly into
`global_place_stages[0]` (`learning_rate`, `optimizer`, `wirelength`,
`Llambda_density_weight_iteration`, `Lsub_iteration`) and top-level weights
(`density_weight`, `gamma`, `stop_overflow`, `gp_noise_ratio`, fillers/scale/degree).
Optional **MedianPruner** drops bad trials using running mean across the calibration
sequence; on clusters use `scripts/slurm/tune_dreamplace_optuna.slurm` (1× GPU,
`MACRO_PLACE_DP_GPU=1`). After search, re-validate winners at production budgets with
full `evaluate --all` and promote only improvements that hold across **feature**
strata, not single benchmark IDs.

Lighter diagnostic: `submissions/dreamplace_cpu_smoke_placer.py` (single short run).

## Next Steps

1. Run `evaluate submissions/replace_pipeline_placer.py --all` as the final
   harness check against the promoted defaults.
2. Promote only general policies that improve true-proxy aggregate selection;
   never branch on benchmark names.
