## How To Run

```bash
source ~/myenv/bin/activate
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

## DREAMPlace Status

DREAMPlace is installed at `external/DREAMPlace/install` and runs with:

- WSL build: use `cmake --build ... -j2`; `-j$(nproc)` caused OOM.
- Torch 2.10 has `_GLIBCXX_USE_CXX11_ABI=True`; configure with
  `-DCMAKE_CXX_ABI=1`.
- Runner compatibility fixes: omit empty `.shapes`, add NumPy 2 `np.string_`
  shim, collect config-specific `.gp.pl` copies so multi-config sweeps are
  actually scored.

Latest DREAMPlace finding:

- DREAMPlace is viable, but the current bridge underperforms the tuned RePlAce
  path. Damped random-global blends improved some cases modestly, while
  true-shape/global-only runs often reduce WL/congestion but lose on challenge
  density after import/legalization.
- Next DREAMPlace work should tune the bridge, not just raw optimizer params:
  soft macro representation, density scaling, import blend, legalizer
  displacement, and NaN/overflow behavior.

Useful DREAMPlace commands:

```bash
python scripts/run_dreamplace_diagnostics.py --benchmark ibm02 \
  --soft-macro-mode row_height --preset random_global \
  --config 0.65:200:0.01:1e-3 --timeout 240

python scripts/run_dreamplace_diagnostics.py --benchmark ibm02 \
  --soft-macro-mode preserve --preset global_only \
  --config 0.65:120:0.01:1e-3 --timeout 240
```

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
- Variations of the `casadi_placer.py` such as `hard_coord_descent_placer.py`, `hard_macro_lns_placer.py`, etc. failed to move the needle.

## Avoid

- Benchmark-name-specific tuning.
- Using CVXPY; the canonicalization overhead isn't worth it.
- More CasADi global/two-phase variants without a new measured mismatch.
- Large local-search sweeps before a stronger global candidate exists.

## Next Steps

1. Run `evaluate submissions/replace_pipeline_placer.py --all` as the final
   harness check against the promoted defaults.
2. Push DREAMPlace bridge tuning for hidden/test diversity, but keep
   `replace_pipeline_placer.py` as the safe submission path.
3. Promote only general policies that improve true-proxy aggregate selection;
   never branch on benchmark names.
