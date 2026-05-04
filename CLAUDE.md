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

## Analytical Descent Stack (experimental)

Pure PyTorch / NumPy pipeline for movable **hard** macros; **soft** macros and
fixed hard macros stay at initial coordinates for the smooth objectives.
Shared code: `submissions/_descent_core.py`. Tuning uses **features only**
(`num_hard_macros`, `num_nets`, canvas size, hard-macro area utilization)—no
benchmark-name branches.

**Objectives:** log-sum-exp **smooth HPWL** (β ~ 1 / (0.008–0.01 × canvas));
**FFT Poisson** density (64²) with Gaussian splats and a **fixed** background
from initial soft + fixed-hard positions; optional **soft Rudy** (sigmoid bins,
same β) vs. routing capacity. **Adam** with optional linear **lr decay**.

**Legalization:** coarse **min-displacement grid** snap + `_hard_legalizer.legalize_hard`.
Production unified placer uses **tiered** repair: stronger rounds after Lloyd
outers, lighter caps after LNS inner steps (runtime dominated by repeated
legalize).

**Unified entry:** `submissions/descent_pipeline_placer.py` — multistart
(true-proxy pick when ICCAD04 plc reload works), Lloyd ↔ legalize with **anchor**
penalty on later outers, LNS (random macro subsets + short global + legal),
periodic **HPWL-only** polish, final **anchored** refinement. Wall clock capped
(~53 min budget) with per-start LNS caps.

**Incremental stage files (evaluate each alone):**

| Stage | File | Idea |
|------:|------|------|
| 1 | `descent_step01_hpwl_only_placer.py` | HPWL only; overlaps expected (diagnostic). |
| 2 | `descent_step02_density_placer.py` | + FFT density, λ ramp; still illegal. |
| 3 | `descent_step03_mvp_legalize_placer.py` | + legalize; first valid MVP. |
| 4 | `descent_step04_lloyd_placer.py` | Lloyd + anchor; no Rudy in global loss. |
| 5 | `descent_step05_rudy_placer.py` | + soft Rudy in global loss. |
| 6 | `descent_step06_lns_placer.py` | + LNS + time budget (`_descent_core.lns_time_budget_sec`). |
| 7 | `descent_step07_multistart_placer.py` | + multistart + proxy selector. |
| 8 | `descent_step08_orient_placer.py` | Same as 7; Tier-1 harness does not apply Klein-4 orientations. |

**Profiling:** `scripts/descent_stage_profile.py` prints proxy vs. wall time for
phases 0–4; `--quick` uses scaled-down iters/legal rounds for fast A/B trends
(not contest-caliber scores).

**Status:** Experimental; average proxy is **not** at the RePlAce-bridge level
yet. Use for research, ablations, or future hybrid/selector glue—not the default
submission unless a sweep proves competitive under the 1h/bench cap.

## DREAMPlace Status

DREAMPlace is installed at `external/DREAMPlace/install` and runs with:

- WSL build: use `cmake --build ... -j2`; `-j$(nproc)` caused OOM.
- Torch 2.10 has `_GLIBCXX_USE_CXX11_ABI=True`; configure with
  `-DCMAKE_CXX_ABI=1`.
- Runner compatibility fixes: omit empty `.shapes`, add NumPy 2 `np.string_`
  shim, collect config-specific `.gp.pl` copies so multi-config sweeps are
  actually scored.

### DREAMPlace-only submission (`dreamplace_bridge_placer.py`)

Evaluator entrypoint for **RePlAce-free** IBM runs: CasADi baseline, then
DREAMPlace Bookshelf jobs over two legal seeds and two feature-driven density
configs (true-proxy selection over all candidates).

- **Layout:** `submissions/dreamplace_bridge_placer.py` orchestrates;
  `submissions/_dreamplace_bridge.py` builds utilization-tuned
  `target_density` schedules. `light_bridge_dreamplace_configs` now returns a
  utilization-anchor config (`clamp(util, 0.66, 0.86)`, 64 bins, low gamma) and
  a spread-anchor config (`clamp(util + 0.12, 0.74, 0.92)`, 128 bins, higher
  gamma). `submissions/_dreamplace_candidates.py` exports, runs DP, hard-only
  and tiny full-tensor baseline blends, import fallbacks, and legalization;
  `submissions/_dreamplace_runner.py` / `_dreamplace_presets.py` wrap the
  process and JSON presets.
- **Seeds:** Run 0 uses the **CasADi** placement as `initial_placement`; run 1
  uses **CasADi + Gaussian jitter** on movable hard macros (µm-scale), then
  `legalize_hard`, so the second DP sees a different basin than a near-duplicate
  legalized-.plc layout (on `ibm01`, legalized initial and CasADi centers differ
  by ~0.01 µm but share the same ~1.038 proxy, so jitter is required for real
  diversity).
- **Soft macros in Bookshelf:** `aspect_cap` when
  `macro_area_utilization >= 0.48`, else `row_height` (cap longest soft side
  at `k × row_height` in `aspect_cap`; `soft_macro_row_cap_mult` default 12).
- **Blends:** Hard-only blends keep soft macros at the seed placement (important
  bug fix; earlier code accidentally kept DREAMPlace-moved soft macros).
  Current bridge scores tiny hard-only alphas (`0.002/0.004/0.006/0.01`) plus
  larger probes (`0.28/0.45/0.62`), and tiny full-tensor alphas
  (`0.004/0.006/0.007/0.008/0.01`). The full-tensor micro-blends are the best
  observed DREAMPlace-only improvement on `ibm01`.
- **Legalization:** Lighter post-import pass than earlier experiments (fewer
  outer passes, no per-pass displacement budget by default). The current bridge
  uses a smaller `legalize_step_fraction=0.1` for the tiny-blend family; this
  preserved the best `ibm01` micro-move better than the default larger step.
- **GPU:** `DreamPlaceConfig.gpu` follows `torch.cuda.is_available()` unless
  overridden; README #2 (DreamPlace++) cites **~37s/bench on GPU**. Force CPU
  with `MACRO_PLACE_DP_CPU=1` if the build has no CUDA.
- **Optional macro preset:** `MACRO_PLACE_DP_MACRO=1` merges
  `macro_place_flag` + `two_stage_density_scaler` into DP JSON (off by default
  so mixed Bookshelf builds stay predictable).
- **Tuning jitter:** `MACRO_PLACE_DP_SEED_SIGMA` overrides the default perturbation
  σ (µm) for the second seed.

**Diagnostics harness:** `scripts/run_dreamplace_diagnostics.py` supports
`--bridge-configs`, `--soft-macro-mode aspect_cap`, bin field on `--config`,
`--include-route`, raw pre-legalization proxy metrics, full-tensor blend
screening (`--blend-full-tensor`), and legalizer CLI knobs; presets live in
`submissions/_dreamplace_presets.py`.

### Measured findings (recent)

- On **`ibm01`**, **CasADi-only** true proxy is **~1.0385** (valid). A full
  `evaluate` of the current `dreamplace_bridge_placer.py` now returns
  **~1.0330** (valid): best observed candidate is a tiny full-tensor blend from
  the spread-density config, with components about
  `wl=0.064`, `density=0.809`, `congestion=1.129`. This is a real improvement
  over CasADi, but still only a micro-move, not the leaderboard-scale jump.
- **RePlAce baseline** proxy on `ibm01` is **~0.998** (evaluator table); closing
  that gap with DREAMPlace-only remains hard. The aspirational `ibm01` target is
  below **0.9**, so the current bridge is still far from enough.
- Raw diagnostics showed the original hard-only blend implementation was
  misleading: it interpolated hard macros but kept DREAMPlace-moved soft macros,
  causing large congestion. Fixing it dropped quick blended candidates from
  around `1.95` to around `1.13`, and tiny hard-only blends later reached
  ~`1.038`.
- Tiny **full-tensor** blends are better than hard-only blends on `ibm01`; the
  best reproduced diagnostic result was around **1.03302** at full blend
  `alpha ~= 0.008` from a `target_density ~= 0.92`, 128-bin, `gamma=1.2e-4`
  DREAMPlace output.
- Sweeps tried and did **not** unlock a large move: macro-aware preset
  (`macro_place_flag`), route-aware export plus routability preset, gift/random
  global initializations, reverse/large hard-only blends, soft-policy
  alternatives (`row_height`/`preserve`, too slow on CPU and produced no usable
  `.pl` before timeout), and sparse combinations of several DREAMPlace
  displacement directions. All plateaued near `1.033`-`1.038`.
- This WSL environment reported `torch.cuda.is_available() == False`, so all
  recent DREAMPlace measurements were CPU-only; README leaderboard runtime
  numbers assume GPU and should not be compared directly.
- **Hybrid RePlAce + DREAMPlace** was prototyped then **removed** per direction
  to stay DP-only and avoid RePlAce wall time; do not resurrect without an
  explicit product decision.
- **`legalize_flag: 0`** (global-only) as the second DP in the light bridge was
  **dropped**: it often fights Bookshelf import + challenge legalization and
  proxy density.
- Import **NaN / missing .pl** handling: try alternate `.pl` paths, fall back to
  seed with `RuntimeWarning` (see `_dreamplace_candidates.py`).

**Older summary (still directionally true):** damped blends help, but only at
very small alphas in current runs; true-shape / global-only DP can win WL then
**lose density/congestion** after import and legalization. Bridge tuning matters
more than raw DP iteration counts alone.

Useful DREAMPlace commands:

```bash
python scripts/run_dreamplace_diagnostics.py --benchmark ibm02 \
  --soft-macro-mode row_height --preset random_global \
  --config 0.65:200:0.01:1e-3 --timeout 240

python scripts/run_dreamplace_diagnostics.py --benchmark ibm02 \
  --soft-macro-mode preserve --preset global_only \
  --config 0.65:120:0.01:1e-3 --timeout 240

python scripts/run_dreamplace_diagnostics.py --benchmark ibm01 \
  --bridge-configs --soft-macro-mode aspect_cap
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
2. Keep pushing **DREAMPlace bridge** research via `dreamplace_bridge_placer.py`
   and `run_dreamplace_diagnostics.py` (prove candidates beat CasADi on proxy
   before expecting IBM-average gains). `replace_pipeline_placer.py` remains the
   safe submission path until a DP-only sweep is competitive under the 1h/bench
   cap.
3. Promote only general policies that improve true-proxy aggregate selection;
   never branch on benchmark names.
