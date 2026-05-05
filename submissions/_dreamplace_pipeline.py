"""Multi-start DREAMPlace + true-proxy selection (feature-aware caps, no benchmark names)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from macro_place.benchmark import Benchmark

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _benchmark_features import benchmark_features  # noqa: E402
from _candidate_select import (  # noqa: E402
    SelectionResult,
    select_best_true_proxy_candidates_only,
)
from _dreamplace_cpu_smoke import (  # noqa: E402
    default_dreamplace_install,
    dreamplace_install_ok,
    run_dreamplace_placement,
)
from _hard_legalizer import legalize_hard  # noqa: E402
from _plc_lookup import PlcLookup  # noqa: E402


def _tuner_progress_enabled() -> bool:
    return os.environ.get("MACRO_PLACE_TUNER_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _clamp_centers(placement: torch.Tensor, benchmark: Benchmark) -> None:
    n = benchmark.num_hard_macros
    if n <= 0:
        return
    movable = ~benchmark.macro_fixed[:n]
    hw = benchmark.macro_sizes[:n, 0] * 0.5
    hh = benchmark.macro_sizes[:n, 1] * 0.5
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    gap = 1e-3
    if bool(movable.any()):
        placement[:n, 0] = torch.where(
            movable,
            torch.clamp(placement[:n, 0], hw + gap, cw - hw - gap),
            placement[:n, 0],
        )
        placement[:n, 1] = torch.where(
            movable,
            torch.clamp(placement[:n, 1], hh + gap, ch - hh - gap),
            placement[:n, 1],
        )
    if benchmark.macro_fixed.any():
        placement[benchmark.macro_fixed] = benchmark.macro_positions[
            benchmark.macro_fixed
        ].to(placement.dtype)


def jitter_hard_centers(
    base: torch.Tensor,
    benchmark: Benchmark,
    *,
    sigma_um: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Gaussian perturbation on movable hard macro centers (µm scale)."""

    out = base.clone()
    n = benchmark.num_hard_macros
    if n <= 0 or sigma_um <= 0:
        return out
    movable = ~benchmark.macro_fixed[:n]
    if not bool(movable.any()):
        return out
    idx = movable.nonzero(as_tuple=False).squeeze(-1)
    noise = torch.randn(
        (idx.numel(), 2),
        device=base.device,
        dtype=base.dtype,
        generator=generator,
    )
    out[idx, :2] = out[idx, :2] + float(sigma_um) * noise
    _clamp_centers(out, benchmark)
    return out


def cap_num_starts(benchmark: Benchmark, requested: int) -> int:
    """Fewer parallel DREAMPlace starts on very large hard-macro counts (runtime)."""

    nh = int(benchmark_features(benchmark)["num_hard_macros"])
    if nh >= 500:
        cap = 2
    elif nh >= 380:
        cap = 3
    else:
        cap = 4
    return max(1, min(int(requested), cap))


def scaled_global_iterations(benchmark: Benchmark, base_iters: int) -> int:
    """Mild feature-based iteration stretch (utilization / size), capped."""

    f = benchmark_features(benchmark)
    util = float(f["hard_area_utilization"])
    nh = int(f["num_hard_macros"])
    mult = 1.0 + 0.12 * max(0.0, util - 0.5) + 0.08 * max(0, nh - 280) / 220.0
    return int(round(float(base_iters) * min(mult, 1.28)))


def _rich_dp_variant_specs(
    benchmark: Benchmark,
    *,
    target_density: float,
    num_bins: int,
) -> List[Tuple[float, int, str, Dict[str, Any]]]:
    """(target_density, num_bins, label_tag, extra_json) modes from utilization / scale only."""

    f = benchmark_features(benchmark)
    util = float(f["hard_area_utilization"])
    td0 = float(target_density)
    b0 = int(num_bins)
    alt_bins = 64 if b0 >= 96 else 128
    # High utilization: encourage spreading (lower target density).
    td_spread = max(0.64, min(0.90, td0 - 0.06 * max(0.0, (util - 0.46) / 0.12)))
    # Low utilization: allow slightly tighter packing.
    td_tight = max(0.64, min(0.90, td0 + 0.05 * max(0.0, (0.50 - util) / 0.10)))
    specs: List[Tuple[float, int, str, Dict[str, Any]]] = [
        (max(0.64, min(0.90, td0)), b0, "base", {}),
        (td_spread, alt_bins, "spread", {}),
        (td_tight, b0, "tight", {}),
    ]
    # Slight density-objective emphasis on spread mode when utilization is stressed.
    if util >= 0.50:
        dw_scale = 1.0 + 0.2 * min(1.0, (util - 0.50) / 0.06)
        specs[1] = (
            specs[1][0],
            specs[1][1],
            specs[1][2],
            {"density_weight_scale": dw_scale},
        )
    return specs


def _apply_density_weight_scale(
    overrides: Dict[str, Any], scale: float
) -> Dict[str, Any]:
    if scale == 1.0:
        return overrides
    out = dict(overrides)
    base_dw = out.get("density_weight")
    if base_dw is not None:
        try:
            out["density_weight"] = float(base_dw) * float(scale)
        except (TypeError, ValueError):
            pass
    else:
        # Default from _dp_json is 8e-5; scale relative to that if user did not set.
        out["density_weight"] = float(8e-5) * float(scale)
    return out


@dataclass(frozen=True)
class DreamPlacePipelineResult:
    """``initial_handoff`` is the loader `.plc` placement (returned if DP yields nothing valid)."""

    placement: torch.Tensor
    initial_handoff: torch.Tensor
    selection: Optional[SelectionResult]
    reason: str

    def diagnostics(self, benchmark_name: str | None = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "benchmark": benchmark_name,
            "reason": self.reason,
        }
        if self.selection is not None:
            out["selected_label"] = self.selection.best.label
            out["scores"] = [
                {
                    "label": s.label,
                    "valid": s.valid,
                    "proxy_cost": float(s.proxy_cost),
                    "overlaps": int(s.overlaps),
                }
                for s in self.selection.scores
            ]
        return out


class DreamPlacePipeline:
    """Multi-start DREAMPlace only; best true proxy among DP runs (no external baseline placer)."""

    def __init__(
        self,
        *,
        plc_lookup: PlcLookup | None = None,
        dreamplace_install: Path | str | None = None,
        num_starts: int = 4,
        jitter_sigma_um: float = 0.028,
        global_iterations: int = 100,
        num_bins: int = 128,
        num_threads: int = 8,
        target_density: float = 0.76,
        timeout_seconds: float = 720.0,
        dreamplace_json_overrides: Optional[Mapping[str, Any]] = None,
        use_gpu: Optional[bool] = None,
        scale_iterations_with_features: bool = True,
        rich_candidate_set: bool = False,
    ):
        self.plc_lookup = plc_lookup or PlcLookup()
        self.dreamplace_install = dreamplace_install
        self.num_starts = int(num_starts)
        self.jitter_sigma_um = float(jitter_sigma_um)
        self.global_iterations = int(global_iterations)
        self.num_bins = int(num_bins)
        self.num_threads = int(num_threads)
        self.target_density = float(target_density)
        self.timeout_seconds = float(timeout_seconds)
        self.dreamplace_json_overrides = (
            dict(dreamplace_json_overrides) if dreamplace_json_overrides else None
        )
        self.use_gpu = use_gpu
        self.scale_iterations_with_features = bool(scale_iterations_with_features)
        self.rich_candidate_set = bool(rich_candidate_set)

    @staticmethod
    def _repair_seed(seed: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Best-effort overlap/bounds repair when DREAMPlace cannot run or select."""

        return legalize_hard(
            seed.clone(),
            benchmark,
            legalize_rounds=1200,
            overlap_gap=1e-3,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self.run(benchmark).placement

    def run(self, benchmark: Benchmark) -> DreamPlacePipelineResult:
        seed = benchmark.macro_positions.clone().float()
        plc = self.plc_lookup.load(benchmark)
        inst = (
            Path(self.dreamplace_install)
            if self.dreamplace_install is not None
            else default_dreamplace_install()
        )

        if plc is None:
            fb = self._repair_seed(seed, benchmark)
            return DreamPlacePipelineResult(
                placement=fb,
                initial_handoff=seed,
                selection=None,
                reason="missing_plc",
            )
        ok, _ = dreamplace_install_ok(inst)
        if not ok:
            fb = self._repair_seed(seed, benchmark)
            return DreamPlacePipelineResult(
                placement=fb,
                initial_handoff=seed,
                selection=None,
                reason="dreamplace_install_missing",
            )

        starts = cap_num_starts(benchmark, self.num_starts)
        iters = (
            scaled_global_iterations(benchmark, self.global_iterations)
            if self.scale_iterations_with_features
            else self.global_iterations
        )

        candidates: List[torch.Tensor] = []
        labels: List[str] = []
        gen = torch.Generator(device=seed.device)
        gen.manual_seed(2026 + int(benchmark.num_hard_macros) + int(benchmark.num_nets))

        variant_specs: Sequence[Tuple[float, int, str, Dict[str, Any]]]
        if self.rich_candidate_set:
            variant_specs = _rich_dp_variant_specs(
                benchmark,
                target_density=self.target_density,
                num_bins=self.num_bins,
            )
        else:
            variant_specs = (
                (self.target_density, self.num_bins, "base", {}),
            )

        for k in range(starts):
            if k == 0:
                init = seed
            else:
                init = jitter_hard_centers(
                    seed,
                    benchmark,
                    sigma_um=self.jitter_sigma_um,
                    generator=gen,
                )
            td_k, bins_k, tag_k, extra_k = variant_specs[k % len(variant_specs)]
            overrides: Dict[str, Any] = (
                dict(self.dreamplace_json_overrides)
                if self.dreamplace_json_overrides
                else {}
            )
            scale = float(extra_k.get("density_weight_scale", 1.0))
            extra_clean = {a: b for a, b in extra_k.items() if a != "density_weight_scale"}
            overrides.update(extra_clean)
            overrides = _apply_density_weight_scale(overrides, scale)
            overrides["random_seed"] = int(9000 + k * 9973 + benchmark.num_macros)

            label = f"dp_{tag_k}_k{k}_seed{overrides['random_seed']}"
            if _tuner_progress_enabled():
                print(
                    f"[tune:dp] {benchmark.name}  Placer {k + 1}/{starts}  "
                    f"iters={iters}  bins={bins_k}  td={td_k:.3f}  tag={tag_k}  "
                    f"timeout={self.timeout_seconds:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
            dp_out = run_dreamplace_placement(
                benchmark,
                plc,
                dreamplace_install=inst,
                global_iterations=iters,
                num_bins=int(bins_k),
                num_threads=self.num_threads,
                target_density=float(td_k),
                timeout_seconds=self.timeout_seconds,
                dreamplace_json_overrides=overrides,
                use_gpu=self.use_gpu,
                initial_placement=init,
            )
            if _tuner_progress_enabled():
                print(
                    f"[tune:dp] {benchmark.name}  Placer {k + 1}/{starts}  "
                    f"finished  placement={'ok' if dp_out is not None else 'None'}",
                    file=sys.stderr,
                    flush=True,
                )
            if dp_out is not None:
                candidates.append(dp_out)
                labels.append(label)

        if not candidates:
            fb = self._repair_seed(seed, benchmark)
            return DreamPlacePipelineResult(
                placement=fb,
                initial_handoff=seed,
                selection=None,
                reason="all_dreamplace_starts_failed",
            )

        try:
            selection = select_best_true_proxy_candidates_only(
                candidates,
                benchmark,
                plc,
                candidate_labels=labels,
            )
        except ValueError:
            fb = self._repair_seed(seed, benchmark)
            return DreamPlacePipelineResult(
                placement=fb,
                initial_handoff=seed,
                selection=None,
                reason="no_valid_dreamplace_candidate",
            )
        except Exception:
            fb = self._repair_seed(seed, benchmark)
            return DreamPlacePipelineResult(
                placement=fb,
                initial_handoff=seed,
                selection=None,
                reason="selection_failed",
            )

        return DreamPlacePipelineResult(
            placement=selection.placement,
            initial_handoff=seed,
            selection=selection,
            reason="ok",
        )

