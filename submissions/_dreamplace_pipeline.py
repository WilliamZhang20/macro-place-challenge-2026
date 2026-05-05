"""Multi-start DREAMPlace + true-proxy selection (feature-aware caps, no benchmark names)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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
from _plc_lookup import PlcLookup  # noqa: E402


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
            return DreamPlacePipelineResult(
                placement=seed,
                initial_handoff=seed,
                selection=None,
                reason="missing_plc",
            )
        ok, _ = dreamplace_install_ok(inst)
        if not ok:
            return DreamPlacePipelineResult(
                placement=seed,
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
            overrides: Dict[str, Any] = (
                dict(self.dreamplace_json_overrides)
                if self.dreamplace_json_overrides
                else {}
            )
            overrides["random_seed"] = int(9000 + k * 9973 + benchmark.num_macros)

            label = f"dp_k{k}_seed{overrides['random_seed']}"
            dp_out = run_dreamplace_placement(
                benchmark,
                plc,
                dreamplace_install=inst,
                global_iterations=iters,
                num_bins=self.num_bins,
                num_threads=self.num_threads,
                target_density=self.target_density,
                timeout_seconds=self.timeout_seconds,
                dreamplace_json_overrides=overrides,
                use_gpu=self.use_gpu,
                initial_placement=init,
            )
            if dp_out is not None:
                candidates.append(dp_out)
                labels.append(label)

        if not candidates:
            return DreamPlacePipelineResult(
                placement=seed,
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
            return DreamPlacePipelineResult(
                placement=seed,
                initial_handoff=seed,
                selection=None,
                reason="no_valid_dreamplace_candidate",
            )
        except Exception:
            return DreamPlacePipelineResult(
                placement=seed,
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

