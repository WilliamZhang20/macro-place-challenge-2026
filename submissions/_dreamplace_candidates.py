"""Generate challenge placement candidates from DREAMPlace runs."""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _dreamplace_runner import (  # noqa: E402
    DreamPlaceConfig,
    DreamPlaceRunResult,
    run_dreamplace,
)
from _hard_legalizer import legalize_hard  # noqa: E402
from _replace_bookshelf import BookshelfExport, write_bookshelf  # noqa: E402
from _replace_import import import_bookshelf_placement  # noqa: E402


@dataclass(frozen=True)
class DreamPlaceCandidate:
    """One imported placement produced by one DREAMPlace output file."""

    placement: torch.Tensor
    pl_path: Path
    run_result: DreamPlaceRunResult
    label: str
    raw_overlap_count: int = 0
    final_overlap_count: int = 0
    legalizer_max_displacement: float = 0.0
    legalizer_mean_displacement: float = 0.0


@dataclass(frozen=True)
class DreamPlaceCandidateBatch:
    """All artifacts from an export plus one or more DREAMPlace runs."""

    export: BookshelfExport
    run_results: List[DreamPlaceRunResult]
    candidates: List[DreamPlaceCandidate]


def generate_dreamplace_candidates(
    benchmark: Benchmark,
    plc,
    work_root: Path | str,
    configs: Sequence[DreamPlaceConfig],
    *,
    bookshelf_name: str | None = None,
    scale: int = 1000,
    dreamplace_root: Path | str = Path("external/DREAMPlace"),
    placer_path: Path | str | None = None,
    timeout_seconds: float = 600.0,
    initial_placement: torch.Tensor | None = None,
    soft_macro_mode: str = "preserve",
    soft_macro_row_cap_mult: float = 12.0,
    blend_alphas: Sequence[float] = (),
    blend_hard_only: bool = True,
    legalize_imported: bool = True,
    legalize_rounds: int = 1800,
    legalize_outer_passes: int = 1,
    legalize_displacement_budget_frac: float | None = None,
    legalize_step_fraction: float = 0.35,
    legalize_iterative_cycles: int = 1,
    legalize_refine_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_partial_results: bool = True,
) -> DreamPlaceCandidateBatch:
    """Export ``benchmark``, run DREAMPlace configs, and import placements."""

    if not configs:
        raise ValueError("at least one DREAMPlace config is required")

    work_root = Path(work_root)
    bs_name = bookshelf_name or benchmark.name
    export = write_bookshelf(
        benchmark,
        plc,
        work_root / "ETC" / bs_name,
        bookshelf_name=bs_name,
        scale=scale,
        include_route=False,
        include_shapes=False,
        soft_macro_mode=soft_macro_mode,
        soft_macro_row_cap_mult=soft_macro_row_cap_mult,
        initial_placement=initial_placement,
    )

    run_results: List[DreamPlaceRunResult] = []
    candidates: List[DreamPlaceCandidate] = []
    seen_pls = set()

    for config in configs:
        result = run_dreamplace(
            export,
            config,
            dreamplace_root=dreamplace_root,
            placer_path=placer_path,
            timeout_seconds=timeout_seconds,
            work_root=work_root / "dreamplace",
        )
        run_results.append(result)
        if not result.usable or (not use_partial_results and not result.ok):
            continue
        if not result.pl_paths:
            continue

        chosen_pl: Path | None = None
        raw_placement: torch.Tensor | None = None
        for pl_path in result.pl_paths:
            trial = _try_import_dreamplace_placement(
                pl_path, export.metadata_path, benchmark
            )
            if trial is not None:
                raw_placement = trial
                chosen_pl = pl_path
                break

        import_fallback = False
        if raw_placement is None:
            if initial_placement is None:
                warnings.warn(
                    f"DREAMPlace run produced no finite import "
                    f"(log {result.log_path})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            warnings.warn(
                f"DREAMPlace .pl failed import; using initial_placement fallback "
                f"(log {result.log_path})",
                RuntimeWarning,
                stacklevel=2,
            )
            raw_placement = initial_placement.clone().float()
            chosen_pl = result.pl_paths[0]
            import_fallback = True

        assert chosen_pl is not None
        resolved = chosen_pl.resolve()
        if resolved in seen_pls:
            continue
        seen_pls.add(resolved)

        base_label = _candidate_label(chosen_pl, result)
        if import_fallback:
            base_label += "_import_fallback"
        candidate_inputs: List[Tuple[torch.Tensor, str]] = [
            (raw_placement, base_label)
        ]
        if initial_placement is not None:
            for alpha in blend_alphas:
                alpha_f = float(alpha)
                if not (0.0 < alpha_f < 1.0):
                    raise ValueError("blend_alphas must be in (0, 1)")
                blended = raw_placement.clone()
                if blend_hard_only:
                    nh = benchmark.num_hard_macros
                    blended[:nh] = (1.0 - alpha_f) * initial_placement[:nh].to(
                        blended.dtype
                    ) + alpha_f * raw_placement[:nh].to(blended.dtype)
                else:
                    blended = (1.0 - alpha_f) * initial_placement.to(
                        raw_placement.dtype
                    ) + alpha_f * raw_placement
                suffix = "_blend" + _label_float(alpha_f)
                candidate_inputs.append((blended, f"{base_label}{suffix}"))

        for raw_candidate, label in candidate_inputs:
            raw_overlap_count = _overlap_count(raw_candidate, benchmark)
            placement = raw_candidate
            if legalize_imported:
                n_cycles = max(1, int(legalize_iterative_cycles))
                for cycle in range(n_cycles):
                    placement = legalize_hard(
                        placement,
                        benchmark,
                        overlap_gap=1e-3,
                        legalize_rounds=int(legalize_rounds),
                        outer_passes=int(legalize_outer_passes),
                        displacement_budget_frac=legalize_displacement_budget_frac,
                        step_fraction=float(legalize_step_fraction),
                    )
                    if (
                        legalize_refine_fn is not None
                        and cycle + 1 < n_cycles
                    ):
                        placement = legalize_refine_fn(placement)
            _clamp_to_canvas(placement, benchmark)
            final_overlap_count = _overlap_count(placement, benchmark)
            max_disp, mean_disp = _displacement_stats(raw_candidate, placement)
            candidates.append(
                DreamPlaceCandidate(
                    placement=placement,
                    pl_path=chosen_pl,
                    run_result=result,
                    label=label,
                    raw_overlap_count=raw_overlap_count,
                    final_overlap_count=final_overlap_count,
                    legalizer_max_displacement=max_disp,
                    legalizer_mean_displacement=mean_disp,
                )
            )

    return DreamPlaceCandidateBatch(
        export=export,
        run_results=run_results,
        candidates=candidates,
    )


def _try_import_dreamplace_placement(
    pl_path: Path,
    metadata_path: Path,
    benchmark: Benchmark,
) -> torch.Tensor | None:
    try:
        t = import_bookshelf_placement(pl_path, metadata_path, benchmark)
    except (ValueError, OSError, KeyError, json.JSONDecodeError):
        return None
    if t.shape != benchmark.macro_positions.shape:
        return None
    if not bool(torch.isfinite(t).all().item()):
        return None
    if not _placement_is_reasonable(t, benchmark):
        return None
    return t


def _placement_is_reasonable(t: torch.Tensor, benchmark: Benchmark) -> bool:
    span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
    if span <= 0:
        return True
    return bool((t.abs() <= span * 50.0).all().item())


def _overlap_count(placement: torch.Tensor, benchmark: Benchmark) -> int:
    return int(compute_overlap_metrics(placement, benchmark)["overlap_count"])


def _candidate_label(pl_path: Path, run_result: DreamPlaceRunResult) -> str:
    config = run_result.config
    den = _label_float(config.target_density)
    return f"{pl_path.parent.name}_dream_den{den}_iter{int(config.iterations)}_{pl_path.name}"


def _label_float(value: float) -> str:
    return f"{float(value):.6g}".replace(".", "p")


def _displacement_stats(before: torch.Tensor, after: torch.Tensor) -> tuple[float, float]:
    if before.numel() == 0:
        return 0.0, 0.0
    disp = torch.linalg.vector_norm((after - before).float(), dim=1)
    return float(disp.max().item()), float(disp.mean().item())


def _clamp_to_canvas(placement: torch.Tensor, benchmark: Benchmark) -> None:
    sizes = benchmark.macro_sizes.to(dtype=placement.dtype, device=placement.device)
    inset = torch.full_like(placement[:, 0], 1e-5)
    half_w = 0.5 * sizes[:, 0]
    half_h = 0.5 * sizes[:, 1]
    min_x = half_w + inset
    max_x = float(benchmark.canvas_width) - half_w - inset
    min_y = half_h + inset
    max_y = float(benchmark.canvas_height) - half_h - inset
    placement[:, 0] = torch.minimum(torch.maximum(placement[:, 0], min_x), max_x)
    placement[:, 1] = torch.minimum(torch.maximum(placement[:, 1], min_y), max_y)
    if benchmark.macro_fixed.any():
        placement[benchmark.macro_fixed] = benchmark.macro_positions[
            benchmark.macro_fixed
        ].to(dtype=placement.dtype, device=placement.device)
