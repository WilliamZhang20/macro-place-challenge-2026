"""Generate challenge placement candidates from RePlAce runs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _replace_bookshelf import BookshelfExport, write_bookshelf  # noqa: E402
from _hard_legalizer import legalize_hard  # noqa: E402
from _replace_import import import_bookshelf_placement  # noqa: E402
from _replace_runner import ReplaceConfig, ReplaceRunResult, run_replace  # noqa: E402


@dataclass(frozen=True)
class ReplaceCandidate:
    """One imported placement produced by one RePlAce output file."""

    placement: torch.Tensor
    pl_path: Path
    run_result: ReplaceRunResult
    label: str
    raw_overlap_count: int = 0
    final_overlap_count: int = 0
    legalizer_max_displacement: float = 0.0
    legalizer_mean_displacement: float = 0.0


@dataclass(frozen=True)
class ReplaceCandidateBatch:
    """All artifacts from an export plus one or more RePlAce runs."""

    export: BookshelfExport
    run_results: List[ReplaceRunResult]
    candidates: List[ReplaceCandidate]


def generate_replace_candidates(
    benchmark: Benchmark,
    plc,
    work_root: Path | str,
    configs: Sequence[ReplaceConfig],
    *,
    bookshelf_name: str | None = None,
    scale: int = 1000,
    binary_path: Path | str = Path(
        "external/MacroPlacement/Flows/util/RePlAceFlow/RePlAce-static"
    ),
    timeout_seconds: float = 600.0,
    initial_placement: torch.Tensor | None = None,
    legalize_imported: bool = True,
    use_partial_results: bool = True,
) -> ReplaceCandidateBatch:
    """Export ``benchmark``, run RePlAce configs, and import all placements.

    This function deliberately does not select or score candidates.  By
    default it does run the shared hard legalizer on imported coordinates,
    because Bookshelf integer round-trips and external placer macro movement can
    create tiny strict-overlap violations.  The returned candidate records keep
    enough legalization accounting for tuning and backend comparison.

    When ``use_partial_results`` is true, placement files from timed-out runs
    are still imported and screened.  True-proxy selection remains the guardrail,
    while diagnostics still record that the backend run was not clean.
    """

    if not configs:
        raise ValueError("at least one RePlAce config is required")

    work_root = Path(work_root)
    bs_name = bookshelf_name or benchmark.name
    export = write_bookshelf(
        benchmark,
        plc,
        work_root / "ETC" / bs_name,
        bookshelf_name=bs_name,
        scale=scale,
        initial_placement=initial_placement,
    )

    run_results: List[ReplaceRunResult] = []
    candidates: List[ReplaceCandidate] = []
    seen_pls = set()

    for config in configs:
        result = run_replace(
            export,
            config,
            binary_path=binary_path,
            timeout_seconds=timeout_seconds,
        )
        run_results.append(result)
        if not result.usable or (not use_partial_results and not result.ok):
            continue
        for pl_path in result.pl_paths:
            resolved = pl_path.resolve()
            if resolved in seen_pls:
                continue
            seen_pls.add(resolved)
            raw_placement = import_bookshelf_placement(
                pl_path,
                export.metadata_path,
                benchmark,
            )
            raw_overlap_count = _overlap_count(raw_placement, benchmark)
            placement = raw_placement
            if legalize_imported:
                placement = legalize_hard(
                    placement,
                    benchmark,
                    overlap_gap=1e-3,
                    legalize_rounds=1800,
                )
                _clamp_to_canvas(placement, benchmark)
            final_overlap_count = _overlap_count(placement, benchmark)
            max_disp, mean_disp = _displacement_stats(raw_placement, placement)
            candidates.append(
                ReplaceCandidate(
                    placement=placement,
                    pl_path=pl_path,
                    run_result=result,
                    label=_candidate_label(pl_path, result),
                    raw_overlap_count=raw_overlap_count,
                    final_overlap_count=final_overlap_count,
                    legalizer_max_displacement=max_disp,
                    legalizer_mean_displacement=mean_disp,
                )
            )

    return ReplaceCandidateBatch(
        export=export,
        run_results=run_results,
        candidates=candidates,
    )


def _overlap_count(placement: torch.Tensor, benchmark: Benchmark) -> int:
    return int(compute_overlap_metrics(placement, benchmark)["overlap_count"])


def _candidate_label(pl_path: Path, run_result: ReplaceRunResult) -> str:
    config = run_result.config
    den = _label_float(config.density)
    pcof = _label_float(config.pcofmax)
    return f"{pl_path.parent.name}_den{den}_pcof{pcof}_{pl_path.name}"


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
