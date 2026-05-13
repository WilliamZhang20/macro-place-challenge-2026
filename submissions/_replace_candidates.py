"""Generate challenge placement candidates from RePlAce runs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _replace_bookshelf import BookshelfExport, write_bookshelf  # noqa: E402
from _candidate_select import score_placement  # noqa: E402
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
    adaptive_top_k: int = 0,
    adaptive_probe_timeout_seconds: float | None = None,
    adaptive_full_timeout_seconds: float | None = None,
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

    def run_one(
        config: ReplaceConfig,
        *,
        run_timeout_seconds: float,
        stop_after_first_pl: bool,
    ) -> List[ReplaceCandidate]:
        result = run_replace(
            export,
            config,
            binary_path=binary_path,
            timeout_seconds=run_timeout_seconds,
            stop_after_first_pl=stop_after_first_pl,
        )
        run_results.append(result)
        if not result.usable or (not use_partial_results and not result.ok):
            return []
        out: List[ReplaceCandidate] = []
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
            candidate = ReplaceCandidate(
                placement=placement,
                pl_path=pl_path,
                run_result=result,
                label=_candidate_label(pl_path, result),
                raw_overlap_count=raw_overlap_count,
                final_overlap_count=final_overlap_count,
                legalizer_max_displacement=max_disp,
                legalizer_mean_displacement=mean_disp,
            )
            candidates.append(candidate)
            out.append(candidate)
        return out

    if adaptive_top_k <= 0:
        for config in configs:
            run_one(
                config,
                run_timeout_seconds=float(timeout_seconds),
                stop_after_first_pl=True,
            )
    else:
        probe_timeout = (
            float(adaptive_probe_timeout_seconds)
            if adaptive_probe_timeout_seconds is not None
            else max(8.0, min(75.0, 0.20 * float(timeout_seconds)))
        )
        full_timeout = (
            float(adaptive_full_timeout_seconds)
            if adaptive_full_timeout_seconds is not None
            else float(timeout_seconds)
        )
        probe_scores: List[Tuple[ReplaceConfig, float]] = []
        for config in configs:
            probe_candidates = run_one(
                config,
                run_timeout_seconds=probe_timeout,
                stop_after_first_pl=True,
            )
            best_proxy = _best_valid_proxy(
                probe_candidates,
                benchmark,
                plc,
            )
            if best_proxy < float("inf"):
                probe_scores.append((config, best_proxy))

        promoted = _promoted_replace_configs(
            probe_scores,
            top_k=adaptive_top_k,
            existing=configs,
        )
        for config in promoted:
            run_one(
                config,
                run_timeout_seconds=full_timeout,
                stop_after_first_pl=False,
            )

    return ReplaceCandidateBatch(
        export=export,
        run_results=run_results,
        candidates=candidates,
    )


def _best_valid_proxy(
    candidates: Sequence[ReplaceCandidate],
    benchmark: Benchmark,
    plc,
) -> float:
    best = float("inf")
    for candidate in candidates:
        score = score_placement(candidate.label, candidate.placement, benchmark, plc)
        if score.valid:
            best = min(best, float(score.proxy_cost))
    return best


def _promoted_replace_configs(
    probe_scores: Sequence[Tuple[ReplaceConfig, float]],
    *,
    top_k: int,
    existing: Sequence[ReplaceConfig],
) -> List[ReplaceConfig]:
    if not probe_scores:
        return []
    existing_keys = {_config_key(c) for c in existing}
    ordered = [config for config, _score in sorted(probe_scores, key=lambda item: item[1])]
    promoted: List[ReplaceConfig] = []
    seen = set()

    def add(config: ReplaceConfig) -> None:
        key = _config_key(config)
        if key in seen:
            return
        seen.add(key)
        promoted.append(config)

    for config in ordered[: max(1, int(top_k))]:
        add(config)
        for neighbor in _neighbor_configs(config):
            key = _config_key(neighbor)
            if key not in existing_keys:
                add(neighbor)
    return promoted


def _neighbor_configs(config: ReplaceConfig) -> List[ReplaceConfig]:
    density = float(config.density)
    pcofmax = float(config.pcofmax)
    extra_args = tuple(str(v) for v in config.extra_args)
    out: List[ReplaceConfig] = []
    for delta in (-0.02, 0.02):
        d = min(0.92, max(0.58, density + delta))
        out.append(ReplaceConfig(density=d, pcofmax=pcofmax, extra_args=extra_args))
    for pcof in (1.03, 1.08, 1.20):
        if abs(pcof - pcofmax) > 1e-9:
            out.append(ReplaceConfig(density=density, pcofmax=pcof, extra_args=extra_args))
    if "-bin" not in extra_args:
        out.append(ReplaceConfig(density=density, pcofmax=pcofmax, extra_args=("-bin", "64")))
        out.append(ReplaceConfig(density=density, pcofmax=pcofmax, extra_args=("-bin", "128")))
    return out


def _config_key(config: ReplaceConfig) -> Tuple[float, float, Tuple[str, ...]]:
    return (
        round(float(config.density), 6),
        round(float(config.pcofmax), 6),
        tuple(str(v) for v in config.extra_args),
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
