"""Top-level orchestration for the RePlAce candidate pipeline."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from macro_place.benchmark import Benchmark

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _candidate_select import SelectionResult, select_best_true_proxy  # noqa: E402
from _benchmark_features import benchmark_features  # noqa: E402
from _plc_lookup import PlcLookup  # noqa: E402
from _replace_candidates import ReplaceCandidateBatch, generate_replace_candidates  # noqa: E402
from _replace_runner import ReplaceConfig  # noqa: E402


BaselineProvider = Callable[[Benchmark], torch.Tensor]

_GENERIC_CONFIGS = (
    ReplaceConfig(density=0.70, pcofmax=1.03, extra_args=("-bin", "64")),
    ReplaceConfig(density=0.72, pcofmax=1.03),
    ReplaceConfig(density=0.72, pcofmax=1.03, extra_args=("-bin", "64")),
    ReplaceConfig(density=0.72, pcofmax=1.03, extra_args=("-bin", "128")),
    ReplaceConfig(density=0.78, pcofmax=1.03),
    ReplaceConfig(density=0.80, pcofmax=1.03),
    ReplaceConfig(density=0.80, pcofmax=1.03, extra_args=("-bin", "64")),
    ReplaceConfig(density=0.80, pcofmax=1.03, extra_args=("-bin", "128")),
    ReplaceConfig(density=0.80, pcofmax=1.03, extra_args=("-pcofmin", "0.98")),
    ReplaceConfig(density=0.80, pcofmax=1.20),
    ReplaceConfig(density=0.80, pcofmax=1.20, extra_args=("-bin", "64")),
    ReplaceConfig(density=0.80, pcofmax=1.20, extra_args=("-bin", "128")),
    ReplaceConfig(density=0.84, pcofmax=1.20, extra_args=("-bin", "128")),
    ReplaceConfig(density=0.84, pcofmax=1.03),
    ReplaceConfig(density=0.84, pcofmax=1.03, extra_args=("-bin", "64")),
    ReplaceConfig(density=0.86, pcofmax=1.03),
    ReplaceConfig(density=0.88, pcofmax=1.03),
)

_COMPACT_EXTRA_CONFIGS = (
    ReplaceConfig(density=0.76, pcofmax=1.08),
    ReplaceConfig(density=0.82, pcofmax=1.08),
    ReplaceConfig(density=0.82, pcofmax=1.08, extra_args=("-pcofmin", "0.90")),
    ReplaceConfig(density=0.82, pcofmax=1.08, extra_args=("-pcofmin", "0.98")),
)


@dataclass(frozen=True)
class ReplacePipelineResult:
    """Full accounting for one pipeline invocation."""

    placement: torch.Tensor
    baseline: torch.Tensor
    selection: Optional[SelectionResult]
    candidate_batch: Optional[ReplaceCandidateBatch]
    reason: str = "ok"

    def diagnostics(self, benchmark_name: str | None = None) -> Dict[str, Any]:
        """Return a JSON-serializable run summary for sweep/tuning logs."""

        best_label = self.selection.best.label if self.selection is not None else None
        out: Dict[str, Any] = {
            "benchmark": benchmark_name,
            "reason": self.reason,
            "selected_label": best_label,
            "num_candidates": 0,
            "runs": [],
            "candidates": [],
            "scores": [],
        }
        if self.candidate_batch is not None:
            out["num_candidates"] = len(self.candidate_batch.candidates)
            out["bookshelf_name"] = self.candidate_batch.export.bookshelf_name
            out["runs"] = [
                {
                    "density": float(run.config.density),
                    "pcofmax": float(run.config.pcofmax),
                    "extra_args": [str(arg) for arg in run.config.extra_args],
                    "returncode": int(run.returncode),
                    "timed_out": bool(run.timed_out),
                    "ok": bool(run.ok),
                    "runtime_seconds": float(run.runtime_seconds),
                    "num_pl_paths": len(run.pl_paths),
                    "log_path": str(run.log_path),
                }
                for run in self.candidate_batch.run_results
            ]
            out["candidates"] = [
                {
                    "label": candidate.label,
                    "pl_path": str(candidate.pl_path),
                    "run_density": float(candidate.run_result.config.density),
                    "run_pcofmax": float(candidate.run_result.config.pcofmax),
                    "raw_overlap_count": int(candidate.raw_overlap_count),
                    "final_overlap_count": int(candidate.final_overlap_count),
                    "legalizer_max_displacement": float(
                        candidate.legalizer_max_displacement
                    ),
                    "legalizer_mean_displacement": float(
                        candidate.legalizer_mean_displacement
                    ),
                }
                for candidate in self.candidate_batch.candidates
            ]
        if self.selection is not None:
            out["scores"] = [
                {
                    "label": score.label,
                    "valid": bool(score.valid),
                    "proxy_cost": float(score.proxy_cost),
                    "wirelength": float(score.wirelength),
                    "density": float(score.density),
                    "congestion": float(score.congestion),
                    "overlaps": int(score.overlaps),
                    "violations": list(score.violations),
                }
                for score in self.selection.scores
            ]
        return out


class ReplacePipeline:
    """Generate RePlAce candidates and select against a baseline by true proxy."""

    def __init__(
        self,
        *,
        configs: Sequence[ReplaceConfig] | None = None,
        baseline_provider: BaselineProvider | None = None,
        plc_lookup: PlcLookup | None = None,
        work_root: Path | str | None = None,
        binary_path: Path | str = Path(
            "external/MacroPlacement/Flows/util/RePlAceFlow/RePlAce-static"
        ),
        timeout_seconds: float = 600.0,
        scale: int = 1000,
    ):
        self.configs = list(configs) if configs is not None else None
        self.baseline_provider = baseline_provider or self._default_baseline_provider()
        self.plc_lookup = plc_lookup or PlcLookup()
        self.work_root = Path(work_root) if work_root is not None else None
        self.binary_path = binary_path
        self.timeout_seconds = float(timeout_seconds)
        self.scale = int(scale)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Return only the selected placement for evaluator compatibility."""

        return self.run(benchmark).placement

    def run(self, benchmark: Benchmark) -> ReplacePipelineResult:
        baseline = self.baseline_provider(benchmark).clone().float()
        plc = self.plc_lookup.load(benchmark)
        if plc is None:
            return ReplacePipelineResult(
                placement=baseline,
                baseline=baseline,
                selection=None,
                candidate_batch=None,
                reason="missing_plc",
            )

        configs = self._configs_for(benchmark)
        if not configs:
            return ReplacePipelineResult(
                placement=baseline,
                baseline=baseline,
                selection=None,
                candidate_batch=None,
                reason="no_external_configs",
            )

        try:
            batch = generate_replace_candidates(
                benchmark,
                plc,
                self._work_root_for(benchmark),
                configs,
                bookshelf_name=benchmark.name,
                scale=self.scale,
                binary_path=self.binary_path,
                timeout_seconds=self.timeout_seconds,
                initial_placement=baseline,
            )
        except Exception:
            return ReplacePipelineResult(
                placement=baseline,
                baseline=baseline,
                selection=None,
                candidate_batch=None,
                reason="candidate_generation_failed",
            )

        candidate_tensors = [candidate.placement for candidate in batch.candidates]
        labels = [candidate.label for candidate in batch.candidates]
        try:
            selection = select_best_true_proxy(
                baseline,
                candidate_tensors,
                benchmark,
                plc,
                candidate_labels=labels,
            )
        except Exception:
            return ReplacePipelineResult(
                placement=baseline,
                baseline=baseline,
                selection=None,
                candidate_batch=batch,
                reason="selection_failed",
            )

        return ReplacePipelineResult(
            placement=selection.placement,
            baseline=baseline,
            selection=selection,
            candidate_batch=batch,
            reason="ok",
        )

    def _work_root_for(self, benchmark: Benchmark) -> Path:
        if self.work_root is not None:
            root = self.work_root
        else:
            root = Path(tempfile.gettempdir()) / "macro_place_replace_pipeline"
        path = root / benchmark.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _configs_for(self, benchmark: Benchmark) -> List[ReplaceConfig]:
        if self.configs is not None:
            return list(self.configs)
        configs = list(_GENERIC_CONFIGS)
        features = benchmark_features(benchmark)
        if _is_compact_external_candidate(features):
            configs.extend(_COMPACT_EXTRA_CONFIGS)
        return configs

    @staticmethod
    def _default_baseline_provider() -> BaselineProvider:
        from casadi_placer import CasadiPlacer

        placer = CasadiPlacer()
        return placer.place


def _is_compact_external_candidate(features: Dict[str, Any]) -> bool:
    return (
        float(features.get("grid_rows", 0)) <= 35
        and float(features.get("grid_cols", 0)) <= 40
        and float(features.get("canvas_area", 0.0)) <= 3500.0
    )
