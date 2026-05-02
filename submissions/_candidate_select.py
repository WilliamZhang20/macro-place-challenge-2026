"""True-proxy candidate selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from macro_place.utils import validate_placement


@dataclass(frozen=True)
class ScoredPlacement:
    """Validation and proxy-cost details for one placement candidate."""

    label: str
    placement: torch.Tensor
    valid: bool
    proxy_cost: float
    wirelength: float
    density: float
    congestion: float
    overlaps: int
    violations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionResult:
    """Winner plus score ledger for all considered candidates."""

    best: ScoredPlacement
    scores: List[ScoredPlacement]

    @property
    def placement(self) -> torch.Tensor:
        return self.best.placement


def select_best_true_proxy(
    baseline: torch.Tensor,
    candidates: Sequence[torch.Tensor],
    benchmark: Benchmark,
    plc,
    *,
    candidate_labels: Optional[Sequence[str]] = None,
    require_zero_overlap: bool = True,
) -> SelectionResult:
    """Return the valid placement with lowest evaluator proxy cost.

    ``baseline`` is always evaluated first and acts as the floor when valid.
    Invalid candidates are recorded with ``proxy_cost=inf`` but never selected.
    This function intentionally uses the real ``compute_proxy_cost`` instead of
    a surrogate, because selection is the last line of defense before returning
    a placement from the RePlAce pipeline.
    """

    if candidate_labels is not None and len(candidate_labels) != len(candidates):
        raise ValueError("candidate_labels length must match candidates length")

    labels = list(candidate_labels) if candidate_labels is not None else [
        f"candidate_{i}" for i in range(len(candidates))
    ]

    scored: List[ScoredPlacement] = []
    scored.append(
        score_placement(
            "baseline",
            baseline,
            benchmark,
            plc,
            require_zero_overlap=require_zero_overlap,
        )
    )
    for label, placement in zip(labels, candidates):
        scored.append(
            score_placement(
                label,
                placement,
                benchmark,
                plc,
                require_zero_overlap=require_zero_overlap,
            )
        )

    valid_scores = [s for s in scored if s.valid]
    if not valid_scores:
        raise ValueError("no valid placement candidates, including baseline")

    best = min(valid_scores, key=lambda s: s.proxy_cost)
    return SelectionResult(best=best, scores=scored)


def score_placement(
    label: str,
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    require_zero_overlap: bool = True,
) -> ScoredPlacement:
    """Validate and score one placement by true proxy."""

    ok, violations = validate_placement(
        placement,
        benchmark,
        check_overlaps=require_zero_overlap,
    )
    overlaps = int(compute_overlap_metrics(placement, benchmark)["overlap_count"])
    if require_zero_overlap and overlaps:
        ok = False
        if not any("overlap" in v.lower() for v in violations):
            violations = [*violations, f"{overlaps} hard macro overlaps"]

    if not ok:
        return ScoredPlacement(
            label=label,
            placement=placement.clone(),
            valid=False,
            proxy_cost=float("inf"),
            wirelength=float("inf"),
            density=float("inf"),
            congestion=float("inf"),
            overlaps=overlaps,
            violations=tuple(violations),
        )

    costs = compute_proxy_cost(placement, benchmark, plc)
    return ScoredPlacement(
        label=label,
        placement=placement.clone(),
        valid=True,
        proxy_cost=float(costs["proxy_cost"]),
        wirelength=float(costs["wirelength_cost"]),
        density=float(costs["density_cost"]),
        congestion=float(costs["congestion_cost"]),
        overlaps=int(costs["overlap_count"]),
        violations=(),
    )
