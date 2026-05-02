"""Feature extraction for general placement policy decisions."""

from __future__ import annotations

import torch

from macro_place.benchmark import Benchmark


def benchmark_features(benchmark: Benchmark) -> dict:
    """Return JSON-friendly benchmark features, without using benchmark names."""

    sizes = benchmark.macro_sizes.float()
    area = sizes[:, 0] * sizes[:, 1]
    canvas_area = float(benchmark.canvas_width) * float(benchmark.canvas_height)
    hard_area = area[: benchmark.num_hard_macros]
    soft_area = area[benchmark.num_hard_macros :]
    movable = (~benchmark.macro_fixed).float()
    fixed_count = int(benchmark.macro_fixed.sum().item())
    degrees = torch.tensor(
        [int(nodes.numel()) for nodes in benchmark.net_nodes],
        dtype=torch.float32,
    )
    return {
        "canvas_area": canvas_area,
        "canvas_aspect": _safe_div(float(benchmark.canvas_width), float(benchmark.canvas_height)),
        "num_macros": int(benchmark.num_macros),
        "num_hard_macros": int(benchmark.num_hard_macros),
        "num_soft_macros": int(benchmark.num_soft_macros),
        "fixed_macro_fraction": _safe_div(fixed_count, int(benchmark.num_macros)),
        "hard_macro_fraction": _safe_div(int(benchmark.num_hard_macros), int(benchmark.num_macros)),
        "macro_area_utilization": _safe_div(float(area.sum().item()), canvas_area),
        "hard_area_utilization": _safe_div(float(hard_area.sum().item()), canvas_area),
        "soft_area_utilization": _safe_div(float(soft_area.sum().item()), canvas_area),
        "mean_macro_area": float(area.mean().item()) if area.numel() else 0.0,
        "max_macro_area": float(area.max().item()) if area.numel() else 0.0,
        "macro_area_cv": _coefficient_of_variation(area),
        "num_nets": int(benchmark.num_nets),
        "mean_net_degree": float(degrees.mean().item()) if degrees.numel() else 0.0,
        "max_net_degree": int(degrees.max().item()) if degrees.numel() else 0,
        "movable_fraction": float(movable.mean().item()) if movable.numel() else 0.0,
        "grid_cols": int(benchmark.grid_cols),
        "grid_rows": int(benchmark.grid_rows),
    }


def _coefficient_of_variation(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    mean = float(values.mean().item())
    if abs(mean) < 1e-12:
        return 0.0
    return float(values.std(unbiased=False).item()) / mean


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0
