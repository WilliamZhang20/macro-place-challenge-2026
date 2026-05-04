"""Utilization-driven DREAMPlace config grids for the challenge bridge."""

from __future__ import annotations

from macro_place.benchmark import Benchmark

from _benchmark_features import benchmark_features
from _dreamplace_presets import dreamplace_preset_params
from _dreamplace_runner import DreamPlaceConfig


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def utilization_density_triplet(macro_area_utilization: float) -> tuple[float, float, float]:
    """Three target_density values from low→high as utilization increases.

    Low-util designs tolerate aggressive under-target density during global place;
    packed designs need targets in the 0.85+ range so post-import legalization
    does not destroy the global solution.
    """
    u = _clamp01(macro_area_utilization)
    t = _clamp01((u - 0.32) / 0.38)
    d_lo = 0.60 + 0.06 * t
    d_mid = 0.68 + 0.12 * t
    d_hi = 0.74 + 0.18 * t
    if u >= 0.55:
        d_hi = max(d_hi, 0.85)
        d_mid = max(d_mid, 0.78)
    return (d_lo, d_mid, d_hi)


def bridge_dreamplace_configs(
    benchmark: Benchmark,
    *,
    preset: str = "basic",
    iterations: int = 200,
    learning_rate: float = 0.01,
    gpu: bool = False,
) -> list[DreamPlaceConfig]:
    """Six (density × bin × gamma) points for true-proxy ensemble selection."""

    util = float(benchmark_features(benchmark)["macro_area_utilization"])
    d_lo, d_mid, d_hi = utilization_density_triplet(util)
    extra = dreamplace_preset_params(preset)
    gammas = (5e-5, 8e-5, 1.2e-4)
    triples: list[tuple[float, int, float]] = [
        (d_lo, 64, gammas[0]),
        (d_lo, 128, gammas[1]),
        (d_mid, 64, gammas[2]),
        (d_mid, 128, gammas[0]),
        (d_hi, 64, gammas[1]),
        (d_hi, 128, gammas[2]),
    ]
    return [
        DreamPlaceConfig(
            target_density=density,
            num_bins_x=bins,
            num_bins_y=bins,
            iterations=int(iterations),
            learning_rate=float(learning_rate),
            density_weight=float(gamma),
            gpu=bool(gpu),
            extra_params=dict(extra),
        )
        for density, bins, gamma in triples
    ]


def light_bridge_dreamplace_configs(
    benchmark: Benchmark,
    *,
    preset: str = "basic",
    iterations: int = 170,
    gpu: bool = False,
) -> list[DreamPlaceConfig]:
    """Two DP points (low- and high-density corners of the six-grid).

    Keeps bin/gamma diversity while respecting a tight per-benchmark budget
    (competition limit is 1h/bench; README #2 DreamPlace++ ~37s/bench on GPU).
    """

    full = bridge_dreamplace_configs(
        benchmark, preset=preset, iterations=iterations, gpu=gpu
    )
    if len(full) < 6:
        return full
    return [full[i] for i in (0, 5)]
