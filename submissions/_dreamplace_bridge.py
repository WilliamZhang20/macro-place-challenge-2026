"""Utilization-driven DREAMPlace config grids for the challenge bridge."""

from __future__ import annotations

import os

from macro_place.benchmark import Benchmark

from _benchmark_features import benchmark_features
from _dreamplace_presets import dreamplace_preset_params
from _dreamplace_runner import DreamPlaceConfig


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def utilization_density_triplet(macro_area_utilization: float) -> tuple[float, float, float]:
    """Three target_density values from low→high as utilization increases.

    Low-util designs tolerate softer targets; packed designs need 0.85+ on the
    high corner.  We then **blend toward utilization-anchored targets** as
    macro fill rises: asking DREAMPlace for ~0.60 effective density when the
    floorplan is ~0.5 macro fill makes the global solution incompatible with the
    challenge proxy density after import/legalization (CLAUDE.md bridge gap).
    """
    u = _clamp01(macro_area_utilization)
    t = _clamp01((u - 0.32) / 0.38)
    d_lo = 0.60 + 0.06 * t
    d_mid = 0.68 + 0.12 * t
    d_hi = 0.74 + 0.18 * t
    if u >= 0.55:
        d_hi = max(d_hi, 0.85)
        d_mid = max(d_mid, 0.78)

    mix = _clamp01((u - 0.44) / 0.30)
    u_lo = max(0.58, min(0.88, u + 0.06 + 0.04 * (1.0 - u)))
    u_mid = max(0.64, min(0.91, u + 0.11 + 0.05 * (1.0 - u)))
    u_hi = max(0.70, min(0.96, u + 0.17 + 0.06 * (1.0 - u)))
    d_lo = d_lo * (1.0 - mix) + u_lo * mix
    d_mid = d_mid * (1.0 - mix) + u_mid * mix
    d_hi = d_hi * (1.0 - mix) + u_hi * mix
    return (d_lo, d_mid, d_hi)


def _bridge_extra_params(preset: str) -> dict:
    """Bookshelf-safe defaults; optional IBM macro knobs (can destabilize some builds)."""

    out = dict(dreamplace_preset_params(preset))
    if os.environ.get("MACRO_PLACE_DP_MACRO", "").lower() in ("1", "true", "yes"):
        out.update(
            {
                "macro_place_flag": 1,
                "two_stage_density_scaler": 1000,
            }
        )
    return out


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
    extra = _bridge_extra_params(preset)
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
    """Two standard DP corners (utilization-anchor, fine high-density).

    Avoid ``legalize_flag:0`` here: global-only outputs often import poorly into
    the challenge legalizer/proxy. Diversity comes from the placer using two
    Bookshelf seeds (see ``dreamplace_bridge_placer``).
    """

    util = float(benchmark_features(benchmark)["macro_area_utilization"])
    extra = _bridge_extra_params(preset)
    util_anchor = DreamPlaceConfig(
        target_density=max(0.66, min(0.86, util)),
        num_bins_x=64,
        num_bins_y=64,
        iterations=int(iterations),
        learning_rate=0.01,
        density_weight=5e-5,
        gpu=bool(gpu),
        extra_params=dict(extra),
    )
    spread_anchor = DreamPlaceConfig(
        target_density=max(0.74, min(0.92, util + 0.12)),
        num_bins_x=128,
        num_bins_y=128,
        iterations=min(int(iterations), 130),
        learning_rate=0.01,
        density_weight=1.2e-4,
        gpu=bool(gpu),
        extra_params=dict(extra),
    )
    full = bridge_dreamplace_configs(
        benchmark, preset=preset, iterations=iterations, gpu=gpu
    )
    if len(full) < 6:
        return [util_anchor, spread_anchor, *full]
    return [util_anchor, spread_anchor]
