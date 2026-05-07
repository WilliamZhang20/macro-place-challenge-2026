"""Strictly improving greedy search on the evaluator proxy (non-regressive).

Proposes random jiggles / occasional swaps on movable hard macros, legalizes,
and accepts only if ``compute_proxy_cost`` strictly decreases. No benchmark names;
attempt budget scales with macro and net counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _hard_legalizer import legalize_hard  # noqa: E402


def greedy_proxy_attempt_budget(benchmark: Benchmark) -> int:
    nh = int(benchmark.num_hard_macros)
    nn = int(benchmark.num_nets)
    return int(min(650, max(100, nh * 15 + nn // 6)))


def greedy_improve_proxy(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    max_attempts: int,
    seed: int,
) -> torch.Tensor:
    """Return a placement whose proxy is never worse than ``placement``."""

    best = placement.detach().clone().float()
    if benchmark.num_hard_macros <= 0:
        return best

    try:
        best_proxy = float(compute_proxy_cost(best, benchmark, plc)["proxy_cost"])
    except Exception:
        return best

    n_hard = int(benchmark.num_hard_macros)
    movable = ~benchmark.macro_fixed[:n_hard]
    if not bool(movable.any()):
        return best

    idx_list = movable.nonzero(as_tuple=False).squeeze(-1).tolist()
    if not idx_list:
        return best

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    sizes = benchmark.macro_sizes[:n_hard].float()
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    for _ in range(max(1, int(max_attempts))):
        prop = best.clone()
        if len(idx_list) >= 2 and float(rng.random()) < 0.11:
            a, b = rng.choice(len(idx_list), size=2, replace=False)
            ia, ib = idx_list[int(a)], idx_list[int(b)]
            tmp = prop[ia, :2].clone()
            prop[ia, :2] = prop[ib, :2]
            prop[ib, :2] = tmp
        else:
            k = int(idx_list[int(rng.integers(0, len(idx_list)))])
            sigma = float(0.010 + 0.038 * rng.random())
            prop[k, 0] = prop[k, 0] + float(rng.normal(0.0, sigma))
            prop[k, 1] = prop[k, 1] + float(rng.normal(0.0, sigma))
            prop[k, 0] = float(torch.clamp(prop[k, 0], hw[k], cw - hw[k]))
            prop[k, 1] = float(torch.clamp(prop[k, 1], hh[k], ch - hh[k]))

        if bool(benchmark.macro_fixed.any()):
            prop[benchmark.macro_fixed] = benchmark.macro_positions[
                benchmark.macro_fixed
            ].to(prop.dtype)

        prop = legalize_hard(
            prop,
            benchmark,
            overlap_gap=1e-3,
            legalize_rounds=320,
            outer_passes=2,
            displacement_budget_frac=0.14,
            step_fraction=0.28,
        )
        ok, _ = validate_placement(prop, benchmark, check_overlaps=True)
        if not ok:
            continue
        try:
            proxy = float(compute_proxy_cost(prop, benchmark, plc)["proxy_cost"])
        except Exception:
            continue
        if proxy < best_proxy - 1e-12:
            best = prop
            best_proxy = proxy

    return best
