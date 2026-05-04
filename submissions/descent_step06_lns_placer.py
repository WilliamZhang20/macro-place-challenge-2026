"""
Step 6 — Step 5 pipeline + LNS: random subsets of hard macros, short global,
re-legalize under a time budget.

Usage:
  uv run evaluate submissions/descent_step06_lns_placer.py -b ibm01
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark

_SUB = Path(__file__).resolve().parent
if str(_SUB) not in sys.path:
    sys.path.insert(0, str(_SUB))

from _descent_core import (  # noqa: E402
    build_net_pin_batch,
    congestion_grid_shape,
    finalize_hard_legalization,
    global_adam_optimize,
    legal_grid_div_heuristic,
    lns_time_budget_sec,
    movable_hard_indices,
    project_inside_canvas,
)


def run_lns_placer_from(
    benchmark: Benchmark,
    init: torch.Tensor,
    *,
    rng_seed: int = 0,
    enable_lns: bool = True,
) -> torch.Tensor:
    """Core step-6 logic from an initial full placement (CPU tensor)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    free_idx = movable_hard_indices(benchmark)
    if not free_idx:
        return init.clone()

    beta = 1.0 / (0.0085 * max(benchmark.canvas_width, benchmark.canvas_height))
    batch = build_net_pin_batch(benchmark, device)
    cr, cc = congestion_grid_shape(benchmark)
    grid_div = legal_grid_div_heuristic(benchmark)
    cur = init.to(device=device, dtype=torch.float32)
    project_inside_canvas(cur, benchmark)

    outer_iters = 4
    lr = 0.042
    density_grid = 64
    lam_start = 0.008
    lam_end = 0.30
    cong_weight = 0.38

    for outer in range(outer_iters):
        if outer == 0:
            inner_iters, mu_s, mu_e = 480, 0.0, 0.0
            anchor = None
        else:
            inner_iters, mu_s, mu_e = 360, 0.05, 0.40 + 0.1 * float(outer - 1)
            anchor = cur.clone()
        cur = global_adam_optimize(
            benchmark,
            cur,
            free_idx,
            batch,
            iters=inner_iters,
            lr=lr,
            beta=beta,
            density_grid=density_grid,
            lam_start=lam_start,
            lam_end=lam_end,
            anchor=anchor,
            mu_anchor_start=mu_s,
            mu_anchor_end=mu_e,
            cong_weight=cong_weight,
            cong_rows=cr,
            cong_cols=cc,
        )
        cur = finalize_hard_legalization(
            cur.cpu(), benchmark, grid_div=grid_div
        ).to(device)

    if not enable_lns:
        return cur.cpu()

    rng = np.random.default_rng(rng_seed + len(free_idx))
    budget = lns_time_budget_sec(benchmark)
    t0 = time.time()
    nh = len(free_idx)

    lns_lr = 0.05
    lns_iters = 120
    lns_lam_end = 0.26
    lns_cong = 0.24

    while time.time() - t0 < budget:
        k_hi = min(28, max(8, nh // 4))
        k_lo = max(2, min(14, nh // 5))
        if k_lo > k_hi:
            k_lo, k_hi = k_hi, k_lo
        k = int(rng.integers(k_lo, k_hi + 1))
        k = min(k, nh)
        pick = rng.choice(np.array(free_idx, dtype=np.int64), size=k, replace=False)
        subset = [int(x) for x in pick.tolist()]
        cur = global_adam_optimize(
            benchmark,
            cur,
            subset,
            batch,
            iters=lns_iters,
            lr=lns_lr,
            beta=beta,
            density_grid=density_grid,
            lam_start=0.02,
            lam_end=lns_lam_end,
            anchor=None,
            mu_anchor_start=0.0,
            mu_anchor_end=0.0,
            cong_weight=lns_cong,
            cong_rows=cr,
            cong_cols=cc,
        )
        cur = finalize_hard_legalization(
            cur.cpu(), benchmark, grid_div=grid_div
        ).to(device)

    return cur.cpu()


class DescentStep06LnsPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return run_lns_placer_from(
            benchmark,
            benchmark.macro_positions.clone(),
            rng_seed=0,
            enable_lns=True,
        )
