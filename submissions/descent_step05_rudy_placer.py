"""
Step 5 — Same Lloyd loop as step 4 plus differentiable soft Rudy (proxy-shaped).

Congestion weight ramps inside each global phase; grid size follows benchmark
routing grid (clamped).

Usage:
  uv run evaluate submissions/descent_step05_rudy_placer.py -b ibm01
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    movable_hard_indices,
)


class DescentStep05RudyPlacer:
    def __init__(self):
        self.outer_iters = 4
        self.lr = 0.042
        self.density_grid = 64
        self.lam_start = 0.008
        self.lam_end = 0.30
        self.cong_weight = 0.38

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        free_idx = movable_hard_indices(benchmark)
        if not free_idx:
            return benchmark.macro_positions.clone()

        beta = 1.0 / (0.0085 * max(benchmark.canvas_width, benchmark.canvas_height))
        batch = build_net_pin_batch(benchmark, device)
        cr, cc = congestion_grid_shape(benchmark)
        grid_div = legal_grid_div_heuristic(benchmark)
        cur = benchmark.macro_positions.to(device=device, dtype=torch.float32)

        for outer in range(self.outer_iters):
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
                lr=self.lr,
                beta=beta,
                density_grid=self.density_grid,
                lam_start=self.lam_start,
                lam_end=self.lam_end,
                anchor=anchor,
                mu_anchor_start=mu_s,
                mu_anchor_end=mu_e,
                cong_weight=self.cong_weight,
                cong_rows=cr,
                cong_cols=cc,
            )
            cur = finalize_hard_legalization(
                cur.cpu(), benchmark, grid_div=grid_div
            ).to(device)

        return cur.cpu()
