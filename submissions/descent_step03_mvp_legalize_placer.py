"""
Step 3 — MVP: Step-2 global objective + coarse grid min-displacement legalizer.

First scoreable zero-overlap baseline for the descent chain.

Usage:
  uv run evaluate submissions/descent_step03_mvp_legalize_placer.py -b ibm01
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
    fft_density_energy,
    finalize_hard_legalization,
    legal_grid_div_heuristic,
    movable_hard_indices,
    project_inside_canvas,
    scatter_free_into_placement,
    smooth_hpwl_loss,
)


class DescentStep03MvpLegalizePlacer:
    def __init__(self):
        self.iters = 480
        self.lr = 0.045
        self.density_grid = 64
        self.lam_start = 0.008
        self.lam_end = 0.32

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = benchmark.macro_positions.to(device=device, dtype=torch.float32)
        free_idx = movable_hard_indices(benchmark)
        if not free_idx:
            return base.cpu()

        beta = 1.0 / (0.0085 * max(benchmark.canvas_width, benchmark.canvas_height))
        batch = build_net_pin_batch(benchmark, device)
        free_xy = torch.nn.Parameter(base[free_idx].clone())
        opt = torch.optim.Adam([free_xy], lr=self.lr)

        for t in range(self.iters):
            alpha = t / max(self.iters - 1, 1)
            lam = self.lam_start + alpha * (self.lam_end - self.lam_start)
            opt.zero_grad(set_to_none=True)
            pl = scatter_free_into_placement(base, free_idx, free_xy)
            loss = smooth_hpwl_loss(pl, benchmark, batch, beta=beta)
            loss = loss + lam * fft_density_energy(
                pl, benchmark, grid_n=self.density_grid, base=base
            )
            loss.backward()
            opt.step()
            with torch.no_grad():
                pl = scatter_free_into_placement(base, free_idx, free_xy)
                project_inside_canvas(pl, benchmark)
                free_xy.copy_(pl[free_idx])

        out = scatter_free_into_placement(base, free_idx, free_xy).detach()
        project_inside_canvas(out, benchmark)
        grid_div = legal_grid_div_heuristic(benchmark)
        return finalize_hard_legalization(out.cpu(), benchmark, grid_div=grid_div)
