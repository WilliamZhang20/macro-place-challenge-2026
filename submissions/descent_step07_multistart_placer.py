"""
Step 7 — Multi-start: run step-6 chain from M perturbed initializations; keep
best placement by true proxy (PlacementCost) when the ICCAD04 testcase is available.

Usage:
  uv run evaluate submissions/descent_step07_multistart_placer.py -b ibm01
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

_SUB = Path(__file__).resolve().parent
if str(_SUB) not in sys.path:
    sys.path.insert(0, str(_SUB))

from _descent_core import (  # noqa: E402
    movable_hard_indices,
    project_inside_canvas,
    try_load_iccad04_plc,
)
from descent_step06_lns_placer import run_lns_placer_from  # noqa: E402


class DescentStep07MultistartPlacer:
    def __init__(self):
        self.num_starts = 4
        self.noise_frac = 0.0105

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        free_idx = movable_hard_indices(benchmark)
        plc = try_load_iccad04_plc(benchmark)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        canvas = float(max(benchmark.canvas_width, benchmark.canvas_height))
        noise_scale = self.noise_frac * canvas

        best_pl: torch.Tensor | None = None
        best_cost = float("inf")

        for s in range(self.num_starts):
            init = benchmark.macro_positions.clone()
            if free_idx and s > 0:
                rng = np.random.default_rng(10007 + s * 9973)
                noise = rng.normal(0.0, noise_scale, (len(free_idx), 2))
                idx_t = torch.tensor(free_idx, dtype=torch.long)
                init[idx_t] = init[idx_t] + torch.from_numpy(noise).to(init.dtype)
                tmp = init.to(device=device, dtype=torch.float32)
                project_inside_canvas(tmp, benchmark)
                init = tmp.cpu()

            out = run_lns_placer_from(
                benchmark, init, rng_seed=s * 17, enable_lns=True
            )
            if plc is not None:
                cost = float(compute_proxy_cost(out, benchmark, plc)["proxy_cost"])
                if cost < best_cost:
                    best_cost = cost
                    best_pl = out.clone()
            elif best_pl is None:
                best_pl = out.clone()

        assert best_pl is not None
        return best_pl
