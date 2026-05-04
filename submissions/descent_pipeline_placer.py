"""
Unified analytical descent placer: smooth HPWL + FFT density + soft Rudy,
Lloyd alternation with legalization, adaptive multi-start (true-proxy selector),
LNS with periodic HPWL-only polish, and a final anchored refinement pass.

Tuning is derived only from benchmark features (macro/net counts, utilization,
canvas size). No benchmark-name rules.

Usage:
  evaluate submissions/descent_pipeline_placer.py -b ibm01
  evaluate submissions/descent_pipeline_placer.py --all
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple, Optional
import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

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
    try_load_iccad04_plc,
)


def _hard_utilization(benchmark: Benchmark) -> float:
    nh = benchmark.num_hard_macros
    a = benchmark.macro_sizes[:nh, 0] * benchmark.macro_sizes[:nh, 1]
    return float(a.sum() / (benchmark.canvas_width * benchmark.canvas_height))


class PipelineConfig(NamedTuple):
    num_starts: int
    outer_iters: int
    inner0: int
    inner1: int
    lr: float
    lr_end: float
    density_grid: int
    lam_start: float
    lam_end: float
    cong_weight: float
    lns_budget_scale: float
    lns_iters: int
    lns_lr: float
    lns_lam_end: float
    lns_cong: float
    polish_every: int
    polish_iters: int
    polish_lr: float
    final_pass: bool
    final_iters: int
    final_lr: float
    final_mu: float
    noise_frac: float


def derive_config(benchmark: Benchmark) -> PipelineConfig:
    nh = int(benchmark.num_hard_macros)
    nn = int(benchmark.num_nets)
    util = _hard_utilization(benchmark)

    est = 95.0 + 0.41 * nh + 0.0010 * nn + 195.0 * max(0.0, util - 0.43)
    est *= 1.05 + 0.35 * max(0.0, util - 0.50)
    # Stay under ~50 min/bench including final polish + multistart (1h contest cap).
    target = 2850.0
    ns = int(np.clip(target / max(est, 72.0), 2.0, 8.0))
    if nh > 495:
        ns = min(ns, 4)
    elif nh > 430:
        ns = min(ns, 5)
    elif nh < 265:
        ns = max(ns, 6)
    while ns * est > 2700.0 and ns > 2:
        ns -= 1

    outer = 5 if util > 0.486 else 4
    inner0 = int(500 + 55 * max(0.0, util - 0.45) + 0.04 * max(0, nh - 320))
    inner1 = int(348 + 40 * max(0.0, util - 0.46))

    lam_end = 0.245 + 0.20 * max(0.0, util - 0.415)
    lam_end = float(np.clip(lam_end, 0.22, 0.48))

    cong_w = 0.305 + 0.26 * max(0.0, util - 0.415)
    cong_w = float(np.clip(cong_w, 0.28, 0.48))

    lns_scale = 0.88 + 0.34 * (0.53 - util)
    lns_scale = float(np.clip(lns_scale, 0.74, 1.32))

    lr = float(np.clip(0.0465 - 0.0048 * max(0.0, util - 0.46), 0.038, 0.048))
    lr_end = lr * 0.52

    lns_iters = int(np.clip(108 + 0.05 * nh, 100, 155))
    noise = float(np.clip(0.0078 + 0.006 * max(0.0, 0.50 - util), 0.0065, 0.014))

    return PipelineConfig(
        num_starts=ns,
        outer_iters=outer,
        inner0=inner0,
        inner1=inner1,
        lr=lr,
        lr_end=lr_end,
        density_grid=64,
        lam_start=0.0065,
        lam_end=lam_end,
        cong_weight=cong_w,
        lns_budget_scale=lns_scale,
        lns_iters=lns_iters,
        lns_lr=0.051,
        lns_lam_end=float(np.clip(0.21 + 0.12 * max(0.0, util - 0.44), 0.19, 0.34)),
        lns_cong=float(np.clip(0.19 + 0.14 * max(0.0, util - 0.43), 0.16, 0.34)),
        polish_every=3,
        polish_iters=95,
        polish_lr=0.036,
        final_pass=True,
        final_iters=int(295 + 0.22 * nh),
        final_lr=0.032,
        final_mu=float(np.clip(0.11 + 0.35 * max(0.0, util - 0.44), 0.09, 0.55)),
        noise_frac=noise,
    )


def _run_single_start(
    benchmark: Benchmark,
    init: torch.Tensor,
    cfg: PipelineConfig,
    rng_seed: int,
    *,
    deadline: Optional[float] = None,
    lns_cap_sec: Optional[float] = None,
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    free_idx = movable_hard_indices(benchmark)
    if not free_idx:
        return init.clone()

    beta = 1.0 / (0.0083 * max(benchmark.canvas_width, benchmark.canvas_height))
    batch = build_net_pin_batch(benchmark, device)
    cr, cc = congestion_grid_shape(benchmark)
    grid_div = legal_grid_div_heuristic(benchmark)
    cur = init.to(device=device, dtype=torch.float32)
    project_inside_canvas(cur, benchmark)

    for outer in range(cfg.outer_iters):
        if deadline is not None and time.time() > deadline:
            break
        if outer == 0:
            inner_iters = cfg.inner0
            mu_s, mu_e = 0.0, 0.0
            anchor = None
        else:
            inner_iters = cfg.inner1
            mu_s = 0.045 + 0.012 * float(outer - 1)
            mu_e = 0.38 + 0.11 * float(outer - 1) + 0.06 * max(
                0.0, _hard_utilization(benchmark) - 0.46
            )
            anchor = cur.clone()
        cur = global_adam_optimize(
            benchmark,
            cur,
            free_idx,
            batch,
            iters=inner_iters,
            lr=cfg.lr,
            lr_end=cfg.lr_end,
            beta=beta,
            density_grid=cfg.density_grid,
            lam_start=cfg.lam_start,
            lam_end=cfg.lam_end,
            anchor=anchor,
            mu_anchor_start=mu_s,
            mu_anchor_end=mu_e,
            cong_weight=cfg.cong_weight,
            cong_rows=cr,
            cong_cols=cc,
        )
        cur = finalize_hard_legalization(
            cur.cpu(), benchmark, grid_div=grid_div
        ).to(device)

    rng = np.random.default_rng(rng_seed + len(free_idx))
    budget = lns_time_budget_sec(benchmark) * cfg.lns_budget_scale
    if lns_cap_sec is not None:
        budget = min(budget, lns_cap_sec)
    t0 = time.time()
    nh = len(free_idx)
    rnd = 0

    while time.time() - t0 < budget:
        if deadline is not None and time.time() > deadline:
            break
        k_hi = min(30, max(8, nh // 4))
        k_lo = max(2, min(15, nh // 5))
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
            iters=cfg.lns_iters,
            lr=cfg.lns_lr,
            lr_end=cfg.lns_lr * 0.55,
            beta=beta,
            density_grid=cfg.density_grid,
            lam_start=0.018,
            lam_end=cfg.lns_lam_end,
            anchor=None,
            mu_anchor_start=0.0,
            mu_anchor_end=0.0,
            cong_weight=cfg.lns_cong,
            cong_rows=cr,
            cong_cols=cc,
        )
        cur = finalize_hard_legalization(
            cur.cpu(), benchmark, grid_div=grid_div
        ).to(device)

        rnd += 1
        if rnd % cfg.polish_every == 0:
            cur = global_adam_optimize(
                benchmark,
                cur,
                free_idx,
                batch,
                iters=cfg.polish_iters,
                lr=cfg.polish_lr,
                lr_end=cfg.polish_lr * 0.5,
                beta=beta,
                density_grid=cfg.density_grid,
                lam_start=0.01,
                lam_end=0.09,
                anchor=None,
                mu_anchor_start=0.0,
                mu_anchor_end=0.0,
                cong_weight=0.0,
                cong_rows=cr,
                cong_cols=cc,
            )
            cur = finalize_hard_legalization(
                cur.cpu(), benchmark, grid_div=grid_div
            ).to(device)

    return cur.cpu()


def _final_anchor_polish(
    benchmark: Benchmark,
    placement: torch.Tensor,
    cfg: PipelineConfig,
    *,
    deadline: Optional[float] = None,
) -> torch.Tensor:
    if not cfg.final_pass:
        return placement
    if deadline is not None and time.time() > deadline:
        return placement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    free_idx = movable_hard_indices(benchmark)
    if not free_idx:
        return placement
    beta = 1.0 / (0.0083 * max(benchmark.canvas_width, benchmark.canvas_height))
    batch = build_net_pin_batch(benchmark, device)
    cr, cc = congestion_grid_shape(benchmark)
    grid_div = legal_grid_div_heuristic(benchmark)
    cur = placement.to(device=device, dtype=torch.float32)
    anchor = cur.clone()
    cur = global_adam_optimize(
        benchmark,
        cur,
        free_idx,
        batch,
        iters=cfg.final_iters,
        lr=cfg.final_lr,
        lr_end=cfg.final_lr * 0.48,
        beta=beta,
        density_grid=cfg.density_grid,
        lam_start=0.04,
        lam_end=0.20,
        anchor=anchor,
        mu_anchor_start=cfg.final_mu * 0.25,
        mu_anchor_end=cfg.final_mu,
        cong_weight=0.26,
        cong_rows=cr,
        cong_cols=cc,
    )
    return finalize_hard_legalization(cur.cpu(), benchmark, grid_div=grid_div)


class DescentPipelinePlacer:
    """Production entry point for the descent stack."""

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_wall0 = time.time()
        wall_limit = 3480.0
        deadline = t_wall0 + wall_limit

        cfg = derive_config(benchmark)
        plc = try_load_iccad04_plc(benchmark)
        free_idx = movable_hard_indices(benchmark)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        canvas = float(max(benchmark.canvas_width, benchmark.canvas_height))
        noise_scale = cfg.noise_frac * canvas

        per_start_lns = (wall_limit - 420.0 - cfg.final_iters * 0.12) / max(
            cfg.num_starts, 1
        )
        per_start_lns = float(np.clip(per_start_lns, 28.0, 420.0))

        best: torch.Tensor | None = None
        best_cost = float("inf")

        for s in range(cfg.num_starts):
            if time.time() > deadline - 90.0:
                break
            init = benchmark.macro_positions.clone()
            if free_idx and s > 0:
                rng = np.random.default_rng(13007 + s * 10009 + len(free_idx))
                noise = rng.normal(0.0, noise_scale, (len(free_idx), 2))
                idx_t = torch.tensor(free_idx, dtype=torch.long)
                init[idx_t] = init[idx_t] + torch.from_numpy(noise).to(init.dtype)
                tmp = init.to(device=device, dtype=torch.float32)
                project_inside_canvas(tmp, benchmark)
                init = tmp.cpu()

            out = _run_single_start(
                benchmark,
                init,
                cfg,
                rng_seed=s * 31,
                deadline=deadline,
                lns_cap_sec=per_start_lns,
            )
            if plc is not None:
                cost = float(compute_proxy_cost(out, benchmark, plc)["proxy_cost"])
                if cost < best_cost:
                    best_cost = cost
                    best = out.clone()
            elif best is None:
                best = out.clone()

        assert best is not None
        best = _final_anchor_polish(benchmark, best, cfg, deadline=deadline)
        return best
