#!/usr/bin/env python3
"""
Measure proxy cost and wall time for incremental descent phases on one ICCAD04 case.

Phases (each builds on the previous idea, not on the previous placement):
  0 — initial .plc proxy (reference)
  1 — single-start Lloyd only (global + legalize per outer, LNS disabled)
  2 — single-start full (adds LNS + periodic HPWL polish)
  3 — adds final anchored polish pass (matches production tail)
  4 — two random starts + proxy pick (scaled-down multistart)

Example:
  python scripts/descent_stage_profile.py -b ibm01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SUB = ROOT / "submissions"
if str(SUB) not in sys.path:
    sys.path.insert(0, str(SUB))

import numpy as np
import torch

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from _descent_core import (  # noqa: E402
    movable_hard_indices,
    project_inside_canvas,
    try_load_iccad04_plc,
)
from descent_pipeline_placer import (  # noqa: E402
    _final_anchor_polish,
    _run_single_start,
    derive_config,
)


def _proxy(placement, benchmark, plc) -> float:
    return float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-b",
        "--benchmark",
        default="ibm01",
        help="ICCAD04 benchmark name (under Testcases/ICCAD04)",
    )
    parser.add_argument(
        "--max-phase",
        type=int,
        default=4,
        help="Run phases 0..max_phase inclusive (default 4)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Scale down iterations/budgets (~5–8× faster) for A/B trends, not absolute proxy",
    )
    args = parser.parse_args()

    bench_dir = ROOT / "external/MacroPlacement/Testcases/ICCAD04" / args.benchmark
    if not bench_dir.is_dir():
        print(f"Missing {bench_dir}", file=sys.stderr)
        sys.exit(1)

    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    plc_ref = try_load_iccad04_plc(benchmark)
    if plc_ref is None:
        plc_ref = plc

    cfg = derive_config(benchmark)
    if args.quick:
        cfg = cfg._replace(
            outer_iters=min(2, cfg.outer_iters),
            inner0=max(64, min(110, cfg.inner0 // 2)),
            inner1=max(56, min(95, cfg.inner1 // 2)),
            lns_iters=max(40, cfg.lns_iters - 36),
            lns_budget_scale=cfg.lns_budget_scale * 0.38,
            polish_iters=max(28, cfg.polish_iters // 2),
            polish_every=max(7, cfg.polish_every + 2),
            final_iters=max(72, cfg.final_iters // 2),
            legal_lloyd_rounds=72,
            legal_lns_rounds=58,
            legal_lns_grid_cap=3200,
            legal_final_rounds=95,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    free_idx = movable_hard_indices(benchmark)
    canvas = float(max(benchmark.canvas_width, benchmark.canvas_height))
    noise_scale = cfg.noise_frac * canvas

    print(
        f"benchmark={args.benchmark}  nh={benchmark.num_hard_macros}  starts_cfg={cfg.num_starts}  quick={args.quick}",
        flush=True,
    )
    print(f"{'phase':<6} {'description':<42} {'sec':>8} {'proxy':>10}", flush=True)
    print("-" * 70, flush=True)

    t0 = time.time()
    base = benchmark.macro_positions.clone()
    p0 = _proxy(base, benchmark, plc_ref)
    print(f"{0:<6} {'initial placement':<42} {time.time() - t0:8.2f} {p0:10.4f}", flush=True)

    if args.max_phase < 1:
        return

    t1 = time.time()
    out1 = _run_single_start(
        benchmark, base.clone(), cfg, rng_seed=0, deadline=None, lns_cap_sec=0.0
    )
    print(
        f"{1:<6} {'+ Lloyd (LNS off)':<42} {time.time() - t1:8.2f} {_proxy(out1, benchmark, plc_ref):10.4f}",
        flush=True,
    )

    if args.max_phase < 2:
        return

    t2 = time.time()
    out2 = _run_single_start(
        benchmark,
        base.clone(),
        cfg,
        rng_seed=1,
        deadline=None,
        lns_cap_sec=None,
    )
    print(
        f"{2:<6} {'+ LNS + polish (single start)':<42} {time.time() - t2:8.2f} {_proxy(out2, benchmark, plc_ref):10.4f}",
        flush=True,
    )

    if args.max_phase < 3:
        return

    t3 = time.time()
    out3 = _final_anchor_polish(benchmark, out2.clone(), cfg, deadline=None)
    print(
        f"{3:<6} {'+ final anchor polish':<42} {time.time() - t3:8.2f} {_proxy(out3, benchmark, plc_ref):10.4f}",
        flush=True,
    )

    if args.max_phase < 4:
        return

    t4 = time.time()
    best_c = float("inf")
    best_p = out3
    for s in range(2):
        init = benchmark.macro_positions.clone()
        if free_idx and s > 0:
            rng = np.random.default_rng(13007 + s * 10009)
            noise = rng.normal(0.0, noise_scale, (len(free_idx), 2))
            idx_t = torch.tensor(free_idx, dtype=torch.long)
            init[idx_t] = init[idx_t] + torch.from_numpy(noise).to(init.dtype)
            tmp = init.to(device=device, dtype=torch.float32)
            project_inside_canvas(tmp, benchmark)
            init = tmp.cpu()
        o = _run_single_start(
            benchmark, init, cfg, rng_seed=s * 31, deadline=None, lns_cap_sec=None
        )
        o = _final_anchor_polish(benchmark, o, cfg, deadline=None)
        c = _proxy(o, benchmark, plc_ref)
        if c < best_c:
            best_c = c
            best_p = o
    print(
        f"{4:<6} {'+ multistart (2) + final':<42} {time.time() - t4:8.2f} {_proxy(best_p, benchmark, plc_ref):10.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
