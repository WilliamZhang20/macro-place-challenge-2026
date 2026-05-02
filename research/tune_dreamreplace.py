"""Fast tuning harness for global_dreamRePlace.

Loads ibm01 once, exposes a single function that runs ONE start with a
given config and prints per-iter trajectory + final true proxy.  Skip the
CasADi baseline and skip the full multi-start so iterations are fast.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions"))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402

import global_dreamRePlace as gdr  # noqa: E402
from _hard_legalizer import legalize_hard  # noqa: E402

BENCH_DIR = ROOT / "external/MacroPlacement/Testcases/ICCAD04/ibm01"


def load_once():
    benchmark, plc = load_benchmark_from_dir(str(BENCH_DIR))
    return benchmark, plc


def run_one_config(benchmark, plc, cfg, *, init_pos=None, verbose=True):
    """Run a single ePlace pass and return (final_pos, final_proxy, trajectory).

    Trajectory is a list of (t, hpwl, density, overflow, density_w, gamma).
    Patches gdr.GlobalDreamRePlacer._one_start to record telemetry.
    """
    placer = gdr.GlobalDreamRePlacer(max_seconds=600.0, configs=[cfg])
    placer._start = time.monotonic()
    flat_net, flat_kind, flat_idx, port_xy = placer._flat_pins(benchmark)
    if init_pos is None:
        init_pos = benchmark.macro_positions.clone().float()
    rudy = placer._static_rudy(init_pos, benchmark)

    traj = []
    orig = gdr.GlobalDreamRePlacer._one_start

    def _instrumented(self, baseline, bench, fn, fk, fi, ports, rudy_map, c, seed):
        import math as _math
        iters, lr, gamma_frac, density_w0, rudy_w, hard_pair_w, init_mode = c
        torch.manual_seed(20260501 + seed)
        dev = self.device
        dtype = torch.float32
        cw = float(bench.canvas_width)
        ch = float(bench.canvas_height)
        sizes = bench.macro_sizes.detach().to(device=dev, dtype=dtype)
        hw = 0.5 * sizes[:, 0]
        hh = 0.5 * sizes[:, 1]
        movable = (~bench.macro_fixed).detach().to(device=dev)
        fixed_pos = bench.macro_positions.detach().to(device=dev, dtype=dtype)
        pos = self._init_positions(baseline, bench, init_mode, dev, dtype)
        pos[~movable] = fixed_pos[~movable]
        pos = torch.nn.Parameter(pos)
        canvas_scale = _math.hypot(cw, ch)
        gamma0 = max(1e-3, float(gamma_frac) * canvas_scale)
        d_rows = min(self.density_grid_max, max(8, int(bench.grid_rows) // 2))
        d_cols = min(self.density_grid_max, max(8, int(bench.grid_cols) // 2))
        bin_w = cw / d_cols
        bin_h = ch / d_rows
        bin_cx = (torch.arange(d_cols, device=dev, dtype=dtype) + 0.5) * bin_w
        bin_cy = (torch.arange(d_rows, device=dev, dtype=dtype) + 0.5) * bin_h
        total_area = float((sizes[:, 0] * sizes[:, 1]).sum().item())
        target_density = max(0.1, min(1.0, total_area / max(1e-9, cw * ch)))
        target_density = 1.05 * target_density
        fn = fn.to(device=dev, dtype=torch.long)
        fk = fk.to(device=dev, dtype=torch.long)
        fi = fi.to(device=dev, dtype=torch.long)
        ports = ports.to(device=dev, dtype=dtype)
        rudy_map = rudy_map.to(device=dev, dtype=dtype)
        opt = torch.optim.Adam([pos], lr=float(lr), betas=(0.9, 0.999))
        density_w = float(density_w0)
        prev_overflow = 1.0
        gamma = gamma0
        n_hard = int(bench.num_hard_macros)
        hard_idx = torch.arange(n_hard, device=dev, dtype=torch.long)
        hard_movable = movable[:n_hard]
        for t in range(int(iters)):
            opt.zero_grad()
            hpwl = self._wa_hpwl(pos, fn, fk, fi, ports, bench.num_nets, gamma, canvas_scale)
            density, overflow = self._bin_density(
                pos, sizes, bin_cx, bin_cy, bin_w, bin_h, target_density
            )
            rudy_loss = self._rudy_pressure(pos, sizes, bench, rudy_map, dev, dtype)
            boundary = self._boundary(pos, hw, hh, cw, ch)
            hard_pair = self._hard_pair_penalty(pos, sizes, hard_idx, hard_movable, n_hard)
            loss = (
                hpwl
                + density_w * density
                + float(rudy_w) * rudy_loss
                + float(hard_pair_w) * hard_pair
                + 50.0 * boundary
            )
            loss.backward()
            with torch.no_grad():
                if (~movable).any():
                    pos.grad[~movable] = 0.0
            opt.step()
            with torch.no_grad():
                pos[:, 0].clamp_(hw + 1e-4, cw - hw - 1e-4)
                pos[:, 1].clamp_(hh + 1e-4, ch - hh - 1e-4)
                if (~movable).any():
                    pos[~movable] = fixed_pos[~movable]
            ov = float(overflow.detach().item())
            stalled = abs(prev_overflow - ov) < 1e-3
            prev_overflow = ov
            if ov > 0.10:
                density_w *= 1.06 if not stalled else 1.30
            else:
                density_w *= 1.02
            density_w = min(density_w, 50.0)
            gamma = gamma0 * max(0.10, min(1.0, ov + 0.10))
            traj.append((
                t,
                float(hpwl.detach().item()),
                float(density.detach().item()),
                ov,
                density_w,
                gamma,
                float(hard_pair.detach().item()),
            ))
        out = pos.detach().to(dtype=torch.float32, device="cpu").clone()
        if bench.macro_fixed.any():
            out[bench.macro_fixed] = bench.macro_positions[bench.macro_fixed].to(out.dtype)
        return out

    gdr.GlobalDreamRePlacer._one_start = _instrumented
    try:
        cand = placer._one_start(
            init_pos, benchmark, flat_net, flat_kind, flat_idx, port_xy, rudy, cfg, 0
        )
        legal = placer._legalize(cand, benchmark)
        ok, _ = validate_placement(legal, benchmark, check_overlaps=True)
        valid = ok and int(compute_overlap_metrics(legal, benchmark)["overlap_count"]) == 0
        if valid:
            costs = compute_proxy_cost(legal, benchmark, plc)
            proxy = float(costs["proxy_cost"])
            wl = float(costs["wirelength_cost"])
            den = float(costs["density_cost"])
            cong = float(costs["congestion_cost"])
        else:
            proxy = float("inf")
            wl = den = cong = float("nan")
    finally:
        gdr.GlobalDreamRePlacer._one_start = orig

    if verbose:
        # Print every ~10% of iters
        n = len(traj)
        step = max(1, n // 10)
        print(f"  iter | hpwl     density  overflow  d_w   gamma   hard_pair")
        for r in traj[::step]:
            print(f"  {r[0]:4d} | {r[1]:8.4f} {r[2]:8.4f} {r[3]:8.4f} {r[4]:6.2f} {r[5]:6.3f} {r[6]:9.4f}")
        if traj and traj[-1][0] % step != 0:
            r = traj[-1]
            print(f"  {r[0]:4d} | {r[1]:8.4f} {r[2]:8.4f} {r[3]:8.4f} {r[4]:6.2f} {r[5]:6.3f} {r[6]:9.4f}")
        print(f"  -> proxy={proxy:.4f}  wl={wl:.3f} den={den:.3f} cong={cong:.3f}  valid={valid}")
    return cand, legal, proxy, traj


def run_grid(configs):
    bench, plc = load_once()
    print(f"loaded {bench.name}: {bench.num_hard_macros} hard, {bench.num_soft_macros} soft, {bench.num_nets} nets")
    print(f"canvas {bench.canvas_width:.1f} x {bench.canvas_height:.1f}, grid {bench.grid_rows} x {bench.grid_cols}")
    print(f"\n{'name':30s} {'proxy':>8s} {'wl':>6s} {'den':>6s} {'cong':>6s} {'time':>6s}")
    print("-" * 70)
    results = []
    for name, cfg in configs:
        t0 = time.monotonic()
        _, legal, proxy, _ = run_one_config(bench, plc, cfg, verbose=False)
        if np.isfinite(proxy):
            costs = compute_proxy_cost(legal, bench, plc)
            wl = float(costs["wirelength_cost"])
            den = float(costs["density_cost"])
            cong = float(costs["congestion_cost"])
        else:
            wl = den = cong = float("nan")
        dt = time.monotonic() - t0
        print(f"{name:30s} {proxy:8.4f} {wl:6.3f} {den:6.3f} {cong:6.3f} {dt:6.1f}s")
        results.append((name, cfg, proxy))
    return results


if __name__ == "__main__":
    # Probe: same iter budget, vary init strategy and jitter scale.
    # Budget 150 iters for fast feedback.
    grid = [
        ("jitter_initial 0.04",   (150, 0.045, 0.045, 0.10, 0.020, 0.30, "jitter_initial")),
        ("jitter_initial 0.02",   (150, 0.025, 0.045, 0.10, 0.020, 0.30, "jitter_initial_small")),
        ("jitter_initial 0.005",  (150, 0.015, 0.045, 0.10, 0.020, 0.30, "jitter_initial_tiny")),
        ("jitter_baseline 0.02",  (150, 0.025, 0.045, 0.10, 0.020, 0.30, "jitter_baseline")),
        ("scatter_center",        (150, 0.045, 0.045, 0.10, 0.020, 0.30, "scatter_center")),
        ("from_initial 0jitter",  (150, 0.020, 0.045, 0.10, 0.020, 0.30, "jitter_initial_zero")),
    ]
    run_grid(grid)
