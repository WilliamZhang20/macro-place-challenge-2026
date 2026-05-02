"""Fast Rudy-style routing congestion estimator.

For each net we compute a pin/port bounding box and distribute the net's wire
demand uniformly across the bins it spans.  The result is normalized by the
benchmark's per-direction routing capacity and reduced via per-bin
``max(h_util, v_util)``.

This is a surrogate for the proxy congestion term, not a calibrated estimator.
The intended use is to bias hard-macro move/site selection so the surrogate
tracks roughly the same regions the proxy congestion penalizes, instead of the
density-only signal LNS and coordinate descent currently use.
"""

from __future__ import annotations

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def compute_rudy_map(placement: torch.Tensor, benchmark: Benchmark) -> np.ndarray:
    """Return a [grid_rows, grid_cols] routing-utilization map.

    Larger values = more routing pressure relative to capacity.  Values are
    not bounded by 1.0; calibration is left to the caller.
    """
    rows = int(benchmark.grid_rows)
    cols = int(benchmark.grid_cols)
    if rows <= 0 or cols <= 0:
        return np.zeros((max(1, rows), max(1, cols)), dtype=np.float64)

    bin_w = float(benchmark.canvas_width) / cols
    bin_h = float(benchmark.canvas_height) / rows
    if bin_w <= 0.0 or bin_h <= 0.0:
        return np.zeros((rows, cols), dtype=np.float64)

    pos = placement.detach().cpu().numpy().astype(np.float64, copy=False)
    n_macros = int(benchmark.num_macros)

    if benchmark.port_positions.numel() > 0:
        ports = benchmark.port_positions.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        ports = np.zeros((0, 2), dtype=np.float64)
    n_ports = int(ports.shape[0])

    if benchmark.net_weights.numel() > 0:
        weights = benchmark.net_weights.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        weights = None

    h_demand = np.zeros((rows, cols), dtype=np.float64)
    v_demand = np.zeros((rows, cols), dtype=np.float64)

    inv_bin_w = 1.0 / bin_w
    inv_bin_h = 1.0 / bin_h

    for net_idx, net in enumerate(benchmark.net_nodes):
        if hasattr(net, "numel"):
            if net.numel() < 2:
                continue
            nodes_np = net.detach().cpu().numpy()
        else:
            nodes_np = np.asarray(net)
            if nodes_np.size < 2:
                continue

        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for u in nodes_np:
            u = int(u)
            if 0 <= u < n_macros:
                px = pos[u, 0]
                py = pos[u, 1]
            elif n_ports > 0:
                p = u - n_macros
                if 0 <= p < n_ports:
                    px = ports[p, 0]
                    py = ports[p, 1]
                else:
                    continue
            else:
                continue
            if px < x_min:
                x_min = px
            if px > x_max:
                x_max = px
            if py < y_min:
                y_min = py
            if py > y_max:
                y_max = py

        if not np.isfinite(x_min) or not np.isfinite(x_max):
            continue

        bbox_w = max(0.0, float(x_max - x_min))
        bbox_h = max(0.0, float(y_max - y_min))
        if bbox_w + bbox_h <= 0.0:
            continue

        c_lo = int(x_min * inv_bin_w)
        c_hi = int(x_max * inv_bin_w)
        r_lo = int(y_min * inv_bin_h)
        r_hi = int(y_max * inv_bin_h)
        if c_lo < 0:
            c_lo = 0
        if r_lo < 0:
            r_lo = 0
        if c_hi >= cols:
            c_hi = cols - 1
        if r_hi >= rows:
            r_hi = rows - 1
        if c_hi < c_lo or r_hi < r_lo:
            continue

        n_bins = (c_hi - c_lo + 1) * (r_hi - r_lo + 1)
        if n_bins <= 0:
            continue

        w = 1.0
        if weights is not None and net_idx < weights.shape[0]:
            w = float(weights[net_idx])

        # Each bin in the bbox gets an equal share of total horizontal wire
        # (~bbox_w) and vertical wire (~bbox_h).  This is the simplest Rudy
        # variant; it overestimates spread for tall/wide nets but tracks the
        # same regions that the proxy congestion term penalizes.
        h_per = w * bbox_w / n_bins
        v_per = w * bbox_h / n_bins

        h_demand[r_lo : r_hi + 1, c_lo : c_hi + 1] += h_per
        v_demand[r_lo : r_hi + 1, c_lo : c_hi + 1] += v_per

    h_cap = float(benchmark.hroutes_per_micron) * bin_h * bin_w
    v_cap = float(benchmark.vroutes_per_micron) * bin_w * bin_h
    if h_cap <= 0.0:
        h_cap = 1e-9
    if v_cap <= 0.0:
        v_cap = 1e-9

    return np.maximum(h_demand / h_cap, v_demand / v_cap)


def normalize_map(arr: np.ndarray) -> np.ndarray:
    """Scale a non-negative map so its mean is 1.0 (or zero if all-zero)."""
    if arr.size == 0:
        return arr
    mean = float(arr.mean())
    if mean <= 0.0:
        return np.zeros_like(arr)
    return arr / mean
