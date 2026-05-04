"""Shared differentiable objectives and legalization for descent-based placers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from macro_place.benchmark import Benchmark


def try_load_iccad04_plc(benchmark: Benchmark):
    """Reload PlacementCost for IBM ICCAD04 paths (proxy scoring / orientation trials)."""
    from macro_place.loader import load_benchmark_from_dir

    root = Path(__file__).resolve().parents[1]
    d = root / "external/MacroPlacement/Testcases/ICCAD04" / benchmark.name
    if not d.is_dir():
        return None
    try:
        return load_benchmark_from_dir(str(d))[1]
    except OSError:
        return None


@dataclass
class NetPinBatch:
    """Padded pin connectivity for vectorized smooth HPWL / Rudy."""

    max_p: int
    mask: torch.Tensor  # [num_nets, max_p] bool, True = real pin
    owner: torch.Tensor  # [num_nets, max_p] int64, benchmark macro index or port id
    kind: torch.Tensor  # [num_nets, max_p] int8: 0=hard, 1=soft, 2=port
    off_x: torch.Tensor  # [num_nets, max_p]
    off_y: torch.Tensor
    port_x: torch.Tensor  # [num_nets, max_p] xy for port pins, else 0
    port_y: torch.Tensor
    weight: torch.Tensor  # [num_nets]


def build_net_pin_batch(benchmark: Benchmark, device: torch.device) -> NetPinBatch:
    num_hard = benchmark.num_hard_macros
    num_macros = benchmark.num_macros
    ports = benchmark.port_positions.to(device=device, dtype=torch.float32)
    weights = benchmark.net_weights.to(device=device, dtype=torch.float32)

    if benchmark.net_pin_nodes:
        nets = benchmark.net_pin_nodes
    else:
        nets = []
        for nodes in benchmark.net_nodes:
            if nodes.numel() < 2:
                nets.append(torch.zeros(0, 2, dtype=torch.long))
            else:
                pins = torch.stack(
                    [torch.tensor([int(n), 0], dtype=torch.long) for n in nodes], dim=0
                )
                nets.append(pins)

    max_p = max((t.shape[0] for t in nets), default=1)
    n_nets = len(nets)
    mask = torch.zeros(n_nets, max_p, dtype=torch.bool, device=device)
    owner = torch.zeros(n_nets, max_p, dtype=torch.long, device=device)
    kind = torch.zeros(n_nets, max_p, dtype=torch.int8, device=device)
    off_x = torch.zeros(n_nets, max_p, dtype=torch.float32, device=device)
    off_y = torch.zeros(n_nets, max_p, dtype=torch.float32, device=device)
    port_x = torch.zeros(n_nets, max_p, dtype=torch.float32, device=device)
    port_y = torch.zeros(n_nets, max_p, dtype=torch.float32, device=device)

    pin_offsets = benchmark.macro_pin_offsets

    for i, pins in enumerate(nets):
        p = int(pins.shape[0])
        if p < 2:
            continue
        mask[i, :p] = True
        for j in range(p):
            o = int(pins[j, 0].item())
            s = int(pins[j, 1].item())
            owner[i, j] = o
            if o >= num_macros:
                pi = o - num_macros
                kind[i, j] = 2
                if 0 <= pi < ports.shape[0]:
                    port_x[i, j] = ports[pi, 0]
                    port_y[i, j] = ports[pi, 1]
            elif o >= num_hard:
                kind[i, j] = 1
            else:
                kind[i, j] = 0
                if o < len(pin_offsets) and pin_offsets[o].numel() > 0:
                    po = pin_offsets[o]
                    if s < po.shape[0]:
                        off_x[i, j] = po[s, 0]
                        off_y[i, j] = po[s, 1]

    w = weights[:n_nets] if weights.numel() >= n_nets else torch.ones(n_nets, device=device)
    return NetPinBatch(
        max_p=max_p,
        mask=mask,
        owner=owner,
        kind=kind,
        off_x=off_x,
        off_y=off_y,
        port_x=port_x,
        port_y=port_y,
        weight=w,
    )


def pin_xy_from_placement(
    placement: torch.Tensor,
    batch: NetPinBatch,
    *,
    num_hard: int,
    num_macros: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Differentiable pin coordinates [num_nets, max_p]. """
    n_nets, max_p = batch.mask.shape
    device = placement.device
    px = torch.zeros(n_nets, max_p, device=device, dtype=placement.dtype)
    py = torch.zeros(n_nets, max_p, device=device, dtype=placement.dtype)

    m = batch.mask
    if not m.any():
        return px, py

    k = batch.kind
    ow = batch.owner

    hard_m = m & (k == 0)
    soft_m = m & (k == 1)
    port_m = m & (k == 2)

    if hard_m.any():
        idx = ow[hard_m]
        px[hard_m] = placement[idx, 0] + batch.off_x[hard_m]
        py[hard_m] = placement[idx, 1] + batch.off_y[hard_m]
    if soft_m.any():
        idx = ow[soft_m]
        px[soft_m] = placement[idx, 0]
        py[soft_m] = placement[idx, 1]
    if port_m.any():
        px[port_m] = batch.port_x[port_m]
        py[port_m] = batch.port_y[port_m]

    return px, py


def smooth_hpwl_loss(
    placement: torch.Tensor,
    benchmark: Benchmark,
    batch: NetPinBatch,
    *,
    beta: float,
) -> torch.Tensor:
    """Log-sum-exp smoothed HPWL (weighted)."""
    num_hard = benchmark.num_hard_macros
    num_macros = benchmark.num_macros
    px, py = pin_xy_from_placement(placement, batch, num_hard=num_hard, num_macros=num_macros)
    m = batch.mask
    neg_inf = torch.finfo(px.dtype).min / 4
    px_e = torch.where(m, px, torch.full_like(px, neg_inf))
    py_e = torch.where(m, py, torch.full_like(py, neg_inf))
    px_neg = torch.where(m, -px, torch.full_like(px, neg_inf))
    py_neg = torch.where(m, -py, torch.full_like(py, neg_inf))

    b = float(beta)
    smax_x = torch.logsumexp(b * px_e, dim=1) / b
    smin_x = -torch.logsumexp(b * px_neg, dim=1) / b
    smax_y = torch.logsumexp(b * py_e, dim=1) / b
    smin_y = -torch.logsumexp(b * py_neg, dim=1) / b
    wl = (smax_x - smin_x) + (smax_y - smin_y)
    valid = m.sum(dim=1) >= 2
    return (wl * batch.weight * valid.float()).sum()


def smooth_bbox(
    px: torch.Tensor,
    py: torch.Tensor,
    m: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    neg_inf = torch.finfo(px.dtype).min / 4
    px_e = torch.where(m, px, torch.full_like(px, neg_inf))
    py_e = torch.where(m, py, torch.full_like(py, neg_inf))
    px_neg = torch.where(m, -px, torch.full_like(px, neg_inf))
    py_neg = torch.where(m, -py, torch.full_like(py, neg_inf))
    b = float(beta)
    xmax = torch.logsumexp(b * px_e, dim=1) / b
    xmin = -torch.logsumexp(b * px_neg, dim=1) / b
    ymax = torch.logsumexp(b * py_e, dim=1) / b
    ymin = -torch.logsumexp(b * py_neg, dim=1) / b
    return xmin, xmax, ymin, ymax


def soft_rudy_penalty(
    placement: torch.Tensor,
    benchmark: Benchmark,
    batch: NetPinBatch,
    *,
    grid_rows: int,
    grid_cols: int,
    beta: float,
    tau_bins: float = 0.35,
) -> torch.Tensor:
    """Differentiable congestion proxy: soft Rudy + utilization vs capacity."""
    W = float(benchmark.canvas_width)
    H = float(benchmark.canvas_height)
    num_hard = benchmark.num_hard_macros
    num_macros = benchmark.num_macros
    px, py = pin_xy_from_placement(placement, batch, num_hard=num_hard, num_macros=num_macros)
    m = batch.mask
    xmin, xmax, ymin, ymax = smooth_bbox(px, py, m, beta)

    rows, cols = int(grid_rows), int(grid_cols)
    device = placement.device
    dtype = placement.dtype
    bin_w = W / cols
    bin_h = H / rows
    cx = (torch.arange(cols, device=device, dtype=dtype) + 0.5) * bin_w
    cy = (torch.arange(rows, device=device, dtype=dtype) + 0.5) * bin_h
    gy, gx = torch.meshgrid(cy, cx, indexing="ij")  # [rows, cols]

    left = gx - 0.5 * bin_w
    right = gx + 0.5 * bin_w
    bottom = gy - 0.5 * bin_h
    top = gy + 0.5 * bin_h
    tau = float(tau_bins) * min(bin_w, bin_h)

    def soft_in_interval(t: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((t - lo) / tau) * torch.sigmoid((hi - t) / tau)

    # [num_nets] vs [rows, cols] -> broadcast [num_nets, rows, cols]
    x1 = xmin[:, None, None]
    x2 = xmax[:, None, None]
    y1 = ymin[:, None, None]
    y2 = ymax[:, None, None]
    inside = (
        soft_in_interval(gx, x1, x2) * soft_in_interval(gy, y1, y2)
    )  # [num_nets, rows, cols]
    mass = inside.sum(dim=(1, 2)).clamp_min(1e-6)
    bbox_w = (xmax - xmin).clamp_min(1e-6)
    bbox_h = (ymax - ymin).clamp_min(1e-6)
    w = batch.weight[:, None, None]
    h_share = w * bbox_w[:, None, None] * inside / mass[:, None, None]
    v_share = w * bbox_h[:, None, None] * inside / mass[:, None, None]

    h_demand = h_share.sum(dim=0)
    v_demand = v_share.sum(dim=0)
    h_cap = float(benchmark.hroutes_per_micron) * bin_h * bin_w
    v_cap = float(benchmark.vroutes_per_micron) * bin_w * bin_h
    h_util = h_demand / max(h_cap, 1e-9)
    v_util = v_demand / max(v_cap, 1e-9)
    util = torch.maximum(h_util, v_util)
    return torch.relu(util - 1.0).pow(2).mean()


def _splat_gaussians_centers(
    centers_xy: torch.Tensor,
    sizes_hw: torch.Tensor,
    gx: torch.Tensor,
    gy: torch.Tensor,
) -> torch.Tensor:
    """Sum of 2D Gaussian blobs (area-weighted); centers_xy [N,2], sizes [N,2]."""
    if centers_xy.numel() == 0:
        return torch.zeros_like(gx)
    n = centers_xy.shape[0]
    mx = centers_xy[:, 0].view(n, 1, 1)
    my = centers_xy[:, 1].view(n, 1, 1)
    wi = sizes_hw[:, 0].view(n, 1, 1)
    hi = sizes_hw[:, 1].view(n, 1, 1)
    area = (wi * hi).clamp_min(1e-6)
    sigma = 0.22 * torch.maximum(wi, hi)
    r2 = (gx - mx) ** 2 + (gy - my) ** 2
    return (area * torch.exp(-r2 / (2.0 * sigma * sigma))).sum(0)


def fft_density_energy(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    grid_n: int,
    base: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ePlace-style electrostatic energy; optional fixed background from ``base``."""
    W = float(benchmark.canvas_width)
    H = float(benchmark.canvas_height)
    device = placement.device
    dtype = placement.dtype
    rows, cols = int(grid_n), int(grid_n)

    ys = (torch.arange(rows, device=device, dtype=dtype) + 0.5) * (H / rows)
    xs = (torch.arange(cols, device=device, dtype=dtype) + 0.5) * (W / cols)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    num_hard = benchmark.num_hard_macros
    num_macros = benchmark.num_macros
    sizes_all = benchmark.macro_sizes.to(device=device, dtype=dtype)
    fixed = benchmark.macro_fixed[:num_hard].to(device=device)
    movable = ~fixed

    if base is None:
        base = placement.detach()

    # Fixed background: soft macros + fixed hard (no grad) — repels movable from clusters.
    rho_fixed = torch.zeros_like(gx)
    with torch.no_grad():
        if num_hard < num_macros:
            soft_idx = torch.arange(num_hard, num_macros, device=device)
            rho_fixed = rho_fixed + _splat_gaussians_centers(
                base[soft_idx], sizes_all[soft_idx], gx, gy
            )
        if fixed.any():
            fi = torch.where(fixed)[0]
            rho_fixed = rho_fixed + _splat_gaussians_centers(
                base[fi], sizes_all[fi], gx, gy
            )

    ph = placement[:num_hard]
    if movable.any():
        mi = torch.where(movable)[0]
        rho_mov = _splat_gaussians_centers(ph[mi], sizes_all[mi], gx, gy)
    else:
        rho_mov = torch.zeros_like(gx)

    rho = rho_fixed + rho_mov
    rho = rho - rho.mean()
    cell_w = W / cols
    cell_h = H / rows

    kx = 2.0 * torch.pi * torch.fft.fftfreq(cols, d=cell_w, device=device, dtype=dtype)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(rows, d=cell_h, device=device, dtype=dtype)
    kxv, kyv = torch.meshgrid(kx, ky, indexing="ij")
    denom = kxv * kxv + kyv * kyv
    denom = denom.clone()
    denom[0, 0] = 1.0

    rho_hat = torch.fft.fft2(rho)
    phi_hat = -rho_hat / denom
    phi_hat[0, 0] = 0.0
    phi = torch.fft.ifft2(phi_hat).real
    energy = 0.5 * (phi * rho).sum() * cell_w * cell_h
    # Scale-stable across canvas sizes and macro counts (feature-derived only).
    scale = (W * H) * max(1, int(movable.sum().item()))
    return energy / max(float(scale), 1.0)


def anchor_penalty(
    placement: torch.Tensor,
    anchor: torch.Tensor,
    movable_mask: torch.Tensor,
) -> torch.Tensor:
    d = placement[movable_mask] - anchor[movable_mask]
    return (d * d).sum()


def legalize_min_displacement_grid(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    grid_div: int = 40,
    overlap_gap: float = 1e-3,
    max_candidates: int = 8000,
) -> torch.Tensor:
    """Greedy legalizer: large macros first, snap to coarse grid near current spot."""
    out = placement.clone()
    num_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:num_hard].cpu().numpy()
    fixed = benchmark.macro_fixed[:num_hard].cpu().numpy()
    movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()

    step = max(cw, ch) / float(max(grid_div, 8))
    xs = np.arange(step * 0.5, cw, step, dtype=np.float64)
    ys = np.arange(step * 0.5, ch, step, dtype=np.float64)
    grid_pts = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)

    half_w = 0.5 * sizes[:, 0]
    half_h = 0.5 * sizes[:, 1]
    areas = sizes[:, 0] * sizes[:, 1]
    order = np.argsort(-areas)

    pos = out[:num_hard].cpu().numpy().astype(np.float64).copy()

    def overlap_pair(i: int, j: int, pi: np.ndarray) -> bool:
        dx = abs(pi[i, 0] - pi[j, 0])
        dy = abs(pi[i, 1] - pi[j, 1])
        sep_x = half_w[i] + half_w[j] + overlap_gap
        sep_y = half_h[i] + half_h[j] + overlap_gap
        return dx < sep_x and dy < sep_y

    def legal_with_placed(i: int, cand: np.ndarray, placed: List[int]) -> bool:
        if cand[0] < half_w[i] or cand[0] > cw - half_w[i]:
            return False
        if cand[1] < half_h[i] or cand[1] > ch - half_h[i]:
            return False
        for j in placed:
            dx = abs(cand[0] - pos[j, 0])
            dy = abs(cand[1] - pos[j, 1])
            if dx < half_w[i] + half_w[j] + overlap_gap and dy < half_h[i] + half_h[j] + overlap_gap:
                return False
        return True

    placed: List[int] = []
    for i in order:
        if fixed[i] or not movable[i]:
            placed.append(i)
            continue
        cx, cy = float(pos[i, 0]), float(pos[i, 1])
        dist2 = (grid_pts[:, 0] - cx) ** 2 + (grid_pts[:, 1] - cy) ** 2
        idx_sort = np.argsort(dist2)
        found = None
        for t in idx_sort[:max_candidates]:
            cand = grid_pts[int(t)]
            if legal_with_placed(i, cand, placed):
                found = cand
                break
        if found is None:
            found = np.array([np.clip(cx, half_w[i], cw - half_w[i]), np.clip(cy, half_h[i], ch - half_h[i])])
        pos[i] = found
        placed.append(i)

    out[:num_hard] = torch.from_numpy(pos).to(out.dtype)
    return out


def finalize_hard_legalization(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    grid_div: int,
    overlap_gap: float = 1.2e-3,
) -> torch.Tensor:
    """Coarse grid snap then overlap repair (bounded displacement)."""
    from _hard_legalizer import legalize_hard

    nh = benchmark.num_hard_macros
    max_cand = min(20000, 3000 + 50 * nh)
    out = legalize_min_displacement_grid(
        placement,
        benchmark,
        grid_div=grid_div,
        overlap_gap=overlap_gap,
        max_candidates=max_cand,
    )
    return legalize_hard(out, benchmark, overlap_gap=overlap_gap, legalize_rounds=320)


def legal_grid_div_heuristic(benchmark: Benchmark) -> int:
    """Finer grid when there are more movable hard macros (no benchmark names).."""
    nh = int(benchmark.num_hard_macros)
    movable = int((~benchmark.macro_fixed[:nh]).sum().item())
    return int(np.clip(36.0 + 0.12 * float(movable), 36.0, 72.0))


def adam_step(
    params: List[torch.Tensor],
    grads: List[Optional[torch.Tensor]],
    m_states: List[torch.Tensor],
    v_states: List[torch.Tensor],
    *,
    step: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    for i, p in enumerate(params):
        g = grads[i]
        if g is None:
            continue
        m_states[i] = beta1 * m_states[i] + (1.0 - beta1) * g
        v_states[i] = beta2 * v_states[i] + (1.0 - beta2) * (g * g)
        m_hat = m_states[i] / (1.0 - beta1 ** step)
        v_hat = v_states[i] / (1.0 - beta2 ** step)
        p.data.addcdiv_(-lr, m_hat, v_hat.sqrt() + eps)


def project_inside_canvas(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    num_hard: Optional[int] = None,
) -> None:
    nh = int(num_hard or benchmark.num_hard_macros)
    sizes = benchmark.macro_sizes[:nh].to(placement.device)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    hw = 0.5 * sizes[:, 0]
    hh = 0.5 * sizes[:, 1]
    placement[:nh, 0].clamp_(hw, cw - hw)
    placement[:nh, 1].clamp_(hh, ch - hh)


def scatter_free_into_placement(
    base: torch.Tensor,
    free_idx: List[int],
    free_xy: torch.Tensor,
) -> torch.Tensor:
    out = base.clone()
    out[free_idx] = free_xy
    return out


def movable_hard_indices(benchmark: Benchmark) -> List[int]:
    m = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    return torch.where(m)[0].tolist()


def congestion_grid_shape(benchmark: Benchmark) -> Tuple[int, int]:
    """Routing-grid-shaped soft Rudy resolution (bounded, feature-derived)."""
    r = int(benchmark.grid_rows)
    c = int(benchmark.grid_cols)
    r = int(np.clip(float(r), 24.0, 56.0))
    c = int(np.clip(float(c), 24.0, 56.0))
    return r, c


def lns_time_budget_sec(benchmark: Benchmark) -> float:
    nh = int(benchmark.num_hard_macros)
    return float(np.clip(40.0 + 0.28 * float(nh), 40.0, 220.0))


def global_adam_optimize(
    benchmark: Benchmark,
    state: torch.Tensor,
    optimize_idx: List[int],
    batch: NetPinBatch,
    *,
    iters: int,
    lr: float,
    beta: float,
    density_grid: int,
    lam_start: float,
    lam_end: float,
    anchor: Optional[torch.Tensor] = None,
    mu_anchor_start: float = 0.0,
    mu_anchor_end: float = 0.0,
    cong_weight: float = 0.0,
    cong_rows: int = 32,
    cong_cols: int = 32,
    lr_end: Optional[float] = None,
) -> torch.Tensor:
    """Adam on a subset of macros; HPWL + ramped density + optional anchor + soft Rudy."""
    if not optimize_idx:
        return state
    device = state.device
    dtype = state.dtype
    density_ref = benchmark.macro_positions.to(device=device, dtype=dtype)
    base = state.clone()
    free_xy = torch.nn.Parameter(state[optimize_idx].clone())
    opt = torch.optim.Adam([free_xy], lr=lr)
    lr_0 = float(lr)
    lr_1 = float(lr_end) if lr_end is not None else lr_0

    for t in range(iters):
        alpha = t / max(iters - 1, 1)
        lam = lam_start + alpha * (lam_end - lam_start)
        mu = mu_anchor_start + alpha * (mu_anchor_end - mu_anchor_start)
        eta = cong_weight * (0.2 + 0.8 * alpha) if cong_weight > 0 else 0.0
        lr_t = lr_0 + alpha * (lr_1 - lr_0)
        for pg in opt.param_groups:
            pg["lr"] = lr_t

        opt.zero_grad(set_to_none=True)
        pl = scatter_free_into_placement(base, optimize_idx, free_xy)
        loss = smooth_hpwl_loss(pl, benchmark, batch, beta=beta)
        loss = loss + lam * fft_density_energy(
            pl, benchmark, grid_n=density_grid, base=density_ref
        )
        if anchor is not None and (mu_anchor_start > 0 or mu_anchor_end > 0):
            d = pl[optimize_idx] - anchor[optimize_idx]
            loss = loss + mu * (d * d).sum()
        if eta > 0:
            loss = loss + eta * soft_rudy_penalty(
                pl,
                benchmark,
                batch,
                grid_rows=cong_rows,
                grid_cols=cong_cols,
                beta=beta,
            )
        loss.backward()
        opt.step()
        with torch.no_grad():
            pl = scatter_free_into_placement(base, optimize_idx, free_xy)
            project_inside_canvas(pl, benchmark)
            free_xy.copy_(pl[optimize_idx])

    out = scatter_free_into_placement(base, optimize_idx, free_xy).detach()
    project_inside_canvas(out, benchmark)
    return out
