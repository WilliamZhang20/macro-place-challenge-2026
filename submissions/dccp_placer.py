"""
DCCP-based macro placer (draft).

Uses the dccp package CCP implementation (``dccp.problem.dccp``) on a
cvxpy ``Problem``, after ``is_dccp`` validation.  Optimization is in
**displacements** ``(dx, dy)`` from the **baseline** initial placement
``(p0_x, p0_y)`` (centers = p0 + d), so the DCCP step is explicitly a refinement
on top of the legal input placement.

Circle packing is only a **relaxation** for rectangles; after DCCP we run a
light axis push legalization.

Debug: ``MACRO_PLACE_DEBUG_DCCP=1`` logs repair progress on stderr.

DCCP program (per solve, variables ``dx, dy``):
    minimize    ||dx||^2 + ||dy||^2 + lam * ||A dx||^2 + lam * ||A dy||^2
                            + mu * ||A (p0+dx)||^2 + mu * ||A (p0+dy)||^2
  subject to  (r_i+r_j)^2 - ||(p0+dp)_i - (p0+dp)_j||^2 <= 0   (concave)
              centers stay inside canvas (linear in dx, dy)

Usage: see README.md
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import cvxpy as cp
import numpy as np
import torch
from dccp import is_dccp
from dccp.problem import dccp as dccp_ccp

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics
from macro_place.utils import validate_placement

# NG45 commercial flows use different .plc handoffs; IBM ICCAD04 matches dccp_old path.
_NG45_NAMES = frozenset({"ariane133", "ariane136", "mempool_tile", "nvdla"})


def _is_ng45_benchmark(benchmark: Benchmark) -> bool:
    """
    True for NanGate45 challenge handoffs.

    ``load_benchmark(..., initial.plc)`` sets ``benchmark.name`` to the parent folder of
    ``netlist.pb.txt`` (e.g. ``output_CT_Grouping``), not ``ariane133``. Detect that layout
    id as well as canonical short names.
    """
    if benchmark.name in _NG45_NAMES:
        return True
    # MacroPlacement NG45 flow packages netlist under .../<design>/netlist/output_CT_Grouping/
    if benchmark.name == "output_CT_Grouping":
        return True
    return False


def _debug(msg: str) -> None:
    if os.environ.get("MACRO_PLACE_DEBUG_DCCP"):
        print(f"[DccpPlacer] {msg}", file=sys.stderr, flush=True)


def _movable_bbox_exceeds_canvas(
    placement: torch.Tensor, benchmark: Benchmark, margin: float = 1e-4
) -> bool:
    """True if the axis-aligned bbox of all movable macros is wider/taller than the canvas."""
    n = benchmark.num_macros
    if n == 0:
        return False
    movable = (~benchmark.macro_fixed[:n]).detach().cpu().numpy()
    if not movable.any():
        return False
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    w = benchmark.macro_sizes[:n, 0].detach().cpu().numpy().astype(np.float64)
    h = benchmark.macro_sizes[:n, 1].detach().cpu().numpy().astype(np.float64)
    pos = placement[:n, :].detach().cpu().numpy().astype(np.float64)
    idx = np.flatnonzero(movable)
    min_lx = float(np.min(pos[idx, 0] - w[idx] / 2.0))
    min_ly = float(np.min(pos[idx, 1] - h[idx] / 2.0))
    max_ux = float(np.max(pos[idx, 0] + w[idx] / 2.0))
    max_uy = float(np.max(pos[idx, 1] + h[idx] / 2.0))
    span_x = max_ux - min_lx
    span_y = max_uy - min_ly
    # Strictly larger than canvas (float noise on ibm01: span_y ≈ ch triggers false shelf).
    tol = 1e-3 + max(0.0, float(margin))
    return span_x > cw + tol or span_y > ch + tol


def _shelf_pack_movable_hard_macros(placement: torch.Tensor, benchmark: Benchmark) -> None:
    """
    In-place greedy row packing for movable **hard** macros only (demo placer logic).
    Used when the initial floorplan's footprint is larger than the canvas (some NG45
    handoffs): translation cannot fit, and scaling centers would break non-overlap.
    """
    movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    movable_indices = torch.where(movable)[0].tolist()
    if not movable_indices:
        return
    sizes = benchmark.macro_sizes
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    movable_indices.sort(key=lambda i: -sizes[i, 1].item())
    gap = 0.001
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    for idx in movable_indices:
        w = sizes[idx, 0].item()
        h = sizes[idx, 1].item()
        if cursor_x + w > canvas_w:
            cursor_x = 0.0
            cursor_y += row_height + gap
            row_height = 0.0
        if cursor_y + h > canvas_h:
            placement[idx, 0] = w / 2.0
            placement[idx, 1] = h / 2.0
            continue
        placement[idx, 0] = cursor_x + w / 2.0
        placement[idx, 1] = cursor_y + h / 2.0
        cursor_x += w + gap
        row_height = max(row_height, h)


def _fit_all_macros_in_canvas(placement: torch.Tensor, benchmark: Benchmark) -> None:
    """
    In-place translation so **movable** macro bboxes fit inside the canvas.

    NanGate45 ``initial.plc`` handoffs are often strictly overlap-free but slightly
    out of bounds. Clamping centers directly then piles macros on the border and
    creates massive overlaps. This preserves relative layout when only a shift is
    needed; designs already in-bounds are unchanged. Fixed macros
    are not moved (IBM handoffs).
    """
    n = benchmark.num_macros
    if n == 0:
        return
    movable = (~benchmark.macro_fixed[:n]).detach().cpu().numpy()
    if not movable.any():
        return
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    if cw <= 0 or ch <= 0:
        return

    w = benchmark.macro_sizes[:n, 0].detach().cpu().numpy().astype(np.float64)
    h = benchmark.macro_sizes[:n, 1].detach().cpu().numpy().astype(np.float64)
    pos = placement[:n, :].detach().cpu().numpy().astype(np.float64)
    idx = np.flatnonzero(movable)
    min_lx = float(np.min(pos[idx, 0] - w[idx] / 2.0))
    min_ly = float(np.min(pos[idx, 1] - h[idx] / 2.0))
    max_ux = float(np.max(pos[idx, 0] + w[idx] / 2.0))
    max_uy = float(np.max(pos[idx, 1] + h[idx] / 2.0))
    span_x = max_ux - min_lx
    span_y = max_uy - min_ly
    margin = 1e-4

    def write_movable() -> None:
        for i in idx:
            placement[i, 0] = float(pos[i, 0])
            placement[i, 1] = float(pos[i, 1])

    # No uniform scale of centers (sizes are fixed): it does not preserve non-overlap.
    # If the movable footprint cannot fit by translation alone, skip and let the caller
    # repack hard macros (shelf) before retrying this fit.
    tol = 1e-3
    if span_x > cw + tol or span_y > ch + tol:
        return

    dx_lo = -min_lx + margin if min_lx < margin else 0.0
    dx_hi = cw - margin - max_ux if max_ux > cw - margin else 0.0
    dy_lo = -min_ly + margin if min_ly < margin else 0.0
    dy_hi = ch - margin - max_uy if max_uy > ch - margin else 0.0
    dx = dx_lo + dx_hi
    dy = dy_lo + dy_hi
    pos[idx, 0] += dx
    pos[idx, 1] += dy
    write_movable()


def _clamp_movable_to_canvas(
    placement: torch.Tensor, benchmark: Benchmark, *, edge_inset: bool = True
) -> None:
    """In-place clamp of every non-fixed macro center so its bbox stays inside the canvas."""
    movable = ~benchmark.macro_fixed
    if not movable.any():
        return
    hw = benchmark.macro_sizes[:, 0] * 0.5
    hh = benchmark.macro_sizes[:, 1] * 0.5
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    # Tiny inset (NG45): avoids float32 edge overflow in strict validate_placement.
    # ICCAD04 / IBM: match dccp_old — no inset so repair + DCCP see the same geometry.
    inset = 1e-3 if edge_inset else 0.0
    placement[:, 0] = torch.where(
        movable,
        torch.clamp(placement[:, 0], hw + inset, cw - hw - inset),
        placement[:, 0],
    )
    placement[:, 1] = torch.where(
        movable,
        torch.clamp(placement[:, 1], hh + inset, ch - hh - inset),
        placement[:, 1],
    )


def _placement_needs_repair(placement: torch.Tensor, benchmark: Benchmark) -> bool:
    if int(compute_overlap_metrics(placement, benchmark)["overlap_count"]) > 0:
        return True
    ok, _ = validate_placement(placement, benchmark, check_overlaps=True)
    return not ok


def _repair_loaded_floorplan(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    iccad04_style: bool = False,
    max_seconds: Optional[float] = 72.0,
    legalize_max_seconds: Optional[float] = None,
) -> torch.Tensor:
    """
    Some ICCAD04 .plc tensors are slightly illegal in Python (bounds epsilon, hard overlaps
    under PLC overlap threshold). Repair with clamp + shuffled axis legalization + rare jitter.
    Bounded by *max_seconds* when set so dense NG45 floorplans cannot stall indefinitely.
    IBM benchmarks pass max_seconds=None and iccad04_style=True (match dccp_old).

    *legalize_max_seconds* caps each axis-legalization call (NG45 must set this: otherwise one
    pass can run to max_rounds on ~100+ macros and hang for many minutes while the outer loop
    only checks time between attempts).
    """
    rng = np.random.default_rng(0)
    t0 = time.monotonic()
    for attempt in range(22):
        if max_seconds is not None and (time.monotonic() - t0) > max_seconds:
            break
        nh = benchmark.num_hard_macros
        oc_hint = int(compute_overlap_metrics(placement, benchmark)["overlap_count"])
        denom = max(1, nh * (nh - 1) // 2)
        overlap_density = float(oc_hint) / float(denom)
        pair_ops, max_rounds, idle_cap = _adaptive_legalize_budget(
            nh,
            overlap_density,
            iccad04_style=iccad04_style,
        )
        _clamp_movable_to_canvas(placement, benchmark, edge_inset=not iccad04_style)
        placement = _legalize_hard_macros_tensor(
            placement,
            benchmark,
            max_pair_ops=pair_ops,
            max_rounds=max_rounds,
            gap=1e-3,
            rng=rng,
            idle_cap=idle_cap,
            max_seconds=legalize_max_seconds,
        )
        _clamp_movable_to_canvas(placement, benchmark, edge_inset=not iccad04_style)
        oc = int(compute_overlap_metrics(placement, benchmark)["overlap_count"])
        ok, viol = validate_placement(placement, benchmark, check_overlaps=True)
        _debug(
            f"repair attempt {attempt}: hard_overlap_pairs={oc} valid={ok}"
            + (f" first_viol={viol[0]!r}" if viol else "")
        )
        if oc == 0 and ok:
            return placement
        for i in range(nh):
            if bool(benchmark.macro_fixed[i].item()):
                continue
            placement[i, 0] = float(placement[i, 0].item()) + float(rng.normal(0.0, 0.12))
            placement[i, 1] = float(placement[i, 1].item()) + float(rng.normal(0.0, 0.12))
    return placement


def _inscribed_radius(w: float, h: float, shrink: float) -> float:
    """Circle radius inside the macro; shrink < 1 relaxes packing for feasibility."""
    return shrink * min(w, h) / 2.0


def _inject_movable_centers(
    baseline: torch.Tensor,
    idx_list: Sequence[int],
    centers: np.ndarray,
) -> torch.Tensor:
    """Build full placement tensor: baseline with movable hard centers overwritten."""
    out = baseline.clone()
    for k, tensor_i in enumerate(idx_list):
        out[tensor_i, 0] = float(centers[k, 0])
        out[tensor_i, 1] = float(centers[k, 1])
    return out


def _edge_quadratic_cost(centers: np.ndarray, edges: Sequence[Tuple[int, int]]) -> float:
    """Light wirelength surrogate over local edges."""
    if not edges:
        return 0.0
    cost = 0.0
    for i, j in edges:
        dx = float(centers[i, 0] - centers[j, 0])
        dy = float(centers[i, 1] - centers[j, 1])
        cost += dx * dx + dy * dy
    return cost


def _weighted_edge_quadratic_cost(
    centers: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    weights: np.ndarray,
) -> float:
    """Weighted quadratic edge span (wire surrogate)."""
    if not edges:
        return 0.0
    if weights.shape[0] != len(edges):
        return _edge_quadratic_cost(centers, edges)
    cost = 0.0
    for k, (i, j) in enumerate(edges):
        dx = float(centers[i, 0] - centers[j, 0])
        dy = float(centers[i, 1] - centers[j, 1])
        cost += float(weights[k]) * (dx * dx + dy * dy)
    return cost


def _net_objective_edges(
    benchmark: Benchmark,
    idx_list: Sequence[int],
    centers: np.ndarray,
    max_neighbors: int = 10,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Build sparse weighted macro-macro edges from net co-occurrence.

    Uses pairwise expansion for small nets and centroid-anchored sparse links for
    large nets to keep objective size bounded.
    """
    n = len(idx_list)
    if n <= 1 or benchmark.num_nets <= 0:
        return [], np.zeros(0, dtype=np.float64)

    g2l = {int(g): i for i, g in enumerate(idx_list)}
    pair_weight: dict[Tuple[int, int], float] = {}

    for net_id, net in enumerate(benchmark.net_nodes):
        nodes = net.detach().cpu().tolist() if hasattr(net, "detach") else list(net)
        locals_in_net = [g2l[int(u)] for u in nodes if int(u) in g2l]
        m = len(locals_in_net)
        if m <= 1:
            continue
        net_w = float(benchmark.net_weights[net_id].item()) if benchmark.num_nets > net_id else 1.0
        base_w = max(1e-4, net_w / float(max(1, m - 1)))

        if m <= 10:
            for a in range(m):
                ia = locals_in_net[a]
                for b in range(a + 1, m):
                    ib = locals_in_net[b]
                    p = (ia, ib) if ia < ib else (ib, ia)
                    pair_weight[p] = pair_weight.get(p, 0.0) + base_w
            continue

        local_arr = np.asarray(locals_in_net, dtype=np.int64)
        pts = centers[local_arr]
        centroid = np.mean(pts, axis=0)
        d2c = np.sum((pts - centroid) ** 2, axis=1)
        anchor = int(local_arr[int(np.argmin(d2c))])
        for v in local_arr:
            iv = int(v)
            if iv == anchor:
                continue
            p = (anchor, iv) if anchor < iv else (iv, anchor)
            pair_weight[p] = pair_weight.get(p, 0.0) + 0.7 * base_w

        # Local ring by x-order keeps large nets connected with few edges.
        order = np.argsort(pts[:, 0])
        for t in range(m - 1):
            ia = int(local_arr[int(order[t])])
            ib = int(local_arr[int(order[t + 1])])
            p = (ia, ib) if ia < ib else (ib, ia)
            pair_weight[p] = pair_weight.get(p, 0.0) + 0.3 * base_w

    if not pair_weight:
        return [], np.zeros(0, dtype=np.float64)

    # Per-node top-k pruning prevents huge dense objectives.
    incident: List[List[Tuple[float, int, int]]] = [[] for _ in range(n)]
    for (i, j), w in pair_weight.items():
        incident[i].append((w, i, j))
        incident[j].append((w, i, j))

    keep: Set[Tuple[int, int]] = set()
    for i in range(n):
        if not incident[i]:
            continue
        incident[i].sort(key=lambda t: t[0], reverse=True)
        for w, a, b in incident[i][: max(1, max_neighbors)]:
            if w <= 0:
                continue
            keep.add((a, b))

    edges = sorted(keep)
    weights = np.asarray([pair_weight[e] for e in edges], dtype=np.float64)
    if weights.size > 0:
        w_mean = float(np.mean(weights))
        if w_mean > 0:
            weights = weights / w_mean
    return edges, weights


def _rms_displacement(curr: np.ndarray, base: np.ndarray) -> float:
    if curr.size == 0:
        return 0.0
    d2 = np.sum((curr - base) ** 2, axis=1)
    return float(np.sqrt(np.mean(d2)))


def _adaptive_refinement_policy(
    n: int,
    pair_density: float,
    ng45: bool,
    base_outer: int,
    base_dccp_iter: int,
    canvas_diag: float,
) -> Tuple[int, int, float, int]:
    """Adaptive refinement caps from size and current overlap-constraint density."""
    if n <= 1:
        return 0, base_dccp_iter, 0.0, 800
    if n >= 190:
        trust = max(0.25, 0.08 * canvas_diag / math.sqrt(float(n)))
        return 0, min(base_dccp_iter, 20), trust, 700

    size_scale = 1.0 / (1.0 + max(0.0, float(n - 48)) / 58.0)
    dens_scale = 1.0 / (1.0 + 2.2 * max(0.0, pair_density - 0.10))
    ng_scale = 0.84 if ng45 else 1.0
    scale = size_scale * dens_scale * ng_scale

    outer_cap = max(1, int(round(base_outer * scale)))
    dccp_iter = max(18, int(round(base_dccp_iter * max(0.36, scale))))
    if n > 130:
        outer_cap = min(outer_cap, 2)
        dccp_iter = min(dccp_iter, 28)

    trust = max(0.25, 0.12 * canvas_diag / math.sqrt(float(n)))
    if ng45:
        trust *= 1.15

    legalize_iters = int(min(6000, max(900, 700 + 24 * n * (1.0 + 0.6 * pair_density))))
    return outer_cap, dccp_iter, trust, legalize_iters


def _adaptive_legalize_budget(
    num_hard: int,
    overlap_density: float,
    *,
    iccad04_style: bool,
) -> Tuple[int, int, int]:
    """Adaptive legalizer budgets from macro count and observed overlap density."""
    density_boost = 1.0 + 2.0 * max(0.0, overlap_density)
    pair_ops = int((420_000 + 6500 * num_hard) * density_boost)
    max_rounds = int((1100 + 19 * num_hard) * density_boost)
    idle_cap = 14 if iccad04_style else 24
    if overlap_density > 0.06:
        idle_cap += 10
    pair_ops = max(160_000, min(pair_ops, 1_800_000))
    max_rounds = max(700, min(max_rounds, 7200))
    return pair_ops, max_rounds, idle_cap


def _spatial_candidate_pairs(
    centers: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    *,
    dense_fallback_n: int = 32,
) -> List[Tuple[int, int]]:
    """Candidate overlap pairs from a conservative bbox grid index."""
    n = centers.shape[0]
    if n <= 1:
        return []
    if n <= dense_fallback_n:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    widths = 2.0 * half_w
    heights = 2.0 * half_h
    cell_x = float(max(1e-3, np.percentile(widths, 75)))
    cell_y = float(max(1e-3, np.percentile(heights, 75)))

    lx = centers[:, 0] - half_w
    ux = centers[:, 0] + half_w
    ly = centers[:, 1] - half_h
    uy = centers[:, 1] + half_h
    gx0 = np.floor(lx / cell_x).astype(np.int64)
    gx1 = np.floor(ux / cell_x).astype(np.int64)
    gy0 = np.floor(ly / cell_y).astype(np.int64)
    gy1 = np.floor(uy / cell_y).astype(np.int64)

    buckets: dict[Tuple[int, int], List[int]] = {}
    total_slots = 0
    max_slots = max(2000, 24 * n)
    for i in range(n):
        nx = int(gx1[i] - gx0[i] + 1)
        ny = int(gy1[i] - gy0[i] + 1)
        slots = nx * ny
        total_slots += slots
        if total_slots > max_slots:
            return [(a, b) for a in range(n) for b in range(a + 1, n)]
        for gx in range(int(gx0[i]), int(gx1[i]) + 1):
            for gy in range(int(gy0[i]), int(gy1[i]) + 1):
                buckets.setdefault((gx, gy), []).append(i)

    pairs: Set[Tuple[int, int]] = set()
    for members in buckets.values():
        m = len(members)
        if m <= 1:
            continue
        for a in range(m):
            ia = members[a]
            for b in range(a + 1, m):
                ib = members[b]
                if ia < ib:
                    pairs.add((ia, ib))
                else:
                    pairs.add((ib, ia))
    return list(pairs)


def _placement_knn_wire_proxy(placement: torch.Tensor, benchmark: Benchmark, k: int = 4) -> float:
    """Proxy-like tie-breaker for legalizer seeds: lower local edge span is better."""
    nh = benchmark.num_hard_macros
    if nh <= 1:
        return 0.0
    pos = placement[:nh, :].detach().cpu().numpy().astype(np.float64)
    edges = _knn_edges(pos, k=min(k, nh - 1))
    return _edge_quadratic_cost(pos, edges)


def _legalize_hard_macros_tensor(
    placement: torch.Tensor,
    benchmark: Benchmark,
    max_pair_ops: int = 800_000,
    max_rounds: int = 2500,
    gap: float = 1e-3,
    rng: Optional[np.random.Generator] = None,
    idle_cap: int = 14,
    max_seconds: Optional[float] = None,
) -> torch.Tensor:
    """
    Remove axis-aligned overlaps among all hard macros. Fixed macros act as obstacles.
    Uses shuffled pair order when *rng* is set to escape local jamming; stops when no strict
    overlaps remain, or budgets exhaust, or many idle rounds pass with overlaps left.
    Optional *max_seconds* stops pathological long runs on dense designs (caller may retry).
    """
    num_h = benchmark.num_hard_macros
    if num_h <= 1:
        return placement

    out = placement.clone()
    pos = out[:num_h, :].detach().cpu().numpy().astype(np.float64).copy()
    sizes = benchmark.macro_sizes[:num_h].detach().cpu().numpy().astype(np.float64)
    movable = (~benchmark.macro_fixed[:num_h]).detach().cpu().numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0

    def clamp_i(i: int) -> None:
        pos[i, 0] = float(np.clip(pos[i, 0], hw[i], cw - hw[i]))
        pos[i, 1] = float(np.clip(pos[i, 1], hh[i], ch - hh[i]))

    def strict_overlap(i: int, j: int) -> bool:
        adx = abs(pos[i, 0] - pos[j, 0])
        ady = abs(pos[i, 1] - pos[j, 1])
        return adx < hw[i] + hw[j] and ady < hh[i] + hh[j]

    all_pairs = [(i, j) for i in range(num_h) for j in range(i + 1, num_h)]

    def candidate_pairs() -> List[Tuple[int, int]]:
        cand = _spatial_candidate_pairs(pos, hw, hh)
        if not cand:
            return []
        if len(cand) > int(0.92 * len(all_pairs)):
            return all_pairs
        return cand

    def any_overlap_fast() -> bool:
        for ii, jj in candidate_pairs():
            if strict_overlap(ii, jj):
                return True
        return False

    if not any_overlap_fast():
        return placement

    ops = 0
    rounds = 0
    idle = 0
    t0 = time.monotonic()
    while ops < max_pair_ops and rounds < max_rounds:
        if max_seconds is not None and (time.monotonic() - t0) > max_seconds:
            break
        rounds += 1
        moved_round = False
        pair_order = candidate_pairs()
        if not pair_order:
            break
        if rng is not None:
            rng.shuffle(pair_order)
        for i, j in pair_order:
            if not strict_overlap(i, j):
                continue
            mi, mj = bool(movable[i]), bool(movable[j])
            if not mi and not mj:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            sep_x = hw[i] + hw[j] + gap
            sep_y = hh[i] + hh[j] + gap
            ox = sep_x - abs(dx)
            oy = sep_y - abs(dy)
            if ox <= 0 or oy <= 0:
                continue
            moved_round = True
            ops += 1
            if mi and mj:
                if ox <= oy:
                    s = 1.0 if dx >= 0 else -1.0
                    shift = ox / 2.0 + gap * 0.5
                    pos[i, 0] += s * shift
                    pos[j, 0] -= s * shift
                else:
                    s = 1.0 if dy >= 0 else -1.0
                    shift = oy / 2.0 + gap * 0.5
                    pos[i, 1] += s * shift
                    pos[j, 1] -= s * shift
                clamp_i(i)
                clamp_i(j)
            elif mi:
                if ox <= oy:
                    s = 1.0 if dx >= 0 else -1.0
                    pos[i, 0] += s * (ox + gap * 0.5)
                else:
                    s = 1.0 if dy >= 0 else -1.0
                    pos[i, 1] += s * (oy + gap * 0.5)
                clamp_i(i)
            else:
                if ox <= oy:
                    s = 1.0 if dx >= 0 else -1.0
                    pos[j, 0] -= s * (ox + gap * 0.5)
                else:
                    s = 1.0 if dy >= 0 else -1.0
                    pos[j, 1] -= s * (oy + gap * 0.5)
                clamp_i(j)
            if ops >= max_pair_ops:
                break
        if not moved_round:
            if not any_overlap_fast():
                break
            idle += 1
            if idle >= idle_cap:
                break
        else:
            idle = 0

    out[:num_h, 0] = torch.from_numpy(pos[:, 0]).to(out.device, dtype=out.dtype)
    out[:num_h, 1] = torch.from_numpy(pos[:, 1]).to(out.device, dtype=out.dtype)
    return out


def _legality_sort_key(placement: torch.Tensor, benchmark: Benchmark) -> Tuple[int, int]:
    """Lower is better: (overlap_pairs, 1 if any validate violation else 0)."""
    oc = int(compute_overlap_metrics(placement, benchmark)["overlap_count"])
    ok, _ = validate_placement(placement, benchmark, check_overlaps=True)
    return oc, 0 if ok else 1


def _legalize_hard_macros_multi_seed(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    num_seeds: int = 14,
    max_pair_ops: int = 1_100_000,
    max_rounds: int = 4000,
    idle_cap: int = 40,
    gap: float = 1e-3,
    max_seconds: float = 34.0,
    per_seed_legalize_seconds: float = 14.0,
    elite_k: int = 3,
    edge_inset: bool = True,
) -> torch.Tensor:
    """
    Dense floorplans (e.g. NG45) often trap single-pass axis legalization in a local jam.
    Re-run the same push heuristic with shuffled pair orders and larger budgets; keep the
    best result by (overlap_count, validation). Cheap when the first pass is already legal.

    *max_seconds* caps total wall time; *per_seed_legalize_seconds* caps each inner
    legalization so one seed cannot block for tens of minutes on 100+ macros.
    """
    seed0 = placement.clone()
    seed0_key = _legality_sort_key(seed0, benchmark)
    if seed0_key == (0, 0):
        return seed0

    elites: List[Tuple[Tuple[int, int, float], torch.Tensor]] = []

    def keep_elite(key: Tuple[int, int, float], cand: torch.Tensor) -> None:
        elites.append((key, cand.clone()))
        elites.sort(key=lambda x: x[0])
        if len(elites) > max(1, elite_k):
            elites.pop()

    keep_elite((seed0_key[0], seed0_key[1], _placement_knn_wire_proxy(seed0, benchmark)), seed0)
    t0 = time.monotonic()
    for seed in range(num_seeds):
        if time.monotonic() - t0 > max_seconds:
            break
        rng = np.random.default_rng(seed)
        remain = max_seconds - (time.monotonic() - t0)
        inner_cap = min(per_seed_legalize_seconds, max(4.0, remain))
        cand = _legalize_hard_macros_tensor(
            placement.clone(),
            benchmark,
            max_pair_ops=max_pair_ops,
            max_rounds=max_rounds,
            gap=gap,
            rng=rng,
            idle_cap=idle_cap,
            max_seconds=inner_cap,
        )
        _clamp_movable_to_canvas(cand, benchmark, edge_inset=edge_inset)
        key = _legality_sort_key(cand, benchmark)
        proxy = _placement_knn_wire_proxy(cand, benchmark)
        keep_elite((key[0], key[1], proxy), cand)
        if elites and elites[0][0][0] == 0 and elites[0][0][1] == 0 and len(elites) >= max(1, elite_k):
            break

    return elites[0][1]


def _legalize_centers(
    centers: np.ndarray,
    sizes: np.ndarray,
    cw: float,
    ch: float,
    max_iters: int = 5000,
    gap: float = 1e-3,
) -> np.ndarray:
    """
    Remove axis-aligned rectangle overlaps by pushing pairs apart (deterministic).

    Does not optimize wirelength; used only to repair DCCP circle relaxations,
    which do not imply rectangle legality. Clamps centers to stay in-bounds.
    """
    c = np.asarray(centers, dtype=np.float64).copy()
    n = c.shape[0]
    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0

    def clamp_one(i: int) -> None:
        c[i, 0] = float(np.clip(c[i, 0], hw[i], cw - hw[i]))
        c[i, 1] = float(np.clip(c[i, 1], hh[i], ch - hh[i]))

    for _ in range(max_iters):
        moved = False
        pair_order = _spatial_candidate_pairs(c, hw, hh)
        if not pair_order:
            break
        for i, j in pair_order:
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            sep_x = hw[i] + hw[j] + gap
            sep_y = hh[i] + hh[j] + gap
            ox = sep_x - abs(dx)
            oy = sep_y - abs(dy)
            if ox <= 0 or oy <= 0:
                continue
            moved = True
            # Separate along the axis that needs less total motion.
            if ox <= oy:
                s = 1.0 if dx >= 0 else -1.0
                shift = ox / 2.0 + gap * 0.5
                c[i, 0] += s * shift
                c[j, 0] -= s * shift
            else:
                s = 1.0 if dy >= 0 else -1.0
                shift = oy / 2.0 + gap * 0.5
                c[i, 1] += s * shift
                c[j, 1] -= s * shift
            clamp_one(i)
            clamp_one(j)
        if not moved:
            break
    return c


def _knn_edges(pos: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """Undirected edges from k-nearest neighbors in 2D."""
    n = pos.shape[0]
    if n <= 1:
        return []
    k = min(k, n - 1)
    edges: Set[Tuple[int, int]] = set()
    for i in range(n):
        d2 = np.sum((pos - pos[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn = np.argpartition(d2, k - 1)[:k]
        for j in nn:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return list(edges)


def _build_circle_packing_dccp_problem(
    n: int,
    p0: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    radii: np.ndarray,
    cw: float,
    ch: float,
    pairs: Sequence[Tuple[int, int]],
    reg_edges: Sequence[Tuple[int, int]],
    wire_edges: Sequence[Tuple[int, int]],
    wire_weights: np.ndarray | None = None,
) -> Tuple[cp.Problem, cp.Variable, cp.Variable, np.ndarray, np.ndarray]:
    """
    DCCP on displacements dx, dy from baseline p0: x = p0_x + dx, y = p0_y + dy.

    k-NN term uses A @ dx and A @ dy (equivalent to preserving relative offsets
    from baseline). Vectorized for fast CVXPY canonicalization.
    """
    px = np.asarray(p0[:, 0], dtype=np.float64)
    py = np.asarray(p0[:, 1], dtype=np.float64)
    dx = cp.Variable(n)
    dy = cp.Variable(n)
    x = px + dx
    y = py + dy

    obj = cp.sum_squares(dx) + cp.sum_squares(dy)
    edge_density = min(1.0, float(len(reg_edges)) / float(max(1, n)))
    wire_density = min(1.0, float(len(wire_edges)) / float(max(1, n)))
    lam = 0.09 + 0.08 * edge_density
    mu = 0.018 + 0.040 * wire_density

    if reg_edges:
        A_e = np.zeros((len(reg_edges), n), dtype=np.float64)
        for k, (i, j) in enumerate(reg_edges):
            A_e[k, i] = 1.0
            A_e[k, j] = -1.0
        obj += lam * cp.sum_squares(A_e @ dx) + lam * cp.sum_squares(A_e @ dy)

    if wire_edges:
        A_w = np.zeros((len(wire_edges), n), dtype=np.float64)
        for k, (i, j) in enumerate(wire_edges):
            A_w[k, i] = 1.0
            A_w[k, j] = -1.0
        if wire_weights is not None and wire_weights.shape[0] == len(wire_edges):
            sw = np.sqrt(np.maximum(wire_weights, 1e-6)).astype(np.float64)
            Awx = cp.multiply(sw, A_w @ x)
            Awy = cp.multiply(sw, A_w @ y)
            obj += mu * cp.sum_squares(Awx) + mu * cp.sum_squares(Awy)
        else:
            obj += mu * cp.sum_squares(A_w @ x) + mu * cp.sum_squares(A_w @ y)

    cons: List = [
        x >= half_w,
        x <= cw - half_w,
        y >= half_h,
        y <= ch - half_h,
    ]

    if pairs:
        A_p = np.zeros((len(pairs), n), dtype=np.float64)
        rij = np.zeros(len(pairs), dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            A_p[k, i] = 1.0
            A_p[k, j] = -1.0
            rij[k] = radii[i] + radii[j]
        Bx = A_p @ x
        By = A_p @ y
        cons.append(rij**2 - cp.square(Bx) - cp.square(By) <= 0)

    prob = cp.Problem(cp.Minimize(obj), cons)
    return prob, dx, dy, px, py


class DccpPlacer:
    """
    Place movable hard macros with DCCP (circle packing + convex objective).

    Parameters
    ----------
    max_outer_iters : int
        Iteratively add overlap pairs and re-solve until clean or limit.
    dccp_max_iter : int
        Inner DCCP iterations per solve.
    knn_k : int
        Neighbors for the "preserve relative displacement" convex term.
    circle_shrink : float
        In (0,1]; smaller inscribed circles ease feasibility.
    """

    def __init__(
        self,
        max_outer_iters: int = 8,
        dccp_max_iter: int = 80,
        knn_k: int = 6,
        circle_shrink: float = 0.92,
        trust_region_scale: float = 1.0,
        surrogate_tolerance: float = 0.03,
    ):
        self.max_outer_iters = max_outer_iters
        self.dccp_max_iter = dccp_max_iter
        self.knn_k = knn_k
        self.circle_shrink = circle_shrink
        self.trust_region_scale = trust_region_scale
        self.surrogate_tolerance = surrogate_tolerance

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # True handoff from loader (may be epsilon-out-of-bounds / micro-overlapping vs strict Python checks).
        ng45 = _is_ng45_benchmark(benchmark)
        initial = benchmark.macro_positions.detach().clone()
        placement = initial.clone()
        # NG45-only: shelf + translate-to-fit (IBM handoffs are in-family for dccp_old; these steps can disturb legality).
        if ng45:
            if _movable_bbox_exceeds_canvas(placement, benchmark):
                _debug(f"{benchmark.name}: movable bbox exceeds canvas — shelf-packing hard macros")
                _shelf_pack_movable_hard_macros(placement, benchmark)
            _fit_all_macros_in_canvas(placement, benchmark)

        _clamp_movable_to_canvas(placement, benchmark, edge_inset=ng45)

        if _placement_needs_repair(placement, benchmark):
            _debug(
                f"{benchmark.name}: loaded floorplan fails strict validation — running repair "
                f"(overlap_pairs={int(compute_overlap_metrics(placement, benchmark)['overlap_count'])})"
            )
            placement = _repair_loaded_floorplan(
                placement,
                benchmark,
                iccad04_style=not ng45,
                max_seconds=(72.0 if ng45 else None),
                legalize_max_seconds=(14.0 if ng45 else None),
            )
        if _placement_needs_repair(placement, benchmark) and ng45:
            _debug(
                f"{benchmark.name}: repair incomplete — multi-seed axis legalization "
                f"(overlap_pairs={int(compute_overlap_metrics(placement, benchmark)['overlap_count'])})"
            )
            placement = _legalize_hard_macros_multi_seed(placement, benchmark)

        # Legal starting point for DCCP + safe fallback if refinement fails.
        baseline = placement.clone()
        hard_movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_idx = torch.where(hard_movable)[0]
        if movable_idx.numel() == 0:
            return placement

        idx_list = [int(i) for i in movable_idx.tolist()]
        n = len(idx_list)
        sizes = benchmark.macro_sizes[idx_list].numpy().astype(np.float64)
        p0 = placement[idx_list].numpy().astype(np.float64)
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)

        half_w = sizes[:, 0] / 2.0
        half_h = sizes[:, 1] / 2.0

        raw_radii = 0.5 * np.minimum(sizes[:, 0], sizes[:, 1])
        radii = self.circle_shrink * raw_radii

        # Baseline centers (legal initial placement from benchmark)
        centers = p0.copy()
        pair_set: Set[Tuple[int, int]] = self._initial_pairs(centers, radii, n)
        reg_edges = _knn_edges(p0, self.knn_k)
        wire_edges, wire_weights = _net_objective_edges(benchmark, idx_list, p0, max_neighbors=max(8, self.knn_k + 2))
        if not wire_edges:
            wire_edges = reg_edges
            wire_weights = np.ones(len(wire_edges), dtype=np.float64)

        pair_count_max = max(1, n * (n - 1) // 2)
        pair_density = float(len(pair_set)) / float(pair_count_max)
        outer_cap, dccp_mi, trust_radius, legalize_iters = _adaptive_refinement_policy(
            n,
            pair_density,
            ng45,
            self.max_outer_iters,
            self.dccp_max_iter,
            math.hypot(cw, ch),
        )
        trust_radius *= max(0.5, self.trust_region_scale)

        current_overlap = int(
            compute_overlap_metrics(_inject_movable_centers(baseline, idx_list, centers), benchmark)["overlap_count"]
        )
        current_surrogate = _weighted_edge_quadratic_cost(centers, wire_edges, wire_weights)
        stagnant_rounds = 0

        for outer in range(outer_cap):
            shrink_iter = min(0.985, self.circle_shrink + 0.012 * float(outer))
            radii_iter = raw_radii * shrink_iter
            x_sol, y_sol = self._solve_dccp(
                p0=p0,
                half_w=half_w,
                half_h=half_h,
                radii=radii_iter,
                cw=cw,
                ch=ch,
                pairs=sorted(pair_set),
                reg_edges=reg_edges,
                wire_edges=wire_edges,
                wire_weights=wire_weights,
                n=n,
                dccp_max_iter=dccp_mi,
                x_init=centers[:, 0],
                y_init=centers[:, 1],
            )
            if x_sol is None or y_sol is None:
                break

            cand_centers = np.empty_like(centers)
            cand_centers[:, 0] = x_sol
            cand_centers[:, 1] = y_sol
            # DCCP uses a circle relaxation; axis push often clears rectangle overlaps here.
            # Must check *after* legalization: pre-legalize overlap can be large while a single
            # legalize pass fixes everything — otherwise we keep adding DCCP pairs, later solves
            # get worse, and the final iterate may not legalize within max_iters.
            cand_centers = _legalize_centers(cand_centers, sizes, cw, ch, max_iters=legalize_iters)

            # Movable-only pair check misses movable–fixed overlaps; match the evaluator.
            tentative = _inject_movable_centers(baseline, idx_list, cand_centers)
            cand_overlap = int(compute_overlap_metrics(tentative, benchmark)["overlap_count"])
            cand_surrogate = _weighted_edge_quadratic_cost(cand_centers, wire_edges, wire_weights)
            cand_disp = _rms_displacement(cand_centers, p0)

            trust_ok = cand_disp <= trust_radius
            overlap_ok = cand_overlap <= current_overlap
            surrogate_ok = cand_surrogate <= current_surrogate * (1.0 + self.surrogate_tolerance)
            accepted = False
            if trust_ok and overlap_ok and (cand_overlap < current_overlap or surrogate_ok):
                centers = cand_centers
                current_overlap = cand_overlap
                current_surrogate = cand_surrogate
                accepted = True

            viol = self._overlapping_pairs(cand_centers, sizes, margin=1e-3)
            if not viol:
                # May still be illegal vs fixed hard macros; final _legalize_hard_macros_tensor repairs.
                if current_overlap == 0:
                    break
                break
            prev_pair_count = len(pair_set)
            for i, j in viol:
                a, b = (i, j) if i < j else (j, i)
                pair_set.add((a, b))
            pair_density = float(len(pair_set)) / float(pair_count_max)
            _, _, trust_radius_new, legalize_iters = _adaptive_refinement_policy(
                n,
                pair_density,
                ng45,
                self.max_outer_iters,
                self.dccp_max_iter,
                math.hypot(cw, ch),
            )
            trust_radius = trust_radius_new * max(0.5, self.trust_region_scale)

            if accepted:
                stagnant_rounds = 0
            elif len(pair_set) == prev_pair_count and cand_overlap >= current_overlap:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0

            if len(pair_set) == prev_pair_count and current_overlap == 0:
                break
            if stagnant_rounds >= 2:
                break

        # Final pass (noop if already legal); cheap insurance if the last outer iter broke early.
        centers = _legalize_centers(centers, sizes, cw, ch, max_iters=max(700, legalize_iters // 2))

        for k, tensor_i in enumerate(idx_list):
            placement[tensor_i, 0] = float(centers[k, 0])
            placement[tensor_i, 1] = float(centers[k, 1])

        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = baseline[fixed_mask]

        # Float32 placement tensor can be epsilon-outside canvas after numpy→torch (validate fails
        # and we incorrectly revert to baseline).
        _clamp_movable_to_canvas(placement, benchmark, edge_inset=ng45)

        # Repair hard–hard overlaps including vs fixed obstacles (not in the DCCP subproblem).
        nh_leg = benchmark.num_hard_macros
        oc_hint = int(compute_overlap_metrics(placement, benchmark)["overlap_count"])
        denom = max(1, nh_leg * (nh_leg - 1) // 2)
        overlap_density = float(oc_hint) / float(denom)
        pair_ops, max_rounds, idle_cap = _adaptive_legalize_budget(
            nh_leg,
            overlap_density,
            iccad04_style=not ng45,
        )
        placement = _legalize_hard_macros_tensor(
            placement,
            benchmark,
            max_pair_ops=pair_ops,
            max_rounds=max_rounds,
            max_seconds=(52.0 if (ng45 and nh_leg > 90) else None),
            idle_cap=(max(idle_cap, 32) if (ng45 and nh_leg > 90) else idle_cap),
        )
        _clamp_movable_to_canvas(placement, benchmark, edge_inset=ng45)

        # NG45-scale density: single-pass legalization can stall; multi-seed only if still illegal.
        if ng45 and _placement_needs_repair(placement, benchmark):
            placement = _legalize_hard_macros_multi_seed(
                placement,
                benchmark,
                max_seconds=48.0,
                per_seed_legalize_seconds=14.0,
                edge_inset=True,
            )
            _clamp_movable_to_canvas(placement, benchmark, edge_inset=True)

        # Evaluator + harness validation (overlap geometry must match).
        if _placement_needs_repair(placement, benchmark):
            if ng45:
                repaired_baseline = _legalize_hard_macros_multi_seed(
                    baseline.clone(),
                    benchmark,
                    num_seeds=14,
                    max_pair_ops=1_400_000,
                    max_rounds=4500,
                    idle_cap=42,
                    max_seconds=48.0,
                    per_seed_legalize_seconds=14.0,
                    edge_inset=True,
                )
                _clamp_movable_to_canvas(repaired_baseline, benchmark, edge_inset=True)
                if not _placement_needs_repair(repaired_baseline, benchmark):
                    return repaired_baseline
            return baseline.clone()
        return placement

    def _initial_pairs(self, centers: np.ndarray, radii: np.ndarray, n: int) -> Set[Tuple[int, int]]:
        pair_set: Set[Tuple[int, int]] = set()
        margin = 1e-3
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.hypot(dx, dy)
                need = radii[i] + radii[j] + margin
                if dist < need * 3.0:
                    pair_set.add((i, j))
        if len(pair_set) == 0 and n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    pair_set.add((i, j))
        return pair_set

    def _solve_dccp(
        self,
        p0: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        radii: np.ndarray,
        cw: float,
        ch: float,
        pairs: Sequence[Tuple[int, int]],
        reg_edges: Sequence[Tuple[int, int]],
        wire_edges: Sequence[Tuple[int, int]],
        wire_weights: np.ndarray | None,
        n: int,
        dccp_max_iter: int | None = None,
        x_init: np.ndarray | None = None,
        y_init: np.ndarray | None = None,
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        prob, dx, dy, px, py = _build_circle_packing_dccp_problem(
            n=n,
            p0=p0,
            half_w=half_w,
            half_h=half_h,
            radii=radii,
            cw=cw,
            ch=ch,
            pairs=pairs,
            reg_edges=reg_edges,
            wire_edges=wire_edges,
            wire_weights=wire_weights,
        )

        if not is_dccp(prob):
            return None, None

        # CCP warm-start: use provided centers when available, else baseline (zero displacement).
        if (
            x_init is not None
            and y_init is not None
            and np.shape(x_init)[0] == n
            and np.shape(y_init)[0] == n
        ):
            dx.value = np.asarray(x_init, dtype=np.float64).ravel() - px
            dy.value = np.asarray(y_init, dtype=np.float64).ravel() - py
        else:
            dx.value = np.zeros(n, dtype=np.float64)
            dy.value = np.zeros(n, dtype=np.float64)

        try:
            dccp_ccp(
                prob,
                max_iter=dccp_max_iter if dccp_max_iter is not None else self.dccp_max_iter,
                solver=cp.CLARABEL,
                ep=1e-4,
                max_slack=1e-2,
            )
        except Exception:
            return None, None

        if dx.value is None or dy.value is None:
            return None, None
        vx = np.asarray(dx.value, dtype=np.float64).ravel()
        vy = np.asarray(dy.value, dtype=np.float64).ravel()
        if not np.all(np.isfinite(vx)) or not np.all(np.isfinite(vy)):
            return None, None

        return px + vx, py + vy

    def _overlapping_pairs(
        self, centers: np.ndarray, sizes: np.ndarray, margin: float
    ) -> List[Tuple[int, int]]:
        """Axis-aligned rectangle overlap (local indices)."""
        out: List[Tuple[int, int]] = []
        hw = sizes[:, 0] / 2.0
        hh = sizes[:, 1] / 2.0
        for i, j in _spatial_candidate_pairs(centers, hw, hh):
            xi, yi = centers[i]
            xj, yj = centers[j]
            wi, hi = sizes[i, 0], sizes[i, 1]
            wj, hj = sizes[j, 0], sizes[j, 1]
            dx = abs(xi - xj)
            dy = abs(yi - yj)
            sep_x = (wi + wj) / 2.0 + margin
            sep_y = (hi + hj) / 2.0 + margin
            if dx < sep_x and dy < sep_y:
                out.append((i, j))
        return out


def _iccad04_dir(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / name


def _run_local_evaluation() -> None:
    """Load one ICCAD04 case, place, validate, print costs (like examples README)."""
    parser = argparse.ArgumentParser(description="Run DccpPlacer on one IBM ICCAD04 benchmark.")
    parser.add_argument(
        "-b",
        "--benchmark",
        default="ibm01",
        help="Benchmark name (default: ibm01)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After evaluation, compare proxy cost and positions to the initial (baseline) placement.",
    )
    args = parser.parse_args()

    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    bench_dir = _iccad04_dir(args.benchmark)
    if not bench_dir.is_dir():
        print(f"Benchmark directory not found: {bench_dir}")
        print("Run from the repository root with: git submodule update --init external/MacroPlacement")
        raise SystemExit(1)

    print("=" * 60)
    print("DCCP Placer — local evaluation")
    print("=" * 60)

    print(f"\n[1/4] Loading {bench_dir}...")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"  Loaded {benchmark.name}")
    print(f"    - Macros: {benchmark.num_macros} ({benchmark.num_hard_macros} hard, {benchmark.num_soft_macros} soft)")
    print(f"    - Nets: {benchmark.num_nets}")
    print(f"    - Canvas: {benchmark.canvas_width:.1f} x {benchmark.canvas_height:.1f} um")

    initial = benchmark.macro_positions.clone()
    baseline_costs = compute_proxy_cost(initial, benchmark, plc)

    print("\n[2/4] Running placer...")
    placer = DccpPlacer()
    placement = placer.place(benchmark)
    print("  Finished place().")

    print("\n[3/4] Validating placement...")
    is_valid, violations = validate_placement(placement, benchmark)
    if is_valid:
        print("  Placement is valid.")
    else:
        print(f"  Invalid: {violations}")

    print("\n[4/4] Computing proxy cost...")
    costs = compute_proxy_cost(placement, benchmark, plc)
    print(f"  Wirelength:  {costs['wirelength_cost']:.6f}")
    print(f"  Density:     {costs['density_cost']:.6f}")
    print(f"  Congestion:  {costs['congestion_cost']:.6f}")
    print(f"  Proxy cost:  {costs['proxy_cost']:.6f}")
    print(f"  Overlaps:    {costs['overlap_count']}")

    if args.verify:
        max_delta = float((placement - initial).abs().max().item())
        same = torch.allclose(placement, initial, atol=1e-5, rtol=0.0)
        print("\n[verify] Baseline (initial .plc) proxy cost (same evaluator):")
        print(f"         Proxy: {baseline_costs['proxy_cost']:.6f}  WL: {baseline_costs['wirelength_cost']:.6f}")
        print(f"[verify] After placer:  Proxy: {costs['proxy_cost']:.6f}")
        print(f"[verify] max |placement - initial| (all macros): {max_delta:.6g} um")
        print(f"[verify] Placement equals initial (atol ~1e-5): {same}")
        if same or max_delta < 1e-4:
            print(
                "[verify] Interpretation: score matches the benchmark's hand-placed initial "
                "floorplan — DCCP did not meaningfully move macros (or reverted to baseline)."
            )
        else:
            print("[verify] Interpretation: placer changed positions vs initial; cost delta is real movement.")
    print()


if __name__ == "__main__":
    _run_local_evaluation()
