"""
Pure DCCP macro placement (control / experiment).

Implements the slide formulation without circle packing or k-NN regularizers,
plus lightweight greedy warm-start/post nudges for improved validity:

  * Variables: rectangle lower-left corners (equivalently centers; we use
    **shifted** centers so the canvas contains the origin and the centroid
    constraint is feasible).
  * Objective: minimize sum_{(i,j) in E} ||c_i - c_j||_2^2 over net-derived
    undirected pairs among movable hard macros.
  * Convex coupling: (1/n) sum_i c_i = 0 in shifted coordinates (origin at
    canvas center), i.e. sum(c_x) = 0 and sum(c_y) = 0.
  * Non-overlap (DCCP): for every pair of relevant rectangles,
        min{ x_i+w_i-x_j, x_j+w_j-x_i, y_i+h_i-y_j, y_j+h_j-y_i } <= 0
    in physical lower-left coordinates (vectorized with one min per pair).

Solver: Convex–concave procedure from the ``dccp`` package with Clarabel.

This file is for experimentation; large benchmarks build O(n^2) non-overlap
constraints and can be slow or memory-heavy.

Usage:
    uv run evaluate submissions/dccp_control.py -b ibm01
    python submissions/dccp_control.py -b ibm01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cvxpy as cp
import numpy as np
import torch
from dccp import is_dccp
from dccp.problem import dccp as dccp_ccp

from macro_place.benchmark import Benchmark


def _net_macro_pairs(
    benchmark: Benchmark, global_indices: List[int]
) -> List[Tuple[int, int]]:
    """Undirected macro–macro pairs from nets (both endpoints movable & in *global_indices*)."""
    gset = set(global_indices)
    loc = {g: k for k, g in enumerate(global_indices)}
    edges: Set[Tuple[int, int]] = set()
    for net in benchmark.net_nodes:
        nodes = net.detach().cpu().tolist() if hasattr(net, "detach") else list(net)
        macs = [int(u) for u in nodes if int(u) in gset]
        m = len(macs)
        for a in range(m):
            for b in range(a + 1, m):
                i, j = macs[a], macs[b]
                if i == j:
                    continue
                p, q = (i, j) if i < j else (j, i)
                edges.add((loc[p], loc[q]))
    return sorted(edges)


def _pair_incidence_matrix(n: int, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """Rows are (e_i - e_j) for each undirected pair (i, j), i < j."""
    p = len(pairs)
    if p == 0:
        return np.zeros((0, n), dtype=np.float64)
    D = np.zeros((p, n), dtype=np.float64)
    r = np.arange(p, dtype=np.int64)
    i_idx = np.array([a for a, _ in pairs], dtype=np.int64)
    j_idx = np.array([b for _, b in pairs], dtype=np.int64)
    D[r, i_idx] = 1.0
    D[r, j_idx] = -1.0
    return D


def _triu_pair_incidence(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All i < j pairs: D @ c gives c_i - c_j for each pair."""
    i_idx, j_idx = np.triu_indices(n, k=1)
    p = len(i_idx)
    if p == 0:
        return i_idx, j_idx, np.zeros((0, n), dtype=np.float64)
    D = np.zeros((p, n), dtype=np.float64)
    r = np.arange(p, dtype=np.int64)
    D[r, i_idx] = 1.0
    D[r, j_idx] = -1.0
    return i_idx, j_idx, D


def _build_pure_dccp_problem(
    *,
    n: int,
    w: np.ndarray,
    h: np.ndarray,
    cw: float,
    ch: float,
    edges: List[Tuple[int, int]],
    i_mm: np.ndarray,
    j_mm: np.ndarray,
    D_mm: np.ndarray,
    mf_i: np.ndarray,
    mf_const0: np.ndarray,
    mf_const1: np.ndarray,
    mf_const2: np.ndarray,
    mf_const3: np.ndarray,
) -> Tuple[cp.Problem, cp.Variable, cp.Variable]:
    """
    Variables cx, cy are macro centers in coordinates shifted by (cw/2, ch/2).

    Physical lower-left: x_ll = cx + cw/2 - w/2, y_ll = cy + ch/2 - h/2.
    """
    cx = cp.Variable(n)
    cy = cp.Variable(n)
    half_w = w * 0.5
    half_h = h * 0.5

    cons: List = [
        cx >= -cw / 2.0 + half_w,
        cx <= cw / 2.0 - half_w,
        cy >= -ch / 2.0 + half_h,
        cy <= ch / 2.0 - half_h,
        cp.sum(cx) == 0,
        cp.sum(cy) == 0,
    ]

    if edges:
        A = _pair_incidence_matrix(n, edges)
        obj = cp.sum_squares(A @ cx) + cp.sum_squares(A @ cy)
    else:
        # Feasibility-only when no net pairs among movable macros
        obj = 0.0 * cp.sum(cx)

    # --- Movable–movable non-overlap (slide DC inequality), vectorized ---
    if D_mm.shape[0] > 0:
        w_sum = w[i_mm] + w[j_mm]
        h_sum = h[i_mm] + h[j_mm]
        t0 = D_mm @ cx + w_sum * 0.5
        t1 = (-D_mm) @ cx + w_sum * 0.5
        t2 = D_mm @ cy + h_sum * 0.5
        t3 = (-D_mm) @ cy + h_sum * 0.5
        mmin = cp.minimum(cp.minimum(t0, t1), cp.minimum(t2, t3))
        cons.append(mmin <= 0)

    # --- Movable–fixed hard macro non-overlap (same min form; j constants) ---
    if mf_i.shape[0] > 0:
        R = np.zeros((mf_i.shape[0], n), dtype=np.float64)
        R[np.arange(mf_i.shape[0], dtype=np.int64), mf_i] = 1.0
        t0 = R @ cx + mf_const0
        t1 = (-R) @ cx + mf_const1
        t2 = R @ cy + mf_const2
        t3 = (-R) @ cy + mf_const3
        mmin_f = cp.minimum(cp.minimum(t0, t1), cp.minimum(t2, t3))
        cons.append(mmin_f <= 0)

    prob = cp.Problem(cp.Minimize(obj), cons)
    return prob, cx, cy


def _clamp_centers(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    cw: float,
    ch: float,
    inset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp centers so macro bounding boxes remain inside canvas."""
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    lbx = 0.5 * w + inset
    ubx = cw - 0.5 * w - inset
    lby = 0.5 * h + inset
    uby = ch - 0.5 * h - inset
    bad_x = lbx > ubx
    bad_y = lby > uby
    if np.any(bad_x):
        midx = 0.5 * cw
        lbx[bad_x] = midx
        ubx[bad_x] = midx
    if np.any(bad_y):
        midy = 0.5 * ch
        lby[bad_y] = midy
        uby[bad_y] = midy
    x = np.minimum(np.maximum(x, lbx), ubx)
    y = np.minimum(np.maximum(y, lby), uby)
    return x, y


def _greedy_nudge_hard_centers(
    *,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    cw: float,
    ch: float,
    fixed_x: np.ndarray,
    fixed_y: np.ndarray,
    fixed_w: np.ndarray,
    fixed_h: np.ndarray,
    rounds: int,
    restarts: int,
    seed: int,
    gap: float,
    inset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy axis pushes to reduce hard-macro overlaps before/after DCCP."""
    x0 = np.asarray(x, dtype=np.float64)
    y0 = np.asarray(y, dtype=np.float64)
    n = int(x0.shape[0])
    if n <= 1 or rounds <= 0:
        return _clamp_centers(x0, y0, w, h, cw, ch, inset)

    mm_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    nf = int(fixed_x.shape[0])

    def overlap_count(xx: np.ndarray, yy: np.ndarray) -> int:
        c = 0
        for i in range(n):
            wi = float(w[i])
            hi = float(h[i])
            for j in range(i + 1, n):
                dx = abs(xx[i] - xx[j])
                dy = abs(yy[i] - yy[j])
                if dx < 0.5 * (wi + float(w[j])) and dy < 0.5 * (hi + float(h[j])):
                    c += 1
            if nf > 0:
                for j in range(nf):
                    dx = abs(xx[i] - float(fixed_x[j]))
                    dy = abs(yy[i] - float(fixed_y[j]))
                    if dx < 0.5 * (wi + float(fixed_w[j])) and dy < 0.5 * (hi + float(fixed_h[j])):
                        c += 1
        return c

    rng = np.random.default_rng(int(seed))
    best_x, best_y = _clamp_centers(x0, y0, w, h, cw, ch, inset)
    best_count = overlap_count(best_x, best_y)
    if best_count == 0:
        return best_x, best_y

    restarts = max(1, int(restarts))
    eps = 1e-12
    for r in range(restarts):
        cur_x = best_x.copy() if r == 0 else x0.copy()
        cur_y = best_y.copy() if r == 0 else y0.copy()
        if r > 0:
            jitter = 0.03
            cur_x = cur_x + rng.normal(0.0, jitter, size=n)
            cur_y = cur_y + rng.normal(0.0, jitter, size=n)
        cur_x, cur_y = _clamp_centers(cur_x, cur_y, w, h, cw, ch, inset)

        for _ in range(int(rounds)):
            moved = False

            pair_order = list(mm_pairs)
            rng.shuffle(pair_order)
            for i, j in pair_order:
                wi = float(w[i])
                hi = float(h[i])
                wj = float(w[j])
                hj = float(h[j])
                dx = cur_x[i] - cur_x[j]
                dy = cur_y[i] - cur_y[j]
                req_x = 0.5 * (wi + wj) + gap
                req_y = 0.5 * (hi + hj) + gap
                ox = req_x - abs(dx)
                oy = req_y - abs(dy)
                if ox <= 0.0 or oy <= 0.0:
                    continue

                moved = True
                if ox <= oy:
                    s = 1.0 if dx >= 0.0 else -1.0
                    if abs(dx) < eps:
                        s = 1.0 if ((i + j) & 1) == 0 else -1.0
                    shift = 0.5 * ox
                    cur_x[i] += s * shift
                    cur_x[j] -= s * shift
                else:
                    s = 1.0 if dy >= 0.0 else -1.0
                    if abs(dy) < eps:
                        s = 1.0 if ((i + j) & 1) == 0 else -1.0
                    shift = 0.5 * oy
                    cur_y[i] += s * shift
                    cur_y[j] -= s * shift

                xi, yi = _clamp_centers(cur_x[i : i + 1], cur_y[i : i + 1], w[i : i + 1], h[i : i + 1], cw, ch, inset)
                xj, yj = _clamp_centers(cur_x[j : j + 1], cur_y[j : j + 1], w[j : j + 1], h[j : j + 1], cw, ch, inset)
                cur_x[i], cur_y[i] = float(xi[0]), float(yi[0])
                cur_x[j], cur_y[j] = float(xj[0]), float(yj[0])

            if nf > 0:
                macro_order = np.arange(n, dtype=np.int64)
                rng.shuffle(macro_order)
                fixed_order = np.arange(nf, dtype=np.int64)
                for i in macro_order:
                    wi = float(w[i])
                    hi = float(h[i])
                    rng.shuffle(fixed_order)
                    for j in fixed_order:
                        dx = cur_x[i] - float(fixed_x[j])
                        dy = cur_y[i] - float(fixed_y[j])
                        req_x = 0.5 * (wi + float(fixed_w[j])) + gap
                        req_y = 0.5 * (hi + float(fixed_h[j])) + gap
                        ox = req_x - abs(dx)
                        oy = req_y - abs(dy)
                        if ox <= 0.0 or oy <= 0.0:
                            continue

                        moved = True
                        if ox <= oy:
                            s = 1.0 if dx >= 0.0 else -1.0
                            if abs(dx) < eps:
                                s = 1.0 if ((i + int(j)) & 1) == 0 else -1.0
                            cur_x[i] += s * ox
                        else:
                            s = 1.0 if dy >= 0.0 else -1.0
                            if abs(dy) < eps:
                                s = 1.0 if ((i + int(j)) & 1) == 0 else -1.0
                            cur_y[i] += s * oy

                        xi, yi = _clamp_centers(cur_x[i : i + 1], cur_y[i : i + 1], w[i : i + 1], h[i : i + 1], cw, ch, inset)
                        cur_x[i], cur_y[i] = float(xi[0]), float(yi[0])

            cur_count = overlap_count(cur_x, cur_y)
            if cur_count < best_count:
                best_x, best_y = cur_x.copy(), cur_y.copy()
                best_count = cur_count
                if best_count == 0:
                    return best_x, best_y

            if not moved:
                break

    return best_x, best_y


def _clamp_nonfixed_macros_to_canvas(
    placement: torch.Tensor,
    benchmark: Benchmark,
    inset: float,
) -> None:
    """In-place clamp for all non-fixed macros (hard + soft)."""
    if benchmark.num_macros == 0:
        return
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    for i in range(benchmark.num_macros):
        if bool(benchmark.macro_fixed[i].item()):
            continue
        wi = float(benchmark.macro_sizes[i, 0].item())
        hi = float(benchmark.macro_sizes[i, 1].item())
        xl = wi * 0.5 + inset
        xh = cw - wi * 0.5 - inset
        yl = hi * 0.5 + inset
        yh = ch - hi * 0.5 - inset
        if xl > xh:
            placement[i, 0] = float(cw * 0.5)
        else:
            placement[i, 0] = float(min(max(float(placement[i, 0].item()), xl), xh))
        if yl > yh:
            placement[i, 1] = float(ch * 0.5)
        else:
            placement[i, 1] = float(min(max(float(placement[i, 1].item()), yl), yh))


def _legalize_hard_macros_tensor(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    max_pair_ops: int,
    max_rounds: int,
    gap: float,
    rng: Optional[np.random.Generator],
    idle_cap: int,
) -> torch.Tensor:
    """Greedy axis-push legalization among hard macros (fixed macros are obstacles)."""
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

    pairs = [(i, j) for i in range(num_h) for j in range(i + 1, num_h)]
    ops = 0
    rounds = 0
    idle = 0
    while ops < max_pair_ops and rounds < max_rounds:
        rounds += 1
        moved_round = False
        pair_order = list(pairs)
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
            if ox <= 0.0 or oy <= 0.0:
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
            idle += 1
            if idle >= idle_cap:
                break
        else:
            idle = 0

    out[:num_h, 0] = torch.from_numpy(pos[:, 0]).to(out.device, dtype=out.dtype)
    out[:num_h, 1] = torch.from_numpy(pos[:, 1]).to(out.device, dtype=out.dtype)
    return out


def _hard_overlap_count(placement: torch.Tensor, benchmark: Benchmark) -> int:
    """Count strict overlaps among hard macros (same geometry as validate_placement)."""
    num_h = benchmark.num_hard_macros
    if num_h <= 1:
        return 0
    pos = placement[:num_h, :].detach().cpu().numpy().astype(np.float64)
    sizes = benchmark.macro_sizes[:num_h, :].detach().cpu().numpy().astype(np.float64)
    c = 0
    for i in range(num_h):
        lxi = pos[i, 0] - 0.5 * sizes[i, 0]
        uxi = pos[i, 0] + 0.5 * sizes[i, 0]
        lyi = pos[i, 1] - 0.5 * sizes[i, 1]
        uyi = pos[i, 1] + 0.5 * sizes[i, 1]
        for j in range(i + 1, num_h):
            lxj = pos[j, 0] - 0.5 * sizes[j, 0]
            uxj = pos[j, 0] + 0.5 * sizes[j, 0]
            lyj = pos[j, 1] - 0.5 * sizes[j, 1]
            uyj = pos[j, 1] + 0.5 * sizes[j, 1]
            if not (lxi >= uxj or uxi <= lxj or lyi >= uyj or uyi <= lyj):
                c += 1
    return c


def _legalize_hard_macros_multi_seed(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    num_seeds: int,
    jitter_std: float,
    boundary_inset: float,
    max_pair_ops: int,
    max_rounds: int,
    gap: float,
    idle_cap: int,
    seed: int,
) -> torch.Tensor:
    """Run greedy legalization with jittered restarts; keep best (fewest hard overlaps)."""
    num_seeds = max(1, int(num_seeds))
    jitter_std = float(max(0.0, jitter_std))

    best = placement.clone()
    best_count = _hard_overlap_count(best, benchmark)
    if best_count == 0:
        return best

    num_h = benchmark.num_hard_macros
    for s in range(num_seeds):
        trial = placement.clone()
        if s > 0 and jitter_std > 0.0 and num_h > 0:
            rng_j = np.random.default_rng(seed + 9973 * s)
            for i in range(num_h):
                if bool(benchmark.macro_fixed[i].item()):
                    continue
                trial[i, 0] = float(trial[i, 0].item()) + float(rng_j.normal(0.0, jitter_std))
                trial[i, 1] = float(trial[i, 1].item()) + float(rng_j.normal(0.0, jitter_std))
            _clamp_nonfixed_macros_to_canvas(trial, benchmark, boundary_inset)

        trial = _legalize_hard_macros_tensor(
            trial,
            benchmark,
            max_pair_ops=max_pair_ops,
            max_rounds=max_rounds,
            gap=gap,
            rng=np.random.default_rng(seed + 131 * s),
            idle_cap=idle_cap,
        )
        cnt = _hard_overlap_count(trial, benchmark)
        if cnt < best_count:
            best = trial
            best_count = cnt
            if best_count == 0:
                return best

    return best


class DccpControlPlacer:
    """
    Pure slide DCCP placement on movable hard macros (no hybrid heuristics).

    Parameters
    ----------
    dccp_max_iter : int
        Inner iterations passed to ``dccp.problem.dccp``.
    ep, max_slack : float
        DCCP tolerances (see dccp package).
    """

    def __init__(
        self,
        dccp_max_iter: int = 80,
        ep: float = 1e-4,
        max_slack: float = 1e-2,
        greedy_warm_start_rounds: int = 8,
        greedy_post_rounds: int = 10,
        greedy_restarts: int = 1,
        post_legalize_rounds: int = 2500,
        post_legalize_pair_ops: int = 900000,
        post_legalize_seeds: int = 8,
        post_legalize_jitter_std: float = 0.03,
        greedy_gap: float = 1e-3,
        boundary_inset: float = 1e-3,
    ):
        self.dccp_max_iter = dccp_max_iter
        self.ep = ep
        self.max_slack = max_slack
        self.greedy_warm_start_rounds = max(0, int(greedy_warm_start_rounds))
        self.greedy_post_rounds = max(0, int(greedy_post_rounds))
        self.greedy_restarts = max(1, int(greedy_restarts))
        self.post_legalize_rounds = max(0, int(post_legalize_rounds))
        self.post_legalize_pair_ops = max(0, int(post_legalize_pair_ops))
        self.post_legalize_seeds = max(1, int(post_legalize_seeds))
        self.post_legalize_jitter_std = float(max(0.0, post_legalize_jitter_std))
        self.greedy_gap = float(max(0.0, greedy_gap))
        self.boundary_inset = float(max(0.0, boundary_inset))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.detach().clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        idx_list = [int(i) for i in torch.where(movable)[0].tolist()]
        if not idx_list:
            return placement

        n = len(idx_list)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        w = benchmark.macro_sizes[idx_list, 0].detach().cpu().numpy().astype(np.float64)
        h = benchmark.macro_sizes[idx_list, 1].detach().cpu().numpy().astype(np.float64)
        pc = placement[idx_list].detach().cpu().numpy().astype(np.float64)

        # Movable–fixed hard pairs and fixed hard geometry.
        nh = benchmark.num_hard_macros
        global_set = set(idx_list)
        fixed_idx = [
            fi
            for fi in range(nh)
            if bool(benchmark.macro_fixed[fi].item()) and fi not in global_set
        ]
        fixed_x = (
            benchmark.macro_positions[fixed_idx, 0].detach().cpu().numpy().astype(np.float64)
            if fixed_idx
            else np.zeros(0, dtype=np.float64)
        )
        fixed_y = (
            benchmark.macro_positions[fixed_idx, 1].detach().cpu().numpy().astype(np.float64)
            if fixed_idx
            else np.zeros(0, dtype=np.float64)
        )
        fixed_w = (
            benchmark.macro_sizes[fixed_idx, 0].detach().cpu().numpy().astype(np.float64)
            if fixed_idx
            else np.zeros(0, dtype=np.float64)
        )
        fixed_h = (
            benchmark.macro_sizes[fixed_idx, 1].detach().cpu().numpy().astype(np.float64)
            if fixed_idx
            else np.zeros(0, dtype=np.float64)
        )

        # Greedy warm start in physical coordinates, then shift to DCCP frame.
        warm_x, warm_y = _clamp_centers(pc[:, 0], pc[:, 1], w, h, cw, ch, self.boundary_inset)
        if self.greedy_warm_start_rounds > 0:
            warm_x, warm_y = _greedy_nudge_hard_centers(
                x=warm_x,
                y=warm_y,
                w=w,
                h=h,
                cw=cw,
                ch=ch,
                fixed_x=fixed_x,
                fixed_y=fixed_y,
                fixed_w=fixed_w,
                fixed_h=fixed_h,
                rounds=self.greedy_warm_start_rounds,
                restarts=self.greedy_restarts,
                seed=0,
                gap=self.greedy_gap,
                inset=self.boundary_inset,
            )
        cx0 = warm_x - cw / 2.0
        cy0 = warm_y - ch / 2.0

        edges = _net_macro_pairs(benchmark, idx_list)

        # All movable–movable pairs for non-overlap
        i_mm, j_mm, D_mm = _triu_pair_incidence(n)

        mf_i_list: List[int] = []
        mf_c0: List[float] = []
        mf_c1: List[float] = []
        mf_c2: List[float] = []
        mf_c3: List[float] = []
        for fi in fixed_idx:
            pfx = float(benchmark.macro_positions[fi, 0].item())
            pfy = float(benchmark.macro_positions[fi, 1].item())
            wf = float(benchmark.macro_sizes[fi, 0].item())
            hf = float(benchmark.macro_sizes[fi, 1].item())
            for loc_k in range(n):
                wi = float(w[loc_k])
                hi = float(h[loc_k])
                # t0 = cx_k + cw/2 + wi/2 - pfx + wf/2
                mf_i_list.append(loc_k)
                mf_c0.append(cw / 2.0 + wi / 2.0 - pfx + wf / 2.0)
                mf_c1.append(pfx + wf / 2.0 - cw / 2.0 + wi / 2.0)
                mf_c2.append(ch / 2.0 + hi / 2.0 - pfy + hf / 2.0)
                mf_c3.append(pfy + hf / 2.0 - ch / 2.0 + hi / 2.0)

        mf_i = np.asarray(mf_i_list, dtype=np.int64) if mf_i_list else np.zeros(0, dtype=np.int64)
        mf_const0 = np.asarray(mf_c0, dtype=np.float64) if mf_c0 else np.zeros(0, dtype=np.float64)
        mf_const1 = np.asarray(mf_c1, dtype=np.float64) if mf_c1 else np.zeros(0, dtype=np.float64)
        mf_const2 = np.asarray(mf_c2, dtype=np.float64) if mf_c2 else np.zeros(0, dtype=np.float64)
        mf_const3 = np.asarray(mf_c3, dtype=np.float64) if mf_c3 else np.zeros(0, dtype=np.float64)

        prob, cx, cy = _build_pure_dccp_problem(
            n=n,
            w=w,
            h=h,
            cw=cw,
            ch=ch,
            edges=edges,
            i_mm=i_mm,
            j_mm=j_mm,
            D_mm=D_mm,
            mf_i=mf_i,
            mf_const0=mf_const0,
            mf_const1=mf_const1,
            mf_const2=mf_const2,
            mf_const3=mf_const3,
        )

        if not is_dccp(prob):
            return placement

        cx.value = np.asarray(cx0, dtype=np.float64)
        cy.value = np.asarray(cy0, dtype=np.float64)

        try:
            dccp_ccp(
                prob,
                max_iter=self.dccp_max_iter,
                solver=cp.CLARABEL,
                ep=self.ep,
                max_slack=self.max_slack,
            )
        except Exception:
            return placement

        if cx.value is None or cy.value is None:
            return placement
        solx = np.asarray(cx.value, dtype=np.float64).ravel()
        soly = np.asarray(cy.value, dtype=np.float64).ravel()
        if solx.shape[0] != n or not np.all(np.isfinite(solx)) or not np.all(np.isfinite(soly)):
            return placement

        # Physical centers from DCCP candidate.
        phys_x = solx + cw / 2.0
        phys_y = soly + ch / 2.0
        phys_x, phys_y = _clamp_centers(phys_x, phys_y, w, h, cw, ch, self.boundary_inset)

        if self.greedy_post_rounds > 0:
            phys_x, phys_y = _greedy_nudge_hard_centers(
                x=phys_x,
                y=phys_y,
                w=w,
                h=h,
                cw=cw,
                ch=ch,
                fixed_x=fixed_x,
                fixed_y=fixed_y,
                fixed_w=fixed_w,
                fixed_h=fixed_h,
                rounds=self.greedy_post_rounds,
                restarts=self.greedy_restarts,
                seed=13,
                gap=self.greedy_gap,
                inset=self.boundary_inset,
            )

        for k, g in enumerate(idx_list):
            placement[g, 0] = float(phys_x[k])
            placement[g, 1] = float(phys_y[k])

        if self.post_legalize_rounds > 0 and self.post_legalize_pair_ops > 0:
            placement = _legalize_hard_macros_multi_seed(
                placement,
                benchmark,
                num_seeds=self.post_legalize_seeds,
                jitter_std=self.post_legalize_jitter_std,
                boundary_inset=self.boundary_inset,
                max_pair_ops=self.post_legalize_pair_ops,
                max_rounds=self.post_legalize_rounds,
                gap=self.greedy_gap,
                idle_cap=28,
                seed=7,
            )

        # Soft macros are not optimized by DCCP; clamp them to avoid strict OOB invalidations.
        _clamp_nonfixed_macros_to_canvas(placement, benchmark, self.boundary_inset)

        return placement


def _iccad04_dir(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / name


def _run_local_evaluation() -> None:
    parser = argparse.ArgumentParser(description="Run DccpControlPlacer on one IBM ICCAD04 benchmark.")
    parser.add_argument("-b", "--benchmark", default="ibm01", help="Benchmark name (default: ibm01)")
    args = parser.parse_args()

    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    bench_dir = _iccad04_dir(args.benchmark)
    if not bench_dir.is_dir():
        print(f"Benchmark directory not found: {bench_dir}", file=sys.stderr)
        raise SystemExit(1)

    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"Loaded {benchmark.name}, movable hard: {(benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).sum().item()}")

    placer = DccpControlPlacer()
    placement = placer.place(benchmark)
    ok, viol = validate_placement(placement, benchmark)
    print(f"valid={ok} violations={viol[:3]!r}..." if len(viol) > 3 else f"valid={ok} violations={viol!r}")
    costs = compute_proxy_cost(placement, benchmark, plc)
    print(f"proxy={costs['proxy_cost']:.6f} overlaps={costs['overlap_count']}")


if __name__ == "__main__":
    _run_local_evaluation()
