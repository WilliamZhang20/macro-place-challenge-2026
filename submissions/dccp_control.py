"""
Pure DCCP macro placement (control / experiment).

Implements the slide formulation without circle packing or k-NN regularizers.

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

**Warm start:** Uses the benchmark hand-off placement (README: initial floorplan
as reference), clamped in-bounds and projected onto the centroid-zero affine
subspace in shifted coordinates — not a greedy overlap heuristic.

**Feasibility:** The ``dccp`` CCP attaches slack to non-DCP constraints and
increases penalty each iteration via ``tau <- min(tau * mu, tau_max)`` (see
``dccp.problem.dccp``). We tune ``tau``, ``mu``, ``tau_max``, and run a small
number of **manual** restarts with jittered initial points. We do *not* use
``ccp_times > 1``, which forces random initialization and discards the hand-off.

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


def _project_shifted_centroid(
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    cw: float,
    ch: float,
    inset: float,
    iters: int = 28,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project (cx, cy) onto box bounds in shifted coordinates while approximately
    enforcing sum(cx)=sum(cy)=0 (required linear equalities in the DCCP problem).
    """
    half_w = w * 0.5
    half_h = h * 0.5
    lbx = -cw / 2.0 + half_w + inset
    ubx = cw / 2.0 - half_w - inset
    lby = -ch / 2.0 + half_h + inset
    uby = ch / 2.0 - half_h - inset
    x = np.asarray(cx, dtype=np.float64).copy()
    y = np.asarray(cy, dtype=np.float64).copy()
    for _ in range(iters):
        x = np.minimum(np.maximum(x, lbx), ubx)
        y = np.minimum(np.maximum(y, lby), uby)
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
    x = np.minimum(np.maximum(x, lbx), ubx)
    y = np.minimum(np.maximum(y, lby), uby)
    return x, y


def _run_dccp_with_recovery(
    prob: cp.Problem,
    cx: cp.Variable,
    cy: cp.Variable,
    *,
    base_cx0: np.ndarray,
    base_cy0: np.ndarray,
    dccp_max_iter: int,
    ep: float,
    max_slack: float,
    tau: float,
    mu: float,
    tau_max: float,
    restarts: int,
    restart_jitter: float,
    seed: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve with DCCP slack-penalty schedule (tau, mu, tau_max). Manual restarts
    with jitter preserve the hand-off as restart 0; ``ccp_times>1`` is avoided.
    """
    n = int(base_cx0.shape[0])
    rng = np.random.default_rng(int(seed))
    restarts = max(1, int(restarts))

    # Tiered schedules: stronger slack weight helps feasibility recovery.
    schedules: List[Tuple[float, float, float, int]] = [
        (tau, mu, tau_max, dccp_max_iter),
        (max(tau * 4.0, 0.02), max(mu, 1.25), max(tau_max, 1e9), max(dccp_max_iter, 96)),
        (max(tau * 16.0, 0.08), max(mu * 1.05, 1.35), max(tau_max * 10.0, 1e10), max(dccp_max_iter, 120)),
    ]

    best_x: Optional[np.ndarray] = None
    best_y: Optional[np.ndarray] = None
    best_key: Optional[Tuple[int, float]] = None  # (converged: 0 better, cost)

    for r in range(restarts):
        cx0 = np.asarray(base_cx0, dtype=np.float64).copy()
        cy0 = np.asarray(base_cy0, dtype=np.float64).copy()
        if r > 0 and restart_jitter > 0.0:
            cx0 = cx0 + rng.normal(0.0, restart_jitter, size=n)
            cy0 = cy0 + rng.normal(0.0, restart_jitter, size=n)

        for t_sched, m_sched, tmax_sched, mi_sched in schedules:
            cx.value = cx0.copy()
            cy.value = cy0.copy()
            try:
                dccp_ccp(
                    prob,
                    max_iter=int(mi_sched),
                    solver=cp.CLARABEL,
                    ep=ep,
                    max_slack=max_slack,
                    tau=float(t_sched),
                    mu=float(m_sched),
                    tau_max=float(tmax_sched),
                    ccp_times=1,
                )
            except Exception:
                continue

            if cx.value is None or cy.value is None:
                continue
            sx = np.asarray(cx.value, dtype=np.float64).ravel()
            sy = np.asarray(cy.value, dtype=np.float64).ravel()
            if sx.shape[0] != n or not np.all(np.isfinite(sx)) or not np.all(np.isfinite(sy)):
                continue

            status = getattr(prob, "_status", "Not_converged")
            converged = 1 if status == "Converged" else 0
            cost = float(prob.value) if prob.value is not None else float("inf")
            key = (-converged, cost)
            if best_key is None or key < best_key:
                best_key = key
                best_x, best_y = sx.copy(), sy.copy()
            if converged:
                return best_x, best_y

            cx0, cy0 = sx, sy

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


class DccpControlPlacer:
    """
    Pure slide DCCP placement on movable hard macros (no hybrid heuristics).

    Parameters
    ----------
    dccp_max_iter : int
        Inner iterations passed to ``dccp.problem.dccp`` (first penalty tier).
    ep, max_slack : float
        DCCP tolerances (see ``dccp.problem.dccp``).
    tau, mu, tau_max : float
        Slack penalty schedule: ``tau`` increases by ``mu`` each iteration up to ``tau_max``.
    dccp_restarts : int
        Manual restarts with ``restart_jitter`` on shifted centers (``ccp_times`` stays 1).
    """

    def __init__(
        self,
        dccp_max_iter: int = 80,
        ep: float = 1e-4,
        max_slack: float = 1e-2,
        tau: float = 0.005,
        mu: float = 1.2,
        tau_max: float = 1e8,
        dccp_restarts: int = 3,
        restart_jitter: float = 0.02,
        boundary_inset: float = 1e-3,
    ):
        self.dccp_max_iter = dccp_max_iter
        self.ep = ep
        self.max_slack = max_slack
        self.tau = float(tau)
        self.mu = float(mu)
        self.tau_max = float(tau_max)
        self.dccp_restarts = max(1, int(dccp_restarts))
        self.restart_jitter = float(max(0.0, restart_jitter))
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

        # Benchmark hand-off (README initial floorplan): clamp, then shifted coords + centroid.
        warm_x, warm_y = _clamp_centers(pc[:, 0], pc[:, 1], w, h, cw, ch, self.boundary_inset)
        cx0, cy0 = _project_shifted_centroid(
            warm_x - cw / 2.0,
            warm_y - ch / 2.0,
            w,
            h,
            cw,
            ch,
            self.boundary_inset,
        )

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

        solx, soly = _run_dccp_with_recovery(
            prob,
            cx,
            cy,
            base_cx0=cx0,
            base_cy0=cy0,
            dccp_max_iter=self.dccp_max_iter,
            ep=self.ep,
            max_slack=self.max_slack,
            tau=self.tau,
            mu=self.mu,
            tau_max=self.tau_max,
            restarts=self.dccp_restarts,
            restart_jitter=self.restart_jitter,
            seed=0,
        )
        if solx is None or soly is None:
            return placement

        # Physical centers from DCCP candidate.
        phys_x = solx + cw / 2.0
        phys_y = soly + ch / 2.0
        phys_x, phys_y = _clamp_centers(phys_x, phys_y, w, h, cw, ch, self.boundary_inset)

        for k, g in enumerate(idx_list):
            placement[g, 0] = float(phys_x[k])
            placement[g, 1] = float(phys_y[k])

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
