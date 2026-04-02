"""
DCCP-based macro placer (draft).

Uses the dccp package CCP implementation (``dccp.problem.dccp``) on a
cvxpy ``Problem``, after ``is_dccp`` validation.  Optimization is in
**displacements** ``(dx, dy)`` from the **baseline** initial placement
``(p0_x, p0_y)`` (centers = p0 + d), so the DCCP step is explicitly a refinement
on top of the legal input placement.

Circle packing is only a **relaxation** for rectangles; after DCCP we run a
light axis push legalization. If anything is still illegal, we return a full
copy of the benchmark **initial placement** (same overlap metric as
``compute_proxy_cost`` / ``compute_overlap_metrics``).

DCCP program (per solve, variables ``dx, dy``):
  minimize    ||dx||^2 + ||dy||^2 + lam * ||A dx||^2 + lam * ||A dy||^2
  subject to  (r_i+r_j)^2 - ||(p0+dp)_i - (p0+dp)_j||^2 <= 0   (concave)
              centers stay inside canvas (linear in dx, dy)

Usage:
    uv run evaluate submissions/dccp_placer.py -b ibm01
    uv run evaluate submissions/dccp_placer.py --all

    # Standalone (from repo root): load ibm01, validate, print proxy cost
    python submissions/dccp_placer.py
    python submissions/dccp_placer.py -b ibm03
    python submissions/dccp_placer.py --verify   # compare vs initial placement
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Set, Tuple

import cvxpy as cp
import numpy as np
import torch
from dccp import is_dccp
from dccp.problem import dccp as dccp_ccp

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics
from macro_place.utils import validate_placement


def _inscribed_radius(w: float, h: float, shrink: float) -> float:
    """Circle radius inside the macro; shrink < 1 relaxes packing for feasibility."""
    return shrink * min(w, h) / 2.0


def _legalize_centers(
    centers: np.ndarray,
    sizes: np.ndarray,
    cw: float,
    ch: float,
    max_iters: int = 2000,
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
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                sep_x = hw[i] + hw[j] + gap
                sep_y = hh[i] + hh[j] + gap
                ox = sep_x - abs(dx)
                oy = sep_y - abs(dy)
                if ox <= 0 or oy <= 0:
                    continue
                moved = True
                # Separate along the axis that needs less total motion
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
    edges: Sequence[Tuple[int, int]],
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
    lam = 0.15
    if edges:
        A_e = np.zeros((len(edges), n), dtype=np.float64)
        for k, (i, j) in enumerate(edges):
            A_e[k, i] = 1.0
            A_e[k, j] = -1.0
        obj += lam * cp.sum_squares(A_e @ dx) + lam * cp.sum_squares(A_e @ dy)

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
    ):
        self.max_outer_iters = max_outer_iters
        self.dccp_max_iter = dccp_max_iter
        self.knn_k = knn_k
        self.circle_shrink = circle_shrink

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Snapshot before any edits; used as guaranteed legal fallback (matches initial .plc).
        baseline = benchmark.macro_positions.detach().clone()
        placement = baseline.clone()
        hard_movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_idx = torch.where(hard_movable)[0]
        if movable_idx.numel() == 0:
            return placement

        idx_list = [int(i) for i in movable_idx.tolist()]
        n = len(idx_list)
        sizes = benchmark.macro_sizes[idx_list].numpy().astype(np.float64)
        p0 = benchmark.macro_positions[idx_list].numpy().astype(np.float64)
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)

        half_w = sizes[:, 0] / 2.0
        half_h = sizes[:, 1] / 2.0

        radii = np.array(
            [_inscribed_radius(float(sizes[i, 0]), float(sizes[i, 1]), self.circle_shrink) for i in range(n)],
            dtype=np.float64,
        )

        # Baseline centers (legal initial placement from benchmark)
        centers = p0.copy()
        pair_set: Set[Tuple[int, int]] = self._initial_pairs(centers, radii, n)
        edges = _knn_edges(p0, self.knn_k)

        for _outer in range(self.max_outer_iters):
            x_sol, y_sol = self._solve_dccp(
                p0=p0,
                half_w=half_w,
                half_h=half_h,
                radii=radii,
                cw=cw,
                ch=ch,
                pairs=sorted(pair_set),
                edges=edges,
                n=n,
            )
            if x_sol is None or y_sol is None:
                break

            centers[:, 0] = x_sol
            centers[:, 1] = y_sol
            # DCCP uses a circle relaxation; axis push often clears rectangle overlaps here.
            # Must check *after* legalization: pre-legalize overlap can be large while a single
            # legalize pass fixes everything — otherwise we keep adding DCCP pairs, later solves
            # get worse, and the final iterate may not legalize within max_iters.
            centers = _legalize_centers(centers, sizes, cw, ch)

            viol = self._overlapping_pairs(centers, sizes, margin=1e-3)
            if not viol:
                break
            for i, j in viol:
                a, b = (i, j) if i < j else (j, i)
                pair_set.add((a, b))

        # Final pass (noop if already legal); cheap insurance if the last outer iter broke early.
        centers = _legalize_centers(centers, sizes, cw, ch)

        for k, tensor_i in enumerate(idx_list):
            placement[tensor_i, 0] = float(centers[k, 0])
            placement[tensor_i, 1] = float(centers[k, 1])

        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = baseline[fixed_mask]

        # Float32 placement tensor can be epsilon-outside canvas after numpy→torch (validate fails
        # and we incorrectly revert to baseline).
        movable_mask = ~fixed_mask
        if movable_mask.any():
            hw = benchmark.macro_sizes[:, 0] * 0.5
            hh = benchmark.macro_sizes[:, 1] * 0.5
            cw = float(benchmark.canvas_width)
            ch = float(benchmark.canvas_height)
            placement[:, 0] = torch.where(
                movable_mask,
                torch.clamp(placement[:, 0], hw, cw - hw),
                placement[:, 0],
            )
            placement[:, 1] = torch.where(
                movable_mask,
                torch.clamp(placement[:, 1], hh, ch - hh),
                placement[:, 1],
            )

        # Evaluator uses compute_overlap_metrics (must be 0 for VALID in evaluate harness).
        if compute_overlap_metrics(placement, benchmark)["overlap_count"] > 0:
            return baseline.clone()
        ok, _ = validate_placement(placement, benchmark, check_overlaps=False)
        if not ok:
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
        edges: Sequence[Tuple[int, int]],
        n: int,
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
            edges=edges,
        )

        if not is_dccp(prob):
            return None, None

        # CCP: start from baseline (zero displacement).
        dx.value = np.zeros(n, dtype=np.float64)
        dy.value = np.zeros(n, dtype=np.float64)

        try:
            dccp_ccp(
                prob,
                max_iter=self.dccp_max_iter,
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
        n = centers.shape[0]
        out: List[Tuple[int, int]] = []
        for i in range(n):
            xi, yi = centers[i]
            wi, hi = sizes[i, 0], sizes[i, 1]
            for j in range(i + 1, n):
                xj, yj = centers[j]
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
