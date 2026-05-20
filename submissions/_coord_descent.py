"""Coordinate-descent macro polish — best-of-both-worlds.

Combines the TILOS reference algorithm
(``external/MacroPlacement/CodeElements/Plc_client/coordinate_descent_placer.py``)
with our pipeline's direct-proxy evaluation + wall-clock budget:

  * **k-distance bounded search**: per macro, search all grid cells within
    Manhattan distance ``k`` of the current cell.  ``k`` defaults to
    ``max(cols, rows) // 3`` per TILOS, but is adapted down on heavy
    benchmarks so per-pass cost stays bounded.
  * **``plc.get_node_mask`` feasibility filter**: only candidate cells that
    the evaluator reports as overlap-free are scored.  Drops 50-80% of
    proposals before they hit the expensive scoring path.
  * **``cell_search_prob`` random subsampling**: thin the candidate set per
    macro further when k×k neighborhoods are too large.
  * **Descending-size node order** (TILOS empirical winner): bigger macros
    get first crack at their best cell, reducing backtracking from
    small-macro pollution.
  * **Direct-proxy cost** via ``compute_proxy_cost`` — same proxy the
    final scorer uses.  Strictly-improving acceptance: monotone-improving
    from the caller's POV.
  * **Time budget cap**: ``time_budget_s`` upper-bounds wall time so the
    polish stays inside the per-benchmark budget; ``max_passes`` caps
    epochs (TILOS uses 10 by default; we usually do 1-2).

The caller passes the ``benchmark`` and ``plc``; this module borrows the
plc's place/unplace API (same plc passed through the rescue stage).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _hard_legalizer import legalize_hard  # noqa: E402


def _adaptive_k(grid_cols: int, grid_rows: int, n_hard: int) -> int:
    """Choose a k-distance bound that keeps per-macro candidate counts
    manageable.  Wider on small benchmarks, narrower on big ones."""
    base = max(grid_cols, grid_rows) // 3
    if n_hard <= 100:
        return max(2, base)
    if n_hard <= 250:
        return max(2, min(base, 8))
    if n_hard <= 450:
        return max(2, min(base, 5))
    if n_hard <= 800:
        return max(2, min(base, 3))
    return 2


def _macro_to_plc_index(benchmark: Benchmark, plc) -> List[int]:
    """Map hard macro torch indices (0..num_hard_macros-1) to plc node
    indices.  Uses benchmark.hard_macro_indices when available; falls
    back to enumerating plc.hard_macro_indices."""
    n_hard = int(benchmark.num_hard_macros)
    if hasattr(benchmark, "hard_macro_indices"):
        indices = benchmark.hard_macro_indices
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        return [int(i) for i in indices][:n_hard]
    # Fallback — plc has its own hard_macro_indices list.
    return list(plc.hard_macro_indices)[:n_hard]


def coord_descent_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    time_budget_s: float,
    max_passes: int = 1,
    k_distance_bound: Optional[int] = None,
    cell_search_prob: float = 1.0,
    node_order: str = "descending_size",
    seed: int = 20260520,
    use_mask: bool = True,
    log_progress: bool = False,
) -> Tuple[torch.Tensor, float, int]:
    """Return ``(best_placement, best_proxy, accepted_moves)``.

    Strictly-improving CD.  ``k_distance_bound=None`` -> adaptive based
    on grid size and macro count.
    """
    n_hard = int(benchmark.num_hard_macros)
    if n_hard < 2 or time_budget_s <= 0.0:
        return placement.clone(), float("inf"), 0
    movable_mask = (~benchmark.macro_fixed[:n_hard]).detach().cpu().numpy().astype(bool)
    movable_idx = np.flatnonzero(movable_mask)
    if movable_idx.size == 0:
        return placement.clone(), float("inf"), 0

    # Set the plc to the starting placement and score it.
    best = placement.detach().clone().float()
    try:
        best_proxy = float(compute_proxy_cost(best, benchmark, plc)["proxy_cost"])
    except Exception:
        return best, float("inf"), 0

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    grid_col = max(1, int(benchmark.grid_cols))
    grid_row = max(1, int(benchmark.grid_rows))
    bin_w = cw / grid_col
    bin_h = ch / grid_row
    sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
    half_w = 0.5 * sizes[:, 0]
    half_h = 0.5 * sizes[:, 1]
    macro_area = sizes[:, 0] * sizes[:, 1]

    fixed_mask = benchmark.macro_fixed
    has_fixed = bool(fixed_mask.any())
    if has_fixed:
        fixed_positions = benchmark.macro_positions[fixed_mask]

    if k_distance_bound is None:
        k_bound = _adaptive_k(grid_col, grid_row, n_hard)
    else:
        k_bound = max(1, int(k_distance_bound))
    cell_search_prob = float(min(1.0, max(0.01, cell_search_prob)))

    # Node order: descending size by default (TILOS empirical winner).
    if node_order == "descending_size":
        order = sorted(movable_idx.tolist(), key=lambda k: -macro_area[k])
    elif node_order == "random":
        rng_order = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        order = movable_idx.tolist()
        rng_order.shuffle(order)
    elif node_order == "random_macro_first":
        rng_order = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        order = movable_idx.tolist()
        rng_order.shuffle(order)
    else:
        order = movable_idx.tolist()

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    plc_node_indices = _macro_to_plc_index(benchmark, plc)

    accepted = 0
    start_time = time.monotonic()

    if log_progress:
        print(
            f"[coord-desc] start  proxy={best_proxy:.4f} budget_s={time_budget_s:.0f} "
            f"passes={max_passes} k_bound={k_bound} cell_prob={cell_search_prob:.2f} "
            f"order={node_order} use_mask={use_mask} n_movable={movable_idx.size}",
            file=sys.stderr,
            flush=True,
        )

    # Cache current x/y by macro torch index — we update on accept.
    cur_pos = best[:n_hard].detach().cpu().numpy().astype(np.float64).copy()

    def _candidate_cells_for(k: int, curr_cell: int) -> List[int]:
        """Return list of candidate grid cell indices within Manhattan
        distance ``k_bound`` of ``curr_cell``, with random subsampling
        by ``cell_search_prob``.  Includes the current cell."""
        curr_row, curr_col = divmod(curr_cell, grid_col)
        cells: List[int] = [curr_cell]
        for r_off in range(-k_bound, k_bound + 1):
            r = curr_row + r_off
            if r < 0 or r >= grid_row:
                continue
            cmax_off = k_bound - abs(r_off)
            for c_off in range(-cmax_off, cmax_off + 1):
                if r_off == 0 and c_off == 0:
                    continue
                c = curr_col + c_off
                if c < 0 or c >= grid_col:
                    continue
                if cell_search_prob >= 1.0 or float(rng.random()) <= cell_search_prob:
                    cells.append(r * grid_col + c)
        return cells

    for pass_idx in range(int(max_passes)):
        if time.monotonic() - start_time >= time_budget_s:
            break
        moved_this_pass = 0
        # Re-shuffle for random orders each pass.
        if node_order in ("random", "random_macro_first"):
            rng_order = np.random.default_rng(int(seed) + pass_idx * 9973)
            order = list(order)
            rng_order.shuffle(order)

        for k_int in order:
            if time.monotonic() - start_time >= time_budget_s:
                break
            k = int(k_int)
            cur_x = float(cur_pos[k, 0])
            cur_y = float(cur_pos[k, 1])
            cur_col = int(cur_x / bin_w)
            cur_row = int(cur_y / bin_h)
            cur_col = max(0, min(grid_col - 1, cur_col))
            cur_row = max(0, min(grid_row - 1, cur_row))
            curr_cell = cur_row * grid_col + cur_col

            candidate_cells = _candidate_cells_for(k_bound, curr_cell)

            # Optional mask-based feasibility prefilter.
            if use_mask:
                try:
                    plc_node = plc_node_indices[k]
                    # The plc's node_mask is computed for the current
                    # placement.  We don't unplace because our pipeline
                    # uses ``set_pos`` semantics, not plc.place_node.
                    # The mask tells us which cells are ovr-free for
                    # *this node* given everything else.  Cells where
                    # mask == 1 are feasible.
                    mask = plc.get_node_mask(plc_node)
                    if mask is not None:
                        feasible = set(int(i) for i, m in enumerate(mask) if m > 0)
                        # Always include current cell so we don't drop
                        # the macro out of its own slot.
                        feasible.add(curr_cell)
                        candidate_cells = [c for c in candidate_cells if c in feasible]
                except Exception:
                    pass  # mask unavailable — fall through to legalize-check

            local_best_proxy = best_proxy
            local_best = None
            for cell in candidate_cells:
                if time.monotonic() - start_time >= time_budget_s:
                    break
                if cell == curr_cell:
                    continue
                row = cell // grid_col
                col = cell % grid_col
                new_x = (col + 0.5) * bin_w
                new_y = (row + 0.5) * bin_h
                new_x = float(np.clip(new_x, half_w[k] + 1e-3, cw - half_w[k] - 1e-3))
                new_y = float(np.clip(new_y, half_h[k] + 1e-3, ch - half_h[k] - 1e-3))
                if abs(new_x - cur_x) < 1e-6 and abs(new_y - cur_y) < 1e-6:
                    continue
                prop = best.clone()
                prop[k, 0] = new_x
                prop[k, 1] = new_y
                if has_fixed:
                    prop[fixed_mask] = fixed_positions.to(prop.dtype)
                prop = legalize_hard(
                    prop,
                    benchmark,
                    overlap_gap=1e-3,
                    legalize_rounds=120,
                    outer_passes=1,
                    displacement_budget_frac=0.08,
                    step_fraction=0.30,
                )
                ok, _ = validate_placement(prop, benchmark, check_overlaps=True)
                if not ok:
                    continue
                try:
                    new_proxy = float(
                        compute_proxy_cost(prop, benchmark, plc)["proxy_cost"]
                    )
                except Exception:
                    continue
                if new_proxy < local_best_proxy - 1e-12:
                    local_best_proxy = new_proxy
                    local_best = prop
            if local_best is not None:
                best = local_best
                best_proxy = local_best_proxy
                cur_pos = best[:n_hard].detach().cpu().numpy().astype(np.float64).copy()
                accepted += 1
                moved_this_pass += 1
                # Re-set placement on plc so subsequent get_node_mask
                # calls reflect the new layout.  compute_proxy_cost on
                # `best` already did this.
                if log_progress:
                    print(
                        f"[coord-desc] pass={pass_idx} macro={k} "
                        f"moved ({cur_x:.1f},{cur_y:.1f})->({new_x:.1f},{new_y:.1f}) "
                        f"new_proxy={best_proxy:.4f}",
                        file=sys.stderr,
                        flush=True,
                    )
        if log_progress:
            print(
                f"[coord-desc] pass={pass_idx} done  moves={moved_this_pass} "
                f"best={best_proxy:.4f}  elapsed={time.monotonic()-start_time:.0f}s",
                file=sys.stderr,
                flush=True,
            )
        if moved_this_pass == 0:
            break

    if log_progress:
        print(
            f"[coord-desc] final  proxy={best_proxy:.4f} accepted={accepted} "
            f"elapsed={time.monotonic()-start_time:.0f}s",
            file=sys.stderr,
            flush=True,
        )
    return best, best_proxy, accepted
