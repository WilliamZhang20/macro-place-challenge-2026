"""Go-With-The-Winners (GWTW) Simulated Annealing — Python multiprocess.

Replicates the population-based meta-heuristic from
``external/MacroPlacement/CodeElements/SimulatedAnnealingGWTW``: instead of
a single SA run, ``num_workers`` independent SA workers explore the search
space in parallel.  After each ``syncup_freq`` fraction of total steps the
population is sorted by current cost, the top-K winners are kept, and the
remaining ``num_workers - top_k`` workers are replaced with clones of the
winners' placements (round-robin per ``resource_assignment``).  Cooling
continues across syncs; each worker's next batch picks up where it left
off but biased toward the winning placements.

Differences from the TILOS C++ reference:
  * **Process-level parallelism** (Python ``multiprocessing.Pool``) rather
    than OpenMP.  ``num_workers`` is upper-bounded by available CPUs.
  * Each worker initializes its own ``PlacementCost`` via ``PlcLookup``
    (PlacementCost is not robustly picklable; loading per-worker is the
    cleanest path).
  * Cost is computed by the real evaluator (``compute_proxy_cost``) —
    direct proxy in the inner loop, matching the rest of this codebase.

The move set reuses ``_tilos_moves`` helpers
(swap/shift/mirror/move/shuffle); GWTW owns the geometric cooling schedule.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

# Critical for slurm: switch torch shared-memory to filename-based so
# per-tensor fds don't accumulate when GWTW SA Pool workers share state.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

from macro_place.benchmark import Benchmark

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ---------------------------------------------------------------------------
# Worker-process globals (set once per process via the pool initializer).
# ---------------------------------------------------------------------------
_W_BENCHMARK: Optional[Benchmark] = None
_W_PLC = None


def _worker_init(benchmark: Benchmark) -> None:
    """Load PlacementCost in each worker process once."""
    global _W_BENCHMARK, _W_PLC
    # Re-add submissions to path inside the child process.
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from _plc_lookup import PlcLookup  # noqa: PLC0415

    _W_BENCHMARK = benchmark
    _W_PLC = PlcLookup().load(benchmark)


def _worker_sa_batch(args) -> Tuple[np.ndarray, float, np.ndarray, float, int, int, int]:
    """Run ``num_steps`` SA actions in this worker; return final state.

    Returns ``(current_array, current_proxy, best_array, best_proxy, accepted,
    evaluated, worker_id)``.  ``current_array`` is the worker's Metropolis
    state so subsequent syncs operate on a coherent state/cost pair.
    ``best_array``/``best_proxy`` track the lowest proxy seen by this worker
    within the batch, so the master can preserve true best-so-far placements.

    ``per_worker_budget_s`` caps each worker's wall time so a slow benchmark
    cannot blow the global budget within a single sync round.
    """
    if len(args) == 11:
        (
            worker_id,
            start_placement_arr,
            num_steps,
            seed,
            t_max,
            t_min,
            global_step_offset,
            cool_total_steps,
            action_probs,
            per_worker_budget_s,
            sa_disp_budget_frac,
        ) = args
    else:
        (
            worker_id,
            start_placement_arr,
            num_steps,
            seed,
            t_max,
            t_min,
            global_step_offset,
            cool_total_steps,
            action_probs,
            per_worker_budget_s,
        ) = args
        sa_disp_budget_frac = 0.10

    benchmark = _W_BENCHMARK
    plc = _W_PLC
    if benchmark is None or plc is None:
        # Worker init failed — fall through cleanly.
        return (
            start_placement_arr,
            float("inf"),
            start_placement_arr,
            float("inf"),
            0,
            0,
            worker_id,
        )

    from _hard_legalizer import legalize_hard  # noqa: PLC0415
    from _tilos_moves import (  # noqa: PLC0415
        _propose_swap,
        _propose_shift,
        _propose_mirror,
        _propose_move,
        _propose_shuffle,
    )
    from macro_place.objective import compute_proxy_cost  # noqa: PLC0415
    from macro_place.utils import validate_placement  # noqa: PLC0415

    n_hard = int(benchmark.num_hard_macros)
    movable_mask = (
        (~benchmark.macro_fixed[:n_hard]).detach().cpu().numpy().astype(bool)
    )
    movable_idx = np.flatnonzero(movable_mask)
    if movable_idx.size == 0:
        return start_placement_arr, float("inf"), 0, 0, worker_id

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
    half_w = 0.5 * sizes[:, 0]
    half_h = 0.5 * sizes[:, 1]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    grid_col = max(1, int(benchmark.grid_cols))
    grid_row = max(1, int(benchmark.grid_rows))
    bin_w = cw / grid_col
    bin_h = ch / grid_row

    probs = list(action_probs)
    if len(probs) != 5:
        probs = [0.2] * 5
    total = sum(max(0.0, float(p)) for p in probs)
    if total <= 0.0:
        probs = [0.2] * 5
        total = 1.0
    probs = [max(0.0, float(p)) / total for p in probs]
    thresholds = []
    acc = 0.0
    for p in probs:
        acc += p
        thresholds.append(acc)

    current = torch.from_numpy(np.array(start_placement_arr, dtype=np.float32, copy=True))
    try:
        current_proxy = float(compute_proxy_cost(current, benchmark, plc)["proxy_cost"])
    except Exception:
        return (
            start_placement_arr,
            float("inf"),
            start_placement_arr,
            float("inf"),
            0,
            0,
            worker_id,
        )
    best = current.clone()
    best_proxy = current_proxy

    log_t_ratio = math.log(max(t_min, 1e-12) / max(t_max, 1e-12))
    cool_total = max(1, int(cool_total_steps))

    fixed_mask = benchmark.macro_fixed
    has_fixed = bool(fixed_mask.any())
    if has_fixed:
        fixed_positions = benchmark.macro_positions[fixed_mask]

    accepted = 0
    evaluated = 0
    worker_start = time.monotonic()
    for local_step in range(int(num_steps)):
        if per_worker_budget_s > 0.0 and time.monotonic() - worker_start >= per_worker_budget_s:
            break
        cool_step = min(global_step_offset + local_step, cool_total - 1)
        t = t_max * math.exp(log_t_ratio * cool_step / max(1, cool_total - 1))

        action_roll = float(rng.random())
        if action_roll < thresholds[0]:
            prop = _propose_swap(
                current, movable_idx, half_w, half_h, cw, ch, sizes, rng
            )
        elif action_roll < thresholds[1]:
            prop = _propose_shift(
                current, movable_idx, half_w, half_h, cw, ch, bin_w, bin_h, rng
            )
        elif action_roll < thresholds[2]:
            prop = _propose_mirror(
                current, movable_idx, half_w, half_h, cw, ch, rng
            )
        elif action_roll < thresholds[3]:
            prop = _propose_move(
                current,
                movable_idx,
                half_w,
                half_h,
                cw,
                ch,
                grid_col,
                grid_row,
                bin_w,
                bin_h,
                rng,
            )
        else:
            prop = _propose_shuffle(
                current, movable_idx, half_w, half_h, cw, ch, rng
            )

        if has_fixed:
            prop[fixed_mask] = fixed_positions.to(prop.dtype)

        prop = legalize_hard(
            prop,
            benchmark,
            overlap_gap=1e-3,
            legalize_rounds=140,
            outer_passes=1,
            displacement_budget_frac=sa_disp_budget_frac,
            step_fraction=0.30,
        )
        ok, _ = validate_placement(prop, benchmark, check_overlaps=True)
        if not ok:
            continue
        try:
            new_proxy = float(compute_proxy_cost(prop, benchmark, plc)["proxy_cost"])
        except Exception:
            continue
        evaluated += 1

        delta = new_proxy - current_proxy
        if delta <= 0.0:
            current = prop
            current_proxy = new_proxy
            accepted += 1
        else:
            try:
                pacc = math.exp(-delta / max(t, 1e-12))
            except OverflowError:
                pacc = 0.0
            if float(rng.random()) < pacc:
                current = prop
                current_proxy = new_proxy
                accepted += 1

        if current_proxy < best_proxy - 1e-12:
            best = current.clone()
            best_proxy = current_proxy

    # Worker returns its current Metropolis state for population sync, plus
    # the batch best placement for global-best tracking.
    return (
        current.detach().cpu().numpy().astype(np.float32, copy=False),
        float(current_proxy),
        best.detach().cpu().numpy().astype(np.float32, copy=False),
        float(best_proxy),
        int(accepted),
        int(evaluated),
        int(worker_id),
    )


def _resource_assignment(num_workers: int, top_k: int) -> List[int]:
    """For each worker i in 0..num_workers-1, return the source-worker
    index it should clone from at sync.  Top-K workers clone from
    themselves (no-op); the remaining workers round-robin among 0..top_k-1.

    Mirrors the TILOS GWTW C++ logic in main.cpp lines 80-95.
    """
    top_k = max(1, min(int(top_k), int(num_workers)))
    assignment = list(range(num_workers))
    equal = num_workers // top_k
    total = equal * top_k
    next_worker = top_k
    counts = [equal] * top_k
    for i in range(top_k):
        if total < num_workers:
            counts[i] += 1
            total += 1
    for i in range(top_k):
        for _ in range(counts[i] - 1):
            if next_worker < num_workers:
                assignment[next_worker] = i
                next_worker += 1
    return assignment


def tilos_gwtw_sa_refine(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,  # unused in master; workers each load their own via PlcLookup.
    *,
    num_workers: int,
    num_iters: int,
    syncup_freq: float,
    top_k: int,
    time_budget_s: float,
    seed: int,
    t_max: float = 8e-5,
    t_min: float = 1e-8,
    action_probs: Sequence[float] = (0.20, 0.20, 0.20, 0.20, 0.20),
    log_progress: bool = False,
    sa_disp_budget_frac: Optional[float] = 0.10,
) -> Tuple[torch.Tensor, float, int, int]:
    """Population-based GWTW SA. Returns ``(best_placement, best_proxy, total_accepted, total_evaluated)``.

    The starting placement seeds ALL workers.  Each worker uses a distinct
    random seed so the initial SA trajectories diverge.  Between syncs,
    bottom workers are replaced by clones of the top-``top_k`` winners.
    """
    import multiprocessing as mp  # noqa: PLC0415

    n_hard = int(benchmark.num_hard_macros)
    if n_hard < 2 or num_iters <= 0 or num_workers <= 0 or time_budget_s <= 0.0:
        return placement.clone(), float("inf"), 0, 0

    num_workers = max(1, int(num_workers))
    top_k = max(1, min(int(top_k), num_workers))
    sync_iter = max(1, int(round(num_iters * float(syncup_freq))))
    cool_total = int(num_iters)
    assignment = _resource_assignment(num_workers, top_k)

    base = placement.detach().clone().float()
    base_arr = base.cpu().numpy().astype(np.float32, copy=True)

    # Worker placement state — initially all start from the same placement.
    worker_states: List[np.ndarray] = [base_arr.copy() for _ in range(num_workers)]
    worker_costs: List[float] = [float("inf")] * num_workers
    global_best_placement = base.clone()
    global_best_proxy = float("inf")
    # Master-side seed compute for the initial proxy reference.
    try:
        from macro_place.objective import compute_proxy_cost  # noqa: PLC0415

        global_best_proxy = float(
            compute_proxy_cost(base, benchmark, plc)["proxy_cost"]
        )
    except Exception:
        global_best_proxy = float("inf")

    total_accepted = 0
    total_evaluated = 0
    start_time = time.monotonic()
    iter_done = 0

    if log_progress:
        print(
            f"[gwtw-sa] start  workers={num_workers} top_k={top_k} "
            f"sync_iter={sync_iter} cool_total={cool_total} "
            f"budget_s={time_budget_s:.0f} init_proxy={global_best_proxy:.4f}",
            file=sys.stderr,
            flush=True,
        )

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(benchmark,),
    )
    try:
        sync_round = 0
        while iter_done < cool_total:
            if time.monotonic() - start_time >= time_budget_s:
                break
            batch_steps = min(sync_iter, cool_total - iter_done)
            # Cap each worker's batch time so a single sync round can't
            # blow the global budget.  Budget remaining → per-round wall
            # time; workers run in parallel so per-worker = per-round.
            remaining = time_budget_s - (time.monotonic() - start_time)
            # Reserve some time for sync overhead + future rounds.
            target_rounds_left = max(1, int((cool_total - iter_done) / max(1, sync_iter)))
            per_worker_budget = max(15.0, remaining / target_rounds_left * 0.85)
            tasks = []
            for w in range(num_workers):
                tasks.append(
                    (
                        w,
                        worker_states[w],
                        batch_steps,
                        int(seed) + sync_round * 9973 + w * 7919,
                        t_max,
                        t_min,
                        iter_done,
                        cool_total,
                        tuple(action_probs),
                        per_worker_budget,
                        sa_disp_budget_frac,
                    )
                )

            # Run all workers' batches in parallel; collect results.
            results = pool.map(_worker_sa_batch, tasks)

            for result in results:
                if len(result) == 7:
                    (
                        new_arr,
                        w_current_proxy,
                        w_best_arr,
                        w_best_proxy,
                        w_acc,
                        w_eval,
                        w_id,
                    ) = result
                else:
                    # Backward-compatible parse for any stale/imported helper.
                    new_arr, w_best_proxy, w_acc, w_eval, w_id = result
                    w_current_proxy = w_best_proxy
                    w_best_arr = new_arr
                worker_states[w_id] = new_arr
                worker_costs[w_id] = w_current_proxy
                total_accepted += w_acc
                total_evaluated += w_eval
                if w_best_proxy < global_best_proxy - 1e-12:
                    global_best_proxy = w_best_proxy
                    global_best_placement = torch.from_numpy(
                        w_best_arr.copy()
                    ).float()
                    if log_progress:
                        print(
                            f"[gwtw-sa] round={sync_round} worker={w_id}  "
                            f"new_global_best proxy={global_best_proxy:.4f}",
                            file=sys.stderr,
                            flush=True,
                        )

            iter_done += batch_steps
            sync_round += 1

            if iter_done >= cool_total or time.monotonic() - start_time >= time_budget_s:
                break

            # Sync: sort workers by cost, replace bottom (num_workers - top_k)
            # with clones of their assigned top-K worker.
            order = sorted(range(num_workers), key=lambda i: worker_costs[i])
            # Re-map states/costs into the sorted order so assignment uses
            # post-sort indices.
            sorted_states = [worker_states[i].copy() for i in order]
            sorted_costs = [worker_costs[i] for i in order]
            new_states = [sorted_states[i].copy() for i in range(num_workers)]
            new_costs = list(sorted_costs)
            for i in range(top_k, num_workers):
                src = assignment[i]  # index into top-K (post-sort)
                new_states[i] = sorted_states[src].copy()
                new_costs[i] = sorted_costs[src]
            worker_states = new_states
            worker_costs = new_costs
            if log_progress:
                top_str = ", ".join(f"{worker_costs[k]:.4f}" for k in range(top_k))
                print(
                    f"[gwtw-sa] sync round={sync_round} iter={iter_done}/{cool_total} "
                    f"top_k_costs=[{top_str}]  global_best={global_best_proxy:.4f}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        pool.close()
        pool.join()

    if log_progress:
        elapsed = time.monotonic() - start_time
        print(
            f"[gwtw-sa] done  iters={iter_done}/{cool_total} sync_rounds={sync_round} "
            f"evaluated={total_evaluated} accepted={total_accepted} "
            f"final_best={global_best_proxy:.4f} elapsed={elapsed:.0f}s",
            file=sys.stderr,
            flush=True,
        )
    return global_best_placement, float(global_best_proxy), total_accepted, total_evaluated
