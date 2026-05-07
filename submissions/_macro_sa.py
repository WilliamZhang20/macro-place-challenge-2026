"""Direct macro-position simulated annealing with overlap-penalty acceptance.

Cost
----
``cost = WL_norm + density_weight * D_norm + λ(t) * overlap_area_norm``

where:
  * ``WL_norm`` and ``D_norm`` are the same surrogate components used by
    the B*-tree path (``_btree_sa._hpwl_movable``, ``_density_penalty``).
  * ``overlap_area_norm`` is total pairwise hard-macro overlap area
    divided by canvas area.
  * ``λ(t)`` ramps geometrically from a small ``lambda_init`` to a large
    ``lambda_final`` over the time budget. Early in the search, overlap is
    cheap so SA can explore; late in the search, overlap is heavily
    penalized so the trajectory exits in a feasible configuration.

We deliberately skip routing congestion in the surrogate: the RUDY
formulation we tried was anti-correlated with TILOS true congestion at
the operating point of this challenge (see post-mortem on the B*-tree
branch). Ranking is decided by the true TILOS proxy at selection time.

Moves
-----
* ``jiggle``: pick one movable macro, propose new center ``current +
  N(0, σ(t))``, σ annealed from 10% of canvas to 0.5%.
* ``swap``: pick two macros, swap positions.
* ``lns`` (every ``lns_period`` iterations): destroy a random fraction
  ``lns_destroy_frac`` of macros, re-place each at the HPWL centroid of
  its net-neighbors plus small jitter.

Acceptance is standard Metropolis on the full scalar cost. The cost
delta is recomputed by full surrogate recomputation per move (sub-ms in
numpy at ibm01 scale; faster than maintaining incremental net bboxes).
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _btree_sa import (  # noqa: E402
    SAContext,
    build_sa_context,
    _bin_overlap_xy,
    _density_penalty,
    _hpwl_movable,
)


# ---------------------------------------------------------------------------
# Pairwise overlap (vectorized)
# ---------------------------------------------------------------------------


def _pairwise_overlap_metrics(
    centers: np.ndarray, sizes: np.ndarray
) -> Tuple[float, float, int]:
    """Symmetric pairwise overlap among rectangles. Returns
    ``(total_area, total_penetration, num_pairs)``.
    """
    n = centers.shape[0]
    if n < 2:
        return 0.0, 0.0, 0
    half = 0.5 * sizes
    ll = centers - half
    ur = centers + half
    ox = np.maximum(
        0.0,
        np.minimum(ur[:, 0:1], ur[:, 0][None, :])
        - np.maximum(ll[:, 0:1], ll[:, 0][None, :]),
    )
    oy = np.maximum(
        0.0,
        np.minimum(ur[:, 1:2], ur[:, 1][None, :])
        - np.maximum(ll[:, 1:2], ll[:, 1][None, :]),
    )
    np.fill_diagonal(ox, 0.0)
    np.fill_diagonal(oy, 0.0)
    overlap_area = ox * oy
    pen = np.minimum(ox, oy)
    pair_mask = (ox > 0.0) & (oy > 0.0)
    n_pairs = int(pair_mask.sum() // 2)
    return float(0.5 * overlap_area.sum()), float(0.5 * pen.sum()), n_pairs


def _crosswise_overlap_metrics(
    centers_a: np.ndarray,
    sizes_a: np.ndarray,
    centers_b: np.ndarray,
    sizes_b: np.ndarray,
) -> Tuple[float, float, int]:
    """Asymmetric overlap between two disjoint sets A and B (e.g. movable
    macros vs fixed macros). Returns ``(total_area, total_penetration,
    num_pairs)`` over all (a, b) pairs.
    """
    na = centers_a.shape[0]
    nb = centers_b.shape[0]
    if na == 0 or nb == 0:
        return 0.0, 0.0, 0
    ll_a = centers_a - 0.5 * sizes_a
    ur_a = centers_a + 0.5 * sizes_a
    ll_b = centers_b - 0.5 * sizes_b
    ur_b = centers_b + 0.5 * sizes_b
    ox = np.maximum(
        0.0,
        np.minimum(ur_a[:, 0:1], ur_b[:, 0][None, :])
        - np.maximum(ll_a[:, 0:1], ll_b[:, 0][None, :]),
    )
    oy = np.maximum(
        0.0,
        np.minimum(ur_a[:, 1:2], ur_b[:, 1][None, :])
        - np.maximum(ll_a[:, 1:2], ll_b[:, 1][None, :]),
    )
    overlap_area = ox * oy
    pen = np.minimum(ox, oy)
    pair_mask = (ox > 0.0) & (oy > 0.0)
    return float(overlap_area.sum()), float(pen.sum()), int(pair_mask.sum())


# Back-compat / convenience wrappers
def _pairwise_overlap_area(centers: np.ndarray, sizes: np.ndarray) -> float:
    return _pairwise_overlap_metrics(centers, sizes)[0]


def _pairwise_overlap_count(centers: np.ndarray, sizes: np.ndarray, tol: float = 0.0) -> int:
    return _pairwise_overlap_metrics(centers, sizes)[2]


# ---------------------------------------------------------------------------
# Surrogate (movable-macro coords as the optimization variable)
# ---------------------------------------------------------------------------


def _full_centers(centers_movable: np.ndarray, ctx: SAContext) -> np.ndarray:
    """Build full (num_macros, 2) center array from movable subset."""
    full = ctx.fixed_centers.copy()
    full[ctx.movable_idx] = centers_movable
    return full


def _surrogate_eval(
    centers_movable: np.ndarray,
    ctx: SAContext,
    *,
    density_weight: float,
    lam: float,
    bins: int,
) -> Tuple[float, dict]:
    """Compute scalar surrogate cost and component breakdown."""
    full = _full_centers(centers_movable, ctx)

    # WL
    hpwl_raw = _hpwl_movable(full, ctx)
    wl_norm = hpwl_raw / max(ctx.hpwl_norm, 1e-9)

    # Density (movable + fixed macros, on a coarse grid)
    sizes_movable = ctx.sizes[ctx.movable_idx]
    fixed_macro_mask = np.ones(ctx.fixed_centers.shape[0], dtype=bool)
    fixed_macro_mask[ctx.movable_idx] = False
    fixed_centers = ctx.fixed_centers[fixed_macro_mask]
    fixed_sizes = ctx.sizes[fixed_macro_mask]
    if density_weight > 0:
        d_norm = _density_penalty(
            centers_movable,
            sizes_movable,
            fixed_centers,
            fixed_sizes,
            ctx.canvas,
            bins=bins,
        )
    else:
        d_norm = 0.0

    # Overlap penalty: hybrid of per-pair fixed cost (so grid-aligned
    # layouts can't hide many sliver-thin overlaps in float-precision noise)
    # and penetration depth (smooth gradient toward feasibility).
    #     pen_term = pair_cost_um * num_pairs + sum_{pairs} min(dx, dy)
    #     pen_norm = pen_term / canvas_semi_perimeter
    # ``pair_cost_um`` is the per-pair fixed cost in microns; choose so
    # that one overlapping pair costs about ``min(macro_size)``.
    ov_area, ov_pen, ov_pairs = _pairwise_overlap_metrics(
        centers_movable, sizes_movable
    )
    fixed_macro_mask = np.ones(ctx.fixed_centers.shape[0], dtype=bool)
    fixed_macro_mask[ctx.movable_idx] = False
    fixed_centers = ctx.fixed_centers[fixed_macro_mask]
    fixed_sizes = ctx.sizes[fixed_macro_mask]
    ov_fix_area, ov_fix_pen, ov_fix_pairs = _crosswise_overlap_metrics(
        centers_movable,
        sizes_movable,
        fixed_centers,
        fixed_sizes,
    )
    cw, ch = ctx.canvas
    norm_len = 0.5 * (cw + ch)
    pair_cost_um = 0.005  # 5 nm; outweighs typical WL-from-jiggle improvement
    pen_pairs = ov_pairs + ov_fix_pairs
    pen_term = pair_cost_um * pen_pairs + ov_pen + ov_fix_pen
    pen_norm = pen_term / max(norm_len, 1e-9)

    cost = wl_norm + density_weight * d_norm + lam * pen_norm
    return cost, {
        "WL_norm": wl_norm,
        "D_norm": d_norm,
        "overlap_area": ov_area,
        "overlap_pen": ov_pen,
        "overlap_pairs": ov_pairs,
        "overlap_fixed_area": ov_fix_area,
        "overlap_fixed_pen": ov_fix_pen,
        "overlap_fixed_pairs": ov_fix_pairs,
        "overlap_total_pairs": pen_pairs,
        "pen_norm": pen_norm,
        "lambda": lam,
    }


# ---------------------------------------------------------------------------
# LNS repair (greedy by HPWL centroid)
# ---------------------------------------------------------------------------


def _macro_to_local_nets(ctx: SAContext) -> List[np.ndarray]:
    """For each movable macro (local index), list of net IDs it belongs to."""
    n_move = ctx.movable_idx.size
    out: List[List[int]] = [[] for _ in range(n_move)]
    starts = ctx.move_csr_offsets[:-1]
    ends = ctx.move_csr_offsets[1:]
    for net_id, (s, e) in enumerate(zip(starts, ends)):
        if s == e:
            continue
        for local in ctx.move_csr_local[s:e]:
            out[int(local)].append(net_id)
    return [np.asarray(x, dtype=np.int64) for x in out]


def _hpwl_centroid_for_macro(
    local_idx: int,
    centers_movable: np.ndarray,
    ctx: SAContext,
    macro_nets: List[np.ndarray],
) -> Tuple[float, float]:
    """Return the HPWL-optimal point for one macro given current others.

    HPWL-optimal-for-one-pin is the median of net-bbox medians; we use the
    weighted average of net-bbox centers across this macro's nets, which
    is a good cheap approximation.
    """
    nets = macro_nets[local_idx]
    if nets.size == 0:
        cw, ch = ctx.canvas
        return 0.5 * cw, 0.5 * ch

    full = _full_centers(centers_movable, ctx)
    move_idx = ctx.move_csr_local
    starts = ctx.move_csr_offsets[:-1]
    ends = ctx.move_csr_offsets[1:]

    sx = sy = sw = 0.0
    for net_id in nets.tolist():
        s, e = int(starts[net_id]), int(ends[net_id])
        # All movable members in this net
        members_local = move_idx[s:e]
        # Use their current centers to compute net bbox (excluding this macro)
        # but it's fine to include it — we recompute the centroid anyway.
        c = full[ctx.movable_idx][members_local]
        # Include fixed members of this net via precomputed bounds
        xmin = c[:, 0].min()
        xmax = c[:, 0].max()
        ymin = c[:, 1].min()
        ymax = c[:, 1].max()
        if np.isfinite(ctx.fixed_xmin[net_id]):
            xmin = min(xmin, ctx.fixed_xmin[net_id])
            xmax = max(xmax, ctx.fixed_xmax[net_id])
            ymin = min(ymin, ctx.fixed_ymin[net_id])
            ymax = max(ymax, ctx.fixed_ymax[net_id])
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        w = float(ctx.net_weights[net_id])
        sx += w * cx
        sy += w * cy
        sw += w
    if sw <= 0:
        cw, ch = ctx.canvas
        return 0.5 * cw, 0.5 * ch
    return sx / sw, sy / sw


def _lns_destroy_repair(
    centers_movable: np.ndarray,
    ctx: SAContext,
    macro_nets: List[np.ndarray],
    rng: np.random.Generator,
    *,
    destroy_frac: float,
    jitter_sigma: float,
) -> np.ndarray:
    """Destroy a random subset of macros and repair them at HPWL centroids.

    Returns a NEW centers_movable array (caller decides whether to keep).
    """
    n_move = centers_movable.shape[0]
    if n_move < 2:
        return centers_movable.copy()
    k = max(1, int(round(destroy_frac * n_move)))
    destroy = rng.choice(n_move, size=k, replace=False)
    new_centers = centers_movable.copy()
    cw, ch = ctx.canvas
    sizes_movable = ctx.sizes[ctx.movable_idx]

    # Repair sequentially: for each destroyed macro, place at HPWL centroid
    # using the *current* placements of all others (including freshly-placed
    # ones from this LNS step).
    for k_local in destroy.tolist():
        cx, cy = _hpwl_centroid_for_macro(k_local, new_centers, ctx, macro_nets)
        cx += rng.normal(0.0, jitter_sigma)
        cy += rng.normal(0.0, jitter_sigma)
        # Clip to canvas keeping macro inside
        hw = 0.5 * sizes_movable[k_local, 0]
        hh = 0.5 * sizes_movable[k_local, 1]
        cx = float(np.clip(cx, hw, cw - hw))
        cy = float(np.clip(cy, hh, ch - hh))
        new_centers[k_local] = (cx, cy)

    return new_centers


# ---------------------------------------------------------------------------
# SA loop
# ---------------------------------------------------------------------------


@dataclass
class MacroSAConfig:
    time_budget_s: float = 240.0
    seed: int = 0

    init_temp: Optional[float] = None
    final_temp_ratio: float = 1e-3

    # Overlap penalty schedule (geometric vs wall-clock).
    lambda_init: float = 0.5
    lambda_final: float = 200.0

    # Move sigma schedule (fraction of canvas, geometric).
    jiggle_sigma_init_frac: float = 0.10
    jiggle_sigma_final_frac: float = 0.005

    # Move probabilities.
    swap_prob: float = 0.10

    # LNS escape.
    lns_period_iters: int = 1500
    lns_destroy_frac: float = 0.10
    lns_jitter_frac: float = 0.02

    # Surrogate.
    density_weight: float = 0.5
    bins: int = 16

    log_every_iters: int = 2000


@dataclass
class MacroSAResult:
    # Lowest surrogate cost ever observed (under the λ at the time it was
    # found; may have small overlaps if found early).
    best_centers_movable: np.ndarray  # (n_move, 2)
    best_full_centers: np.ndarray     # (num_macros, 2)
    best_cost: float
    best_components: dict
    # Lowest surrogate cost among states with zero hard overlap (movable vs
    # movable). May be ``None`` if SA never observed a zero-overlap state.
    best_feasible_centers_movable: Optional[np.ndarray]
    best_feasible_full_centers: Optional[np.ndarray]
    best_feasible_cost: float
    best_feasible_components: Optional[dict]
    # Last accepted state (may have residual overlaps; rerun feasibility
    # check before using).
    final_centers_movable: np.ndarray
    final_full_centers: np.ndarray
    final_cost: float
    final_components: dict
    iters: int
    accepted: int
    rate: float
    history: List[Tuple[int, float, float, float]] = field(default_factory=list)


def _calibrate_temperature(
    centers0: np.ndarray,
    ctx: SAContext,
    config: MacroSAConfig,
    rng: np.random.Generator,
    samples: int = 32,
) -> float:
    """Estimate T0 from typical uphill move magnitude with current σ."""
    cw, ch = ctx.canvas
    sigma = config.jiggle_sigma_init_frac * 0.5 * (cw + ch)
    sizes_movable = ctx.sizes[ctx.movable_idx]
    base_cost, _ = _surrogate_eval(
        centers0,
        ctx,
        density_weight=config.density_weight,
        lam=config.lambda_init,
        bins=config.bins,
    )
    deltas = []
    for _ in range(samples):
        k = int(rng.integers(0, centers0.shape[0]))
        new_c = centers0.copy()
        new_c[k] = centers0[k] + rng.normal(0.0, sigma, size=2)
        # Clamp
        hw = 0.5 * sizes_movable[k, 0]
        hh = 0.5 * sizes_movable[k, 1]
        new_c[k, 0] = float(np.clip(new_c[k, 0], hw, cw - hw))
        new_c[k, 1] = float(np.clip(new_c[k, 1], hh, ch - hh))
        new_cost, _ = _surrogate_eval(
            new_c,
            ctx,
            density_weight=config.density_weight,
            lam=config.lambda_init,
            bins=config.bins,
        )
        d = new_cost - base_cost
        if d > 0:
            deltas.append(d)
    if not deltas:
        return 1.0
    avg = float(np.mean(deltas))
    # P(accept uphill at T0) ~ 0.85
    return max(avg / -math.log(0.85), 1e-6)


def run_macro_sa(
    init_centers_full: np.ndarray,
    ctx: SAContext,
    config: MacroSAConfig = MacroSAConfig(),
    *,
    verbose: bool = False,
) -> MacroSAResult:
    """Run direct macro-position SA from ``init_centers_full``.

    ``init_centers_full`` must be a (num_macros, 2) array with the .plc
    initial centers. Only the rows in ``ctx.movable_idx`` are modified.
    """
    rng = np.random.default_rng(config.seed)
    cw, ch = ctx.canvas
    sigma_init = config.jiggle_sigma_init_frac * 0.5 * (cw + ch)
    sigma_final = config.jiggle_sigma_final_frac * 0.5 * (cw + ch)
    macro_nets = _macro_to_local_nets(ctx)
    sizes_movable = ctx.sizes[ctx.movable_idx]

    centers = init_centers_full[ctx.movable_idx].astype(np.float64).copy()
    # Clamp inside canvas
    centers[:, 0] = np.clip(centers[:, 0], 0.5 * sizes_movable[:, 0], cw - 0.5 * sizes_movable[:, 0])
    centers[:, 1] = np.clip(centers[:, 1], 0.5 * sizes_movable[:, 1], ch - 0.5 * sizes_movable[:, 1])

    cur_cost, cur_comps = _surrogate_eval(
        centers,
        ctx,
        density_weight=config.density_weight,
        lam=config.lambda_init,
        bins=config.bins,
    )
    T0 = config.init_temp
    if T0 is None:
        T0 = _calibrate_temperature(centers, ctx, config, rng)
    T = T0

    best_centers = centers.copy()
    best_cost = cur_cost
    best_comps = dict(cur_comps)
    # Best feasible (zero movable-vs-movable and movable-vs-fixed overlap pairs)
    best_feas_centers: Optional[np.ndarray] = None
    best_feas_cost: float = float("inf")
    best_feas_comps: Optional[dict] = None
    if int(cur_comps.get("overlap_total_pairs", 0)) == 0:
        best_feas_centers = centers.copy()
        best_feas_cost = cur_cost
        best_feas_comps = dict(cur_comps)

    start = time.monotonic()
    budget = max(0.1, config.time_budget_s)
    history: List[Tuple[int, float, float, float]] = [(0, T0, config.lambda_init, best_cost)]
    iters = 0
    accepted = 0
    last_lns_iter = 0

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= budget:
            break
        progress = elapsed / budget
        # Schedules
        T = T0 * (config.final_temp_ratio ** progress)
        sigma = sigma_init * ((sigma_final / sigma_init) ** progress)
        lam = config.lambda_init * ((config.lambda_final / config.lambda_init) ** progress)

        # Choose move
        do_lns = (iters - last_lns_iter) >= config.lns_period_iters
        if do_lns:
            jitter = config.lns_jitter_frac * 0.5 * (cw + ch)
            cand = _lns_destroy_repair(
                centers,
                ctx,
                macro_nets,
                rng,
                destroy_frac=config.lns_destroy_frac,
                jitter_sigma=jitter,
            )
            new_cost, new_comps = _surrogate_eval(
                cand,
                ctx,
                density_weight=config.density_weight,
                lam=lam,
                bins=config.bins,
            )
            d = new_cost - cur_cost
            # LNS is large; use a tempered acceptance with a separate
            # "LNS gate": accept unconditionally if better, else with
            # standard Metropolis.
            if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-12)):
                centers = cand
                cur_cost = new_cost
                cur_comps = new_comps
                accepted += 1
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_centers = centers.copy()
                    best_comps = dict(cur_comps)
                if int(cur_comps.get("overlap_total_pairs", 0)) == 0 and cur_cost < best_feas_cost:
                    best_feas_cost = cur_cost
                    best_feas_centers = centers.copy()
                    best_feas_comps = dict(cur_comps)
            last_lns_iter = iters
            iters += 1
            continue

        # Local move
        if rng.random() < config.swap_prob and centers.shape[0] >= 2:
            a, b = rng.choice(centers.shape[0], size=2, replace=False)
            cand = centers.copy()
            cand[a], cand[b] = centers[b].copy(), centers[a].copy()
        else:
            k = int(rng.integers(0, centers.shape[0]))
            cand = centers.copy()
            cand[k] = centers[k] + rng.normal(0.0, sigma, size=2)
            hw = 0.5 * sizes_movable[k, 0]
            hh = 0.5 * sizes_movable[k, 1]
            cand[k, 0] = float(np.clip(cand[k, 0], hw, cw - hw))
            cand[k, 1] = float(np.clip(cand[k, 1], hh, ch - hh))

        new_cost, new_comps = _surrogate_eval(
            cand,
            ctx,
            density_weight=config.density_weight,
            lam=lam,
            bins=config.bins,
        )
        d = new_cost - cur_cost
        if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-12)):
            centers = cand
            cur_cost = new_cost
            cur_comps = new_comps
            accepted += 1
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_centers = centers.copy()
                best_comps = dict(cur_comps)
            if int(cur_comps.get("overlap_total_pairs", 0)) == 0 and cur_cost < best_feas_cost:
                best_feas_cost = cur_cost
                best_feas_centers = centers.copy()
                best_feas_comps = dict(cur_comps)
        iters += 1

        if iters % max(1, config.log_every_iters) == 0:
            history.append((iters, T, lam, best_cost))
            if verbose:
                print(
                    f"[macro-sa] iter={iters} T={T:.3g} λ={lam:.2f} "
                    f"cur={cur_cost:.3f} best={best_cost:.3f} "
                    f"WL_n={cur_comps['WL_norm']:.3f} D_n={cur_comps['D_norm']:.3f} "
                    f"ov_pen={cur_comps['overlap_pen']:.3f} "
                    f"pairs={cur_comps['overlap_pairs']}",
                    flush=True,
                )

    history.append((iters, T, lam if iters > 0 else config.lambda_init, best_cost))
    elapsed_total = max(time.monotonic() - start, 1e-9)

    # Re-evaluate "best" under FINAL lambda; with shifting cost, the early
    # best may have hidden tiny overlaps that the late schedule now hates.
    lam_final = config.lambda_final
    best_cost_final, best_comps_final = _surrogate_eval(
        best_centers,
        ctx,
        density_weight=config.density_weight,
        lam=lam_final,
        bins=config.bins,
    )
    final_cost_final, final_comps_final = _surrogate_eval(
        centers,
        ctx,
        density_weight=config.density_weight,
        lam=lam_final,
        bins=config.bins,
    )
    if best_feas_centers is not None:
        feas_full = _full_centers(best_feas_centers, ctx)
        feas_cost_final, feas_comps_final = _surrogate_eval(
            best_feas_centers,
            ctx,
            density_weight=config.density_weight,
            lam=lam_final,
            bins=config.bins,
        )
    else:
        feas_full = None
        feas_cost_final = float("inf")
        feas_comps_final = None

    return MacroSAResult(
        best_centers_movable=best_centers,
        best_full_centers=_full_centers(best_centers, ctx),
        best_cost=best_cost_final,
        best_components=best_comps_final,
        best_feasible_centers_movable=best_feas_centers,
        best_feasible_full_centers=feas_full,
        best_feasible_cost=feas_cost_final,
        best_feasible_components=feas_comps_final,
        final_centers_movable=centers,
        final_full_centers=_full_centers(centers, ctx),
        final_cost=final_cost_final,
        final_components=final_comps_final,
        iters=iters,
        accepted=accepted,
        rate=iters / elapsed_total,
        history=history,
    )
