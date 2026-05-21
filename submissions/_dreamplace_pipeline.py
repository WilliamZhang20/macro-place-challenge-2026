"""Multi-start DREAMPlace + true-proxy selection (feature-aware caps, no benchmark names)."""

from __future__ import annotations

import os
import sys
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# Switch torch.multiprocessing from the default 'file_descriptor'
# strategy (one fd per shared tensor — accumulated 51k+ fds on the
# 2026-05-20 slurm sweep, hitting cgroup cap 51200 and crashing GWTW
# SA with OSError(24)) to 'file_system' which uses /dev/shm filenames
# instead of fds.  Must be set BEFORE any tensor is shared.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _benchmark_features import benchmark_features  # noqa: E402
from _candidate_select import (  # noqa: E402
    SelectionResult,
    score_placement,
    select_best_true_proxy_candidates_only,
)
from _dreamplace_cpu_smoke import (  # noqa: E402
    default_dreamplace_install,
    deep_merge_dreamplace_json,
    dreamplace_install_ok,
    run_dreamplace_placement,
)
from _coord_descent import coord_descent_polish  # noqa: E402
from _hard_legalizer import legalize_hard, legalize_hard_spiral  # noqa: E402
from _plc_lookup import PlcLookup  # noqa: E402
from _tilos_gwtw_sa import tilos_gwtw_sa_refine  # noqa: E402


def _tuner_progress_enabled() -> bool:
    return os.environ.get("MACRO_PLACE_TUNER_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _clamp_centers(placement: torch.Tensor, benchmark: Benchmark) -> None:
    n = benchmark.num_hard_macros
    if n <= 0:
        return
    movable = ~benchmark.macro_fixed[:n]
    hw = benchmark.macro_sizes[:n, 0] * 0.5
    hh = benchmark.macro_sizes[:n, 1] * 0.5
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    gap = 1e-3
    if bool(movable.any()):
        placement[:n, 0] = torch.where(
            movable,
            torch.clamp(placement[:n, 0], hw + gap, cw - hw - gap),
            placement[:n, 0],
        )
        placement[:n, 1] = torch.where(
            movable,
            torch.clamp(placement[:n, 1], hh + gap, ch - hh - gap),
            placement[:n, 1],
        )
    if benchmark.macro_fixed.any():
        placement[benchmark.macro_fixed] = benchmark.macro_positions[
            benchmark.macro_fixed
        ].to(placement.dtype)


def _clamp_all_macro_centers(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    gap: float = 0.0,
    hard_only: bool = False,
) -> None:
    """Clamp macro centers to legal canvas bounds without moving fixed macros."""

    with torch.no_grad():
        count = int(benchmark.num_hard_macros) if hard_only else int(benchmark.num_macros)
        if count <= 0:
            return
        sizes = benchmark.macro_sizes[:count].to(placement.dtype)
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        cw = torch.tensor(float(benchmark.canvas_width), dtype=placement.dtype, device=placement.device)
        ch = torch.tensor(float(benchmark.canvas_height), dtype=placement.dtype, device=placement.device)
        gap_t = torch.tensor(float(gap), dtype=placement.dtype, device=placement.device)

        lo_x = half_w + gap_t
        hi_x = cw - half_w - gap_t
        lo_y = half_h + gap_t
        hi_y = ch - half_h - gap_t

        cur_x = placement[:count, 0]
        cur_y = placement[:count, 1]
        mid_x = torch.full_like(cur_x, float(benchmark.canvas_width) * 0.5)
        mid_y = torch.full_like(cur_y, float(benchmark.canvas_height) * 0.5)
        placement[:count, 0] = torch.where(
            hi_x >= lo_x,
            torch.minimum(torch.maximum(cur_x, lo_x), hi_x),
            mid_x,
        )
        placement[:count, 1] = torch.where(
            hi_y >= lo_y,
            torch.minimum(torch.maximum(cur_y, lo_y), hi_y),
            mid_y,
        )
        if benchmark.macro_fixed.any():
            placement[benchmark.macro_fixed] = benchmark.macro_positions[
                benchmark.macro_fixed
            ].to(placement.dtype)


def _legalized_reference_seed(seed: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    """Legalize the README hand-crafted `.plc` reference for scoring guardrails."""

    ref = seed.clone().float()
    _clamp_all_macro_centers(ref, benchmark, gap=0.0)
    ref = legalize_hard(
        ref,
        benchmark,
        overlap_gap=1e-3,
        legalize_rounds=6000,
        outer_passes=6,
        displacement_budget_frac=None,
        step_fraction=0.45,
    )
    # `legalize_hard` only touches hard macros.  Soft macros can still be
    # outside the canvas in the raw `.plc`, so clamp them after hard legalization.
    if benchmark.num_macros > benchmark.num_hard_macros:
        _clamp_all_macro_centers(ref, benchmark, gap=0.0)
    return ref


def _legalized_generated_start(seed: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    """Cheaper legalization for generated pre-DP start candidates."""

    out = seed.clone().float()
    _clamp_all_macro_centers(out, benchmark, gap=0.0)
    out = legalize_hard_spiral(
        out,
        benchmark,
        overlap_gap=1e-3,
        max_rings=160,
        batch_rings=20,
        repair_rounds=24,
    )
    if benchmark.num_macros > benchmark.num_hard_macros:
        _clamp_all_macro_centers(out, benchmark, gap=0.0)
    return out


# Base DP overrides.  Tight stop_overflow + moderate noise to ensure DP
# converges to a real basin (loosened 0.045 → 0.065 caused undercooked
# placements with 2x congestion on ibm08).  Trimmed noise slightly
# (0.070 → 0.060) since that lever was contributing to OOB failures
# without much basin-exploration benefit.
_AGGRESSIVE_DP_OVERRIDES: Dict[str, Any] = {
    "density_weight": 2.15e-4,
    "gamma": 3.3,
    "gp_noise_ratio": 0.060,
    "stop_overflow": 0.050,
    "global_place_stages": [
        {
            "learning_rate": 0.013,
            "Llambda_density_weight_iteration": 2,
            "Lsub_iteration": 3,
        }
    ],
}


def jitter_hard_centers(
    base: torch.Tensor,
    benchmark: Benchmark,
    *,
    sigma_um: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Gaussian perturbation on movable hard macro centers (µm scale)."""

    out = base.clone()
    n = benchmark.num_hard_macros
    if n <= 0 or sigma_um <= 0:
        return out
    movable = ~benchmark.macro_fixed[:n]
    if not bool(movable.any()):
        return out
    idx = movable.nonzero(as_tuple=False).squeeze(-1)
    noise = torch.randn(
        (idx.numel(), 2),
        device=base.device,
        dtype=base.dtype,
        generator=generator,
    )
    out[idx, :2] = out[idx, :2] + float(sigma_um) * noise
    _clamp_centers(out, benchmark)
    return out


def cap_num_starts(benchmark: Benchmark, requested: int) -> int:
    """Feature-aware cap.  Per-start DP cost is roughly linear in
    `macros × nets`, so the caps use both as a runtime proxy.

    ibm10 (387 macros × 12k nets) was the regression case: under the old
    `nh<450 → cap=50` rule it spent 9952 s — 2.77× the 3600 s/benchmark
    rule cap.  The new mid-tier (`nh>=350` and `nets>=10k`) drops it to
    24 starts so the total fits the budget.
    """

    f = benchmark_features(benchmark)
    nh = int(f["num_hard_macros"])
    nn = int(f.get("num_nets", 0))
    if nh >= 1600:
        cap = 8
    elif nh >= 1000:
        cap = 14
    elif nh >= 700:
        cap = 20
    elif nh >= 450:
        cap = 32
    elif nh >= 350 and nn >= 10000:
        # ibm10 class: enough macros + nets that 50 starts blows the
        # per-benchmark wall budget.
        cap = 24
    else:
        cap = 50
    return max(1, min(int(requested), cap))


def scaled_global_iterations(benchmark: Benchmark, base_iters: int) -> int:
    """Mild feature-based iteration stretch.  Cap at 1.25× base (was 1.55×)
    so the lower base_iters default actually shows up at runtime."""

    f = benchmark_features(benchmark)
    util = float(f["hard_area_utilization"])
    nh = int(f["num_hard_macros"])
    mult = 1.0 + 0.15 * max(0.0, util - 0.46) / 0.10 + 0.10 * max(0, nh - 260) / 300.0
    return int(round(float(base_iters) * min(mult, 1.25)))


def _is_sparse_high_net_case(benchmark: Benchmark) -> bool:
    f = benchmark_features(benchmark)
    return (
        float(f["hard_area_utilization"]) < 0.32
        and int(f["num_nets"]) >= 20000
        and 250 <= int(f["num_hard_macros"]) <= 520
    )


def _movable_hard_indices(benchmark: Benchmark) -> torch.Tensor:
    n = benchmark.num_hard_macros
    if n <= 0:
        return torch.empty(0, dtype=torch.long, device=benchmark.macro_positions.device)
    return (~benchmark.macro_fixed[:n]).nonzero(as_tuple=False).squeeze(-1)


def _normalized_rms_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    benchmark: Benchmark,
    movable_idx: torch.Tensor,
) -> float:
    if movable_idx.numel() == 0:
        return 0.0
    scale = torch.tensor(
        [max(float(benchmark.canvas_width), 1e-6), max(float(benchmark.canvas_height), 1e-6)],
        device=a.device,
        dtype=a.dtype,
    )
    delta = (a[movable_idx, :2] - b[movable_idx, :2]) / scale
    return float(torch.sqrt(torch.mean(delta * delta)).item())


def _transform_seed(
    base: torch.Tensor,
    benchmark: Benchmark,
    mode: str,
    *,
    strength: float = 0.0,
    anchor: Tuple[float, float] | None = None,
) -> torch.Tensor:
    out = base.clone()
    idx = _movable_hard_indices(benchmark)
    if idx.numel() == 0:
        return out
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    x = base[idx, 0]
    y = base[idx, 1]
    if mode == "identity":
        nx, ny = x, y
    elif mode == "mirror_x":
        nx, ny = cw - x, y
    elif mode == "mirror_y":
        nx, ny = x, ch - y
    elif mode == "mirror_xy":
        nx, ny = cw - x, ch - y
    elif mode == "transpose":
        nx, ny = (y / max(ch, 1e-6)) * cw, (x / max(cw, 1e-6)) * ch
    elif mode == "anti_transpose":
        nx, ny = cw - (y / max(ch, 1e-6)) * cw, ch - (x / max(cw, 1e-6)) * ch
    elif mode == "anchor" and anchor is not None:
        ax = float(anchor[0]) * cw
        ay = float(anchor[1]) * ch
        nx = (1.0 - strength) * x + strength * ax
        ny = (1.0 - strength) * y + strength * ay
    else:
        nx, ny = x, y
    out[idx, 0] = nx
    out[idx, 1] = ny
    _clamp_centers(out, benchmark)
    return out


def make_diverse_initial_placements(
    base: torch.Tensor,
    benchmark: Benchmark,
    *,
    num_starts: int,
    jitter_sigma_um: float,
    generator: torch.Generator,
) -> List[Tuple[str, torch.Tensor]]:
    """Build maximin-diverse DREAMPlace handoff seeds from a small candidate pool.

    The pool deliberately mixes global symmetries with edge/corner-biased starts,
    then spends a little time selecting starts whose movable hard macro centers are
    far apart in normalized RMS distance.  This keeps the DP calls from all
    beginning in the same local basin.
    """

    requested = max(1, int(num_starts))
    movable_idx = _movable_hard_indices(benchmark)
    if movable_idx.numel() == 0:
        return [("fixed", base.clone())]

    pool: List[Tuple[str, torch.Tensor]] = []
    sparse_high_net = _is_sparse_high_net_case(benchmark)
    if sparse_high_net:
        # These cases have lots of nets but low hard-macro utilization.  Full
        # mirroring/transposition tends to shred the initial connectivity shape
        # and creates routing congestion, so keep starts diverse but local.
        pool.append(("identity", _transform_seed(base, benchmark, "identity")))
    else:
        for mode in (
            "identity",
            "mirror_x",
            "mirror_y",
            "mirror_xy",
            "transpose",
            "anti_transpose",
        ):
            pool.append((mode, _transform_seed(base, benchmark, mode)))

    anchors = (
        (0.14, 0.14),
        (0.86, 0.14),
        (0.14, 0.86),
        (0.86, 0.86),
        (0.50, 0.12),
        (0.88, 0.50),
        (0.50, 0.88),
        (0.12, 0.50),
    )
    anchor_strengths = (0.10, 0.18, 0.26) if sparse_high_net else (0.28, 0.44)
    for i, anchor in enumerate(anchors):
        for strength in anchor_strengths:
            pool.append(
                (
                    f"anchor{i}_s{int(strength * 100)}",
                    _transform_seed(
                        base,
                        benchmark,
                        "anchor",
                        strength=strength,
                        anchor=anchor,
                    ),
                )
            )

    jitter_scales = (0.35, 0.70, 1.05, 1.40) if sparse_high_net else (0.75, 1.25, 1.75, 2.35)
    for label, placement in list(pool):
        for scale in jitter_scales:
            pool.append(
                (
                    f"{label}_jit{scale:.2f}",
                    jitter_hard_centers(
                        placement,
                        benchmark,
                        sigma_um=float(jitter_sigma_um) * scale,
                        generator=generator,
                    ),
                )
            )

    if requested <= 1:
        return [("initial_plc", base.clone())]

    selected: List[Tuple[str, torch.Tensor]] = []
    native = jitter_hard_centers(
        base,
        benchmark,
        sigma_um=max(1e-6, 0.35 * float(jitter_sigma_um)),
        generator=generator,
    )
    selected.append(("native_jit", native))

    # Reserve the final slot for the exact hand-crafted `.plc` reference so
    # diagnostics include it without shifting the empirically useful early
    # start/variant pairings.
    diverse_budget = requested - 1
    while len(selected) < diverse_budget and pool:
        best_i = 0
        best_score = -1.0
        for i, (_, candidate) in enumerate(pool):
            min_dist = min(
                _normalized_rms_distance(candidate, prev, benchmark, movable_idx)
                for _, prev in selected
            )
            native_dist = _normalized_rms_distance(candidate, base, benchmark, movable_idx)
            native_weight = 0.05 if sparse_high_net else 0.20
            score = min_dist + native_weight * native_dist
            if score > best_score:
                best_i = i
                best_score = score
        selected.append(pool.pop(best_i))

    selected.append(("initial_plc", base.clone()))
    return selected[:requested]


def _merge_preferred_starts(
    base_starts: Sequence[Tuple[str, torch.Tensor]],
    preferred_starts: Sequence[Tuple[str, torch.Tensor]],
    *,
    requested: int,
) -> List[Tuple[str, torch.Tensor]]:
    """Prefer scored valid starts while preserving the exact `.plc` slot."""

    exact_ref = [(label, p) for label, p in base_starts if label == "initial_plc"]
    budget = max(1, int(requested)) - len(exact_ref[:1])
    merged: List[Tuple[str, torch.Tensor]] = []
    seen: set[str] = set()

    def add(label: str, placement: torch.Tensor) -> None:
        if len(merged) >= budget or label in seen:
            return
        seen.add(label)
        merged.append((label, placement))

    for label, placement in preferred_starts:
        add(label, placement)
    for label, placement in base_starts:
        if label == "initial_plc":
            continue
        add(label, placement)
    merged.extend(exact_ref[:1])
    return merged[: max(1, int(requested))]


def _valid_start_surrogate(
    placement: torch.Tensor,
    benchmark: Benchmark,
    movable_idx: torch.Tensor,
) -> float:
    """Cheap HPWL+density score used only to order valid start candidates."""

    pos = placement.detach().cpu().numpy()
    canvas_norm = max(float(benchmark.canvas_width) + float(benchmark.canvas_height), 1e-9)
    hpwl = 0.0
    for nodes in benchmark.net_nodes:
        if nodes.numel() <= 1:
            continue
        idx = nodes.detach().cpu().numpy()
        idx = idx[idx < pos.shape[0]]
        if idx.size <= 1:
            continue
        xs = pos[idx, 0]
        ys = pos[idx, 1]
        hpwl += float(xs.max() - xs.min() + ys.max() - ys.min())
    wl_norm = hpwl / (max(1, int(benchmark.num_nets)) * canvas_norm)

    rows = max(1, int(benchmark.grid_rows))
    cols = max(1, int(benchmark.grid_cols))
    density = np.zeros((rows, cols), dtype=np.float64)
    sizes = benchmark.macro_sizes.detach().cpu().numpy()
    cw = max(float(benchmark.canvas_width), 1e-9)
    ch = max(float(benchmark.canvas_height), 1e-9)
    bin_w = cw / cols
    bin_h = ch / rows
    bin_area = max(bin_w * bin_h, 1e-9)
    for i in range(int(benchmark.num_hard_macros)):
        c = int(np.clip(pos[i, 0] / bin_w, 0, cols - 1))
        r = int(np.clip(pos[i, 1] / bin_h, 0, rows - 1))
        density[r, c] += float(sizes[i, 0] * sizes[i, 1]) / bin_area
    top_k = max(1, density.size // 10)
    top_density = float(np.mean(np.sort(density.ravel())[-top_k:]))
    spread = float(np.std(density))

    center_penalty = 0.0
    if movable_idx.numel() > 0:
        xy = placement[movable_idx, :2].detach().cpu().numpy()
        center = np.array([[0.5 * cw, 0.5 * ch]], dtype=np.float64)
        scale = np.array([[cw, ch]], dtype=np.float64)
        center_penalty = float(np.mean(np.linalg.norm((xy - center) / scale, axis=1)))

    return wl_norm + 0.020 * top_density + 0.010 * spread + 0.010 * center_penalty


def _select_diverse_valid_starts(
    entries: Sequence[Tuple[float, str, torch.Tensor]],
    benchmark: Benchmark,
    movable_idx: torch.Tensor,
    *,
    wanted: int,
) -> List[Tuple[float, str, torch.Tensor]]:
    """Maximin diverse selection with a light surrogate tie-break."""

    if wanted <= 0:
        return []
    if len(entries) <= wanted:
        return list(entries)

    sorted_entries = sorted(entries, key=lambda x: x[0])
    selected: List[Tuple[float, str, torch.Tensor]] = [sorted_entries[0]]
    remaining = sorted_entries[1:]
    scale = torch.tensor(
        [max(float(benchmark.canvas_width), 1e-6), max(float(benchmark.canvas_height), 1e-6)],
        dtype=selected[0][2].dtype,
        device=selected[0][2].device,
    )
    s_vals = np.asarray([e[0] for e in sorted_entries], dtype=np.float64)
    s_min = float(s_vals.min())
    s_span = float(max(s_vals.max() - s_min, 1e-9))

    def dist(a: torch.Tensor, b: torch.Tensor) -> float:
        if movable_idx.numel() == 0:
            return 0.0
        d = (a[movable_idx, :2] - b[movable_idx, :2]) / scale
        return float(torch.sqrt(torch.mean(d * d)).item())

    while remaining and len(selected) < wanted:
        best_i = 0
        best_score = -float("inf")
        for i, entry in enumerate(remaining):
            surrogate, _, placement = entry
            min_dist = min(dist(placement, prev[2]) for prev in selected)
            surrogate_bonus = 1.0 - (float(surrogate) - s_min) / s_span
            score = min_dist + 0.08 * surrogate_bonus
            if score > best_score:
                best_i = i
                best_score = score
        selected.append(remaining.pop(best_i))
    return selected


def _discover_valid_proxy_starts(
    base: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    requested: int,
    pool_size: int,
    selection_mode: str,
    proxy_eval_limit: int,
    jitter_sigma_um: float,
    generator: torch.Generator,
) -> List[Tuple[str, torch.Tensor]]:
    """Legalize generated starts, rank by true proxy, and optionally SA-refine.

    This is a pre-DREAMPlace discovery stage: it finds valid, real-proxy-scored
    basins before asking DREAMPlace to optimize from them.
    """

    wanted = max(0, int(requested))
    if wanted <= 0:
        return []

    pool_base = _legalized_reference_seed(base, benchmark)
    raw_pool = make_diverse_initial_placements(
        pool_base,
        benchmark,
        num_starts=max(wanted + 1, int(pool_size)),
        jitter_sigma_um=jitter_sigma_um,
        generator=generator,
    )
    scored: List[Tuple[float, str, torch.Tensor]] = []
    seen_labels: set[str] = set()
    movable_idx = _movable_hard_indices(benchmark)

    for label, placement in raw_pool:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        try:
            valid = placement.clone().float()
            _clamp_all_macro_centers(valid, benchmark, gap=0.0)
            ok, _ = validate_placement(valid, benchmark, check_overlaps=True)
            if not ok:
                valid = (
                    _legalized_reference_seed(placement, benchmark)
                    if label == "initial_plc"
                    else _legalized_generated_start(placement, benchmark)
                )
                ok, _ = validate_placement(valid, benchmark, check_overlaps=True)
                if not ok:
                    continue
            if selection_mode == "proxy":
                costs = compute_proxy_cost(valid, benchmark, plc)
                score = float(costs["proxy_cost"])
            else:
                score = _valid_start_surrogate(valid, benchmark, movable_idx)
            scored.append((float(score), label, valid))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0])
    mode = str(selection_mode).strip().lower()
    if mode == "diverse":
        selected = _select_diverse_valid_starts(
            scored,
            benchmark,
            movable_idx,
            wanted=wanted,
        )
    elif mode == "hybrid":
        shortlist_n = min(len(scored), max(wanted, int(proxy_eval_limit)))
        rescored: List[Tuple[float, str, torch.Tensor]] = []
        for _, label, valid in scored[:shortlist_n]:
            try:
                costs = compute_proxy_cost(valid, benchmark, plc)
                rescored.append((float(costs["proxy_cost"]), label, valid))
            except Exception:
                continue
        selected = _select_diverse_valid_starts(
            rescored or scored,
            benchmark,
            movable_idx,
            wanted=wanted,
        )
    else:
        selected = scored[:wanted]

    if _tuner_progress_enabled() and scored:
        preview_src = selected if selected else scored[: min(5, len(scored))]
        preview = ", ".join(f"{label}:{score:.4f}" for score, label, _ in preview_src[:5])
        print(
            f"[tune:dp] {benchmark.name}  pre_dp_valid_start_pool "
            f"mode={mode or 'proxy'} valid={len(scored)}/{len(raw_pool)} selected=[{preview}]",
            file=sys.stderr,
            flush=True,
        )

    selected = list(selected)
    if mode != "diverse":
        selected.sort(key=lambda x: x[0])
    out: List[Tuple[str, torch.Tensor]] = []
    for proxy, label, placement in selected[:wanted]:
        safe_label = label.replace(" ", "_").replace("/", "_")
        tag = "p" if mode in ("proxy", "hybrid") else "s"
        out.append((f"valid_{safe_label}_{tag}{proxy:.4f}", placement))
    return out


def _evaluator_aligned_num_bins(benchmark: Benchmark, *, axis_multiplier: int) -> int:
    """Pick a DP bin count that aligns with the benchmark's evaluator grid.

    The evaluator's density cost is the top-10% of bins on a
    ``grid_rows x grid_cols`` grid (e.g., 38x34 for ibm08), so a DP density
    loss computed on a finer-but-aligned grid will correlate better with the
    true density penalty than the default 64/128 power-of-two choices.
    """

    axis = max(int(benchmark.grid_rows), int(benchmark.grid_cols))
    if axis <= 0:
        return 128
    raw = int(axis) * max(1, int(axis_multiplier))
    # DREAMPlace's FFT prefers power-of-two bins; round up to the closest
    # power of two within the safe range.
    p = 32
    while p < raw and p < 512:
        p *= 2
    return max(64, min(256, p))


def _rich_dp_variant_specs(
    benchmark: Benchmark,
    *,
    target_density: float,
    num_bins: int,
) -> List[Tuple[float, int, str, Dict[str, Any]]]:
    """(target_density, num_bins, label_tag, extra_json) modes from utilization / scale only."""

    f = benchmark_features(benchmark)
    util = float(f["hard_area_utilization"])
    nets = int(f["num_nets"])
    td0 = float(target_density)
    b0 = int(num_bins)
    alt_bins = 64 if b0 >= 96 else 128
    if _is_sparse_high_net_case(benchmark):
        td_wire = max(0.76, min(0.88, td0 + 0.02))
        td_loose = max(0.72, td_wire - 0.04)
        td_tight_sparse = min(0.90, td_wire + 0.04)
        return [
            (
                td_wire,
                b0,
                "wire",
                {
                    "density_weight_scale": 0.34,
                    "gp_noise_ratio": 0.022,
                    "stop_overflow": 0.095,
                    "enable_fillers": 0,
                    "global_place_stages": [
                        {
                            "learning_rate": 0.012,
                            "Llambda_density_weight_iteration": 4,
                            "Lsub_iteration": 3,
                        }
                    ],
                },
            ),
            (
                td_tight_sparse,
                b0,
                "wire_tight",
                {
                    "density_weight_scale": 0.42,
                    "gp_noise_ratio": 0.018,
                    "stop_overflow": 0.085,
                    "enable_fillers": 0,
                    "global_place_stages": [
                        {
                            "learning_rate": 0.012,
                            "optimizer": "yogi",
                            "Llambda_density_weight_iteration": 4,
                            "Lsub_iteration": 3,
                        }
                    ],
                },
            ),
            (
                td_loose,
                alt_bins,
                "wire_loose",
                {
                    "density_weight_scale": 0.52,
                    "gp_noise_ratio": 0.030,
                    "stop_overflow": 0.115,
                    "gamma": 3.2,
                },
            ),
            (
                td_wire,
                64,
                "wire_coarse",
                {
                    "density_weight_scale": 0.46,
                    "gp_noise_ratio": 0.026,
                    "stop_overflow": 0.105,
                    "enable_fillers": 0,
                },
            ),
            (
                min(0.90, td_wire + 0.02),
                128,
                "wire_fine",
                {
                    "density_weight_scale": 0.58,
                    "gp_noise_ratio": 0.020,
                    "stop_overflow": 0.095,
                    "gamma": 3.0,
                },
            ),
            (
                max(0.72, td_wire - 0.02),
                b0,
                "mild_spread",
                {
                    "density_weight_scale": 0.78,
                    "gp_noise_ratio": 0.038,
                    "stop_overflow": 0.110,
                },
            ),
            (
                min(0.91, td_wire + 0.06),
                b0,
                "dense_wire",
                {
                    "density_weight_scale": 0.30,
                    "gp_noise_ratio": 0.016,
                    "stop_overflow": 0.080,
                    "enable_fillers": 0,
                },
            ),
            (
                max(0.70, td_wire - 0.06),
                alt_bins,
                "soft_relax",
                {
                    "density_weight_scale": 0.64,
                    "gp_noise_ratio": 0.034,
                    "stop_overflow": 0.125,
                    "gamma": 3.4,
                },
            ),
        ]
    # High utilization: encourage spreading (lower target density).
    td_spread = max(0.64, min(0.90, td0 - 0.06 * max(0.0, (util - 0.46) / 0.12)))
    # Low utilization: allow slightly tighter packing.
    td_tight = max(0.64, min(0.90, td0 + 0.05 * max(0.0, (0.50 - util) / 0.10)))
    aligned_bins_fine = _evaluator_aligned_num_bins(benchmark, axis_multiplier=4)
    aligned_bins_coarse = _evaluator_aligned_num_bins(benchmark, axis_multiplier=2)
    # Orthogonal multistart: ~30 DP variants spanning (target_density,
    # bins, density_weight_scale, gp_noise_ratio, stop_overflow, gamma).
    # Goal is per-start orthogonality — each variant tries a meaningfully
    # different DP hyperparam corner so the top-K filter has real basin
    # diversity to choose from.  The first 8 are the proven empirical
    # winners from the rich-variant set; the remainder expand the
    # exploration grid.
    specs: List[Tuple[float, int, str, Dict[str, Any]]] = [
        # --- empirical-winner core (proven validity envelope) ---
        (max(0.60, min(0.86, td0)), b0, "base", {}),
        (
            max(0.60, td_spread - 0.015),
            alt_bins,
            "spread",
            {"density_weight_scale": 1.30, "stop_overflow": 0.065},
        ),
        (td_tight, b0, "tight", {"density_weight_scale": 1.05, "stop_overflow": 0.070}),
        (
            max(0.58, td_spread - 0.035),
            alt_bins,
            "xspread",
            {"density_weight_scale": 1.30, "gp_noise_ratio": 0.045, "stop_overflow": 0.075},
        ),
        (
            min(0.88, td_tight + 0.015),
            b0,
            "xtight",
            {"density_weight_scale": 0.90, "gamma": 2.7, "stop_overflow": 0.080},
        ),
        (
            max(0.62, td0 - 0.020),
            aligned_bins_coarse,
            "aligned_coarse",
            {"density_weight_scale": 1.20, "gp_noise_ratio": 0.045, "stop_overflow": 0.070},
        ),
        (
            max(0.60, min(0.86, td0 - 0.010)),
            aligned_bins_fine,
            "aligned_fine",
            {"density_weight_scale": 1.10, "gamma": 3.0, "stop_overflow": 0.075},
        ),
        (
            max(0.62, td0 - 0.040),
            alt_bins,
            "salvaged_explore",
            {"density_weight_scale": 1.40, "gp_noise_ratio": 0.060, "stop_overflow": 0.070},
        ),
        # --- orthogonal density / gamma exploration (low-mid utilization) ---
        (0.65, 128, "td065_g25", {"density_weight_scale": 0.95, "gamma": 2.5, "stop_overflow": 0.075}),
        (0.68, 64, "td068_b64", {"density_weight_scale": 1.10, "stop_overflow": 0.072}),
        (0.70, 128, "td070_g34", {"density_weight_scale": 1.00, "gamma": 3.4, "stop_overflow": 0.068}),
        (0.72, 256, "td072_b256", {"density_weight_scale": 1.05, "stop_overflow": 0.070}),
        (0.74, 128, "td074_lr_low", {"density_weight_scale": 1.15, "stop_overflow": 0.068,
                                       "global_place_stages": [{"learning_rate": 0.010, "Llambda_density_weight_iteration": 2, "Lsub_iteration": 3}]}),
        (0.76, 128, "td076_lr_high", {"density_weight_scale": 1.05, "stop_overflow": 0.070,
                                        "global_place_stages": [{"learning_rate": 0.020, "Llambda_density_weight_iteration": 2, "Lsub_iteration": 3}]}),
        (0.78, 128, "td078_tight", {"density_weight_scale": 0.95, "gamma": 3.0, "stop_overflow": 0.072}),
        (0.80, 256, "td080_b256", {"density_weight_scale": 0.85, "gamma": 3.0, "stop_overflow": 0.075}),
        # --- noise / overflow exploration ---
        (0.72, 128, "noise_low",  {"density_weight_scale": 1.10, "gp_noise_ratio": 0.015, "stop_overflow": 0.060}),
        (0.72, 128, "noise_mid",  {"density_weight_scale": 1.10, "gp_noise_ratio": 0.040, "stop_overflow": 0.060}),
        (0.72, 128, "noise_high", {"density_weight_scale": 1.10, "gp_noise_ratio": 0.080, "stop_overflow": 0.060}),
        # Loosened 'ovf_tight' from 0.040 → 0.055; 0.040 was too aggressive
        # and pushed many variants into the OOB pile.  Still tighter than
        # the default 0.075, so it explores a different convergence basin.
        (0.74, 128, "ovf_tight",  {"density_weight_scale": 1.10, "stop_overflow": 0.055}),
        (0.74, 128, "ovf_loose",  {"density_weight_scale": 1.10, "stop_overflow": 0.100}),
        # --- aggressive spread / aggressive tight (different basins) ---
        # NOTE: trimmed dw_scale (was 1.80 / 0.80 → 1.30 / 0.85) and
        # loosened stop_overflow (>= 0.070) to avoid the OOB pile.  We
        # rely on the WIDE variant grid for orthogonality, not extremity.
        (
            max(0.60, td_spread - 0.040),
            alt_bins,
            "wide_spread",
            {"density_weight_scale": 1.30, "gp_noise_ratio": 0.045, "stop_overflow": 0.072},
        ),
        (
            min(0.88, td_tight + 0.030),
            b0,
            "wide_tight",
            {"density_weight_scale": 0.85, "gamma": 2.7, "stop_overflow": 0.082},
        ),
        # --- bin alignment + density extremes ---
        (0.66, aligned_bins_fine, "aligned_lowD", {"density_weight_scale": 1.20, "stop_overflow": 0.068}),
        (0.82, aligned_bins_coarse, "aligned_highD", {"density_weight_scale": 0.95, "stop_overflow": 0.072}),
        # --- gamma sweeps (gradient softness) ---
        (0.72, 128, "gamma_low",  {"density_weight_scale": 1.10, "gamma": 2.4, "stop_overflow": 0.070}),
        (0.72, 128, "gamma_high", {"density_weight_scale": 1.10, "gamma": 3.6, "stop_overflow": 0.070}),
        # --- two-stage cooling (drawn-out runs) ---
        (
            0.74, 128, "twostage_cool",
            {
                "density_weight_scale": 1.05,
                "stop_overflow": 0.060,
                "global_place_stages": [
                    {"learning_rate": 0.018, "Llambda_density_weight_iteration": 2, "Lsub_iteration": 3},
                    {"learning_rate": 0.008, "Llambda_density_weight_iteration": 3, "Lsub_iteration": 3},
                ],
            },
        ),
        # --- congestion-friendly variants (lower density push) ---
        (0.72, 128, "cong_friendly_mid",  {"density_weight_scale": 0.75, "gp_noise_ratio": 0.030, "stop_overflow": 0.080}),
        (0.78, 128, "cong_friendly_high", {"density_weight_scale": 0.75, "gp_noise_ratio": 0.030, "stop_overflow": 0.080}),
    ]
    # Slight density-objective emphasis on spread mode when utilization is stressed.
    if util >= 0.50:
        dw_scale = 1.0 + 0.2 * min(1.0, (util - 0.50) / 0.06)
        specs[1] = (
            specs[1][0],
            specs[1][1],
            specs[1][2],
            {"density_weight_scale": dw_scale},
        )
    elif nets >= 20000:
        # Net-heavy cases are congestion-sensitive; trim noisiest variants
        # (the last "cong_friendly_high" stays — net-heavy cases benefit
        # from softer density push).
        specs[7] = (
            max(0.64, td0 - 0.030),
            alt_bins,
            "net_escape",
            {"density_weight_scale": 1.20, "gp_noise_ratio": 0.060, "stop_overflow": 0.085},
        )
    return specs


def _apply_density_weight_scale(
    overrides: Dict[str, Any], scale: float
) -> Dict[str, Any]:
    if scale == 1.0:
        return overrides
    out = dict(overrides)
    base_dw = out.get("density_weight")
    if base_dw is not None:
        try:
            out["density_weight"] = float(base_dw) * float(scale)
        except (TypeError, ValueError):
            pass
    else:
        # Default from _dp_json is 8e-5; scale relative to that if user did not set.
        out["density_weight"] = float(8e-5) * float(scale)
    return out


@dataclass(frozen=True)
class _DreamPlaceSpec:
    index: int
    tag: str
    start_tag: str
    init: torch.Tensor
    target_density: float
    num_bins: int
    overrides: Dict[str, Any]


def _successive_halving_resources(
    *,
    min_iterations: int,
    max_iterations: int,
    eta: int,
) -> List[int]:
    max_i = max(1, int(max_iterations))
    cur = max(1, min(int(min_iterations), max_i))
    resources = [cur]
    while resources[-1] < max_i:
        nxt = min(max_i, max(resources[-1] + 1, int(math.ceil(resources[-1] * max(2, eta)))))
        resources.append(nxt)
    return resources


def _log_selection_scores(prefix: str, selection: SelectionResult) -> None:
    for score in sorted(selection.scores, key=lambda s: s.proxy_cost)[:12]:
        violation = f" violation={score.violations[0]!r}" if score.violations else ""
        print(
            f"[tune:dp-score:{prefix}] {score.label} "
            f"valid={int(score.valid)} proxy={score.proxy_cost:.4f} "
            f"wl={score.wirelength:.3f} den={score.density:.3f} "
            f"cong={score.congestion:.3f} overlaps={score.overlaps}{violation}",
            file=sys.stderr,
            flush=True,
        )


@dataclass(frozen=True)
class DreamPlacePipelineResult:
    """``initial_handoff`` is the loader `.plc` placement (returned if DP yields nothing valid)."""

    placement: torch.Tensor
    initial_handoff: torch.Tensor
    selection: Optional[SelectionResult]
    reason: str

    def diagnostics(self, benchmark_name: str | None = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "benchmark": benchmark_name,
            "reason": self.reason,
        }
        if self.selection is not None:
            out["selected_label"] = self.selection.best.label
            out["scores"] = [
                {
                    "label": s.label,
                    "valid": s.valid,
                    "proxy_cost": float(s.proxy_cost),
                    "overlaps": int(s.overlaps),
                }
                for s in self.selection.scores
            ]
        return out


class DreamPlacePipeline:
    """Multi-start DREAMPlace; best true proxy with initial-placement guardrail."""

    def __init__(
        self,
        *,
        plc_lookup: PlcLookup | None = None,
        dreamplace_install: Path | str | None = None,
        num_starts: int = 6,
        # Number of top DP candidates (by true proxy) to seed RePlAce
        # rescue from.  RePlAce is the dominant runtime; running just K
        # configs × top_dp_for_rescue seeds = K × top_dp_for_rescue
        # RePlAce invocations.  Default 5 keeps the rescue stage to
        # ~15 runs at K=3.
        top_dp_for_rescue: int = 5,
        jitter_sigma_um: float = 0.115,
        # Default global iterations: 240.  ibm08 evidence:
        # 122 iters crashed quality (proxy 2.8 vs 1.39 baseline) because
        # DP couldn't converge.  180 is the floor where convergence
        # is reliable; still ~25% cheaper than 240.  Combined with
        # reduced subprocess startup overhead, per-start drops from
        # ~30s to ~15-20s.
        global_iterations: int = 240,
        num_bins: int = 128,
        num_threads: int = 8,
        target_density: float = 0.72,
        timeout_seconds: float = 720.0,
        dreamplace_json_overrides: Optional[Mapping[str, Any]] = None,
        use_gpu: Optional[bool] = None,
        scale_iterations_with_features: bool = False,
        rich_candidate_set: bool = True,
        # Pre-DREAMPlace start discovery.  Defaults use 6 empirical base starts
        # + 44 valid/diverse starts = 50 DP attempts on small IBM cases.
        pre_dp_valid_starts: int = 44,
        pre_dp_valid_pool_size: int = 56,
        pre_dp_valid_selection: str = "diverse",
        pre_dp_proxy_eval_limit: int = 8,
        explicit_legalize_dp_outputs: bool = True,
        # Coordinate-descent polish — TILOS-style k-distance bounded
        # search with mask-based feasibility, run on the rescue winner.
        # Strictly-improving (never regresses).  Macro-count-adaptive
        # k_bound keeps per-benchmark wall time bounded.
        # Coord descent dropped 2026-05-20 — bang-per-minute analysis on
        # ibm01/07/10/14 showed it found 0–1 moves vs GWTW SA's 10+.
        # GWTW SA gets the freed budget.  Set this >0 to re-enable.
        post_rescue_coord_descent_seconds: float = 0.0,
        post_rescue_coord_descent_max_passes: int = 1,
        post_rescue_coord_descent_k_bound: Optional[int] = None,
        post_rescue_coord_descent_cell_search_prob: float = 1.0,
        post_rescue_coord_descent_node_order: str = "descending_size",
        # TILOS-style Go-With-The-Winners (GWTW) SA — multi-worker pool
        # exploring in parallel, periodic sync replaces bottom workers
        # with clones of top winners.  Replicates
        # external/MacroPlacement/CodeElements/SimulatedAnnealingGWTW in
        # Python multiprocessing.  Each worker uses direct
        # compute_proxy_cost evaluation.
        #
        # "Light + aggressive": small wall budget (180s), but t_max
        # high enough that early uphill moves are accepted with
        # meaningful probability.  exp(-0.001/0.005) ≈ 0.82 at t_max,
        # so a 0.1% uphill is mostly accepted early; cools to greedy by
        # end of schedule.  This is real SA behaviour rather than the
        # near-greedy descent of t_max=8e-5.
        post_rescue_gwtw_seconds: float = 360.0,
        post_rescue_gwtw_num_workers: int = 8,
        post_rescue_gwtw_num_iters: int = 120,
        post_rescue_gwtw_syncup_freq: float = 0.20,
        post_rescue_gwtw_top_k: int = 2,
        post_rescue_gwtw_t_max: float = 5e-3,
        post_rescue_gwtw_t_min: float = 5e-6,
        # Bias toward bigger jumps: less mirror (often degrades validity),
        # more shuffle (4-macro permutation — only move that meaningfully
        # changes topology).  Observed on ibm14: 93% acceptance with
        # equal probs meant most proposals were trivially-accepted small
        # moves; rebalancing should produce more basin-escape attempts.
        post_rescue_gwtw_action_probs: Tuple[float, float, float, float, float] = (
            0.20,  # swap
            0.20,  # shift
            0.10,  # mirror (was 0.20)
            0.20,  # move
            0.30,  # shuffle (was 0.20)
        ),
        replace_rescue: bool = True,
        replace_rescue_trigger_proxy: float = 0.0,
        replace_rescue_timeout_seconds: float = 150.0,
        hyperband_enabled: bool = False,
        hyperband_eta: int = 3,
        hyperband_min_iterations: int = 48,
    ):
        self.plc_lookup = plc_lookup or PlcLookup()
        self.dreamplace_install = dreamplace_install
        self.num_starts = int(num_starts)
        self.top_dp_for_rescue = max(1, int(top_dp_for_rescue))
        self.jitter_sigma_um = float(jitter_sigma_um)
        self.global_iterations = int(global_iterations)
        self.num_bins = int(num_bins)
        self.num_threads = int(num_threads)
        self.target_density = float(target_density)
        self.timeout_seconds = float(timeout_seconds)
        overrides = dict(_AGGRESSIVE_DP_OVERRIDES)
        if dreamplace_json_overrides:
            overrides = deep_merge_dreamplace_json(overrides, dict(dreamplace_json_overrides))
        self.dreamplace_json_overrides = overrides
        self.use_gpu = use_gpu
        self.scale_iterations_with_features = bool(scale_iterations_with_features)
        self.rich_candidate_set = bool(rich_candidate_set)
        self.pre_dp_valid_starts = int(pre_dp_valid_starts)
        self.pre_dp_valid_pool_size = int(pre_dp_valid_pool_size)
        self.pre_dp_valid_selection = str(pre_dp_valid_selection)
        self.pre_dp_proxy_eval_limit = int(pre_dp_proxy_eval_limit)
        self.explicit_legalize_dp_outputs = bool(explicit_legalize_dp_outputs)
        self.post_rescue_coord_descent_seconds = float(post_rescue_coord_descent_seconds)
        self.post_rescue_coord_descent_max_passes = int(post_rescue_coord_descent_max_passes)
        self.post_rescue_coord_descent_k_bound = post_rescue_coord_descent_k_bound
        self.post_rescue_coord_descent_cell_search_prob = float(post_rescue_coord_descent_cell_search_prob)
        self.post_rescue_coord_descent_node_order = str(post_rescue_coord_descent_node_order)
        self.post_rescue_gwtw_seconds = float(post_rescue_gwtw_seconds)
        self.post_rescue_gwtw_num_workers = int(post_rescue_gwtw_num_workers)
        self.post_rescue_gwtw_num_iters = int(post_rescue_gwtw_num_iters)
        self.post_rescue_gwtw_syncup_freq = float(post_rescue_gwtw_syncup_freq)
        self.post_rescue_gwtw_top_k = int(post_rescue_gwtw_top_k)
        self.post_rescue_gwtw_t_max = float(post_rescue_gwtw_t_max)
        self.post_rescue_gwtw_t_min = float(post_rescue_gwtw_t_min)
        self.post_rescue_gwtw_action_probs = tuple(post_rescue_gwtw_action_probs)
        self.replace_rescue = bool(replace_rescue)
        self.replace_rescue_trigger_proxy = float(replace_rescue_trigger_proxy)
        self.replace_rescue_timeout_seconds = float(replace_rescue_timeout_seconds)
        # Short DREAMPlace runs are not a monotone proxy for full-budget starts:
        # May 13 hyperband sweeps promoted early winners that regressed badly on
        # ibm07/ibm15.  Keep halving opt-in for experiments, never the default.
        self.hyperband_enabled = bool(hyperband_enabled)
        self.hyperband_eta = max(2, int(hyperband_eta))
        self.hyperband_min_iterations = max(1, int(hyperband_min_iterations))

    @staticmethod
    def _repair_seed(seed: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Best-effort overlap/bounds repair when DREAMPlace cannot run or select."""

        return legalize_hard(
            seed.clone(),
            benchmark,
            legalize_rounds=1200,
            overlap_gap=1e-3,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self.run(benchmark).placement

    def run(self, benchmark: Benchmark) -> DreamPlacePipelineResult:
        # Raise NOFILE soft limit before spawning subprocesses or
        # multiprocessing pools.  On SLURM nodes the soft cap is often
        # 1024, which the 50 DREAMPlace subprocesses + 12 RePlAce
        # subprocesses + GWTW SA Pool(8) sequence can exhaust — observed
        # as `OSError(24, "Too many open files")` killing the GWTW SA
        # stage on ibm10 and nvdla in the 2026-05-20 sweep.
        try:
            import resource as _resource  # noqa: PLC0415
            _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
            _target = min(int(_hard), 65536)
            if _soft < _target:
                _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))
        except Exception:
            pass

        pipeline_t0 = time.monotonic()
        # Per-benchmark wall budget the pipeline aims to respect.  The
        # official cap is 1h = 3600s; we target 3000s with 600s margin so
        # one slow stage (e.g. RePlAce hitting its 240s timeout on 12
        # configs = 2880s worst case) does not blow the cap.
        pipeline_wall_budget_s = float(
            os.environ.get("MACRO_PLACE_PIPELINE_WALL_BUDGET_S", "2700")
        )

        seed = benchmark.macro_positions.clone().float()
        plc = self.plc_lookup.load(benchmark)
        inst = (
            Path(self.dreamplace_install)
            if self.dreamplace_install is not None
            else default_dreamplace_install()
        )

        if plc is None:
            fb = self._repair_seed(seed, benchmark)
            return DreamPlacePipelineResult(
                placement=fb,
                initial_handoff=seed,
                selection=None,
                reason="missing_plc",
            )
        ok, _ = dreamplace_install_ok(inst)
        if not ok:
            fb = self._repair_seed(seed, benchmark)
            return DreamPlacePipelineResult(
                placement=fb,
                initial_handoff=seed,
                selection=None,
                reason="dreamplace_install_missing",
            )

        starts = cap_num_starts(benchmark, self.num_starts)
        iters = (
            scaled_global_iterations(benchmark, self.global_iterations)
            if self.scale_iterations_with_features
            else self.global_iterations
        )

        # GUARDRAIL: the initial `.plc` placement is "hand-crafted, serves
        # as reference" (README).  The raw reference can contain overlaps
        # or OOB macros, so keep the exact raw reference as a DREAMPlace
        # start and score this minimally repaired version as the proxy
        # floor DREAMPlace must beat.
        initial_legalized = _legalized_reference_seed(seed, benchmark)
        candidates: List[torch.Tensor] = [initial_legalized.clone().float()]
        labels: List[str] = ["initial_plc_reference_legalized"]
        gen = torch.Generator(device=seed.device)
        gen.manual_seed(2026 + int(benchmark.num_hard_macros) + int(benchmark.num_nets))

        variant_specs: Sequence[Tuple[float, int, str, Dict[str, Any]]]
        if self.rich_candidate_set:
            variant_specs = _rich_dp_variant_specs(
                benchmark,
                target_density=self.target_density,
                num_bins=self.num_bins,
            )
        else:
            variant_specs = (
                (self.target_density, self.num_bins, "base", {}),
            )

        initial_starts = make_diverse_initial_placements(
            seed,
            benchmark,
            num_starts=starts,
            jitter_sigma_um=self.jitter_sigma_um,
            generator=gen,
        )
        if self.pre_dp_valid_starts > 0:
            # Feature-aware pre-DP cap.  legalize_hard_spiral + true-proxy
            # scoring is O(nh^2 * legalize_rounds) per pool placement.
            # On ibm10 (nh=786) the default 56-placement pool + 44 selected
            # starts was costing >12 minutes alone before DP even began —
            # blowing the per-benchmark wall budget in the 2026-05-20 sweep.
            nh_pre = int(benchmark.num_hard_macros)
            if nh_pre >= 1000:
                _pre_pool = min(self.pre_dp_valid_pool_size, 10)
                _pre_req = min(self.pre_dp_valid_starts, 6)
            elif nh_pre >= 600:
                _pre_pool = min(self.pre_dp_valid_pool_size, 18)
                _pre_req = min(self.pre_dp_valid_starts, 12)
            elif nh_pre >= 400:
                _pre_pool = min(self.pre_dp_valid_pool_size, 30)
                _pre_req = min(self.pre_dp_valid_starts, 22)
            else:
                _pre_pool = self.pre_dp_valid_pool_size
                _pre_req = self.pre_dp_valid_starts
            if _tuner_progress_enabled() and (
                _pre_pool != self.pre_dp_valid_pool_size
                or _pre_req != self.pre_dp_valid_starts
            ):
                print(
                    f"[tune:dp] {benchmark.name}  pre_dp cap  "
                    f"nh={nh_pre}  pool {self.pre_dp_valid_pool_size}->{_pre_pool}  "
                    f"requested {self.pre_dp_valid_starts}->{_pre_req}",
                    file=sys.stderr,
                    flush=True,
                )
            discovered_starts = _discover_valid_proxy_starts(
                seed,
                benchmark,
                plc,
                requested=_pre_req,
                pool_size=_pre_pool,
                selection_mode=self.pre_dp_valid_selection,
                proxy_eval_limit=self.pre_dp_proxy_eval_limit,
                jitter_sigma_um=self.jitter_sigma_um,
                generator=gen,
            )
            if discovered_starts:
                # Preserve the empirically useful start/variant pairings from
                # the base portfolio; discovered starts are extra DP attempts.
                initial_starts = list(initial_starts) + list(discovered_starts)
                starts = len(initial_starts)
                for label, placement in discovered_starts:
                    labels.append(f"pre_dp_{label}")
                    candidates.append(placement)
                if _tuner_progress_enabled():
                    names = ", ".join(label for label, _ in initial_starts)
                    print(
                        f"[tune:dp] {benchmark.name}  pre_dp_start_order "
                        f"{len(initial_starts)}=[{names}]",
                        file=sys.stderr,
                        flush=True,
                    )

        dp_specs: List[_DreamPlaceSpec] = []
        for k, (start_tag, init) in enumerate(initial_starts):
            td_k, bins_k, tag_k, extra_k = variant_specs[k % len(variant_specs)]
            overrides: Dict[str, Any] = (
                dict(self.dreamplace_json_overrides)
                if self.dreamplace_json_overrides
                else {}
            )
            scale = float(extra_k.get("density_weight_scale", 1.0))
            extra_clean = {a: b for a, b in extra_k.items() if a != "density_weight_scale"}
            if extra_clean:
                overrides = deep_merge_dreamplace_json(overrides, extra_clean)
            overrides = _apply_density_weight_scale(overrides, scale)
            overrides["random_seed"] = int(9000 + k * 9973 + benchmark.num_macros)
            dp_specs.append(
                _DreamPlaceSpec(
                    index=k,
                    tag=tag_k,
                    start_tag=start_tag,
                    init=init,
                    target_density=float(td_k),
                    num_bins=int(bins_k),
                    overrides=overrides,
                )
            )

        def run_spec(spec: _DreamPlaceSpec, resource_iters: int, round_idx: int) -> Optional[Tuple[str, torch.Tensor]]:
            label = (
                f"dp_{spec.tag}_{spec.start_tag}_k{spec.index}"
                f"_r{round_idx}_it{int(resource_iters)}_seed{spec.overrides['random_seed']}"
            )
            timeout_k = self.timeout_seconds
            if self.hyperband_enabled and iters > 0:
                frac = max(0.12, min(1.0, float(resource_iters) / float(max(1, iters))))
                timeout_k = max(90.0, min(self.timeout_seconds, self.timeout_seconds * frac * 1.35))
            if _tuner_progress_enabled():
                print(
                    f"[tune:dp] {benchmark.name}  Placer {spec.index + 1}/{starts}  "
                    f"round={round_idx}  iters={resource_iters}  bins={spec.num_bins}  "
                    f"td={spec.target_density:.3f}  tag={spec.tag}  "
                    f"start={spec.start_tag}  timeout={timeout_k:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
            dp_out = run_dreamplace_placement(
                benchmark,
                plc,
                dreamplace_install=inst,
                global_iterations=int(resource_iters),
                num_bins=spec.num_bins,
                num_threads=self.num_threads,
                target_density=spec.target_density,
                timeout_seconds=timeout_k,
                dreamplace_json_overrides=spec.overrides,
                use_gpu=self.use_gpu,
                initial_placement=spec.init,
            )
            if _tuner_progress_enabled():
                print(
                    f"[tune:dp] {benchmark.name}  Placer {spec.index + 1}/{starts}  "
                    f"round={round_idx} finished  placement={'ok' if dp_out is not None else 'None'}",
                    file=sys.stderr,
                    flush=True,
                )
            if dp_out is None:
                return None
            if self.explicit_legalize_dp_outputs:
                dp_out = _legalized_generated_start(dp_out, benchmark)
            return label, dp_out

        use_halving = (
            self.hyperband_enabled
            and len(dp_specs) > 1
            and iters > max(1, self.hyperband_min_iterations)
        )
        if use_halving:
            resources = _successive_halving_resources(
                min_iterations=self.hyperband_min_iterations,
                max_iterations=iters,
                eta=self.hyperband_eta,
            )
            active = list(dp_specs)
            for round_idx, resource_iters in enumerate(resources):
                round_entries: List[Tuple[_DreamPlaceSpec, str, torch.Tensor]] = []
                for spec in active:
                    result = run_spec(spec, resource_iters, round_idx)
                    if result is None:
                        continue
                    label, placement = result
                    labels.append(label)
                    candidates.append(placement)
                    round_entries.append((spec, label, placement))

                if round_idx == len(resources) - 1 or len(active) <= 1:
                    break
                keep = max(1, int(math.ceil(float(len(active)) / float(self.hyperband_eta))))
                promoted: List[_DreamPlaceSpec] = []
                if round_entries:
                    try:
                        round_selection = select_best_true_proxy_candidates_only(
                            [entry[2] for entry in round_entries],
                            benchmark,
                            plc,
                            candidate_labels=[entry[1] for entry in round_entries],
                        )
                        label_to_spec = {entry[1]: entry[0] for entry in round_entries}
                        valid_scores = [s for s in round_selection.scores if s.valid]
                        valid_scores.sort(key=lambda s: s.proxy_cost)
                        promoted = [
                            label_to_spec[s.label]
                            for s in valid_scores[:keep]
                            if s.label in label_to_spec
                        ]
                    except Exception:
                        promoted = []
                if not promoted:
                    promoted = active[:keep]
                active = promoted
                if _tuner_progress_enabled():
                    names = ", ".join(f"k{s.index}:{s.tag}/{s.start_tag}" for s in active)
                    print(
                        f"[tune:dp] {benchmark.name}  hyperband promote "
                        f"{len(active)} after round {round_idx}: {names}",
                        file=sys.stderr,
                        flush=True,
                    )
        else:
            # spec_results lets us identify which initial specs produced
            # in-bounds (potentially valid) placements vs which went OOB so
            # we can salvage with a benchmark-specific safe template.
            spec_results: List[Tuple[_DreamPlaceSpec, str, torch.Tensor]] = []
            # Reserve roughly 35% of the pipeline budget for everything
            # AFTER the initial DP pass: rescue (up to 12 × 240s), salvage
            # reruns, coord descent (240s), GWTW SA (180s), and overhead.
            # That leaves ~65% for DP starts.  Without this guard, ibm10
            # with 387 macros + 12k nets blew 9952s — 2.77× the 3600s cap.
            # DP early-exit at 55% (was 65%) of budget so rescue gets a
            # bigger slice — rescue contributes ~0.1+ proxy on the hard
            # benchmarks, whereas DP runs past start 10 contribute
            # diminishingly.
            dp_stage_budget_s = pipeline_wall_budget_s * 0.55
            for k_spec, spec in enumerate(dp_specs):
                elapsed = time.monotonic() - pipeline_t0
                if elapsed > dp_stage_budget_s and k_spec >= 8:
                    if _tuner_progress_enabled():
                        print(
                            f"[tune:dp] {benchmark.name}  DP stage early-exit "
                            f"at {k_spec}/{len(dp_specs)}: elapsed={elapsed:.0f}s "
                            f"> dp_budget={dp_stage_budget_s:.0f}s "
                            f"(pipeline_budget={pipeline_wall_budget_s:.0f}s)",
                            file=sys.stderr,
                            flush=True,
                        )
                    break
                result = run_spec(spec, iters, 0)
                if result is None:
                    continue
                label, placement = result
                labels.append(label)
                candidates.append(placement)
                spec_results.append((spec, label, placement))

            # Adaptive valid-yield salvage: on benchmarks where DP yield is
            # low (ibm08 had 3/16 valid before this pass), use one of the
            # initial pass's actual valid configs as a "safe template" and
            # rerun every OOB slot with that template + heavier jitter on
            # the slot's original start placement.  Evidence from ibm08:
            # every valid initial run used a conservative variant
            # (tight/xtight) paired with heavy jitter (jit2.35); the
            # combination is what survives gradient placement without
            # flying macros OOB.  Borrowing that combination per-benchmark
            # converts most invalids to valids without benchmark-name
            # tuning.
            valid_initial: List[Tuple[_DreamPlaceSpec, str, torch.Tensor]] = [
                (s, l, p)
                for s, l, p in spec_results
                if validate_placement(p, benchmark, check_overlaps=False)[0]
            ]
            invalid_initial: List[Tuple[_DreamPlaceSpec, str, torch.Tensor]] = [
                (s, l, p)
                for s, l, p in spec_results
                if not validate_placement(p, benchmark, check_overlaps=False)[0]
            ]
            # Salvage budget: only retry enough invalid slots to reach a
            # healthy pool size — no point salvaging when we already have
            # plenty of valid candidates.  Target: max(8, 2 * top_dp_for_rescue)
            # valid candidates total, so we have a buffer above the top-K
            # rescue seeds.  With 50 starts and ~50% valid baseline, this
            # caps salvage at ~8-10 retries instead of trying all ~25.
            target_valid = max(8, 2 * self.top_dp_for_rescue)
            salvage_budget = max(0, target_valid - len(valid_initial))
            if invalid_initial and valid_initial and salvage_budget > 0:
                # Rotate among ALL valid initial specs as safe templates,
                # not just the first one.  Different valid specs landed in
                # different DP basins; rotating templates gives the salvage
                # retries basin diversity (rather than all producing
                # near-identical placements).  Heavy-jitter the slot's
                # original start so the retry doesn't collapse to the
                # template's start either.
                # Cap retry list to salvage_budget invalids.
                invalid_initial = invalid_initial[:salvage_budget]
                safe_templates = [s for s, _, _ in valid_initial]
                if _tuner_progress_enabled():
                    tmpls = ", ".join(
                        f"{s.tag}/{s.start_tag}" for s in safe_templates[:4]
                    )
                    print(
                        f"[tune:dp] {benchmark.name}  salvage start "
                        f"valid={len(valid_initial)} invalid={len(invalid_initial)} "
                        f"templates=[{tmpls}{'...' if len(safe_templates) > 4 else ''}]",
                        file=sys.stderr,
                        flush=True,
                    )
                heavy_scale = 3.0
                for retry_idx, (inv_spec, _, _) in enumerate(invalid_initial):
                    safe_spec = safe_templates[retry_idx % len(safe_templates)]
                    heavy_init = jitter_hard_centers(
                        inv_spec.init.clone(),
                        benchmark,
                        sigma_um=self.jitter_sigma_um * heavy_scale,
                        generator=gen,
                    )
                    retry_overrides = dict(safe_spec.overrides)
                    retry_overrides["random_seed"] = (
                        int(safe_spec.overrides.get("random_seed", 0))
                        ^ (0xC0FFEE + retry_idx * 137 + inv_spec.index * 9001)
                    )
                    retry_spec = _DreamPlaceSpec(
                        index=inv_spec.index,
                        tag=f"{safe_spec.tag}_salv",
                        start_tag=f"{inv_spec.start_tag}_h{heavy_scale:.1f}",
                        init=heavy_init,
                        target_density=safe_spec.target_density,
                        num_bins=safe_spec.num_bins,
                        overrides=retry_overrides,
                    )
                    retry_result = run_spec(retry_spec, iters, 0)
                    if retry_result is None:
                        continue
                    rlabel, rplacement = retry_result
                    labels.append(rlabel)
                    candidates.append(rplacement)
                    if _tuner_progress_enabled():
                        r_ok, _ = validate_placement(
                            rplacement, benchmark, check_overlaps=False
                        )
                        print(
                            f"[tune:dp] {benchmark.name}  salvage k{inv_spec.index} "
                            f"{inv_spec.tag}->{retry_spec.tag} valid={int(r_ok)}",
                            file=sys.stderr,
                            flush=True,
                        )
            elif invalid_initial and not valid_initial:
                # No initial candidate was valid — fall back to a hardcoded
                # conservative default.  This is the "all 16 failed" case
                # which should be rare but needs a path forward.  Use the
                # tight variant config (proven safest in the rich variant
                # set: density_weight_scale=1.05, stop_overflow=0.070).
                fallback_overrides = (
                    dict(self.dreamplace_json_overrides)
                    if self.dreamplace_json_overrides
                    else {}
                )
                fallback_overrides = _apply_density_weight_scale(
                    fallback_overrides, 1.05
                )
                fallback_overrides["stop_overflow"] = 0.075
                fallback_overrides["gp_noise_ratio"] = 0.045
                if _tuner_progress_enabled():
                    print(
                        f"[tune:dp] {benchmark.name}  salvage fallback "
                        f"(no initial valid) invalid={len(invalid_initial)}",
                        file=sys.stderr,
                        flush=True,
                    )
                heavy_scale = 3.0
                for retry_idx, (inv_spec, _, _) in enumerate(invalid_initial):
                    heavy_init = jitter_hard_centers(
                        inv_spec.init.clone(),
                        benchmark,
                        sigma_um=self.jitter_sigma_um * heavy_scale,
                        generator=gen,
                    )
                    retry_overrides = dict(fallback_overrides)
                    retry_overrides["random_seed"] = (
                        0xDEADBEEF ^ (retry_idx * 137 + inv_spec.index * 9001)
                    )
                    retry_spec = _DreamPlaceSpec(
                        index=inv_spec.index,
                        tag=f"safe_fallback",
                        start_tag=f"{inv_spec.start_tag}_h{heavy_scale:.1f}",
                        init=heavy_init,
                        target_density=0.76,
                        num_bins=128,
                        overrides=retry_overrides,
                    )
                    retry_result = run_spec(retry_spec, iters, 0)
                    if retry_result is None:
                        continue
                    rlabel, rplacement = retry_result
                    labels.append(rlabel)
                    candidates.append(rplacement)

        if not candidates:
            fb = self._repair_seed(seed, benchmark)
            return DreamPlacePipelineResult(
                placement=fb,
                initial_handoff=seed,
                selection=None,
                reason="all_dreamplace_starts_failed",
            )

        try:
            preliminary = select_best_true_proxy_candidates_only(
                candidates,
                benchmark,
                plc,
                candidate_labels=labels,
            )
            if _tuner_progress_enabled():
                _log_selection_scores("pre", preliminary)
        except ValueError:
            fb = self._repair_seed(seed, benchmark)
            return DreamPlacePipelineResult(
                placement=fb,
                initial_handoff=seed,
                selection=None,
                reason="no_valid_dreamplace_candidate",
            )
        except Exception:
            fb = self._repair_seed(seed, benchmark)
            return DreamPlacePipelineResult(
                placement=fb,
                initial_handoff=seed,
                selection=None,
                reason="selection_failed",
            )
        selection = preliminary

        if (
            self.replace_rescue
            and selection.best.valid
            and float(selection.best.proxy_cost) >= self.replace_rescue_trigger_proxy
        ):
            try:
                from _replace_pipeline import ReplacePipeline  # noqa: PLC0415
                from _replace_runner import ReplaceConfig  # noqa: PLC0415

                # Wide density sweep used for the always-on initial-seeded
                # rescue.  Density+pcofmax dominates RePlAce's final basin;
                # ibm08 and ibm09 both confirmed that different starts
                # converge to identical final placements when (density,
                # pcofmax) match.  Config diversity is the only thing that
                # matters — seed diversity (the previous top-K DP rescue
                # path) was redundant and wasteful.
                #
                # Low-density configs (0.66, 0.68, 0.70) abort on lower
                # utilization benchmarks like ibm09 ("no more tier to
                # assign") but are critical on packed cases.  High-density
                # configs (0.80, 0.84) won on high-cost benchmarks
                # (ibm14/17/18) in the old high_cost_initial_configs path
                # — included here so the simplification doesn't regress
                # those cases.  adaptive probe-then-commit picks the
                # winning configs per benchmark without burning compute on
                # ones that won't converge.
                # Expanded portfolio: more (density, pcofmax) pairs explore
                # more distinct RePlAce basins.  adaptive_top_k=3 keeps the
                # compute bounded — probes are short, only top 3 commit
                # fully.  Added (0.70,1.08), (0.72,1.20), (0.76,1.03),
                # (0.78,1.03/1.08), (0.80,1.03/1.08), (0.82,1.20), so we
                # cover the (density 0.66-0.84) × (pcofmax 1.03/1.08/1.20)
                # grid for the dense bands that have historically won.
                # Portfolio mixes `-bin 128` (current default) with `-bin 64`
                # and `-pcofmin 0.98` flag variants from the historical
                # _GENERIC_CONFIGS — ibm01 baseline 0.9219 was likely found
                # by one of these and my -bin 128-only version regressed
                # to 0.9321 because it explored fewer basins.
                # Trimmed orthogonal portfolio.  Dropped near-duplicates
                # in the old 30-config sweep (e.g. multiple bin=64 vs
                # bin=128 variants at the same density usually find the
                # same basin) in favor of FEWER configs that target
                # DIFFERENT basins — aggressive density, aggressive
                # pcofmax, very loose overflow, etc.  Goal is that each
                # config produces an orthogonal placement so adaptive
                # selection actually picks the best basin per benchmark
                # rather than averaging over near-duplicates.
                # ORTHOGONAL portfolio: 12 distinct (density, pcofmax,
                # bin, overflow, pcofmin, racnt*) combinations.  Memory
                # confirms basin is determined by (density, pcofmax) — so
                # seed diversity is wasted unless paired with config
                # diversity.  Slots 0-2 are the proven congestion-attack
                # winners; slots 3-7 are historical _GENERIC_CONFIGS
                # winners (incl. the bin=64 paths that found the ibm01
                # 0.9219 baseline); slots 8-11 fill in (density × pcofmax)
                # corners not yet covered.
                ortho_rescue_configs = (
                    # (0) Mid-density congestion-aggressive (proven).
                    ReplaceConfig(
                        density=0.74, pcofmax=1.20,
                        extra_args=("-bin", "128", "-overflow", "0.04", "-pcofmin", "0.90"),
                    ),
                    # (1) High-density very-aggressive (proven).
                    ReplaceConfig(
                        density=0.82, pcofmax=1.50,
                        extra_args=("-bin", "128", "-overflow", "0.05", "-pcofmin", "0.85"),
                    ),
                    # (2) Routability-mode (proven).
                    ReplaceConfig(
                        density=0.72, pcofmax=1.08,
                        extra_args=("-bin", "128", "-overflow", "0.06", "-racnti", "5", "-racnto", "10"),
                    ),
                    # (3) High-density tight-pcofmin (_GENERIC winner).
                    ReplaceConfig(
                        density=0.80, pcofmax=1.03,
                        extra_args=("-bin", "128", "-pcofmin", "0.98"),
                    ),
                    # (4) High-density mid-pcofmax (_GENERIC winner).
                    ReplaceConfig(
                        density=0.80, pcofmax=1.20,
                        extra_args=("-bin", "128",),
                    ),
                    # (5) Very-high-density aggressive (_GENERIC winner).
                    ReplaceConfig(
                        density=0.84, pcofmax=1.20,
                        extra_args=("-bin", "128",),
                    ),
                    # (6) Low-density 64-bin (likely ibm01 0.9219 source).
                    ReplaceConfig(
                        density=0.70, pcofmax=1.03,
                        extra_args=("-bin", "64",),
                    ),
                    # (7) Very-high-density gentle 64-bin (_GENERIC).
                    ReplaceConfig(
                        density=0.84, pcofmax=1.03,
                        extra_args=("-bin", "64",),
                    ),
                    # (8) Mid-density compact (new ortho fill).
                    ReplaceConfig(
                        density=0.76, pcofmax=1.08,
                        extra_args=("-bin", "128", "-overflow", "0.05"),
                    ),
                    # (9) Mid-high aggressive (new ortho fill).
                    ReplaceConfig(
                        density=0.78, pcofmax=1.20,
                        extra_args=("-bin", "128", "-overflow", "0.05", "-pcofmin", "0.92"),
                    ),
                    # (10) Very-high gentle (new ortho fill).
                    ReplaceConfig(
                        density=0.86, pcofmax=1.08,
                        extra_args=("-bin", "128",),
                    ),
                    # (11) Aggressive routability (new ortho fill).
                    ReplaceConfig(
                        density=0.74, pcofmax=1.50,
                        extra_args=("-bin", "128", "-overflow", "0.05", "-racnti", "8", "-racnto", "12"),
                    ),
                )
                # Latin-square assignment: each (seed, config) pair is
                # unique, and densities are matched to seed character so
                # RePlAce does not abort with "no more tier to assign" on
                # the dense .plc seed (observed on ibm01 + ibm07 when
                # configs with `density<=0.74` were assigned to .plc).
                #
                # .plc seed = the raw hand-crafted initial placement, which
                # is already dense. Only HIGH-density configs (>=0.80) are
                # safe here. DP-output seeds are progressively more spread
                # as rank increases (rank 0 = best/densest, rank 4 =
                # worst/most-spread), so low-density configs go on later
                # ranks where they actually converge.
                initial_seed_configs = (
                    ortho_rescue_configs[3],   # (0.80, 1.03, bin=128, pcofmin=0.98)
                    ortho_rescue_configs[4],   # (0.80, 1.20, bin=128)
                    ortho_rescue_configs[5],   # (0.84, 1.20, bin=128)
                )
                # TRIMMED PORTFOLIO (2026-05-20): dropped DP ranks 1, 2, 3.
                # Evidence from per-benchmark winners:
                #   ibm01 → dp_rank4 + config 6 (low-density)
                #   ibm07 → replace_initial (high-density)
                #   ibm10 → dp_rank0 (high-density)
                #   ibm14 → replace_initial (high-density)
                # Ranks 1/2/3 never won outright across the IBM tests; their
                # ~10-12 min of cumulative wall time pushed ibm10 over the
                # 1h cap.  Keep only .plc + densest-DP + most-spread-DP.
                # Each empty tuple is a no-op slot that the rescue loop
                # skips, preserving rank numbering for log readability.
                dp_seed_config_groups: Tuple[Tuple[ReplaceConfig, ...], ...] = (
                    (ortho_rescue_configs[1], ortho_rescue_configs[7]),   # rank 0 (dense DP) → high-density: (0.82,1.50), (0.84,1.03 bin=64)
                    (),                                                    # rank 1 — dropped
                    (),                                                    # rank 2 — dropped
                    (),                                                    # rank 3 — dropped
                    (ortho_rescue_configs[6],),                           # rank 4 (most-spread DP) → low-density (0.70, bin=64)
                )

                # Reuse rescue's internal scores instead of re-scoring all
                # candidates in the outer selection.  On ibm14 (152k nets)
                # the outer re-scoring was the dominant runtime (~30 min
                # for 25 candidates) and was duplicate work: the rescue
                # pipeline already runs select_best_true_proxy_candidates_only
                # on its own outputs.  Tracking the best-so-far against the
                # rescue's already-scored outputs lets us skip the outer
                # full scoring pass entirely.
                rescue_scores: List = []

                import tempfile as _tempfile  # noqa: PLC0415
                import threading as _threading  # noqa: PLC0415
                rescue_scores_lock = _threading.Lock()

                def merge_rescue(
                    rescue_seed: torch.Tensor,
                    rescue_configs: Tuple[ReplaceConfig, ...],
                    prefix: str,
                    *,
                    adaptive_top_k: int = 3,
                    timeout_seconds: float | None = None,
                    work_root: Path | None = None,
                ) -> bool:
                    # Rescue-stage wall-time guard.  Each rescue
                    # invocation can burn up to `timeout * len(configs)`
                    # seconds; on nvdla the 3 failing rescues each
                    # consumed near-full timeout before validation
                    # rejected them, contributing to the 4663s runtime
                    # violation.  If we've already used most of the
                    # pipeline budget, skip remaining rescues so coord
                    # descent + GWTW SA still get to run.
                    elapsed_now = time.monotonic() - pipeline_t0
                    rescue_skip_threshold = pipeline_wall_budget_s * 0.85
                    if elapsed_now > rescue_skip_threshold:
                        if _tuner_progress_enabled():
                            print(
                                f"[tune:dp] {benchmark.name}  replace_rescue skip "
                                f"prefix={prefix}: elapsed={elapsed_now:.0f}s "
                                f"> rescue_skip={rescue_skip_threshold:.0f}s "
                                f"(reserving budget for polish)",
                                file=sys.stderr,
                                flush=True,
                            )
                        return False
                    # Per-call timeout cap: if remaining budget is tight,
                    # shrink the per-config timeout so this rescue can't
                    # singlehandedly blow the budget.
                    remaining = pipeline_wall_budget_s - elapsed_now
                    # Polish reserve.  Coord descent is now disabled (was
                    # 240s); only GWTW SA remains at 360s budget.  Reserve
                    # ~6 min total so rescue can use the rest.
                    polish_reserve = 360.0
                    rescue_budget = max(60.0, remaining - polish_reserve)
                    requested_timeout = (
                        float(timeout_seconds)
                        if timeout_seconds is not None
                        else float(self.replace_rescue_timeout_seconds)
                    )
                    # Cap per-config timeout so worst-case
                    # (configs * timeout) fits the remaining rescue
                    # budget.
                    n_cfg = max(1, len(rescue_configs))
                    eff_timeout = min(requested_timeout, rescue_budget / n_cfg)
                    eff_timeout = max(30.0, eff_timeout)
                    try:
                        if _tuner_progress_enabled():
                            print(
                                f"[tune:dp] {benchmark.name}  replace_rescue start "
                                f"prefix={prefix} configs={len(rescue_configs)} "
                                f"timeout={eff_timeout:.0f}s (req {requested_timeout:.0f}s) "
                                f"adaptive_top_k={adaptive_top_k} "
                                f"elapsed={elapsed_now:.0f}s",
                                file=sys.stderr,
                                flush=True,
                            )
                        rescue = ReplacePipeline(
                            configs=rescue_configs,
                            baseline_provider=lambda _benchmark, _seed=rescue_seed: _seed,
                            plc_lookup=self.plc_lookup,
                            work_root=work_root,
                            timeout_seconds=eff_timeout,
                            adaptive_top_k=adaptive_top_k,
                        ).run(benchmark)
                    except Exception as e:
                        if _tuner_progress_enabled():
                            print(
                                f"[tune:dp] {benchmark.name}  replace_rescue "
                                f"exception prefix={prefix}: {e!r}",
                                file=sys.stderr,
                                flush=True,
                            )
                        return False
                    if rescue.selection is None:
                        if _tuner_progress_enabled():
                            print(
                                f"[tune:dp] {benchmark.name}  replace_rescue "
                                f"no selection prefix={prefix}",
                                file=sys.stderr,
                                flush=True,
                            )
                        return False
                    local_scores = [
                        type(sc)(
                            label=f"{prefix}_{sc.label}",
                            placement=sc.placement,
                            valid=sc.valid,
                            proxy_cost=sc.proxy_cost,
                            wirelength=sc.wirelength,
                            density=sc.density,
                            congestion=sc.congestion,
                            overlaps=sc.overlaps,
                            violations=sc.violations,
                        )
                        for sc in rescue.selection.scores
                    ]
                    with rescue_scores_lock:
                        rescue_scores.extend(local_scores)
                    if _tuner_progress_enabled():
                        valid_scores = [
                            sc for sc in rescue.selection.scores if sc.valid
                        ]
                        best_proxy = (
                            min(sc.proxy_cost for sc in valid_scores)
                            if valid_scores
                            else float("inf")
                        )
                        print(
                            f"[tune:dp] {benchmark.name}  replace_rescue done "
                            f"prefix={prefix} scores={len(rescue.selection.scores)} "
                            f"best={best_proxy:.4f}",
                            file=sys.stderr,
                            flush=True,
                        )
                    return True

                # === New strategy: seed RePlAce from the TOP-K DP
                # outputs (by true proxy), not the original .plc.  DP
                # provides 50 orthogonal starting basins; we run the 3
                # aggressive RePlAce configs from each of the top-K to
                # explore different polished basins.  Total RePlAce
                # invocations = 3 * top_dp_for_rescue (= 15 by default).
                # Also always-include the original .plc seed as a
                # guardrail in case DP outputs are all worse than the
                # initial placement.

                # Rank DP candidates by their pre-SA proxy (already
                # computed in `preliminary.scores`).
                dp_label_to_placement = dict(zip(labels, candidates))
                valid_dp_scores = sorted(
                    [
                        s
                        for s in preliminary.scores
                        if s.valid and s.label.startswith("dp_") and s.label in dp_label_to_placement
                    ],
                    key=lambda s: s.proxy_cost,
                )

                # Build the list of rescue tasks (one per surviving
                # (seed, configs) group).  Each task gets its own
                # work_root subdir so concurrent ReplacePipeline calls
                # don't overwrite each other's Bookshelf export.
                rescue_tasks: List[Tuple[torch.Tensor, Tuple[ReplaceConfig, ...], str, Path]] = []
                _rescue_root_base = Path(_tempfile.gettempdir()) / "macro_place_replace_pipeline"
                rescue_tasks.append((
                    seed.clone().float(),
                    initial_seed_configs,
                    "replace_initial",
                    _rescue_root_base / f"{benchmark.name}__replace_initial",
                ))

                # DP-output-seeded rescues: each rank gets a UNIQUE config
                # subset (Latin-square pairing).
                top_k = min(self.top_dp_for_rescue, len(valid_dp_scores))
                if _tuner_progress_enabled() and top_k > 0:
                    top_proxies = [f"{s.proxy_cost:.4f}" for s in valid_dp_scores[:top_k]]
                    print(
                        f"[tune:dp] {benchmark.name}  rescue DP-seeded top_{top_k} "
                        f"dp_proxies=[{', '.join(top_proxies)}]",
                        file=sys.stderr,
                        flush=True,
                    )
                for rank, dp_score in enumerate(valid_dp_scores[:top_k]):
                    if rank >= len(dp_seed_config_groups):
                        break
                    rank_configs = dp_seed_config_groups[rank]
                    if not rank_configs:
                        continue
                    dp_seed_placement = dp_label_to_placement[dp_score.label].clone().float()
                    prefix = f"replace_dp_rank{rank}"
                    rescue_tasks.append((
                        dp_seed_placement,
                        rank_configs,
                        prefix,
                        _rescue_root_base / f"{benchmark.name}__{prefix}",
                    ))


                # Parallel dispatch.  Each task launches a RePlAce
                # subprocess (CPU-only — no GPU contention with the
                # already-finished DP loop) so we can run them
                # concurrently via a ThreadPoolExecutor.  3 workers by
                # default (matches the 3 surviving rescue groups after
                # the trim).
                import concurrent.futures as _futures  # noqa: PLC0415
                _rescue_workers = max(1, min(
                    len(rescue_tasks),
                    int(os.environ.get("MACRO_PLACE_RESCUE_WORKERS", "3")),
                ))
                changed = False
                if _rescue_workers <= 1 or len(rescue_tasks) == 1:
                    for seed_t, cfgs, pfx, wr in rescue_tasks:
                        if merge_rescue(seed_t, cfgs, pfx, adaptive_top_k=0, work_root=wr):
                            changed = True
                else:
                    if _tuner_progress_enabled():
                        print(
                            f"[tune:dp] {benchmark.name}  rescue parallel "
                            f"workers={_rescue_workers} tasks={len(rescue_tasks)}",
                            file=sys.stderr,
                            flush=True,
                        )
                    with _futures.ThreadPoolExecutor(max_workers=_rescue_workers) as _exec:
                        _futs = [
                            _exec.submit(
                                merge_rescue, seed_t, cfgs, pfx,
                                adaptive_top_k=0, work_root=wr,
                            )
                            for seed_t, cfgs, pfx, wr in rescue_tasks
                        ]
                        for _f in _futures.as_completed(_futs):
                            try:
                                if _f.result():
                                    changed = True
                            except Exception as _e:
                                if _tuner_progress_enabled():
                                    print(
                                        f"[tune:dp] {benchmark.name}  rescue task "
                                        f"raised: {_e!r}",
                                        file=sys.stderr,
                                        flush=True,
                                    )

                if changed and rescue_scores:
                    # Merge cached selection scores with rescue scores; pick
                    # best across both pools.  No re-scoring of placements.
                    all_scores = list(selection.scores) + rescue_scores
                    valid_scores = [s for s in all_scores if s.valid]
                    if valid_scores:
                        best = min(valid_scores, key=lambda s: s.proxy_cost)
                        selection = SelectionResult(best=best, scores=all_scores)
                        if _tuner_progress_enabled():
                            _log_selection_scores("post_replace", selection)
            except Exception as e:
                if _tuner_progress_enabled():
                    print(
                        f"[tune:dp] {benchmark.name}  replace_rescue outer "
                        f"exception: {e!r}",
                        file=sys.stderr,
                        flush=True,
                    )

        # Post-rescue coordinate descent polish.
        # k-distance-bounded search per macro, mask-based feasibility,
        # descending-size node order.  Strictly-improving in true proxy.
        # Runs BEFORE GWTW SA so the SA starts from an already-polished
        # state (CD finds easy improvements cheaply; SA covers harder
        # ones that need uphill moves).
        # Hard wall-time guard before polish stages: if we're already
        # >92% of the per-benchmark budget, skip coord_desc + GWTW so
        # the run finishes inside the 1h rule cap.  Each polish stage
        # alone could otherwise add 240s + 180s = 7 min.
        _polish_skip_threshold = pipeline_wall_budget_s * 0.92
        _polish_skip_reason = None
        if (time.monotonic() - pipeline_t0) > _polish_skip_threshold:
            _polish_skip_reason = (
                f"elapsed={time.monotonic() - pipeline_t0:.0f}s > "
                f"polish_skip={_polish_skip_threshold:.0f}s"
            )
        # Also shrink each polish stage's per-stage budget to fit
        # what's left of the pipeline wall budget.
        _budget_left = max(0.0, pipeline_wall_budget_s - (time.monotonic() - pipeline_t0))
        _cd_budget = min(self.post_rescue_coord_descent_seconds, _budget_left * 0.55)
        _gwtw_budget = min(self.post_rescue_gwtw_seconds, _budget_left * 0.40)

        if (
            self.post_rescue_coord_descent_seconds > 0.0
            and selection.best.valid
            and benchmark.num_hard_macros > 1
            and _polish_skip_reason is None
            and _cd_budget >= 30.0
        ):
            try:
                if _tuner_progress_enabled():
                    print(
                        f"[tune:dp] {benchmark.name}  coord_desc start  "
                        f"budget_s={_cd_budget:.0f} (req={self.post_rescue_coord_descent_seconds:.0f}) "
                        f"passes={self.post_rescue_coord_descent_max_passes} "
                        f"k_bound={self.post_rescue_coord_descent_k_bound} "
                        f"cell_prob={self.post_rescue_coord_descent_cell_search_prob:.2f} "
                        f"order={self.post_rescue_coord_descent_node_order} "
                        f"current_proxy={selection.best.proxy_cost:.4f}",
                        file=sys.stderr,
                        flush=True,
                    )
                cd_best, cd_proxy, cd_acc = coord_descent_polish(
                    selection.placement.clone(),
                    benchmark,
                    plc,
                    time_budget_s=float(_cd_budget),
                    max_passes=int(self.post_rescue_coord_descent_max_passes),
                    k_distance_bound=self.post_rescue_coord_descent_k_bound,
                    cell_search_prob=float(self.post_rescue_coord_descent_cell_search_prob),
                    node_order=self.post_rescue_coord_descent_node_order,
                    seed=(
                        20260521
                        + int(benchmark.num_macros) * 47
                        + int(benchmark.num_nets) * 13
                    ),
                    log_progress=False,
                )
                if not torch.equal(cd_best, selection.placement):
                    cd_score = score_placement(
                        f"coord_desc", cd_best, benchmark, plc
                    )
                    if (
                        cd_score.valid
                        and cd_score.proxy_cost
                        < float(selection.best.proxy_cost) - 1e-9
                    ):
                        if _tuner_progress_enabled():
                            print(
                                f"[tune:dp] {benchmark.name}  coord_desc win  "
                                f"proxy={cd_score.proxy_cost:.4f} "
                                f"(was {selection.best.proxy_cost:.4f}) "
                                f"moves={cd_acc}",
                                file=sys.stderr,
                                flush=True,
                            )
                        new_scores = list(selection.scores) + [cd_score]
                        selection = SelectionResult(
                            best=cd_score, scores=new_scores
                        )
                    elif _tuner_progress_enabled():
                        print(
                            f"[tune:dp] {benchmark.name}  coord_desc no win  "
                            f"moves={cd_acc}",
                            file=sys.stderr,
                            flush=True,
                        )
            except Exception as e:
                if _tuner_progress_enabled():
                    print(
                        f"[tune:dp] {benchmark.name}  coord_desc exception: {e!r}",
                        file=sys.stderr,
                        flush=True,
                    )

        # Post-rescue PRIMARY POLISH (NEW): TILOS Go-With-The-Winners SA.
        # Population of ``num_workers`` SA workers explores in parallel
        # from the rescue winner.  Each worker uses direct
        # compute_proxy_cost evaluation.  Every ``syncup_freq * num_iters``
        # steps the population is sorted by cost and the bottom workers
        # are replaced with clones of the top ``top_k`` winners.  This is
        # the throughput-amplified variant of single-worker SA — with 8
        # workers we get roughly 8x more proposals in the same wall time.
        # Re-evaluate budget left and skip GWTW if we've run out.  CD
        # may have consumed _cd_budget so subtract that.
        _budget_left2 = max(0.0, pipeline_wall_budget_s - (time.monotonic() - pipeline_t0))
        _gwtw_budget = min(self.post_rescue_gwtw_seconds, _budget_left2 * 0.85)
        if (
            self.post_rescue_gwtw_seconds > 0.0
            and self.post_rescue_gwtw_num_workers > 0
            and self.post_rescue_gwtw_num_iters > 0
            and selection.best.valid
            and benchmark.num_hard_macros > 1
            and _gwtw_budget >= 30.0
            and (time.monotonic() - pipeline_t0) <= _polish_skip_threshold
        ):
            try:
                # Net-count-scaled num_iters cap — per-worker proxy cost
                # is O(num_nets), so cap iters on big benchmarks to keep
                # the population SA bounded.
                nn = int(benchmark.num_nets)
                if nn >= 150_000:
                    gwtw_iters = min(self.post_rescue_gwtw_num_iters, 800)
                elif nn >= 80_000:
                    gwtw_iters = min(self.post_rescue_gwtw_num_iters, 1500)
                elif nn >= 40_000:
                    gwtw_iters = min(self.post_rescue_gwtw_num_iters, 2400)
                elif nn >= 15_000:
                    gwtw_iters = min(self.post_rescue_gwtw_num_iters, 3000)
                else:
                    gwtw_iters = int(self.post_rescue_gwtw_num_iters)
                if _tuner_progress_enabled():
                    print(
                        f"[tune:dp] {benchmark.name}  gwtw_sa start  "
                        f"workers={self.post_rescue_gwtw_num_workers} "
                        f"iters={gwtw_iters} sync_freq={self.post_rescue_gwtw_syncup_freq:.2f} "
                        f"top_k={self.post_rescue_gwtw_top_k} "
                        f"budget_s={self.post_rescue_gwtw_seconds:.0f} "
                        f"current_proxy={selection.best.proxy_cost:.4f}",
                        file=sys.stderr,
                        flush=True,
                    )
                gwtw_best, gwtw_best_proxy, gwtw_acc, gwtw_eval = tilos_gwtw_sa_refine(
                    selection.placement.clone(),
                    benchmark,
                    plc,
                    num_workers=int(self.post_rescue_gwtw_num_workers),
                    num_iters=gwtw_iters,
                    syncup_freq=float(self.post_rescue_gwtw_syncup_freq),
                    top_k=int(self.post_rescue_gwtw_top_k),
                    time_budget_s=float(_gwtw_budget),
                    seed=(
                        20260520
                        + int(benchmark.num_macros) * 53
                        + int(benchmark.num_nets) * 11
                    ),
                    t_max=float(self.post_rescue_gwtw_t_max),
                    t_min=float(self.post_rescue_gwtw_t_min),
                    action_probs=self.post_rescue_gwtw_action_probs,
                    log_progress=False,
                )
                if not torch.equal(gwtw_best, selection.placement):
                    gwtw_score = score_placement(
                        f"gwtw_sa", gwtw_best, benchmark, plc
                    )
                    if (
                        gwtw_score.valid
                        and gwtw_score.proxy_cost
                        < float(selection.best.proxy_cost) - 1e-9
                    ):
                        if _tuner_progress_enabled():
                            print(
                                f"[tune:dp] {benchmark.name}  gwtw_sa win  "
                                f"proxy={gwtw_score.proxy_cost:.4f} "
                                f"(was {selection.best.proxy_cost:.4f}) "
                                f"accepted={gwtw_acc}/{gwtw_eval}",
                                file=sys.stderr,
                                flush=True,
                            )
                        new_scores = list(selection.scores) + [gwtw_score]
                        selection = SelectionResult(
                            best=gwtw_score, scores=new_scores
                        )
                    elif _tuner_progress_enabled():
                        print(
                            f"[tune:dp] {benchmark.name}  gwtw_sa no win  "
                            f"accepted={gwtw_acc}/{gwtw_eval}",
                            file=sys.stderr,
                            flush=True,
                        )
            except Exception as e:
                if _tuner_progress_enabled():
                    print(
                        f"[tune:dp] {benchmark.name}  gwtw_sa exception: {e!r}",
                        file=sys.stderr,
                        flush=True,
                    )

        return DreamPlacePipelineResult(
            placement=selection.placement,
            initial_handoff=seed,
            selection=selection,
            reason="ok",
        )
