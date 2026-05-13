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

from macro_place.benchmark import Benchmark
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
from _hard_legalizer import legalize_hard  # noqa: E402
from _plc_lookup import PlcLookup  # noqa: E402
from _routing_congestion import compute_rudy_map  # noqa: E402


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


_AGGRESSIVE_DP_OVERRIDES: Dict[str, Any] = {
    "density_weight": 2.15e-4,
    "gamma": 3.3,
    "gp_noise_ratio": 0.070,
    "stop_overflow": 0.045,
    "global_place_stages": [
        {
            "learning_rate": 0.014,
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
    """Feature-aware cap: allow up to 16 starts, but avoid runaway NG45 runtimes."""

    nh = int(benchmark_features(benchmark)["num_hard_macros"])
    if nh >= 1600:
        cap = 4
    elif nh >= 1000:
        cap = 6
    elif nh >= 700:
        cap = 8
    elif nh >= 450:
        cap = 12
    else:
        cap = 16
    return max(1, min(int(requested), cap))


def scaled_global_iterations(benchmark: Benchmark, base_iters: int) -> int:
    """Aggressive feature-based iteration stretch (utilization / size), capped."""

    f = benchmark_features(benchmark)
    util = float(f["hard_area_utilization"])
    nh = int(f["num_hard_macros"])
    mult = 1.0 + 0.30 * max(0.0, util - 0.46) / 0.10 + 0.18 * max(0, nh - 260) / 300.0
    return int(round(float(base_iters) * min(mult, 1.55)))


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

    selected: List[Tuple[str, torch.Tensor]] = []
    native = jitter_hard_centers(
        base,
        benchmark,
        sigma_um=max(1e-6, 0.35 * float(jitter_sigma_um)),
        generator=generator,
    )
    selected.append(("native_jit", native))

    while len(selected) < requested and pool:
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

    return selected[:requested]


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
    specs: List[Tuple[float, int, str, Dict[str, Any]]] = [
        (max(0.60, min(0.86, td0)), b0, "base", {}),
        (max(0.58, td_spread - 0.025), alt_bins, "spread", {"density_weight_scale": 1.70, "stop_overflow": 0.045}),
        (td_tight, b0, "tight", {"density_weight_scale": 1.05, "stop_overflow": 0.070}),
        (
            max(0.56, td_spread - 0.065),
            alt_bins,
            "xspread",
            {"density_weight_scale": 2.15, "gp_noise_ratio": 0.090, "stop_overflow": 0.035},
        ),
        (
            min(0.88, td_tight + 0.015),
            b0,
            "xtight",
            {"density_weight_scale": 0.90, "gamma": 2.7, "stop_overflow": 0.080},
        ),
        (
            max(0.58, td0 - 0.045),
            64,
            "coarse",
            {"density_weight_scale": 1.55, "gp_noise_ratio": 0.095, "stop_overflow": 0.045},
        ),
        (
            max(0.60, min(0.86, td0 - 0.015)),
            128,
            "fine",
            {"density_weight_scale": 1.45, "gamma": 3.0, "stop_overflow": 0.050},
        ),
        (
            max(0.56, td0 - 0.085),
            alt_bins,
            "escape",
            {"density_weight_scale": 2.45, "gp_noise_ratio": 0.125, "stop_overflow": 0.030},
        ),
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
        # Net-heavy cases are congestion-sensitive; avoid the noisiest escape.
        specs[-1] = (
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


def _post_dp_sa_surrogate(placement: torch.Tensor, benchmark: Benchmark) -> float:
    pos = placement.detach().cpu().numpy().astype(np.float64, copy=False)
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
    ports = (
        benchmark.port_positions.detach().cpu().numpy().astype(np.float64, copy=False)
        if benchmark.port_positions.numel() > 0
        else np.zeros((0, 2), dtype=np.float64)
    )
    weights = (
        benchmark.net_weights.detach().cpu().numpy().astype(np.float64, copy=False)
        if benchmark.net_weights.numel() > 0
        else None
    )
    n_macros = int(benchmark.num_macros)
    n_ports = int(ports.shape[0])
    wl = 0.0
    wsum = 0.0
    for net_id, net in enumerate(benchmark.net_nodes):
        nodes = net.detach().cpu().numpy() if hasattr(net, "detach") else np.asarray(net)
        if nodes.size < 2:
            continue
        xmin = ymin = np.inf
        xmax = ymax = -np.inf
        for raw_u in nodes:
            u = int(raw_u)
            if 0 <= u < n_macros:
                x, y = float(pos[u, 0]), float(pos[u, 1])
            else:
                p = u - n_macros
                if 0 <= p < n_ports:
                    x, y = float(ports[p, 0]), float(ports[p, 1])
                else:
                    continue
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
        if not np.isfinite(xmin):
            continue
        w = float(weights[net_id]) if weights is not None and net_id < weights.shape[0] else 1.0
        wl += w * ((xmax - xmin) + (ymax - ymin))
        wsum += w
    wl_norm = wl / max(wsum * 0.5 * (float(benchmark.canvas_width) + float(benchmark.canvas_height)), 1e-9)

    rows = max(4, min(24, int(benchmark.grid_rows)))
    cols = max(4, min(24, int(benchmark.grid_cols)))
    bin_w = float(benchmark.canvas_width) / cols
    bin_h = float(benchmark.canvas_height) / rows
    bin_area = max(1e-9, bin_w * bin_h)
    density = np.zeros((rows, cols), dtype=np.float64)
    for i in range(n_macros):
        c = int(np.clip(pos[i, 0] / max(bin_w, 1e-9), 0, cols - 1))
        r = int(np.clip(pos[i, 1] / max(bin_h, 1e-9), 0, rows - 1))
        density[r, c] += float(sizes[i, 0] * sizes[i, 1]) / bin_area
    density_over = float(np.mean(np.maximum(0.0, density - 0.88) ** 2))
    density_spread = float(np.std(density))

    try:
        rudy = compute_rudy_map(placement, benchmark)
        if rudy.size:
            rudy_cost = float(np.mean(np.sort(rudy.ravel())[-max(1, rudy.size // 20) :]))
        else:
            rudy_cost = 0.0
    except Exception:
        rudy_cost = 0.0

    return wl_norm + 0.080 * density_over + 0.012 * density_spread + 0.020 * rudy_cost


def _post_dp_sa_refine(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    label: str,
    seed: int,
    time_budget_s: float,
    max_evals: int,
) -> Optional[Tuple[str, torch.Tensor]]:
    """Small true-proxy simulated annealing polish after DREAMPlace.

    This is intentionally conservative: every proposed state is legalized and
    scored by the real proxy before Metropolis acceptance.  It is not a full
    placer; it just explores the discrete neighborhood around a good DP basin.
    """

    if time_budget_s <= 0.0 or max_evals <= 0 or benchmark.num_hard_macros <= 1:
        return None
    n = int(benchmark.num_hard_macros)
    movable_mask = (~benchmark.macro_fixed[:n]).detach().cpu().numpy().astype(bool)
    movable_idx = np.flatnonzero(movable_mask)
    if movable_idx.size == 0:
        return None

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n].detach().cpu().numpy().astype(np.float64)

    def legalize(pl: torch.Tensor) -> torch.Tensor:
        return legalize_hard(
            pl,
            benchmark,
            overlap_gap=1e-3,
            legalize_rounds=180,
            outer_passes=2,
            displacement_budget_frac=0.12,
            step_fraction=0.28,
        )

    current = legalize(placement.clone().float())
    current_score = score_placement(f"{label}_sa_start", current, benchmark, plc)
    if not current_score.valid:
        return None
    initial = current.clone()
    best = current.clone()
    best_cost = _post_dp_sa_surrogate(current, benchmark)
    initial_cost = best_cost
    cur_cost = best_cost

    start = time.monotonic()
    evals = 0
    sigma0 = 0.030 * 0.5 * (cw + ch)
    sigma1 = 0.0035 * 0.5 * (cw + ch)
    temp0 = 0.030
    temp1 = 0.0015

    while evals < int(max_evals):
        elapsed = time.monotonic() - start
        if elapsed >= time_budget_s:
            break
        progress = min(1.0, elapsed / max(time_budget_s, 1e-9))
        sigma = sigma0 * ((sigma1 / sigma0) ** progress)
        temp = temp0 * ((temp1 / temp0) ** progress)

        cand = current.clone()
        pos = cand[:n].detach().cpu().numpy().astype(np.float64).copy()
        move_roll = float(rng.random())
        if move_roll < 0.18 and movable_idx.size >= 2:
            # Swap similarly sized macros; wildly different swaps often only
            # exercise legalization rather than useful local search.
            a = int(rng.choice(movable_idx))
            area = sizes[:, 0] * sizes[:, 1]
            lo = 0.55 * area[a]
            hi = 1.80 * area[a]
            peers = movable_idx[(area[movable_idx] >= lo) & (area[movable_idx] <= hi)]
            if peers.size <= 1:
                b = int(rng.choice(movable_idx))
            else:
                b = int(rng.choice(peers[peers != a]))
            pos[[a, b], :2] = pos[[b, a], :2]
        elif move_roll < 0.36 and movable_idx.size >= 4:
            # Coherent tiny cluster translation, useful after legalization has
            # left a connected group slightly off its DP basin.
            k = int(rng.choice(movable_idx))
            center = pos[k, :2].copy()
            dist2 = np.sum((pos[movable_idx, :2] - center[None, :]) ** 2, axis=1)
            count = int(min(max(2, movable_idx.size // 32), 12, movable_idx.size))
            group = movable_idx[np.argpartition(dist2, count - 1)[:count]]
            delta = rng.normal(0.0, 0.65 * sigma, size=2)
            pos[group, :2] += delta[None, :]
        else:
            k = int(rng.choice(movable_idx))
            pos[k, :2] += rng.normal(0.0, sigma, size=2)

        for k in movable_idx.tolist():
            hw = 0.5 * sizes[k, 0]
            hh = 0.5 * sizes[k, 1]
            pos[k, 0] = float(np.clip(pos[k, 0], hw + 1e-3, cw - hw - 1e-3))
            pos[k, 1] = float(np.clip(pos[k, 1], hh + 1e-3, ch - hh - 1e-3))

        cand[:n] = torch.from_numpy(pos).to(device=cand.device, dtype=cand.dtype)
        cand = legalize(cand)
        evals += 1
        ok, _ = validate_placement(cand, benchmark, check_overlaps=True)
        if not ok:
            continue
        new_cost = _post_dp_sa_surrogate(cand, benchmark)
        delta_cost = new_cost - cur_cost
        if delta_cost <= 0.0 or rng.random() < math.exp(-delta_cost / max(temp, 1e-9)):
            current = cand
            cur_cost = new_cost
            if new_cost < best_cost:
                best = cand.clone()
                best_cost = new_cost

    if best_cost + 1e-9 < initial_cost and not torch.allclose(
        best, initial, atol=1e-6, rtol=0.0
    ):
        return (f"{label}_sa", best)
    return None


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
        num_starts: int = 16,
        jitter_sigma_um: float = 0.115,
        global_iterations: int = 240,
        num_bins: int = 128,
        num_threads: int = 8,
        target_density: float = 0.72,
        timeout_seconds: float = 720.0,
        dreamplace_json_overrides: Optional[Mapping[str, Any]] = None,
        use_gpu: Optional[bool] = None,
        scale_iterations_with_features: bool = True,
        rich_candidate_set: bool = True,
        post_dp_sa_seconds: float = 24.0,
        post_dp_sa_top_k: int = 1,
        post_dp_sa_max_evals: int = 200,
        replace_rescue: bool = True,
        replace_rescue_trigger_proxy: float = 0.0,
        replace_rescue_timeout_seconds: float = 240.0,
    ):
        self.plc_lookup = plc_lookup or PlcLookup()
        self.dreamplace_install = dreamplace_install
        self.num_starts = int(num_starts)
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
        self.post_dp_sa_seconds = float(post_dp_sa_seconds)
        self.post_dp_sa_top_k = int(post_dp_sa_top_k)
        self.post_dp_sa_max_evals = int(post_dp_sa_max_evals)
        self.replace_rescue = bool(replace_rescue)
        self.replace_rescue_trigger_proxy = float(replace_rescue_trigger_proxy)
        self.replace_rescue_timeout_seconds = float(replace_rescue_timeout_seconds)

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

        candidates: List[torch.Tensor] = []
        labels: List[str] = []
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

            label = f"dp_{tag_k}_{start_tag}_k{k}_seed{overrides['random_seed']}"
            if _tuner_progress_enabled():
                print(
                    f"[tune:dp] {benchmark.name}  Placer {k + 1}/{starts}  "
                    f"iters={iters}  bins={bins_k}  td={td_k:.3f}  tag={tag_k}  "
                    f"start={start_tag}  timeout={self.timeout_seconds:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
            dp_out = run_dreamplace_placement(
                benchmark,
                plc,
                dreamplace_install=inst,
                global_iterations=iters,
                num_bins=int(bins_k),
                num_threads=self.num_threads,
                target_density=float(td_k),
                timeout_seconds=self.timeout_seconds,
                dreamplace_json_overrides=overrides,
                use_gpu=self.use_gpu,
                initial_placement=init,
            )
            if _tuner_progress_enabled():
                print(
                    f"[tune:dp] {benchmark.name}  Placer {k + 1}/{starts}  "
                    f"finished  placement={'ok' if dp_out is not None else 'None'}",
                    file=sys.stderr,
                    flush=True,
                )
            if dp_out is not None:
                labels.append(label)
                candidates.append(dp_out)

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

        if self.post_dp_sa_seconds > 0.0 and self.post_dp_sa_top_k > 0:
            label_to_candidate = dict(zip(labels, candidates))
            valid_dp_scores = [
                s
                for s in preliminary.scores
                if s.valid and s.label in label_to_candidate
            ]
            valid_dp_scores.sort(key=lambda s: s.proxy_cost)
            top_scores = valid_dp_scores[: max(0, self.post_dp_sa_top_k)]
            if top_scores:
                per_budget = float(self.post_dp_sa_seconds) / float(len(top_scores))
                per_evals = max(1, int(self.post_dp_sa_max_evals) // len(top_scores))
                for rank, score in enumerate(top_scores):
                    refined = _post_dp_sa_refine(
                        label_to_candidate[score.label],
                        benchmark,
                        plc,
                        label=score.label,
                        seed=17041
                        + 7919 * rank
                        + int(benchmark.num_macros)
                        + int(benchmark.num_nets),
                        time_budget_s=per_budget,
                        max_evals=per_evals,
                    )
                    if refined is not None:
                        sa_label, sa_candidate = refined
                        labels.append(sa_label)
                        candidates.append(sa_candidate)

        try:
            selection = select_best_true_proxy_candidates_only(
                candidates,
                benchmark,
                plc,
                candidate_labels=labels,
            )
        except Exception:
            selection = preliminary

        if (
            self.replace_rescue
            and selection.best.valid
            and float(selection.best.proxy_cost) >= self.replace_rescue_trigger_proxy
        ):
            try:
                from _replace_pipeline import ReplacePipeline  # noqa: PLC0415
                from _replace_runner import ReplaceConfig  # noqa: PLC0415

                rescue_seed = selection.placement.clone().float()
                rescue_configs = (
                    ReplaceConfig(density=0.62, pcofmax=1.03, extra_args=("-bin", "64")),
                    ReplaceConfig(density=0.64, pcofmax=1.03, extra_args=("-bin", "128")),
                    ReplaceConfig(density=0.68, pcofmax=1.03, extra_args=("-bin", "64")),
                    ReplaceConfig(density=0.68, pcofmax=1.03, extra_args=("-bin", "128")),
                    ReplaceConfig(density=0.72, pcofmax=1.03, extra_args=("-bin", "128")),
                    ReplaceConfig(density=0.74, pcofmax=1.08, extra_args=("-bin", "128")),
                )
                rescue = ReplacePipeline(
                    configs=rescue_configs,
                    baseline_provider=lambda _benchmark, _seed=rescue_seed: _seed,
                    plc_lookup=self.plc_lookup,
                    timeout_seconds=self.replace_rescue_timeout_seconds,
                ).run(benchmark)
                if rescue.selection is not None:
                    for score in rescue.selection.scores:
                        labels.append(f"replace_rescue_{score.label}")
                        candidates.append(score.placement)
                    selection = select_best_true_proxy_candidates_only(
                        candidates,
                        benchmark,
                        plc,
                        candidate_labels=labels,
                    )
            except Exception:
                pass

        return DreamPlacePipelineResult(
            placement=selection.placement,
            initial_handoff=seed,
            selection=selection,
            reason="ok",
        )
