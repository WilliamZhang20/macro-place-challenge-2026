"""Shared TILOS-style move proposal helpers for GWTW refinement."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _clip_to_canvas(
    x: float,
    y: float,
    half_w: float,
    half_h: float,
    cw: float,
    ch: float,
) -> Tuple[float, float]:
    eps = 1e-3
    return (
        float(min(max(x, half_w + eps), cw - half_w - eps)),
        float(min(max(y, half_h + eps), ch - half_h - eps)),
    )


def _propose_swap(
    placement: torch.Tensor,
    movable_idx: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    sizes: np.ndarray,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Swap two movable macros' (x, y).  Prefer similar sizes to keep
    feasibility — wildly-mismatched swaps almost always shred legalization."""
    if movable_idx.size < 2:
        return placement.clone()
    out = placement.clone()
    a = int(rng.choice(movable_idx))
    area_a = float(sizes[a, 0] * sizes[a, 1])
    lo = 0.55 * area_a
    hi = 1.85 * area_a
    area_movable = sizes[movable_idx, 0] * sizes[movable_idx, 1]
    eligible = movable_idx[(area_movable >= lo) & (area_movable <= hi)]
    eligible = eligible[eligible != a]
    if eligible.size == 0:
        b = int(rng.choice(movable_idx[movable_idx != a]))
    else:
        b = int(rng.choice(eligible))
    pos_a = out[a, :2].clone()
    out[a, 0] = out[b, 0]
    out[a, 1] = out[b, 1]
    out[b, 0] = pos_a[0]
    out[b, 1] = pos_a[1]
    for k in (a, b):
        nx, ny = _clip_to_canvas(
            float(out[k, 0]),
            float(out[k, 1]),
            float(half_w[k]),
            float(half_h[k]),
            cw,
            ch,
        )
        out[k, 0] = nx
        out[k, 1] = ny
    return out


def _propose_shift(
    placement: torch.Tensor,
    movable_idx: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    bin_w: float,
    bin_h: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Shift one movable macro by ±1 evaluator-grid cell in a random cardinal
    direction."""
    out = placement.clone()
    k = int(rng.choice(movable_idx))
    direction = int(rng.integers(0, 4))
    dx, dy = (
        (bin_w, 0.0),
        (-bin_w, 0.0),
        (0.0, bin_h),
        (0.0, -bin_h),
    )[direction]
    new_x = float(out[k, 0]) + dx
    new_y = float(out[k, 1]) + dy
    nx, ny = _clip_to_canvas(new_x, new_y, float(half_w[k]), float(half_h[k]), cw, ch)
    out[k, 0] = nx
    out[k, 1] = ny
    return out


def _propose_mirror(
    placement: torch.Tensor,
    movable_idx: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Mirror one movable macro's position across the canvas centerline(s).

    We have no per-macro orientation field, so TILOS's ``flip`` maps to a
    canvas-position mirror: x → cw - x (vertical mirror, 1/3 prob), y → ch
    - y (horizontal mirror, 1/3 prob), or both (1/3 prob).
    """
    out = placement.clone()
    k = int(rng.choice(movable_idx))
    mode = float(rng.random())
    x = float(out[k, 0])
    y = float(out[k, 1])
    if mode < 1.0 / 3.0:
        new_x, new_y = cw - x, y
    elif mode < 2.0 / 3.0:
        new_x, new_y = x, ch - y
    else:
        new_x, new_y = cw - x, ch - y
    nx, ny = _clip_to_canvas(new_x, new_y, float(half_w[k]), float(half_h[k]), cw, ch)
    out[k, 0] = nx
    out[k, 1] = ny
    return out


def _propose_move(
    placement: torch.Tensor,
    movable_idx: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    grid_col: int,
    grid_row: int,
    bin_w: float,
    bin_h: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Move one movable macro to the centre of a random evaluator-grid cell."""
    out = placement.clone()
    k = int(rng.choice(movable_idx))
    col = int(rng.integers(0, max(1, grid_col)))
    row = int(rng.integers(0, max(1, grid_row)))
    new_x = (col + 0.5) * bin_w
    new_y = (row + 0.5) * bin_h
    nx, ny = _clip_to_canvas(new_x, new_y, float(half_w[k]), float(half_h[k]), cw, ch)
    out[k, 0] = nx
    out[k, 1] = ny
    return out


def _propose_shuffle(
    placement: torch.Tensor,
    movable_idx: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Pick 4 movable macros and randomly permute their positions."""
    out = placement.clone()
    if movable_idx.size < 4:
        # Fall back to a swap if we don't have enough movable macros.
        if movable_idx.size >= 2:
            i, j = rng.choice(movable_idx, size=2, replace=False)
            tmp = out[int(i), :2].clone()
            out[int(i), 0] = out[int(j), 0]
            out[int(i), 1] = out[int(j), 1]
            out[int(j), 0] = tmp[0]
            out[int(j), 1] = tmp[1]
        return out
    picks = rng.choice(movable_idx, size=4, replace=False)
    positions = [
        (float(out[int(k), 0]), float(out[int(k), 1])) for k in picks
    ]
    permutation = list(range(4))
    while True:
        rng.shuffle(permutation)
        if any(p != i for i, p in enumerate(permutation)):
            break
    for src_idx, dst_idx in enumerate(permutation):
        k = int(picks[dst_idx])
        new_x, new_y = positions[src_idx]
        nx, ny = _clip_to_canvas(
            new_x, new_y, float(half_w[k]), float(half_h[k]), cw, ch
        )
        out[k, 0] = nx
        out[k, 1] = ny
    return out
