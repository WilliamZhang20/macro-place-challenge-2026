"""Hard-macro legalization helpers.

This module contains the deterministic overlap repair that used to live inside
``dreamplace_moreau_placer.py``. It is intentionally independent of global
placement/objective code: callers pass a placement and get a repaired placement.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def legalize_hard(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    overlap_gap: float = 1e-3,
    legalize_rounds: int = 260,
) -> torch.Tensor:
    """Repair hard-macro overlaps with bounded displacement."""
    num_hard = benchmark.num_hard_macros
    if num_hard <= 1:
        return placement

    out = placement.clone()
    pos = out[:num_hard].cpu().numpy().astype(np.float64).copy()
    sizes = benchmark.macro_sizes[:num_hard].cpu().numpy().astype(np.float64)

    half_w = 0.5 * sizes[:, 0]
    half_h = 0.5 * sizes[:, 1]
    movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    gap = float(overlap_gap)

    sep_x = half_w[:, None] + half_w[None, :] + gap
    sep_y = half_h[:, None] + half_h[None, :] + gap
    tri_mask = np.triu(np.ones((num_hard, num_hard), dtype=bool), k=1)

    for i in range(num_hard):
        pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
        pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

    for _ in range(int(legalize_rounds)):
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        ovx = sep_x - np.abs(dx)
        ovy = sep_y - np.abs(dy)
        overlap = (ovx > 0.0) & (ovy > 0.0) & tri_mask
        if not np.any(overlap):
            break

        ii, jj = np.where(overlap)
        ox = ovx[ii, jj]
        oy = ovy[ii, jj]
        dxp = dx[ii, jj]
        dyp = dy[ii, jj]
        choose_x = ox <= oy

        mi = movable[ii]
        mj = movable[jj]
        active = mi | mj
        if not np.any(active):
            break

        ii = ii[active]
        jj = jj[active]
        ox = ox[active]
        oy = oy[active]
        dxp = dxp[active]
        dyp = dyp[active]
        choose_x = choose_x[active]
        mi = mi[active]
        mj = mj[active]

        sx = np.where(dxp >= 0.0, 1.0, -1.0)
        sy = np.where(dyp >= 0.0, 1.0, -1.0)
        px = ox + gap
        py = oy + gap

        both = mi & mj
        only_i = mi & (~mj)
        only_j = (~mi) & mj

        dix = np.zeros_like(px)
        diy = np.zeros_like(py)
        djx = np.zeros_like(px)
        djy = np.zeros_like(py)

        m = both & choose_x
        dix[m] = 0.5 * sx[m] * px[m]
        djx[m] = -0.5 * sx[m] * px[m]
        m = both & ~choose_x
        diy[m] = 0.5 * sy[m] * py[m]
        djy[m] = -0.5 * sy[m] * py[m]
        m = only_i & choose_x
        dix[m] = sx[m] * px[m]
        m = only_i & ~choose_x
        diy[m] = sy[m] * py[m]
        m = only_j & choose_x
        djx[m] = -sx[m] * px[m]
        m = only_j & ~choose_x
        djy[m] = -sy[m] * py[m]

        moves = np.zeros_like(pos)
        np.add.at(moves[:, 0], ii, dix)
        np.add.at(moves[:, 1], ii, diy)
        np.add.at(moves[:, 0], jj, djx)
        np.add.at(moves[:, 1], jj, djy)

        moved_norm = 0.0
        for i in range(num_hard):
            if not movable[i]:
                continue
            dx_i = float(np.clip(moves[i, 0], -0.35 * sizes[i, 0], 0.35 * sizes[i, 0]))
            dy_i = float(np.clip(moves[i, 1], -0.35 * sizes[i, 1], 0.35 * sizes[i, 1]))
            pos[i, 0] = np.clip(pos[i, 0] + dx_i, half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(pos[i, 1] + dy_i, half_h[i], ch - half_h[i])
            moved_norm += abs(dx_i) + abs(dy_i)

        if moved_norm < 1e-8:
            break

    remaining = _collect_overlapping_macros(pos, sizes)
    if remaining:
        for i in remaining:
            if movable[i]:
                _reinsert_one(i, pos, sizes, movable, cw, ch, gap)

    for _ in range(24):
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        ovx = sep_x - np.abs(dx)
        ovy = sep_y - np.abs(dy)
        overlap = (ovx > 0.0) & (ovy > 0.0) & tri_mask
        if not np.any(overlap):
            break
        ii, jj = np.where(overlap)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if not movable[i] and not movable[j]:
                continue
            px = ovx[i, j] + gap
            py = ovy[i, j] + gap
            if px <= py:
                s = 1.0 if dx[i, j] >= 0.0 else -1.0
                if movable[i] and movable[j]:
                    pos[i, 0] = np.clip(pos[i, 0] + 0.5 * s * px, half_w[i], cw - half_w[i])
                    pos[j, 0] = np.clip(pos[j, 0] - 0.5 * s * px, half_w[j], cw - half_w[j])
                elif movable[i]:
                    pos[i, 0] = np.clip(pos[i, 0] + s * px, half_w[i], cw - half_w[i])
                else:
                    pos[j, 0] = np.clip(pos[j, 0] - s * px, half_w[j], cw - half_w[j])
            else:
                s = 1.0 if dy[i, j] >= 0.0 else -1.0
                if movable[i] and movable[j]:
                    pos[i, 1] = np.clip(pos[i, 1] + 0.5 * s * py, half_h[i], ch - half_h[i])
                    pos[j, 1] = np.clip(pos[j, 1] - 0.5 * s * py, half_h[j], ch - half_h[j])
                elif movable[i]:
                    pos[i, 1] = np.clip(pos[i, 1] + s * py, half_h[i], ch - half_h[i])
                else:
                    pos[j, 1] = np.clip(pos[j, 1] - s * py, half_h[j], ch - half_h[j])

    out[:num_hard] = torch.tensor(pos, dtype=out.dtype)
    if benchmark.macro_fixed.any():
        out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    return out


def _collect_overlapping_macros(pos: np.ndarray, sizes: np.ndarray) -> List[int]:
    n = pos.shape[0]
    if n <= 1:
        return []
    hw = 0.5 * sizes[:, 0]
    hh = 0.5 * sizes[:, 1]
    bad = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (
                abs(pos[i, 0] - pos[j, 0]) < hw[i] + hw[j]
                and abs(pos[i, 1] - pos[j, 1]) < hh[i] + hh[j]
            ):
                bad.add(i)
                bad.add(j)
    return sorted(bad)


def _reinsert_one(
    idx: int,
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    gap: float,
) -> None:
    if not movable[idx]:
        return
    w = sizes[idx, 0]
    h = sizes[idx, 1]
    hw = 0.5 * w
    hh = 0.5 * h
    base_x = float(np.clip(pos[idx, 0], hw, canvas_w - hw))
    base_y = float(np.clip(pos[idx, 1], hh, canvas_h - hh))

    def legal(x: float, y: float) -> bool:
        for j in range(pos.shape[0]):
            if j == idx:
                continue
            sep_x = 0.5 * (w + sizes[j, 0]) + gap
            sep_y = 0.5 * (h + sizes[j, 1]) + gap
            if abs(x - pos[j, 0]) < sep_x and abs(y - pos[j, 1]) < sep_y:
                return False
        return True

    if legal(base_x, base_y):
        pos[idx, 0] = base_x
        pos[idx, 1] = base_y
        return

    step = max(0.15 * max(w, h), 0.02)
    best = None
    best_d2 = float("inf")
    for r in range(1, 81):
        samples = max(16, 8 * r)
        radius = r * step
        for s in range(samples):
            theta = 2.0 * np.pi * (s / samples)
            x = float(np.clip(base_x + radius * np.cos(theta), hw, canvas_w - hw))
            y = float(np.clip(base_y + radius * np.sin(theta), hh, canvas_h - hh))
            if not legal(x, y):
                continue
            d2 = (x - base_x) ** 2 + (y - base_y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (x, y)
        if best is not None:
            break

    if best is not None:
        pos[idx, 0], pos[idx, 1] = best
