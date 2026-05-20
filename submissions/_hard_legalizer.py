"""Hard-macro legalization helpers.

This module contains deterministic overlap repair (`legalize_hard`). It is
intentionally independent of global placement/objective code: callers pass a
placement and get a repaired placement.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _build_ring_offsets(max_r: int) -> List[np.ndarray]:
    offsets: List[np.ndarray] = []
    for r in range(1, int(max_r) + 1):
        edge = np.arange(-r, r + 1)
        top = np.column_stack([edge, np.full(len(edge), r)])
        bot = np.column_stack([edge, np.full(len(edge), -r)])
        inner = np.arange(-r + 1, r)
        left = np.column_stack([np.full(len(inner), -r), inner])
        right = np.column_stack([np.full(len(inner), r), inner])
        offsets.append(np.vstack([top, bot, left, right]))
    return offsets


_RING_OFFSETS = _build_ring_offsets(200)


class _SpatialGrid:
    __slots__ = ("cell_size", "grid", "positions", "half_sizes")

    def __init__(self, max_macro_dim: float):
        self.cell_size = max(float(max_macro_dim) * 1.2, 1.0)
        self.grid: dict[tuple[int, int], List[int]] = {}
        self.positions: dict[int, tuple[float, float]] = {}
        self.half_sizes: dict[int, tuple[float, float]] = {}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(x / self.cell_size), int(y / self.cell_size)

    def add(self, idx: int, cx: float, cy: float, half_w: float, half_h: float) -> None:
        self.positions[idx] = (cx, cy)
        self.half_sizes[idx] = (half_w, half_h)
        self.grid.setdefault(self._cell(cx, cy), []).append(idx)

    def check_overlap(
        self,
        idx: int,
        cx: float,
        cy: float,
        half_w: float,
        half_h: float,
        gap: float,
    ) -> bool:
        sx_self = half_w + gap
        sy_self = half_h + gap
        gx_min = int((cx - sx_self - self.cell_size) / self.cell_size)
        gx_max = int((cx + sx_self + self.cell_size) / self.cell_size)
        gy_min = int((cy - sy_self - self.cell_size) / self.cell_size)
        gy_max = int((cy + sy_self + self.cell_size) / self.cell_size)
        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                for j in self.grid.get((gx, gy), ()):
                    if j == idx:
                        continue
                    jx, jy = self.positions[j]
                    jhw, jhh = self.half_sizes[j]
                    if (
                        abs(cx - jx) < half_w + jhw + gap
                        and abs(cy - jy) < half_h + jhh + gap
                    ):
                        return True
        return False

    def nearby_arrays(
        self,
        cx: float,
        cy: float,
        half_w: float,
        half_h: float,
        gap: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        sx_self = half_w + gap
        sy_self = half_h + gap
        gx_min = int((cx - sx_self - self.cell_size) / self.cell_size)
        gx_max = int((cx + sx_self + self.cell_size) / self.cell_size)
        gy_min = int((cy - sy_self - self.cell_size) / self.cell_size)
        gy_max = int((cy + sy_self + self.cell_size) / self.cell_size)
        nearby: List[int] = []
        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                nearby.extend(self.grid.get((gx, gy), ()))
        if not nearby:
            return None
        px = np.array([self.positions[j][0] for j in nearby], dtype=np.float64)
        py = np.array([self.positions[j][1] for j in nearby], dtype=np.float64)
        phw = np.array([self.half_sizes[j][0] for j in nearby], dtype=np.float64)
        phh = np.array([self.half_sizes[j][1] for j in nearby], dtype=np.float64)
        return px, py, phw, phh


def legalize_hard_spiral(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    overlap_gap: float = 1e-3,
    max_rings: int = 160,
    batch_rings: int = 20,
    repair_rounds: int = 30,
) -> torch.Tensor:
    """Fast greedy hard-macro legalizer using spatial-grid spiral search."""

    num_hard = int(benchmark.num_hard_macros)
    if num_hard <= 1:
        return placement.clone()

    out = placement.clone()
    pos = out[:num_hard].detach().cpu().numpy().astype(np.float64).copy()
    sizes = benchmark.macro_sizes[:num_hard].detach().cpu().numpy().astype(np.float64)
    movable = (~benchmark.macro_fixed[:num_hard]).detach().cpu().numpy().astype(bool)
    half_w = 0.5 * sizes[:, 0]
    half_h = 0.5 * sizes[:, 1]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    gap = float(overlap_gap)

    np.clip(pos[:, 0], half_w, cw - half_w, out=pos[:, 0])
    np.clip(pos[:, 1], half_h, ch - half_h, out=pos[:, 1])

    legal = pos.copy()
    order = sorted(range(num_hard), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    grid = _SpatialGrid(float(sizes.max()) if sizes.size else 1.0)
    max_ring = min(int(max_rings), len(_RING_OFFSETS))
    batch = max(1, int(batch_rings))

    for idx in order:
        if not movable[idx]:
            legal[idx] = benchmark.macro_positions[idx].detach().cpu().numpy()
            legal[idx, 0] = np.clip(legal[idx, 0], half_w[idx], cw - half_w[idx])
            legal[idx, 1] = np.clip(legal[idx, 1], half_h[idx], ch - half_h[idx])
            grid.add(idx, legal[idx, 0], legal[idx, 1], half_w[idx], half_h[idx])
            continue
        if not grid.check_overlap(
            idx, legal[idx, 0], legal[idx, 1], half_w[idx], half_h[idx], gap
        ):
            grid.add(idx, legal[idx, 0], legal[idx, 1], half_w[idx], half_h[idx])
            continue

        base_step = max(float(sizes[idx, 0]), float(sizes[idx, 1]), 1e-6) * 0.20
        best = legal[idx].copy()
        best_d = float("inf")
        for r_start in range(0, max_ring, batch):
            r_end = min(r_start + batch, max_ring)
            if r_start >= 80:
                step = base_step * 2.0
            elif r_start >= 40:
                step = base_step * 1.5
            else:
                step = base_step
            offsets = np.vstack(_RING_OFFSETS[r_start:r_end])
            cx = np.clip(pos[idx, 0] + offsets[:, 0] * step, half_w[idx], cw - half_w[idx])
            cy = np.clip(pos[idx, 1] + offsets[:, 1] * step, half_h[idx], ch - half_h[idx])
            nearby = grid.nearby_arrays(
                pos[idx, 0],
                pos[idx, 1],
                half_w[idx] + (r_end + 1) * step,
                half_h[idx] + (r_end + 1) * step,
                gap,
            )
            if nearby is None:
                d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                bi = int(np.argmin(d))
                best = np.array([cx[bi], cy[bi]], dtype=np.float64)
                best_d = float(d[bi])
                break
            px, py, phw, phh = nearby
            overlaps = (
                np.abs(cx[:, None] - px[None, :]) < (half_w[idx] + phw + gap)[None, :]
            ) & (
                np.abs(cy[:, None] - py[None, :]) < (half_h[idx] + phh + gap)[None, :]
            )
            valid = ~overlaps.any(axis=1)
            if valid.any():
                d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                d[~valid] = float("inf")
                bi = int(np.argmin(d))
                if float(d[bi]) < best_d:
                    best = np.array([cx[bi], cy[bi]], dtype=np.float64)
                break
        legal[idx] = best
        grid.add(idx, best[0], best[1], half_w[idx], half_h[idx])

    for _ in range(max(0, int(repair_rounds))):
        dx = legal[:, 0:1] - legal[:, 0:1].T
        dy = legal[:, 1:2] - legal[:, 1:2].T
        min_dx = half_w[:, None] + half_w[None, :] + gap
        min_dy = half_h[:, None] + half_h[None, :] + gap
        overlap = (np.abs(dx) < min_dx) & (np.abs(dy) < min_dy)
        np.fill_diagonal(overlap, False)
        overlap = np.triu(overlap, k=1)
        if not overlap.any():
            break
        ii, jj = np.where(overlap)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if not movable[i] and not movable[j]:
                continue
            dxij = legal[i, 0] - legal[j, 0]
            dyij = legal[i, 1] - legal[j, 1]
            need_x = half_w[i] + half_w[j] + gap - abs(dxij)
            need_y = half_h[i] + half_h[j] + gap - abs(dyij)
            if need_x <= 0.0 or need_y <= 0.0:
                continue
            if movable[i] and movable[j]:
                share_i = share_j = 0.5
            elif movable[i]:
                share_i, share_j = 1.0, 0.0
            else:
                share_i, share_j = 0.0, 1.0
            if need_x <= need_y:
                s = 1.0 if dxij >= 0.0 else -1.0
                if share_i:
                    legal[i, 0] = np.clip(legal[i, 0] + s * need_x * share_i, half_w[i], cw - half_w[i])
                if share_j:
                    legal[j, 0] = np.clip(legal[j, 0] - s * need_x * share_j, half_w[j], cw - half_w[j])
            else:
                s = 1.0 if dyij >= 0.0 else -1.0
                if share_i:
                    legal[i, 1] = np.clip(legal[i, 1] + s * need_y * share_i, half_h[i], ch - half_h[i])
                if share_j:
                    legal[j, 1] = np.clip(legal[j, 1] - s * need_y * share_j, half_h[j], ch - half_h[j])

    out[:num_hard] = torch.tensor(legal, dtype=out.dtype, device=out.device)
    if benchmark.macro_fixed.any():
        out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed].to(out.dtype)
    return out


def legalize_hard(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    overlap_gap: float = 1e-3,
    legalize_rounds: int = 260,
    outer_passes: int = 1,
    displacement_budget_frac: float | None = None,
    step_fraction: float = 0.35,
) -> torch.Tensor:
    """Repair hard-macro overlaps with bounded displacement.

    When ``outer_passes`` > 1 and/or ``displacement_budget_frac`` is set, each
    outer pass clamps cumulative movement (per macro, L∞ from the start of that
    pass) so legalization can iterate without a single pass shattering density.
    """
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
    sf = float(step_fraction)
    if sf <= 0.0:
        raise ValueError("step_fraction must be positive")

    sep_x = half_w[:, None] + half_w[None, :] + gap
    sep_y = half_h[:, None] + half_h[None, :] + gap
    tri_mask = np.triu(np.ones((num_hard, num_hard), dtype=bool), k=1)

    for i in range(num_hard):
        pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
        pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

    n_outer = max(1, int(outer_passes))
    for _outer in range(n_outer):
        pass_start = pos.copy()
        caps = None
        if displacement_budget_frac is not None:
            bf = float(displacement_budget_frac)
            if bf <= 0.0:
                raise ValueError("displacement_budget_frac must be positive when set")
            caps = bf * np.minimum(sizes[:, 0], sizes[:, 1])

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
                dx_i = float(
                    np.clip(moves[i, 0], -sf * sizes[i, 0], sf * sizes[i, 0])
                )
                dy_i = float(
                    np.clip(moves[i, 1], -sf * sizes[i, 1], sf * sizes[i, 1])
                )
                pos[i, 0] = np.clip(pos[i, 0] + dx_i, half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + dy_i, half_h[i], ch - half_h[i])
                moved_norm += abs(dx_i) + abs(dy_i)

            if caps is not None:
                for i in range(num_hard):
                    if not movable[i]:
                        continue
                    delta = pos[i] - pass_start[i]
                    pos[i, 0] = pass_start[i, 0] + float(
                        np.clip(delta[0], -caps[i], caps[i])
                    )
                    pos[i, 1] = pass_start[i, 1] + float(
                        np.clip(delta[1], -caps[i], caps[i])
                    )
                    pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
                    pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

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
                        pos[i, 0] = np.clip(
                            pos[i, 0] + 0.5 * s * px, half_w[i], cw - half_w[i]
                        )
                        pos[j, 0] = np.clip(
                            pos[j, 0] - 0.5 * s * px, half_w[j], cw - half_w[j]
                        )
                    elif movable[i]:
                        pos[i, 0] = np.clip(
                            pos[i, 0] + s * px, half_w[i], cw - half_w[i]
                        )
                    else:
                        pos[j, 0] = np.clip(
                            pos[j, 0] - s * px, half_w[j], cw - half_w[j]
                        )
                else:
                    s = 1.0 if dy[i, j] >= 0.0 else -1.0
                    if movable[i] and movable[j]:
                        pos[i, 1] = np.clip(
                            pos[i, 1] + 0.5 * s * py, half_h[i], ch - half_h[i]
                        )
                        pos[j, 1] = np.clip(
                            pos[j, 1] - 0.5 * s * py, half_h[j], ch - half_h[j]
                        )
                    elif movable[i]:
                        pos[i, 1] = np.clip(
                            pos[i, 1] + s * py, half_h[i], ch - half_h[i]
                        )
                    else:
                        pos[j, 1] = np.clip(
                            pos[j, 1] - s * py, half_h[j], ch - half_h[j]
                        )

        if caps is not None:
            for i in range(num_hard):
                if not movable[i]:
                    continue
                delta = pos[i] - pass_start[i]
                pos[i, 0] = pass_start[i, 0] + float(
                    np.clip(delta[0], -caps[i], caps[i])
                )
                pos[i, 1] = pass_start[i, 1] + float(
                    np.clip(delta[1], -caps[i], caps[i])
                )
                pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

        if not _collect_overlapping_macros(pos, sizes):
            break

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
