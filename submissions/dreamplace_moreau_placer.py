"""
DREAMPlace-style Moreau-HPWL placer with lightweight spreading and force-based legalization

Design:
1) Optimize movable hard macros with accelerated gradient descent on a smooth
   objective formed by the Moreau envelope of exact net HPWL.
2) Add a lightweight smooth spreading term to reduce overlap pressure early.
3) Legalize hard macros with a force-style overlap resolver and local reinsertion,
   preserving the optimized layout as much as possible.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from macro_place.benchmark import Benchmark


class _NetData:
    def __init__(self, var_ids: np.ndarray, const_x: np.ndarray, const_y: np.ndarray, weight: float):
        self.var_ids = var_ids
        self.const_x = const_x
        self.const_y = const_y
        self.weight = weight


class DreamplaceMoreauPlacer:
    def __init__(
        self,
        seed: int = 7,
        base_iters: int = 48,
        max_iters: int = 96,
        smoothing_t: float = 0.08,
        step_size: float | None = None,
        anchor_weight: float = 0.010,
        spread_weight: float = 0.030,
        overlap_gap: float = 1e-3,
        legalize_rounds: int = 260,
    ):
        self.seed = seed
        self.base_iters = base_iters
        self.max_iters = max_iters
        self.smoothing_t = float(max(1e-4, smoothing_t))
        self.step_size = step_size
        self.anchor_weight = anchor_weight
        self.spread_weight = spread_weight
        self.overlap_gap = overlap_gap
        self.legalize_rounds = legalize_rounds

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        placement = benchmark.macro_positions.clone().float()
        num_hard = benchmark.num_hard_macros
        if num_hard <= 1:
            return placement

        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable_mask)[0].tolist()
        if not movable_indices:
            return placement

        nets = self._build_nets(benchmark, movable_indices)
        if not nets:
            return self._legalize_hard(placement, benchmark)

        sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
        movable_sizes = sizes[movable_indices]

        x0 = placement[movable_indices].cpu().numpy().astype(np.float64)
        x_prev = x0.copy()
        y = x0.copy()

        half_w = 0.5 * movable_sizes[:, 0]
        half_h = 0.5 * movable_sizes[:, 1]
        x_lo = half_w
        x_hi = float(benchmark.canvas_width) - half_w
        y_lo = half_h
        y_hi = float(benchmark.canvas_height) - half_h

        t = self.smoothing_t
        lr = self.step_size if self.step_size is not None else t

        n_mov = len(movable_indices)
        extra = int(0.12 * np.sqrt(float(n_mov)))
        iters = min(self.max_iters, self.base_iters + extra)

        spread_scale = max(1e-3, float(np.mean(np.maximum(movable_sizes[:, 0], movable_sizes[:, 1]))))

        for k in range(1, iters + 1):
            grad = self._moreau_hpwl_grad(y, nets, t)

            if self.anchor_weight > 0.0:
                grad += self.anchor_weight * (y - x0)

            if self.spread_weight > 0.0:
                grad += self.spread_weight * self._spread_grad(y, movable_sizes, spread_scale)

            # Light gradient clipping improves stability on very high-degree nets.
            gnorm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad = grad / np.maximum(1.0, gnorm / 3.5)

            x_new = y - lr * grad
            x_new[:, 0] = np.clip(x_new[:, 0], x_lo, x_hi)
            x_new[:, 1] = np.clip(x_new[:, 1], y_lo, y_hi)

            beta = (k - 1.0) / (k + 2.0)
            y = x_new + beta * (x_new - x_prev)
            y[:, 0] = np.clip(y[:, 0], x_lo, x_hi)
            y[:, 1] = np.clip(y[:, 1], y_lo, y_hi)
            x_prev = x_new

        placement[movable_indices] = torch.tensor(x_prev, dtype=placement.dtype)

        fixed = benchmark.macro_fixed
        if fixed.any():
            placement[fixed] = benchmark.macro_positions[fixed]

        placement = self._legalize_hard(placement, benchmark)
        return placement

    def _build_nets(self, benchmark: Benchmark, movable_indices: List[int]) -> List[_NetData]:
        global_to_local = {g: i for i, g in enumerate(movable_indices)}

        macro_pos = benchmark.macro_positions.cpu().numpy().astype(np.float64)
        port_pos = benchmark.port_positions.cpu().numpy().astype(np.float64)

        nets: List[_NetData] = []
        for net_id, nodes_tensor in enumerate(benchmark.net_nodes):
            nodes = nodes_tensor.tolist()
            var_ids: List[int] = []
            const_x: List[float] = []
            const_y: List[float] = []

            for node in nodes:
                if node < benchmark.num_macros:
                    local = global_to_local.get(node)
                    if local is not None:
                        var_ids.append(local)
                    else:
                        const_x.append(float(macro_pos[node, 0]))
                        const_y.append(float(macro_pos[node, 1]))
                else:
                    port_idx = node - benchmark.num_macros
                    if 0 <= port_idx < port_pos.shape[0]:
                        const_x.append(float(port_pos[port_idx, 0]))
                        const_y.append(float(port_pos[port_idx, 1]))

            if len(var_ids) + len(const_x) < 2:
                continue

            weight = (
                float(benchmark.net_weights[net_id].item())
                if net_id < benchmark.net_weights.shape[0]
                else 1.0
            )
            nets.append(
                _NetData(
                    var_ids=np.asarray(var_ids, dtype=np.int32),
                    const_x=np.asarray(const_x, dtype=np.float64),
                    const_y=np.asarray(const_y, dtype=np.float64),
                    weight=weight,
                )
            )

        return nets

    def _moreau_hpwl_grad(self, pos: np.ndarray, nets: List[_NetData], t: float) -> np.ndarray:
        grad = np.zeros_like(pos)

        for net in nets:
            ids = net.var_ids
            n_var = ids.shape[0]
            if n_var == 0:
                continue

            x_var = pos[ids, 0]
            y_var = pos[ids, 1]

            if net.const_x.size > 0:
                x_all = np.concatenate((x_var, net.const_x), axis=0)
                y_all = np.concatenate((y_var, net.const_y), axis=0)
            else:
                x_all = x_var.copy()
                y_all = y_var.copy()

            gx_all = self._moreau_hpwl_1d_grad(x_all, t)
            gy_all = self._moreau_hpwl_1d_grad(y_all, t)

            np.add.at(grad[:, 0], ids, net.weight * gx_all[:n_var])
            np.add.at(grad[:, 1], ids, net.weight * gy_all[:n_var])

        return grad

    def _moreau_hpwl_1d_grad(self, values: np.ndarray, t: float) -> np.ndarray:
        """
        Exact Moreau envelope gradient for 1D HPWL = max(values) - min(values).

        prox_{t*HPWL}(x):
            min_u (max(u) - min(u)) + (1/(2t)) ||u - x||^2

        KKT yields a clipped structure u = clip(x, b, a), where a and b satisfy:
            sum_i (x_i - a)_+ = t
            sum_i (b - x_i)_+ = t
        via water-filling.

        Gradient:
            grad e_t(x) = (x - prox_{t*HPWL}(x)) / t
        """
        if values.size <= 1:
            return np.zeros_like(values)

        a = self._upper_threshold(values, t)
        b = self._lower_threshold(values, t)

        if b <= a:
            prox = np.clip(values, b, a)
        else:
            prox = np.full_like(values, np.mean(values))

        return (values - prox) / t

    def _upper_threshold(self, values: np.ndarray, t: float) -> float:
        y = np.sort(values)[::-1]
        prefix = 0.0
        n = y.size

        for k in range(1, n + 1):
            prefix += y[k - 1]
            a = (prefix - t) / float(k)
            hi = y[k - 1]
            lo = y[k] if k < n else -np.inf
            if a <= hi + 1e-12 and a >= lo - 1e-12:
                return float(a)

        return float((np.sum(y) - t) / float(n))

    def _lower_threshold(self, values: np.ndarray, t: float) -> float:
        y = np.sort(values)
        prefix = 0.0
        n = y.size

        for k in range(1, n + 1):
            prefix += y[k - 1]
            b = (prefix + t) / float(k)
            lo = y[k - 1]
            hi = y[k] if k < n else np.inf
            if b >= lo - 1e-12 and b <= hi + 1e-12:
                return float(b)

        return float((np.sum(y) + t) / float(n))

    def _spread_grad(self, pos: np.ndarray, sizes: np.ndarray, scale: float) -> np.ndarray:
        """
        Smooth pairwise Gaussian repulsion for early overlap avoidance.

        phi_ij = exp(-(||p_i-p_j||^2)/sigma^2),
        grad_i phi = -(2/sigma^2) * (p_i-p_j) * phi.
        """
        n = pos.shape[0]
        if n <= 1:
            return np.zeros_like(pos)

        sigma = max(1e-3, scale)
        inv_sigma2 = 1.0 / (sigma * sigma)

        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]

        # Slightly anisotropic scaling by macro extents improves behavior on tall/wide blocks.
        mean_w = max(1e-3, float(np.mean(sizes[:, 0])))
        mean_h = max(1e-3, float(np.mean(sizes[:, 1])))
        q = (dx * dx) / (mean_w * mean_w) + (dy * dy) / (mean_h * mean_h)

        k = np.exp(-q * inv_sigma2)
        np.fill_diagonal(k, 0.0)

        grad = np.zeros_like(pos)
        grad[:, 0] = -2.0 * inv_sigma2 * np.sum(dx * k, axis=1)
        grad[:, 1] = -2.0 * inv_sigma2 * np.sum(dy * k, axis=1)
        return grad

    def _legalize_hard(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        num_hard = benchmark.num_hard_macros
        if num_hard <= 1:
            return placement

        out = placement.clone()
        pos = out[:num_hard].cpu().numpy().astype(np.float64).copy()
        sizes = benchmark.macro_sizes[:num_hard].cpu().numpy().astype(np.float64)

        half_w = 0.5 * sizes[:, 0]
        half_h = 0.5 * sizes[:, 1]
        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        movable_f = movable.astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        gap = self.overlap_gap

        sep_x = half_w[:, None] + half_w[None, :] + gap
        sep_y = half_h[:, None] + half_h[None, :] + gap
        tri_mask = np.triu(np.ones((num_hard, num_hard), dtype=bool), k=1)

        for i in range(num_hard):
            pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

        # Stage 1: vectorized overlap-force relaxation.
        for _ in range(self.legalize_rounds):
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

            use_x = choose_x
            use_y = ~choose_x

            m = both & use_x
            dix[m] = 0.5 * sx[m] * px[m]
            djx[m] = -0.5 * sx[m] * px[m]
            m = both & use_y
            diy[m] = 0.5 * sy[m] * py[m]
            djy[m] = -0.5 * sy[m] * py[m]

            m = only_i & use_x
            dix[m] = sx[m] * px[m]
            m = only_i & use_y
            diy[m] = sy[m] * py[m]

            m = only_j & use_x
            djx[m] = -sx[m] * px[m]
            m = only_j & use_y
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

        # Stage 2: targeted local reinsertion for any remaining overlaps.
        remaining = self._collect_overlapping_macros(pos, sizes)
        if remaining:
            for i in remaining:
                if not movable[i]:
                    continue
                self._reinsert_one(i, pos, sizes, movable, cw, ch, gap)

        # Stage 3: one short vectorized polish after reinsertion.
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
                    elif movable[j]:
                        pos[j, 0] = np.clip(pos[j, 0] - s * px, half_w[j], cw - half_w[j])
                else:
                    s = 1.0 if dy[i, j] >= 0.0 else -1.0
                    if movable[i] and movable[j]:
                        pos[i, 1] = np.clip(pos[i, 1] + 0.5 * s * py, half_h[i], ch - half_h[i])
                        pos[j, 1] = np.clip(pos[j, 1] - 0.5 * s * py, half_h[j], ch - half_h[j])
                    elif movable[i]:
                        pos[i, 1] = np.clip(pos[i, 1] + s * py, half_h[i], ch - half_h[i])
                    elif movable[j]:
                        pos[j, 1] = np.clip(pos[j, 1] - s * py, half_h[j], ch - half_h[j])

        out[:num_hard] = torch.tensor(pos, dtype=out.dtype)
        if benchmark.macro_fixed.any():
            out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return out

    def _collect_overlapping_macros(self, pos: np.ndarray, sizes: np.ndarray) -> List[int]:
        n = pos.shape[0]
        if n <= 1:
            return []

        hw = 0.5 * sizes[:, 0]
        hh = 0.5 * sizes[:, 1]
        bad = set()

        for i in range(n):
            for j in range(i + 1, n):
                if abs(pos[i, 0] - pos[j, 0]) < (hw[i] + hw[j]) and abs(pos[i, 1] - pos[j, 1]) < (
                    hh[i] + hh[j]
                ):
                    bad.add(i)
                    bad.add(j)

        return sorted(bad)

    def _reinsert_one(
        self,
        idx: int,
        pos: np.ndarray,
        sizes: np.ndarray,
        movable: np.ndarray,
        canvas_w: float,
        canvas_h: float,
        gap: float,
    ) -> None:
        """Search near current location for a legal position with minimal displacement."""
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

        max_r = 80
        for r in range(1, max_r + 1):
            samples = max(16, 8 * r)
            radius = r * step
            for s in range(samples):
                theta = 2.0 * np.pi * (s / samples)
                x = base_x + radius * np.cos(theta)
                y = base_y + radius * np.sin(theta)
                x = float(np.clip(x, hw, canvas_w - hw))
                y = float(np.clip(y, hh, canvas_h - hh))
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
