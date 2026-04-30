"""
DREAMPlace-style Moreau-HPWL placer with L-BFGS-B optimization.

Uses scipy L-BFGS-B (quasi-Newton, second-order) instead of Nesterov gradient
descent. L-BFGS-B converges in far fewer iterations on smooth objectives and
handles canvas bounds natively — no step-size tuning required.

Objective: Moreau-envelope HPWL + anchor regularizer + Gaussian spread repulsion.
The spread term acts as a density proxy, pushing macros away from crowded regions.
"""

from __future__ import annotations

import math
from typing import List
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize, Bounds

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
        lbfgs_iters: int = 300,
        smoothing_t: float = 0.08,
        anchor_weight: float = 0.005,
        spread_weight: float = 0.1,
        density_weight: float = 0.0,
        overlap_gap: float = 1e-3,
        legalize_rounds: int = 260,
        oracle_refine_steps: int = 0,
        oracle_radius_scale: float = 0.14,
        oracle_temperature: float = 0.004,
        hpwl_model: str = "moreau",
    ):
        self.seed = seed
        self.lbfgs_iters = lbfgs_iters
        self.smoothing_t = float(max(1e-4, smoothing_t))
        self.anchor_weight = anchor_weight
        self.spread_weight = spread_weight
        self.density_weight = density_weight
        self.overlap_gap = overlap_gap
        self.legalize_rounds = legalize_rounds
        self.oracle_refine_steps = oracle_refine_steps
        self.oracle_radius_scale = oracle_radius_scale
        self.oracle_temperature = oracle_temperature
        self.hpwl_model = hpwl_model

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

        sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
        movable_sizes = sizes[movable_indices]

        x0 = placement[movable_indices].cpu().numpy().astype(np.float64)

        half_w = 0.5 * movable_sizes[:, 0]
        half_h = 0.5 * movable_sizes[:, 1]
        x_lo = half_w
        x_hi = float(benchmark.canvas_width) - half_w
        y_lo = half_h
        y_hi = float(benchmark.canvas_height) - half_h

        if self.lbfgs_iters > 0:
            nets = self._build_nets(benchmark, movable_indices)
            if nets:
                # Pass full sizes (all hard macros) for density — fixed macros still
                # contribute density and must be accounted for in the bin grid.
                all_hard_pos = placement[:num_hard].cpu().numpy().astype(np.float64)
                all_hard_sizes = sizes[:num_hard]
                optimized = self._run_lbfgsb(
                    x0.copy(), nets, x_lo, x_hi, y_lo, y_hi,
                    movable_sizes, movable_indices, x0, self.smoothing_t,
                    all_hard_pos, all_hard_sizes,
                    int(benchmark.grid_rows), int(benchmark.grid_cols),
                    float(benchmark.canvas_width), float(benchmark.canvas_height),
                )
            else:
                optimized = x0.copy()
        else:
            optimized = x0.copy()

        placement[movable_indices] = torch.tensor(optimized, dtype=placement.dtype)

        fixed = benchmark.macro_fixed
        if fixed.any():
            placement[fixed] = benchmark.macro_positions[fixed]

        placement = self._legalize_hard(placement, benchmark)

        if self.oracle_refine_steps > 0:
            placement = self._oracle_refine(placement, benchmark)

        return placement

    # ------------------------------------------------------------------
    # L-BFGS-B optimizer
    # ------------------------------------------------------------------

    def _run_lbfgsb(
        self,
        pos_init: np.ndarray,
        nets: List[_NetData],
        x_lo: np.ndarray,
        x_hi: np.ndarray,
        y_lo: np.ndarray,
        y_hi: np.ndarray,
        movable_sizes: np.ndarray,
        movable_indices: List[int],
        x0: np.ndarray,
        t: float,
        all_hard_pos: np.ndarray,
        all_hard_sizes: np.ndarray,
        grid_rows: int,
        grid_cols: int,
        canvas_w: float,
        canvas_h: float,
    ) -> np.ndarray:
        n_mov = pos_init.shape[0]
        n_hard = all_hard_pos.shape[0]
        spread_scale = max(1e-3, float(np.mean(np.maximum(movable_sizes[:, 0], movable_sizes[:, 1]))))

        # Fixed macros contribute a constant density offset — precompute once.
        movable_set = set(movable_indices)
        fixed_indices = [i for i in range(n_hard) if i not in movable_set]
        fixed_pos = all_hard_pos[fixed_indices] if fixed_indices else np.zeros((0, 2))
        fixed_sizes = all_hard_sizes[fixed_indices] if fixed_indices else np.zeros((0, 2))

        # Local index within movable array for each hard-macro index
        hard_to_local = {g: k for k, g in enumerate(movable_indices)}

        def obj_and_grad(flat: np.ndarray):
            pos = flat.reshape(n_mov, 2)
            val, grad = self._hpwl_obj_and_grad(pos, nets, t)

            if self.anchor_weight > 0.0:
                diff = pos - x0
                val += 0.5 * self.anchor_weight * float(np.sum(diff ** 2))
                grad = grad + self.anchor_weight * diff

            if self.spread_weight > 0.0:
                val += self.spread_weight * self._spread_obj(pos, movable_sizes, spread_scale)
                grad = grad + self.spread_weight * self._spread_grad(pos, movable_sizes, spread_scale)

            if self.density_weight > 0.0:
                dv, dg = self._density_obj_and_grad(
                    pos, movable_sizes,
                    fixed_pos, fixed_sizes,
                    grid_rows, grid_cols, canvas_w, canvas_h,
                )
                val += self.density_weight * dv
                grad = grad + self.density_weight * dg

            return float(val), grad.ravel()

        lo_2d = np.column_stack([x_lo, y_lo])
        hi_2d = np.column_stack([x_hi, y_hi])
        bounds = Bounds(lb=lo_2d.ravel(), ub=hi_2d.ravel())

        result = minimize(
            obj_and_grad,
            pos_init.ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self.lbfgs_iters, "ftol": 1e-10, "gtol": 1e-6},
        )

        return np.clip(result.x.reshape(n_mov, 2), lo_2d, hi_2d)

    # ------------------------------------------------------------------
    # Bin-based density — objective + gradient
    # ------------------------------------------------------------------

    def _density_obj_and_grad(
        self,
        pos: np.ndarray,
        sizes: np.ndarray,
        fixed_pos: np.ndarray,
        fixed_sizes: np.ndarray,
        grid_rows: int,
        grid_cols: int,
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, np.ndarray]:
        """
        Smooth bin-density penalty: 0.5 * sum_{bins} density[r,c]^2.

        density[r,c] = (sum of macro area overlapping bin (r,c)) / bin_area.

        The gradient is piecewise-linear (kinks only at bin edges), which
        L-BFGS-B handles well. Fixed macros contribute a constant density
        offset included in the objective but not differentiated.
        """
        bin_w = canvas_w / grid_cols
        bin_h = canvas_h / grid_rows
        bin_area = bin_w * bin_h

        # Bin edge coordinates
        bx = np.linspace(0.0, canvas_w, grid_cols + 1)  # (grid_cols+1,)
        by = np.linspace(0.0, canvas_h, grid_rows + 1)  # (grid_rows+1,)
        bx_lo, bx_hi = bx[:-1], bx[1:]  # (grid_cols,)
        by_lo, by_hi = by[:-1], by[1:]  # (grid_rows,)

        def overlap_x(cx, hw):
            """Overlap of a 1D interval [cx-hw, cx+hw] with each x-bin. Shape (n, grid_cols)."""
            lx = cx[:, None] - hw[:, None]
            rx = cx[:, None] + hw[:, None]
            return np.maximum(0.0, np.minimum(rx, bx_hi[None, :]) - np.maximum(lx, bx_lo[None, :]))

        def overlap_y(cy, hh):
            lyi = cy[:, None] - hh[:, None]
            ryi = cy[:, None] + hh[:, None]
            return np.maximum(0.0, np.minimum(ryi, by_hi[None, :]) - np.maximum(lyi, by_lo[None, :]))

        hw = 0.5 * sizes[:, 0]
        hh = 0.5 * sizes[:, 1]

        ov_x = overlap_x(pos[:, 0], hw)   # (n_mov, grid_cols)
        ov_y = overlap_y(pos[:, 1], hh)   # (n_mov, grid_rows)

        # density from movable macros: einsum('ic,ir->rc', ov_x, ov_y) / bin_area
        # (n_mov, grid_cols)^T @ (n_mov, grid_rows) = (grid_cols, grid_rows) → transpose
        density = (ov_x.T @ ov_y).T / bin_area   # (grid_rows, grid_cols)

        # Add fixed-macro density (constant, no gradient)
        if fixed_pos.shape[0] > 0:
            fhw = 0.5 * fixed_sizes[:, 0]
            fhh = 0.5 * fixed_sizes[:, 1]
            fov_x = overlap_x(fixed_pos[:, 0], fhw)
            fov_y = overlap_y(fixed_pos[:, 1], fhh)
            density = density + (fov_x.T @ fov_y).T / bin_area

        obj = 0.5 * float(np.sum(density ** 2))

        # Gradient w.r.t. movable positions:
        # d(obj)/d(x_i) = density_weight * sum_{r,c} density[r,c] * d(density[r,c])/d(x_i)
        # d(density[r,c])/d(x_i) = ov_y[i,r] * d(ov_x[i,c])/d(x_i) / bin_area
        # d(ov_x[i,c])/d(x_i) = I[rx_i < bx_hi_c] - I[lx_i > bx_lo_c], when ov_x[i,c] > 0

        # w_x[i,c] = sum_r density[r,c] * ov_y[i,r]  →  (n_mov, grid_cols)
        w_x = ov_y @ density        # (n_mov, grid_rows) @ (grid_rows, grid_cols) = (n_mov, grid_cols)

        cx = pos[:, 0]
        lx_i = cx - hw
        rx_i = cx + hw
        active_x = ov_x > 0
        drx = (rx_i[:, None] <= bx_hi[None, :]).astype(np.float64)
        dlx = (lx_i[:, None] >= bx_lo[None, :]).astype(np.float64)
        dov_x = (drx - dlx) * active_x   # (n_mov, grid_cols)

        # w_y[i,r] = sum_c density[r,c] * ov_x[i,c]  →  (n_mov, grid_rows)
        w_y = ov_x @ density.T      # (n_mov, grid_cols) @ (grid_cols, grid_rows) = (n_mov, grid_rows)

        cy = pos[:, 1]
        ly_i = cy - hh
        ry_i = cy + hh
        active_y = ov_y > 0
        dry = (ry_i[:, None] <= by_hi[None, :]).astype(np.float64)
        dly = (ly_i[:, None] >= by_lo[None, :]).astype(np.float64)
        dov_y = (dry - dly) * active_y   # (n_mov, grid_rows)

        grad = np.zeros_like(pos)
        grad[:, 0] = np.sum(w_x * dov_x, axis=1) / bin_area
        grad[:, 1] = np.sum(w_y * dov_y, axis=1) / bin_area

        return obj, grad

    # ------------------------------------------------------------------
    # Moreau HPWL — objective + gradient (combined for efficiency)
    # ------------------------------------------------------------------

    def _hpwl_obj_and_grad(
        self, pos: np.ndarray, nets: List[_NetData], t: float
    ) -> tuple[float, np.ndarray]:
        if self.hpwl_model == "lse":
            return self._lse_hpwl_obj_and_grad(pos, nets, t)
        return self._moreau_hpwl_obj_and_grad(pos, nets, t)

    def _moreau_hpwl_obj_and_grad(
        self, pos: np.ndarray, nets: List[_NetData], t: float
    ) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(pos)
        obj = 0.0

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

            vx, gx_all = self._moreau_hpwl_1d_obj_and_grad(x_all, t)
            vy, gy_all = self._moreau_hpwl_1d_obj_and_grad(y_all, t)

            np.add.at(grad[:, 0], ids, net.weight * gx_all[:n_var])
            np.add.at(grad[:, 1], ids, net.weight * gy_all[:n_var])
            obj += net.weight * (vx + vy)

        return obj, grad

    def _moreau_hpwl_1d_obj_and_grad(
        self, values: np.ndarray, t: float
    ) -> tuple[float, np.ndarray]:
        """
        Moreau envelope value and gradient for 1D HPWL = max(v) - min(v).

        prox_{t*HPWL}(v) = clip(v, b, a) where a, b are water-filling thresholds.
        e_t(v) = HPWL(prox) + (1/2t)||v - prox||^2
        grad    = (v - prox) / t
        """
        if values.size <= 1:
            return 0.0, np.zeros_like(values)

        a = self._upper_threshold(values, t)
        b = self._lower_threshold(values, t)

        if b <= a:
            prox = np.clip(values, b, a)
        else:
            prox = np.full_like(values, np.mean(values))

        residual = values - prox
        grad = residual / t
        hpwl = float(np.max(prox) - np.min(prox))
        val = hpwl + 0.5 / t * float(np.dot(residual, residual))
        return val, grad

    def _lse_hpwl_obj_and_grad(
        self, pos: np.ndarray, nets: List[_NetData], t: float
    ) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(pos)
        obj = 0.0
        gamma = max(1e-4, float(t))

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

            vx, gx_all = self._lse_hpwl_1d_obj_and_grad(x_all, gamma)
            vy, gy_all = self._lse_hpwl_1d_obj_and_grad(y_all, gamma)
            np.add.at(grad[:, 0], ids, net.weight * gx_all[:n_var])
            np.add.at(grad[:, 1], ids, net.weight * gy_all[:n_var])
            obj += net.weight * (vx + vy)

        return obj, grad

    def _lse_hpwl_1d_obj_and_grad(
        self, values: np.ndarray, gamma: float
    ) -> tuple[float, np.ndarray]:
        if values.size <= 1:
            return 0.0, np.zeros_like(values)

        z_hi = values / gamma
        m_hi = float(np.max(z_hi))
        e_hi = np.exp(z_hi - m_hi)
        s_hi = float(np.sum(e_hi))

        z_lo = -values / gamma
        m_lo = float(np.max(z_lo))
        e_lo = np.exp(z_lo - m_lo)
        s_lo = float(np.sum(e_lo))

        val = gamma * (m_hi + math.log(s_hi) + m_lo + math.log(s_lo))
        grad = e_hi / s_hi - e_lo / s_lo
        return float(val), grad

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

    # ------------------------------------------------------------------
    # Spread repulsion (density proxy) — objective + gradient
    # ------------------------------------------------------------------

    def _spread_obj(self, pos: np.ndarray, sizes: np.ndarray, scale: float) -> float:
        """Sum of pairwise Gaussian repulsion potentials."""
        n = pos.shape[0]
        if n <= 1:
            return 0.0
        sigma = max(1e-3, scale)
        inv_sigma2 = 1.0 / (sigma * sigma)
        mean_w = max(1e-3, float(np.mean(sizes[:, 0])))
        mean_h = max(1e-3, float(np.mean(sizes[:, 1])))
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        q = (dx * dx) / (mean_w * mean_w) + (dy * dy) / (mean_h * mean_h)
        k = np.exp(-q * inv_sigma2)
        np.fill_diagonal(k, 0.0)
        return float(0.5 * np.sum(k))

    def _spread_grad(self, pos: np.ndarray, sizes: np.ndarray, scale: float) -> np.ndarray:
        n = pos.shape[0]
        if n <= 1:
            return np.zeros_like(pos)
        sigma = max(1e-3, scale)
        inv_sigma2 = 1.0 / (sigma * sigma)
        mean_w = max(1e-3, float(np.mean(sizes[:, 0])))
        mean_h = max(1e-3, float(np.mean(sizes[:, 1])))
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        q = (dx * dx) / (mean_w * mean_w) + (dy * dy) / (mean_h * mean_h)
        k = np.exp(-q * inv_sigma2)
        np.fill_diagonal(k, 0.0)
        grad = np.zeros_like(pos)
        grad[:, 0] = -2.0 * inv_sigma2 * np.sum(dx * k, axis=1)
        grad[:, 1] = -2.0 * inv_sigma2 * np.sum(dy * k, axis=1)
        return grad

    # ------------------------------------------------------------------
    # Net builder
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Legalization (vectorized overlap-force + reinsert)
    # ------------------------------------------------------------------

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

            ii = ii[active]; jj = jj[active]
            ox = ox[active]; oy = oy[active]
            dxp = dxp[active]; dyp = dyp[active]
            choose_x = choose_x[active]
            mi = mi[active]; mj = mj[active]

            sx = np.where(dxp >= 0.0, 1.0, -1.0)
            sy = np.where(dyp >= 0.0, 1.0, -1.0)
            px = ox + gap
            py = oy + gap

            both = mi & mj
            only_i = mi & (~mj)
            only_j = (~mi) & mj

            dix = np.zeros_like(px); diy = np.zeros_like(py)
            djx = np.zeros_like(px); djy = np.zeros_like(py)

            m = both & choose_x
            dix[m] = 0.5 * sx[m] * px[m]; djx[m] = -0.5 * sx[m] * px[m]
            m = both & ~choose_x
            diy[m] = 0.5 * sy[m] * py[m]; djy[m] = -0.5 * sy[m] * py[m]
            m = only_i & choose_x; dix[m] = sx[m] * px[m]
            m = only_i & ~choose_x; diy[m] = sy[m] * py[m]
            m = only_j & choose_x; djx[m] = -sx[m] * px[m]
            m = only_j & ~choose_x; djy[m] = -sy[m] * py[m]

            moves = np.zeros_like(pos)
            np.add.at(moves[:, 0], ii, dix); np.add.at(moves[:, 1], ii, diy)
            np.add.at(moves[:, 0], jj, djx); np.add.at(moves[:, 1], jj, djy)

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

        # Stage 2: targeted local reinsertion for remaining overlaps.
        remaining = self._collect_overlapping_macros(pos, sizes)
        if remaining:
            for i in remaining:
                if not movable[i]:
                    continue
                self._reinsert_one(i, pos, sizes, movable, cw, ch, gap)

        # Stage 3: short polish pass.
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

    # ------------------------------------------------------------------
    # Oracle SA refinement on true proxy cost
    # ------------------------------------------------------------------

    def _oracle_refine(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        plc = self._load_plc_for_benchmark(benchmark)
        if plc is None:
            return placement

        try:
            from macro_place.objective import compute_proxy_cost
        except Exception:
            return placement

        num_hard = benchmark.num_hard_macros
        if num_hard <= 1:
            return placement

        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable)[0].tolist()
        if not movable_indices:
            return placement

        sizes = benchmark.macro_sizes[:num_hard].cpu().numpy().astype(np.float64)
        half_w = 0.5 * sizes[:, 0]
        half_h = 0.5 * sizes[:, 1]
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        gap = self.overlap_gap

        current = placement.clone()
        best = current.clone()
        try:
            best_cost = float(compute_proxy_cost(current, benchmark, plc)["proxy_cost"])
        except Exception:
            return placement

        current_cost = best_cost
        pos = current[:num_hard].cpu().numpy().astype(np.float64)

        adjacency = self._build_macro_adjacency(benchmark, num_hard)
        canvas_scale = max(cw, ch)
        r0 = max(0.02, self.oracle_radius_scale * canvas_scale)
        temp0 = max(1e-6, self.oracle_temperature)
        rng = np.random.default_rng(self.seed + 97)

        steps = max(0, int(self.oracle_refine_steps))
        if steps == 0:
            return current

        for step in range(steps):
            frac = step / max(1, steps - 1)
            radius = r0 * (1.0 - 0.85 * frac)
            temp = temp0 * (1.0 - 0.75 * frac)

            idx = int(rng.choice(movable_indices))
            old_x = float(pos[idx, 0])
            old_y = float(pos[idx, 1])

            tx, ty = self._anchor_target(idx, current, benchmark, adjacency[idx])
            cand_x = old_x + 0.38 * (tx - old_x) + float(rng.normal(0.0, radius))
            cand_y = old_y + 0.38 * (ty - old_y) + float(rng.normal(0.0, radius))
            cand_x = float(np.clip(cand_x, half_w[idx], cw - half_w[idx]))
            cand_y = float(np.clip(cand_y, half_h[idx], ch - half_h[idx]))

            if self._overlaps_hard(pos, sizes, idx, cand_x, cand_y, gap):
                accepted_probe = False
                for _ in range(6):
                    ang = float(rng.uniform(0.0, 2.0 * np.pi))
                    rx = old_x + radius * np.cos(ang)
                    ry = old_y + radius * np.sin(ang)
                    rx = float(np.clip(rx, half_w[idx], cw - half_w[idx]))
                    ry = float(np.clip(ry, half_h[idx], ch - half_h[idx]))
                    if not self._overlaps_hard(pos, sizes, idx, rx, ry, gap):
                        cand_x, cand_y = rx, ry
                        accepted_probe = True
                        break
                if not accepted_probe:
                    continue

            pos[idx, 0] = cand_x; pos[idx, 1] = cand_y
            current[idx, 0] = cand_x; current[idx, 1] = cand_y

            try:
                cand_cost = float(compute_proxy_cost(current, benchmark, plc)["proxy_cost"])
            except Exception:
                cand_cost = float("inf")

            delta = cand_cost - current_cost
            accept = delta <= 0.0
            if not accept and temp > 0.0:
                accept = rng.uniform(0.0, 1.0) < np.exp(-delta / temp)

            if accept:
                current_cost = cand_cost
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best = current.clone()
            else:
                pos[idx, 0] = old_x; pos[idx, 1] = old_y
                current[idx, 0] = old_x; current[idx, 1] = old_y

        return best

    def _load_plc_for_benchmark(self, benchmark: Benchmark):
        try:
            from macro_place.loader import load_benchmark_from_dir, load_benchmark
        except Exception:
            return None

        root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark.name
        if root.exists():
            try:
                _, plc = load_benchmark_from_dir(str(root))
                return plc
            except Exception:
                return None

        ng45_map = {
            "ariane133": "ariane133",
            "ariane136": "ariane136",
            "mempool_tile": "mempool_tile",
            "nvdla": "nvdla",
        }
        design = ng45_map.get(benchmark.name)
        if design is None:
            return None

        ng45_dir = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / design / "netlist" / "output_CT_Grouping"
        )
        netlist_file = ng45_dir / "netlist.pb.txt"
        plc_file = ng45_dir / "initial.plc"
        if not netlist_file.exists() or not plc_file.exists():
            return None

        try:
            _, plc = load_benchmark(str(netlist_file), str(plc_file), name=benchmark.name)
            return plc
        except Exception:
            return None

    def _build_macro_adjacency(self, benchmark: Benchmark, num_hard: int) -> List[List[int]]:
        adjacency: List[List[int]] = [[] for _ in range(num_hard)]
        for nodes_t in benchmark.net_nodes:
            nodes = nodes_t.tolist()
            if len(nodes) <= 1:
                continue
            for node in nodes:
                if 0 <= node < num_hard:
                    adjacency[node].extend([other for other in nodes if other != node])
        return adjacency

    def _anchor_target(self, hard_idx, placement, benchmark, neighbors):
        if not neighbors:
            return float(placement[hard_idx, 0].item()), float(placement[hard_idx, 1].item())
        xs: List[float] = []
        ys: List[float] = []
        for n in neighbors:
            if n < benchmark.num_macros:
                xs.append(float(placement[n, 0].item()))
                ys.append(float(placement[n, 1].item()))
            else:
                port_idx = n - benchmark.num_macros
                if 0 <= port_idx < benchmark.port_positions.shape[0]:
                    xs.append(float(benchmark.port_positions[port_idx, 0].item()))
                    ys.append(float(benchmark.port_positions[port_idx, 1].item()))
        if not xs:
            return float(placement[hard_idx, 0].item()), float(placement[hard_idx, 1].item())
        return float(np.median(np.asarray(xs))), float(np.median(np.asarray(ys)))

    def _overlaps_hard(self, hard_pos, sizes, idx, cand_x, cand_y, gap):
        hw_i = 0.5 * sizes[idx, 0]
        hh_i = 0.5 * sizes[idx, 1]
        for j in range(hard_pos.shape[0]):
            if j == idx:
                continue
            sep_x = hw_i + 0.5 * sizes[j, 0] + gap
            sep_y = hh_i + 0.5 * sizes[j, 1] + gap
            if abs(cand_x - hard_pos[j, 0]) < sep_x and abs(cand_y - hard_pos[j, 1]) < sep_y:
                return True
        return False

    def _collect_overlapping_macros(self, pos, sizes) -> List[int]:
        n = pos.shape[0]
        if n <= 1:
            return []
        hw = 0.5 * sizes[:, 0]
        hh = 0.5 * sizes[:, 1]
        bad = set()
        for i in range(n):
            for j in range(i + 1, n):
                if (abs(pos[i, 0] - pos[j, 0]) < hw[i] + hw[j] and
                        abs(pos[i, 1] - pos[j, 1]) < hh[i] + hh[j]):
                    bad.add(i)
                    bad.add(j)
        return sorted(bad)

    def _reinsert_one(self, idx, pos, sizes, movable, canvas_w, canvas_h, gap):
        if not movable[idx]:
            return
        w = sizes[idx, 0]; h = sizes[idx, 1]
        hw = 0.5 * w; hh = 0.5 * h
        base_x = float(np.clip(pos[idx, 0], hw, canvas_w - hw))
        base_y = float(np.clip(pos[idx, 1], hh, canvas_h - hh))

        def legal(x, y):
            for j in range(pos.shape[0]):
                if j == idx:
                    continue
                sep_x = 0.5 * (w + sizes[j, 0]) + gap
                sep_y = 0.5 * (h + sizes[j, 1]) + gap
                if abs(x - pos[j, 0]) < sep_x and abs(y - pos[j, 1]) < sep_y:
                    return False
            return True

        if legal(base_x, base_y):
            pos[idx, 0] = base_x; pos[idx, 1] = base_y
            return

        step = max(0.15 * max(w, h), 0.02)
        best = None; best_d2 = float("inf")
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
                    best_d2 = d2; best = (x, y)
            if best is not None:
                break

        if best is not None:
            pos[idx, 0], pos[idx, 1] = best
