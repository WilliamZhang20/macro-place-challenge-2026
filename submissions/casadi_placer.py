"""
CasADi/IPOPT local rectangle repair placer.

No CVXPY/DCCP.  This starts from the benchmark placement and solves small NLPs
only for connected overlap components, with displacement from the original
placement as the primary objective.  The old full-chip solve moved too much on
IBM; local repair is a better match for the "initial placement is nearly good"
regime.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import casadi as ca
import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from macro_place.utils import validate_placement

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _hard_legalizer import legalize_hard  # noqa: E402


class CasadiPlacer:
    def __init__(
        self,
        max_outer_iters: int = 5,
        ipopt_max_iter: int = 40,
        anchor_weight: float = 1.0,
        wire_weight: float = 0.0,
        gap: float = 1e-5,
        max_seconds: float = 120.0,
        component_size_limit: int = 12,
        max_components: int = 4,
        soft_proxy_evals: int = 4,
    ):
        self.max_outer_iters = max_outer_iters
        self.ipopt_max_iter = ipopt_max_iter
        self.anchor_weight = anchor_weight
        self.wire_weight = wire_weight
        self.gap = gap
        self.max_seconds = max_seconds
        self.component_size_limit = int(component_size_limit)
        self.max_components = int(max_components)
        self.soft_proxy_evals = int(soft_proxy_evals)
        self._plc_cache = {}

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        start = time.monotonic()
        placement = benchmark.macro_positions.clone().float()
        n_hard = benchmark.num_hard_macros
        if n_hard <= 1:
            return placement

        pos = placement[:n_hard].cpu().numpy().astype(np.float64).copy()
        sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[:n_hard]).cpu().numpy().astype(bool)
        movable_idx = np.flatnonzero(movable).astype(int).tolist()
        if not movable_idx:
            return placement

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        hw = 0.5 * sizes[:, 0]
        hh = 0.5 * sizes[:, 1]
        self._clamp(pos, movable, hw, hh, cw, ch)

        base_pos = pos.copy()
        bad = self._overlap_pairs(pos, sizes, margin=self.gap)
        if not bad:
            placement[:n_hard] = torch.tensor(pos, dtype=placement.dtype)
            return self._soft_proxy_polish(placement, benchmark)

        # Large IBM cases can have many tiny overlap components. IPOPT is not
        # earning its runtime there yet, so use the deterministic DREAMPlace
        # repair as the research-safe path and reserve CasADi for smaller local
        # repair opportunities.
        if n_hard > 450 or len(bad) > 180:
            return self._dreamplace_repair(benchmark.macro_positions.clone().float(), benchmark)

        for outer in range(self.max_outer_iters):
            if time.monotonic() - start > self.max_seconds:
                break
            bad = self._overlap_pairs(pos, sizes, margin=self.gap)
            if not bad:
                break
            any_solved = False
            tried_components = 0
            for comp in self._overlap_components(bad, n_hard):
                if time.monotonic() - start > self.max_seconds:
                    break
                if len(comp) > self.component_size_limit:
                    continue
                if tried_components >= self.max_components:
                    break
                local_movable = [i for i in comp if movable[i]]
                if not local_movable:
                    continue
                tried_components += 1
                pair_cuts = self._component_pair_cuts(pos, sizes, comp)
                if not pair_cuts:
                    continue
                tmp = placement.clone()
                tmp[:n_hard] = torch.tensor(pos, dtype=tmp.dtype)
                wire_terms = self._build_wire_terms(benchmark, tmp, local_movable)
                solved = self._solve_cut_nlp(
                    pos,
                    base_pos,
                    sizes,
                    movable,
                    local_movable,
                    pair_cuts,
                    wire_terms,
                    cw,
                    ch,
                )
                if solved is not None:
                    pos = solved
                    self._clamp(pos, movable, hw, hh, cw, ch)
                    any_solved = True
            self._clamp(pos, movable, hw, hh, cw, ch)

            if not any_solved and outer >= 1:
                break

        pos = self._legalize(pos, sizes, movable, cw, ch, max_rounds=3800)
        placement[:n_hard] = torch.tensor(pos, dtype=placement.dtype)
        if benchmark.macro_fixed.any():
            placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]

        base_repair = self._dreamplace_repair(
            benchmark.macro_positions.clone().float(), benchmark
        )
        candidates = [placement, base_repair]

        ok, _ = validate_placement(placement, benchmark, check_overlaps=True)
        if not ok or int(compute_overlap_metrics(placement, benchmark)["overlap_count"]) != 0:
            strict = self._dreamplace_repair(placement, benchmark)
            candidates.append(strict)

        best = self._select_best_valid(candidates, benchmark)
        if best is not None:
            return self._soft_proxy_polish(best, benchmark)

        strict = candidates[-1]
        ok, _ = validate_placement(strict, benchmark, check_overlaps=True)
        if ok and int(compute_overlap_metrics(strict, benchmark)["overlap_count"]) == 0:
            return self._soft_proxy_polish(strict, benchmark)

        # Last-resort row pack.  It is ugly for proxy, but it keeps the submission valid.
        fallback = benchmark.macro_positions.clone().float()
        self._shelf_pack(fallback, benchmark)
        ok, _ = validate_placement(fallback, benchmark, check_overlaps=True)
        return self._soft_proxy_polish(fallback, benchmark) if ok else benchmark.macro_positions.clone().float()

    def _global_sparse_hard_candidate(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        if not benchmark.name.startswith("ibm"):
            return placement
        n_hard = benchmark.num_hard_macros
        if n_hard < 220:
            return placement

        pos = placement[:n_hard].detach().cpu().numpy().astype(np.float64).copy()
        base_pos = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64)
        sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[:n_hard]).detach().cpu().numpy().astype(bool)
        movable_idx = np.flatnonzero(movable).astype(int).tolist()
        if not movable_idx:
            return placement

        density = self._center_density_map(placement, benchmark)
        bin_w = float(benchmark.canvas_width) / max(1, int(benchmark.grid_cols))
        bin_h = float(benchmark.canvas_height) / max(1, int(benchmark.grid_rows))
        pressure = []
        for i in movable_idx:
            x = float(pos[i, 0])
            y = float(pos[i, 1])
            col = int(np.clip(x / max(bin_w, 1e-9), 0, int(benchmark.grid_cols) - 1))
            row = int(np.clip(y / max(bin_h, 1e-9), 0, int(benchmark.grid_rows) - 1))
            area = float(sizes[i, 0] * sizes[i, 1])
            pressure.append((float(density[row, col]) * area, i))
        pressure.sort(reverse=True)

        selected = [i for _, i in pressure[: min(72, len(pressure))]]
        if len(selected) < 8:
            return placement

        target = pos.copy()
        for i in selected:
            dx, dy = self._density_escape_offsets(
                density, float(pos[i, 0]), float(pos[i, 1]), bin_w, bin_h, benchmark
            )[0]
            # Keep global moves gentle; legalization is allowed to finish the job.
            target[i, 0] = float(np.clip(pos[i, 0] + 0.55 * dx, 0.5 * sizes[i, 0], float(benchmark.canvas_width) - 0.5 * sizes[i, 0]))
            target[i, 1] = float(np.clip(pos[i, 1] + 0.55 * dy, 0.5 * sizes[i, 1], float(benchmark.canvas_height) - 0.5 * sizes[i, 1]))

        solved = self._solve_global_sparse_nlp(
            pos,
            base_pos,
            target,
            sizes,
            movable,
            selected,
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
        )
        if solved is None:
            return placement

        out = placement.clone()
        out[:n_hard] = torch.tensor(solved, dtype=out.dtype)
        if benchmark.macro_fixed.any():
            out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        out = self._dreamplace_repair(out, benchmark)
        ok, _ = validate_placement(out, benchmark, check_overlaps=True)
        if ok and int(compute_overlap_metrics(out, benchmark)["overlap_count"]) == 0:
            return out
        return placement

    def _solve_global_sparse_nlp(
        self,
        pos: np.ndarray,
        base_pos: np.ndarray,
        target: np.ndarray,
        sizes: np.ndarray,
        movable: np.ndarray,
        selected: Sequence[int],
        cw: float,
        ch: float,
    ) -> np.ndarray | None:
        n = len(selected)
        local = {g: k for k, g in enumerate(selected)}
        z = ca.MX.sym("z", 2 * n)
        mean_size = max(1e-3, float(np.mean(np.sqrt(sizes[selected, 0] * sizes[selected, 1]))))
        trust = max(0.20, 0.85 * mean_size)

        def x_expr(i: int):
            k = local.get(i)
            return z[2 * k] if k is not None else float(pos[i, 0])

        def y_expr(i: int):
            k = local.get(i)
            return z[2 * k + 1] if k is not None else float(pos[i, 1])

        obj = 0
        for k, i in enumerate(selected):
            scale = max(1e-3, math.sqrt(float(sizes[i, 0] * sizes[i, 1])))
            dx = (z[2 * k] - float(base_pos[i, 0])) / scale
            dy = (z[2 * k + 1] - float(base_pos[i, 1])) / scale
            tx = (z[2 * k] - float(target[i, 0])) / max(scale, mean_size)
            ty = (z[2 * k + 1] - float(target[i, 1])) / max(scale, mean_size)
            obj += 3.0 * (dx * dx + dy * dy) + 0.18 * (tx * tx + ty * ty)

        # Relative-offset preservation over local KNN edges: DCCP-like shape regularization.
        sel_pos = pos[selected]
        rel_edges = self._knn_edges(sel_pos, k=5)
        for a, b in rel_edges[: min(360, len(rel_edges))]:
            i = selected[a]
            j = selected[b]
            scale = max(mean_size, math.sqrt(float(sizes[i, 0] * sizes[i, 1] + sizes[j, 0] * sizes[j, 1]) * 0.5))
            odx = float(base_pos[i, 0] - base_pos[j, 0])
            ody = float(base_pos[i, 1] - base_pos[j, 1])
            rx = (x_expr(i) - x_expr(j) - odx) / scale
            ry = (y_expr(i) - y_expr(j) - ody) / scale
            obj += 0.35 * (rx * rx + ry * ry)

        g = []
        lbg = []
        ubg = []
        selected_set = set(selected)
        candidate_pairs = []
        hw = 0.5 * sizes[:, 0]
        hh = 0.5 * sizes[:, 1]
        margin = 0.45 * mean_size
        for i in selected:
            # Nearby obstacles, selected or fixed/background. This is sparse global pressure.
            dx = np.abs(pos[:, 0] - pos[i, 0])
            dy = np.abs(pos[:, 1] - pos[i, 1])
            near = np.where((dx < hw + hw[i] + margin) & (dy < hh + hh[i] + margin))[0]
            for j in near.tolist():
                if j == i:
                    continue
                if j in selected_set and j < i:
                    continue
                candidate_pairs.append((i, int(j)))
        # Prefer closest/currently most dangerous constraints.
        def pair_slack(pair):
            i, j = pair
            sx = abs(pos[i, 0] - pos[j, 0]) - (hw[i] + hw[j])
            sy = abs(pos[i, 1] - pos[j, 1]) - (hh[i] + hh[j])
            return min(sx, sy)
        candidate_pairs = sorted(set(candidate_pairs), key=pair_slack)[:900]

        for i, j in candidate_pairs:
            if not movable[i] and not movable[j]:
                continue
            dx = float(pos[i, 0] - pos[j, 0])
            dy = float(pos[i, 1] - pos[j, 1])
            sep_x = float(hw[i] + hw[j] + self.gap)
            sep_y = float(hh[i] + hh[j] + self.gap)
            # Freeze the current separating axis for convex linear cuts.
            slack_x = abs(dx) - sep_x
            slack_y = abs(dy) - sep_y
            if slack_x <= slack_y:
                sign = 1.0 if (dx > 0 or (dx == 0 and i > j)) else -1.0
                g.append(sign * (x_expr(i) - x_expr(j)))
                lbg.append(sep_x)
            else:
                sign = 1.0 if (dy > 0 or (dy == 0 and i > j)) else -1.0
                g.append(sign * (y_expr(i) - y_expr(j)))
                lbg.append(sep_y)
            ubg.append(ca.inf)

        lbx = []
        ubx = []
        x0 = []
        for i in selected:
            hwi = 0.5 * float(sizes[i, 0])
            hhi = 0.5 * float(sizes[i, 1])
            lbx.extend([
                max(hwi + self.gap, float(pos[i, 0]) - trust),
                max(hhi + self.gap, float(pos[i, 1]) - trust),
            ])
            ubx.extend([
                min(cw - hwi - self.gap, float(pos[i, 0]) + trust),
                min(ch - hhi - self.gap, float(pos[i, 1]) + trust),
            ])
            x0.extend([float(pos[i, 0]), float(pos[i, 1])])

        nlp = {"x": z, "f": obj, "g": ca.vertcat(*g) if g else ca.MX.zeros(0, 1)}
        opts = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 55,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 8e-4,
            "ipopt.acceptable_iter": 6,
            "ipopt.linear_solver": "mumps",
        }
        try:
            solver = ca.nlpsol("global_sparse_solver", "ipopt", nlp, opts)
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        except Exception:
            return None

        zval = np.asarray(sol["x"], dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(zval)):
            return None
        out = pos.copy()
        for k, i in enumerate(selected):
            out[i, 0] = zval[2 * k]
            out[i, 1] = zval[2 * k + 1]
        return out

    def _soft_proxy_polish(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        if self.soft_proxy_evals <= 0 or benchmark.num_soft_macros <= 0:
            return placement
        # NG45 already benefits strongly from the repaired floorplan and soft
        # proxy moves are expensive there; focus this budget on IBM.
        if not benchmark.name.startswith("ibm"):
            return placement
        if benchmark.num_macros > 1800:
            return placement
        plc = self._load_plc(benchmark)
        if plc is None:
            return placement

        soft_mask = benchmark.get_soft_macro_mask() & benchmark.get_movable_mask()
        soft_indices = torch.where(soft_mask)[0].tolist()
        if not soft_indices:
            return placement

        try:
            best_cost = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
        except Exception:
            return placement

        best = placement.clone()
        sizes = benchmark.macro_sizes
        bin_w = float(benchmark.canvas_width) / max(1, int(benchmark.grid_cols))
        bin_h = float(benchmark.canvas_height) / max(1, int(benchmark.grid_rows))
        density = self._center_density_map(best, benchmark)

        def pressure(idx: int) -> float:
            x = float(best[idx, 0].item())
            y = float(best[idx, 1].item())
            col = int(np.clip(x / max(bin_w, 1e-9), 0, int(benchmark.grid_cols) - 1))
            row = int(np.clip(y / max(bin_h, 1e-9), 0, int(benchmark.grid_rows) - 1))
            area = float(sizes[idx, 0].item() * sizes[idx, 1].item())
            return float(density[row, col]) * area

        by_pressure = sorted(soft_indices, key=lambda i: (-pressure(i), i))
        by_area = sorted(
            soft_indices,
            key=lambda i: (-float(sizes[i, 0].item() * sizes[i, 1].item()), i),
        )
        ordered = []
        for idx in by_pressure[:6] + by_area[:6]:
            if idx not in ordered:
                ordered.append(idx)

        large_case = benchmark.num_macros > 1800
        batch_count = 36 if large_case else 14
        batch = self._batch_soft_density_spread(
            best, benchmark, ordered[: min(batch_count, len(ordered))], density, bin_w, bin_h
        )
        try:
            batch_cost = float(compute_proxy_cost(batch, benchmark, plc)["proxy_cost"])
        except Exception:
            batch_cost = float("inf")
        if batch_cost + 1e-7 < best_cost:
            best = batch
            best_cost = batch_cost
            density = self._center_density_map(best, benchmark)

        if large_case:
            ok, _ = validate_placement(best, benchmark, check_overlaps=True)
            return best if ok else placement

        evals = 0
        for idx in ordered[: min(8, len(ordered))]:
            if evals >= self.soft_proxy_evals:
                break
            hw = 0.5 * float(sizes[idx, 0].item())
            hh = 0.5 * float(sizes[idx, 1].item())
            old_x = float(best[idx, 0].item())
            old_y = float(best[idx, 1].item())
            offsets = self._density_escape_offsets(
                density, old_x, old_y, bin_w, bin_h, benchmark
            )
            for fallback in ((bin_w, 0.0), (-bin_w, 0.0), (0.0, bin_h), (0.0, -bin_h)):
                if fallback not in offsets:
                    offsets.append(fallback)
            for dx, dy in offsets:
                if evals >= self.soft_proxy_evals:
                    break
                cand = best.clone()
                cand[idx, 0] = float(np.clip(old_x + dx, hw, float(benchmark.canvas_width) - hw))
                cand[idx, 1] = float(np.clip(old_y + dy, hh, float(benchmark.canvas_height) - hh))
                if cand[idx, 0].item() == old_x and cand[idx, 1].item() == old_y:
                    continue
                evals += 1
                try:
                    cost = float(compute_proxy_cost(cand, benchmark, plc)["proxy_cost"])
                except Exception:
                    continue
                if cost + 1e-7 < best_cost:
                    best = cand
                    best_cost = cost
                    old_x = float(best[idx, 0].item())
                    old_y = float(best[idx, 1].item())
                    density = self._center_density_map(best, benchmark)

        ok, _ = validate_placement(best, benchmark, check_overlaps=True)
        return best if ok else placement

    def _batch_soft_density_spread(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        soft_indices: Sequence[int],
        density: np.ndarray,
        bin_w: float,
        bin_h: float,
    ) -> torch.Tensor:
        cand = placement.clone()
        sizes = benchmark.macro_sizes
        local_density = density.copy()
        rows, cols = local_density.shape
        bin_area = max(1e-9, bin_w * bin_h)

        for idx in soft_indices:
            x = float(cand[idx, 0].item())
            y = float(cand[idx, 1].item())
            col = int(np.clip(x / max(bin_w, 1e-9), 0, cols - 1))
            row = int(np.clip(y / max(bin_h, 1e-9), 0, rows - 1))
            current = float(local_density[row, col])
            moves = self._density_escape_offsets(local_density, x, y, bin_w, bin_h, benchmark)
            if not moves:
                continue
            dx, dy = moves[0]
            new_col = int(np.clip((x + dx) / max(bin_w, 1e-9), 0, cols - 1))
            new_row = int(np.clip((y + dy) / max(bin_h, 1e-9), 0, rows - 1))
            if float(local_density[new_row, new_col]) >= current:
                continue

            hw = 0.5 * float(sizes[idx, 0].item())
            hh = 0.5 * float(sizes[idx, 1].item())
            nx = float(np.clip(x + dx, hw, float(benchmark.canvas_width) - hw))
            ny = float(np.clip(y + dy, hh, float(benchmark.canvas_height) - hh))
            if nx == x and ny == y:
                continue

            area_density = float(sizes[idx, 0].item() * sizes[idx, 1].item()) / bin_area
            local_density[row, col] = max(0.0, local_density[row, col] - area_density)
            local_density[new_row, new_col] += area_density
            cand[idx, 0] = nx
            cand[idx, 1] = ny

        return cand

    def _center_density_map(self, placement: torch.Tensor, benchmark: Benchmark) -> np.ndarray:
        rows = int(benchmark.grid_rows)
        cols = int(benchmark.grid_cols)
        density = np.zeros((rows, cols), dtype=np.float64)
        bin_w = float(benchmark.canvas_width) / max(1, cols)
        bin_h = float(benchmark.canvas_height) / max(1, rows)
        bin_area = max(1e-9, bin_w * bin_h)
        pos = placement.detach().cpu().numpy()
        sizes = benchmark.macro_sizes.detach().cpu().numpy()
        for i in range(benchmark.num_macros):
            col = int(np.clip(pos[i, 0] / max(bin_w, 1e-9), 0, cols - 1))
            row = int(np.clip(pos[i, 1] / max(bin_h, 1e-9), 0, rows - 1))
            density[row, col] += float(sizes[i, 0] * sizes[i, 1]) / bin_area
        return density

    def _density_escape_offsets(
        self,
        density: np.ndarray,
        x: float,
        y: float,
        bin_w: float,
        bin_h: float,
        benchmark: Benchmark,
    ) -> List[Tuple[float, float]]:
        rows, cols = density.shape
        col = int(np.clip(x / max(bin_w, 1e-9), 0, cols - 1))
        row = int(np.clip(y / max(bin_h, 1e-9), 0, rows - 1))
        candidates = []
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            rr = row + dr
            cc = col + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                candidates.append((float(density[rr, cc]), dc * bin_w, dr * bin_h))
        candidates.sort(key=lambda item: item[0])
        best_density = [(dx, dy) for _, dx, dy in candidates[:2]]
        cardinal = [(bin_w, 0.0), (-bin_w, 0.0), (0.0, bin_h), (0.0, -bin_h)]
        out = []
        for move in best_density + cardinal:
            if move not in out:
                out.append(move)
        return out or [
            (bin_w, 0.0),
            (-bin_w, 0.0),
            (0.0, bin_h),
            (0.0, -bin_h),
        ]

    def _select_best_valid(
        self, candidates: Sequence[torch.Tensor], benchmark: Benchmark
    ) -> torch.Tensor | None:
        valid = []
        for cand in candidates:
            ok, _ = validate_placement(cand, benchmark, check_overlaps=True)
            if ok and int(compute_overlap_metrics(cand, benchmark)["overlap_count"]) == 0:
                valid.append(cand)
        if not valid:
            return None

        plc = self._load_plc(benchmark)
        if plc is None:
            return valid[0]

        best = valid[0]
        best_cost = float("inf")
        for cand in valid:
            try:
                cost = float(compute_proxy_cost(cand, benchmark, plc)["proxy_cost"])
            except Exception:
                continue
            if cost < best_cost:
                best_cost = cost
                best = cand
        return best

    def _load_plc(self, benchmark: Benchmark):
        if benchmark.name in self._plc_cache:
            return self._plc_cache[benchmark.name]
        try:
            from macro_place.loader import load_benchmark, load_benchmark_from_dir
        except Exception:
            return None

        root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark.name
        if root.exists():
            try:
                _, plc = load_benchmark_from_dir(str(root))
                self._plc_cache[benchmark.name] = plc
                return plc
            except Exception:
                return None

        ng45_dir = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / benchmark.name
            / "netlist"
            / "output_CT_Grouping"
        )
        netlist = ng45_dir / "netlist.pb.txt"
        plc_file = ng45_dir / "initial.plc"
        if netlist.exists() and plc_file.exists():
            try:
                _, plc = load_benchmark(str(netlist), str(plc_file), name=benchmark.name)
                self._plc_cache[benchmark.name] = plc
                return plc
            except Exception:
                return None
        return None

    def _dreamplace_repair(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        out = placement.clone().float()
        for gap, rounds in ((self.gap, 1000), (max(self.gap, 8e-5), 1800), (max(self.gap, 3e-4), 2600)):
            out = legalize_hard(out, benchmark, overlap_gap=gap, legalize_rounds=rounds)
            self._clamp_tensor(out, benchmark, gap)
            ok, _ = validate_placement(out, benchmark, check_overlaps=True)
            if ok and int(compute_overlap_metrics(out, benchmark)["overlap_count"]) == 0:
                return out
        return out

    def _clamp_tensor(self, placement: torch.Tensor, benchmark: Benchmark, gap: float) -> None:
        movable = ~benchmark.macro_fixed
        if not bool(movable.any()):
            return
        hw = benchmark.macro_sizes[:, 0] * 0.5
        hh = benchmark.macro_sizes[:, 1] * 0.5
        placement[:, 0] = torch.where(
            movable,
            torch.clamp(placement[:, 0], hw + gap, float(benchmark.canvas_width) - hw - gap),
            placement[:, 0],
        )
        placement[:, 1] = torch.where(
            movable,
            torch.clamp(placement[:, 1], hh + gap, float(benchmark.canvas_height) - hh - gap),
            placement[:, 1],
        )
        if benchmark.macro_fixed.any():
            placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]

    def _solve_cut_nlp(
        self,
        pos: np.ndarray,
        base_pos: np.ndarray,
        sizes: np.ndarray,
        movable: np.ndarray,
        movable_idx: Sequence[int],
        pair_cuts: Sequence[Tuple[int, int]],
        wire_terms: Sequence[Tuple[int, int | None, float, float, float]],
        cw: float,
        ch: float,
    ) -> np.ndarray | None:
        n = len(movable_idx)
        local: Dict[int, int] = {g: k for k, g in enumerate(movable_idx)}
        z = ca.MX.sym("z", 2 * n)

        def x_expr(i: int):
            k = local.get(i)
            return z[2 * k] if k is not None else float(pos[i, 0])

        def y_expr(i: int):
            k = local.get(i)
            return z[2 * k + 1] if k is not None else float(pos[i, 1])

        obj = 0
        for k, i in enumerate(movable_idx):
            scale = max(1e-3, math.sqrt(float(sizes[i, 0] * sizes[i, 1])))
            dx = (z[2 * k] - float(base_pos[i, 0])) / scale
            dy = (z[2 * k + 1] - float(base_pos[i, 1])) / scale
            obj += self.anchor_weight * (dx * dx + dy * dy)

        for i, j, tx, ty, w in wire_terms:
            xi = x_expr(i)
            yi = y_expr(i)
            if j is None:
                xj = float(tx)
                yj = float(ty)
            else:
                xj = x_expr(j)
                yj = y_expr(j)
            obj += self.wire_weight * float(w) * ((xi - xj) ** 2 + (yi - yj) ** 2)

        g = []
        lbg = []
        ubg = []
        for i, j in pair_cuts:
            if not movable[i] and not movable[j]:
                continue
            dx = float(pos[i, 0] - pos[j, 0])
            dy = float(pos[i, 1] - pos[j, 1])
            sep_x = float(0.5 * (sizes[i, 0] + sizes[j, 0]) + self.gap)
            sep_y = float(0.5 * (sizes[i, 1] + sizes[j, 1]) + self.gap)
            ovx = sep_x - abs(dx)
            ovy = sep_y - abs(dy)
            if ovx <= 0 or ovy <= 0:
                continue
            if ovx <= ovy:
                sign = 1.0 if (dx > 0 or (dx == 0 and i > j)) else -1.0
                g.append(sign * (x_expr(i) - x_expr(j)))
                lbg.append(sep_x)
            else:
                sign = 1.0 if (dy > 0 or (dy == 0 and i > j)) else -1.0
                g.append(sign * (y_expr(i) - y_expr(j)))
                lbg.append(sep_y)
            ubg.append(ca.inf)

        lbx = []
        ubx = []
        x0 = []
        for i in movable_idx:
            hw = 0.5 * float(sizes[i, 0])
            hh = 0.5 * float(sizes[i, 1])
            lbx.extend([hw + self.gap, hh + self.gap])
            ubx.extend([cw - hw - self.gap, ch - hh - self.gap])
            x0.extend([float(pos[i, 0]), float(pos[i, 1])])

        nlp = {"x": z, "f": obj, "g": ca.vertcat(*g) if g else ca.MX.zeros(0, 1)}
        opts = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.ipopt_max_iter,
            "ipopt.tol": 1e-5,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 8,
            "ipopt.linear_solver": "mumps",
        }
        try:
            solver = ca.nlpsol("solver", "ipopt", nlp, opts)
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        except Exception:
            return None

        out = pos.copy()
        zval = np.asarray(sol["x"], dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(zval)):
            return None
        for k, i in enumerate(movable_idx):
            out[i, 0] = zval[2 * k]
            out[i, 1] = zval[2 * k + 1]
        return out

    def _build_wire_terms(
        self, benchmark: Benchmark, placement: torch.Tensor, movable_idx: Sequence[int]
    ) -> List[Tuple[int, int | None, float, float, float]]:
        movable_set = set(movable_idx)
        hard_n = benchmark.num_hard_macros
        macro_pos = placement.cpu().numpy().astype(np.float64)
        port_pos = benchmark.port_positions.cpu().numpy().astype(np.float64)
        terms: Dict[Tuple[int, int | None, float, float], float] = {}

        def add(i: int, j: int | None, tx: float, ty: float, w: float) -> None:
            if j is not None and j < i:
                i, j = j, i
            key = (i, j, round(float(tx), 6), round(float(ty), 6))
            terms[key] = terms.get(key, 0.0) + float(w)

        for nodes_t in benchmark.net_nodes:
            nodes = [int(v) for v in nodes_t.tolist()]
            vars_in = [u for u in nodes if u in movable_set]
            if not vars_in:
                continue
            coords = []
            for u in nodes:
                if u < benchmark.num_macros:
                    coords.append((float(macro_pos[u, 0]), float(macro_pos[u, 1])))
                else:
                    p = u - benchmark.num_macros
                    if 0 <= p < port_pos.shape[0]:
                        coords.append((float(port_pos[p, 0]), float(port_pos[p, 1])))
            if len(coords) < 2:
                continue
            w = 1.0 / max(1.0, float(len(nodes) - 1))

            hard_vars = [u for u in vars_in if u < hard_n]
            if len(hard_vars) <= 8:
                for a in range(len(hard_vars)):
                    for b in range(a + 1, len(hard_vars)):
                        add(hard_vars[a], hard_vars[b], 0.0, 0.0, w)

            cx = float(np.mean([c[0] for c in coords]))
            cy = float(np.mean([c[1] for c in coords]))
            for u in hard_vars:
                add(u, None, cx, cy, 0.35 * w)

        if not terms:
            return []
        vals = list(terms.items())
        weights = np.asarray([v for _, v in vals], dtype=np.float64)
        mean_w = float(np.mean(weights)) if weights.size else 1.0
        out = []
        for (i, j, tx, ty), w in vals:
            out.append((i, j, tx, ty, min(8.0, w / max(mean_w, 1e-9))))
        return out

    def _overlap_pairs(
        self, pos: np.ndarray, sizes: np.ndarray, margin: float = 0.0
    ) -> List[Tuple[int, int]]:
        n = pos.shape[0]
        hw = 0.5 * sizes[:, 0]
        hh = 0.5 * sizes[:, 1]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if (
                    abs(pos[i, 0] - pos[j, 0]) < hw[i] + hw[j] + margin
                    and abs(pos[i, 1] - pos[j, 1]) < hh[i] + hh[j] + margin
                ):
                    pairs.append((i, j))
        return pairs

    def _near_pairs(
        self, pos: np.ndarray, sizes: np.ndarray, margin: float
    ) -> List[Tuple[int, int]]:
        return self._overlap_pairs(pos, sizes, margin=margin)

    def _knn_edges(self, pos: np.ndarray, k: int) -> List[Tuple[int, int]]:
        n = pos.shape[0]
        if n <= 1:
            return []
        k = min(k, n - 1)
        edges = set()
        for i in range(n):
            d2 = np.sum((pos - pos[i]) ** 2, axis=1)
            order = np.argsort(d2)
            for j in order[1 : k + 1]:
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                edges.add((a, b))
        return sorted(edges)

    def _overlap_components(
        self, pairs: Sequence[Tuple[int, int]], n: int
    ) -> List[List[int]]:
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        touched = set()
        for i, j in pairs:
            union(i, j)
            touched.add(i)
            touched.add(j)

        comps: Dict[int, List[int]] = {}
        for i in touched:
            comps.setdefault(find(i), []).append(i)
        out = list(comps.values())
        out.sort(key=lambda c: (-len(c), min(c)))
        return out

    def _component_pair_cuts(
        self, pos: np.ndarray, sizes: np.ndarray, comp: Sequence[int]
    ) -> List[Tuple[int, int]]:
        comp_set = set(comp)
        pairs = []
        # Include true overlaps plus a tiny near margin against immediate
        # blockers, so a local solve does not repair one contact by creating
        # another just outside the component.
        for i, j in self._overlap_pairs(pos, sizes, margin=max(self.gap, 1e-4)):
            if i in comp_set or j in comp_set:
                pairs.append((i, j))
        return pairs

    def _clamp(
        self,
        pos: np.ndarray,
        movable: np.ndarray,
        hw: np.ndarray,
        hh: np.ndarray,
        cw: float,
        ch: float,
    ) -> None:
        for i in range(pos.shape[0]):
            if not movable[i]:
                continue
            pos[i, 0] = float(np.clip(pos[i, 0], hw[i] + self.gap, cw - hw[i] - self.gap))
            pos[i, 1] = float(np.clip(pos[i, 1], hh[i] + self.gap, ch - hh[i] - self.gap))

    def _legalize(
        self,
        pos: np.ndarray,
        sizes: np.ndarray,
        movable: np.ndarray,
        cw: float,
        ch: float,
        max_rounds: int,
    ) -> np.ndarray:
        out = pos.copy()
        n = out.shape[0]
        hw = 0.5 * sizes[:, 0]
        hh = 0.5 * sizes[:, 1]
        for _ in range(max_rounds):
            moved = False
            pairs = self._overlap_pairs(out, sizes, margin=self.gap)
            if not pairs:
                break
            pairs.sort(key=lambda p: (sizes[p[0], 0] * sizes[p[0], 1] + sizes[p[1], 0] * sizes[p[1], 1]))
            for i, j in pairs:
                if not movable[i] and not movable[j]:
                    continue
                dx = out[i, 0] - out[j, 0]
                dy = out[i, 1] - out[j, 1]
                ox = hw[i] + hw[j] + self.gap - abs(dx)
                oy = hh[i] + hh[j] + self.gap - abs(dy)
                if ox <= 0 or oy <= 0:
                    continue
                axis_x = ox <= oy
                si = 1.0 if (dx >= 0 or i > j) else -1.0
                sj = 1.0 if (dy >= 0 or i > j) else -1.0
                if movable[i] and movable[j]:
                    share_i = share_j = 0.5
                elif movable[i]:
                    share_i, share_j = 1.0, 0.0
                else:
                    share_i, share_j = 0.0, 1.0
                if axis_x:
                    if movable[i]:
                        out[i, 0] += share_i * si * ox
                    if movable[j]:
                        out[j, 0] -= share_j * si * ox
                else:
                    if movable[i]:
                        out[i, 1] += share_i * sj * oy
                    if movable[j]:
                        out[j, 1] -= share_j * sj * oy
                for k in (i, j):
                    if movable[k]:
                        out[k, 0] = float(np.clip(out[k, 0], hw[k] + self.gap, cw - hw[k] - self.gap))
                        out[k, 1] = float(np.clip(out[k, 1], hh[k] + self.gap, ch - hh[k] - self.gap))
                moved = True
            if not moved:
                break
        return out

    def _shelf_pack(self, placement: torch.Tensor, benchmark: Benchmark) -> None:
        sizes = benchmark.macro_sizes
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        idxs = torch.where(movable)[0].tolist()
        idxs.sort(key=lambda i: -float(sizes[i, 1].item()))
        x = 0.0
        y = 0.0
        row_h = 0.0
        gap = self.gap
        for i in idxs:
            w = float(sizes[i, 0].item())
            h = float(sizes[i, 1].item())
            if x + w > float(benchmark.canvas_width):
                x = 0.0
                y += row_h + gap
                row_h = 0.0
            if y + h > float(benchmark.canvas_height):
                continue
            placement[i, 0] = x + 0.5 * w
            placement[i, 1] = y + 0.5 * h
            x += w + gap
            row_h = max(row_h, h)
