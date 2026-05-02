"""
Hard-only incremental coordinate descent.

Starts from the current strict repair baseline.  For a pressure-ranked subset
of hard macros, generate legal nearby/cool grid-cell moves, score them with a
cheap density/connectivity/displacement surrogate, and evaluate only a small
finalist set with the true proxy.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from macro_place.utils import validate_placement

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from casadi_placer import CasadiPlacer  # noqa: E402


class HardCoordDescentPlacer:
    def __init__(
        self,
        max_macros: int = 56,
        max_sites_per_macro: int = 72,
        true_proxy_finalists: int = 5,
        max_seconds: float = 900.0,
    ):
        self.max_macros = int(max_macros)
        self.max_sites_per_macro = int(max_sites_per_macro)
        self.true_proxy_finalists = int(true_proxy_finalists)
        self.max_seconds = float(max_seconds)
        self.baseline = CasadiPlacer(soft_proxy_evals=0)
        self.repair = CasadiPlacer(soft_proxy_evals=0)
        self._start = 0.0
        self._plc_cache = {}

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        self._start = time.monotonic()
        base = self.baseline.place(benchmark)
        if not benchmark.name.startswith("ibm"):
            return base

        plc = self._load_plc(benchmark)
        candidates = [base]
        move_candidates = self._ranked_single_moves(base, benchmark)

        for _, idx, x, y in move_candidates[: self.true_proxy_finalists]:
            if self._expired():
                break
            cand = base.clone()
            cand[idx, 0] = float(x)
            cand[idx, 1] = float(y)
            cand = self.repair._dreamplace_repair(cand, benchmark)
            if self._valid(cand, benchmark):
                candidates.append(cand)

        batch = self._batch_apply_moves(base, benchmark, move_candidates[: min(12, len(move_candidates))])
        if self._valid(batch, benchmark):
            candidates.append(batch)

        best = self._select(candidates, benchmark, plc)
        return best if best is not None else base

    def _ranked_single_moves(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> List[Tuple[float, int, float, float]]:
        n_hard = benchmark.num_hard_macros
        pos = placement[:n_hard].detach().cpu().numpy().astype(np.float64)
        sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)
        density = self._density_map(placement, benchmark)
        targets = self._hard_net_targets(placement, benchmark)
        rows, cols = density.shape
        bin_w = float(benchmark.canvas_width) / max(1, cols)
        bin_h = float(benchmark.canvas_height) / max(1, rows)

        macro_scores = []
        for i in range(n_hard):
            if fixed[i]:
                continue
            c = int(np.clip(pos[i, 0] / max(bin_w, 1e-9), 0, cols - 1))
            r = int(np.clip(pos[i, 1] / max(bin_h, 1e-9), 0, rows - 1))
            area = float(sizes[i, 0] * sizes[i, 1])
            macro_scores.append((float(density[r, c]) * area, i))
        macro_scores.sort(reverse=True)

        occupied = [i for i in range(n_hard)]
        moves: List[Tuple[float, int, float, float]] = []
        for _, i in macro_scores[: self.max_macros]:
            if self._expired():
                break
            sites = self._sites_for_macro(i, pos, sizes, density, benchmark, self.max_sites_per_macro)
            tx, ty = targets.get(i, (float(pos[i, 0]), float(pos[i, 1])))
            current_c = int(np.clip(pos[i, 0] / max(bin_w, 1e-9), 0, cols - 1))
            current_r = int(np.clip(pos[i, 1] / max(bin_h, 1e-9), 0, rows - 1))
            current_den = float(density[current_r, current_c])
            for den, x, y in sites:
                if abs(x - pos[i, 0]) < 1e-9 and abs(y - pos[i, 1]) < 1e-9:
                    continue
                if self._overlaps_any(i, x, y, pos, sizes, occupied):
                    continue
                disp = (abs(x - pos[i, 0]) / max(1e-9, float(benchmark.canvas_width)))
                disp += (abs(y - pos[i, 1]) / max(1e-9, float(benchmark.canvas_height)))
                netd = abs(x - tx) / max(1e-9, float(benchmark.canvas_width))
                netd += abs(y - ty) / max(1e-9, float(benchmark.canvas_height))
                # Negative density delta is good; keep displacement restrained.
                score = (float(den) - current_den) + 0.28 * disp + 0.22 * netd
                moves.append((score, i, x, y))
        moves.sort(key=lambda t: t[0])
        return moves

    def _batch_apply_moves(
        self,
        base: torch.Tensor,
        benchmark: Benchmark,
        moves: Sequence[Tuple[float, int, float, float]],
    ) -> torch.Tensor:
        out = base.clone()
        n_hard = benchmark.num_hard_macros
        pos = out[:n_hard].detach().cpu().numpy().astype(np.float64).copy()
        sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
        moved = set()
        for score, i, x, y in moves:
            if i in moved:
                continue
            if self._overlaps_any(i, x, y, pos, sizes, range(n_hard)):
                continue
            pos[i, 0] = x
            pos[i, 1] = y
            moved.add(i)
            if len(moved) >= 8:
                break
        out[:n_hard] = torch.tensor(pos, dtype=out.dtype)
        if benchmark.macro_fixed.any():
            out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        out = self.repair._dreamplace_repair(out, benchmark)
        return out

    def _sites_for_macro(
        self,
        i: int,
        pos: np.ndarray,
        sizes: np.ndarray,
        density: np.ndarray,
        benchmark: Benchmark,
        cap: int,
    ) -> List[Tuple[float, float, float]]:
        rows, cols = density.shape
        bin_w = float(benchmark.canvas_width) / max(1, cols)
        bin_h = float(benchmark.canvas_height) / max(1, rows)
        c0 = int(np.clip(pos[i, 0] / max(bin_w, 1e-9), 0, cols - 1))
        r0 = int(np.clip(pos[i, 1] / max(bin_h, 1e-9), 0, rows - 1))
        hw = 0.5 * sizes[i, 0]
        hh = 0.5 * sizes[i, 1]

        candidates = []
        radii = [1, 2, 4, 7, 11]
        for rad in radii:
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if abs(dr) + abs(dc) != rad:
                        continue
                    r = r0 + dr
                    c = c0 + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        x = float(np.clip((c + 0.5) * bin_w, hw, float(benchmark.canvas_width) - hw))
                        y = float(np.clip((r + 0.5) * bin_h, hh, float(benchmark.canvas_height) - hh))
                        candidates.append((float(density[r, c]) + 0.015 * rad, x, y))

        # Add global cool cells as escape hatches.
        cool = []
        for r in range(rows):
            for c in range(cols):
                edge = 0.04 if r in (0, rows - 1) or c in (0, cols - 1) else 0.0
                cool.append((float(density[r, c]) + edge, r, c))
        cool.sort(key=lambda t: t[0])
        for den, r, c in cool[: max(12, cap // 4)]:
            x = float(np.clip((c + 0.5) * bin_w, hw, float(benchmark.canvas_width) - hw))
            y = float(np.clip((r + 0.5) * bin_h, hh, float(benchmark.canvas_height) - hh))
            candidates.append((den, x, y))

        # Deduplicate by rounded center.
        seen = set()
        unique = []
        for den, x, y in sorted(candidates, key=lambda t: t[0]):
            key = (round(x, 5), round(y, 5))
            if key in seen:
                continue
            seen.add(key)
            unique.append((den, x, y))
            if len(unique) >= cap:
                break
        return unique

    def _density_map(self, placement: torch.Tensor, benchmark: Benchmark) -> np.ndarray:
        rows = int(benchmark.grid_rows)
        cols = int(benchmark.grid_cols)
        density = np.zeros((rows, cols), dtype=np.float64)
        bin_w = float(benchmark.canvas_width) / max(1, cols)
        bin_h = float(benchmark.canvas_height) / max(1, rows)
        bin_area = max(1e-9, bin_w * bin_h)
        pos = placement.detach().cpu().numpy()
        sizes = benchmark.macro_sizes.detach().cpu().numpy()
        for i in range(benchmark.num_macros):
            c = int(np.clip(pos[i, 0] / max(bin_w, 1e-9), 0, cols - 1))
            r = int(np.clip(pos[i, 1] / max(bin_h, 1e-9), 0, rows - 1))
            density[r, c] += float(sizes[i, 0] * sizes[i, 1]) / bin_area
        return density

    def _hard_net_targets(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> Dict[int, Tuple[float, float]]:
        pos = placement.detach().cpu().numpy()
        ports = benchmark.port_positions.detach().cpu().numpy()
        accum: Dict[int, List[float]] = {}
        for net in benchmark.net_nodes:
            nodes = [int(v) for v in net.tolist()]
            coords = []
            for u in nodes:
                if u < benchmark.num_macros:
                    coords.append((float(pos[u, 0]), float(pos[u, 1])))
                else:
                    p = u - benchmark.num_macros
                    if 0 <= p < ports.shape[0]:
                        coords.append((float(ports[p, 0]), float(ports[p, 1])))
            if len(coords) < 2:
                continue
            cx = float(np.mean([x for x, _ in coords]))
            cy = float(np.mean([y for _, y in coords]))
            for u in nodes:
                if 0 <= u < benchmark.num_hard_macros:
                    a = accum.setdefault(u, [0.0, 0.0, 0.0])
                    a[0] += cx
                    a[1] += cy
                    a[2] += 1.0
        return {i: (v[0] / v[2], v[1] / v[2]) for i, v in accum.items() if v[2] > 0}

    def _overlaps_any(
        self,
        i: int,
        x: float,
        y: float,
        pos: np.ndarray,
        sizes: np.ndarray,
        others,
    ) -> bool:
        hw = 0.5 * sizes[i, 0]
        hh = 0.5 * sizes[i, 1]
        for j in others:
            if j == i:
                continue
            if abs(x - pos[j, 0]) < hw + 0.5 * sizes[j, 0] + 1e-4 and abs(y - pos[j, 1]) < hh + 0.5 * sizes[j, 1] + 1e-4:
                return True
        return False

    def _select(self, candidates: Sequence[torch.Tensor], benchmark: Benchmark, plc):
        best = None
        best_cost = float("inf")
        for cand in candidates:
            if not self._valid(cand, benchmark):
                continue
            if plc is None:
                return cand
            try:
                cost = float(compute_proxy_cost(cand, benchmark, plc)["proxy_cost"])
            except Exception:
                continue
            if cost < best_cost:
                best_cost = cost
                best = cand
        return best

    def _valid(self, placement: torch.Tensor, benchmark: Benchmark) -> bool:
        ok, _ = validate_placement(placement, benchmark, check_overlaps=True)
        if not ok:
            return False
        return int(compute_overlap_metrics(placement, benchmark)["overlap_count"]) == 0

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

    def _expired(self) -> bool:
        return (time.monotonic() - self._start) > self.max_seconds
