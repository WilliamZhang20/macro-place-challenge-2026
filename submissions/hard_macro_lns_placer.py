"""
Hard-macro large-neighborhood search.

Starts from strict minimal repair, then makes topology-changing hard macro
reinsertions from crowded bins into cooler legal sites.  Soft clusters remain
where the benchmark put them; the goal of this branch is to test whether hard
macro topology changes can move the late IBM cases off their bad plateau.
"""

from __future__ import annotations

import random
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
from _routing_congestion import compute_rudy_map, normalize_map  # noqa: E402


class HardMacroLnsPlacer:
    def __init__(
        self,
        seed: int = 19,
        max_candidates: int = 4,
        max_seconds: float = 900.0,
        rudy_weight: float = 0.5,
    ):
        self.seed = int(seed)
        self.max_candidates = int(max_candidates)
        self.max_seconds = float(max_seconds)
        # Weight of normalized Rudy congestion in the bin-pressure surrogate.
        # 0.5 mirrors the proxy formula's 0.5 weight on congestion.
        self.rudy_weight = float(rudy_weight)
        self.repair = CasadiPlacer(soft_proxy_evals=0)
        self.baseline = CasadiPlacer()
        self._start = 0.0
        self._plc_cache = {}

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        self._start = time.monotonic()
        plc = self._load_plc(benchmark)
        base = self.baseline.place(benchmark)
        candidates = [base]

        # LNS was useful on some high-congestion mid/late IBM cases and mostly
        # wasted time elsewhere. Keep the first pass intentionally targeted.
        if not self._should_try_lns(benchmark):
            return base

        for cand in self._lns_candidates(base, benchmark):
            if self._expired():
                break
            cand = self.repair._dreamplace_repair(cand, benchmark)
            if self._valid(cand, benchmark):
                candidates.append(cand)
            if len(candidates) >= self.max_candidates + 1:
                break

        best = self._select(candidates, benchmark, plc)
        return best if best is not None else base

    def _should_try_lns(self, benchmark: Benchmark) -> bool:
        if not benchmark.name.startswith("ibm"):
            return False
        return benchmark.num_hard_macros >= 360

    def _lns_candidates(self, base: torch.Tensor, benchmark: Benchmark) -> List[torch.Tensor]:
        rng = random.Random(self.seed + sum(ord(c) for c in benchmark.name))
        hot = self._hot_hard_macros(base, benchmark)
        if not hot:
            return []

        n_hard = benchmark.num_hard_macros
        if n_hard >= 650:
            plans = [(24, 180), (40, 240)]
        elif n_hard >= 350:
            plans = [(16, 140), (28, 220), (44, 320)]
        else:
            plans = [(10, 100), (18, 160)]

        out = []
        for k in (8, 16, 28):
            out.append(self._swap_hot_cool(base, benchmark, hot, k, rng))

        for k, site_cap in plans:
            subset = [i for _, i in hot[: min(k, len(hot))]]
            out.append(self._reinsert_subset(base, benchmark, subset, site_cap, rng))

        # Randomized hot subsets help avoid one deterministic bad evacuation order.
        pool = [i for _, i in hot[: min(96, len(hot))]]
        for k, site_cap in plans[:1]:
            if len(pool) <= k:
                continue
            subset = rng.sample(pool, k)
            out.append(self._reinsert_subset(base, benchmark, subset, site_cap, rng))
        return out

    def _swap_hot_cool(
        self,
        base: torch.Tensor,
        benchmark: Benchmark,
        hot: Sequence[Tuple[float, int]],
        swap_count: int,
        rng: random.Random,
    ) -> torch.Tensor:
        out = base.clone()
        n_hard = benchmark.num_hard_macros
        pos = out[:n_hard].detach().cpu().numpy().astype(np.float64).copy()
        sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)
        targets = self._hard_net_targets(out, benchmark)

        hot_ids = [i for _, i in hot if not fixed[i]][: min(96, len(hot))]
        cool_ids = [i for _, i in reversed(hot) if not fixed[i]][: min(160, len(hot))]
        if not hot_ids or not cool_ids:
            return out

        used = set()
        swaps = []
        for i in hot_ids:
            if len(swaps) >= swap_count:
                break
            if i in used:
                continue
            ti = targets.get(i)
            if ti is None:
                continue
            ai = float(sizes[i, 0] * sizes[i, 1])
            best = None
            for j in cool_ids:
                if i == j or j in used:
                    continue
                aj = float(sizes[j, 0] * sizes[j, 1])
                ratio = max(ai, aj) / max(1e-9, min(ai, aj))
                if ratio > 2.4:
                    continue
                tj = targets.get(j, (float(pos[j, 0]), float(pos[j, 1])))
                # Prefer swaps that put each macro closer to its net centroid.
                before = abs(pos[i, 0] - ti[0]) + abs(pos[i, 1] - ti[1])
                before += abs(pos[j, 0] - tj[0]) + abs(pos[j, 1] - tj[1])
                after = abs(pos[j, 0] - ti[0]) + abs(pos[j, 1] - ti[1])
                after += abs(pos[i, 0] - tj[0]) + abs(pos[i, 1] - tj[1])
                score = after - before + 0.08 * ratio
                if best is None or score < best[0]:
                    best = (score, j)
            if best is None or best[0] >= 0.0:
                continue
            _, j = best
            swaps.append((i, j))
            used.add(i)
            used.add(j)

        # Small randomness: keep deterministic best swaps mostly, shuffle order
        # to vary legalization side-effects.
        rng.shuffle(swaps)
        for i, j in swaps:
            xi, yi = float(pos[i, 0]), float(pos[i, 1])
            xj, yj = float(pos[j, 0]), float(pos[j, 1])
            pos[i, 0], pos[i, 1] = self._clamped_center_for_size(
                xj, yj, sizes[i], benchmark
            )
            pos[j, 0], pos[j, 1] = self._clamped_center_for_size(
                xi, yi, sizes[j], benchmark
            )

        out[:n_hard] = torch.tensor(pos, dtype=out.dtype)
        if benchmark.macro_fixed.any():
            out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return out

    def _clamped_center_for_size(
        self, x: float, y: float, size: np.ndarray, benchmark: Benchmark
    ) -> Tuple[float, float]:
        hw = 0.5 * float(size[0])
        hh = 0.5 * float(size[1])
        return (
            float(np.clip(x, hw, float(benchmark.canvas_width) - hw)),
            float(np.clip(y, hh, float(benchmark.canvas_height) - hh)),
        )

    def _reinsert_subset(
        self,
        base: torch.Tensor,
        benchmark: Benchmark,
        subset: Sequence[int],
        site_cap: int,
        rng: random.Random,
    ) -> torch.Tensor:
        out = base.clone()
        n_hard = benchmark.num_hard_macros
        pos = out[:n_hard].detach().cpu().numpy().astype(np.float64).copy()
        sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)
        pressure = self._pressure_map(out, benchmark)
        sites = self._candidate_sites(pressure, benchmark, site_cap)
        targets = self._hard_net_targets(out, benchmark)

        removed = set(int(i) for i in subset if 0 <= int(i) < n_hard and not fixed[int(i)])
        if not removed:
            return out

        order = sorted(removed, key=lambda i: -float(sizes[i, 0] * sizes[i, 1]))
        placed_removed: List[int] = []
        obstacles = [i for i in range(n_hard) if i not in removed]

        for i in order:
            best = None
            hw = 0.5 * sizes[i, 0]
            hh = 0.5 * sizes[i, 1]
            for den, x, y in sites:
                x = float(np.clip(x, hw, float(benchmark.canvas_width) - hw))
                y = float(np.clip(y, hh, float(benchmark.canvas_height) - hh))
                if self._overlaps_any(i, x, y, pos, sizes, obstacles + placed_removed):
                    continue
                tx, ty = targets.get(i, (float(base[i, 0].item()), float(base[i, 1].item())))
                disp = abs(x - float(base[i, 0].item())) / max(1e-9, float(benchmark.canvas_width))
                disp += abs(y - float(base[i, 1].item())) / max(1e-9, float(benchmark.canvas_height))
                netd = abs(x - tx) / max(1e-9, float(benchmark.canvas_width))
                netd += abs(y - ty) / max(1e-9, float(benchmark.canvas_height))
                score = float(den) + 0.20 * disp + 0.35 * netd
                if best is None or score < best[0]:
                    best = (score, x, y)
            if best is None:
                placed_removed.append(i)
                continue
            _, x, y = best
            pos[i, 0] = x
            pos[i, 1] = y
            placed_removed.append(i)

            # Rotate sites so later macros see a little diversity among similarly good cells.
            if len(sites) > 8:
                shift = rng.randrange(0, min(8, len(sites)))
                sites = sites[shift:] + sites[:shift]

        out[:n_hard] = torch.tensor(pos, dtype=out.dtype)
        if benchmark.macro_fixed.any():
            out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return out

    def _hot_hard_macros(self, placement: torch.Tensor, benchmark: Benchmark) -> List[Tuple[float, int]]:
        pressure = self._pressure_map(placement, benchmark)
        rows, cols = pressure.shape
        bin_w = float(benchmark.canvas_width) / max(1, cols)
        bin_h = float(benchmark.canvas_height) / max(1, rows)
        sizes = benchmark.macro_sizes
        scored = []
        for i in range(benchmark.num_hard_macros):
            if bool(benchmark.macro_fixed[i].item()):
                continue
            x = float(placement[i, 0].item())
            y = float(placement[i, 1].item())
            c = int(np.clip(x / max(bin_w, 1e-9), 0, cols - 1))
            r = int(np.clip(y / max(bin_h, 1e-9), 0, rows - 1))
            area = float(sizes[i, 0].item() * sizes[i, 1].item())
            scored.append((float(pressure[r, c]) * area, i))
        scored.sort(reverse=True)
        return scored

    def _candidate_sites(
        self, pressure: np.ndarray, benchmark: Benchmark, site_cap: int
    ) -> List[Tuple[float, float, float]]:
        rows, cols = pressure.shape
        bin_w = float(benchmark.canvas_width) / max(1, cols)
        bin_h = float(benchmark.canvas_height) / max(1, rows)
        cells = []
        for r in range(rows):
            for c in range(cols):
                edge_penalty = 0.03 if r in (0, rows - 1) or c in (0, cols - 1) else 0.0
                cells.append((float(pressure[r, c]) + edge_penalty, (c + 0.5) * bin_w, (r + 0.5) * bin_h))
        cells.sort(key=lambda t: t[0])
        return cells[: max(16, min(site_cap, len(cells)))]

    def _pressure_map(self, placement: torch.Tensor, benchmark: Benchmark) -> np.ndarray:
        density = self._density_map(placement, benchmark)
        if self.rudy_weight <= 0.0:
            return density
        rudy = compute_rudy_map(placement, benchmark)
        return density + self.rudy_weight * normalize_map(rudy)

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
        others: Sequence[int],
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
