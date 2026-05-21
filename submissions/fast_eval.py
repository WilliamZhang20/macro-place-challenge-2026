"""
FastEvaluator: ~100-500x faster proxy cost for SA inner loops.

Matches compute_proxy_cost() structure using vectorized PyTorch ops:
  WL     = sum(HPWL_net × weight) / ((W+H) × net_cnt)   [macro-center approx]
  Density = 0.5 × avg_top10%(cell_area_fraction)
  Congestion = abu(V+H routing demand, 0.05)              [RUDY approximation]

Usage:
  fast = FastEvaluator(benchmark, plc)
  r = fast.calibrate(benchmark, plc)           # 5 oracle calls, fits linear scale
  cost = fast.evaluate(positions)              # ~2ms vs ~500ms oracle

Incremental SA use:
  old_cost = fast.evaluate(positions)
  for each proposed move (macro_i, new_xy):
      delta = fast.delta_wl(positions, macro_i, new_xy)  # ~0.05ms
      if accept(old_cost + delta):
          positions[macro_i] = new_xy
          old_cost += delta
          if step % 500 == 0:
              old_cost = fast.evaluate(positions)  # re-sync density/cong
"""

import math
import random
import time
import numpy as np
import torch
from typing import List, Optional, Tuple

from macro_place.benchmark import Benchmark


class FastEvaluator:
    """Calibrated fast proxy cost evaluator for SA inner loops."""

    def __init__(self, benchmark: Benchmark, plc=None):
        self._n_macro = benchmark.num_macros
        self._n_hard  = benchmark.num_hard_macros
        self._cw      = benchmark.canvas_width
        self._ch      = benchmark.canvas_height

        self._build_wl_tables(benchmark)
        self._build_density_tables(benchmark)
        self._build_congestion_tables(benchmark)

        # Linear calibration: official ≈ _a * raw + _b
        self._a: float = 1.0
        self._b: float = 0.0

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_wl_tables(self, benchmark: Benchmark):
        net_nodes = benchmark.net_nodes
        total = sum(len(n) for n in net_nodes)

        flat_nodes = torch.empty(total, dtype=torch.long)
        net_ids    = torch.empty(total, dtype=torch.long)
        k = 0
        for i, nodes in enumerate(net_nodes):
            n = len(nodes)
            flat_nodes[k:k+n] = nodes
            net_ids[k:k+n]    = i
            k += n

        self._flat_nodes  = flat_nodes          # [total_pins]
        self._net_ids     = net_ids             # [total_pins]
        self._net_weights = benchmark.net_weights.float()
        self._num_nets    = benchmark.num_nets
        self._wl_norm     = (benchmark.canvas_width + benchmark.canvas_height) * benchmark.num_nets

        # Port positions (fixed): appended after macro positions
        if benchmark.port_positions is not None and len(benchmark.port_positions) > 0:
            self._port_pos = benchmark.port_positions.float()
        else:
            self._port_pos = None

        # Macro → list of net indices (for incremental delta)
        # net_nodes contains both macro indices [0, n_macro) and port indices [n_macro, ...)
        # Only track the macro ones.
        n_macro = benchmark.num_macros
        macro_to_nets: List[List[int]] = [[] for _ in range(n_macro)]
        for i, nodes in enumerate(net_nodes):
            for nd in nodes.tolist():
                if nd < n_macro:
                    macro_to_nets[nd].append(i)
        # Deduplicate
        self._macro_to_nets: List[List[int]] = [sorted(set(lst)) for lst in macro_to_nets]

        # Per-net node lists as numpy arrays (for fast incremental eval)
        self._net_node_arrays = [nodes.numpy() for nodes in net_nodes]

    def _build_density_tables(self, benchmark: Benchmark):
        self._macro_sizes = benchmark.macro_sizes.float()
        self._grid_rows   = benchmark.grid_rows
        self._grid_cols   = benchmark.grid_cols
        self._cell_w      = benchmark.canvas_width  / benchmark.grid_cols
        self._cell_h      = benchmark.canvas_height / benchmark.grid_rows
        self._cell_area   = self._cell_w * self._cell_h

        c = torch.arange(benchmark.grid_cols, dtype=torch.float32)
        r = torch.arange(benchmark.grid_rows, dtype=torch.float32)
        self._cx0 = c * self._cell_w          # [C]
        self._cx1 = (c + 1) * self._cell_w
        self._cy0 = r * self._cell_h          # [R]
        self._cy1 = (r + 1) * self._cell_h

    def _build_congestion_tables(self, benchmark: Benchmark):
        self._hroutes = benchmark.hroutes_per_micron
        self._vroutes = benchmark.vroutes_per_micron
        self._v_cap   = self._cell_w * benchmark.vroutes_per_micron
        self._h_cap   = self._cell_h * benchmark.hroutes_per_micron

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _all_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Concatenate macro positions + fixed port positions."""
        if self._port_pos is not None:
            return torch.cat([positions, self._port_pos], dim=0)
        return positions

    def _raw_wl(self, positions: torch.Tensor) -> torch.Tensor:
        all_pos  = self._all_positions(positions)
        pin_pos  = all_pos[self._flat_nodes]             # [total_pins, 2]
        x, y     = pin_pos[:, 0], pin_pos[:, 1]
        nids     = self._net_ids
        neg_inf  = torch.full((self._num_nets,), float('-inf'))
        pos_inf  = torch.full((self._num_nets,), float('inf'))
        x_max    = neg_inf.scatter_reduce(0, nids, x, reduce='amax', include_self=True)
        x_min    = pos_inf.scatter_reduce(0, nids, x, reduce='amin', include_self=True)
        y_max    = neg_inf.scatter_reduce(0, nids, y, reduce='amax', include_self=True)
        y_min    = pos_inf.scatter_reduce(0, nids, y, reduce='amin', include_self=True)
        hpwl     = (x_max - x_min + y_max - y_min) * self._net_weights
        return hpwl.sum() / self._wl_norm

    def _raw_density(self, positions: torch.Tensor) -> torch.Tensor:
        N       = len(positions)
        hw      = self._macro_sizes[:N, 0] / 2
        hh_     = self._macro_sizes[:N, 1] / 2
        x0      = positions[:, 0] - hw              # [N]
        x1      = positions[:, 0] + hw
        y0      = positions[:, 1] - hh_
        y1      = positions[:, 1] + hh_

        ov_x    = torch.clamp(
            torch.min(x1.unsqueeze(1), self._cx1.unsqueeze(0)) -
            torch.max(x0.unsqueeze(1), self._cx0.unsqueeze(0)),
            min=0.0
        )  # [N, C]
        ov_y    = torch.clamp(
            torch.min(y1.unsqueeze(1), self._cy1.unsqueeze(0)) -
            torch.max(y0.unsqueeze(1), self._cy0.unsqueeze(0)),
            min=0.0
        )  # [N, R]

        density = (ov_x.unsqueeze(1) * ov_y.unsqueeze(2)).sum(0).flatten() / self._cell_area  # [G]

        G     = self._grid_rows * self._grid_cols
        top_k = max(1, int(G * 0.1))
        occ   = density[density > 0]
        if len(occ) == 0:
            return torch.tensor(0.0)
        top_density = torch.topk(occ, min(top_k, len(occ))).values
        return 0.5 * top_density.mean()

    def _raw_congestion(self, positions: torch.Tensor) -> torch.Tensor:
        all_pos = self._all_positions(positions)
        pin_pos = all_pos[self._flat_nodes]
        x, y    = pin_pos[:, 0], pin_pos[:, 1]
        nids    = self._net_ids
        zeros   = torch.zeros(self._num_nets)

        nx_max  = zeros.clone().scatter_reduce(0, nids, x, reduce='amax', include_self=False)
        nx_min  = zeros.clone().scatter_reduce(0, nids, x, reduce='amin', include_self=False)
        ny_max  = zeros.clone().scatter_reduce(0, nids, y, reduce='amax', include_self=False)
        ny_min  = zeros.clone().scatter_reduce(0, nids, y, reduce='amin', include_self=False)

        R, C   = self._grid_rows, self._grid_cols
        cw, ch = self._cell_w, self._cell_h

        c0s = np.clip((nx_min.numpy() / cw).astype(np.int32), 0, C - 1)
        c1s = np.clip((nx_max.numpy() / cw).astype(np.int32), 0, C - 1)
        r0s = np.clip((ny_min.numpy() / ch).astype(np.int32), 0, R - 1)
        r1s = np.clip((ny_max.numpy() / ch).astype(np.int32), 0, R - 1)
        dxs = (nx_max - nx_min).numpy()
        dys = (ny_max - ny_min).numpy()
        ws  = self._net_weights.numpy()

        # Vectorized RUDY using 2D prefix-sum ("corner trick"):
        #   For each net: add uniform demand to bbox. Instead of a Python
        #   slice loop, apply ±value at 4 corners then 2D cumsum → O(N+RC).
        nc_arr = (c1s - c0s + 1).astype(np.float32)
        nr_arr = (r1s - r0s + 1).astype(np.float32)
        n_cells = nc_arr * nr_arr
        n_cells = np.maximum(n_cells, 1.0)

        v_per = (ws * dys / np.maximum(n_cells * ch, 1e-9)).astype(np.float32)
        h_per = (ws * dxs / np.maximum(n_cells * cw, 1e-9)).astype(np.float32)

        # Pad by 1 in each dim so c1+1/r1+1 never goes out of bounds
        V_c = np.zeros((R + 1, C + 1), dtype=np.float32)
        H_c = np.zeros((R + 1, C + 1), dtype=np.float32)

        np.add.at(V_c, (r0s,     c0s),       v_per)
        np.add.at(V_c, (r1s + 1, c0s),      -v_per)
        np.add.at(V_c, (r0s,     c1s + 1),  -v_per)
        np.add.at(V_c, (r1s + 1, c1s + 1),   v_per)

        np.add.at(H_c, (r0s,     c0s),       h_per)
        np.add.at(H_c, (r1s + 1, c0s),      -h_per)
        np.add.at(H_c, (r0s,     c1s + 1),  -h_per)
        np.add.at(H_c, (r1s + 1, c1s + 1),   h_per)

        V_cong = np.cumsum(np.cumsum(V_c[:R, :C], axis=0), axis=1)
        H_cong = np.cumsum(np.cumsum(H_c[:R, :C], axis=0), axis=1)

        V_norm   = torch.from_numpy(V_cong.flatten()) / self._v_cap
        H_norm   = torch.from_numpy(H_cong.flatten()) / self._h_cap
        combined = torch.cat([V_norm, H_norm])

        top_k = max(1, int(len(combined) * 0.05))
        return torch.topk(combined, top_k).values.mean()

    def congestion_map(self, positions: torch.Tensor) -> np.ndarray:
        """
        Return per-bin total routing congestion [R, C] (V+H, normalized).
        ~3ms. Used by SA to compute congestion gradient for guided moves.
        """
        all_pos = self._all_positions(positions)
        pin_pos = all_pos[self._flat_nodes]
        x, y    = pin_pos[:, 0], pin_pos[:, 1]
        nids    = self._net_ids
        zeros   = torch.zeros(self._num_nets)
        nx_max  = zeros.clone().scatter_reduce(0, nids, x, reduce='amax', include_self=False)
        nx_min  = zeros.clone().scatter_reduce(0, nids, x, reduce='amin', include_self=False)
        ny_max  = zeros.clone().scatter_reduce(0, nids, y, reduce='amax', include_self=False)
        ny_min  = zeros.clone().scatter_reduce(0, nids, y, reduce='amin', include_self=False)
        R, C = self._grid_rows, self._grid_cols
        cw, ch = self._cell_w, self._cell_h
        c0s = np.clip((nx_min.numpy() / cw).astype(np.int32), 0, C-1)
        c1s = np.clip((nx_max.numpy() / cw).astype(np.int32), 0, C-1)
        r0s = np.clip((ny_min.numpy() / ch).astype(np.int32), 0, R-1)
        r1s = np.clip((ny_max.numpy() / ch).astype(np.int32), 0, R-1)
        dxs, dys = (nx_max - nx_min).numpy(), (ny_max - ny_min).numpy()
        ws = self._net_weights.numpy()
        nc_arr = np.maximum((c1s - c0s + 1).astype(np.float32), 1.0)
        nr_arr = np.maximum((r1s - r0s + 1).astype(np.float32), 1.0)
        n_cells = nc_arr * nr_arr
        v_per = (ws * dys / np.maximum(n_cells * ch, 1e-9)).astype(np.float32)
        h_per = (ws * dxs / np.maximum(n_cells * cw, 1e-9)).astype(np.float32)
        V_c = np.zeros((R+1, C+1), dtype=np.float32)
        H_c = np.zeros((R+1, C+1), dtype=np.float32)
        for arr, v in [(V_c, v_per), (H_c, h_per)]:
            np.add.at(arr, (r0s,     c0s),       v)
            np.add.at(arr, (r1s + 1, c0s),      -v)
            np.add.at(arr, (r0s,     c1s + 1),  -v)
            np.add.at(arr, (r1s + 1, c1s + 1),   v)
        V_cong = np.cumsum(np.cumsum(V_c[:R, :C], axis=0), axis=1)
        H_cong = np.cumsum(np.cumsum(H_c[:R, :C], axis=0), axis=1)
        return V_cong / self._v_cap + H_cong / self._h_cap  # [R, C]

    def macro_congestion_score(self, positions: torch.Tensor, n_hard: int) -> np.ndarray:
        """
        Per-macro congestion contribution score [n_hard].
        Score = sum of congestion in bins the macro occupies.
        ~4ms. Use to select which macros to move for congestion relief.
        """
        cmap = self.congestion_map(positions)  # [R, C]
        scores = np.zeros(n_hard, dtype=np.float32)
        R, C = self._grid_rows, self._grid_cols
        cw, ch = self._cell_w, self._cell_h
        sizes = self._macro_sizes[:n_hard]
        pos_np = positions[:n_hard].numpy()
        hw = sizes[:, 0].numpy() / 2
        hh = sizes[:, 1].numpy() / 2
        for i in range(n_hard):
            c0 = max(0, int((pos_np[i, 0] - hw[i]) / cw))
            c1 = min(C-1, int((pos_np[i, 0] + hw[i]) / cw))
            r0 = max(0, int((pos_np[i, 1] - hh[i]) / ch))
            r1 = min(R-1, int((pos_np[i, 1] + hh[i]) / ch))
            scores[i] = float(cmap[r0:r1+1, c0:c1+1].sum())
        return scores

    def _raw_evaluate(self, positions: torch.Tensor) -> float:
        with torch.no_grad():
            wl   = self._raw_wl(positions)
            dens = self._raw_density(positions)
            cong = self._raw_congestion(positions)
            return (wl + 0.5 * dens + 0.5 * cong).item()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, positions: torch.Tensor) -> float:
        """
        Calibrated proxy cost estimate. ~2-5ms vs ~500ms oracle.
        Includes WL + density + congestion.
        """
        raw = self._raw_evaluate(positions)
        return self._a * raw + self._b

    def evaluate_wl_density(self, positions: torch.Tensor) -> float:
        """
        WL + density only (no congestion). ~0.5-1ms.
        Use in inner SA loops; recompute congestion every N steps.
        """
        with torch.no_grad():
            wl   = self._raw_wl(positions)
            dens = self._raw_density(positions)
            raw  = (wl + 0.5 * dens).item()
        return self._a * raw + self._b

    def delta_wl(
        self,
        positions: torch.Tensor,
        macro_i: int,
        new_x: float,
        new_y: float,
    ) -> float:
        """
        Incremental WL delta for single hard-macro move. ~0.05ms.
        Returns the change in *calibrated* WL cost.
        Does NOT update density or congestion (caller must track separately
        or re-call evaluate() periodically to re-sync).

        Usage:
            fast_cost = fast.evaluate(pos)
            ...
            d = fast.delta_wl(pos, i, nx, ny)
            if accept(fast_cost + d):
                pos[i] = [nx, ny]
                fast_cost += d
        """
        nets = self._macro_to_nets[macro_i]
        if not nets:
            return 0.0

        all_pos   = self._all_positions(positions).numpy()
        old_x     = float(positions[macro_i, 0])
        old_y     = float(positions[macro_i, 1])

        delta_hpwl = 0.0
        for ni in nets:
            nodes    = self._net_node_arrays[ni]
            xs       = all_pos[nodes, 0]
            ys       = all_pos[nodes, 1]
            old_hpwl = (xs.max() - xs.min() + ys.max() - ys.min())

            # Replace macro_i's contribution
            mask     = (nodes == macro_i)
            xs[mask] = new_x
            ys[mask] = new_y
            new_hpwl = (xs.max() - xs.min() + ys.max() - ys.min())

            w         = float(self._net_weights[ni])
            delta_hpwl += w * (new_hpwl - old_hpwl)

        raw_delta = delta_hpwl / self._wl_norm
        return self._a * raw_delta   # calibrated delta (offset cancels)

    def calibrate(
        self,
        benchmark: Benchmark,
        plc,
        n_samples: int = 8,
        seed: int = 0,
    ) -> float:
        """
        Fit linear scale fast → official using n_samples oracle evaluations.
        Returns Pearson r (target > 0.95).

        Mutates self._a and self._b.
        """
        from macro_place.objective import compute_proxy_cost

        rng = random.Random(seed)
        positions = benchmark.macro_positions.clone().float()
        movable   = (~benchmark.macro_fixed).numpy()
        cw, ch    = benchmark.canvas_width, benchmark.canvas_height
        hw_arr    = (benchmark.macro_sizes[:, 0] / 2).numpy()
        hh_arr    = (benchmark.macro_sizes[:, 1] / 2).numpy()

        fast_vals:     List[float] = []
        official_vals: List[float] = []

        sigmas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
        for idx in range(n_samples):
            sigma = sigmas[idx % len(sigmas)]
            pts   = positions.clone()
            if sigma > 0:
                for mi in range(benchmark.num_hard_macros):
                    if movable[mi]:
                        nx = float(pts[mi, 0]) + rng.gauss(0, cw * sigma)
                        ny = float(pts[mi, 1]) + rng.gauss(0, ch * sigma)
                        pts[mi, 0] = float(max(hw_arr[mi], min(cw - hw_arr[mi], nx)))
                        pts[mi, 1] = float(max(hh_arr[mi], min(ch - hh_arr[mi], ny)))

            fast_vals.append(self._raw_evaluate(pts))
            official_vals.append(
                compute_proxy_cost(pts, benchmark, plc)["proxy_cost"]
            )

        fv  = np.array(fast_vals, dtype=np.float64)
        ov  = np.array(official_vals, dtype=np.float64)
        A   = np.vstack([fv, np.ones(len(fv))]).T
        res = np.linalg.lstsq(A, ov, rcond=None)
        a, b = float(res[0][0]), float(res[0][1])

        # Pearson r between fitted and official
        pred = fv * a + b
        r    = float(np.corrcoef(pred, ov)[0, 1])

        self._a = a
        self._b = b
        print(
            f"  [fast_eval] calibrated: a={a:.4f} b={b:.4f} r={r:.4f} "
            f"(fast_range=[{fv.min():.4f},{fv.max():.4f}] "
            f"official_range=[{ov.min():.4f},{ov.max():.4f}])"
        )
        return r


# ── Standalone calibration / timing test ──────────────────────────────────────

if __name__ == "__main__":
    import sys, argparse
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", default="ibm01")
    args = parser.parse_args()

    from macro_place.loader    import load_benchmark
    from macro_place.objective import compute_proxy_cost

    base = f"external/MacroPlacement/Testcases/ICCAD04/{args.benchmark}"
    bench, plc = load_benchmark(
        f"{base}/netlist.pb.txt",
        f"{base}/initial.plc",
    )
    print(f"Loaded {bench}")

    fast = FastEvaluator(bench, plc)
    r    = fast.calibrate(bench, plc, n_samples=10)

    # Timing test
    pos = bench.macro_positions.clone().float()
    N   = 100

    t0 = time.time()
    for _ in range(N):
        compute_proxy_cost(pos, bench, plc)["proxy_cost"]
    oracle_ms = (time.time() - t0) / N * 1000

    t0 = time.time()
    for _ in range(N):
        fast.evaluate(pos)
    fast_ms = (time.time() - t0) / N * 1000

    t0 = time.time()
    for _ in range(N * 10):
        fast.delta_wl(pos, 0, float(pos[0, 0]) + 1.0, float(pos[0, 1]))
    delta_ms = (time.time() - t0) / (N * 10) * 1000

    print(f"\nTiming on {args.benchmark}:")
    print(f"  oracle:         {oracle_ms:.1f} ms/call")
    print(f"  fast.evaluate:  {fast_ms:.2f} ms/call   ({oracle_ms/fast_ms:.0f}x speedup)")
    print(f"  fast.delta_wl:  {delta_ms:.3f} ms/call  ({oracle_ms/delta_ms:.0f}x speedup)")
    print(f"  Calibration r = {r:.4f}")
