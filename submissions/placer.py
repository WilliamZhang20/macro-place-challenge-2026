"""
KoralPlacer â€” multi-seed Xplace GP + SA for the Partcl Macro Placement Challenge.

Pipeline:
  1. Multi-fidelity Xplace seed search (Phase A coarse â†’ B full GP â†’ C warm SA)
  2. Best GP start â†’ sequential SA (CD â†’ LNS â†’ Adam â†’ FD â†’ hSA â†’ oSA with oracle sync)

Usage:
  uv run evaluate submissions/koral/placer.py -b ibm01
  uv run evaluate submissions/koral/placer.py --all
"""

import sys
import os
import math
import random
import time
import tempfile
import numpy as np
import torch
from pathlib import Path

# Unbuffered output for live progress in nohup/pipe contexts; UTF-8 for Windows cp1252 compat
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

from macro_place.benchmark import Benchmark
from macro_place.objective  import compute_proxy_cost, compute_overlap_metrics

# Import bookshelf writer from same directory
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from bookshelf import write_bookshelf


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_plc(benchmark: Benchmark):
    """Reload PlacementCost for a benchmark â€” needed to call compute_proxy_cost."""
    try:
        from macro_place.loader import load_benchmark

        # Use forward-slash strings so plc_client_os.rsplit('/') works on Windows too
        base_ibm = "external/MacroPlacement/Testcases/ICCAD04/" + benchmark.name
        netlist = base_ibm + "/netlist.pb.txt"
        plc_file = base_ibm + "/initial.plc"
        if Path(netlist).exists():
            _, plc = load_benchmark(
                netlist,
                plc_file if Path(plc_file).exists() else None,
            )
            return plc

        # NG45 paths
        ng45_map = {
            "ariane133": "ariane133", "ariane136": "ariane136",
            "nvdla": "nvdla",         "mempool_tile": "mempool_tile",
        }
        ng45_name = ng45_map.get(benchmark.name)
        if ng45_name:
            base_ng = ("external/MacroPlacement/Flows/NanGate45/"
                       + ng45_name + "/netlist/output_CT_Grouping")
            ng_netlist = base_ng + "/netlist.pb.txt"
            ng_plc = base_ng + "/initial.plc"
            if Path(ng_netlist).exists():
                _, plc = load_benchmark(
                    ng_netlist,
                    ng_plc if Path(ng_plc).exists() else None,
                )
                return plc
    except Exception as e:
        print(f"[warn] Could not load PlacementCost for {benchmark.name}: {e}")
    return None


class _WorkerStream:
    """Prefix every output line with worker ID for interleaved log readability."""
    def __init__(self, prefix):
        self._prefix = prefix
        self._buf    = ""
        self._real   = sys.__stdout__
    def write(self, s):
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self._real.write(f"{self._prefix}{line}\n")
        self._real.flush()
    def flush(self): self._real.flush()
    def reconfigure(self, **kw): pass


def _polish_worker_queued(q, args):
    """Module-level wrapper so spawn context can pickle the target."""
    try:
        q.put(_polish_worker(args))
    except Exception as e:
        q.put((args[1], float('inf')))


def _pool_worker_loop(idx, task_q, result_q):
    """Long-lived SA worker: loops accepting tasks from successive benchmarks."""
    while True:
        task = task_q.get()
        if task is None:
            break
        bench_name, pos_np, sa_budget, seed = task
        try:
            result = _polish_worker((bench_name, pos_np, sa_budget, seed))
            result_q.put(result)
        except Exception as e:
            result_q.put((pos_np, float('inf')))


def _polish_worker(args):
    """Module-level worker for parallel SA (spawn context).

    Reloads benchmark + plc from disk so each worker has its own independent
    PlacementCost state. Returns (placement_np, proxy_cost).
    """
    bench_name, placement_np, sa_budget, seed = args
    worker_id = seed - 42
    sys.stdout = _WorkerStream(f"[w{worker_id}] ")
    try:
        from macro_place.loader import load_benchmark
        base_ibm = "external/MacroPlacement/Testcases/ICCAD04/" + bench_name
        netlist   = base_ibm + "/netlist.pb.txt"
        plc_file  = base_ibm + "/initial.plc"
        if not Path(netlist).exists():
            ng45_map = {
                "ariane133": "ariane133", "ariane136": "ariane136",
                "nvdla": "nvdla", "mempool_tile": "mempool_tile",
            }
            ng45_name = ng45_map.get(bench_name)
            if ng45_name:
                base_ng  = ("external/MacroPlacement/Flows/NanGate45/"
                            + ng45_name + "/netlist/output_CT_Grouping")
                netlist  = base_ng + "/netlist.pb.txt"
                plc_file = base_ng + "/initial.plc"
        bench, plc = load_benchmark(
            netlist, plc_file if Path(plc_file).exists() else None
        )
        placement = torch.from_numpy(placement_np).clone()
        placer    = KoralPlacer(sa_time_budget=sa_budget, seed=seed)
        result    = placer._cd_lns_polish(placement, bench, plc)
        cost      = compute_proxy_cost(result, bench, plc)["proxy_cost"]
        print(f"done, cost={cost:.4f}", flush=True)
        return result.numpy(), cost
    except Exception as e:
        import traceback
        print(f"ERROR: {e}\n{traceback.format_exc()}", flush=True)
        return placement_np, float('inf')


# â”€â”€ Time budget tracker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TimeBudget:
    """Tracks total wall-clock budget across all placement phases.

    Usage:
        budget = TimeBudget(3540)
        budget.log("Xplace done", f"best={cost:.4f}")
        sa_seconds = budget.allocate(0.50)  # 50% of remaining
    """

    def __init__(self, total_seconds: float):
        self._t0 = time.time()
        self.total = total_seconds

    def elapsed(self) -> float:
        return time.time() - self._t0

    def remaining(self) -> float:
        return max(0.0, self.total - self.elapsed())

    def pct(self) -> float:
        return min(100.0, self.elapsed() / self.total * 100.0)

    def allocate(self, fraction: float, max_seconds: float = None) -> float:
        alloc = self.remaining() * fraction
        if max_seconds is not None:
            alloc = min(alloc, max_seconds)
        return max(0.0, alloc)

    def log(self, phase: str, extra: str = ""):
        r = self.remaining()
        e = self.elapsed()
        parts = [f"  [budget] {phase}: elapsed={e:.0f}s  remaining={r:.0f}s  ({self.pct():.0f}% used)"]
        if extra:
            parts.append(f"  {extra}")
        print("".join(parts))


# â”€â”€ Main placer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class KoralPlacer:
    def __init__(
        self,
        target_density: float = 0.0,     # 0 = auto from utilization
        density_weight: float = 8e-5,
        gamma: float = 4.0,
        sa_time_budget: int = int(os.environ.get("KORAL_SA_BUDGET", "3480")),  
        seed: int = 42,
    ):
        self.target_density  = target_density
        self.density_weight  = density_weight
        self.gamma           = gamma
        self.sa_time_budget  = sa_time_budget
        self.seed            = seed

        # Pre-fork SA worker pool BEFORE any CUDA usage.
        # Fork here while CUDA is clean; workers loop accepting tasks from successive benchmarks.
        self._n_pool = 16 if sys.platform != 'win32' else 0
        self._pool_task_qs  = []
        self._pool_result_q = None
        self._pool_procs    = []

        if self._n_pool > 0:
            import multiprocessing as _mp
            _ctx = _mp.get_context('fork')
            self._pool_result_q = _ctx.Queue()
            self._pool_task_qs  = [_ctx.Queue() for _ in range(self._n_pool)]
            self._pool_procs    = [
                _ctx.Process(target=_pool_worker_loop,
                             args=(i, self._pool_task_qs[i], self._pool_result_q))
                for i in range(self._n_pool)
            ]
            for _p in self._pool_procs:
                _p.start()
            print(f"  [KoralPlacer] {self._n_pool} SA workers pre-forked (CUDA-clean at init)")

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Total wall-clock budget (KORAL_TOTAL_BUDGET overrides; default = SA_BUDGET + 120s for GP phases)
        _total_budget = float(os.environ.get(
            "KORAL_TOTAL_BUDGET",
            str(self.sa_time_budget + 120)
        ))
        budget = TimeBudget(_total_budget)
        budget.log("start", f"benchmark={benchmark.name}  total={_total_budget:.0f}s")

        plc = _load_plc(benchmark)

        # Start from CT positions (fallback when Xplace unavailable).
        # CT positions: already near-optimal from Circuit Training (good congestion/density).
        ct = benchmark.macro_positions.clone()
        # Full legalization for significant (>4nm) overlaps; micro-legalize for sub-4nm.
        # CT positions for ibm06 have 47 sub-4nm float64 overlaps. SA preserves these
        # because SA only rejects NEW candidate overlaps, never fixes existing ones.
        # micro_legalize here resolves CT overlaps before SA sees them (CT macros are
        # grid-placed, not at boundaries, so micro_legalize converges in 1-2 passes).
        if self._count_significant_overlaps(ct, benchmark, threshold=0.004) > 0:
            ct_legal = self._legalize_hard(ct, benchmark)
        else:
            ct_legal = ct
            # Note: pre-SA micro_legalize causes cascade amplification on dense benchmarks
            # (ibm06: 47 pairs chain-react, each 7nm push creates new overlaps â†’ 15Î¼m max
            # displacement â†’ oracle 1.87 vs CT's 1.66). SA preserves existing sub-4nm
            # overlaps but also won't place macros INTO overlapping positions (via _overlaps
            # filter). Post-SA micro_legalize handles any residual overlaps.

        placement = ct_legal  # default: CT positions
        xpl_tops  = None    # top-N Xplace seeds for parallel SA

        n_mv_hard = sum(1 for i in range(benchmark.num_hard_macros)
                        if not benchmark.macro_fixed[i])

        # Try Xplace multi-seed GP (routability-aware, requires CUDA).
        # KORAL_USE_CACHE=1 skips Xplace and loads saved tops for fast SA iteration.
        _cache_dir  = os.path.join(os.path.dirname(__file__), "placements")
        _xpl_cache  = os.path.join(_cache_dir, f"{benchmark.name}_xplace.pt")
        _use_cache  = os.environ.get("KORAL_USE_CACHE", "0") == "1" and os.path.exists(_xpl_cache)

        if self.sa_time_budget >= 300:
            ct_cost = compute_proxy_cost(ct_legal, benchmark, plc)["proxy_cost"] if plc else float('inf')
            best_ap_cost = ct_cost
            best_placement = ct_legal
            budget.log("CT baseline", f"ct_cost={ct_cost:.4f}")

            if _use_cache:
                _c = torch.load(_xpl_cache)
                xpl_tops = [torch.from_numpy(p) if isinstance(p, np.ndarray) else p for p in _c["xpl_tops"]]
                best_ap_cost = _c["best_cost"]
                best_placement = xpl_tops[0]
                print(f"  [cache] Loaded Xplace tops from {_xpl_cache} (best={best_ap_cost:.4f})")
                budget.log("Xplace skipped (cache)", f"best_gp={best_ap_cost:.4f}")
            else:
                _fast_early = None
                try:
                    from fast_eval import FastEvaluator as _FE
                    _fast_early = _FE(benchmark, plc)
                    _fe_r = _fast_early.calibrate(benchmark, plc, n_samples=5)
                    if _fe_r < 0.80:
                        print(f"  [fast_eval] early calibration r={_fe_r:.4f} too low, disabling")
                        _fast_early = None
                    else:
                        print(f"  [fast_eval] early calibration r={_fe_r:.4f} (from CT positions)")
                except Exception as e:
                    print(f"  [fast_eval] early calibration failed: {e}")

                try:
                    xpl_tops = self._run_xplace_multiseed(benchmark, budget, _fast_early, plc)
                    if xpl_tops:
                        xpl_cost = compute_proxy_cost(xpl_tops[0], benchmark, plc)["proxy_cost"]
                        if xpl_cost < best_ap_cost:
                            print(f"  [start] Xplace best={xpl_cost:.4f} < CT {best_ap_cost:.4f}"
                                  f"  top-{len(xpl_tops)} seeds for parallel SA")
                            best_ap_cost = xpl_cost
                            best_placement = xpl_tops[0]
                        else:
                            print(f"  [start] Xplace best={xpl_cost:.4f} >= CT {best_ap_cost:.4f}, using CT")
                            xpl_tops = None
                except Exception as e:
                    import traceback
                    print(f"  [start] Xplace multi-seed failed: {e}\n{traceback.format_exc()[-300:]}")
                budget.log("Xplace done", f"best_gp={best_ap_cost:.4f}")

                # Save Xplace tops for fast SA-only re-runs (KORAL_USE_CACHE=1).
                if xpl_tops:
                    try:
                        os.makedirs(_cache_dir, exist_ok=True)
                        torch.save({"xpl_tops": [t.numpy() for t in xpl_tops], "best_cost": best_ap_cost}, _xpl_cache)
                        print(f"  [cache] Saved Xplace tops → {_xpl_cache}")
                    except Exception as _ce:
                        print(f"  [cache] Save failed: {_ce}")

            placement = best_placement

        if plc is not None and self.sa_time_budget > 0:
            sa_budget = max(60.0, budget.remaining() - 120.0)
            budget.log("SA start", f"sa_budget={sa_budget:.0f}s")

            _start_positions = [p.numpy() for p in xpl_tops] if xpl_tops else [placement.numpy()]
            n_actual = min(len(_start_positions), self._n_pool) if self._n_pool > 0 else 0

            if n_actual > 0:
                print(f"  [parallel-SA] {n_actual} workers, pool (CUDA-clean at init)")
                for i, pos_np in enumerate(_start_positions[:n_actual]):
                    self._pool_task_qs[i].put((benchmark.name, pos_np, sa_budget, self.seed + i))
                results = [self._pool_result_q.get() for _ in range(n_actual)]

                best_cost, best_np = float('inf'), None
                for res_np, cost in results:
                    if cost < best_cost:
                        best_cost, best_np = cost, res_np
                if best_np is not None:
                    placement = torch.from_numpy(best_np)
                print(f"  [parallel-SA] best={best_cost:.4f}")
            else:
                self.sa_time_budget = sa_budget
                placement = self._cd_lns_polish(placement, benchmark, plc, budget=budget)

        budget.log("done", f"benchmark={benchmark.name}")

        # Save final placement so results can be inspected / reused later.
        try:
            _save_dir = os.path.join(os.path.dirname(__file__), "placements")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{benchmark.name}_final.pt")
            torch.save({"positions": placement, "benchmark_name": benchmark.name}, _save_path)
            print(f"  [save] placement → {_save_path}")
        except Exception as _e:
            print(f"  [save] skipped: {_e}")

        return placement

    @staticmethod
    def _xplace_available():
        """Return (xplace_home, xplace_main) or (None, None) if unavailable."""
        import os
        if os.name == 'nt':
            return None, None  # Windows: no GPU, Git Bash converts paths
        xplace_home = os.environ.get('XPLACE_HOME', '/opt/xplace')
        xplace_main = os.path.join(xplace_home, 'main.py')
        if not os.path.exists(xplace_main):
            return None, None
        if not torch.cuda.is_available():
            return None, None
        return xplace_home, xplace_main

    def _xplace_parse_pl(self, pl_path: str, benchmark: Benchmark):
        """Parse a Bookshelf .pl file → positions tensor, or None on failure."""
        positions = benchmark.macro_positions.clone().float()
        macro_sizes = benchmark.macro_sizes.float()
        parsed = 0
        try:
            with open(pl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3 or not parts[0].startswith('n'):
                        continue
                    idx_str = parts[0][1:]
                    if not idx_str.isdigit():
                        continue
                    idx = int(idx_str)
                    if idx >= benchmark.num_macros or benchmark.macro_fixed[idx]:
                        continue
                    try:
                        xl_nm, yl_nm = float(parts[1]), float(parts[2])
                    except ValueError:
                        continue
                    w = float(macro_sizes[idx, 0]); h = float(macro_sizes[idx, 1])
                    cx = max(w/2, min(benchmark.canvas_width  - w/2, xl_nm/1000.0 + w/2.0))
                    cy = max(h/2, min(benchmark.canvas_height - h/2, yl_nm/1000.0 + h/2.0))
                    positions[idx, 0] = cx; positions[idx, 1] = cy
                    parsed += 1
        except Exception as e:
            print(f"  [Xplace] pl parse error: {e}")
            return None
        if parsed < benchmark.num_hard_macros // 2:
            print(f"  [Xplace] too few positions ({parsed}), discarding")
            return None
        return positions

    def _xplace_run_one(self, benchmark: Benchmark, tmpdir: str, aux_path: str,
                        xplace_home: str, xplace_main: str, target_density: float,
                        seed: int, inner_iter: int,
                        noise_ratio: float = 0.025, stop_overflow: float = 0.05,
                        use_route_force: bool = False) -> "tuple[torch.Tensor | None, float]":
        """Run one Xplace GP. Returns (positions, elapsed_seconds) or (None, elapsed)."""
        import subprocess, glob as _glob
        result_dir  = os.path.join(tmpdir, f"results_s{seed}")
        exp_id      = f"xpl_s{seed}"
        output_dir  = "out"
        output_prefix = "placement"
        os.makedirs(os.path.join(result_dir, exp_id, output_dir), exist_ok=True)

        custom_path = (f"aux:{aux_path},benchmark:ispd2005,design_name:{benchmark.name}")
        cmd = [
            sys.executable, xplace_main,
            "--custom_path",           custom_path,
            "--load_from_raw",         "True",
            "--global_placement",      "True",
            "--legalization",          "False",
            "--detail_placement",      "False",
            "--write_placement",       "True",
            "--write_global_placement","True",
            "--inner_iter",            str(inner_iter),
            "--use_filler",            "False",
            "--noise_ratio",           str(noise_ratio),
            "--target_density",        str(target_density),
            "--stop_overflow",         str(stop_overflow),
            "--mixed_size",            "True",
            "--gpu",                   "0",
            "--num_threads",           "8",
            "--seed",                  str(seed),
            "--deterministic",         "True",
            "--use_route_force",       "True" if use_route_force else "False",
            "--result_dir",            result_dir,
            "--exp_id",                exp_id,
            "--output_dir",            output_dir,
            "--output_prefix",         output_prefix,
            "--log_name",              "xplace.log",
            "--verbose_cpp_log",       "False",
            "--cpp_log_level",         "2",
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=xplace_home,
                env={**os.environ, "PYTHONPATH": f"{xplace_home}:{os.environ.get('PYTHONPATH', '')}"}
            )
            elapsed = time.time() - t0
            if proc.returncode != 0:
                tail = (proc.stdout or "")[-400:] + (proc.stderr or "")[-200:]
                print(f"  [Xplace] seed={seed} failed rc={proc.returncode} in {elapsed:.0f}s: {tail[-100:]}")
                return None, elapsed
        except Exception as e:
            return None, time.time() - t0

        elapsed = time.time() - t0
        pl_path = os.path.join(result_dir, exp_id, output_dir,
                               f"{output_prefix}_{benchmark.name}_gp.pl")
        if not os.path.exists(pl_path):
            candidates = _glob.glob(os.path.join(result_dir, "**", "*_gp.pl"), recursive=True)
            if candidates:
                pl_path = candidates[0]
            else:
                return None, elapsed

        positions = self._xplace_parse_pl(pl_path, benchmark)
        return positions, elapsed

    def _run_warm_sa(self, pos: torch.Tensor, benchmark: Benchmark, plc,
                     budget_seconds: float = 300.0) -> "tuple[torch.Tensor, float]":
        """Short oracle SA to assess basin quality. Returns (best_pos, best_cost)."""
        if plc is None:
            c = float('inf')
            return pos, c
        pos_np = pos.numpy().copy() if isinstance(pos, torch.Tensor) else pos.copy()
        sizes  = benchmark.macro_sizes.numpy()
        hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
        n_hard = benchmark.num_hard_macros
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        movable      = benchmark.get_movable_mask().numpy()
        movable_hard = [i for i in range(n_hard) if movable[i]]
        if not movable_hard:
            cost = compute_proxy_cost(torch.tensor(pos_np, dtype=torch.float32), benchmark, plc)["proxy_cost"]
            return pos, cost

        # Precompute separation thresholds for overlap detection
        sep_x = hw[:n_hard, None] + hw[None, :n_hard]
        sep_y = hh[:n_hard, None] + hh[None, :n_hard]

        def _ov(i, cx, cy):
            dx = np.abs(cx - pos_np[:n_hard, 0]); dy = np.abs(cy - pos_np[:n_hard, 1])
            mask = (dx < sep_x[i]) & (dy < sep_y[i]); mask[i] = False
            return bool(mask.any())

        best_cost = compute_proxy_cost(torch.tensor(pos_np, dtype=torch.float32), benchmark, plc)["proxy_cost"]
        best_pos  = pos_np.copy()
        scale = max(cw, ch) * 0.06
        T     = best_cost * 0.005
        t0    = time.time()
        oracle_count = 1

        while time.time() - t0 < budget_seconds:
            t_frac    = min(1.0, (time.time() - t0) / budget_seconds)
            cur_scale = scale * max(0.25, 1.0 - 0.75 * t_frac)
            T_cur     = T * max(0.01, 1.0 - 0.99 * t_frac)
            ci = random.choice(movable_hard)
            cx = float(np.clip(pos_np[ci, 0] + random.gauss(0, cur_scale), hw[ci], cw - hw[ci]))
            cy = float(np.clip(pos_np[ci, 1] + random.gauss(0, cur_scale), hh[ci], ch - hh[ci]))
            if _ov(ci, cx, cy):
                continue
            old_x, old_y = float(pos_np[ci, 0]), float(pos_np[ci, 1])
            pos_np[ci, 0] = cx; pos_np[ci, 1] = cy
            cost = compute_proxy_cost(torch.tensor(pos_np, dtype=torch.float32), benchmark, plc)["proxy_cost"]
            oracle_count += 1
            delta = cost - best_cost
            if delta < 0 or (delta < T_cur and random.random() < math.exp(-delta / max(T_cur, 1e-12))):
                if cost < best_cost:
                    best_cost = cost; best_pos = pos_np.copy()
                    print(f"  [warmSA] t={time.time()-t0:.0f}s oracle={cost:.4f}")
            else:
                pos_np[ci, 0] = old_x; pos_np[ci, 1] = old_y

        print(f"  [warmSA] {oracle_count} oracle calls, best={best_cost:.4f} in {time.time()-t0:.0f}s")
        return torch.tensor(best_pos, dtype=torch.float32), best_cost

    def _run_xplace_multiseed(self, benchmark: Benchmark, budget: "TimeBudget",
                               fast, plc) -> "torch.Tensor | None":
        """
        Multi-fidelity Xplace seed search:
          Phase A: 100-200 coarse seeds (inner_iter=500, ~5s each) → rank by fast.evaluate
          Phase B: full GP on top-10 coarse winners (inner_iter=5000) → rank by oracle
          Phase C: warm oracle SA on top-3 → pick winner by post-warmup oracle cost
        Returns best positions tensor, or None if Xplace unavailable.
        """
        import shutil

        xplace_home, xplace_main = self._xplace_available()
        if xplace_home is None:
            print("  [Xplace] not available")
            return None

        macro_area   = (benchmark.macro_sizes[:, 0] * benchmark.macro_sizes[:, 1]).sum().item()
        target_density = min(0.95, macro_area / (benchmark.canvas_width * benchmark.canvas_height) + 0.15)

        tmpdir = tempfile.mkdtemp(prefix=f"koral_{benchmark.name}_xpl_")
        try:
            aux_path = write_bookshelf(benchmark, tmpdir, fix_soft=False)

            # ── Multi-seed GP sweep (inner_iter=5000, proven to converge) ────
            # Phase A (inner_iter=500) was dropped: partial convergence produces
            # hard-macro overlaps that legalization can't resolve, wasting the
            # entire phase budget with zero usable seeds.
            # 8000 iterations: enough for dense benchmarks to converge; stop_overflow=0.05
            # terminates early for small/easy benchmarks so extra iters cost nothing.
            # Adaptive noise schedule based on netlist connectivity.
            # Hyperconnected benchmarks (ibm18-like: ~92 nets/macro) benefit from
            # low noise because CT clustering is already good — high noise destroys it.
            # Normal benchmarks (ibm01: 24 nets/macro) need higher noise to escape the
            # CT local minimum. Threshold 60 has clear margin from all known IBM cases.
            _nets_per_macro = benchmark.num_nets / max(benchmark.num_hard_macros, 1)
            if _nets_per_macro > 60:
                _noise_sched = [0.02, 0.03, 0.05, 0.07, 0.10, 0.13]
            else:
                _noise_sched = [0.05, 0.10, 0.13, 0.15, 0.18, 0.20]
            MAX_SEED_SECONDS = 1800
            MIN_SA_SECONDS   = 400
            budget.log("Xplace seeds start", f"max={MAX_SEED_SECONDS}s  inner_iter=8000")
            seed_t0      = time.time()
            full_results = []  # list of (oracle_cost, seed, pos)

            for seed_idx in range(400):   # budget governs actual count
                elapsed_seeds = time.time() - seed_t0
                if elapsed_seeds >= MAX_SEED_SECONDS:
                    break
                if budget.remaining() < MIN_SA_SECONDS + 60:
                    break
                seed    = self.seed + seed_idx
                noise_r = _noise_sched[seed_idx % len(_noise_sched)]
                pos, elapsed = self._xplace_run_one(
                    benchmark, tmpdir, aux_path, xplace_home, xplace_main,
                    target_density, seed=seed, inner_iter=8000, noise_ratio=noise_r,
                    # stop_overflow=0.01, use_route_force=True are the new defaults
                )
                if pos is None:
                    if seed_idx == 0:
                        print(f"  [Xplace] first seed failed — aborting")
                        return None
                    continue
                pos = self._legalize_hard(pos, benchmark)
                if self._count_hard_overlaps_f32(pos, benchmark) > 0:
                    continue
                oc      = compute_proxy_cost(pos, benchmark, plc)["proxy_cost"] if plc else float('inf')
                is_best = (not full_results) or oc < full_results[0][0]
                full_results.append((oc, seed, pos))
                full_results.sort(key=lambda x: x[0])
                print(f"  [Xplace] seed={seed} noise={noise_r} oracle={oc:.4f} {elapsed:.0f}s"
                      f"  ({len(full_results)} ok)" + ("  ← new best!" if is_best else ""))

            if not full_results:
                print("  [Xplace] no valid seeds")
                return None
            top_positions = [pos for (_, _, pos) in full_results[:16]]
            budget.log("Xplace seeds done",
                       f"n_seeds={len(full_results)}  best={full_results[0][0]:.4f}  "
                       f"top={len(top_positions)}")
            return top_positions

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Legalization helpers ──────────────────────────────────────────────────


    @staticmethod
    def _count_hard_overlaps_f32(placement: torch.Tensor, benchmark: Benchmark) -> int:
        """Count overlapping hard macro pairs using float32 (matches evaluator precision)."""
        n = benchmark.num_hard_macros
        pos = placement[:n].numpy().astype(np.float32)
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float32)
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        dx = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
        dy = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
        ov = (sep_x - dx > 0) & (sep_y - dy > 0)
        np.fill_diagonal(ov, False)
        return int(ov.sum()) // 2

    @staticmethod
    def _count_significant_overlaps(placement: torch.Tensor, benchmark: Benchmark,
                                     threshold: float = 0.004,
                                     skip_fixed_fixed: bool = False) -> int:
        """Count hard macro pairs with overlap depth > threshold microns.
        TILOS ignores sub-4nm overlaps (float32 precision); use threshold=0.004 to match.
        skip_fixed_fixed=True: ignore fixed-fixed overlaps that no legalization can fix."""
        n = benchmark.num_hard_macros
        pos = placement[:n].numpy().astype(np.float32)
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float32)
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        dx = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
        dy = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
        ov = ((sep_x - dx) > threshold) & ((sep_y - dy) > threshold)
        np.fill_diagonal(ov, False)
        if skip_fixed_fixed:
            movable = benchmark.get_movable_mask()[:n].numpy().astype(bool)
            at_least_one_movable = movable[:, np.newaxis] | movable[np.newaxis, :]
            ov = ov & at_least_one_movable
        return int(ov.sum()) // 2

    def _legalize_hard(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """
        Fast O(NÂ² Ã— passes) push-based legalization of hard macros.
        Each pass scans all pairs; overlapping pairs are resolved by pushing
        the movable macro in the minimum-displacement direction (x or y).
        Converges in a few passes for typical Xplace/CT outputs.
        """
        GAP = 0.005  # 5nm gap to ensure float32 conversion doesn't re-introduce overlaps

        n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n].numpy().astype(bool)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        pos = placement[:n].numpy().copy().astype(np.float64)

        # Clamp all movable macros into bounds first
        for i in range(n):
            if movable[i]:
                pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

        # Exact separations for overlap detection; gap-inflated for resolution
        sep_x_exact = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y_exact = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        sep_x = sep_x_exact + GAP
        sep_y = sep_y_exact + GAP

        pass_idx = 0
        for pass_idx in range(2000):
            # Vectorized overlap detection (any positive physical overlap)
            dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
            dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
            is_ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
            np.fill_diagonal(is_ov, False)
            if not is_ov.any():
                break

            # Sequential (Gauss-Seidel) push â€” handles boundary clamping correctly:
            # when one macro is clamped, the full push goes to its unclamped neighbor.
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not movable[i] and not movable[j]:
                        continue
                    dx = abs(pos[i, 0] - pos[j, 0])
                    dy = abs(pos[i, 1] - pos[j, 1])
                    ox = sep_x[i, j] - dx
                    oy = sep_y[i, j] - dy
                    if ox <= 0 or oy <= 0:
                        continue
                    both = movable[i] and movable[j]
                    if ox <= oy:
                        need = ox
                        sign = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
                        if both:
                            xi_new = np.clip(pos[i, 0] + sign * need / 2, half_w[i], cw - half_w[i])
                            xj_new = np.clip(pos[j, 0] - sign * need / 2, half_w[j], cw - half_w[j])
                            got_i  = abs(xi_new - pos[i, 0])
                            got_j  = abs(xj_new - pos[j, 0])
                            if got_i + got_j < need - 1e-9:
                                xi_new = np.clip(pos[i, 0] + sign * (need - got_j), half_w[i], cw - half_w[i])
                                xj_new = np.clip(pos[j, 0] - sign * (need - got_i), half_w[j], cw - half_w[j])
                        else:
                            xi_new = np.clip(pos[i, 0] + sign * need, half_w[i], cw - half_w[i]) if movable[i] else pos[i, 0]
                            xj_new = np.clip(pos[j, 0] - sign * need, half_w[j], cw - half_w[j]) if movable[j] else pos[j, 0]
                        if abs(xi_new - pos[i, 0]) > 1e-9 or abs(xj_new - pos[j, 0]) > 1e-9:
                            changed = True
                        pos[i, 0] = xi_new; pos[j, 0] = xj_new
                    else:
                        need = oy
                        sign = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
                        if both:
                            yi_new = np.clip(pos[i, 1] + sign * need / 2, half_h[i], ch - half_h[i])
                            yj_new = np.clip(pos[j, 1] - sign * need / 2, half_h[j], ch - half_h[j])
                            got_i  = abs(yi_new - pos[i, 1])
                            got_j  = abs(yj_new - pos[j, 1])
                            if got_i + got_j < need - 1e-9:
                                yi_new = np.clip(pos[i, 1] + sign * (need - got_j), half_h[i], ch - half_h[i])
                                yj_new = np.clip(pos[j, 1] - sign * (need - got_i), half_h[j], ch - half_h[j])
                        else:
                            yi_new = np.clip(pos[i, 1] + sign * need, half_h[i], ch - half_h[i]) if movable[i] else pos[i, 1]
                            yj_new = np.clip(pos[j, 1] - sign * need, half_h[j], ch - half_h[j]) if movable[j] else pos[j, 1]
                        if abs(yi_new - pos[i, 1]) > 1e-9 or abs(yj_new - pos[j, 1]) > 1e-9:
                            changed = True
                        pos[i, 1] = yi_new; pos[j, 1] = yj_new
            if not changed:
                break

        # Diagonal push stage for any remaining stuck pairs:
        # push in BOTH x and y to break cycles that the min-direction approach can't escape
        for diag_pass in range(200):
            dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
            dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
            is_ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
            np.fill_diagonal(is_ov, False)
            if not is_ov.any():
                break
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not is_ov[i, j] or (not movable[i] and not movable[j]):
                        continue
                    dx = abs(pos[i, 0] - pos[j, 0])
                    dy = abs(pos[i, 1] - pos[j, 1])
                    ox = sep_x[i, j] - dx
                    oy = sep_y[i, j] - dy
                    if ox <= 0 and oy <= 0:
                        continue
                    both = movable[i] and movable[j]
                    sx = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
                    sy = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
                    fx = max(ox, 0) * sx; fy = max(oy, 0) * sy
                    s = 0.5 if both else 1.0
                    if movable[i]:
                        xi_new = np.clip(pos[i, 0] + fx * s, half_w[i], cw - half_w[i])
                        yi_new = np.clip(pos[i, 1] + fy * s, half_h[i], ch - half_h[i])
                        if abs(xi_new-pos[i,0]) > 1e-9 or abs(yi_new-pos[i,1]) > 1e-9:
                            changed = True
                        pos[i, 0] = xi_new; pos[i, 1] = yi_new
                    if movable[j]:
                        xj_new = np.clip(pos[j, 0] - fx * s, half_w[j], cw - half_w[j])
                        yj_new = np.clip(pos[j, 1] - fy * s, half_h[j], ch - half_h[j])
                        if abs(xj_new-pos[j,0]) > 1e-9 or abs(yj_new-pos[j,1]) > 1e-9:
                            changed = True
                        pos[j, 0] = xj_new; pos[j, 1] = yj_new
            if not changed:
                break

        # Report remaining overlaps for debugging
        dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
        dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
        ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
        np.fill_diagonal(ov, False)
        n_ov = int(ov.sum()) // 2
        if n_ov > 0:
            print(f"  [legalize] {pass_idx+1}+diag passes, {n_ov} hard macro overlaps remain")
        result = placement.clone()
        result[:n] = torch.tensor(pos, dtype=torch.float32)
        return result

    def _micro_legalize(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """
        Minimal-displacement legalization to eliminate ALL physical overlaps.
        Uses 3nm gap (vs 5nm in _legalize_hard) to minimize cost impact.
        Called before SA (to fix CT overlaps) and after SA (to fix any residual).
        """
        GAP = 0.003  # 3nm â€” survives float32 conversion
        n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n].numpy().astype(bool)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        pos = placement[:n].numpy().copy().astype(np.float64)
        sep_x_exact = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y_exact = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        sep_x = sep_x_exact + GAP
        sep_y = sep_y_exact + GAP

        for _ in range(2000):
            dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
            dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
            is_ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
            np.fill_diagonal(is_ov, False)
            if not is_ov.any():
                break
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not movable[i] and not movable[j]:
                        continue
                    dx = abs(pos[i, 0] - pos[j, 0])
                    dy = abs(pos[i, 1] - pos[j, 1])
                    ox = sep_x[i, j] - dx
                    oy = sep_y[i, j] - dy
                    if ox <= 0 or oy <= 0:
                        continue
                    both = movable[i] and movable[j]
                    old_i, old_j = pos[i].copy(), pos[j].copy()
                    # Resolve in minimum-overlap direction first.
                    # Only try the other direction if: (a) primary fails AND (b) other
                    # direction requires < 10nm movement. This fixes stuck pairs (boundary-
                    # clamped macros with tiny overlaps in both dims) without triggering
                    # large movements for ibm06-type pairs (small x-overlap, large y-extent).
                    # Primary direction (minimum overlap)
                    _TINY = 0.100  # 100nm: max secondary movement to avoid cascade
                    if ox <= oy:
                        need = ox; sign = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
                        if both:
                            xi_n = np.clip(pos[i, 0] + sign * need / 2, half_w[i], cw - half_w[i])
                            xj_n = np.clip(pos[j, 0] - sign * need / 2, half_w[j], cw - half_w[j])
                            got_i = abs(xi_n - pos[i, 0]); got_j = abs(xj_n - pos[j, 0])
                            if got_i + got_j < need - 1e-9:
                                xi_n = np.clip(pos[i, 0] + sign * (need - got_j), half_w[i], cw - half_w[i])
                                xj_n = np.clip(pos[j, 0] - sign * (need - got_i), half_w[j], cw - half_w[j])
                            pos[i, 0] = xi_n; pos[j, 0] = xj_n
                        elif movable[i]:
                            pos[i, 0] = np.clip(pos[i, 0] + sign * need, half_w[i], cw - half_w[i])
                        else:
                            pos[j, 0] = np.clip(pos[j, 0] - sign * need, half_w[j], cw - half_w[j])
                        # If still overlapping after primary (clamped in x), try y
                        # ONLY if the required y movement is tiny (< 10nm) to avoid cascade.
                        dx2 = abs(pos[i, 0] - pos[j, 0]); dy2 = abs(pos[i, 1] - pos[j, 1])
                        if (sep_x_exact[i, j] - dx2 > 0) and (sep_y_exact[i, j] - dy2 > 0):
                            oy2 = sep_y[i, j] - dy2
                            if 0 < oy2 <= _TINY:
                                sy = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
                                if both:
                                    pos[i, 1] = np.clip(pos[i, 1] + sy * oy2 / 2, half_h[i], ch - half_h[i])
                                    pos[j, 1] = np.clip(pos[j, 1] - sy * oy2 / 2, half_h[j], ch - half_h[j])
                                elif movable[i]:
                                    pos[i, 1] = np.clip(pos[i, 1] + sy * oy2, half_h[i], ch - half_h[i])
                                else:
                                    pos[j, 1] = np.clip(pos[j, 1] - sy * oy2, half_h[j], ch - half_h[j])
                    else:
                        need = oy; sign = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
                        if both:
                            yi_n = np.clip(pos[i, 1] + sign * need / 2, half_h[i], ch - half_h[i])
                            yj_n = np.clip(pos[j, 1] - sign * need / 2, half_h[j], ch - half_h[j])
                            got_i = abs(yi_n - pos[i, 1]); got_j = abs(yj_n - pos[j, 1])
                            if got_i + got_j < need - 1e-9:
                                yi_n = np.clip(pos[i, 1] + sign * (need - got_j), half_h[i], ch - half_h[i])
                                yj_n = np.clip(pos[j, 1] - sign * (need - got_i), half_h[j], ch - half_h[j])
                            pos[i, 1] = yi_n; pos[j, 1] = yj_n
                        elif movable[i]:
                            pos[i, 1] = np.clip(pos[i, 1] + sign * need, half_h[i], ch - half_h[i])
                        else:
                            pos[j, 1] = np.clip(pos[j, 1] - sign * need, half_h[j], ch - half_h[j])
                        # If still overlapping after primary (clamped in y), try x if tiny
                        dx2 = abs(pos[i, 0] - pos[j, 0]); dy2 = abs(pos[i, 1] - pos[j, 1])
                        if (sep_x_exact[i, j] - dx2 > 0) and (sep_y_exact[i, j] - dy2 > 0):
                            ox2 = sep_x[i, j] - dx2
                            if 0 < ox2 <= _TINY:
                                sx = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
                                if both:
                                    pos[i, 0] = np.clip(pos[i, 0] + sx * ox2 / 2, half_w[i], cw - half_w[i])
                                    pos[j, 0] = np.clip(pos[j, 0] - sx * ox2 / 2, half_w[j], cw - half_w[j])
                                elif movable[i]:
                                    pos[i, 0] = np.clip(pos[i, 0] + sx * ox2, half_w[i], cw - half_w[i])
                                else:
                                    pos[j, 0] = np.clip(pos[j, 0] - sx * ox2, half_w[j], cw - half_w[j])
                    if not (np.allclose(pos[i], old_i) and np.allclose(pos[j], old_j)):
                        changed = True
            if not changed:
                break

        # Diagonal push for any remaining cyclic stuck pairs (1000 passes)
        for _ in range(1000):
            dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
            dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
            is_ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
            np.fill_diagonal(is_ov, False)
            if not is_ov.any():
                break
            for i, j in zip(*np.where(is_ov)):
                if i >= j or (not movable[i] and not movable[j]):
                    continue
                dx_v = pos[i, 0] - pos[j, 0]; dy_v = pos[i, 1] - pos[j, 1]
                d = math.sqrt(dx_v * dx_v + dy_v * dy_v)
                fx, fy = (1.0, 0.0) if d < 1e-9 else (dx_v / d, dy_v / d)
                need_x = sep_x[i, j] - abs(dx_v)
                need_y = sep_y[i, j] - abs(dy_v)
                s = max(need_x, need_y) * 0.5
                if movable[i]:
                    pos[i, 0] = np.clip(pos[i, 0] + fx * s, half_w[i], cw - half_w[i])
                    pos[i, 1] = np.clip(pos[i, 1] + fy * s, half_h[i], ch - half_h[i])
                if movable[j]:
                    pos[j, 0] = np.clip(pos[j, 0] - fx * s, half_w[j], cw - half_w[j])
                    pos[j, 1] = np.clip(pos[j, 1] - fy * s, half_h[j], ch - half_h[j])

        result = placement.clone()
        result[:n] = torch.tensor(pos, dtype=torch.float32)
        return result

    # â”€â”€ HPWL data structures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_hpwl_data(self, benchmark: Benchmark, pos: np.ndarray):
        """
        Precompute connectivity data for fast incremental HPWL in SA.

        Returns:
            macro_nets: list[list[int]] â€” for each movable hard macro, net IDs it belongs to
            net_hard:   list[list[int]] â€” for each net, movable hard macro indices in it
            net_fbbox:  np.ndarray [num_nets, 4] â€” fixed-node bbox (min_x,max_x,min_y,max_y)
            net_wts:    np.ndarray [num_nets] â€” net weights
        """
        n_hard  = benchmark.num_hard_macros
        n_mac   = benchmark.num_macros
        prt_pos = benchmark.port_positions.numpy()  # [n_ports, 2]
        movable = benchmark.get_movable_mask().numpy()

        # Unified position array: macros (0..n_mac) then ports (n_mac..)
        all_pos = np.vstack([pos, prt_pos]) if prt_pos.shape[0] > 0 else pos

        num_nets = benchmark.num_nets
        macro_nets: list = [[] for _ in range(n_hard)]
        net_hard:   list = []
        # Fixed-node bbox for each net (from soft macros + ports + fixed hard macros)
        net_fbbox = np.full((num_nets, 4), fill_value=np.inf)
        net_fbbox[:, 1] = -np.inf
        net_fbbox[:, 3] = -np.inf

        for k in range(num_nets):
            nodes = benchmark.net_nodes[k].numpy()
            hard_in = []
            for n in nodes.tolist():
                if n < n_hard and movable[n]:
                    hard_in.append(n)
                    macro_nets[n].append(k)
                else:
                    # Fixed node â€” contribute to fixed bbox
                    px = float(all_pos[n, 0])
                    py = float(all_pos[n, 1])
                    if px < net_fbbox[k, 0]: net_fbbox[k, 0] = px
                    if px > net_fbbox[k, 1]: net_fbbox[k, 1] = px
                    if py < net_fbbox[k, 2]: net_fbbox[k, 2] = py
                    if py > net_fbbox[k, 3]: net_fbbox[k, 3] = py
            net_hard.append(hard_in)

        return macro_nets, net_hard, net_fbbox, benchmark.net_weights.numpy()

    @staticmethod
    def _hpwl_nets(net_ids, pos, net_hard, net_fbbox, net_wts):
        """Compute weighted HPWL for the given subset of nets."""
        total = 0.0
        for k in net_ids:
            fb   = net_fbbox[k]
            mnx  = fb[0]; mxx = fb[1]; mny = fb[2]; mxy = fb[3]
            for h in net_hard[k]:
                px = pos[h, 0]; py = pos[h, 1]
                if px < mnx: mnx = px
                if px > mxx: mxx = px
                if py < mny: mny = py
                if py > mxy: mxy = py
            if mnx <= mxx:
                total += net_wts[k] * ((mxx - mnx) + (mxy - mny))
        return total

    def _update_soft_macros(self, pos: np.ndarray, benchmark: Benchmark) -> np.ndarray:
        """
        Move each soft macro to the centroid of its connected hard macros and ports.
        One pass of force-directed placement for soft macros; fast (O(total_pins)).
        """
        n_hard = benchmark.num_hard_macros
        n_mac  = benchmark.num_macros
        port_pos = benchmark.port_positions.numpy()
        movable  = benchmark.get_movable_mask().numpy()
        cw, ch   = benchmark.canvas_width, benchmark.canvas_height
        half_w   = benchmark.macro_sizes[:, 0].numpy() / 2
        half_h   = benchmark.macro_sizes[:, 1].numpy() / 2

        # Unified position: macros[0..n_mac) then ports[n_mac..)
        all_pos = np.vstack([pos, port_pos]) if port_pos.shape[0] > 0 else pos

        # Accumulate weighted centroid for each soft macro from non-soft nodes
        n_soft  = n_mac - n_hard
        sum_x   = np.zeros(n_soft)
        sum_y   = np.zeros(n_soft)
        weight  = np.zeros(n_soft)

        net_wts = benchmark.net_weights.numpy()
        for k in range(benchmark.num_nets):
            nodes = benchmark.net_nodes[k].numpy().tolist()
            soft_in = [n for n in nodes if n_hard <= n < n_mac and movable[n]]
            other   = [n for n in nodes if n < n_hard or n >= n_mac]
            if not soft_in or not other:
                continue
            other_pos = all_pos[other]
            cx = float(other_pos[:, 0].mean())
            cy = float(other_pos[:, 1].mean())
            wt = float(net_wts[k])
            for m in soft_in:
                idx = m - n_hard
                sum_x[idx]  += wt * cx
                sum_y[idx]  += wt * cy
                weight[idx] += wt

        for s in range(n_soft):
            m = n_hard + s
            if not movable[m] or weight[s] == 0:
                continue
            pos[m, 0] = np.clip(sum_x[s] / weight[s], half_w[m], cw - half_w[m])
            pos[m, 1] = np.clip(sum_y[s] / weight[s], half_h[m], ch - half_h[m])

        return pos

    # â”€â”€ CD + LNS polish stage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _cd_lns_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        budget: "TimeBudget | None" = None,
    ) -> torch.Tensor:
        """
        Coordinate descent + LNS polish on hard macros.

        Inner loop (fast): for each movable macro, tries 20 Gaussian-shifted candidate
        positions + 3 orientation flips and accepts whichever minimises delta-HPWL.
        delta-HPWL is O(degree) using precomputed "fixed bbox" per (net, macro) pair.

        Outer loop: evaluates full proxy-cost oracle once per CD pass (~0.5s).
        Reverts positions to best-known if oracle doesn't confirm improvement.

        LNS: periodically tries all pairwise position-swaps within spatial clusters of
        k=8 macros; verifies with oracle.

        Final step: tries updating soft macros to centroids; reverts if no gain.
        """
        n_hard = benchmark.num_hard_macros
        n_mac  = benchmark.num_macros
        pp     = benchmark.port_positions.numpy()
        n_ports = pp.shape[0]
        cw, ch  = float(benchmark.canvas_width), float(benchmark.canvas_height)
        sizes   = benchmark.macro_sizes.numpy()
        hw, hh  = sizes[:, 0] / 2, sizes[:, 1] / 2
        movable = benchmark.get_movable_mask().numpy()
        movable_hard = [i for i in range(n_hard) if movable[i]]
        n_mv = len(movable_hard)
        nw   = benchmark.net_weights.numpy()
        poffs = benchmark.macro_pin_offsets  # list[Tensor[num_pins, 2]] per hard macro
        has_poffs = len(poffs) > 0

        if not movable_hard:
            return placement

        pos    = placement.numpy().copy()
        # Current orientation per hard macro: 0=N, 1=FN(flipX pins), 2=FS(flipY), 3=S(flipXY)
        ori    = np.zeros(n_hard, dtype=np.int8)

        # â”€â”€ Precompute pin-level net data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # all_cx/cy: positions of ALL nodes (hard macros, soft macros, ports)
        all_cx = np.zeros(n_mac + n_ports)
        all_cy = np.zeros(n_mac + n_ports)
        all_cx[:n_mac] = pos[:n_mac, 0]
        all_cy[:n_mac] = pos[:n_mac, 1]
        if n_ports > 0:
            all_cx[n_mac:] = pp[:, 0]
            all_cy[n_mac:] = pp[:, 1]

        # Precompute pin offsets for hard macros (base offsets before orientation)
        # pin_ox[i] = np.array of x-offsets for macro i's pins (one per pin)
        pin_ox = []
        pin_oy = []
        if has_poffs:
            for i in range(n_hard):
                if i < len(poffs) and poffs[i].numel() > 0:
                    pin_ox.append(poffs[i][:, 0].numpy())
                    pin_oy.append(poffs[i][:, 1].numpy())
                else:
                    pin_ox.append(np.zeros(1))
                    pin_oy.append(np.zeros(1))
        else:
            for i in range(n_hard):
                pin_ox.append(np.zeros(1))
                pin_oy.append(np.zeros(1))

        def _oriented_offsets(i, o):
            """Return (ox_arr, oy_arr) for macro i with orientation o."""
            ox = pin_ox[i].copy(); oy = pin_oy[i].copy()
            if o == 1: ox = -ox
            elif o == 2: oy = -oy
            elif o == 3: ox = -ox; oy = -oy
            return ox, oy

        # Build net pin list using net_pin_nodes if available, else net_nodes
        use_pinlevel = len(benchmark.net_pin_nodes) > 0 and has_poffs

        # net_entries[k] = list of (unified_node_idx, base_ox, base_oy, hard_macro_idx_or_-1)
        # For hard macros: hard_macro_idx = owner_idx; for others: -1
        net_entries = []
        macro_to_nets = [[] for _ in range(n_hard)]  # k for each movable hard macro

        for k in range(benchmark.num_nets):
            if use_pinlevel and k < len(benchmark.net_pin_nodes):
                pn = benchmark.net_pin_nodes[k]
                entries = []
                for row in pn.tolist():
                    owner, slot = row
                    if owner < n_hard:
                        ox = float(pin_ox[owner][slot]) if slot < len(pin_ox[owner]) else 0.0
                        oy = float(pin_oy[owner][slot]) if slot < len(pin_oy[owner]) else 0.0
                        entries.append((owner, ox, oy, owner))
                    elif owner < n_mac:
                        entries.append((owner, 0.0, 0.0, -1))
                    else:
                        entries.append((n_mac + (owner - n_mac), 0.0, 0.0, -1))
            else:
                nodes = benchmark.net_nodes[k].tolist()
                entries = []
                for owner in nodes:
                    if owner < n_mac:
                        entries.append((owner, 0.0, 0.0, owner if owner < n_hard else -1))
                    else:
                        entries.append((n_mac + (owner - n_mac), 0.0, 0.0, -1))

            if len(entries) < 2:
                net_entries.append(None)
                continue
            net_entries.append(entries)

            seen = set()
            for (_, _, _, mi) in entries:
                if mi >= 0 and movable[mi] and mi not in seen:
                    macro_to_nets[mi].append(k)
                    seen.add(mi)

        # Precompute "fixed bbox" for each (net k, hard macro i):
        # bbox of all pins in net k EXCLUDING macro i's pins.
        # Stored as flat arrays for speed.
        # fix_bbox[k][i] = (min_x, max_x, min_y, max_y)
        # We store per-macro dicts keyed by k.
        macro_fix_bbox = [{} for _ in range(n_hard)]  # macro_fix_bbox[i][k] = (x0,x1,y0,y1)

        def _compute_fix_bbox(k, excl_mi):
            entries = net_entries[k]
            if entries is None:
                return None
            xs, ys = [], []
            for (node, ox, oy, mi) in entries:
                if mi == excl_mi:
                    continue  # skip this macro's pins
                # Pin pos: for hard macro apply orientation, else use position
                if mi >= 0 and mi != excl_mi:
                    aox, aoy = _oriented_offsets(mi, ori[mi])
                    # Find which slot this is? We stored base offsets so just use ox,oy
                    # Apply orientation to stored ox,oy
                    aox_v = ox; aoy_v = oy
                    o = ori[mi]
                    if o == 1: aox_v = -aox_v
                    elif o == 2: aoy_v = -aoy_v
                    elif o == 3: aox_v = -aox_v; aoy_v = -aoy_v
                    xs.append(all_cx[node] + aox_v)
                    ys.append(all_cy[node] + aoy_v)
                else:
                    xs.append(all_cx[node] + ox)
                    ys.append(all_cy[node] + oy)
            if not xs:
                return (0.0, 0.0, 0.0, 0.0)
            return (min(xs), max(xs), min(ys), max(ys))

        # Build fix_bbox for all (macro, net) pairs
        for i in movable_hard:
            for k in macro_to_nets[i]:
                fb = _compute_fix_bbox(k, i)
                if fb is not None:
                    macro_fix_bbox[i][k] = fb

        def _rebuild_fix_bbox_for_nets(nets_to_update):
            """Recompute fix_bbox for all hard macros in the given nets."""
            for k in nets_to_update:
                entries = net_entries[k]
                if entries is None:
                    continue
                for (_, _, _, mi) in entries:
                    if mi >= 0 and movable[mi]:
                        fb = _compute_fix_bbox(k, mi)
                        if fb is not None:
                            macro_fix_bbox[mi][k] = fb

        def _delta_hpwl(i, new_cx, new_cy, new_o):
            """O(degree) delta-HPWL for moving macro i to (new_cx, new_cy) with orient new_o."""
            delta = 0.0
            old_cx, old_cy = all_cx[i], all_cy[i]
            for k in macro_to_nets[i]:
                fb = macro_fix_bbox[i].get(k)
                if fb is None:
                    continue
                fx0, fx1, fy0, fy1 = fb

                # Compute old and new bounding box contribution from macro i's pins
                entries = net_entries[k]
                old_min_x = old_max_x = old_min_y = old_max_y = None
                new_min_x = new_max_x = new_min_y = new_max_y = None

                for (node, ox, oy, mi) in entries:
                    if mi != i:
                        continue
                    # Old position with old orientation
                    aox_o, aoy_o = ox, oy
                    o_old = int(ori[i])
                    if o_old == 1: aox_o = -aox_o
                    elif o_old == 2: aoy_o = -aoy_o
                    elif o_old == 3: aox_o = -aox_o; aoy_o = -aoy_o
                    px_o, py_o = old_cx + aox_o, old_cy + aoy_o

                    # New position with new orientation
                    aox_n, aoy_n = ox, oy
                    if new_o == 1: aox_n = -aox_n
                    elif new_o == 2: aoy_n = -aoy_n
                    elif new_o == 3: aox_n = -aox_n; aoy_n = -aoy_n
                    px_n, py_n = new_cx + aox_n, new_cy + aoy_n

                    if old_min_x is None or px_o < old_min_x: old_min_x = px_o
                    if old_max_x is None or px_o > old_max_x: old_max_x = px_o
                    if old_min_y is None or py_o < old_min_y: old_min_y = py_o
                    if old_max_y is None or py_o > old_max_y: old_max_y = py_o

                    if new_min_x is None or px_n < new_min_x: new_min_x = px_n
                    if new_max_x is None or px_n > new_max_x: new_max_x = px_n
                    if new_min_y is None or py_n < new_min_y: new_min_y = py_n
                    if new_max_y is None or py_n > new_max_y: new_max_y = py_n

                if old_min_x is None:
                    continue  # macro i not in this net (shouldn't happen)

                old_wl = (max(fx1, old_max_x) - min(fx0, old_min_x) +
                          max(fy1, old_max_y) - min(fy0, old_min_y)) * nw[k]
                new_wl = (max(fx1, new_max_x) - min(fx0, new_min_x) +
                          max(fy1, new_max_y) - min(fy0, new_min_y)) * nw[k]
                delta += new_wl - old_wl
            return delta

        def _accept_move(i, new_cx, new_cy, new_o):
            """Apply move and update all_cx/cy and fix_bboxes for affected nets."""
            all_cx[i] = new_cx; all_cy[i] = new_cy
            pos[i, 0] = new_cx; pos[i, 1] = new_cy
            ori[i] = new_o
            # Rebuild fix_bbox for all macros in all nets containing i
            affected_nets = macro_to_nets[i]
            _rebuild_fix_bbox_for_nets(affected_nets)

        # Overlap check matrices
        _sg = 0.005
        sep_x = (sizes[:n_hard, 0:1] + sizes[:n_hard, 0:1].T) / 2 + _sg
        sep_y = (sizes[:n_hard, 1:2] + sizes[:n_hard, 1:2].T) / 2 + _sg

        def _overlaps(i, cx, cy):
            dx = np.abs(cx - pos[:n_hard, 0]); dy = np.abs(cy - pos[:n_hard, 1])
            mask = (dx < sep_x[i]) & (dy < sep_y[i]); mask[i] = False
            return bool(mask.any())

        def _rebuild_state(ref_pos, ref_ori):
            pos[:] = ref_pos; ori[:] = ref_ori
            all_cx[:n_mac] = pos[:n_mac, 0]; all_cy[:n_mac] = pos[:n_mac, 1]
            for i in movable_hard:
                for k in macro_to_nets[i]:
                    fb = _compute_fix_bbox(k, i)
                    if fb is not None:
                        macro_fix_bbox[i][k] = fb

        # â”€â”€ Initial oracle evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        init_result = compute_proxy_cost(
            torch.tensor(pos, dtype=torch.float32), benchmark, plc
        )
        init_cost = init_result["proxy_cost"]
        _wl_frac = init_result["wirelength_cost"] / max(init_cost, 1e-9)
        best_cost = init_cost
        best_pos  = pos.copy()
        best_ori  = ori.copy()
        print(f"  [CD] initial={init_cost:.4f} (wl_frac={_wl_frac:.2%})")

        # â”€â”€ Fast evaluator (300-4000x speedup for SA inner loops) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Calibrate once here; oracle stays synced from initial eval above.
        try:
            from fast_eval import FastEvaluator as _FE
            _fast = _FE(benchmark, plc)
            _fast_r = _fast.calibrate(benchmark, plc, n_samples=5)
            _fast_ok = _fast_r > 0.90
        except Exception as _fe_err:
            print(f"  [fast_eval] unavailable: {_fe_err}")
            _fast = None; _fast_ok = False

        # Build congestion-guided macro weights (static snapshot after initial eval).
        # plc is now synced; query per-cell routing congestion and map movable macros to cells.
        _cong_weights = None   # None = uniform random selection (high-cong bias)
        _inv_weights   = None  # None = uniform random selection (low-cong bias for swap targets)
        try:
            if hasattr(benchmark, 'hard_macro_indices') and len(benchmark.hard_macro_indices) > 0:
                _plc_ids = [int(benchmark.hard_macro_indices[i]) for i in movable_hard]
                _h_cong  = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float32)
                _v_cong  = np.array(plc.get_vertical_routing_congestion(), dtype=np.float32)
                _total   = _h_cong + _v_cong
                _cells   = [plc.get_grid_cell_of_node(pid) for pid in _plc_ids]
                _mc      = np.array([_total[c] if 0 <= c < len(_total) else 0.0 for c in _cells])
                _thr     = float(np.percentile(_mc, 50))
                _wts     = np.maximum(0.0, _mc - _thr).astype(np.float64)
                if _wts.sum() > 0:
                    _cong_weights = _wts / _wts.sum()
                    # Inverse weights: prefer low-congestion macros for swap targets
                    _inv_wts = 1.0 / (_cong_weights + 1e-9)
                    _inv_weights = (_inv_wts / _inv_wts.sum()).astype(np.float64)
        except Exception:
            pass

        deadline = time.time() + self.sa_time_budget
        oracle_calls = 1
        pass_num = 0
        no_improve_streak = 0

        def _budget_log(phase: str, extra: str = ""):
            if budget is not None:
                budget.log(phase, extra)
            else:
                remaining = max(0.0, deadline - time.time())
                print(f"  [budget] {phase}: remaining={remaining:.0f}s  {extra}")

        _budget_log("CD start", f"initial={best_cost:.4f}  wl_frac={_wl_frac:.1%}")

        # â”€â”€ Coordinate Descent â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        while time.time() < deadline:
            frac = max(0.0, (deadline - time.time()) / self.sa_time_budget)
            # Annealed move scale: start large, end small
            scale = max(cw, ch) * (0.08 * frac + 0.005)
            n_cands = 50

            order = random.sample(movable_hard, n_mv)
            n_accepted = 0

            for i in order:
                if time.time() >= deadline:
                    break

                best_delta = 0.0
                best_cx = pos[i, 0]; best_cy = pos[i, 1]; best_o = int(ori[i])
                found = False

                # Position candidates (Gaussian shifts)
                for _ in range(n_cands):
                    cx = float(np.clip(pos[i, 0] + random.gauss(0, scale), hw[i], cw - hw[i]))
                    cy = float(np.clip(pos[i, 1] + random.gauss(0, scale), hh[i], ch - hh[i]))
                    if _overlaps(i, cx, cy):
                        continue
                    d = _delta_hpwl(i, cx, cy, int(ori[i]))
                    if d < best_delta:
                        best_delta = d; best_cx = cx; best_cy = cy; best_o = int(ori[i]); found = True

                # Orientation flips at current position
                if has_poffs:
                    cur_cx, cur_cy = float(pos[i, 0]), float(pos[i, 1])
                    for o in (1, 2, 3):
                        d = _delta_hpwl(i, cur_cx, cur_cy, o)
                        if d < best_delta:
                            best_delta = d; best_cx = cur_cx; best_cy = cur_cy; best_o = o; found = True

                if found:
                    _accept_move(i, best_cx, best_cy, best_o)
                    n_accepted += 1

            if n_accepted == 0:
                break

            # â”€â”€ Oracle call once per pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            cost = compute_proxy_cost(
                torch.tensor(pos, dtype=torch.float32), benchmark, plc
            )["proxy_cost"]
            oracle_calls += 1

            if cost < best_cost:
                best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                print(f"  [CD] pass {pass_num}: {cost:.4f}  ({n_accepted} accepted)")
                no_improve_streak = 0
            else:
                _rebuild_state(best_pos, best_ori)
                no_improve_streak += 1
                if no_improve_streak >= 5:
                    break  # HPWL gradient not correlated with proxy; stop wasting oracle calls

            pass_num += 1

        _budget_log("LNS start", f"best={best_cost:.4f}  oracle_calls={oracle_calls}")

        # â”€â”€ LNS: pairwise swaps in spatial clusters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # _overlaps_excl: check if macro i at (cx,cy) overlaps anything except macro excl
        def _overlaps_excl(i, cx, cy, excl):
            dx = np.abs(cx - pos[:n_hard, 0]); dy = np.abs(cy - pos[:n_hard, 1])
            mask = (dx < sep_x[i]) & (dy < sep_y[i])
            mask[i] = False; mask[excl] = False
            return bool(mask.any())

        # _swap_delta_hpwl: exact delta-HPWL for swapping positions of i and j
        # (handles shared nets correctly by temporarily applying both changes)
        def _swap_delta_hpwl(i, j):
            old_ci_x, old_ci_y = float(pos[i, 0]), float(pos[i, 1])
            old_cj_x, old_cj_y = float(pos[j, 0]), float(pos[j, 1])
            affected = set(macro_to_nets[i]) | set(macro_to_nets[j])
            delta = 0.0
            for k in affected:
                entries = net_entries[k]
                if entries is None:
                    continue
                # Old HPWL for net k
                old_xmin = old_xmax = old_ymin = old_ymax = None
                new_xmin = new_xmax = new_ymin = new_ymax = None
                for (node, ox, oy, mi) in entries:
                    # Old pin position
                    if mi == i:
                        o = int(ori[i])
                        aox, aoy = ox, oy
                        if o==1: aox=-aox
                        elif o==2: aoy=-aoy
                        elif o==3: aox,aoy=-aox,-aoy
                        px_o, py_o = old_ci_x+aox, old_ci_y+aoy
                        px_n, py_n = old_cj_x+aox, old_cj_y+aoy  # swapped pos
                    elif mi == j:
                        o = int(ori[j])
                        aox, aoy = ox, oy
                        if o==1: aox=-aox
                        elif o==2: aoy=-aoy
                        elif o==3: aox,aoy=-aox,-aoy
                        px_o, py_o = old_cj_x+aox, old_cj_y+aoy
                        px_n, py_n = old_ci_x+aox, old_ci_y+aoy  # swapped pos
                    else:
                        aox_e, aoy_e = ox, oy
                        if mi >= 0:
                            o = int(ori[mi])
                            if o==1: aox_e=-aox_e
                            elif o==2: aoy_e=-aoy_e
                            elif o==3: aox_e,aoy_e=-aox_e,-aoy_e
                        px_o = px_n = all_cx[node]+aox_e; py_o = py_n = all_cy[node]+aoy_e
                    if old_xmin is None or px_o<old_xmin: old_xmin=px_o
                    if old_xmax is None or px_o>old_xmax: old_xmax=px_o
                    if old_ymin is None or py_o<old_ymin: old_ymin=py_o
                    if old_ymax is None or py_o>old_ymax: old_ymax=py_o
                    if new_xmin is None or px_n<new_xmin: new_xmin=px_n
                    if new_xmax is None or px_n>new_xmax: new_xmax=px_n
                    if new_ymin is None or py_n<new_ymin: new_ymin=py_n
                    if new_ymax is None or py_n>new_ymax: new_ymax=py_n
                if old_xmin is not None:
                    old_wl = (old_xmax-old_xmin+old_ymax-old_ymin)*nw[k]
                    new_wl = (new_xmax-new_xmin+new_ymax-new_ymin)*nw[k]
                    delta += new_wl - old_wl
            return delta

        lns_k = 12
        lns_improvements = 0
        lns_no_swap_streak = 0
        # Cap LNS time adaptively based on WL fraction of initial proxy cost.
        # Low-WL benchmarks (ibm06, wl_frac=3%): 15% cap (90s) â†’ oracle SA gets 460s+
        # to exploit improvements that oracle SA CAN find for such benchmarks.
        # High-WL benchmarks (ibm03+, wl_frac=20%+): up to 50% (300s) since LNS is
        # the primary optimizer for them. WL-dominated benchmarks that exit LNS
        # naturally before the cap are unaffected.
        lns_frac = max(0.15, min(0.50, _wl_frac * 3.0))
        lns_deadline = min(deadline - 2.0, time.time() + self.sa_time_budget * lns_frac)
        while time.time() < lns_deadline:
            center = random.choice(movable_hard)
            cx0, cy0 = pos[center, 0], pos[center, 1]
            dists = sorted([(abs(pos[j, 0]-cx0)+abs(pos[j, 1]-cy0), j)
                            for j in movable_hard if j != center])
            cluster = [center] + [idx for _, idx in dists[:lns_k - 1]]

            best_swap_delta = 0.0
            best_swap = None
            for ai in range(len(cluster)):
                for bi in range(ai + 1, len(cluster)):
                    i, j = cluster[ai], cluster[bi]
                    old_ci = (float(pos[i, 0]), float(pos[i, 1]))
                    old_cj = (float(pos[j, 0]), float(pos[j, 1]))
                    if _overlaps_excl(i, old_cj[0], old_cj[1], j) or \
                       _overlaps_excl(j, old_ci[0], old_ci[1], i):
                        continue
                    delta = _swap_delta_hpwl(i, j)
                    if delta < best_swap_delta:
                        best_swap_delta = delta; best_swap = (i, j, old_ci, old_cj)

            if best_swap is not None:
                i, j, old_ci, old_cj = best_swap
                _accept_move(i, old_cj[0], old_cj[1], int(ori[i]))
                _accept_move(j, old_ci[0], old_ci[1], int(ori[j]))
                lns_improvements += 1
                lns_no_swap_streak = 0
                # Verify with oracle every 10 accepted swaps
                if lns_improvements % 10 == 0 or time.time() > deadline - 10:
                    cost = compute_proxy_cost(
                        torch.tensor(pos, dtype=torch.float32), benchmark, plc
                    )["proxy_cost"]
                    oracle_calls += 1
                    if cost < best_cost:
                        best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                        print(f"  [LNS] {cost:.4f}")
                    else:
                        _rebuild_state(best_pos, best_ori)
            else:
                lns_no_swap_streak += 1
                if lns_no_swap_streak >= n_mv * 2:
                    break  # explored enough clusters; HPWL-improving swaps exhausted

        # â”€â”€ Adam WL+density spread: fast autograd pre-step before FD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # fast.evaluate() only has differentiable WL + density (congestion uses numpy â†’
        # zero grad). Adam on WL+density runs 500-1000 steps in 60s vs FD's 6-10 oracle
        # steps; FD still follows to capture the full congestion-sensitive gradient.
        # Oracle verifies every 50 Adam steps; reverts if no improvement.
        if _fast is not None and n_mv >= 1 and deadline - time.time() > 20:
            _rebuild_state(best_pos, best_ori)
            _adam_budget = max(10.0, min(60.0, (deadline - time.time()) * 0.08))  # 8% remaining, min 10s
            _hard_t = torch.tensor(pos[:n_hard].copy(), dtype=torch.float32, requires_grad=True)
            _soft_t = torch.tensor(pos[n_hard:].copy(), dtype=torch.float32)
            _adam_opt = torch.optim.Adam([_hard_t], lr=max(cw, ch) * 0.002)
            _adam_best = best_cost; _adam_best_np = pos.copy()
            _adam_improved = 0; _adam_step = 0; _adam_t0 = time.time()
            print(f"  [Adam] WL+density descent, budget={_adam_budget:.0f}s")
            while time.time() - _adam_t0 < _adam_budget:
                _full_t = torch.cat([_hard_t, _soft_t], dim=0)
                _loss = _fast._raw_wl(_full_t) + _fast._raw_density(_full_t[:n_hard])
                _adam_opt.zero_grad(); _loss.backward(); _adam_opt.step()
                with torch.no_grad():
                    for _ami in range(n_hard):
                        if not movable[_ami]:
                            _hard_t[_ami, 0] = float(pos[_ami, 0])
                            _hard_t[_ami, 1] = float(pos[_ami, 1])
                        else:
                            _hard_t[_ami, 0].clamp_(float(hw[_ami]), float(cw - hw[_ami]))
                            _hard_t[_ami, 1].clamp_(float(hh[_ami]), float(ch - hh[_ami]))
                if _adam_step % 50 == 49:
                    _trial = pos.copy(); _trial[:n_hard] = _hard_t.detach().numpy()
                    _trial_t = self._legalize_hard(
                        torch.tensor(_trial, dtype=torch.float32), benchmark)
                    if self._count_hard_overlaps_f32(_trial_t, benchmark) == 0:
                        _oc = compute_proxy_cost(_trial_t, benchmark, plc)["proxy_cost"]
                        oracle_calls += 1
                        if _oc < _adam_best:
                            _adam_best = _oc; _adam_best_np = _trial_t.numpy()
                            _adam_improved += 1
                            with torch.no_grad():
                                _hard_t[:] = torch.tensor(_adam_best_np[:n_hard])
                            print(f"  [Adam] step={_adam_step} oracle={_oc:.4f}")
                _adam_step += 1
            if _adam_best < best_cost:
                best_cost = _adam_best; best_pos = _adam_best_np
                pos[:] = best_pos
                all_cx[:n_mac] = pos[:n_mac, 0]; all_cy[:n_mac] = pos[:n_mac, 1]
            print(f"  [Adam] {_adam_step} steps, {_adam_improved} improvements, "
                  f"best={best_cost:.4f} in {time.time()-_adam_t0:.0f}s")
            _rebuild_state(best_pos, best_ori)

        _budget_log("FD start", f"best={best_cost:.4f}  oracle_calls={oracle_calls}")

        # â”€â”€ FD gradient descent (targets congestion/density, blind to HPWL gradient) â”€â”€
        # For each of the top-k congested macros: compute âˆ‚proxy/âˆ‚x and âˆ‚proxy/âˆ‚y via
        # 1-sided FD (probe at pos+Î´, compare to best_cost). Apply normalized gradient
        # step, legalize, verify. Repeats until no improvement or step size collapses.
        # Cost: 2*fd_k oracle calls per gradient step + 1 verify â‰ˆ 41 calls/step at k=20.
        oracle_time_est = max(0.5, (time.time() - (deadline - self.sa_time_budget)) / max(1, oracle_calls))
        fd_k = min(n_mv, 20)
        # Run FD for ALL benchmarks. For congestion-dominated (wl_frac < 4%), use larger
        # probe distance (5% canvas) so position changes are large enough to perturb
        # congestion grid bins and produce a non-zero gradient signal.
        if deadline - time.time() > oracle_time_est * (fd_k * 2 + 3) and n_mv >= 1:
            _rebuild_state(best_pos, best_ori)
            _fd_probe = 0.05 if _wl_frac < 0.04 else 0.02
            fd_delta_x = cw * _fd_probe
            fd_delta_y = ch * _fd_probe
            # Step size: scale with available room. For dense benchmarks (>75% utilization),
            # use smaller steps since macros are closely packed and large steps cause many
            # legalization passes. For sparse benchmarks, larger steps explore better.
            macro_area_frac = (benchmark.macro_sizes[:n_hard, 0] * benchmark.macro_sizes[:n_hard, 1]).sum().item() / (cw * ch)
            fd_step_size = max(cw, ch) * (0.02 if macro_area_frac > 0.75 else 0.05)
            fd_no_improve = 0; fd_improved = 0; fd_step = 0
            print(f"  [FD] starting, k={fd_k}, step={fd_step_size:.1f}, remaining={deadline-time.time():.0f}s")
            while time.time() < deadline - oracle_time_est * (fd_k * 2 + 4):
                if _cong_weights is not None:
                    top_k = [movable_hard[int(i)] for i in np.argsort(_cong_weights)[-fd_k:][::-1]]
                else:
                    top_k = movable_hard[:fd_k]
                f0 = best_cost
                grad_x = np.zeros(len(top_k))
                grad_y = np.zeros(len(top_k))
                for ii, mi in enumerate(top_k):
                    if time.time() >= deadline - oracle_time_est * ((len(top_k) - ii) * 2 + 4):
                        break
                    ox, oy = float(pos[mi, 0]), float(pos[mi, 1])
                    # x-gradient probe
                    nx = float(np.clip(ox + fd_delta_x, hw[mi], cw - hw[mi]))
                    if abs(nx - ox) > fd_delta_x * 0.1 and not _overlaps(mi, nx, oy):
                        pos[mi, 0] = nx
                        fp = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), benchmark, plc)["proxy_cost"]
                        oracle_calls += 1
                        grad_x[ii] = (fp - f0) / fd_delta_x
                        pos[mi, 0] = ox
                    # y-gradient probe
                    ny = float(np.clip(oy + fd_delta_y, hh[mi], ch - hh[mi]))
                    if abs(ny - oy) > fd_delta_y * 0.1 and not _overlaps(mi, ox, ny):
                        pos[mi, 1] = ny
                        fp = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), benchmark, plc)["proxy_cost"]
                        oracle_calls += 1
                        grad_y[ii] = (fp - f0) / fd_delta_y
                        pos[mi, 1] = oy
                gnorm = math.sqrt(float(np.sum(grad_x**2 + grad_y**2)))
                if gnorm < 1e-12:
                    break
                old_snap = pos.copy()
                for ii, mi in enumerate(top_k):
                    pos[mi, 0] = float(np.clip(pos[mi, 0] - fd_step_size * grad_x[ii] / gnorm, hw[mi], cw - hw[mi]))
                    pos[mi, 1] = float(np.clip(pos[mi, 1] - fd_step_size * grad_y[ii] / gnorm, hh[mi], ch - hh[mi]))
                pos_t = self._legalize_hard(torch.tensor(pos, dtype=torch.float32), benchmark)
                pos[:] = pos_t.numpy()
                all_cx[:n_mac] = pos[:n_mac, 0]; all_cy[:n_mac] = pos[:n_mac, 1]
                cost = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), benchmark, plc)["proxy_cost"]
                oracle_calls += 1
                if cost < best_cost:
                    best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                    fd_improved += 1; fd_no_improve = 0
                    fd_step_size = min(fd_step_size * 1.2, max(cw, ch) * 0.15)
                    print(f"  [FD] step {fd_step}: {cost:.4f}")
                    for i in movable_hard:
                        for k_ in macro_to_nets[i]:
                            fb_ = _compute_fix_bbox(k_, i)
                            if fb_ is not None:
                                macro_fix_bbox[i][k_] = fb_
                    if _cong_weights is not None:
                        try:
                            _h2 = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float32)
                            _v2 = np.array(plc.get_vertical_routing_congestion(), dtype=np.float32)
                            _mc2 = np.array([(_h2+_v2)[c] if 0<=c<len(_h2) else 0.0
                                             for c in [plc.get_grid_cell_of_node(p) for p in _plc_ids]], dtype=np.float32)
                            _thr2 = float(np.percentile(_mc2, 50))
                            _wts2 = np.maximum(0.0, _mc2 - _thr2).astype(np.float64)
                            if _wts2.sum() > 0:
                                _cong_weights = _wts2 / _wts2.sum()
                                _inv_w2 = 1.0 / (_cong_weights + 1e-9)
                                _inv_weights = (_inv_w2 / _inv_w2.sum()).astype(np.float64)
                        except Exception:
                            pass
                else:
                    pos[:] = old_snap
                    all_cx[:n_mac] = pos[:n_mac, 0]; all_cy[:n_mac] = pos[:n_mac, 1]
                    fd_no_improve += 1; fd_step_size *= 0.5
                    if fd_no_improve >= 4 or fd_step_size < max(cw, ch) * 0.002:
                        break
                fd_step += 1
            if fd_step > 0:
                print(f"  [FD] {fd_improved}/{fd_step} steps improved, best={best_cost:.4f}")
            _rebuild_state(best_pos, best_ori)

        # Re-calibrate FastEvaluator: CD/LNS/FD have moved positions far from CT.
        if _fast_ok and _fast is not None:
            _fast_r2 = _fast.calibrate(benchmark, plc, n_samples=8)
            _fast_ok = _fast_r2 > 0.90
            print(f"  [fast_eval] pre-hSA recalibrate: r={_fast_r2:.4f}")

        # For congestion-dominated benchmarks (wl_frac < 7%), use fast.evaluate()
        # as the hSA acceptance criterion instead of delta_hpwl. This includes
        # RUDY congestion (~50-70% of proxy) rather than WL-only guidance.
        _fast_hsa_cur = None
        _T_fhsa_start = 0.0; _T_fhsa_end = 0.0
        if _fast_ok and _fast is not None and _wl_frac < 0.07:
            _fast_hsa_cur = _fast.evaluate(torch.tensor(pos, dtype=torch.float32))
            # Calibrate temperature in fast-eval space from delta distribution (same method as hpwl_T).
            _fhsa_deltas = []
            for _ in range(min(200, n_mv * 15)):
                _i = random.choice(movable_hard)
                _cx2 = float(np.clip(pos[_i, 0] + random.gauss(0, max(cw, ch) * 0.08), hw[_i], cw - hw[_i]))
                _cy2 = float(np.clip(pos[_i, 1] + random.gauss(0, max(cw, ch) * 0.08), hh[_i], ch - hh[_i]))
                if not _overlaps(_i, _cx2, _cy2):
                    _saved = pos[_i].copy(); pos[_i, 0] = _cx2; pos[_i, 1] = _cy2
                    _fd = abs(_fast.evaluate(torch.tensor(pos, dtype=torch.float32)) - _fast_hsa_cur)
                    pos[_i] = _saved
                    if _fd > 0: _fhsa_deltas.append(_fd)
            if len(_fhsa_deltas) >= 10:
                _T_fhsa_start = float(np.percentile(_fhsa_deltas, 60)) * 2.0
                _T_fhsa_end   = float(np.percentile(_fhsa_deltas,  5)) * 0.05
            else:
                _T_fhsa_start = _fast_hsa_cur * 0.005
                _T_fhsa_end   = _fast_hsa_cur * 0.00005
            print(f"  [fast_eval] hSA mode: fast.evaluate() T={_T_fhsa_start:.5f}→{_T_fhsa_end:.6f} (wl_frac={_wl_frac:.1%})")

        _budget_log("hSA start", f"best={best_cost:.4f}  oracle_calls={oracle_calls}")

        #â”€â”€ HPWL-guided SA with periodic oracle checkpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Uses HPWL as fast surrogate (O(degree), ~0.01ms/call) for SA acceptance,
        # with proxy oracle checkpoint every HPWL_CKPT_N accepted moves (~0.5s/call).
        # Explores ~50Ã— more positions per oracle call budget vs pure oracle SA.
        # Move mix: 65% Gaussian translation, 15% rotation, 20% swap (dense only).
        # Adaptive HPWL_CKPT_N: frequent oracle for congestion-dominated benchmarks.
        remaining = deadline - time.time()
        oracle_time_est = max(0.5, (time.time() - (deadline - self.sa_time_budget)) / max(1, oracle_calls))
        # More frequent oracle for congestion-dominated benchmarks (HPWL â‰  proxy there).
        HPWL_CKPT_N = max(5, min(50, int(50 * _wl_frac * 1.5)))

        if remaining > oracle_time_est * 8:
            _rebuild_state(best_pos, best_ori)

            # Calibrate HPWL temperature from distribution of |delta_hpwl| on random valid moves
            sample_deltas = []
            for _ in range(min(400, n_mv * 20)):
                i = random.choice(movable_hard)
                cx = float(np.clip(pos[i, 0] + random.gauss(0, max(cw, ch) * 0.08), hw[i], cw - hw[i]))
                cy = float(np.clip(pos[i, 1] + random.gauss(0, max(cw, ch) * 0.08), hh[i], ch - hh[i]))
                if not _overlaps(i, cx, cy):
                    d = abs(_delta_hpwl(i, cx, cy, int(ori[i])))
                    if d > 0:
                        sample_deltas.append(d)

            if len(sample_deltas) >= 10:
                hpwl_T_start = float(np.percentile(sample_deltas, 60)) * 2.0
                hpwl_T_end   = float(np.percentile(sample_deltas,  5)) * 0.05
            else:
                hpwl_T_start = (cw + ch) * 0.3
                hpwl_T_end   = (cw + ch) * 0.003
            print(f"  [hSA] T={hpwl_T_start:.2f}->{hpwl_T_end:.4f}, ckpt={HPWL_CKPT_N}, remaining={remaining:.0f}s")

            accepted_since_ckpt = 0
            hsa_improved = 0; hsa_accepted = 0; hsa_step = 0; hsa_oracle = 0
            _hsa_gauss_tried = 0; _hsa_gauss_valid = 0; _hsa_swap_enabled = False
            t_hsa_start = time.time()
            t_hsa_end   = deadline - oracle_time_est * 2
            # Reheating state: if no oracle improvement in K calls, reheat temperature
            _reheat_no_improve = 0; _reheat_count = 0; _reheat_K = 20; _reheat_max = 4
            T_hsa_current_start = hpwl_T_start  # tracks current cycle start temp for reheat

            while time.time() < t_hsa_end:
                frac = (time.time() - t_hsa_start) / max(1e-9, t_hsa_end - t_hsa_start)
                T_hsa = T_hsa_current_start * math.exp(
                    math.log(max(hpwl_T_end, T_hsa_current_start * 1e-6) / T_hsa_current_start) * frac
                )
                # Multi-phase sigma: broad exploration â†’ focused exploitation â†’ fine-tuning.
                # Resets after each reheat since frac restarts from 0.
                if frac < 0.15:
                    scale_hsa = max(cw, ch) * 0.12
                elif frac < 0.75:
                    scale_hsa = max(cw, ch) * 0.05
                else:
                    scale_hsa = max(cw, ch) * 0.012

                # Pick macro: 70% congestion-guided, 30% random
                if _cong_weights is not None and random.random() < 0.7:
                    ci = movable_hard[int(np.random.choice(n_mv, p=_cong_weights))]
                else:
                    ci = random.choice(movable_hard)

                hsa_step += 1
                r = random.random()

                if r < 0.15:
                    # Rotation move: try next orientation (no translation)
                    new_o = (int(ori[ci]) + 1) % 4
                    if _fast_hsa_cur is not None:
                        _old_o2 = int(ori[ci])
                        _accept_move(ci, pos[ci, 0], pos[ci, 1], new_o)
                        _fc2 = _fast.evaluate(torch.tensor(pos, dtype=torch.float32))
                        _T_r = (_T_fhsa_start * math.exp(math.log(max(_T_fhsa_end, _T_fhsa_start * 1e-6) / _T_fhsa_start) * frac)
                                if _T_fhsa_start > 0 else 0.0)
                        _df2 = _fc2 - _fast_hsa_cur
                        if _df2 < 0 or (_T_r > 0 and random.random() < math.exp(-_df2 / _T_r)):
                            _fast_hsa_cur = _fc2; hsa_accepted += 1; accepted_since_ckpt += 1
                        else:
                            _accept_move(ci, pos[ci, 0], pos[ci, 1], _old_o2)
                    else:
                        delta_h = _delta_hpwl(ci, pos[ci, 0], pos[ci, 1], new_o)
                        if delta_h < 0 or (T_hsa > 0 and random.random() < math.exp(
                                -max(delta_h, 0.0) / T_hsa)):
                            _accept_move(ci, pos[ci, 0], pos[ci, 1], new_o)
                            hsa_accepted += 1; accepted_since_ckpt += 1

                elif r < 0.35 and _hsa_swap_enabled and n_mv >= 2:
                    # Swap move: useful for dense benchmarks where Gaussian has <3% valid rate
                    j_idx = random.randrange(n_mv); j = movable_hard[j_idx]
                    while j == ci: j_idx = (j_idx + 1) % n_mv; j = movable_hard[j_idx]
                    old_ci = (pos[ci, 0], pos[ci, 1]); old_cj = (pos[j, 0], pos[j, 1])
                    if not _overlaps_excl(ci, old_cj[0], old_cj[1], j) and \
                       not _overlaps_excl(j, old_ci[0], old_ci[1], ci):
                        delta_h = _swap_delta_hpwl(ci, j)
                        if delta_h < 0 or (T_hsa > 0 and random.random() < math.exp(
                                -max(delta_h, 0.0) / T_hsa)):
                            _accept_move(ci, old_cj[0], old_cj[1], int(ori[ci]))
                            _accept_move(j, old_ci[0], old_ci[1], int(ori[j]))
                            hsa_accepted += 1; accepted_since_ckpt += 1

                else:
                    # Gaussian translation move (primary)
                    cx = float(np.clip(pos[ci, 0] + random.gauss(0, scale_hsa), hw[ci], cw - hw[ci]))
                    cy = float(np.clip(pos[ci, 1] + random.gauss(0, scale_hsa), hh[ci], ch - hh[ci]))
                    _hsa_gauss_tried += 1
                    if _overlaps(ci, cx, cy):
                        # Track validity rate; enable swaps if Gaussian valid rate < 3%
                        if _hsa_gauss_tried % 100 == 0:
                            _hsa_swap_enabled = (_hsa_gauss_valid / _hsa_gauss_tried) < 0.03
                        continue
                    _hsa_gauss_valid += 1
                    if _fast_hsa_cur is not None:
                        # Congestion-dominated: Boltzmann acceptance in fast-eval space.
                        _old_cx, _old_cy, _old_o = float(pos[ci, 0]), float(pos[ci, 1]), int(ori[ci])
                        _accept_move(ci, cx, cy, int(ori[ci]))
                        _fc = _fast.evaluate(torch.tensor(pos, dtype=torch.float32))
                        _delta_f = _fc - _fast_hsa_cur
                        _T_fhsa = (_T_fhsa_start * math.exp(math.log(max(_T_fhsa_end, _T_fhsa_start * 1e-6) / _T_fhsa_start) * frac)
                                   if _T_fhsa_start > 0 else 0.0)
                        if _delta_f < 0 or (_T_fhsa > 0 and random.random() < math.exp(-_delta_f / _T_fhsa)):
                            _fast_hsa_cur = _fc
                            hsa_accepted += 1; accepted_since_ckpt += 1
                        else:
                            _accept_move(ci, _old_cx, _old_cy, _old_o)
                    else:
                        delta_h = _delta_hpwl(ci, cx, cy, int(ori[ci]))
                        if delta_h < 0 or (T_hsa > 0 and random.random() < math.exp(
                                -max(delta_h, 0.0) / T_hsa)):
                            _accept_move(ci, cx, cy, int(ori[ci]))
                            hsa_accepted += 1; accepted_since_ckpt += 1

                if accepted_since_ckpt >= HPWL_CKPT_N:
                    cost = compute_proxy_cost(
                        torch.tensor(pos, dtype=torch.float32), benchmark, plc
                    )["proxy_cost"]
                    oracle_calls += 1; hsa_oracle += 1
                    # Tighter revert for fast.evaluate() mode: temperature allows wider exploration,
                    # but oracle revert keeps state from drifting too far from best_pos.
                    _revert_thr = 1.03 if _wl_frac < 0.05 else (1.05 if _fast_hsa_cur is not None else 1.10)
                    if cost < best_cost:
                        best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                        hsa_improved += 1; _reheat_no_improve = 0
                        print(f"  [hSA] {cost:.4f}")
                    else:
                        _reheat_no_improve += 1
                        if cost > best_cost * _revert_thr:
                            _rebuild_state(best_pos, best_ori)
                    # Reheat on stagnation: if no improvement in _reheat_K oracle calls,
                    # reset temperature to escape local minima.
                    # Last reheat: add perturbation (displace ~5% of macros) for deeper escape.
                    if (_reheat_count >= _reheat_max and _reheat_no_improve >= _reheat_K
                            and deadline - time.time() > oracle_time_est * 30):
                        # All reheats exhausted + stagnating; exit hSA early so oracle SA tail gets time.
                        # Only exit if >30 oracle calls of time remain (worth running oSA).
                        break
                    if (_reheat_no_improve >= _reheat_K and _reheat_count < _reheat_max
                            and time.time() < t_hsa_end - oracle_time_est * 20):
                        if _reheat_count < _reheat_max - 1:
                            # Standard reheat: restore best and raise temperature
                            T_hsa_current_start = hpwl_T_start * 0.4
                            t_hsa_start = time.time()
                            _rebuild_state(best_pos, best_ori)
                        else:
                            # Final reheat: perturb a handful of macros for deeper exploration
                            _perturb = best_pos.copy(); _perturb_ori = best_ori.copy()
                            _n_perturb = max(2, n_mv // 15)
                            _pscale = max(cw, ch) * 0.18
                            _pmacros = random.sample(movable_hard, min(_n_perturb, n_mv))
                            for _pmi in _pmacros:
                                _pcx = float(np.clip(_perturb[_pmi, 0] + random.gauss(0, _pscale), hw[_pmi], cw - hw[_pmi]))
                                _pcy = float(np.clip(_perturb[_pmi, 1] + random.gauss(0, _pscale), hh[_pmi], ch - hh[_pmi]))
                                _perturb[_pmi, 0] = _pcx; _perturb[_pmi, 1] = _pcy
                            # Use micro_legalize (1nm gap) for perturbation â€” full legalize
                            # is too slow for dense benchmarks (ibm06 takes minutes).
                            _perturb_t = self._micro_legalize(torch.tensor(_perturb, dtype=torch.float32), benchmark)
                            _perturb = _perturb_t.numpy()
                            T_hsa_current_start = hpwl_T_start * 0.6
                            t_hsa_start = time.time()
                            _rebuild_state(_perturb, _perturb_ori)
                        _reheat_no_improve = 0; _reheat_count += 1
                        print(f"  [hSA] reheat #{_reheat_count} (stagnated {_reheat_K} oracle calls)")
                    # Refresh congestion map every 30 oracle calls
                    if _cong_weights is not None and hsa_oracle % 30 == 0:
                        try:
                            _h2 = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float32)
                            _v2 = np.array(plc.get_vertical_routing_congestion(), dtype=np.float32)
                            _t2 = _h2 + _v2
                            _mc2 = np.array([_t2[c] if 0 <= c < len(_t2) else 0.0
                                             for c in [plc.get_grid_cell_of_node(p) for p in _plc_ids]],
                                            dtype=np.float32)
                            _thr2 = float(np.percentile(_mc2, 50))
                            _wts2 = np.maximum(0.0, _mc2 - _thr2).astype(np.float64)
                            if _wts2.sum() > 0:
                                _cong_weights = _wts2 / _wts2.sum()
                        except Exception:
                            pass
                    # Periodic soft macro centroid update (every 100 oracle calls).
                    # Fast O(total_pins) pass to reposition soft macros toward connected hard macros.
                    if n_mac > n_hard and hsa_oracle % 100 == 0:
                        _pos_soft = self._update_soft_macros(best_pos.copy(), benchmark)
                        _cost_soft = compute_proxy_cost(
                            torch.tensor(_pos_soft, dtype=torch.float32), benchmark, plc
                        )["proxy_cost"]
                        oracle_calls += 1; hsa_oracle += 1
                        if _cost_soft < best_cost:
                            best_cost = _cost_soft; best_pos = _pos_soft
                            _rebuild_state(best_pos, best_ori)
                            _reheat_no_improve = 0
                    accepted_since_ckpt = 0

            # Final checkpoint for any pending accepted moves
            if accepted_since_ckpt > 0:
                cost = compute_proxy_cost(
                    torch.tensor(pos, dtype=torch.float32), benchmark, plc
                )["proxy_cost"]
                oracle_calls += 1; hsa_oracle += 1
                if cost < best_cost:
                    best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                    hsa_improved += 1
                    print(f"  [hSA] final {cost:.4f}")

            if hsa_oracle > 0:
                print(f"  [hSA] {hsa_improved} improved, {hsa_accepted}/{hsa_step} accepted, "
                      f"{hsa_oracle} oracle, swap={'on' if _hsa_swap_enabled else 'off'}")

        _budget_log("oSA start", f"best={best_cost:.4f}  oracle_calls={oracle_calls}")

        # â”€â”€ Oracle SA tail: direct proxy minimization with remaining time â”€â”€â”€â”€â”€
        # hSA uses HPWL surrogate which doesn't correlate for congestion-dominated
        # benchmarks. After hSA stagnates, use oracle (compute_proxy_cost) directly
        # for single-macro moves. Only helps when micro_legalize will succeed (no
        # legalize_hard needed): oracle SA can find positions with harder legalization
        # that look better by proxy but are actually worse after legalize_hard (+0.08 extra).
        _osa_remain = deadline - time.time() - oracle_time_est * 4
        _osa_pre_micro = self._micro_legalize(torch.tensor(best_pos, dtype=torch.float32), benchmark)
        _osa_micro_ok = compute_overlap_metrics(_osa_pre_micro, benchmark)["overlap_count"] == 0
        if _osa_remain > oracle_time_est * 5 and _osa_micro_ok:
            _pre_osa_best_pos = best_pos.copy(); _pre_osa_best_cost = best_cost  # save for revert
            _rebuild_state(best_pos, best_ori)
            # Adaptive scale: large early (broad exploration) â†’ shrink over time (exploitation).
            # Congestion-dominated benchmarks use a larger base scale (macros need bigger moves to
            # escape congested bins; 4% probe would miss bin boundaries).
            _osa_scale_base = max(cw, ch) * (0.08 if _wl_frac < 0.04 else 0.04)

            # â”€â”€ oSA uses fast evaluator when available (300-17000x speedup) â”€â”€
            # fast.evaluate: 383-763x faster than oracle (full WL+density+cong approx)
            # fast.delta_wl: 4000-17000x faster (WL-only incremental, for translations)
            # Fall back to oracle if fast evaluator is unavailable or miscalibrated.
            _use_fast_osa = _fast_ok and _fast is not None
            if _use_fast_osa:
                _pos_t = torch.tensor(pos, dtype=torch.float32)
                _fast_cur  = _fast.evaluate(_pos_t)
                _fast_best = _fast_cur
                _fast_best_pos = pos.copy()
                # Temperature in fast-eval scale.
                # 0.001 Ã— cost: accepts 91% of moves worsening by 0.1% (one typical step),
                # 39% of moves worsening by 0.1 Ã— cost (far worse). This is more selective
                # than 0.010 (which would accept 99% of typical worsening moves = random walk).
                _osa_T = _fast_cur * 0.001
                # For congestion-dominated benchmarks (wl_frac < 0.15): delta_wl captures only
                # ~6-15% of the cost signal; using it for translation leads to random-walk behavior.
                # Use full fast.evaluate for ALL moves on these benchmarks (383x speedup is still huge).
                # For WL-dominated (wl_frac >= 0.15): delta_wl captures majority of signal â†’ safe.
                _osa_use_delta_wl = (_wl_frac >= 0.15)
                _osa_sync_interval = 500 if _osa_use_delta_wl else 1  # sync=1 means always full evaluate
                _osa_delta_steps   = 0
                print(f"  [oSA] fast_eval active ({_osa_remain:.0f}s, fast_cur={_fast_cur:.4f}, "
                      f"mode={'delta_wl' if _osa_use_delta_wl else 'full_eval'})")
            else:
                _osa_T = best_cost * 0.010

            _osa_improved = 0; _osa_accepted = 0; _osa_tries = 0
            _osa_t0 = time.time(); _osa_deadline = deadline - oracle_time_est * 3
            if not _use_fast_osa:
                print(f"  [oSA] {_osa_remain:.0f}s remaining, starting oracle SA tail")
            # Macro selection weights: congestion-biased for congestion-dominated benchmarks.
            # _cong_weights selects macros in high-congestion cells with higher probability,
            # focusing oracle calls where they can reduce the dominant cost component.
            _osa_mv_arr = np.array(movable_hard, dtype=np.int64)
            _osa_probs = (None if _cong_weights is None
                          else _cong_weights.astype(np.float64) / _cong_weights.sum())
            _osa_n_mv = len(movable_hard)

            # Precompute net-adjacency for guided swap.
            # _osa_neighbors[ci] = list of top-5 hard macros that share most nets with ci.
            # Guided swap (50% chance) biases toward net-adjacent macros: 3-5x more likely
            # to find a WL-improving swap than a random-pair swap.
            _movable_hard_set = set(movable_hard)
            _osa_neighbors = [[] for _ in range(n_hard)]  # List[List[int]]
            for _ci in movable_hard:
                _shared: dict = {}
                for _ni in macro_to_nets[_ci]:
                    for (_nd, _, _, _nm) in (net_entries[_ni] or []):
                        if _nm >= 0 and _nm != _ci and _nm in _movable_hard_set:
                            _shared[_nm] = _shared.get(_nm, 0) + 1
                if _shared:
                    _osa_neighbors[_ci] = [k for k, _ in sorted(_shared.items(), key=lambda x: -x[1])[:5]]
            # Congestion map for gradient-guided moves (refresh every _cong_refresh_n steps)
            _cong_map = None
            _cong_scores = None
            _cong_refresh_n = 2000  # refresh congestion map every N oSA steps
            _cong_last_refresh = -_cong_refresh_n  # force refresh on first iteration

            # Oracle sync state: check oracle every ~20s; snap back when fast evaluator drifts.
            # Root cause of prior failure: oSA ran 1-4M fast moves, found 300-800 "fast-best"
            # updates, but every oracle check at the end failed (proxy drift outside calibration
            # range). Fix: oracle every 20s + immediate check on >0.5% fast improvement.
            # On oracle miss â†’ snap pos back to oracle-confirmed best so drift can't accumulate.
            _osa_last_oracle_t = time.time()
            _osa_oracle_interval_s = 20.0  # oracle sync every 20s (~60s total overhead per 3600s)
            _osa_oracle_best_fast = _fast_best if _use_fast_osa else float('inf')
            _osa_oracle_syncs = 0; _osa_snap_backs = 0

            while time.time() < _osa_deadline:
                # Adaptive scale: decay from base to 25% of base as time runs out
                _t_frac = min(1.0, (time.time() - _osa_t0) / max(1e-9, _osa_deadline - _osa_t0))
                _osa_scale = _osa_scale_base * max(0.25, 1.0 - 0.75 * _t_frac)
                # Temperature annealing: hot â†’ cold over oSA budget.
                # At t=0: T = _osa_T (initial). At t=1: T = 0.01 Ã— _osa_T (near-greedy).
                _osa_T_cur = _osa_T * max(0.01, 1.0 - 0.99 * _t_frac)

                # â”€â”€ Oracle sync: verify fast-best with oracle periodically â”€â”€â”€â”€â”€â”€
                # Trigger on: (a) time interval OR (b) fast improved >0.5% vs oracle baseline.
                # On oracle miss: snap back so fast evaluator re-anchors to a valid region.
                if _use_fast_osa and _osa_tries > 0:
                    _fast_impr = (_osa_oracle_best_fast - _fast_best) / max(1e-9, abs(_osa_oracle_best_fast))
                    _t_since   = time.time() - _osa_last_oracle_t
                    if _t_since >= _osa_oracle_interval_s or _fast_impr > 0.005:
                        _oc = compute_proxy_cost(
                            torch.tensor(_fast_best_pos, dtype=torch.float32), benchmark, plc
                        )["proxy_cost"]
                        oracle_calls += 1; _osa_oracle_syncs += 1
                        _osa_last_oracle_t = time.time()
                        if _oc < best_cost:
                            best_cost = _oc; best_pos = _fast_best_pos.copy(); best_ori = ori.copy()
                            _osa_oracle_best_fast = _fast_best
                            _osa_improved += 1
                            print(f"  [oSA oracle] t={time.time()-_osa_t0:.0f}s  "
                                  f"oracle_best={_oc:.4f}  fast={_fast_best:.4f}")
                        else:
                            # Fast evaluator drifted â€” snap back to oracle-confirmed best
                            pos[:] = best_pos
                            all_cx[:n_mac] = pos[:n_mac, 0]; all_cy[:n_mac] = pos[:n_mac, 1]
                            _fast_cur = _fast.evaluate(torch.tensor(pos, dtype=torch.float32))
                            _fast_best = _fast_cur; _fast_best_pos = pos.copy()
                            _osa_oracle_best_fast = _fast_cur
                            _osa_snap_backs += 1

                # Refresh congestion map periodically when fast evaluator available.
                # Used to select macros biased toward congested regions (10% of moves).
                if _use_fast_osa and (_osa_tries - _cong_last_refresh) >= _cong_refresh_n:
                    try:
                        _cong_scores = _fast.macro_congestion_score(
                            torch.tensor(pos, dtype=torch.float32), n_hard
                        )
                        _cong_last_refresh = _osa_tries
                    except Exception:
                        _cong_scores = None

                # Move selection: 10% congestion-gradient (pick most-congested movable macro),
                # else use existing probs or uniform.
                if (_use_fast_osa and _cong_scores is not None
                        and random.random() < 0.10 and _osa_n_mv > 0):
                    # Pick the movable macro with highest congestion contribution
                    _mv_scores = _cong_scores[_osa_mv_arr]
                    ci = int(_osa_mv_arr[int(np.argmax(_mv_scores))])
                elif _osa_probs is not None:
                    ci = int(np.random.choice(_osa_mv_arr, p=_osa_probs))
                else:
                    ci = random.choice(movable_hard)
                _osa_tries += 1

                # 30% swap move: exchange positions of ci and a second macro.
                # Guided: 50% chance pick from top-5 net-adjacent macros (more likely to
                # improve WL than a random pair), 50% random for diversity.
                if _osa_n_mv > 1 and random.random() < 0.30:
                    _nbrs = _osa_neighbors[ci]
                    if _nbrs and random.random() < 0.5:
                        cj = random.choice(_nbrs)
                    else:
                        cj = random.choice(movable_hard)
                    if cj == ci:
                        continue
                    oxi, oyi = float(pos[ci, 0]), float(pos[ci, 1])
                    oxj, oyj = float(pos[cj, 0]), float(pos[cj, 1])
                    # Bounds check: ci must fit at cj's pos, cj must fit at ci's pos.
                    if (oxj - hw[ci] < 0 or oxj + hw[ci] > cw or oyj - hh[ci] < 0 or oyj + hh[ci] > ch
                            or oxi - hw[cj] < 0 or oxi + hw[cj] > cw or oyi - hh[cj] < 0 or oyi + hh[cj] > ch):
                        continue
                    # Move both simultaneously so _overlaps sees the post-swap state.
                    pos[ci, 0] = oxj; pos[ci, 1] = oyj; all_cx[ci] = oxj; all_cy[ci] = oyj
                    pos[cj, 0] = oxi; pos[cj, 1] = oyi; all_cx[cj] = oxi; all_cy[cj] = oyi
                    if _overlaps(ci, oxj, oyj) or _overlaps(cj, oxi, oyi):
                        pos[ci, 0] = oxi; pos[ci, 1] = oyi; all_cx[ci] = oxi; all_cy[ci] = oyi
                        pos[cj, 0] = oxj; pos[cj, 1] = oyj; all_cx[cj] = oxj; all_cy[cj] = oyj
                        continue
                    if _use_fast_osa:
                        # Swap: use full fast.evaluate (delta_wl doesn't capture 2-macro moves well)
                        cost = _fast.evaluate(torch.tensor(pos, dtype=torch.float32))
                        delta = cost - _fast_cur
                    else:
                        cost = compute_proxy_cost(
                            torch.tensor(pos, dtype=torch.float32), benchmark, plc
                        )["proxy_cost"]
                        oracle_calls += 1
                        delta = cost - best_cost
                    if delta < 0 or (delta < _osa_T_cur and random.random() < math.exp(-delta / _osa_T_cur)):
                        _osa_accepted += 1
                        if _use_fast_osa:
                            _fast_cur = cost
                            if cost < _fast_best:
                                _fast_best = cost; _fast_best_pos = pos.copy()
                                _osa_improved += 1
                        else:
                            if cost < best_cost:
                                best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                                _osa_improved += 1
                                print(f"  [oSA] {cost:.4f}")
                    else:
                        pos[ci, 0] = oxi; pos[ci, 1] = oyi; all_cx[ci] = oxi; all_cy[ci] = oyi
                        pos[cj, 0] = oxj; pos[cj, 1] = oyj; all_cx[cj] = oxj; all_cy[cj] = oyj
                    continue

                # 70%: Gaussian translation of a single macro
                cx = float(np.clip(pos[ci, 0] + random.gauss(0, _osa_scale), hw[ci], cw - hw[ci]))
                cy = float(np.clip(pos[ci, 1] + random.gauss(0, _osa_scale), hh[ci], ch - hh[ci]))
                if _overlaps(ci, cx, cy):
                    continue
                old_x, old_y = float(pos[ci, 0]), float(pos[ci, 1])
                # Apply move to pos (needed for both fast and oracle evaluation)
                pos[ci, 0] = cx; pos[ci, 1] = cy; all_cx[ci] = cx; all_cy[ci] = cy
                if _use_fast_osa:
                    # Translation: delta_wl or periodic full re-sync
                    _osa_delta_steps += 1
                    if _osa_delta_steps >= _osa_sync_interval:
                        # Full re-sync (pos already updated above)
                        cost = _fast.evaluate(torch.tensor(pos, dtype=torch.float32))
                        _osa_delta_steps = 0
                    else:
                        # Incremental: recompute only affected nets (pos already updated)
                        # Temporarily restore old pos to compute delta from correct baseline
                        pos[ci, 0] = old_x; pos[ci, 1] = old_y
                        d_wl = _fast.delta_wl(
                            torch.tensor(pos, dtype=torch.float32), ci, cx, cy
                        )
                        pos[ci, 0] = cx; pos[ci, 1] = cy
                        cost = _fast_cur + d_wl
                    delta = cost - _fast_cur
                if not _use_fast_osa:
                    cost = compute_proxy_cost(
                        torch.tensor(pos, dtype=torch.float32), benchmark, plc
                    )["proxy_cost"]
                    oracle_calls += 1
                    delta = cost - best_cost
                if delta < 0 or (delta < _osa_T_cur and random.random() < math.exp(-delta / _osa_T_cur)):
                    _osa_accepted += 1
                    if _use_fast_osa:
                        _fast_cur = cost
                        if cost < _fast_best:
                            _fast_best = cost; _fast_best_pos = pos.copy()
                            _osa_improved += 1
                            _rebuild_fix_bbox_for_nets(macro_to_nets[ci])
                    else:
                        if cost < best_cost:
                            best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                            _osa_improved += 1
                            _rebuild_fix_bbox_for_nets(macro_to_nets[ci])
                            print(f"  [oSA] {cost:.4f}")
                            # Refresh congestion weights on improvement to track updated hotspots
                            if _osa_probs is not None and _cong_weights is not None:
                                try:
                                    _h2 = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float32)
                                    _v2 = np.array(plc.get_vertical_routing_congestion(), dtype=np.float32)
                                    _mc2 = np.array([(_h2+_v2)[c] if 0<=c<len(_h2) else 0.0
                                                     for c in [plc.get_grid_cell_of_node(p) for p in _plc_ids]],
                                                    dtype=np.float32)
                                    _thr2 = float(np.percentile(_mc2, 50))
                                    _wts2 = np.maximum(0.0, _mc2 - _thr2).astype(np.float64)
                                    if _wts2.sum() > 0:
                                        _cong_weights = _wts2 / _wts2.sum()
                                        _osa_probs = _cong_weights / _cong_weights.sum()
                                except Exception:
                                    pass
                else:
                    pos[ci, 0] = old_x; pos[ci, 1] = old_y
                    all_cx[ci] = old_x; all_cy[ci] = old_y

            # â”€â”€ After fast oSA: best_pos is already oracle-verified (ongoing syncs) â”€
            if _use_fast_osa:
                # Restore pos to oracle-confirmed best (fast moves may have left pos elsewhere)
                pos[:] = best_pos; ori[:] = best_ori
                all_cx[:n_mac] = pos[:n_mac, 0]; all_cy[:n_mac] = pos[:n_mac, 1]
                _osa_improved_verified = best_cost < _pre_osa_best_cost
                print(f"  [oSA] {_osa_tries} fast moves, {_osa_improved} oracle-confirmed improvements, "
                      f"{_osa_oracle_syncs} syncs, {_osa_snap_backs} snap-backs; "
                      f"best={best_cost:.4f} vs pre-oSA={_pre_osa_best_cost:.4f}")
                if not _osa_improved_verified:
                    print(f"  [oSA] no oracle-confirmed improvement over pre-oSA baseline")
            else:
                _osa_improved_verified = _osa_improved > 0
                if _osa_improved > 0 or _osa_accepted > 5:
                    print(f"  [oSA] {_osa_improved} improved, {_osa_accepted}/{_osa_tries} accepted")

            # Verify oSA improvements survive legalization (prevents ibm06-type regression
            # where oSA finds "better" positions with harder legalization cascade)
            if _osa_improved_verified:
                _post_micro = self._micro_legalize(torch.tensor(best_pos, dtype=torch.float32), benchmark)
                if compute_overlap_metrics(_post_micro, benchmark)["overlap_count"] > 0:
                    best_pos = _pre_osa_best_pos; best_cost = _pre_osa_best_cost
                    print(f"  [oSA] reverted: legalization fails for oSA result")
                else:
                    best_pos = _post_micro.numpy()  # use micro-legalized result directly

        elif not _osa_micro_ok and _osa_remain > oracle_time_est * 20:
            # ILS restarts: micro_legalize fails for current best (needs legalize_hard).
            # Try perturbed restarts to find a configuration where micro_legalize succeeds.
            # Key for ibm06: SA tends to push macros to canvas boundaries (creating stuck
            # pairs). Different random perturbations may avoid this.
            _ils_budget_s = max(oracle_time_est * 12, min(oracle_time_est * 30, _osa_remain / 15))
            _ils_scale = max(cw, ch) * 0.10
            _ils_count = 0; _ils_ok_count = 0
            print(f"  [ILS] {_osa_remain:.0f}s, ~{_osa_remain/_ils_budget_s:.0f} restarts planned")
            # Identify boundary-adjacent macros: cascade failures in ibm06 are caused by
            # macros at canvas edges creating stuck fixed-macro overlap pairs.
            _bdy_thresh = max(cw, ch) * 0.05
            _bdy_macros = [i for i in movable_hard
                           if (pos[i,0]-hw[i] < _bdy_thresh or cw-pos[i,0]-hw[i] < _bdy_thresh
                               or pos[i,1]-hh[i] < _bdy_thresh or ch-pos[i,1]-hh[i] < _bdy_thresh)]
            while time.time() < deadline - oracle_time_est * 8:
                _ils_count += 1
                _prt = best_pos.copy()
                _n_p = max(1, n_mv // 12)
                # Alternate: half the restarts target boundary macros (push inward),
                # half use random perturbation to diversify exploration.
                if _bdy_macros and _ils_count % 2 == 0:
                    _targets = random.sample(_bdy_macros, min(max(1, len(_bdy_macros)//2), _n_p))
                    for _pm in _targets:
                        _dx = cw/2 - _prt[_pm, 0]; _dy = ch/2 - _prt[_pm, 1]
                        _d = math.sqrt(_dx*_dx + _dy*_dy)
                        if _d > 0:
                            _prt[_pm, 0] = float(np.clip(_prt[_pm,0]+_dx/_d*_ils_scale, hw[_pm], cw-hw[_pm]))
                            _prt[_pm, 1] = float(np.clip(_prt[_pm,1]+_dy/_d*_ils_scale, hh[_pm], ch-hh[_pm]))
                else:
                    for _pm in random.sample(movable_hard, min(_n_p, n_mv)):
                        _prt[_pm, 0] = float(np.clip(_prt[_pm, 0] + random.gauss(0, _ils_scale), hw[_pm], cw - hw[_pm]))
                        _prt[_pm, 1] = float(np.clip(_prt[_pm, 1] + random.gauss(0, _ils_scale), hh[_pm], ch - hh[_pm]))
                _rebuild_state(_prt, best_ori)
                _r_dead = min(deadline - oracle_time_est * 5, time.time() + _ils_budget_s)
                _r_best_fast = float('inf') if _fast_ok else float('inf')
                _r_best = float('inf'); _r_pos = _prt.copy()
                if _fast_ok and _fast is not None:
                    _r_best_fast = _fast.evaluate(torch.tensor(pos, dtype=torch.float32))
                while time.time() < _r_dead:
                    ci = random.choice(movable_hard)
                    cx = float(np.clip(pos[ci, 0] + random.gauss(0, _ils_scale * 0.5), hw[ci], cw - hw[ci]))
                    cy = float(np.clip(pos[ci, 1] + random.gauss(0, _ils_scale * 0.5), hh[ci], ch - hh[ci]))
                    if _overlaps(ci, cx, cy): continue
                    ox, oy = pos[ci, 0], pos[ci, 1]
                    pos[ci, 0] = cx; pos[ci, 1] = cy; all_cx[ci] = cx; all_cy[ci] = cy
                    if _fast_ok and _fast is not None:
                        r_cost = _fast.evaluate(torch.tensor(pos, dtype=torch.float32))
                        if r_cost < _r_best_fast: _r_best_fast = r_cost; _r_pos = pos.copy()
                        else: pos[ci, 0] = ox; pos[ci, 1] = oy; all_cx[ci] = ox; all_cy[ci] = oy
                    else:
                        r_cost = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), benchmark, plc)["proxy_cost"]
                        oracle_calls += 1
                        if r_cost < _r_best: _r_best = r_cost; _r_pos = pos.copy()
                        else: pos[ci, 0] = ox; pos[ci, 1] = oy; all_cx[ci] = ox; all_cy[ci] = oy
                _r_micro = self._micro_legalize(torch.tensor(_r_pos, dtype=torch.float32), benchmark)
                if compute_overlap_metrics(_r_micro, benchmark)["overlap_count"] == 0:
                    _ils_ok_count += 1
                    _r_mc = compute_proxy_cost(_r_micro, benchmark, plc)["proxy_cost"]
                    oracle_calls += 1
                    if _r_mc < best_cost:
                        best_cost = _r_mc; best_pos = _r_micro.numpy()
                        print(f"  [ILS] restart {_ils_count}: micro OK, {_r_mc:.4f} (new best!)")
                        _osa_micro_ok = True  # final legalization will use micro result
                        break
                _rebuild_state(best_pos, best_ori)
            if _ils_count > 0:
                print(f"  [ILS] {_ils_count} restarts ({_ils_ok_count} micro OK), best={best_cost:.4f}")

        # â”€â”€ Final soft-macro update (revert if no gain) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        _rebuild_state(best_pos, best_ori)
        pos_with_soft_update = self._update_soft_macros(pos.copy(), benchmark)
        soft_cost = compute_proxy_cost(
            torch.tensor(pos_with_soft_update, dtype=torch.float32), benchmark, plc
        )["proxy_cost"]
        oracle_calls += 1
        if soft_cost < best_cost:
            best_cost = soft_cost
            best_pos = pos_with_soft_update
            print(f"  [soft] update improved: {soft_cost:.4f}")
        else:
            print(f"  [soft] update no gain ({soft_cost:.4f} >= {best_cost:.4f}), kept CT positions")

        _budget_log("SA done", f"final={best_cost:.4f}  oracle_calls={oracle_calls}  passes={pass_num}")
        result = torch.tensor(best_pos, dtype=torch.float32)
        # Full legalization if significant (>4nm) overlaps remain (e.g. after Xplace legalization).
        if self._count_significant_overlaps(result, benchmark, threshold=0.004) > 0:
            return self._legalize_hard(result, benchmark)
        # Micro-legalization: fix any residual sub-4nm overlaps.
        # With pre-SA micro_legalize, CT overlaps are resolved before SA sees them;
        # SA's _overlaps filter prevents new overlaps from SA moves. So this should be a no-op.
        micro = self._micro_legalize(result, benchmark)
        # Use exact float64 check (matches harness compute_overlap_metrics).
        if compute_overlap_metrics(micro, benchmark)["overlap_count"] > 0:
            # micro didn't fully converge; start legalize_hard from micro (closer to optimum).
            lh = self._legalize_hard(micro, benchmark)
            return lh
        return micro


