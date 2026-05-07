"""Direct macro-position SA placer with LNS escape.

Pipeline:
  1. Use the .plc warm start (raw, may have small overlaps) as the starting
     configuration. Only hard movable macros are optimized; soft/fixed
     macros stay put.
  2. Run simulated annealing on continuous macro centers with a cost that
     scales overlap-area penalty geometrically over the budget — overlaps
     are cheap early (so SA can explore) and prohibitively expensive late
     (so the trajectory ends in a feasible configuration).
  3. Periodic LNS escape: destroy a random subset of macros and repair
     them at HPWL centroids of their net neighbors.
  4. End-of-budget: ``legalize_hard`` to guarantee zero overlap (small
     touch-up since the SA penalty already drives feasibility).
  5. Score against legalized-.plc baseline via the true TILOS proxy and
     return the winner.

Usage:
    uv run evaluate submissions/macro_sa_placer.py -b ibm01
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _btree_sa import build_sa_context  # noqa: E402
from _candidate_select import select_best_true_proxy  # noqa: E402
from _hard_legalizer import legalize_hard  # noqa: E402
from _macro_sa import MacroSAConfig, run_macro_sa, _pairwise_overlap_count  # noqa: E402
from _plc_lookup import PlcLookup  # noqa: E402


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class MacroSAPlacer:
    """Direct macro-position SA, anchored on .plc, with LNS escape."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.time_budget_s = _env_float("MACRO_SA_BUDGET_S", 240.0)
        self.seed = _env_int("MACRO_SA_SEED", 0)
        self.density_weight = _env_float("MACRO_SA_DENSITY_W", 0.5)
        self.lambda_final = _env_float("MACRO_SA_LAM_FINAL", 200.0)
        self.lns_destroy_frac = _env_float("MACRO_SA_LNS_FRAC", 0.10)
        self.lns_period = _env_int("MACRO_SA_LNS_PERIOD", 1500)
        self._plc_lookup = PlcLookup()

    def place(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        t_start = time.monotonic()
        baseline = benchmark.macro_positions.clone().float()
        n_hard = benchmark.num_hard_macros
        if n_hard <= 1:
            return baseline

        ctx = build_sa_context(benchmark)
        if ctx.movable_idx.size < 2:
            return baseline

        sa_budget = max(20.0, self.time_budget_s - 30.0)
        config = MacroSAConfig(
            time_budget_s=sa_budget,
            seed=self.seed,
            density_weight=self.density_weight,
            lambda_final=self.lambda_final,
            lns_destroy_frac=self.lns_destroy_frac,
            lns_period_iters=self.lns_period,
        )

        if self.verbose:
            print(
                f"[macro-sa] benchmark={benchmark.name} "
                f"n_hard_movable={ctx.movable_idx.size} "
                f"canvas={benchmark.canvas_width:.2f}x{benchmark.canvas_height:.2f} "
                f"sa_budget={sa_budget:.1f}s seed={self.seed}",
                flush=True,
            )

        init_full = baseline.cpu().numpy().astype(np.float64)
        result = run_macro_sa(init_full, ctx, config, verbose=self.verbose)

        sizes_movable = ctx.sizes[ctx.movable_idx]
        if self.verbose:
            print(
                f"[macro-sa] SA done iters={result.iters} "
                f"accepted={result.accepted} "
                f"({result.accepted / max(result.iters, 1) * 100:.1f}%) "
                f"rate={result.rate:.0f} it/s",
                flush=True,
            )
            comp_rows = [
                ("best", result.best_cost, result.best_components),
                ("feasible", result.best_feasible_cost, result.best_feasible_components),
                ("final", result.final_cost, result.final_components),
            ]
            for label, cost, comps in comp_rows:
                if comps is None:
                    print(f"[macro-sa]   {label:<8} (none observed)")
                    continue
                print(
                    f"[macro-sa]   {label:<8} cost={cost:.4f} "
                    f"WL_n={comps['WL_norm']:.3f} D_n={comps['D_norm']:.3f} "
                    f"ov_pen={comps['overlap_pen']:.4f} "
                    f"pairs={comps['overlap_pairs']}",
                    flush=True,
                )

        def _legal(centers_movable):
            pl_np = baseline.cpu().numpy().astype(np.float64)
            pl_np[ctx.movable_idx] = centers_movable
            pl = torch.from_numpy(pl_np).float()
            # Larger displacement budget so legalize_hard can preserve more
            # SA gain when the SA result has small residual overlaps.
            return legalize_hard(
                pl,
                benchmark,
                overlap_gap=1e-3,
                legalize_rounds=400,
                outer_passes=3,
                displacement_budget_frac=0.20,
                step_fraction=0.3,
            )

        sa_best_legal = _legal(result.best_centers_movable)
        sa_final_legal = _legal(result.final_centers_movable)
        cand_pls: list = [sa_best_legal, sa_final_legal]
        cand_labels: list = ["sa_best", "sa_final"]
        if result.best_feasible_centers_movable is not None:
            cand_pls.append(_legal(result.best_feasible_centers_movable))
            cand_labels.append("sa_feasible")

        legal_baseline = legalize_hard(
            baseline,
            benchmark,
            overlap_gap=1e-3,
            legalize_rounds=300,
            outer_passes=2,
            displacement_budget_frac=0.10,
            step_fraction=0.3,
        )
        cand_pls.insert(0, legal_baseline)
        cand_labels.insert(0, "legal_plc")

        if plc is None:
            plc = self._plc_lookup.load(benchmark)
        if plc is None:
            if self.verbose:
                print("[macro-sa] no plc; returning legalized SA-best placement")
            return sa_best_legal

        sel = select_best_true_proxy(
            baseline,
            cand_pls,
            benchmark,
            plc,
            candidate_labels=cand_labels,
        )
        if self.verbose:
            for s in sel.scores:
                if s.valid:
                    print(
                        f"[macro-sa]   {s.label:<10} proxy={s.proxy_cost:.4f} "
                        f"WL={s.wirelength:.3f} D={s.density:.3f} "
                        f"C={s.congestion:.3f}"
                    )
                else:
                    print(
                        f"[macro-sa]   {s.label:<10} INVALID overlaps={s.overlaps}"
                    )
            elapsed = time.monotonic() - t_start
            print(
                f"[macro-sa] best={sel.best.label} "
                f"proxy={sel.best.proxy_cost:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
        return sel.placement
