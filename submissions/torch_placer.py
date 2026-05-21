"""GPU-native multi-start macro placer.

Pipeline:
  1. Build a wide seed portfolio: handcrafted (CT + symmetry transforms) +
     spectral (Laplacian eigvecs of macro affinity graph) + noise variants.
  2. Wide differentiable global placement burst on GPU via :class:`CustomGP`
     (HPWL + density + RUDY congestion), with early-kill of seeds that the
     calibrated :class:`FastEvaluator` flags as unpromising.
  3. Full-budget GP on the survivors.
  4. Hard legalization (`legalize_hard`).
  5. Oracle-score (true proxy) and pick the best.
  6. GWTW SA population refinement (`tilos_gwtw_sa_refine`).
  7. Final oracle selection (current best vs SA result).

Environment knobs (all optional):
  TORCH_PLACER_BUDGET            total wall budget seconds (default 3000)
  TORCH_PLACER_SEEDS             initial seed count (default 16)
  TORCH_PLACER_KEEP              survivors after early-kill (default 6)
  TORCH_PLACER_GP_BURST_ITERS    iters per seed in the kill burst (default 120)
  TORCH_PLACER_GP_FULL_ITERS     iters per survivor in the full run (default 600)
  TORCH_PLACER_GWTW_S            GWTW SA budget seconds (default 540)
  TORCH_PLACER_GWTW_WORKERS      GWTW worker count (default 8)
  TORCH_PLACER_GWTW_ITERS        GWTW iters per worker (default 140)
  TORCH_PLACER_CD_S              coordinate-descent pre-polish seconds (default 120)
  TORCH_PLACER_CD_PASSES         coordinate-descent pass count (default 1)
  TORCH_PLACER_CD_K              coordinate-descent k-cell bound (default adaptive)
  TORCH_PLACER_NO_SPECTRAL=1     skip spectral seeding
  TORCH_PLACER_DEBUG=1           extra logging
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

# fd-friendly multiprocessing for slurm + spawn-based GWTW pools.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from custom_gp import CustomGP  # noqa: E402
from fast_eval import FastEvaluator  # noqa: E402
from _hard_legalizer import legalize_hard  # noqa: E402
from _plc_lookup import PlcLookup  # noqa: E402
from _tilos_gwtw_sa import tilos_gwtw_sa_refine  # noqa: E402
from _coord_descent import coord_descent_polish  # noqa: E402


# ── env knobs ───────────────────────────────────────────────────────────────


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


_DEBUG = _env_flag("TORCH_PLACER_DEBUG", default=False)


def _log(msg: str) -> None:
    print(f"  [torch_placer] {msg}", flush=True)


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"  [torch_placer:dbg] {msg}", flush=True)


# ── budget tracker ──────────────────────────────────────────────────────────


class _Budget:
    def __init__(self, total_s: float) -> None:
        self.total = float(total_s)
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def remaining(self) -> float:
        return max(0.0, self.total - self.elapsed())

    def used_pct(self) -> float:
        if self.total <= 0:
            return 100.0
        return min(100.0, self.elapsed() / self.total * 100.0)

    def log(self, phase: str, extra: str = "") -> None:
        msg = (
            f"{phase}: elapsed={self.elapsed():.0f}s "
            f"remaining={self.remaining():.0f}s ({self.used_pct():.0f}%)"
        )
        if extra:
            msg += f" {extra}"
        _log(msg)


# ── seed construction ──────────────────────────────────────────────────────


def _clamp_to_canvas(pos: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    """Clamp every macro centre inside the canvas; preserve fixed-macro positions."""
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.float()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    out = pos.clone().float()
    out[:, 0] = torch.clamp(out[:, 0], half_w, torch.clamp(cw - half_w, min=half_w))
    out[:, 1] = torch.clamp(out[:, 1], half_h, torch.clamp(ch - half_h, min=half_h))
    fixed = benchmark.macro_fixed
    out[fixed] = benchmark.macro_positions[fixed].float()
    return out


def _symmetry_seeds(benchmark: Benchmark) -> List[Tuple[str, torch.Tensor]]:
    """Mirror/transpose transforms of the handoff placement (movable hard macros)."""
    ct = benchmark.macro_positions.float()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_hard = int(benchmark.num_hard_macros)
    movable = (~benchmark.macro_fixed[:n_hard]).clone()

    out: List[Tuple[str, torch.Tensor]] = [("ct", ct.clone())]
    for label, fn in (
        ("mirror_x", lambda p: torch.stack([cw - p[:, 0], p[:, 1]], dim=1)),
        ("mirror_y", lambda p: torch.stack([p[:, 0], ch - p[:, 1]], dim=1)),
        ("mirror_xy", lambda p: torch.stack([cw - p[:, 0], ch - p[:, 1]], dim=1)),
    ):
        s = ct.clone()
        moved = fn(ct[:n_hard])
        s[:n_hard][movable] = moved[movable]
        out.append((label, _clamp_to_canvas(s, benchmark)))

    # Transpose only makes sense for near-square canvases.
    if abs(cw - ch) / max(cw, ch) < 0.2:
        scale_x = cw / ch
        scale_y = ch / cw
        transposed = torch.stack(
            [ct[:n_hard, 1] * scale_x, ct[:n_hard, 0] * scale_y], dim=1
        )
        s = ct.clone()
        s[:n_hard][movable] = transposed[movable]
        out.append(("transpose", _clamp_to_canvas(s, benchmark)))
    return out


def _spectral_seeds(
    benchmark: Benchmark,
    n_seeds: int = 2,
) -> List[Tuple[str, torch.Tensor]]:
    """Spectral (Laplacian eigenvector) seeds for movable hard macros.

    Build a macro-macro affinity matrix from net incidence (clique model:
    each net of size k contributes weight w/(k-1) to every macro pair it
    connects). Compute the lowest few non-trivial eigenvectors of the
    normalized Laplacian; project each pair of eigvecs to canvas via a
    rank-based map so the resulting placement spans the canvas uniformly.
    Strongly-connected macros end up near each other.
    """
    try:
        from scipy.sparse import csr_matrix, eye as sp_eye, diags as sp_diags
        from scipy.sparse.linalg import eigsh
    except Exception as exc:
        _log(f"spectral: scipy unavailable ({exc}); skipping")
        return []

    n_hard = int(benchmark.num_hard_macros)
    if n_hard < 4:
        return []

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for k, nodes in enumerate(benchmark.net_nodes):
        macros = [int(x) for x in nodes.tolist() if 0 <= int(x) < n_hard]
        if len(macros) < 2 or len(macros) > 256:
            # ignore singletons and pathologically wide nets (clock trees etc.)
            continue
        try:
            w = float(benchmark.net_weights[k])
        except Exception:
            w = 1.0
        scale = w / (len(macros) - 1)
        for i_idx in range(len(macros)):
            ai = macros[i_idx]
            for j_idx in range(i_idx + 1, len(macros)):
                aj = macros[j_idx]
                rows.append(ai); cols.append(aj); data.append(scale)
                rows.append(aj); cols.append(ai); data.append(scale)
    if not data:
        return []

    A = csr_matrix((data, (rows, cols)), shape=(n_hard, n_hard))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_safe = np.where(deg > 0, deg, 1.0)
    D_inv_sqrt = sp_diags(1.0 / np.sqrt(deg_safe))
    L_norm = sp_eye(n_hard) - D_inv_sqrt @ A @ D_inv_sqrt

    k_eig = min(6, n_hard - 1)
    try:
        eigvals, eigvecs = eigsh(L_norm.astype(np.float64), k=k_eig, which="SM")
    except Exception as exc:
        _log(f"spectral: eigsh failed ({exc}); skipping")
        return []
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    movable = (~benchmark.macro_fixed[:n_hard]).numpy()

    pair_idx = [(1, 2), (1, 3), (2, 3)]
    seeds: List[Tuple[str, torch.Tensor]] = []
    for kk, (ix, iy) in enumerate(pair_idx[:n_seeds]):
        if iy >= eigvecs.shape[1]:
            break
        vx = eigvecs[:, ix]
        vy = eigvecs[:, iy]

        def _rank_stretch(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
            order_v = np.argsort(v)
            rank = np.empty_like(order_v)
            rank[order_v] = np.arange(len(v))
            return lo + (rank.astype(np.float64) / max(1, len(v) - 1)) * (hi - lo)

        xs = _rank_stretch(vx, 0.05 * cw, 0.95 * cw)
        ys = _rank_stretch(vy, 0.05 * ch, 0.95 * ch)

        s = benchmark.macro_positions.float().clone()
        new_xy = torch.from_numpy(np.stack([xs, ys], axis=1)).float()
        movable_t = torch.from_numpy(movable)
        s[:n_hard][movable_t] = new_xy[movable_t]
        seeds.append((f"spectral_{kk}", _clamp_to_canvas(s, benchmark)))
    return seeds


def _noise_seeds(
    benchmark: Benchmark,
    base: torch.Tensor,
    scales: Sequence[float],
    rng: torch.Generator,
) -> List[Tuple[str, torch.Tensor]]:
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_hard = int(benchmark.num_hard_macros)
    movable = (~benchmark.macro_fixed[:n_hard]).clone()
    out: List[Tuple[str, torch.Tensor]] = []
    for sc in scales:
        s = base.clone()
        noise = torch.zeros_like(s[:n_hard])
        noise[:, 0] = torch.normal(
            0.0, cw * float(sc), size=(n_hard,), generator=rng
        )
        noise[:, 1] = torch.normal(
            0.0, ch * float(sc), size=(n_hard,), generator=rng
        )
        s[:n_hard][movable] = s[:n_hard][movable] + noise[movable]
        out.append((f"noise_{sc:.2f}", _clamp_to_canvas(s, benchmark)))
    return out


def _build_seed_portfolio(
    benchmark: Benchmark,
    *,
    n_seeds: int,
    use_spectral: bool,
    rng: torch.Generator,
) -> List[Tuple[str, torch.Tensor]]:
    portfolio: List[Tuple[str, torch.Tensor]] = []
    portfolio.extend(_symmetry_seeds(benchmark))
    if use_spectral:
        portfolio.extend(_spectral_seeds(benchmark, n_seeds=2))

    ct = benchmark.macro_positions.float()
    portfolio.extend(
        _noise_seeds(benchmark, ct, scales=(0.04, 0.10, 0.20), rng=rng)
    )

    # de-dup and cap
    seen: List[torch.Tensor] = []
    out: List[Tuple[str, torch.Tensor]] = []
    for label, s in portfolio:
        is_dup = False
        for prev in seen:
            if prev.shape == s.shape and torch.allclose(prev, s, atol=1e-3):
                is_dup = True
                break
        if is_dup:
            continue
        seen.append(s)
        out.append((label, s))
        if len(out) >= n_seeds:
            break
    return out


# ── GP burst with early kill ────────────────────────────────────────────────


# GP hyperparam variants — sprayed across seeds so survivors come from
# different basins (dense vs. spread, congestion-heavy vs. wirelength-heavy).
_GP_VARIANTS: Sequence[dict] = (
    dict(target_density=0.70, density_w_final=0.5, rudy_w=0.4, sigma_start=4.0),
    dict(target_density=0.60, density_w_final=0.4, rudy_w=0.6, sigma_start=4.5),
    dict(target_density=0.80, density_w_final=0.6, rudy_w=0.3, sigma_start=3.5),
    dict(target_density=0.65, density_w_final=0.5, rudy_w=0.8, sigma_start=4.0),
    dict(target_density=0.55, density_w_final=0.3, rudy_w=0.5, sigma_start=5.0),
)


def _gp_burst(
    gp: CustomGP,
    seeds: Sequence[Tuple[str, torch.Tensor]],
    *,
    burst_iters: int,
    device: torch.device,
    benchmark: Benchmark,
    fast: Optional[FastEvaluator],
    keep: int,
    time_budget_s: float,
) -> List[Tuple[str, torch.Tensor, float, dict]]:
    """Run a short GP burst on every seed (sprayed across variant configs);
    rank by FastEval; keep top-``keep`` as (label, placement, score, variant)."""
    results: List[Tuple[str, torch.Tensor, float, dict]] = []
    t0 = time.time()
    for vi, (label, seed) in enumerate(seeds):
        if time.time() - t0 >= time_budget_s:
            _log(f"burst time-up after {len(results)}/{len(seeds)} seeds")
            break
        variant = _GP_VARIANTS[vi % len(_GP_VARIANTS)]
        try:
            placed = gp.optimize_v2(
                seed,
                n_iters=burst_iters,
                device=device,
                log_every=0,
                target_density=variant["target_density"],
                density_w_final=variant["density_w_final"],
                rudy_w=variant["rudy_w"],
                sigma_start=variant["sigma_start"],
                # leave proxy gate OFF in burst (would be too noisy on
                # under-converged placements)
            )
        except Exception as exc:
            _dbg(f"burst seed={label} failed: {exc}")
            continue
        if fast is not None:
            try:
                score = float(fast.evaluate(placed))
            except Exception:
                score = float(
                    compute_proxy_cost(placed, benchmark, None)["proxy_cost"]
                )
        else:
            score = float(
                compute_proxy_cost(placed, benchmark, None)["proxy_cost"]
            )
        results.append((label, placed, score, variant))
        _dbg(
            f"burst {label} td={variant['target_density']:.2f} "
            f"rudy_w={variant['rudy_w']:.2f} score={score:.4f}"
        )

    results.sort(key=lambda r: r[2])
    survivors = results[:keep]
    if survivors:
        scores = ", ".join(
            f"{r[0]}(td{r[3]['target_density']:.2f})={r[2]:.3f}"
            for r in survivors
        )
        _log(f"burst survivors ({len(survivors)}/{len(results)}): {scores}")
    return survivors


# ── Full-budget GP on survivors ────────────────────────────────────────────


def _gp_full(
    gp: CustomGP,
    survivors: Sequence[Tuple[str, torch.Tensor, float, dict]],
    *,
    full_iters: int,
    device: torch.device,
    time_budget_s: float,
    fast: Optional[FastEvaluator] = None,
) -> List[Tuple[str, torch.Tensor]]:
    out: List[Tuple[str, torch.Tensor]] = []
    if not survivors:
        return out
    per_seed_budget = max(15.0, time_budget_s / max(1, len(survivors)))
    t0 = time.time()
    for label, placed, _fast_score, variant in survivors:
        if time.time() - t0 >= time_budget_s:
            _log(f"full-GP time-up after {len(out)}/{len(survivors)} seeds")
            break
        remaining = time_budget_s - (time.time() - t0)
        iters = max(200, int(full_iters * min(1.0, remaining / per_seed_budget)))
        try:
            final = gp.optimize_v2(
                placed.to(device),
                n_iters=iters,
                device=device,
                log_every=0,
                # tighter polish: shorter gamma anneal end, full density ramp
                gamma_start=4.0,
                gamma_end=0.5,
                sigma_start=max(2.0, variant["sigma_start"] - 1.0),
                sigma_end=0.5,
                target_density=variant["target_density"],
                density_w_final=variant["density_w_final"],
                rudy_w=variant["rudy_w"],
                proxy_gate_every=50,
                fast_eval=fast,
            )
        except Exception as exc:
            _dbg(f"full-GP seed={label} failed: {exc}")
            continue
        out.append((f"{label}_td{variant['target_density']:.2f}", final))
    return out


# ── Legalize + oracle-score + select ───────────────────────────────────────


def _legalize_and_score(
    candidates: Sequence[Tuple[str, torch.Tensor]],
    benchmark: Benchmark,
    plc,
) -> List[Tuple[str, torch.Tensor, float]]:
    scored: List[Tuple[str, torch.Tensor, float]] = []
    for label, pos in candidates:
        legal = None
        for rounds, frac in ((400, 0.20), (1200, None)):
            try:
                cand = legalize_hard(
                    pos.cpu(),
                    benchmark,
                    overlap_gap=1e-3,
                    legalize_rounds=rounds,
                    outer_passes=2 if frac is not None else 1,
                    displacement_budget_frac=frac,
                    step_fraction=0.35,
                )
            except Exception as exc:
                _dbg(f"legalize {label} rounds={rounds} failed: {exc}")
                continue
            ok, _ = validate_placement(cand, benchmark, check_overlaps=True)
            if ok:
                legal = cand
                break
        if legal is None:
            _dbg(f"{label} invalid after legalize stages, skipping")
            continue
        try:
            cost = float(compute_proxy_cost(legal, benchmark, plc)["proxy_cost"])
        except Exception as exc:
            _dbg(f"{label} oracle score failed: {exc}")
            continue
        scored.append((label, legal, cost))
    scored.sort(key=lambda r: r[2])
    return scored


def _safe_legalize(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    try:
        return legalize_hard(
            placement,
            benchmark,
            overlap_gap=1e-3,
            legalize_rounds=1200,
            outer_passes=2,
            displacement_budget_frac=0.20,
            step_fraction=0.30,
        )
    except Exception:
        return placement


# ── public placer class ────────────────────────────────────────────────────


class TorchPlacer:
    """GPU-native multi-start placer (HPWL + density + RUDY GP → GWTW SA)."""

    def __init__(self) -> None:
        self.total_budget_s = _env_float("TORCH_PLACER_BUDGET", 3000.0)
        self.n_seeds = _env_int("TORCH_PLACER_SEEDS", 16)
        self.keep = _env_int("TORCH_PLACER_KEEP", 6)
        self.burst_iters = _env_int("TORCH_PLACER_GP_BURST_ITERS", 200)
        self.full_iters = _env_int("TORCH_PLACER_GP_FULL_ITERS", 1500)
        self.gwtw_seconds = _env_float("TORCH_PLACER_GWTW_S", 540.0)
        self.gwtw_workers = _env_int("TORCH_PLACER_GWTW_WORKERS", 8)
        self.gwtw_iters = _env_int("TORCH_PLACER_GWTW_ITERS", 280)
        self.gwtw_topk = _env_int("TORCH_PLACER_GWTW_TOPK", 2)
        self.gwtw_syncup = _env_float("TORCH_PLACER_GWTW_SYNC", 0.20)
        self.gwtw_t_max = _env_float("TORCH_PLACER_GWTW_TMAX", 5e-3)
        self.gwtw_t_min = _env_float("TORCH_PLACER_GWTW_TMIN", 5e-6)
        self.cd_seconds = _env_float("TORCH_PLACER_CD_S", 120.0)
        self.cd_passes = _env_int("TORCH_PLACER_CD_PASSES", 1)
        self.cd_k_bound_raw = _env_int("TORCH_PLACER_CD_K", -1)
        self.cd_cell_prob = _env_float("TORCH_PLACER_CD_CELL_PROB", 1.0)
        # Per-step legalize displacement budget inside SA. 0.10 (default in
        # dreamplace_pipeline) yanks big proposed moves back, killing
        # exploration. Use None (unbounded) to let SA actually move.
        self.sa_disp_budget = _env_float("TORCH_PLACER_SA_DISP_BUDGET", -1.0)
        self.use_spectral = not _env_flag("TORCH_PLACER_NO_SPECTRAL", default=False)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _log(
            f"init device={self.device.type}  budget={self.total_budget_s:.0f}s "
            f"seeds={self.n_seeds} keep={self.keep} spectral={self.use_spectral}"
        )

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(42)
        np.random.seed(42)
        rng = torch.Generator(); rng.manual_seed(42)

        budget = _Budget(self.total_budget_s)
        budget.log("start", f"benchmark={benchmark.name}")

        plc = PlcLookup().load(benchmark)
        if plc is None:
            _log("warning: PlacementCost unavailable; returning legalized handoff")
            return _safe_legalize(
                benchmark.macro_positions.float(), benchmark
            )

        gp = CustomGP(benchmark)

        # FastEval calibration: 5-7 oracle calls.
        fast: Optional[FastEvaluator]
        try:
            fast = FastEvaluator(benchmark, plc)
            r = fast.calibrate(benchmark, plc, n_samples=6)
            if r < 0.85:
                _log(
                    f"fast_eval calibration r={r:.3f} (low) — "
                    "falling back to oracle scoring in burst"
                )
                fast = None
            else:
                _log(f"fast_eval calibration r={r:.3f}")
        except Exception as exc:
            _log(f"fast_eval unavailable: {exc}")
            fast = None

        # ── seed portfolio ──────────────────────────────────────────────
        portfolio = _build_seed_portfolio(
            benchmark,
            n_seeds=self.n_seeds,
            use_spectral=self.use_spectral,
            rng=rng,
        )
        _log(f"portfolio: {[lbl for lbl, _ in portfolio]}")

        # ── small-benchmark shortcut ────────────────────────────────────
        # For benchmarks where GP has shown no improvement over the
        # handoff (empirically nh≤300 on IBM), skip GP entirely and put
        # all budget into multi-restart GWTW SA. The handoff + symmetry
        # seeds remain the candidate set.
        # Configurable via TORCH_PLACER_SKIP_GP_NH (default 300, set to 0
        # to always run GP).
        skip_gp_nh = _env_int("TORCH_PLACER_SKIP_GP_NH", 300)
        small_bench = (
            skip_gp_nh > 0 and int(benchmark.num_hard_macros) <= skip_gp_nh
        )
        if small_bench:
            _log(
                f"small bench (nh={benchmark.num_hard_macros}); skipping GP, "
                "feeding handoff + symmetry seeds directly to legalize+SA"
            )
            portfolio_small = _build_seed_portfolio(
                benchmark,
                n_seeds=min(self.n_seeds, 6),
                use_spectral=False,
                rng=rng,
            )
            scored = _legalize_and_score(
                [("handoff", benchmark.macro_positions.float())] + portfolio_small,
                benchmark,
                plc,
            )
            if not scored:
                _log("small-bench: no valid candidates; returning legalized handoff")
                return _safe_legalize(
                    benchmark.macro_positions.float(), benchmark
                )
            best_label, best_placement, best_cost = scored[0]
            scores_str = ", ".join(f"{r[0]}={r[2]:.4f}" for r in scored[:6])
            _log(f"small-bench top scored: {scores_str}")
            budget.log(
                "small-bench oracle select",
                f"best={best_label} proxy={best_cost:.4f}",
            )
            return self._multi_restart_gwtw(
                scored, best_placement, best_cost, best_label,
                benchmark, plc, budget,
            )

        # ── early-kill burst ────────────────────────────────────────────
        # GP is exploratory; cap its share so GWTW SA gets the bulk.
        burst_share = 0.12
        burst_budget = budget.remaining() * burst_share
        survivors = _gp_burst(
            gp,
            portfolio,
            burst_iters=self.burst_iters,
            device=self.device,
            benchmark=benchmark,
            fast=fast,
            keep=self.keep,
            time_budget_s=burst_budget,
        )
        budget.log("burst done", f"{len(survivors)} survivors")

        if not survivors:
            _log("no GP burst survivors — falling back to handoff legalization")
            return _safe_legalize(
                benchmark.macro_positions.float(), benchmark
            )

        # ── full GP on survivors ─────────────────────────────────────────
        # Leave generous headroom for SA: target ~55% of remaining budget
        # for SA, ~30% for full GP, ~15% for legalize+score.
        full_budget_s = max(
            45.0,
            min(
                budget.remaining() * 0.30,
                budget.remaining() - self.gwtw_seconds - 60.0,
            ),
        )
        finals = _gp_full(
            gp,
            survivors,
            full_iters=self.full_iters,
            device=self.device,
            time_budget_s=full_budget_s,
            fast=fast,
        )
        budget.log("full-GP done", f"{len(finals)} final candidates")

        # Always include the raw handoff as a guardrail.
        ct_seed = ("handoff", benchmark.macro_positions.float())
        finals_with_guard: List[Tuple[str, torch.Tensor]] = [ct_seed, *finals]

        # ── legalize + oracle score ─────────────────────────────────────
        scored = _legalize_and_score(finals_with_guard, benchmark, plc)
        if not scored:
            _log("no valid candidates after legalize — returning legalized handoff")
            return _safe_legalize(
                benchmark.macro_positions.float(), benchmark
            )

        scores_str = ", ".join(f"{r[0]}={r[2]:.4f}" for r in scored[:6])
        _log(f"top scored: {scores_str}")
        best_label, best_placement, best_cost = scored[0]
        budget.log(
            "oracle select", f"best={best_label}  proxy={best_cost:.4f}"
        )

        return self._multi_restart_gwtw(
            scored, best_placement, best_cost, best_label,
            benchmark, plc, budget,
        )

    # ------------------------------------------------------------------
    # Multi-restart GWTW SA — call SA from each of the top-K legalized
    # candidates so each gets its own basin-escape attempt; pick global best.
    # ------------------------------------------------------------------

    def _multi_restart_gwtw(
        self,
        scored: List[Tuple[str, torch.Tensor, float]],
        best_placement: torch.Tensor,
        best_cost: float,
        best_label: str,
        benchmark: Benchmark,
        plc,
        budget: _Budget,
    ) -> torch.Tensor:
        remaining = budget.remaining()
        if remaining < 60.0 or not scored:
            _log("budget too low or no candidates — skipping GWTW SA")
            budget.log(
                "done (skip SA)",
                f"best={best_label}  proxy={best_cost:.4f}",
            )
            return best_placement

        # First take the monotone wins that SA does not need to spend
        # temperature on.  CD is strictly improving under the true proxy, so
        # it is safe as a pre-polish even when it finds only a few moves.
        if self.cd_seconds > 0.0 and budget.remaining() >= 90.0:
            cd_budget = min(float(self.cd_seconds), max(0.0, budget.remaining() * 0.20))
            if cd_budget >= 30.0:
                cd_k_bound = None if self.cd_k_bound_raw < 0 else int(self.cd_k_bound_raw)
                _log(
                    f"coord-desc pre-polish: start={best_label} "
                    f"proxy={best_cost:.4f} budget_s={cd_budget:.0f}"
                )
                try:
                    cd_place, cd_proxy, cd_acc = coord_descent_polish(
                        best_placement.clone(),
                        benchmark,
                        plc,
                        time_budget_s=cd_budget,
                        max_passes=int(self.cd_passes),
                        k_distance_bound=cd_k_bound,
                        cell_search_prob=float(self.cd_cell_prob),
                        node_order="descending_size",
                        seed=(
                            20260521
                            + int(benchmark.num_macros) * 47
                            + int(benchmark.num_nets) * 13
                        ),
                        log_progress=_DEBUG,
                    )
                    if cd_proxy < best_cost - 1e-9:
                        ok, _ = validate_placement(
                            cd_place, benchmark, check_overlaps=True
                        )
                        if ok:
                            best_placement = cd_place
                            best_cost = float(cd_proxy)
                            best_label = best_label + "+cd"
                            scored = [(best_label, best_placement, best_cost), *scored]
                            _log(
                                f"coord-desc win: proxy={best_cost:.4f} "
                                f"moves={cd_acc}"
                            )
                        else:
                            _log(f"coord-desc best invalid; moves={cd_acc}")
                    else:
                        _log(f"coord-desc no win; moves={cd_acc}")
                except Exception as exc:
                    _log(f"coord-desc exception: {exc}; continuing")

        # use top-3 (or fewer) for restart diversity
        restarts = scored[: min(3, len(scored))]
        per_restart = max(40.0, (remaining - 20.0) / max(1, len(restarts)))
        t_max_aggr = max(self.gwtw_t_max, 5e-2)
        _log(
            f"multi-restart gwtw_sa: {len(restarts)} restarts "
            f"per_restart={per_restart:.0f}s t_max={t_max_aggr:.3f}"
        )
        for ridx, (rlabel, rplace, rcost) in enumerate(restarts):
            if budget.remaining() < 30.0:
                _log("budget exhausted; skipping remaining restarts")
                break
            this_budget = min(per_restart, budget.remaining() - 15.0)
            _log(
                f"gwtw[{ridx}] from={rlabel} start_proxy={rcost:.4f} "
                f"budget_s={this_budget:.0f}"
            )
            try:
                disp_budget = (
                    None if self.sa_disp_budget < 0
                    else float(self.sa_disp_budget)
                )
                sa_placement, sa_proxy, accepted, evaluated = (
                    tilos_gwtw_sa_refine(
                        rplace,
                        benchmark,
                        plc,
                        num_workers=self.gwtw_workers,
                        num_iters=self.gwtw_iters,
                        syncup_freq=self.gwtw_syncup,
                        top_k=self.gwtw_topk,
                        time_budget_s=this_budget,
                        seed=42 + ridx * 1009,
                        t_max=t_max_aggr,
                        t_min=self.gwtw_t_min,
                        log_progress=_DEBUG,
                        sa_disp_budget_frac=disp_budget,
                    )
                )
                _log(
                    f"gwtw[{ridx}] {rlabel} → proxy={sa_proxy:.4f} "
                    f"accepted={accepted}/{evaluated}"
                )
                if sa_proxy < best_cost - 1e-6:
                    try:
                        sa_proxy = float(
                            compute_proxy_cost(
                                sa_placement, benchmark, plc
                            )["proxy_cost"]
                        )
                    except Exception as exc:
                        _log(f"gwtw[{ridx}] rescore failed: {exc}; skipping")
                        continue
                    ok, _ = validate_placement(
                        sa_placement, benchmark, check_overlaps=True
                    )
                    if ok and sa_proxy < best_cost - 1e-6:
                        best_placement = sa_placement
                        best_cost = sa_proxy
                        best_label = rlabel + f"+gwtw{ridx}"
                        _log(
                            f"new global best from gwtw[{ridx}]: "
                            f"{best_cost:.4f}"
                        )
                    else:
                        _log(
                            f"gwtw[{ridx}] best invalid — skipping"
                        )
            except Exception as exc:
                _log(f"gwtw[{ridx}] exception: {exc}; continuing")

        budget.log("done", f"best={best_label}  proxy={best_cost:.4f}")
        return best_placement


# backward-compatibility alias
KoralPlacer = TorchPlacer
