#!/usr/bin/env python3
"""Bayesian-style hyperparameter search for DREAMPlace pipeline knobs (Optuna / TPE).

Design choices
--------------
1. **Framework: Optuna** with ``TPESampler`` (optionally multivariate) — sample-efficient
   on mixed spaces; SQLite resume.

2. **Objective: one global policy** — the same parameters on every benchmark in the
   calibration set; loss = **mean or median** proxy (+ optional runtime penalty). No
   per-benchmark-name parameters (rule-safe).

3. **Invalid / failures** — failed DREAMPlace or invalid placement → large penalty
   for that benchmark.

4. **DREAMPlace JSON merge** — ``learning_rate``, ``optimizer``, and sub-iter knobs are
   merged into ``global_place_stages[0]`` (see ``deep_merge_dreamplace_json``); top-level
   keys like ``density_weight`` / ``gamma`` stay top-level.

5. **Calibration set** — if ``--benchmarks`` is omitted, use an **evenly spaced** subset
   of ICCAD04 directories (sorted names, *not* hand-picked identities) so the search
   sees small/medium/large cases without optimizing one bench.

6. **Pruning** — optional ``MedianPruner`` with intermediate reports after each
   benchmark (running mean) to drop bad trials early on long sweeps.

7. **Fidelity** — moderate ``global_iterations`` / ``num_starts`` for search; promote
   winners and re-run full ``evaluate --all`` at production budgets.

**Expectations:** Tuning can materially improve the DREAMPlace pipeline, but the gap
to leaderboard-leading hybrid flows (e.g. strong global + RePlAce-style candidates) may
remain architectural — use this to maximize the DREAMPlace-only branch, not as a
guarantee of rank #1.

Usage (repo root, env with torch + optuna)::

  uv sync --extra tuning
  uv run python scripts/tune_dreamplace_optuna.py --n-trials 64
  uv run python scripts/tune_dreamplace_optuna.py --n-trials 64 --write-best-config
  uv run python scripts/tune_dreamplace_optuna.py --benchmarks ibm01,ibm04,ibm12 --n-trials 30

``--write-best-config`` writes JSON suitable for ``MACRO_PLACE_DP_CONFIG`` on
``submissions/dreamplace_pipeline_placer.py`` (see that file).

Progress lines ``[tune]`` / ``[tune:dp]`` go to **stderr** (line-buffered) so you still
see output when stdout is piped or fully buffered. Without a TTY, the Optuna tqdm bar
is disabled and only ``[tune]`` lines are used. Use ``--quiet`` for less detail;
``-v`` / ``--verbose`` for full params + tracebacks.

"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from _dreamplace_pipeline import DreamPlacePipeline

# Repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUBMISSIONS = REPO_ROOT / "submissions"
if str(SUBMISSIONS) not in sys.path:
    sys.path.insert(0, str(SUBMISSIONS))


def _unbuffer_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def _err(msg: str) -> None:
    """Progress goes to stderr so pipes/SLURM/logs still show movement (stdout is often fully buffered)."""

    print(msg, file=sys.stderr, flush=True)


def default_calibration_benchmarks(repo_root: Path, k: int = 6) -> list[str]:
    """Evenly spaced ICCAD04 case names (sorted lexicographically on disk)."""

    iccad = repo_root / "external/MacroPlacement/Testcases/ICCAD04"
    names = sorted(p.name for p in iccad.iterdir() if p.is_dir())
    if not names:
        raise FileNotFoundError(f"no ICCAD04 benchmarks under {iccad}")
    if len(names) <= k:
        return names
    return [names[round(i * (len(names) - 1) / (k - 1))] for i in range(k)]


def _load_benchmark_pair(name: str):
    from macro_place.loader import load_benchmark_from_dir

    root = REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04" / name
    if not root.is_dir():
        raise FileNotFoundError(f"missing {root}")
    return load_benchmark_from_dir(str(root))


def _aggregate(costs: list[float], mode: str) -> float:
    if not costs:
        return float("inf")
    if mode == "median":
        return float(statistics.median(costs))
    return float(statistics.mean(costs))


def split_optuna_params_to_pipeline_and_overrides(
    params: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map flat Optuna ``trial.params`` to :class:`DreamPlacePipeline` kwargs + JSON overrides."""

    p = dict(params)
    pipeline: dict[str, Any] = {
        "num_starts": int(p["num_starts"]),
        "jitter_sigma_um": float(p["jitter_sigma_um"]),
        "global_iterations": int(p["global_iterations"]),
        "num_bins": int(p["num_bins"]),
        "num_threads": int(p["num_threads"]),
        "target_density": float(p["target_density"]),
        "timeout_seconds": float(timeout_seconds),
        "scale_iterations_with_features": True,
    }
    overrides: dict[str, Any] = {
        "density_weight": float(p["density_weight"]),
        "gamma": float(p["gamma"]),
        "stop_overflow": float(p["stop_overflow"]),
        "gp_noise_ratio": float(p["gp_noise_ratio"]),
        "random_center_init_flag": int(p["random_center_init_flag"]),
        "enable_fillers": int(p["enable_fillers"]),
        "scale_factor": float(p["scale_factor"]),
        "ignore_net_degree": int(p["ignore_net_degree"]),
        "global_place_stages": [
            {
                "learning_rate": float(p["learning_rate"]),
                "Llambda_density_weight_iteration": int(
                    p["Llambda_density_weight_iteration"]
                ),
                "Lsub_iteration": int(p["Lsub_iteration"]),
                "wirelength": str(p["wirelength"]),
                "optimizer": str(p["optimizer"]),
            }
        ],
    }
    return pipeline, overrides


def dreamplace_pipeline_from_optuna_params(
    params: Mapping[str, Any],
    *,
    timeout_seconds: float,
    use_gpu: bool | None,
) -> DreamPlacePipeline:
    """Construct ``DreamPlacePipeline`` from a flat Optuna param dict."""

    from _dreamplace_pipeline import DreamPlacePipeline

    pipe_kw, overrides = split_optuna_params_to_pipeline_and_overrides(
        params, timeout_seconds=timeout_seconds
    )
    return DreamPlacePipeline(
        **pipe_kw,
        dreamplace_json_overrides=overrides,
        use_gpu=use_gpu,
    )


def build_best_config_payload(
    params: Mapping[str, Any],
    *,
    study_name: str,
    best_value: float,
    aggregate: str,
    calibration_benchmarks: list[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    """JSON-serializable blob for ``MACRO_PLACE_DP_CONFIG`` + metadata."""

    pipe_kw, overrides = split_optuna_params_to_pipeline_and_overrides(
        params, timeout_seconds=timeout_seconds
    )
    # Eval harness should follow MACRO_PLACE_DP_GPU unless overridden in JSON later.
    pipe_out = dict(pipe_kw)
    pipe_out["use_gpu"] = None
    return {
        "schema_version": 1,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "study_name": study_name,
        "best_value": float(best_value),
        "aggregate": aggregate,
        "calibration_benchmarks": list(calibration_benchmarks),
        "pipeline": pipe_out,
        "dreamplace_json_overrides": overrides,
    }


def main() -> None:
    try:
        import optuna
        from optuna.pruners import MedianPruner
    except ImportError as e:
        raise SystemExit(
            "Install Optuna: uv sync --extra tuning   (or pip install optuna)"
        ) from e

    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="",
        help=(
            "Comma-separated ICCAD04 names (same params for all). "
            "Empty = evenly spaced subset of all ICCAD04 cases (see module doc)."
        ),
    )
    parser.add_argument("--n-trials", type=int, default=64)
    parser.add_argument(
        "--study-name",
        type=str,
        default="dreamplace_proxy_mean",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=f"sqlite:///{REPO_ROOT}/tuning_logs/optuna_dreamplace.db?timeout=120",
        help="Optuna storage URL (directory created as needed).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timeout-per-run",
        type=float,
        default=900.0,
        help=(
            "Subprocess timeout per DREAMPlace *start* (seconds). CPU builds often "
            "need >420s for slow optimizer/wirelength combos; use 1200+ if trials still time out."
        ),
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=1e4,
        help="Proxy substitute when a benchmark fails or placement invalid.",
    )
    parser.add_argument(
        "--runtime-target",
        type=float,
        default=600.0,
        help="Target seconds per benchmark; penalty applies beyond this.",
    )
    parser.add_argument(
        "--runtime-weight",
        type=float,
        default=0.1,
        help="Penalty weight for runtime overruns (added to proxy cost).",
    )
    parser.add_argument(
        "--aggregate",
        choices=("mean", "median"),
        default="mean",
        help="Reduce per-benchmark costs to a single trial objective.",
    )
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=6,
        help="When --benchmarks is empty, use this many evenly spaced ICCAD04 cases.",
    )
    parser.add_argument(
        "--no-pruner",
        action="store_true",
        help="Disable MedianPruner (full evaluation every trial).",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=10,
        help="Trials completed before MedianPruner activates.",
    )
    parser.add_argument(
        "--tpe-startup-trials",
        type=int,
        default=12,
        help="Random exploration trials before TPE narrows the search.",
    )
    parser.add_argument(
        "--multivariate-tpe",
        action="store_true",
        help="Use multivariate TPE (slower per step, can help correlated knobs).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force DREAMPlace CPU even if a GPU is visible.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve benchmarks and tuner config, import Optuna/pipeline, then exit (no trials).",
    )
    parser.add_argument(
        "--write-best-config",
        type=str,
        nargs="?",
        const=str(REPO_ROOT / "tuning_logs/dreamplace_optuna_best.json"),
        default=None,
        help=(
            "After optimization, write best trial to this JSON file (for "
            "`MACRO_PLACE_DP_CONFIG`). Pass flag alone for tuning_logs/dreamplace_optuna_best.json."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less detail: only one stderr line per trial (start/end); no per-benchmark spam.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full trial.params and tracebacks on failure.",
    )
    args = parser.parse_args()
    _unbuffer_stdio()

    if args.benchmarks.strip():
        names = [n.strip() for n in args.benchmarks.split(",") if n.strip()]
    else:
        names = default_calibration_benchmarks(REPO_ROOT, k=max(2, args.calibration_count))
    if not names:
        raise SystemExit("no benchmarks")

    if args.dry_run:
        print("dry-run: OK")
        print("  benchmarks:", ",".join(names))
        print("  n_trials:", args.n_trials)
        print("  aggregate:", args.aggregate)
        print("  storage:", args.storage)
        print("  use_gpu:", "forced CPU" if args.cpu else "auto/MACRO_PLACE_DP_GPU")
        print("  write_best_config:", args.write_best_config or "(disabled)")
        print("  quiet:", args.quiet)
        print("  verbose:", args.verbose)
        return

    if args.quiet:
        os.environ["MACRO_PLACE_TUNER_DEBUG"] = ""
    else:
        os.environ["MACRO_PLACE_TUNER_DEBUG"] = "1"

    storage_path = args.storage
    if storage_path.startswith("sqlite:///"):
        db = storage_path.replace("sqlite:///", "", 1).split("?", 1)[0]
        Path(db).parent.mkdir(parents=True, exist_ok=True)

    use_gpu: bool | None
    if args.cpu:
        use_gpu = False
    else:
        use_gpu = None

    pruner: optuna.pruners.BasePruner | None
    if args.no_pruner:
        pruner = None
    else:
        pruner = MedianPruner(
            n_startup_trials=max(3, args.pruner_startup_trials),
            interval_steps=1,
        )

    _err(
        f"[tune] study={args.study_name!r}  trials={args.n_trials}  "
        f"benches={len(names)} {names[:3]}{'...' if len(names) > 3 else ''}  "
        f"timeout_per_placer={args.timeout_per_run}s  gpu={'off' if args.cpu else 'auto'}"
    )

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("target_density", 0.66, 0.90)
        trial.suggest_float("density_weight", 2e-5, 3.0e-4, log=True)
        trial.suggest_float("gamma", 2.4, 6.2)
        trial.suggest_int("global_iterations", 24, 120)
        trial.suggest_categorical("num_bins", [64, 128])
        trial.suggest_int("num_starts", 1, 4)
        trial.suggest_float("jitter_sigma_um", 0.008, 0.060)
        trial.suggest_categorical("num_threads", [4, 8, 12])
        trial.suggest_float("learning_rate", 0.004, 0.022, log=True)
        trial.suggest_float("stop_overflow", 0.08, 0.18)
        trial.suggest_float("gp_noise_ratio", 0.01, 0.05)
        trial.suggest_categorical("random_center_init_flag", [0, 1])
        trial.suggest_int("Llambda_density_weight_iteration", 1, 4)
        trial.suggest_int("Lsub_iteration", 1, 3)
        trial.suggest_categorical("wirelength", ["weighted_average", "logsumexp"])
        trial.suggest_categorical("optimizer", ["nesterov", "adamw", "yogi"])
        trial.suggest_categorical("enable_fillers", [0, 1])
        trial.suggest_float("scale_factor", 0.88, 1.12)
        trial.suggest_categorical("ignore_net_degree", [80, 100, 200])

        pipeline = dreamplace_pipeline_from_optuna_params(
            trial.params,
            timeout_seconds=args.timeout_per_run,
            use_gpu=use_gpu,
        )

        _err(
            f"[tune] trial {trial.number}/{args.n_trials}  START  "
            f"starts={pipeline.num_starts}  iters={pipeline.global_iterations}  "
            f"bins={pipeline.num_bins}"
        )
        if args.verbose:
            _err(f"[tune] trial {trial.number}  params={trial.params}")

        costs: list[float] = []
        for step, name in enumerate(names):
            try:
                benchmark, plc = _load_benchmark_pair(name)
            except FileNotFoundError:
                if not args.quiet:
                    _err(f"[tune] trial {trial.number}  {name}  SKIP (missing dir)")
                costs.append(float(args.penalty))
            else:
                if not args.quiet:
                    _err(
                        f"[tune] trial {trial.number}  {name}  ({step + 1}/{len(names)})  "
                        f"pipeline.run ... {time.strftime('%H:%M:%S')}"
                    )
                start_t = time.perf_counter()
                try:
                    result = pipeline.run(benchmark)
                except subprocess.TimeoutExpired as ex:
                    costs.append(float(args.penalty))
                    if not args.quiet:
                        _err(f"[tune] trial {trial.number}  {name}  TIMEOUT {ex.timeout}s")
                except Exception as ex:
                    costs.append(float(args.penalty))
                    if not args.quiet:
                        _err(
                            f"[tune] trial {trial.number}  {name}  ERROR {type(ex).__name__}: {ex}"
                        )
                    if args.verbose:
                        import traceback

                        traceback.print_exc()
                else:
                    elapsed = time.perf_counter() - start_t
                    placement = result.placement
                    ok, _ = validate_placement(
                        placement, benchmark, check_overlaps=True
                    )
                    if not ok:
                        costs.append(float(args.penalty))
                        if not args.quiet:
                            _err(
                                f"[tune] trial {trial.number}  {name}  "
                                f"invalid_placement  {elapsed:.1f}s  reason={result.reason!r}"
                            )
                    else:
                        try:
                            c = float(
                                compute_proxy_cost(placement, benchmark, plc)[
                                    "proxy_cost"
                                ]
                            )
                        except Exception:
                            c = float(args.penalty)
                        runtime_penalty = 0.0
                        if args.runtime_weight > 0.0:
                            over = max(
                                0.0, float(elapsed) - float(args.runtime_target)
                            )
                            denom = max(1.0, float(args.runtime_target))
                            runtime_penalty = float(args.runtime_weight) * (
                                over / denom
                            )
                        costs.append(c + runtime_penalty)
                        if not args.quiet:
                            _err(
                                f"[tune] trial {trial.number}  {name}  "
                                f"proxy={c + runtime_penalty:.4f}  {elapsed:.1f}s  "
                                f"reason={result.reason!r}"
                            )

            if pruner is not None:
                intermediate = _aggregate(costs, "mean")
                trial.report(intermediate, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        obj = _aggregate(costs, args.aggregate)
        _err(f"[tune] trial {trial.number}/{args.n_trials}  END  objective={obj:.6f}")
        return obj

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=args.multivariate_tpe,
        n_startup_trials=max(5, args.tpe_startup_trials),
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    # tqdm needs a TTY; otherwise it prints nothing and looks "stuck".
    _use_bar = (not args.quiet) and sys.stderr.isatty()
    if not _use_bar:
        _err("[tune] (no TTY — progress via [tune] lines on stderr; not using tqdm bar)")

    # Default catch=() would abort the whole study on any uncaught error.
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=_use_bar,
        catch=(Exception,),
    )

    try:
        t = study.best_trial
    except ValueError:
        print("No completed trials; cannot report best or write config.")
        return

    print("Best trial:")
    print("  value:", t.value)
    print("  params:", t.params)

    if args.write_best_config:
        out_path = Path(args.write_best_config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_best_config_payload(
            t.params,
            study_name=args.study_name,
            best_value=float(t.value),
            aggregate=args.aggregate,
            calibration_benchmarks=names,
            timeout_seconds=args.timeout_per_run,
        )
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Wrote best config:", out_path.resolve())
        print(f"Load in the placer: MACRO_PLACE_DP_CONFIG={out_path.resolve()}")


if __name__ == "__main__":
    # Avoid thread oversubscription in child DREAMPlace
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    sys.stderr.write(
        "tune_dreamplace_optuna: starting (stderr is line-buffered; watch this stream)\n"
    )
    sys.stderr.flush()
    main()
