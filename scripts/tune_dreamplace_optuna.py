#!/usr/bin/env python3
"""Bayesian-style hyperparameter search for DREAMPlace pipeline knobs (Optuna / TPE).

Design choices
--------------
1. **Framework: Optuna** with ``TPESampler`` — sample-efficient on mixed continuous /
   integer spaces, mature storage/resume, no custom GP code. (Alternatives: BoTorch
   for pure GP-EI; heavier dependency surface.)

2. **Objective: one global policy** — the same suggested parameters are evaluated on
   *every* benchmark in ``--benchmarks``; the trial loss is the **mean proxy cost**
   (contest objective). No per-benchmark-name parameters (rule-safe).

3. **Invalid / failures** — any failed DREAMPlace start set or invalid final
   placement yields a large penalty (``1e4``) for that benchmark; mean reflects
   unreliability.

4. **Fidelity** — default search caps ``global_iterations`` and ``num_starts`` for
   tuning speed. After search, promote winners with ``--confirm-iters`` /
   production placer defaults and re-run ``evaluate --all``.

5. **Storage** — SQLite study DB for resume (``--storage sqlite:///...``).

Usage (repo root, venv with torch + optuna)::

  uv sync --extra tuning
  uv run python scripts/tune_dreamplace_optuna.py --benchmarks ibm01,ibm02 --n-trials 30

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUBMISSIONS = REPO_ROOT / "submissions"
if str(SUBMISSIONS) not in sys.path:
    sys.path.insert(0, str(SUBMISSIONS))


def _load_benchmark_pair(name: str):
    from macro_place.loader import load_benchmark_from_dir

    root = REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04" / name
    if not root.is_dir():
        raise FileNotFoundError(f"missing {root}")
    return load_benchmark_from_dir(str(root))


def main() -> None:
    try:
        import optuna
    except ImportError as e:
        raise SystemExit(
            "Install Optuna: uv sync --extra tuning   (or pip install optuna)"
        ) from e

    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    from _dreamplace_pipeline import DreamPlacePipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="ibm01,ibm02",
        help="Comma-separated ICCAD04 benchmark names (same params for all).",
    )
    parser.add_argument("--n-trials", type=int, default=32)
    parser.add_argument(
        "--study-name",
        type=str,
        default="dreamplace_proxy_mean",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=f"sqlite:///{REPO_ROOT}/tuning_logs/optuna_dreamplace.db",
        help="Optuna storage URL (directory created as needed).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timeout-per-run",
        type=float,
        default=420.0,
        help="Subprocess timeout per DREAMPlace start (seconds).",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=1e4,
        help="Proxy substitute when a benchmark fails or placement invalid.",
    )
    args = parser.parse_args()
    names = [n.strip() for n in args.benchmarks.split(",") if n.strip()]
    if not names:
        raise SystemExit("no benchmarks")

    storage_path = args.storage
    if storage_path.startswith("sqlite:///"):
        db = storage_path.replace("sqlite:///", "", 1)
        Path(db).parent.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        target_density = trial.suggest_float("target_density", 0.66, 0.90)
        density_weight = trial.suggest_float("density_weight", 4e-5, 1.6e-4, log=True)
        gamma = trial.suggest_float("gamma", 2.8, 5.5)
        global_iterations = trial.suggest_int("global_iterations", 22, 72)
        num_bins = trial.suggest_categorical("num_bins", [64, 128])
        num_starts = trial.suggest_int("num_starts", 2, 4)
        jitter_sigma_um = trial.suggest_float("jitter_sigma_um", 0.012, 0.055)
        num_threads = trial.suggest_categorical("num_threads", [4, 8])

        pipeline = DreamPlacePipeline(
            num_starts=num_starts,
            jitter_sigma_um=jitter_sigma_um,
            global_iterations=global_iterations,
            num_bins=int(num_bins),
            num_threads=int(num_threads),
            target_density=target_density,
            timeout_seconds=args.timeout_per_run,
            dreamplace_json_overrides={
                "density_weight": float(density_weight),
                "gamma": float(gamma),
            },
            scale_iterations_with_features=True,
        )

        costs: list[float] = []
        for name in names:
            try:
                benchmark, plc = _load_benchmark_pair(name)
            except FileNotFoundError:
                costs.append(float(args.penalty))
                continue
            result = pipeline.run(benchmark)
            placement = result.placement
            ok, _ = validate_placement(placement, benchmark, check_overlaps=True)
            if not ok:
                costs.append(float(args.penalty))
                continue
            try:
                c = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
            except Exception:
                c = float(args.penalty)
            costs.append(c)

        return float(sum(costs) / max(1, len(costs)))

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("Best trial:")
    t = study.best_trial
    print("  value:", t.value)
    print("  params:", t.params)


if __name__ == "__main__":
    # Avoid thread oversubscription in child DREAMPlace
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
