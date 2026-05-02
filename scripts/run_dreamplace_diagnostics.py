#!/usr/bin/env python3
"""Run DREAMPlace candidates on one benchmark and write diagnostics JSON."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
for path in (REPO_ROOT, SUBMISSIONS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402
from _benchmark_features import benchmark_features  # noqa: E402
from _candidate_select import select_best_true_proxy  # noqa: E402
from _dreamplace_candidates import generate_dreamplace_candidates  # noqa: E402
from _dreamplace_runner import DreamPlaceConfig, dreamplace_available  # noqa: E402
from casadi_placer import CasadiPlacer  # noqa: E402


IBM_ROOT = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
DEFAULT_DREAMPLACE_ROOT = REPO_ROOT / "external" / "DREAMPlace"


def main() -> int:
    args = _parse_args()
    bench_dir = _benchmark_dir(args.benchmark)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    ok, reason = dreamplace_available(args.dreamplace_root)
    if not ok:
        raise SystemExit(f"DREAMPlace unavailable: {reason}")

    baseline = CasadiPlacer().place(benchmark).clone().float()
    configs = _parse_configs(args.config, preset=args.preset)
    work_root = args.work_root / benchmark.name

    batch = generate_dreamplace_candidates(
        benchmark,
        plc,
        work_root,
        configs,
        bookshelf_name=benchmark.name,
        scale=args.scale,
        dreamplace_root=args.dreamplace_root,
        timeout_seconds=args.timeout,
        initial_placement=baseline,
        soft_macro_mode=args.soft_macro_mode,
        blend_alphas=args.blend_alpha,
    )
    selection = select_best_true_proxy(
        baseline,
        [candidate.placement for candidate in batch.candidates],
        benchmark,
        plc,
        candidate_labels=[candidate.label for candidate in batch.candidates],
    )

    final = selection.placement
    valid, violations = validate_placement(final, benchmark)
    costs = compute_proxy_cost(final, benchmark, plc)
    diagnostics = {
        "benchmark": benchmark.name,
        "features": benchmark_features(benchmark),
        "selected_label": selection.best.label,
        "num_candidates": len(batch.candidates),
        "bookshelf_name": batch.export.bookshelf_name,
        "runs": [
            {
                "target_density": float(run.config.target_density),
                "iterations": int(run.config.iterations),
                "learning_rate": float(run.config.learning_rate),
                "density_weight": float(run.config.density_weight),
                "extra_params": run.config.extra_params or {},
                "returncode": int(run.returncode),
                "timed_out": bool(run.timed_out),
                "ok": bool(run.ok),
                "runtime_seconds": float(run.runtime_seconds),
                "num_pl_paths": len(run.pl_paths),
                "log_path": str(run.log_path),
            }
            for run in batch.run_results
        ],
        "candidates": [
            {
                "label": candidate.label,
                "pl_path": str(candidate.pl_path),
                "run_target_density": float(candidate.run_result.config.target_density),
                "run_iterations": int(candidate.run_result.config.iterations),
                "raw_overlap_count": int(candidate.raw_overlap_count),
                "final_overlap_count": int(candidate.final_overlap_count),
                "legalizer_max_displacement": float(
                    candidate.legalizer_max_displacement
                ),
                "legalizer_mean_displacement": float(
                    candidate.legalizer_mean_displacement
                ),
            }
            for candidate in batch.candidates
        ],
        "scores": [
            {
                "label": score.label,
                "valid": bool(score.valid),
                "proxy_cost": float(score.proxy_cost),
                "wirelength": float(score.wirelength),
                "density": float(score.density),
                "congestion": float(score.congestion),
                "overlaps": int(score.overlaps),
                "violations": list(score.violations),
            }
            for score in selection.scores
        ],
        "final": {
            "valid": bool(valid),
            "violations": list(violations),
            "proxy_cost": float(costs["proxy_cost"]),
            "wirelength": float(costs["wirelength_cost"]),
            "density": float(costs["density_cost"]),
            "congestion": float(costs["congestion_cost"]),
            "overlaps": int(costs["overlap_count"]),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    print(
        f"{benchmark.name}: selected={selection.best.label} "
        f"proxy={costs['proxy_cost']:.6f} valid={valid} overlaps={costs['overlap_count']}"
    )
    print(f"diagnostics: {args.output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DREAMPlace diagnostics for one IBM benchmark."
    )
    parser.add_argument(
        "--benchmark",
        default="ibm01",
        help="IBM benchmark name such as ibm01, or a benchmark directory path.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="DENSITY:ITERATIONS[:LR[:DENSITY_WEIGHT]]",
        help=(
            "DREAMPlace config. Repeat for multi-start, e.g. --config 0.80:200 "
            "or --config 0.75:200:0.01:1e-3."
        ),
    )
    parser.add_argument("--dreamplace-root", type=Path, default=DEFAULT_DREAMPLACE_ROOT)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "macro_place_dreamplace_diagnostics",
    )
    parser.add_argument("--output", type=Path, default=Path("dreamplace_diagnostics.json"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument(
        "--soft-macro-mode",
        choices=("preserve", "row_height"),
        default="preserve",
        help="Bookshelf geometry for soft macros.",
    )
    parser.add_argument(
        "--blend-alpha",
        action="append",
        type=float,
        default=[],
        help="Also score baseline-to-DREAMPlace blended candidates at this alpha.",
    )
    parser.add_argument(
        "--preset",
        choices=(
            "basic",
            "global_only",
            "random_global",
            "macro",
            "macro_global",
            "macro_random_global",
            "macro_bb",
            "macro_bb_global",
            "gift",
            "routability",
        ),
        default="basic",
        help="General DREAMPlace parameter preset to apply to every config.",
    )
    return parser.parse_args()


def _benchmark_dir(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    return IBM_ROOT / name_or_path


def _parse_configs(values: list[str] | None, *, preset: str) -> list[DreamPlaceConfig]:
    if not values:
        return [
            DreamPlaceConfig(
                target_density=0.76,
                iterations=200,
                gpu=False,
                extra_params=_preset_params(preset),
            ),
            DreamPlaceConfig(
                target_density=0.82,
                iterations=200,
                gpu=False,
                extra_params=_preset_params(preset),
            ),
        ]
    configs: list[DreamPlaceConfig] = []
    for raw in values:
        try:
            parts = raw.split(":")
            if len(parts) not in {2, 3, 4}:
                raise ValueError
            density_s, iter_s = parts[:2]
            lr = float(parts[2]) if len(parts) >= 3 else 0.01
            density_weight = float(parts[3]) if len(parts) >= 4 else 8e-5
            configs.append(
                DreamPlaceConfig(
                    target_density=float(density_s),
                    iterations=int(iter_s),
                    learning_rate=lr,
                    density_weight=density_weight,
                    gpu=False,
                    extra_params=_preset_params(preset),
                )
            )
        except ValueError as exc:
            raise SystemExit(
                f"invalid --config {raw!r}; expected DENSITY:ITERATIONS[:LR[:DENSITY_WEIGHT]]"
            ) from exc
    return configs


def _preset_params(preset: str) -> dict:
    if preset == "basic":
        return {}
    if preset == "global_only":
        return {
            "legalize_flag": 0,
        }
    if preset == "random_global":
        return {
            "legalize_flag": 0,
            "random_center_init_flag": 1,
        }
    if preset == "macro":
        return {
            "macro_place_flag": 1,
            "two_stage_density_scaler": 1000,
        }
    if preset == "macro_global":
        return {
            "macro_place_flag": 1,
            "two_stage_density_scaler": 1000,
            "legalize_flag": 0,
        }
    if preset == "macro_random_global":
        return {
            "macro_place_flag": 1,
            "two_stage_density_scaler": 1000,
            "legalize_flag": 0,
            "random_center_init_flag": 1,
        }
    if preset == "macro_bb":
        return {
            "macro_place_flag": 1,
            "use_bb": 1,
            "two_stage_density_scaler": 1000,
        }
    if preset == "macro_bb_global":
        return {
            "macro_place_flag": 1,
            "use_bb": 1,
            "two_stage_density_scaler": 1000,
            "legalize_flag": 0,
        }
    if preset == "gift":
        return {
            "gift_init_flag": 1,
            "gift_init_scale": 0.7,
        }
    if preset == "routability":
        return {
            "routability_opt_flag": 1,
            "route_num_bins_x": 64,
            "route_num_bins_y": 64,
            "adjust_rudy_area_flag": 1,
            "adjust_pin_area_flag": 1,
        }
    raise ValueError(f"unknown preset {preset!r}")


if __name__ == "__main__":
    raise SystemExit(main())
