#!/usr/bin/env python3
"""Run the RePlAce pipeline on one benchmark and write diagnostics JSON."""

from __future__ import annotations

import argparse
import json
import sys
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
from _replace_pipeline import ReplacePipeline  # noqa: E402
from _replace_runner import ReplaceConfig  # noqa: E402


IBM_ROOT = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
DEFAULT_BINARY = (
    REPO_ROOT / "external" / "MacroPlacement" / "Flows" / "util" / "RePlAceFlow" / "RePlAce-static"
)


def main() -> int:
    args = _parse_args()
    bench_dir = _benchmark_dir(args.benchmark)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    pipeline = ReplacePipeline(
        configs=_parse_configs(args.config),
        work_root=args.work_root,
        binary_path=args.binary,
        timeout_seconds=args.timeout,
        scale=args.scale,
    )
    result = pipeline.run(benchmark)
    ok, violations = validate_placement(result.placement, benchmark)
    costs = compute_proxy_cost(result.placement, benchmark, plc)

    diagnostics = result.diagnostics(benchmark.name)
    diagnostics["features"] = benchmark_features(benchmark)
    diagnostics["final"] = {
        "valid": bool(ok),
        "violations": list(violations),
        "proxy_cost": float(costs["proxy_cost"]),
        "wirelength": float(costs["wirelength_cost"]),
        "density": float(costs["density_cost"]),
        "congestion": float(costs["congestion_cost"]),
        "overlaps": int(costs["overlap_count"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    selected = diagnostics.get("selected_label") or "none"
    print(
        f"{benchmark.name}: reason={result.reason} selected={selected} "
        f"proxy={costs['proxy_cost']:.6f} valid={ok} overlaps={costs['overlap_count']}"
    )
    print(f"diagnostics: {args.output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RePlAce pipeline diagnostics for one IBM benchmark."
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
        metavar="DENSITY:PCOFMAX[:ARG=VALUE,...]",
        help=(
            "RePlAce config. Repeat for multi-start, e.g. --config 0.78:1.03 "
            "or --config 0.86:1.03:pcofmin=0.9,overflow=0.08."
        ),
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/tmp/macro_place_replace_diagnostics"),
    )
    parser.add_argument("--output", type=Path, default=Path("replace_diagnostics.json"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--scale", type=int, default=1000)
    return parser.parse_args()


def _benchmark_dir(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    return IBM_ROOT / name_or_path


def _parse_configs(values: list[str] | None) -> list[ReplaceConfig] | None:
    if not values:
        return None
    configs: list[ReplaceConfig] = []
    for raw in values:
        try:
            parts = raw.split(":", 2)
            if len(parts) not in {2, 3}:
                raise ValueError
            density_s, pcof_s = parts[:2]
            extra_args = _parse_extra_args(parts[2]) if len(parts) == 3 else []
            configs.append(
                ReplaceConfig(
                    density=float(density_s),
                    pcofmax=float(pcof_s),
                    extra_args=extra_args,
                )
            )
        except ValueError as exc:
            raise SystemExit(
                f"invalid --config {raw!r}; expected DENSITY:PCOFMAX[:ARG=VALUE,...]"
            ) from exc
    return configs


def _parse_extra_args(raw: str) -> list[str]:
    args: list[str] = []
    if not raw:
        return args
    for item in raw.split(","):
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError
        args.extend([f"-{key}", value])
    return args


if __name__ == "__main__":
    raise SystemExit(main())
