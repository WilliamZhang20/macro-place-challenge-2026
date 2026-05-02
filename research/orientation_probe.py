#!/usr/bin/env python3
"""Probe whether legal hard-macro orientation flips move proxy cost."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SUBMISSIONS = ROOT / "submissions"
if str(SUBMISSIONS) not in sys.path:
    sys.path.insert(0, str(SUBMISSIONS))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402


def _load_placer(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import placer {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in vars(mod).values():
        if isinstance(attr, type) and attr.__module__ == path.stem and hasattr(attr, "place"):
            return attr()
    raise RuntimeError(f"no placer class with place() in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placer", default="submissions/casadi_placer.py")
    parser.add_argument("--benchmark", default="ibm01")
    parser.add_argument("--macro", type=int, default=0)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    bench_dir = ROOT / "external/MacroPlacement/Testcases/ICCAD04" / args.benchmark
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placement = _load_placer(ROOT / args.placer).place(benchmark)
    baseline = compute_proxy_cost(placement, benchmark, plc)

    macro_indices = list(range(min(args.top, benchmark.num_hard_macros)))
    if args.macro not in macro_indices and args.macro < benchmark.num_hard_macros:
        macro_indices.insert(0, args.macro)

    print(
        f"baseline proxy={baseline['proxy_cost']:.6f} wl={baseline['wirelength_cost']:.6f} "
        f"den={baseline['density_cost']:.6f} cong={baseline['congestion_cost']:.6f}"
    )
    print("macro,name,orientation,proxy,wl,density,congestion,delta")
    for macro_i in macro_indices:
        rows = []
        for orient in ("N", "FN", "FS", "S"):
            # PlacementCost orientation updates mutate pin offsets in-place, so
            # use a fresh PLC per trial to avoid cumulative flip artifacts.
            _, plc_trial = load_benchmark_from_dir(str(bench_dir))
            plc_idx = benchmark.hard_macro_indices[macro_i]
            plc_trial.update_macro_orientation(plc_idx, orient)
            costs = compute_proxy_cost(placement, benchmark, plc_trial)
            rows.append((costs["proxy_cost"], orient, costs))
        for proxy, orient, costs in sorted(rows):
            print(
                f"{macro_i},{benchmark.macro_names[macro_i]},{orient},"
                f"{proxy:.6f},{costs['wirelength_cost']:.6f},"
                f"{costs['density_cost']:.6f},{costs['congestion_cost']:.6f},"
                f"{proxy - baseline['proxy_cost']:.6f}"
            )


if __name__ == "__main__":
    main()
