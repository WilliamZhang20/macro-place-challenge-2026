#!/usr/bin/env python3
"""Compare the local Rudy surrogate to PlacementCost routing congestion maps."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SUBMISSIONS = ROOT / "submissions"
if str(SUBMISSIONS) not in sys.path:
    sys.path.insert(0, str(SUBMISSIONS))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from submissions._routing_congestion import compute_rudy_map  # noqa: E402


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


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    if x.size != y.size or x.size == 0:
        return float("nan")
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.size, dtype=np.float64)
    sorted_vals = flat[order]
    i = 0
    while i < flat.size:
        j = i + 1
        while j < flat.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks.reshape(x.shape)


def _top_overlap(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    x = a.reshape(-1)
    y = b.reshape(-1)
    k = max(1, int(round(frac * x.size)))
    ax = set(np.argpartition(x, -k)[-k:].tolist())
    by = set(np.argpartition(y, -k)[-k:].tolist())
    return len(ax & by) / float(k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placer", default="submissions/casadi_placer.py")
    parser.add_argument("--benchmarks", nargs="+", default=["ibm01", "ibm10", "ibm17"])
    args = parser.parse_args()

    placer = _load_placer(ROOT / args.placer)
    root = ROOT / "external/MacroPlacement/Testcases/ICCAD04"

    print("benchmark,proxy,wl,density,congestion,pearson,spearman,top10_overlap")
    for name in args.benchmarks:
        benchmark, plc = load_benchmark_from_dir(str(root / name))
        placement = placer.place(benchmark)
        costs = compute_proxy_cost(placement, benchmark, plc)
        rudy = compute_rudy_map(placement, benchmark)
        h = np.asarray(plc.get_horizontal_routing_congestion(), dtype=np.float64).reshape(
            benchmark.grid_rows, benchmark.grid_cols
        )
        v = np.asarray(plc.get_vertical_routing_congestion(), dtype=np.float64).reshape(
            benchmark.grid_rows, benchmark.grid_cols
        )
        true = np.maximum(h, v)
        print(
            f"{name},{costs['proxy_cost']:.6f},{costs['wirelength_cost']:.6f},"
            f"{costs['density_cost']:.6f},{costs['congestion_cost']:.6f},"
            f"{_pearson(rudy, true):.6f},{_pearson(_rankdata(rudy), _rankdata(true)):.6f},"
            f"{_top_overlap(rudy, true, 0.10):.6f}"
        )


if __name__ == "__main__":
    main()
