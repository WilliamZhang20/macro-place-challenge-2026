#!/usr/bin/env python3
"""Summarize RePlAce diagnostics by config across benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
for path in (REPO_ROOT, SUBMISSIONS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from _benchmark_features import benchmark_features  # noqa: E402

IBM_ROOT = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"


def main() -> int:
    args = _parse_args()
    rows = [_load_one(path) for path in sorted(args.diagnostics_dir.glob("*.json"))]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise SystemExit(f"no benchmark diagnostics found in {args.diagnostics_dir}")

    config_stats = _config_stats(rows)
    summary = {
        "num_benchmarks": len(rows),
        "baseline_average": _avg(row["baseline_proxy"] for row in rows),
        "selected_average": _avg(row["selected_proxy"] for row in rows),
        "oracle_candidate_average": _avg(row["oracle_proxy"] for row in rows),
        "configs": config_stats,
        "feature_correlations": _feature_correlations(rows),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _print_summary(summary)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RePlAce diagnostics JSON files.")
    parser.add_argument("diagnostics_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/macro_place_replace_analysis.json"),
    )
    return parser.parse_args()


def _load_one(path: Path) -> dict | None:
    if path.name == "summary.json":
        return None
    data = json.loads(path.read_text())
    scores = data.get("scores") or []
    if not scores:
        return None
    baseline = scores[0]
    score_by_label = {score["label"]: score for score in scores[1:]}
    candidates = []
    for candidate in data.get("candidates", []):
        score = score_by_label.get(candidate["label"])
        if score is None:
            continue
        candidates.append(
            {
                "label": candidate["label"],
                "config": _config_key(
                    candidate["run_density"],
                    candidate["run_pcofmax"],
                ),
                "valid": bool(score["valid"]),
                "proxy_cost": float(score["proxy_cost"]),
                "wirelength": float(score["wirelength"]),
                "density": float(score["density"]),
                "congestion": float(score["congestion"]),
                "raw_overlap_count": int(candidate["raw_overlap_count"]),
                "final_overlap_count": int(candidate["final_overlap_count"]),
                "legalizer_max_displacement": float(
                    candidate["legalizer_max_displacement"]
                ),
                "legalizer_mean_displacement": float(
                    candidate["legalizer_mean_displacement"]
                ),
            }
        )
    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    best = (
        min(valid_candidates, key=lambda candidate: candidate["proxy_cost"])
        if valid_candidates
        else None
    )
    baseline_proxy = float(baseline["proxy_cost"])
    selected_proxy = min(
        [baseline_proxy]
        + [candidate["proxy_cost"] for candidate in valid_candidates]
    )
    oracle_proxy = best["proxy_cost"] if best is not None else baseline_proxy
    return {
        "benchmark": data["benchmark"],
        "features": data.get("features") or _load_features(data["benchmark"]),
        "baseline_proxy": baseline_proxy,
        "selected_proxy": selected_proxy,
        "oracle_proxy": min(baseline_proxy, oracle_proxy),
        "candidates": candidates,
    }


def _config_stats(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        baseline = row["baseline_proxy"]
        for candidate in row["candidates"]:
            grouped[candidate["config"]].append((baseline, candidate))

    out = []
    for config, entries in sorted(grouped.items()):
        valid_entries = [(base, cand) for base, cand in entries if cand["valid"]]
        deltas = [base - cand["proxy_cost"] for base, cand in valid_entries]
        wins = [delta for delta in deltas if delta > 0.0]
        out.append(
            {
                "config": config,
                "candidates": len(entries),
                "valid": len(valid_entries),
                "wins": len(wins),
                "mean_delta": _avg(deltas) if deltas else None,
                "mean_positive_delta": _avg(wins) if wins else None,
                "best_delta": max(deltas) if deltas else None,
                "worst_delta": min(deltas) if deltas else None,
                "mean_raw_overlaps": _avg(
                    cand["raw_overlap_count"] for _, cand in entries
                ),
                "mean_legalizer_max_displacement": _avg(
                    cand["legalizer_max_displacement"] for _, cand in entries
                ),
            }
        )
    return out


def _feature_correlations(rows: list[dict]) -> list[dict]:
    observations = []
    for row in rows:
        best_delta = row["baseline_proxy"] - row["oracle_proxy"]
        observations.append((row["features"], best_delta))
    keys = sorted({key for features, _ in observations for key in features})
    out = []
    for key in keys:
        xs = [features[key] for features, _ in observations if isinstance(features.get(key), (int, float))]
        ys = [delta for features, delta in observations if isinstance(features.get(key), (int, float))]
        if len(xs) < 3:
            continue
        out.append({"feature": key, "pearson_to_best_delta": _pearson(xs, ys)})
    return sorted(out, key=lambda row: abs(row["pearson_to_best_delta"]), reverse=True)


def _config_key(density: float, pcofmax: float) -> str:
    return f"den={float(density):.6g},pcof={float(pcofmax):.6g}"


def _avg(values) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx = _avg(xs)
    my = _avg(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx <= 0 or dy <= 0:
        return 0.0
    return float(num / ((dx * dy) ** 0.5))


def _load_features(name: str) -> dict:
    bench_dir = IBM_ROOT / name
    if not bench_dir.exists():
        return {}
    benchmark, _ = load_benchmark_from_dir(str(bench_dir))
    return benchmark_features(benchmark)


def _print_summary(summary: dict) -> None:
    print(
        f"benchmarks={summary['num_benchmarks']} "
        f"baseline_avg={summary['baseline_average']:.6f} "
        f"selected_avg={summary['selected_average']:.6f}"
    )
    print()
    print(f"{'config':>18} {'valid':>7} {'wins':>6} {'mean d':>10} {'best':>10} {'worst':>10}")
    print("-" * 70)
    for row in summary["configs"]:
        mean_delta = _fmt(row["mean_delta"])
        best = _fmt(row["best_delta"])
        worst = _fmt(row["worst_delta"])
        print(
            f"{row['config']:>18} {row['valid']:>7}/{row['candidates']:<3} "
            f"{row['wins']:>6} {mean_delta:>10} {best:>10} {worst:>10}"
        )
    if summary["feature_correlations"]:
        print()
        print("top feature correlations to best candidate delta:")
        for row in summary["feature_correlations"][:8]:
            print(f"  {row['feature']:<28} {row['pearson_to_best_delta']:+.3f}")


def _fmt(value) -> str:
    return "none" if value is None else f"{float(value):+.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
