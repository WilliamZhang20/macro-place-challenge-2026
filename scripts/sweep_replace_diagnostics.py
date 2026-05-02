#!/usr/bin/env python3
"""Run RePlAce diagnostics over a small IBM benchmark/config sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_replace_diagnostics.py"
DEFAULT_BENCHMARKS = ("ibm01", "ibm03", "ibm09")
DEFAULT_CONFIGS = ("0.72:1.03", "0.78:1.03", "0.80:1.03", "0.84:1.03", "0.88:1.03")
IBM_BENCHMARKS = (
    "ibm01",
    "ibm02",
    "ibm03",
    "ibm04",
    "ibm06",
    "ibm07",
    "ibm08",
    "ibm09",
    "ibm10",
    "ibm11",
    "ibm12",
    "ibm13",
    "ibm14",
    "ibm15",
    "ibm16",
    "ibm17",
    "ibm18",
)


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for benchmark in args.benchmark:
        out_path = args.output_dir / f"{benchmark}.json"
        cmd = [
            sys.executable,
            str(RUNNER),
            "--benchmark",
            benchmark,
            "--timeout",
            str(args.timeout),
            "--work-root",
            str(args.work_root),
            "--output",
            str(out_path),
        ]
        for config in args.config:
            cmd.extend(["--config", config])
        print("running", benchmark, flush=True)
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        rows.append(_summary_row(out_path))

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    _print_table(rows)
    print(f"summary: {summary_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep RePlAce diagnostics.")
    parser.add_argument("--benchmark", action="append", default=None)
    parser.add_argument("--all", action="store_true", help="Run all public IBM benchmarks.")
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument(
        "--pipeline-default-configs",
        action="store_true",
        help="Use ReplacePipeline's general feature-derived default config policy.",
    )
    parser.add_argument("--timeout", type=float, default=80.0)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/tmp/macro_place_replace_sweep_work"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/macro_place_replace_sweep"),
    )
    args = parser.parse_args()
    if args.all:
        args.benchmark = list(IBM_BENCHMARKS)
    else:
        args.benchmark = args.benchmark or list(DEFAULT_BENCHMARKS)
    if args.pipeline_default_configs:
        if args.config:
            raise SystemExit("--pipeline-default-configs cannot be combined with --config")
        args.config = []
    else:
        args.config = args.config or list(DEFAULT_CONFIGS)
    return args


def _summary_row(path: Path) -> dict:
    data = json.loads(path.read_text())
    scores = data.get("scores", [])
    baseline = scores[0] if scores else {}
    candidates = scores[1:]
    valid_candidates = [score for score in candidates if score.get("valid")]
    best_candidate = (
        min(valid_candidates, key=lambda score: score["proxy_cost"])
        if valid_candidates
        else None
    )
    baseline_proxy = float(baseline.get("proxy_cost", float("inf")))
    candidate_proxy = (
        float(best_candidate["proxy_cost"]) if best_candidate is not None else None
    )
    improvement = (
        baseline_proxy - candidate_proxy if candidate_proxy is not None else None
    )
    return {
        "benchmark": data["benchmark"],
        "selected_label": data.get("selected_label"),
        "num_candidates": int(data.get("num_candidates", 0)),
        "baseline_proxy": baseline_proxy,
        "best_candidate_label": best_candidate["label"] if best_candidate else None,
        "best_candidate_proxy": candidate_proxy,
        "best_candidate_delta": improvement,
        "final_proxy": float(data["final"]["proxy_cost"]),
        "final_valid": bool(data["final"]["valid"]),
        "final_overlaps": int(data["final"]["overlaps"]),
        "json_path": str(path),
    }


def _print_table(rows: list[dict]) -> None:
    print()
    print(f"{'bench':>8} {'base':>10} {'best cand':>10} {'delta':>10} {'selected':>18}")
    print("-" * 64)
    for row in rows:
        cand = row["best_candidate_proxy"]
        delta = row["best_candidate_delta"]
        cand_s = f"{cand:.6f}" if cand is not None else "none"
        delta_s = f"{delta:+.6f}" if delta is not None else "none"
        print(
            f"{row['benchmark']:>8} {row['baseline_proxy']:>10.6f} "
            f"{cand_s:>10} {delta_s:>10} {row['selected_label'][:18]:>18}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
