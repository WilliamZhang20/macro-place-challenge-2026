from pathlib import Path
import sys

import pytest
import torch

from macro_place.benchmark import Benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions import _candidate_select as selector  # noqa: E402


def test_select_best_true_proxy_picks_lowest_valid_candidate(monkeypatch):
    benchmark = _toy_benchmark()
    baseline = benchmark.macro_positions.clone()
    good = baseline.clone()
    good[0, 0] = 1.5
    invalid = baseline.clone()
    invalid[1] = invalid[0]

    monkeypatch.setattr(selector, "compute_proxy_cost", _fake_cost)

    result = selector.select_best_true_proxy(
        baseline,
        [invalid, good],
        benchmark,
        plc=object(),
        candidate_labels=["invalid_overlap", "good"],
    )

    assert result.best.label == "good"
    assert torch.equal(result.placement, good)
    assert [s.label for s in result.scores] == ["baseline", "invalid_overlap", "good"]
    assert not result.scores[1].valid
    assert result.scores[1].proxy_cost == float("inf")
    assert result.scores[2].proxy_cost < result.scores[0].proxy_cost


def test_select_best_true_proxy_keeps_valid_baseline_floor(monkeypatch):
    benchmark = _toy_benchmark()
    baseline = benchmark.macro_positions.clone()
    worse = baseline.clone()
    worse[0, 0] = 8.0

    monkeypatch.setattr(selector, "compute_proxy_cost", _fake_cost)

    result = selector.select_best_true_proxy(
        baseline,
        [worse],
        benchmark,
        plc=object(),
    )

    assert result.best.label == "baseline"
    assert torch.equal(result.placement, baseline)
    assert result.scores[1].valid
    assert result.scores[1].proxy_cost > result.scores[0].proxy_cost


def test_select_best_true_proxy_rejects_label_length_mismatch():
    benchmark = _toy_benchmark()
    with pytest.raises(ValueError, match="candidate_labels"):
        selector.select_best_true_proxy(
            benchmark.macro_positions,
            [benchmark.macro_positions],
            benchmark,
            plc=object(),
            candidate_labels=[],
        )


def test_select_best_true_proxy_errors_when_everything_invalid(monkeypatch):
    benchmark = _toy_benchmark()
    bad = benchmark.macro_positions.clone()
    bad[1] = bad[0]

    monkeypatch.setattr(selector, "compute_proxy_cost", _fake_cost)

    with pytest.raises(ValueError, match="no valid"):
        selector.select_best_true_proxy(
            bad,
            [bad],
            benchmark,
            plc=object(),
        )


def _toy_benchmark() -> Benchmark:
    return Benchmark(
        name="toy",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=2,
        num_hard_macros=2,
        num_soft_macros=0,
        macro_positions=torch.tensor([[3.0, 3.0], [7.0, 7.0]], dtype=torch.float32),
        macro_sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        macro_fixed=torch.tensor([False, False]),
        macro_names=["a", "b"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0, dtype=torch.float32),
        grid_rows=2,
        grid_cols=2,
    )


def _fake_cost(placement, benchmark, plc):
    proxy = float(placement[:, 0].sum().item())
    return {
        "proxy_cost": proxy,
        "wirelength_cost": proxy,
        "density_cost": 0.0,
        "congestion_cost": 0.0,
        "overlap_count": 0,
    }
