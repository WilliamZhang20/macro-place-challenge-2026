from pathlib import Path
import sys

import torch

from macro_place.loader import load_benchmark_from_dir
from macro_place.utils import validate_placement

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
if str(SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSIONS_DIR))

import _candidate_select as selector  # noqa: E402
from casadi_placer import CasadiPlacer  # noqa: E402
from submissions._replace_pipeline import ReplacePipeline  # noqa: E402
from submissions._replace_runner import ReplaceConfig  # noqa: E402


class _StaticPlcLookup:
    def __init__(self, plc):
        self.plc = plc

    def load(self, benchmark):
        return self.plc


class _MissingPlcLookup:
    def load(self, benchmark):
        return None


def test_replace_pipeline_selects_imported_replace_candidate(tmp_path: Path, monkeypatch):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    baseline = CasadiPlacer(soft_proxy_evals=0).place(benchmark)
    hard_idx = 1
    fake = _fake_replace_binary(tmp_path / "fake_replace.py", x_shift=-50.0)

    def fake_cost(placement, benchmark, plc):
        proxy = float(placement[hard_idx, 0].item())
        return {
            "proxy_cost": proxy,
            "wirelength_cost": proxy,
            "density_cost": 0.0,
            "congestion_cost": 0.0,
            "overlap_count": 0,
        }

    monkeypatch.setattr(selector, "compute_proxy_cost", fake_cost)

    pipeline = ReplacePipeline(
        configs=[ReplaceConfig(density=0.75, pcofmax=1.03)],
        baseline_provider=lambda _benchmark: baseline,
        plc_lookup=_StaticPlcLookup(plc),
        work_root=tmp_path / "work",
        binary_path=fake,
        timeout_seconds=10,
        scale=1000,
    )
    result = pipeline.run(benchmark)

    assert result.reason == "ok"
    assert result.selection is not None
    assert result.selection.best.label.endswith(".pl")
    assert result.candidate_batch is not None
    assert len(result.candidate_batch.candidates) == 1
    assert torch.equal(result.placement, result.selection.placement)
    ok, violations = validate_placement(result.placement, benchmark, check_overlaps=False)
    assert ok, violations
    assert result.placement[hard_idx, 0] < baseline[hard_idx, 0]

    diagnostics = result.diagnostics(benchmark.name)
    assert diagnostics["benchmark"] == "ibm01"
    assert diagnostics["reason"] == "ok"
    assert diagnostics["selected_label"].endswith(".pl")
    assert diagnostics["num_candidates"] == 1
    assert diagnostics["runs"][0]["density"] == 0.75
    assert diagnostics["runs"][0]["ok"]
    assert diagnostics["candidates"][0]["final_overlap_count"] == 0
    assert diagnostics["scores"][0]["label"] == "baseline"
    assert diagnostics["scores"][1]["label"].endswith(".pl")


def test_replace_pipeline_returns_baseline_without_plc(tmp_path: Path):
    benchmark, _ = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    baseline = CasadiPlacer(soft_proxy_evals=0).place(benchmark)

    pipeline = ReplacePipeline(
        configs=[ReplaceConfig()],
        baseline_provider=lambda _benchmark: baseline,
        plc_lookup=_MissingPlcLookup(),
        work_root=tmp_path / "work",
        binary_path=tmp_path / "does_not_matter",
        timeout_seconds=10,
        scale=100,
    )
    result = pipeline.run(benchmark)

    assert result.reason == "missing_plc"
    assert result.selection is None
    assert result.candidate_batch is None
    assert torch.equal(result.placement, baseline)
    assert result.diagnostics(benchmark.name)["reason"] == "missing_plc"


def test_replace_pipeline_place_returns_selected_tensor(tmp_path: Path, monkeypatch):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    baseline = CasadiPlacer(soft_proxy_evals=0).place(benchmark)
    hard_idx = 1
    fake = _fake_replace_binary(tmp_path / "fake_replace.py", x_shift=-50.0)

    def fake_cost(placement, benchmark, plc):
        proxy = float(placement[hard_idx, 0].item())
        return {
            "proxy_cost": proxy,
            "wirelength_cost": proxy,
            "density_cost": 0.0,
            "congestion_cost": 0.0,
            "overlap_count": 0,
        }

    monkeypatch.setattr(selector, "compute_proxy_cost", fake_cost)

    pipeline = ReplacePipeline(
        configs=[ReplaceConfig()],
        baseline_provider=lambda _benchmark: baseline,
        plc_lookup=_StaticPlcLookup(plc),
        work_root=tmp_path / "work",
        binary_path=fake,
        timeout_seconds=10,
        scale=1000,
    )

    placement = pipeline.place(benchmark)
    assert placement.shape == benchmark.macro_positions.shape
    assert placement[hard_idx, 0] < baseline[hard_idx, 0]


def test_replace_pipeline_default_configs_include_high_density_edge():
    benchmark, _ = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    configs = ReplacePipeline()._configs_for(benchmark)

    assert any(
        config.density == 0.86 and config.pcofmax == 1.03
        for config in configs
    )


def _fake_replace_binary(path: Path, *, x_shift: float) -> Path:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import sys",
                "args = sys.argv[1:]",
                "name = args[args.index('-bmname') + 1]",
                "root = Path('outputs') / 'ETC' / name",
                "existing = [p for p in root.glob('experiment*') if p.is_dir()] if root.exists() else []",
                "exp = root / f'experiment{len(existing)}'",
                "exp.mkdir(parents=True, exist_ok=True)",
                "pl = exp / f'{name}.eplace-mGP2D.pl'",
                "src = Path('ETC') / name / f'{name}.pl'",
                "lines = []",
                "for raw in src.read_text().splitlines():",
                "    parts = raw.split()",
                "    if parts and parts[0] == 'm1':",
                f"        parts[1] = str(float(parts[1]) + {float(x_shift)!r})",
                "        raw = '\\t'.join(parts)",
                "    lines.append(raw)",
                "pl.write_text('\\n'.join(lines) + '\\n')",
                "print('fake replace pipeline ok')",
            ]
        )
        + "\n"
    )
    path.chmod(0o755)
    return path
