from pathlib import Path
import sys

import torch

from macro_place.loader import load_benchmark_from_dir
from macro_place.utils import validate_placement

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions._replace_candidates import generate_replace_candidates  # noqa: E402
from submissions._replace_runner import ReplaceConfig  # noqa: E402


def test_generate_replace_candidates_runs_and_imports_all_configs(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    fake = _fake_replace_binary(tmp_path / "fake_replace.py", x_shift=9.0)

    batch = generate_replace_candidates(
        benchmark,
        plc,
        tmp_path / "work",
        [
            ReplaceConfig(density=0.70, pcofmax=1.02),
            ReplaceConfig(density=0.82, pcofmax=1.05),
        ],
        bookshelf_name="ibm01_export",
        scale=100,
        binary_path=fake,
        timeout_seconds=10,
        legalize_imported=False,
    )

    assert batch.export.bookshelf_name == "ibm01_export"
    assert len(batch.run_results) == 2
    assert all(r.ok for r in batch.run_results)
    assert len(batch.candidates) == 2
    assert len({c.pl_path for c in batch.candidates}) == 2

    for candidate in batch.candidates:
        assert candidate.placement.shape == benchmark.macro_positions.shape
        ok, violations = validate_placement(candidate.placement, benchmark, check_overlaps=False)
        assert ok, violations
        assert candidate.raw_overlap_count >= 0
        assert candidate.final_overlap_count == candidate.raw_overlap_count
        assert candidate.legalizer_max_displacement == 0.0
        assert candidate.legalizer_mean_displacement == 0.0

    # The fake binary shifts m1's Bookshelf lower-left x by 9 scaled units.
    # Imported center x should therefore move by 0.09 microns at scale=100.
    assert torch.isclose(
        batch.candidates[0].placement[1, 0],
        benchmark.macro_positions[1, 0] + 0.09,
        atol=1e-5,
    )


def test_generate_replace_candidates_keeps_failed_runs_out_of_candidates(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    fake = _fake_replace_binary(tmp_path / "fake_replace.py", exit_code=7)

    batch = generate_replace_candidates(
        benchmark,
        plc,
        tmp_path / "work",
        [ReplaceConfig()],
        bookshelf_name="ibm01_export",
        scale=100,
        binary_path=fake,
        timeout_seconds=10,
    )

    assert len(batch.run_results) == 1
    assert batch.run_results[0].returncode == 7
    assert not batch.run_results[0].ok
    assert not batch.run_results[0].usable
    assert batch.candidates == []


def test_generate_replace_candidates_uses_early_stopped_placements(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    fake = _fake_replace_binary(
        tmp_path / "fake_replace.py",
        sleep_after_write_seconds=5,
    )

    batch = generate_replace_candidates(
        benchmark,
        plc,
        tmp_path / "work",
        [ReplaceConfig()],
        bookshelf_name="ibm01_export",
        scale=100,
        binary_path=fake,
        timeout_seconds=0.2,
        legalize_imported=False,
    )

    assert len(batch.run_results) == 1
    assert not batch.run_results[0].timed_out
    assert batch.run_results[0].returncode < 0
    assert not batch.run_results[0].ok
    assert batch.run_results[0].usable
    assert len(batch.candidates) == 1
    assert batch.candidates[0].run_result is batch.run_results[0]


def test_generate_replace_candidates_can_ignore_partial_timeout_placements(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    fake = _fake_replace_binary(
        tmp_path / "fake_replace.py",
        sleep_after_write_seconds=5,
    )

    batch = generate_replace_candidates(
        benchmark,
        plc,
        tmp_path / "work",
        [ReplaceConfig()],
        bookshelf_name="ibm01_export",
        scale=100,
        binary_path=fake,
        timeout_seconds=0.2,
        use_partial_results=False,
    )

    assert len(batch.run_results) == 1
    assert not batch.run_results[0].timed_out
    assert batch.run_results[0].returncode < 0
    assert batch.run_results[0].usable
    assert batch.candidates == []


def test_generate_replace_candidates_uses_fresh_signal_crash_placements(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    fake = _fake_replace_binary(tmp_path / "fake_replace.py", exit_code=-11)

    batch = generate_replace_candidates(
        benchmark,
        plc,
        tmp_path / "work",
        [ReplaceConfig()],
        bookshelf_name="ibm01_export",
        scale=100,
        binary_path=fake,
        timeout_seconds=10,
        legalize_imported=False,
    )

    assert len(batch.run_results) == 1
    assert batch.run_results[0].returncode == -11
    assert not batch.run_results[0].ok
    assert batch.run_results[0].usable
    assert len(batch.candidates) == 1


def test_generate_replace_candidates_rejects_stale_signal_crash_placements(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    fake = _fake_replace_binary(tmp_path / "fake_replace.py", exit_code=-11)
    work_root = tmp_path / "work"

    first = generate_replace_candidates(
        benchmark,
        plc,
        work_root,
        [ReplaceConfig()],
        bookshelf_name="ibm01_export",
        scale=100,
        binary_path=fake,
        timeout_seconds=10,
        legalize_imported=False,
    )
    assert len(first.candidates) == 1

    fake_no_write = _fake_replace_binary(
        tmp_path / "fake_replace_no_write.py",
        exit_code=-11,
        write_output=False,
    )
    second = generate_replace_candidates(
        benchmark,
        plc,
        work_root,
        [ReplaceConfig()],
        bookshelf_name="ibm01_export",
        scale=100,
        binary_path=fake_no_write,
        timeout_seconds=10,
        legalize_imported=False,
    )

    assert len(second.run_results) == 1
    assert second.run_results[0].returncode == -11
    assert not second.run_results[0].usable
    assert second.candidates == []


def test_generate_replace_candidates_records_legalization_accounting(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    fake = _fake_replace_binary(tmp_path / "fake_replace.py", x_shift=0.0)

    batch = generate_replace_candidates(
        benchmark,
        plc,
        tmp_path / "work",
        [ReplaceConfig()],
        bookshelf_name="ibm01_export",
        scale=100,
        binary_path=fake,
        timeout_seconds=10,
    )

    assert len(batch.candidates) == 1
    candidate = batch.candidates[0]
    assert candidate.raw_overlap_count > 0
    assert candidate.final_overlap_count == 0
    assert candidate.legalizer_max_displacement > 0.0
    assert candidate.legalizer_mean_displacement >= 0.0


def _fake_replace_binary(
    path: Path,
    *,
    x_shift: float = 0.0,
    exit_code: int = 0,
    sleep_after_write_seconds: float = 0.0,
    write_output: bool = True,
) -> Path:
    write_lines = [
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
    ]
    script_lines = [
        "#!/usr/bin/env python3",
        "from pathlib import Path",
        "import sys",
        "args = sys.argv[1:]",
        "name = args[args.index('-bmname') + 1]",
        *(write_lines if write_output else ["pass"]),
        f"import time; time.sleep({float(sleep_after_write_seconds)!r})",
        "print('fake replace candidate ok')",
        f"exit_code = {int(exit_code)}",
        "if exit_code < 0:",
        "    import os, signal",
        "    os.kill(os.getpid(), -exit_code)",
        "sys.exit(exit_code)",
    ]
    path.write_text("\n".join(script_lines) + "\n")
    path.chmod(0o755)
    return path
