from pathlib import Path
import json
import sys

from macro_place.loader import load_benchmark_from_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions._dreamplace_runner import (  # noqa: E402
    DreamPlaceConfig,
    dreamplace_available,
    run_dreamplace,
)
from submissions._dreamplace_candidates import generate_dreamplace_candidates  # noqa: E402
from submissions._replace_bookshelf import write_bookshelf  # noqa: E402


def test_dreamplace_runner_writes_params_and_discovers_output(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    export = write_bookshelf(
        benchmark,
        plc,
        tmp_path / "ETC" / "ibm01_export",
        bookshelf_name="ibm01_export",
        scale=100,
    )
    fake = _fake_dreamplace_placer(tmp_path / "fake_dreamplace.py")

    result = run_dreamplace(
        export,
        DreamPlaceConfig(target_density=0.83, iterations=17, gpu=False),
        dreamplace_root=tmp_path,
        placer_path=fake,
        timeout_seconds=10,
        work_root=tmp_path / "dreamplace_work",
    )

    assert result.ok
    assert result.usable
    assert result.returncode == 0
    assert result.params_path.exists()
    assert result.log_path.exists()
    assert len(result.pl_paths) == 1
    assert result.pl_paths[0].name.endswith("_ibm01_export.gp.pl")

    params = json.loads(result.params_path.read_text())
    assert params["aux_input"] == str(export.aux_path.resolve())
    assert params["target_density"] == 0.83
    assert params["global_place_stages"][0]["iteration"] == 17
    assert params["gpu"] == 0
    assert params["random_center_init_flag"] == 0


def test_dreamplace_available_reports_missing_configure(tmp_path: Path):
    root = tmp_path / "DREAMPlace"
    placer = root / "dreamplace" / "Placer.py"
    placer.parent.mkdir(parents=True)
    placer.write_text("print('placeholder')\n")

    ok, reason = dreamplace_available(root)

    assert not ok
    assert "configure.py" in reason


def test_generate_dreamplace_candidates_imports_and_legalizes(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    fake = _fake_dreamplace_placer(tmp_path / "fake_dreamplace.py")

    batch = generate_dreamplace_candidates(
        benchmark,
        plc,
        tmp_path / "work",
        [DreamPlaceConfig(target_density=0.81, iterations=13, gpu=False)],
        bookshelf_name="ibm01_export",
        scale=100,
        dreamplace_root=tmp_path,
        placer_path=fake,
        timeout_seconds=10,
    )

    assert len(batch.run_results) == 1
    assert batch.run_results[0].ok
    assert len(batch.candidates) == 1
    candidate = batch.candidates[0]
    assert candidate.placement.shape == benchmark.macro_positions.shape
    assert candidate.final_overlap_count == 0
    assert "dream_den0p81_iter13" in candidate.label


def _fake_dreamplace_placer(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "from pathlib import Path",
                "params = json.loads(Path(sys.argv[1]).read_text())",
                "aux = Path(params['aux_input'])",
                "name = aux.stem",
                "result_dir = Path(params['result_dir']) / name",
                "result_dir.mkdir(parents=True, exist_ok=True)",
                "src = aux.with_suffix('.pl')",
                "dst = result_dir / f'{name}.gp.pl'",
                "dst.write_text(src.read_text())",
                "print('fake dreamplace ok')",
            ]
        )
        + "\n"
    )
    path.chmod(0o755)
    return path
