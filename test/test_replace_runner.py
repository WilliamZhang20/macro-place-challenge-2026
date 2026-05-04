from pathlib import Path
import sys

from macro_place.loader import load_benchmark_from_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions._replace_bookshelf import write_bookshelf  # noqa: E402
from submissions._replace_runner import (  # noqa: E402
    ReplaceConfig,
    discover_replace_pls,
    run_replace,
)


def test_replace_runner_invokes_binary_and_discovers_new_pl(tmp_path: Path):
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
    fake = _fake_replace_binary(tmp_path / "fake_replace.py")

    result = run_replace(
        export,
        ReplaceConfig(density=0.73, pcofmax=1.07, extra_args=("-abc", "42")),
        binary_path=fake,
        timeout_seconds=10,
    )

    assert result.ok
    assert result.returncode == 0
    assert not result.timed_out
    assert len(result.pl_paths) == 1
    assert result.pl_paths[0].name == "ibm01_export.eplace-mGP2D.pl"
    assert result.pl_paths[0].read_text().startswith("UCLA pl")
    log = result.log_path.read_text()
    assert "-den 0.73" in log
    assert "-pcofmax 1.07" in log
    assert "-abc 42" in log

    discovered = discover_replace_pls(export)
    assert result.pl_paths[0] in discovered


def test_replace_runner_reports_timeout(tmp_path: Path):
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
    fake = _fake_replace_binary(tmp_path / "fake_replace.py", sleep_seconds=5)

    result = run_replace(
        export,
        ReplaceConfig(),
        binary_path=fake,
        timeout_seconds=0.2,
        stop_after_first_pl=False,
    )

    assert not result.ok
    assert not result.usable
    assert result.returncode == 124
    assert result.timed_out
    assert "[TIMEOUT" in result.log_path.read_text()


def _fake_replace_binary(path: Path, sleep_seconds: float = 0.0) -> Path:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import sys, time",
                f"time.sleep({sleep_seconds!r})",
                "args = sys.argv[1:]",
                "name = args[args.index('-bmname') + 1]",
                "root = Path('outputs') / 'ETC' / name",
                "existing = [p for p in root.glob('experiment*') if p.is_dir()] if root.exists() else []",
                "exp = root / f'experiment{len(existing)}'",
                "exp.mkdir(parents=True, exist_ok=True)",
                "pl = exp / f'{name}.eplace-mGP2D.pl'",
                "src = Path('ETC') / name / f'{name}.pl'",
                "pl.write_text(src.read_text())",
                "print('fake replace ok', ' '.join(args))",
            ]
        )
        + "\n"
    )
    path.chmod(0o755)
    return path
