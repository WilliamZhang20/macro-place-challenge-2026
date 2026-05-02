import json
from pathlib import Path
import sys

import torch

from macro_place.loader import load_benchmark_from_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions._replace_bookshelf import write_bookshelf  # noqa: E402
from submissions._replace_import import import_bookshelf_placement  # noqa: E402


def test_bookshelf_import_round_trips_exported_centers(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    export = write_bookshelf(
        benchmark,
        plc,
        tmp_path,
        bookshelf_name="ibm01_export",
        scale=100,
    )

    placement = import_bookshelf_placement(
        export.pl_path,
        export.metadata_path,
        benchmark,
        clamp_to_canvas=False,
    )

    metadata = json.loads(export.metadata_path.read_text())
    for node in metadata["nodes"]:
        if node["kind"] == "port":
            continue
        idx = int(node["bench_idx"])
        expected_x = (float(node["llx"]) + 0.5 * float(node["width"])) / float(
            metadata["scale"]
        )
        expected_y = (float(node["lly"]) + 0.5 * float(node["height"])) / float(
            metadata["scale"]
        )
        assert torch.isclose(placement[idx, 0], torch.tensor(expected_x), atol=1e-5)
        assert torch.isclose(placement[idx, 1], torch.tensor(expected_y), atol=1e-5)


def test_bookshelf_import_uses_surrogate_center_and_pins_fixed(tmp_path: Path):
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    benchmark.macro_fixed[0] = True
    export = write_bookshelf(
        benchmark,
        plc,
        tmp_path,
        bookshelf_name="ibm01_export",
        scale=100,
    )
    metadata = json.loads(export.metadata_path.read_text())
    scale = float(metadata["scale"])

    movable_hard = next(
        n
        for n in metadata["nodes"]
        if n["kind"] == "hard" and not bool(benchmark.macro_fixed[int(n["bench_idx"])])
    )
    movable_soft = next(n for n in metadata["nodes"] if n["kind"] == "soft")
    fixed_hard = next(
        n
        for n in metadata["nodes"]
        if n["kind"] == "hard" and bool(benchmark.macro_fixed[int(n["bench_idx"])])
    )

    replacements = {
        movable_hard["bookshelf_name"]: (
            float(movable_hard["llx"]) + 7.0,
            float(movable_hard["lly"]) + 11.0,
        ),
        movable_soft["bookshelf_name"]: (
            float(movable_soft["llx"]) + 13.0,
            float(movable_soft["lly"]) + 17.0,
        ),
        fixed_hard["bookshelf_name"]: (
            float(fixed_hard["llx"]) + 100.0,
            float(fixed_hard["lly"]) + 100.0,
        ),
    }

    modified = tmp_path / "modified.pl"
    lines = []
    for raw in export.pl_path.read_text().splitlines():
        parts = raw.split()
        if parts and parts[0] in replacements:
            x, y = replacements[parts[0]]
            suffix = " ".join(parts[3:]) if len(parts) > 3 else ": N"
            lines.append(f"{parts[0]}\t{x:.1f}\t{y:.1f}\t{suffix}")
        else:
            lines.append(raw)
    modified.write_text("\n".join(lines) + "\n")

    placement = import_bookshelf_placement(modified, export.metadata_path, benchmark)

    hard_idx = int(movable_hard["bench_idx"])
    expected_hard = torch.tensor(
        [
            (replacements[movable_hard["bookshelf_name"]][0] + 0.5 * movable_hard["width"])
            / scale,
            (replacements[movable_hard["bookshelf_name"]][1] + 0.5 * movable_hard["height"])
            / scale,
        ],
        dtype=placement.dtype,
    )
    assert torch.allclose(placement[hard_idx], expected_hard, atol=1e-5)

    soft_idx = int(movable_soft["bench_idx"])
    expected_soft = torch.tensor(
        [
            (replacements[movable_soft["bookshelf_name"]][0] + 0.5 * movable_soft["width"])
            / scale,
            (replacements[movable_soft["bookshelf_name"]][1] + 0.5 * movable_soft["height"])
            / scale,
        ],
        dtype=placement.dtype,
    )
    assert torch.allclose(placement[soft_idx], expected_soft, atol=1e-5)

    fixed_idx = int(fixed_hard["bench_idx"])
    assert torch.allclose(placement[fixed_idx], benchmark.macro_positions[fixed_idx])
