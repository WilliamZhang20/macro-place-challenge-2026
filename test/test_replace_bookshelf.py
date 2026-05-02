from pathlib import Path
import sys

from macro_place.loader import load_benchmark_from_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submissions._replace_bookshelf import write_bookshelf  # noqa: E402


def test_bookshelf_export_ibm01_has_complete_mapping(tmp_path: Path):
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

    expected_paths = [
        export.aux_path,
        export.nodes_path,
        export.nets_path,
        export.wts_path,
        export.pl_path,
        export.scl_path,
        export.shapes_path,
        export.route_path,
        export.metadata_path,
    ]
    for path in expected_paths:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    assert export.num_nodes == benchmark.num_macros + len(plc.port_indices)
    assert export.num_terminals >= len(plc.port_indices)
    assert export.num_nets > 0
    assert export.num_pins > export.num_nets

    nodes_text = export.nodes_path.read_text()
    nets_text = export.nets_path.read_text()
    pl_text = export.pl_path.read_text()
    aux_text = export.aux_path.read_text()

    assert "NumNodes" in nodes_text
    assert "terminal_NI" in nodes_text
    assert "NetDegree" in nets_text
    assert " : " in nets_text
    assert "/FIXED_NI" in pl_text
    assert "ibm01_export.route" in aux_text
