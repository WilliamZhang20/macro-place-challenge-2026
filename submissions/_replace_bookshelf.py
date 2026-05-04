"""Bookshelf export for the RePlAce pipeline.

The RePlAce binary in ``external/MacroPlacement/Flows/util/RePlAceFlow`` expects
an ISPD-style Bookshelf testcase.  This module writes one directly from the
challenge ``Benchmark`` and the live ``PlacementCost`` object, preserving the
important mapping in a JSON sidecar so a later importer can convert RePlAce
``.pl`` output back to benchmark tensor indices without guessing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from macro_place.benchmark import Benchmark


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class BookshelfExport:
    """Paths and mapping metadata produced by :func:`write_bookshelf`."""

    benchmark_name: str
    bookshelf_name: str
    directory: Path
    aux_path: Path
    nodes_path: Path
    nets_path: Path
    wts_path: Path
    pl_path: Path
    scl_path: Path
    shapes_path: Path
    route_path: Path
    metadata_path: Path
    scale: int
    num_nodes: int
    num_terminals: int
    num_nets: int
    num_pins: int


def write_bookshelf(
    benchmark: Benchmark,
    plc,
    output_dir: Path | str,
    *,
    bookshelf_name: Optional[str] = None,
    scale: int = 1000,
    include_route: bool = True,
    include_shapes: bool = True,
    soft_macro_mode: str = "row_height",
    initial_placement=None,
) -> BookshelfExport:
    """Write an ISPD Bookshelf testcase for ``benchmark``.

    Coordinates in the challenge are floating-point microns and macro positions
    are centers.  Bookshelf uses lower-left coordinates, so all physical values
    are scaled to integers and converted during export.  The sidecar
    ``<name>.metadata.json`` stores enough information for a lossless import of
    node positions back to benchmark center coordinates.
    """

    if scale <= 0:
        raise ValueError("scale must be a positive integer")
    if benchmark.num_macros <= 0:
        raise ValueError("cannot export a benchmark with no macros")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bs_name = _safe_name(bookshelf_name or benchmark.name)

    if soft_macro_mode not in {"row_height", "preserve"}:
        raise ValueError("soft_macro_mode must be 'row_height' or 'preserve'")

    rows = _build_rows(benchmark, scale)
    row_height = rows[0]["height"]
    node_records = _build_node_records(
        benchmark,
        plc,
        scale,
        soft_cell_height=row_height,
        soft_macro_mode=soft_macro_mode,
        initial_placement=initial_placement,
    )
    net_records = _build_net_records(plc, node_records.name_to_bs, scale)

    paths = {
        "aux": output / f"{bs_name}.aux",
        "nodes": output / f"{bs_name}.nodes",
        "nets": output / f"{bs_name}.nets",
        "wts": output / f"{bs_name}.wts",
        "pl": output / f"{bs_name}.pl",
        "scl": output / f"{bs_name}.scl",
        "shapes": output / f"{bs_name}.shapes",
        "route": output / f"{bs_name}.route",
        "metadata": output / f"{bs_name}.metadata.json",
    }

    _write_aux(
        paths["aux"],
        bs_name,
        include_route=include_route,
        include_shapes=include_shapes,
    )
    _write_nodes(paths["nodes"], node_records.nodes)
    _write_nets(paths["nets"], net_records)
    _write_wts(paths["wts"], net_records)
    _write_pl(paths["pl"], node_records.nodes)
    _write_scl(paths["scl"], rows)
    if include_shapes:
        _write_shapes(paths["shapes"])
    if include_route:
        _write_route(paths["route"], benchmark, scale)

    metadata = _metadata_dict(
        benchmark=benchmark,
        bookshelf_name=bs_name,
        scale=scale,
        node_records=node_records.nodes,
        paths=paths,
    )
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return BookshelfExport(
        benchmark_name=benchmark.name,
        bookshelf_name=bs_name,
        directory=output,
        aux_path=paths["aux"],
        nodes_path=paths["nodes"],
        nets_path=paths["nets"],
        wts_path=paths["wts"],
        pl_path=paths["pl"],
        scl_path=paths["scl"],
        shapes_path=paths["shapes"],
        route_path=paths["route"],
        metadata_path=paths["metadata"],
        scale=scale,
        num_nodes=len(node_records.nodes),
        num_terminals=sum(1 for n in node_records.nodes if n["terminal"]),
        num_nets=len(net_records),
        num_pins=sum(len(pins) for _, pins in net_records),
    )


@dataclass
class _NodeRecords:
    nodes: List[dict]
    name_to_bs: Dict[str, str]


def _build_node_records(
    benchmark: Benchmark,
    plc,
    scale: int,
    *,
    soft_cell_height: int,
    soft_macro_mode: str,
    initial_placement=None,
) -> _NodeRecords:
    name_to_bs: Dict[str, str] = {}
    nodes: List[dict] = []
    canvas_w = _q(float(benchmark.canvas_width), scale)
    canvas_h = _q(float(benchmark.canvas_height), scale)
    placement = benchmark.macro_positions if initial_placement is None else initial_placement
    if tuple(placement.shape) != tuple(benchmark.macro_positions.shape):
        raise ValueError(
            "initial_placement must have shape "
            f"{tuple(benchmark.macro_positions.shape)}, got {tuple(placement.shape)}"
        )

    hard_indices = list(getattr(benchmark, "hard_macro_indices", []))
    soft_indices = list(getattr(benchmark, "soft_macro_indices", []))
    plc_indices = hard_indices + soft_indices
    if len(plc_indices) != int(benchmark.num_macros):
        raise ValueError(
            "benchmark/plc macro index mismatch: "
            f"{len(plc_indices)} plc indices for {benchmark.num_macros} macros"
        )

    for bench_idx, plc_idx in enumerate(plc_indices):
        node = plc.modules_w_pins[int(plc_idx)]
        original = node.get_name()
        bs_name = f"m{bench_idx}"
        name_to_bs[original] = bs_name
        x, y = placement[bench_idx].tolist()
        w, h = benchmark.macro_sizes[bench_idx].tolist()
        fixed = bool(benchmark.macro_fixed[bench_idx].item())
        kind = "hard" if bench_idx < benchmark.num_hard_macros else "soft"
        original_w = _q(w, scale)
        original_h = _q(h, scale)
        export_w = original_w
        export_h = original_h
        if kind == "soft" and not fixed and soft_macro_mode == "row_height":
            # RePlAce's row-based placer expects movable standard cells to have
            # row height.  CT soft macros are variable-size density proxies, so
            # export them as equal-area row-height cells and keep their true
            # dimensions in metadata for the challenge-side placement tensor.
            area = max(1, original_w * original_h)
            export_h = max(1, int(soft_cell_height))
            export_w = max(1, int(round(area / export_h)))
        cx = _q(x, scale)
        cy = _q(y, scale)
        llx = _clamp_int(cx - export_w // 2, 0, max(0, canvas_w - export_w))
        lly = _clamp_int(cy - export_h // 2, 0, max(0, canvas_h - export_h))
        nodes.append(
            {
                "bookshelf_name": bs_name,
                "original_name": original,
                "bench_idx": bench_idx,
                "plc_idx": int(plc_idx),
                "kind": kind,
                "terminal": fixed,
                "terminal_ni": False,
                "width": export_w,
                "height": export_h,
                "original_width": original_w,
                "original_height": original_h,
                "center_x": cx,
                "center_y": cy,
                "llx": llx,
                "lly": lly,
                "orientation": _orientation(node),
            }
        )

    for port_offset, plc_idx in enumerate(getattr(plc, "port_indices", [])):
        port = plc.modules_w_pins[int(plc_idx)]
        original = port.get_name()
        bs_name = f"p{port_offset}"
        name_to_bs[original] = bs_name
        x, y = port.get_pos()
        nodes.append(
            {
                "bookshelf_name": bs_name,
                "original_name": original,
                "bench_idx": int(benchmark.num_macros) + port_offset,
                "plc_idx": int(plc_idx),
                "kind": "port",
                "terminal": True,
                "terminal_ni": True,
                "width": 1,
                "height": 1,
                "original_width": 1,
                "original_height": 1,
                "center_x": _q(x, scale),
                "center_y": _q(y, scale),
                "llx": _q(x, scale),
                "lly": _q(y, scale),
                "orientation": "N",
            }
        )

    return _NodeRecords(nodes=nodes, name_to_bs=name_to_bs)


def _build_net_records(
    plc,
    name_to_bs: Mapping[str, str],
    scale: int,
) -> List[Tuple[str, List[Tuple[str, str, int, int]]]]:
    pin_to_owner = _pin_owner_map(plc)
    net_records: List[Tuple[str, List[Tuple[str, str, int, int]]]] = []

    for net_idx, (driver, sinks) in enumerate(getattr(plc, "nets", {}).items()):
        pins: List[Tuple[str, str, int, int]] = []
        for pin_name in [driver] + list(sinks):
            owner_name, ox, oy = _resolve_pin(pin_name, name_to_bs, pin_to_owner, scale)
            if owner_name is None:
                continue
            bs_name = name_to_bs.get(owner_name)
            if bs_name is None:
                continue
            direction = "O" if not pins else "I"
            pins.append((bs_name, direction, ox, oy))
        if len(pins) >= 2:
            net_records.append((f"net{net_idx}", pins))

    return net_records


def _pin_owner_map(plc) -> Dict[str, Tuple[str, int, int]]:
    out: Dict[str, Tuple[str, int, int]] = {}
    for pin_idx in getattr(plc, "hard_macro_pin_indices", []):
        pin = plc.modules_w_pins[int(pin_idx)]
        if not hasattr(pin, "get_name") or not hasattr(pin, "get_macro_name"):
            continue
        macro_name = pin.get_macro_name()
        if not macro_name:
            continue
        ox, oy = pin.get_offset() if hasattr(pin, "get_offset") else (pin.x_offset, pin.y_offset)
        out[pin.get_name()] = (macro_name, ox, oy)
    return out


def _resolve_pin(
    pin_name: str,
    name_to_bs: Mapping[str, str],
    pin_to_owner: Mapping[str, Tuple[str, float, float]],
    scale: int,
) -> Tuple[Optional[str], int, int]:
    if pin_name in pin_to_owner:
        owner, ox, oy = pin_to_owner[pin_name]
        return owner, _q(float(ox), scale), _q(float(oy), scale)
    if pin_name in name_to_bs:
        return pin_name, 0, 0
    parent = pin_name.split("/")[0]
    if parent in name_to_bs:
        return parent, 0, 0
    return None, 0, 0


def _build_rows(benchmark: Benchmark, scale: int) -> List[dict]:
    rows = max(1, int(getattr(benchmark, "grid_rows", 1)))
    canvas_w = _q(float(benchmark.canvas_width), scale)
    canvas_h = _q(float(benchmark.canvas_height), scale)
    row_h = max(1, int(math.ceil(canvas_h / rows)))
    out = []
    y = 0
    for _ in range(rows):
        out.append(
            {
                "coordinate": y,
                "height": row_h,
                "site_width": 1,
                "site_spacing": 1,
                "subrow_origin": 0,
                "num_sites": canvas_w,
            }
        )
        y += row_h
    return out


def _write_aux(
    path: Path,
    bs_name: str,
    *,
    include_route: bool,
    include_shapes: bool,
) -> None:
    files = [
        f"{bs_name}.nodes",
        f"{bs_name}.nets",
        f"{bs_name}.wts",
        f"{bs_name}.pl",
        f"{bs_name}.scl",
    ]
    if include_shapes:
        files.append(f"{bs_name}.shapes")
    if include_route:
        files.append(f"{bs_name}.route")
    path.write_text("RowBasedPlacement : " + " ".join(files) + "\n")


def _write_nodes(path: Path, nodes: Sequence[dict]) -> None:
    terminals = sum(1 for n in nodes if n["terminal"])
    lines = [
        "UCLA nodes 1.0",
        "",
        f"NumNodes      : {len(nodes)}",
        f"NumTerminals  : {terminals}",
        "",
    ]
    for node in nodes:
        suffix = ""
        if node["terminal_ni"]:
            suffix = " terminal_NI"
        elif node["terminal"]:
            suffix = " terminal"
        lines.append(
            f"{node['bookshelf_name']}\t{node['width']}\t{node['height']}{suffix}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_nets(path: Path, net_records: Sequence[Tuple[str, Sequence[Tuple[str, str, int, int]]]]) -> None:
    num_pins = sum(len(pins) for _, pins in net_records)
    lines = [
        "UCLA nets 1.0",
        "",
        f"NumNets : {len(net_records)}",
        f"NumPins : {num_pins}",
        "",
    ]
    for net_name, pins in net_records:
        lines.append(f"NetDegree : {len(pins)} {net_name}")
        for bs_name, direction, ox, oy in pins:
            lines.append(f"\t{bs_name}\t{direction}\t: {ox} {oy}")
    path.write_text("\n".join(lines) + "\n")


def _write_wts(path: Path, net_records: Sequence[Tuple[str, Sequence[Tuple[str, str, int, int]]]]) -> None:
    lines = ["UCLA wts 1.0", ""]
    lines.extend(f"{net_name} 1" for net_name, _ in net_records)
    path.write_text("\n".join(lines) + "\n")


def _write_pl(path: Path, nodes: Sequence[dict]) -> None:
    lines = ["UCLA pl 1.0", ""]
    for node in nodes:
        suffix = ""
        if node["terminal_ni"]:
            suffix = " /FIXED_NI"
        elif node["terminal"]:
            suffix = " /FIXED"
        lines.append(
            f"{node['bookshelf_name']}\t{node['llx']}\t{node['lly']}\t: {node['orientation']}{suffix}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_scl(path: Path, rows: Sequence[dict]) -> None:
    lines = ["UCLA scl 1.0", "", f"NumRows : {len(rows)}", ""]
    for row in rows:
        lines.extend(
            [
                "CoreRow Horizontal",
                f"  Coordinate    : {row['coordinate']}",
                f"  Height        : {row['height']}",
                f"  Sitewidth     : {row['site_width']}",
                f"  Sitespacing   : {row['site_spacing']}",
                "  Siteorient    : N",
                "  Sitesymmetry  : Y",
                f"  SubrowOrigin  : {row['subrow_origin']} NumSites : {row['num_sites']}",
                "End",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def _write_shapes(path: Path) -> None:
    path.write_text("UCLA shapes 1.0\n\nNumNonRectangularNodes : 0\n")


def _write_route(path: Path, benchmark: Benchmark, scale: int) -> None:
    rows = max(1, int(getattr(benchmark, "grid_rows", 1)))
    cols = max(1, int(getattr(benchmark, "grid_cols", 1)))
    tile_w = max(1, int(math.ceil(_q(float(benchmark.canvas_width), scale) / cols)))
    tile_h = max(1, int(math.ceil(_q(float(benchmark.canvas_height), scale) / rows)))
    h_cap = max(1, int(round(float(benchmark.hroutes_per_micron) * tile_w / scale)))
    v_cap = max(1, int(round(float(benchmark.vroutes_per_micron) * tile_h / scale)))
    lines = [
        "UCLA route 1.0",
        "",
        "Grid  : {} {}".format(cols, rows),
        "VerticalCapacity   : {}".format(v_cap),
        "HorizontalCapacity : {}".format(h_cap),
        "MinWireWidth       : 1",
        "MinWireSpacing     : 0",
        "ViaSpacing         : 0",
        "GridOrigin         : 0 0",
        f"TileSize           : {tile_w} {tile_h}",
        "BlockagePorosity   : 0",
    ]
    path.write_text("\n".join(lines) + "\n")


def _metadata_dict(
    *,
    benchmark: Benchmark,
    bookshelf_name: str,
    scale: int,
    node_records: Sequence[dict],
    paths: Mapping[str, Path],
) -> dict:
    return {
        "benchmark_name": benchmark.name,
        "bookshelf_name": bookshelf_name,
        "scale": scale,
        "canvas_width": float(benchmark.canvas_width),
        "canvas_height": float(benchmark.canvas_height),
        "num_macros": int(benchmark.num_macros),
        "num_hard_macros": int(benchmark.num_hard_macros),
        "num_soft_macros": int(benchmark.num_soft_macros),
        "paths": {k: str(v) for k, v in paths.items()},
        "nodes": list(node_records),
    }


def _safe_name(name: str) -> str:
    safe = _SAFE_NAME_RE.sub("_", name.strip())
    safe = safe.strip("_")
    return safe or "benchmark"


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(int(value), hi))


def _q(value: float, scale: int) -> int:
    return int(round(float(value) * scale))


def _orientation(node) -> str:
    orient = node.get_orientation() if hasattr(node, "get_orientation") else None
    if not orient or orient in {"-", "R0"}:
        return "N"
    return str(orient)
