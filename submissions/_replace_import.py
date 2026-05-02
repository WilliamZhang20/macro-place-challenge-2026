"""Import RePlAce Bookshelf ``.pl`` output into challenge tensor coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch

from macro_place.benchmark import Benchmark


def import_bookshelf_placement(
    pl_path: Path | str,
    metadata_path: Path | str,
    benchmark: Benchmark,
    *,
    clamp_to_canvas: bool = True,
    keep_fixed: bool = True,
) -> torch.Tensor:
    """Convert a Bookshelf ``.pl`` file back to ``[num_macros, 2]`` centers.

    The exporter writes challenge coordinates as scaled Bookshelf lower-left
    coordinates.  RePlAce preserves that convention in output.  For movable
    soft macros, the exporter may have used row-height equal-area surrogates;
    the center of that surrogate is imported as the soft cluster center.
    Challenge-side macro sizes are left untouched.
    """

    pl_path = Path(pl_path)
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    scale = float(metadata["scale"])
    if scale <= 0:
        raise ValueError(f"invalid Bookshelf scale {scale!r}")

    node_by_name = _node_map(metadata)
    pl_entries = _read_pl(pl_path)

    placement = benchmark.macro_positions.clone().float()
    imported = torch.zeros((benchmark.num_macros,), dtype=torch.bool)

    for bs_name, (llx, lly) in pl_entries.items():
        node = node_by_name.get(bs_name)
        if node is None:
            # Filler cells and other placer-generated helper nodes are ignored.
            continue
        bench_idx = int(node["bench_idx"])
        if bench_idx < 0 or bench_idx >= benchmark.num_macros:
            continue
        width = float(node["width"])
        height = float(node["height"])
        placement[bench_idx, 0] = float((llx + 0.5 * width) / scale)
        placement[bench_idx, 1] = float((lly + 0.5 * height) / scale)
        imported[bench_idx] = True

    missing = torch.where(~imported)[0].tolist()
    if missing:
        raise ValueError(
            f"Bookshelf placement is missing {len(missing)} macro nodes; "
            f"first missing index: {missing[0]}"
        )

    if keep_fixed and benchmark.macro_fixed.any():
        placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed].to(
            placement.dtype
        )

    if clamp_to_canvas:
        _clamp_to_canvas(placement, benchmark, fixed_mask=benchmark.macro_fixed if keep_fixed else None)

    return placement


def _node_map(metadata: Mapping) -> Dict[str, Mapping]:
    nodes = metadata.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("metadata missing node list")
    out: Dict[str, Mapping] = {}
    for node in nodes:
        if not isinstance(node, dict) or "bookshelf_name" not in node:
            raise ValueError("metadata contains malformed node record")
        out[str(node["bookshelf_name"])] = node
    return out


def _read_pl(path: Path) -> Dict[str, Tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    entries: Dict[str, Tuple[float, float]] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("UCLA"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
        except ValueError:
            continue
        entries[name] = (x, y)
    return entries


def _clamp_to_canvas(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    fixed_mask: torch.Tensor | None,
) -> None:
    movable = torch.ones((benchmark.num_macros,), dtype=torch.bool)
    if fixed_mask is not None:
        movable &= ~fixed_mask.cpu()

    if not movable.any():
        return

    sizes = benchmark.macro_sizes.to(dtype=placement.dtype, device=placement.device)
    hw = 0.5 * sizes[:, 0]
    hh = 0.5 * sizes[:, 1]
    x = torch.clamp(placement[:, 0], hw, float(benchmark.canvas_width) - hw)
    y = torch.clamp(placement[:, 1], hh, float(benchmark.canvas_height) - hh)
    placement[movable, 0] = x[movable]
    placement[movable, 1] = y[movable]
