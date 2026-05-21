"""
Write TILOS Benchmark data to Bookshelf format files consumable by DREAMPlace.

Coordinate conventions:
  - Benchmark:  center (x, y) in MICRONS (floats)
  - Bookshelf:  lower-left (x, y) in NANOMETERS (integers, scale=1000)
  - DREAMPlace: works internally in nm; site_width=1nm → scale_factor=1.0

Node ordering in .nodes (determines DREAMPlace node_id):
  [0, nm)          : movable hard macros first, then movable soft macros
  [nm, nm+nf)      : fixed hard macros  (TERMINAL)
  [nm+nf, nm+nf+np): I/O ports          (TERMINAL_NI)
"""

import math
import os
import numpy as np
import torch
from macro_place.benchmark import Benchmark

# Scale factor: μm → nm (integers required by Bookshelf parser)
SCALE = 1000


def _um_to_nm(v: float) -> int:
    return int(round(v * SCALE))


def write_bookshelf(benchmark: Benchmark, outdir: str, fix_soft: bool = False) -> str:
    """Write Bookshelf files and return the .aux path.

    fix_soft=True: soft macros written as 'terminal' (FIXED), so DREAMPlace only
    optimizes hard macros while soft macros stay at their CT positions.
    """
    os.makedirs(outdir, exist_ok=True)
    name = benchmark.name

    movable_hard = [i for i in range(benchmark.num_hard_macros)
                    if not benchmark.macro_fixed[i]]
    fixed_hard   = [i for i in range(benchmark.num_hard_macros)
                    if benchmark.macro_fixed[i]]
    movable_soft = list(range(benchmark.num_hard_macros, benchmark.num_macros))

    # With fix_soft: movable_hard only; soft macros and fixed_hard are all terminals.
    # Without fix_soft: movable_hard | movable_soft | fixed_hard (standard ordering).
    if fix_soft:
        ordered   = movable_hard + movable_soft + fixed_hard
        fixed_ids = set(movable_soft) | set(fixed_hard)  # all non-hard are terminals
    else:
        ordered   = movable_hard + movable_soft + fixed_hard
        fixed_ids = set(fixed_hard)

    port_pos  = benchmark.port_positions
    num_ports = port_pos.shape[0]

    _write_nodes(benchmark, outdir, name, ordered, fixed_ids, num_ports)
    _write_pl(benchmark, outdir, name, ordered, fixed_ids, num_ports, port_pos)
    _write_nets(benchmark, outdir, name, ordered, num_ports)
    _write_scl(benchmark, outdir, name)
    _write_aux(outdir, name)

    return os.path.join(outdir, f"{name}.aux")


def _mn(idx):
    return f"n{idx}"


def _write_nodes(benchmark, outdir, name, ordered, fixed_ids, num_ports):
    # fixed_ids: set of node indices to mark as terminal (FIXED)
    macro_sizes = benchmark.macro_sizes.numpy()
    fixed_set = set(fixed_ids) if not isinstance(fixed_ids, set) else fixed_ids
    num_fixed = len(fixed_set)
    lines = [
        "UCLA nodes 1.0",
        f"NumNodes : {len(ordered) + num_ports}",
        f"NumTerminals : {num_fixed + num_ports}",
        "",
    ]
    for idx in ordered:
        w = _um_to_nm(macro_sizes[idx, 0])
        h = _um_to_nm(macro_sizes[idx, 1])
        suffix = "\tterminal" if idx in fixed_set else ""
        lines.append(f"\t{_mn(idx)}\t{w}\t{h}{suffix}")
    for pi in range(num_ports):
        lines.append(f"\tport{pi}\t1\t1\tterminal_NI")
    with open(os.path.join(outdir, f"{name}.nodes"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_pl(benchmark, outdir, name, ordered, fixed_ids, num_ports, port_pos):
    macro_pos  = benchmark.macro_positions.numpy()
    macro_size = benchmark.macro_sizes.numpy()
    fixed_set  = set(fixed_ids) if not isinstance(fixed_ids, set) else fixed_ids
    lines = ["UCLA pl 1.0", ""]
    for idx in ordered:
        cx, cy = macro_pos[idx]
        w,  h  = macro_size[idx]
        xl = _um_to_nm(cx - w / 2)
        yl = _um_to_nm(cy - h / 2)
        tag = " /FIXED" if idx in fixed_set else ""
        lines.append(f"{_mn(idx)} {xl} {yl} : N{tag}")
    ppos = port_pos.numpy()
    for pi in range(num_ports):
        x = _um_to_nm(ppos[pi, 0])
        y = _um_to_nm(ppos[pi, 1])
        lines.append(f"port{pi} {x} {y} : N /FIXED_NI")
    with open(os.path.join(outdir, f"{name}.pl"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_nets(benchmark, outdir, name, ordered, num_ports):
    macro_size = benchmark.macro_sizes.numpy()

    def _node_name(bench_idx):
        if bench_idx < benchmark.num_macros:
            return _mn(bench_idx)
        return f"port{bench_idx - benchmark.num_macros}"

    net_lines = []
    total_pins = 0
    for net_id in range(benchmark.num_nets):
        owners = benchmark.net_nodes[net_id]
        if len(owners) < 2:
            continue
        header = f"NetDegree : {len(owners)}   net{net_id}"
        pins = []
        if benchmark.net_pin_nodes and len(benchmark.net_pin_nodes) > net_id:
            for owner, slot in benchmark.net_pin_nodes[net_id].tolist():
                nm = _node_name(owner)
                if owner < benchmark.num_hard_macros and benchmark.macro_pin_offsets:
                    offsets = benchmark.macro_pin_offsets[owner]
                    ox, oy = (offsets[slot].tolist() if slot < len(offsets) else [0.0, 0.0])
                else:
                    ox, oy = 0.0, 0.0
                # Pin offsets must also be in nm (integers)
                ox_nm = _um_to_nm(ox)
                oy_nm = _um_to_nm(oy)
                pins.append(f"\t{nm}\tI : {ox_nm}\t{oy_nm}")
        else:
            for owner in owners.tolist():
                pins.append(f"\t{_node_name(owner)}\tI : 0\t0")
        total_pins += len(pins)
        net_lines.append((header, pins))

    header_lines = [
        "UCLA nets 1.0",
        f"NumNets : {len(net_lines)}",
        f"NumPins : {total_pins}",
        "",
    ]
    body = []
    for hdr, pins in net_lines:
        body.append(hdr)
        body.extend(pins)
    with open(os.path.join(outdir, f"{name}.nets"), "w") as f:
        f.write("\n".join(header_lines + body) + "\n")


def _write_scl(benchmark, outdir, name):
    """
    Write placement rows in integer nm coordinates.
    site_width=1nm → scale_factor=1.0 (no internal scaling in DREAMPlace).
    """
    W_nm = _um_to_nm(benchmark.canvas_width)
    H_nm = _um_to_nm(benchmark.canvas_height)
    site_w = 1    # 1 nm → scale_factor = 1.0
    site_h = 1    # 1 nm row height — all integer-nm cell heights are valid multiples
    num_rows  = math.ceil(H_nm / site_h)
    num_sites = W_nm // site_w

    lines = ["UCLA scl 1.0", f"NumRows : {num_rows}", ""]
    for r in range(num_rows):
        y = r * site_h
        lines += [
            "CoreRow Horizontal",
            f"  Coordinate    :   {y}",
            f"  Height        :   {site_h}",
            f"  Sitewidth     :   {site_w}",
            f"  Sitespacing   :   {site_w}",
            f"  Siteorient    :   1",
            f"  Sitesymmetry  :   1",
            f"  SubrowOrigin  :   0\tNumSites  :  {num_sites}",
            "End",
        ]
    with open(os.path.join(outdir, f"{name}.scl"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_aux(outdir, name):
    with open(os.path.join(outdir, f"{name}.aux"), "w") as f:
        f.write(f"RowBasedPlacement : {name}.nodes {name}.nets {name}.pl {name}.scl\n")


def dreamplace_nodes_to_tensor(placedb, ordered, fixed_hard,
                               num_ports, benchmark) -> torch.Tensor:
    """
    Convert DREAMPlace PlaceDB node positions → [num_macros, 2] center-coord
    tensor in MICRONS.

    site_width=1nm → scale_factor=1.0, shift_factor=[0,0]
    so placedb.node_x/y are in nm (integers) — just divide by SCALE.
    """
    macro_size = benchmark.macro_sizes.numpy()   # in μm
    placement  = benchmark.macro_positions.clone()

    node_x = placedb.node_x   # nm
    node_y = placedb.node_y   # nm

    for dp_id, bench_idx in enumerate(ordered):
        xl_nm = float(node_x[dp_id])
        yl_nm = float(node_y[dp_id])
        w_um  = macro_size[bench_idx, 0]
        h_um  = macro_size[bench_idx, 1]
        # Convert nm lower-left → μm center
        placement[bench_idx, 0] = float(xl_nm / SCALE + w_um / 2.0)
        placement[bench_idx, 1] = float(yl_nm / SCALE + h_um / 2.0)

    return placement
