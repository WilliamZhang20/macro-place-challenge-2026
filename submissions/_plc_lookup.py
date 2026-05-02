"""Helpers for loading the live PlacementCost object for a Benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from macro_place.benchmark import Benchmark


class PlcLookup:
    """Small cache around the repository's benchmark loaders."""

    def __init__(self):
        self._cache: Dict[str, object] = {}

    def load(self, benchmark: Benchmark):
        if benchmark.name in self._cache:
            return self._cache[benchmark.name]

        try:
            from macro_place.loader import load_benchmark, load_benchmark_from_dir
        except Exception:
            return None

        ibm_root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark.name
        if ibm_root.exists():
            try:
                _, plc = load_benchmark_from_dir(str(ibm_root))
                self._cache[benchmark.name] = plc
                return plc
            except Exception:
                return None

        ng45_dir = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / benchmark.name
            / "netlist"
            / "output_CT_Grouping"
        )
        netlist = ng45_dir / "netlist.pb.txt"
        plc_file = ng45_dir / "initial.plc"
        if netlist.exists() and plc_file.exists():
            try:
                _, plc = load_benchmark(str(netlist), str(plc_file), name=benchmark.name)
                self._cache[benchmark.name] = plc
                return plc
            except Exception:
                return None

        return None
