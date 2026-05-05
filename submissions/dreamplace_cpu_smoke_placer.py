"""DREAMPlace smoke integration (CPU or GPU; not tuned for proxy score).

GPU is used automatically when the DREAMPlace install is CUDA-enabled and
``torch.cuda.is_available()``. Override with constructor ``use_gpu`` or env
``MACRO_PLACE_DP_GPU=0|1|auto``.

Run from repo root with the same Python that has PyTorch, after building
``external/DREAMPlace/install`` (``scripts/setup_dreamplace.sh``).

Example:
  source ~/myenv/bin/activate
  evaluate submissions/dreamplace_cpu_smoke_placer.py -b ibm01
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from macro_place.benchmark import Benchmark

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _dreamplace_cpu_smoke import (  # noqa: E402
    deep_merge_dreamplace_json,
    default_dreamplace_install,
    dreamplace_install_ok,
    resolve_dreamplace_gpu,
    run_dreamplace_placement,
)
from _plc_lookup import PlcLookup  # noqa: E402


class DreamplaceCpuSmokePlacer:
    """Run one short DREAMPlace global+legalize pass; fallback to initial placement.

    Tuning for search / Bayesian optimization:

    - Pass ``dreamplace_json_overrides`` with any DREAMPlace JSON keys; they are
      deep-merged over the built-in defaults (see ``_dreamplace_cpu_smoke._dp_json``).
      ``aux_input`` and ``result_dir`` are always overwritten per run.
    - Shorthand kwargs ``global_iterations``, ``num_bins``, ``num_threads``,
      ``target_density`` set the baseline template before merges.
    - For a full replacement of ``global_place_stages``, put a complete list in
      overrides (merging does not splice list elements).

    Device:

    - ``use_gpu=None`` (default): auto-select GPU when the DREAMPlace install is
      CUDA-built and ``torch.cuda.is_available()``.
    - ``use_gpu=False`` / ``True`` forces CPU or requests GPU (clamped if unsupported).
    - Env ``MACRO_PLACE_DP_GPU`` overrides ``use_gpu`` when set to ``0``, ``1``, or ``auto``.
    """

    def __init__(
        self,
        *,
        dreamplace_install: Path | None = None,
        global_iterations: int = 20,
        num_bins: int = 128,
        num_threads: int = 4,
        target_density: float = 0.72,
        timeout_seconds: float = 900.0,
        dreamplace_json_overrides: Optional[Mapping[str, Any]] = None,
        use_gpu: Optional[bool] = None,
    ):
        self.dreamplace_install = dreamplace_install
        self.global_iterations = int(global_iterations)
        self.num_bins = int(num_bins)
        self.num_threads = int(num_threads)
        self.target_density = float(target_density)
        self.timeout_seconds = float(timeout_seconds)
        self.dreamplace_json_overrides = (
            dict(dreamplace_json_overrides) if dreamplace_json_overrides else None
        )
        self.use_gpu = use_gpu
        self._plc = PlcLookup()

    def set_dreamplace_json_overrides(
        self, overrides: Optional[Mapping[str, Any]]
    ) -> None:
        """Replace merged JSON overrides (e.g. update between BO trials)."""

        self.dreamplace_json_overrides = (
            dict(overrides) if overrides is not None else None
        )

    def preview_effective_json(
        self,
        *,
        aux_placeholder: str = "<aux>",
        result_dir_placeholder: str = "<result_dir>",
    ) -> dict[str, Any]:
        """Static shape of the JSON for debugging / BO bounds design (paths are placeholders)."""

        from _dreamplace_cpu_smoke import _dp_json  # noqa: PLC0415

        base = _dp_json(
            aux_abs=Path(aux_placeholder),
            result_dir_abs=Path(result_dir_placeholder),
            global_iterations=self.global_iterations,
            num_bins=self.num_bins,
            num_threads=self.num_threads,
            target_density=self.target_density,
        )
        if self.dreamplace_json_overrides:
            base = deep_merge_dreamplace_json(base, self.dreamplace_json_overrides)
        base["aux_input"] = aux_placeholder
        base["result_dir"] = result_dir_placeholder
        inst = self.dreamplace_install or default_dreamplace_install()
        base["gpu"] = resolve_dreamplace_gpu(
            inst,
            use_gpu=self.use_gpu,
            dreamplace_json_overrides=self.dreamplace_json_overrides,
        )
        return base

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        base = benchmark.macro_positions.clone().float()
        inst = self.dreamplace_install or default_dreamplace_install()
        ok, _ = dreamplace_install_ok(inst)
        if not ok:
            return base

        plc = self._plc.load(benchmark)
        if plc is None:
            return base

        out = run_dreamplace_placement(
            benchmark,
            plc,
            dreamplace_install=inst,
            global_iterations=self.global_iterations,
            num_bins=self.num_bins,
            num_threads=self.num_threads,
            target_density=self.target_density,
            timeout_seconds=self.timeout_seconds,
            dreamplace_json_overrides=self.dreamplace_json_overrides,
            use_gpu=self.use_gpu,
        )
        if out is None:
            return base
        return out
