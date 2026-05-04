"""DREAMPlace bridge submission: utilization-tuned configs + import/legalize stack."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _candidate_select import select_best_true_proxy  # noqa: E402
from _dreamplace_bridge import light_bridge_dreamplace_configs  # noqa: E402
from _dreamplace_candidates import generate_dreamplace_candidates  # noqa: E402
from _dreamplace_runner import dreamplace_available  # noqa: E402
from _plc_lookup import PlcLookup  # noqa: E402
from casadi_placer import CasadiPlacer  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DP_ROOT = _REPO_ROOT / "external" / "DREAMPlace"


def _dreamplace_use_gpu() -> bool:
    if os.environ.get("MACRO_PLACE_DP_CPU", "").lower() in ("1", "true", "yes"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class DreamPlaceBridgePlacer:
    """CasADi baseline seed, utilization-tuned DREAMPlace ensemble, true-proxy pick.

    DREAMPlace-only path (no RePlAce). Two utilization-tuned configs per
    benchmark, GPU when ``torch.cuda.is_available()`` (see README #2 runtime),
    with conservative timeouts so total time stays well under the 1h/bench cap
    after CasADi seeding, imports, and legalization.
    """

    def __init__(
        self,
        *,
        dreamplace_root: Path | None = None,
        dreamplace_iterations: int = 130,
        timeout_seconds: float = 150.0,
        use_gpu: bool | None = None,
    ):
        self.dreamplace_root = Path(dreamplace_root) if dreamplace_root is not None else _DEFAULT_DP_ROOT
        self.dreamplace_iterations = int(dreamplace_iterations)
        self.timeout_seconds = float(timeout_seconds)
        self.use_gpu = _dreamplace_use_gpu() if use_gpu is None else bool(use_gpu)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        baseline = CasadiPlacer().place(benchmark).clone().float()
        plc = PlcLookup().load(benchmark)
        if plc is None:
            return baseline

        ok, _reason = dreamplace_available(self.dreamplace_root)
        if not ok:
            return baseline

        work_root = (
            Path(tempfile.gettempdir()) / "macro_place_dreamplace_bridge" / benchmark.name
        )
        configs = light_bridge_dreamplace_configs(
            benchmark,
            iterations=self.dreamplace_iterations,
            gpu=self.use_gpu,
        )
        try:
            batch = generate_dreamplace_candidates(
                benchmark,
                plc,
                work_root,
                configs,
                bookshelf_name=benchmark.name,
                dreamplace_root=self.dreamplace_root,
                timeout_seconds=self.timeout_seconds,
                initial_placement=baseline,
                soft_macro_mode="aspect_cap",
                soft_macro_row_cap_mult=12.0,
                blend_alphas=(0.55,),
                blend_hard_only=True,
                legalize_outer_passes=2,
                legalize_displacement_budget_frac=0.2,
                legalize_rounds=320,
                legalize_iterative_cycles=1,
            )
        except Exception:
            return baseline

        if not batch.candidates:
            return baseline

        return select_best_true_proxy(
            baseline,
            [c.placement for c in batch.candidates],
            benchmark,
            plc,
            candidate_labels=[c.label for c in batch.candidates],
        ).placement
