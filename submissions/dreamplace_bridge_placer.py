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

from _benchmark_features import benchmark_features  # noqa: E402
from _candidate_select import select_best_true_proxy  # noqa: E402
from _dreamplace_bridge import light_bridge_dreamplace_configs  # noqa: E402
from _dreamplace_candidates import generate_dreamplace_candidates  # noqa: E402
from _dreamplace_runner import dreamplace_available  # noqa: E402
from _hard_legalizer import legalize_hard  # noqa: E402
from _plc_lookup import PlcLookup  # noqa: E402
from casadi_placer import CasadiPlacer  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DP_ROOT = _REPO_ROOT / "external" / "DREAMPlace"


def _dreamplace_use_gpu() -> bool:
    if os.environ.get("MACRO_PLACE_DP_CPU", "").lower() in ("1", "true", "yes"):
        return False
    try:
        import torch as _torch

        return bool(_torch.cuda.is_available())
    except Exception:
        return False


def _perturbed_legalized_seed(
    baseline: torch.Tensor,
    benchmark: Benchmark,
    *,
    sigma_um: float,
) -> torch.Tensor:
    """Break symmetry vs CasADi: jitter movable hard macros, then legalize.

    CasADi and legalized-.plc seeds sit in the same ~1.04 proxy basin on ibm01,
    so both DREAMPlace runs see the same basin unless we explicitly diversify.
    """

    out = baseline.clone()
    nh = int(benchmark.num_hard_macros)
    if nh <= 0:
        return out
    movable = ~benchmark.macro_fixed[:nh]
    if not bool(movable.any()):
        return out
    noise = torch.randn((nh, 2), dtype=out.dtype, device=out.device) * float(sigma_um)
    noise[~movable] = 0.0
    out[:nh] = out[:nh] + noise
    return legalize_hard(
        out,
        benchmark,
        legalize_rounds=480,
        outer_passes=1,
    )


class DreamPlaceBridgePlacer:
    """CasADi baseline + two DREAMPlace runs (two Bookshelf seeds), true-proxy pick.

    If every DREAMPlace candidate loses to the CasADi baseline (common on easy
    cases like ibm01 where both score ~1.038), selection correctly returns
    CasADi — the bridge must beat that proxy to ``move the needle``.  A
    perturbed second seed forces a different global trajectory while staying
    overlap-free before export.
    """

    def __init__(
        self,
        *,
        dreamplace_root: Path | None = None,
        dreamplace_iterations: int = 200,
        timeout_seconds: float = 220.0,
        use_gpu: bool | None = None,
        perturb_sigma_um: float | None = None,
    ):
        self.dreamplace_root = Path(dreamplace_root) if dreamplace_root is not None else _DEFAULT_DP_ROOT
        self.dreamplace_iterations = int(dreamplace_iterations)
        self.timeout_seconds = float(timeout_seconds)
        self.use_gpu = _dreamplace_use_gpu() if use_gpu is None else bool(use_gpu)
        env_sig = os.environ.get("MACRO_PLACE_DP_SEED_SIGMA")
        self.perturb_sigma_um = float(
            perturb_sigma_um
            if perturb_sigma_um is not None
            else (env_sig if env_sig is not None else 0.62)
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        baseline = CasadiPlacer().place(benchmark).clone().float()
        plc = PlcLookup().load(benchmark)
        if plc is None:
            return baseline

        ok, _reason = dreamplace_available(self.dreamplace_root)
        if not ok:
            return baseline

        feats = benchmark_features(benchmark)
        util = float(feats["macro_area_utilization"])
        soft_mode = "aspect_cap" if util >= 0.48 else "row_height"

        work_root = (
            Path(tempfile.gettempdir()) / "macro_place_dreamplace_bridge" / benchmark.name
        )
        configs = light_bridge_dreamplace_configs(
            benchmark,
            iterations=self.dreamplace_iterations,
            gpu=self.use_gpu,
        )
        if len(configs) < 2:
            return baseline

        seed_b = _perturbed_legalized_seed(
            baseline,
            benchmark,
            sigma_um=self.perturb_sigma_um,
        )

        cand_tensors: list[torch.Tensor] = []
        cand_labels: list[str] = []

        for seed_idx, (seed, seed_tag) in enumerate(
            [(baseline, "seed_casadi"), (seed_b, "seed_perturb")]
        ):
            for cfg_idx, config in enumerate(configs):
                try:
                    sub = generate_dreamplace_candidates(
                        benchmark,
                        plc,
                        work_root,
                        [config],
                        bookshelf_name=f"{benchmark.name}_s{seed_idx}_c{cfg_idx}",
                        dreamplace_root=self.dreamplace_root,
                        timeout_seconds=self.timeout_seconds,
                        initial_placement=seed,
                        soft_macro_mode=soft_mode,
                        soft_macro_row_cap_mult=12.0,
                        blend_alphas=(0.002, 0.004, 0.006, 0.01, 0.28, 0.45, 0.62),
                        full_blend_alphas=(
                            0.004,
                            0.006,
                            0.007,
                            0.008,
                            0.009,
                            0.01,
                            0.012,
                        ),
                        blend_hard_only=True,
                        legalize_outer_passes=1,
                        legalize_displacement_budget_frac=None,
                        legalize_rounds=280,
                        legalize_step_fraction=0.1,
                        legalize_iterative_cycles=1,
                    )
                except Exception:
                    continue
                for c in sub.candidates:
                    cand_tensors.append(c.placement)
                    cand_labels.append(f"{seed_tag}:cfg{cfg_idx}:{c.label}")

        if not cand_tensors:
            return baseline

        return select_best_true_proxy(
            baseline,
            cand_tensors,
            benchmark,
            plc,
            candidate_labels=cand_labels,
        ).placement
