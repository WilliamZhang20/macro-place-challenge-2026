"""Submission: multi-start DREAMPlace + true-proxy selection with guardrail.

Requires ``external/DREAMPlace/install`` (``scripts/setup_dreamplace.sh``).
GPU/CPU follows ``MACRO_PLACE_DP_GPU`` / ``DreamPlacePipeline`` defaults.

Environment:
  MACRO_PLACE_DP_CONFIG — path to JSON from ``scripts/tune_dreamplace_optuna.py
  --write-best-config`` (``pipeline`` + ``dreamplace_json_overrides`` keys).
  MACRO_PLACE_DP_RICH_CANDIDATES=0 — disable the default feature-derived
  DREAMPlace mode cycling.  By default the pipeline runs a diverse valid-start
  DREAMPlace portfolio with explicit hard-macro legalization, then pushes the
  best candidate through RePlAce, coordinate descent, and GWTW refinement.
  Optional key ``pipeline.rich_candidate_set`` in the JSON overrides the env flag.
  ``pipeline.hyperband_enabled`` is supported for experiments, but defaults off:
  short-run successive halving was observed to promote starts that later
  regressed proxy cost.

Example:
  evaluate submissions/dreamplace_pipeline_placer.py -b ibm01
  MACRO_PLACE_DP_RICH_CANDIDATES=0 evaluate submissions/dreamplace_pipeline_placer.py -b ibm01
  MACRO_PLACE_DP_CONFIG=tuning_logs/dreamplace_optuna_best.json evaluate submissions/dreamplace_pipeline_placer.py -b ibm01
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _dreamplace_pipeline import DreamPlacePipeline  # noqa: E402

_PIPELINE_JSON_KEYS = frozenset(
    {
        "num_starts",
        "jitter_sigma_um",
        "global_iterations",
        "num_bins",
        "num_threads",
        "target_density",
        "timeout_seconds",
        "scale_iterations_with_features",
        "use_gpu",
        "rich_candidate_set",
        "pre_dp_valid_starts",
        "pre_dp_valid_pool_size",
        "pre_dp_valid_selection",
        "pre_dp_proxy_eval_limit",
        "explicit_legalize_dp_outputs",
        "post_rescue_coord_descent_seconds",
        "post_rescue_coord_descent_max_passes",
        "post_rescue_coord_descent_k_bound",
        "post_rescue_coord_descent_cell_search_prob",
        "post_rescue_coord_descent_node_order",
        "top_dp_for_rescue",
        "post_rescue_gwtw_seconds",
        "post_rescue_gwtw_num_workers",
        "post_rescue_gwtw_num_iters",
        "post_rescue_gwtw_syncup_freq",
        "post_rescue_gwtw_top_k",
        "post_rescue_gwtw_t_max",
        "post_rescue_gwtw_t_min",
        "post_rescue_gwtw_action_probs",
        "replace_rescue",
        "replace_rescue_trigger_proxy",
        "replace_rescue_timeout_seconds",
        "hyperband_enabled",
        "hyperband_eta",
        "hyperband_min_iterations",
    }
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _pipeline_from_tuner_json(
    payload: dict,
    *,
    rich_from_env: bool,
) -> DreamPlacePipeline:
    pipe = payload.get("pipeline")
    if pipe is not None and not isinstance(pipe, dict):
        raise TypeError("JSON 'pipeline' must be an object")
    overrides = payload.get("dreamplace_json_overrides")
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError("JSON 'dreamplace_json_overrides' must be an object")

    kwargs: dict = {"rich_candidate_set": rich_from_env}
    if pipe:
        for key in _PIPELINE_JSON_KEYS:
            if key == "rich_candidate_set":
                continue
            if key not in pipe:
                continue
            val = pipe[key]
            if key == "use_gpu" and val is not None:
                val = bool(val)
            kwargs[key] = val
        if "rich_candidate_set" in pipe:
            kwargs["rich_candidate_set"] = bool(pipe["rich_candidate_set"])
    if overrides is not None:
        kwargs["dreamplace_json_overrides"] = overrides
    return DreamPlacePipeline(**kwargs)


class DreamplacePipelinePlacer:
    """Evaluator-facing wrapper around :class:`DreamPlacePipeline`."""

    def __init__(self):
        rich = _env_flag("MACRO_PLACE_DP_RICH_CANDIDATES", default=True)
        cfg_path = os.environ.get("MACRO_PLACE_DP_CONFIG", "").strip()
        if cfg_path:
            p = Path(cfg_path).expanduser()
            if not p.is_file():
                raise FileNotFoundError(
                    f"MACRO_PLACE_DP_CONFIG={cfg_path!r} does not exist or is not a file"
                )
            payload = json.loads(p.read_text(encoding="utf-8"))
            self.pipeline = _pipeline_from_tuner_json(payload, rich_from_env=rich)
        else:
            self.pipeline = DreamPlacePipeline(rich_candidate_set=rich)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self.pipeline.place(benchmark)
