"""Submission entrypoint for the RePlAce-backed placement pipeline."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _replace_pipeline import ReplacePipeline  # noqa: E402


class ReplacePipelinePlacer:
    """Thin evaluator-facing wrapper around :class:`ReplacePipeline`."""

    def __init__(self):
        self.pipeline = ReplacePipeline(
            adaptive_multistart=_env_flag("MACRO_PLACE_REPLACE_ADAPTIVE", default=True),
            adaptive_top_k=_env_int("MACRO_PLACE_REPLACE_ADAPTIVE_TOP_K", default=3),
            adaptive_probe_timeout_seconds=_env_float(
                "MACRO_PLACE_REPLACE_PROBE_TIMEOUT",
                default=None,
            ),
            adaptive_full_timeout_seconds=_env_float(
                "MACRO_PLACE_REPLACE_FULL_TIMEOUT",
                default=None,
            ),
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self.pipeline.place(benchmark)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)
