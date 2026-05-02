"""Submission entrypoint for the RePlAce-backed placement pipeline."""

from __future__ import annotations

import sys
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
        self.pipeline = ReplacePipeline()

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self.pipeline.place(benchmark)
