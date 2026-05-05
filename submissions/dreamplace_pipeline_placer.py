"""Submission: multi-start DREAMPlace + true-proxy selection among DP runs only.

Requires ``external/DREAMPlace/install`` (``scripts/setup_dreamplace.sh``).
GPU/CPU follows ``MACRO_PLACE_DP_GPU`` / ``DreamPlacePipeline`` defaults.

Example:
  evaluate submissions/dreamplace_pipeline_placer.py -b ibm01
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _dreamplace_pipeline import DreamPlacePipeline  # noqa: E402


class DreamplacePipelinePlacer:
    """Evaluator-facing wrapper around :class:`DreamPlacePipeline`."""

    def __init__(self):
        self.pipeline = DreamPlacePipeline()

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self.pipeline.place(benchmark)
