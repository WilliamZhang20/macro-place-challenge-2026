"""
Step 8 — Same as step 7 (multi-start + LNS). Klein-4 orientation is not consumed by
the Tier-1 harness (PlacementCost is reloaded with .plc orientations); a future
orientations sidecar or Tier-2 flow would be needed for orientation to affect scores.

This file keeps the pipeline slot from the action plan without adding heavy
per-macro PLC reload loops that would burn the runtime budget.

Usage:
  uv run evaluate submissions/descent_step08_orient_placer.py -b ibm01
"""

from __future__ import annotations

import sys
from pathlib import Path

_SUB = Path(__file__).resolve().parent
if str(_SUB) not in sys.path:
    sys.path.insert(0, str(_SUB))

from descent_step07_multistart_placer import (  # noqa: E402
    DescentStep07MultistartPlacer,
)


class DescentStep08OrientPlacer(DescentStep07MultistartPlacer):
    """Reserved for orientation polish when the evaluator exposes it for Tier 1."""

    pass
