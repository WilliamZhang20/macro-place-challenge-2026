"""Short-budget HardMacroLnsPlacer for regression triage."""

from __future__ import annotations

import sys
from pathlib import Path

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from hard_macro_lns_placer import HardMacroLnsPlacer


class HardMacroLnsQuickPlacer(HardMacroLnsPlacer):
    def __init__(self):
        super().__init__(max_candidates=2, max_seconds=120.0, rudy_weight=0.5)
