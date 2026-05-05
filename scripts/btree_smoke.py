"""Quick sanity sweep of the B*-tree coord->tree converter on every public IBM benchmark.

Reports per-benchmark:
  * number of hard macros
  * canvas (w, h)
  * packed bbox (w, h) after recovery
  * mean / max recovery deviation in microns
  * any overlap in the packed layout (must be 0)

Run with: python scripts/btree_smoke.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_btree():
    spec = importlib.util.spec_from_file_location(
        "_btree", "submissions/_btree.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_btree"] = mod
    spec.loader.exec_module(mod)
    return mod


def _has_overlap(ll: np.ndarray, sizes: np.ndarray, tol: float = 1e-9) -> int:
    n = ll.shape[0]
    ur = ll + sizes
    count = 0
    for i in range(n):
        if np.isnan(ll[i, 0]):
            continue
        for j in range(i + 1, n):
            if np.isnan(ll[j, 0]):
                continue
            if (
                ll[i, 0] + tol < ur[j, 0]
                and ll[j, 0] + tol < ur[i, 0]
                and ll[i, 1] + tol < ur[j, 1]
                and ll[j, 1] + tol < ur[i, 1]
            ):
                count += 1
    return count


def main() -> int:
    btree_mod = _load_btree()
    BStarTree = btree_mod.BStarTree
    ll_from_centers = btree_mod.ll_from_centers

    from macro_place.benchmark import Benchmark

    pt_dir = Path("benchmarks/processed/public")
    cases = sorted(p.stem for p in pt_dir.glob("ibm*.pt"))
    if not cases:
        print(f"no benchmarks found under {pt_dir}")
        return 1

    header = (
        f"{'bench':<8} {'nhard':>5} {'canvas':>16} {'bbox':>16} "
        f"{'max_d':>8} {'mean_d':>8} {'overlaps':>9}"
    )
    print(header)
    print("-" * len(header))

    failures = 0
    for stem in cases:
        b = Benchmark.load(str(pt_dir / f"{stem}.pt"))
        n_hard = b.num_hard_macros
        sizes = b.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        centers = b.macro_positions[:n_hard].cpu().numpy().astype(np.float64)
        ll_target = ll_from_centers(centers, sizes)

        try:
            t = BStarTree.from_centers(sizes, centers)
            t.validate()
            ll_packed = t.pack()
            err = t.recovery_error(ll_target)
            overlaps = _has_overlap(ll_packed, sizes)
        except Exception as e:  # noqa: BLE001
            print(f"{stem:<8}  ERROR: {e}")
            failures += 1
            continue

        cw, ch = float(b.canvas_width), float(b.canvas_height)
        bw, bh = err["bbox_packed"]
        max_d = max(err["max_dx"], err["max_dy"])
        mean_d = 0.5 * (err["mean_dx"] + err["mean_dy"])
        print(
            f"{stem:<8} {n_hard:>5} "
            f"{cw:>7.2f}x{ch:<7.2f} "
            f"{bw:>7.2f}x{bh:<7.2f} "
            f"{max_d:>8.3f} {mean_d:>8.3f} {overlaps:>9d}"
        )
        if overlaps:
            failures += 1

    print()
    print(f"failures: {failures}/{len(cases)}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
