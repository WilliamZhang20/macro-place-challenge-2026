"""Unit tests for the B*-tree data structure (M1) and coord -> tree (M2)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_btree_module():
    spec = importlib.util.spec_from_file_location(
        "_btree", "submissions/_btree.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_btree"] = mod
    spec.loader.exec_module(mod)
    return mod


btree_mod = _load_btree_module()
BStarTree = btree_mod.BStarTree
ll_from_centers = btree_mod.ll_from_centers
centers_from_ll = btree_mod.centers_from_ll


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_overlap(ll: np.ndarray, sizes: np.ndarray, tol: float = 1e-9) -> bool:
    """Brute-force overlap check (O(n^2)). Use only on small instances."""
    n = ll.shape[0]
    ur = ll + sizes
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
                return True
    return False


# ---------------------------------------------------------------------------
# M1: data structure & packing
# ---------------------------------------------------------------------------


def test_empty_and_single_node():
    t = BStarTree(np.zeros((0, 2)))
    assert t.n == 0
    assert t.pack().shape == (0, 2)

    sizes = np.array([[3.0, 2.0]])
    t = BStarTree(sizes)
    t.root = 0
    ll = t.pack()
    assert ll.shape == (1, 2)
    assert ll[0, 0] == 0.0 and ll[0, 1] == 0.0
    assert t.bounding_box() == (3.0, 2.0)


def test_left_skew_is_a_row():
    """Left-skew tree puts every node in a row at y=0."""
    sizes = np.array([[2.0, 1.0], [3.0, 1.5], [1.0, 1.2], [4.0, 0.8]])
    t = BStarTree.from_left_skew(sizes, [0, 1, 2, 3])
    t.validate()
    ll = t.pack()
    # All at y=0, x cumulative
    assert np.allclose(ll[:, 1], 0.0)
    expected_x = np.array([0.0, 2.0, 5.0, 6.0])
    assert np.allclose(ll[:, 0], expected_x)
    assert not _has_overlap(ll, sizes)
    assert t.bounding_box() == (10.0, 1.5)


def test_hand_checked_5_node_layout():
    """Hand-build a B*-tree and verify packing matches expected coordinates.

    Tree:
              0 (3x2)
             / \\
           1     2 (1.5x4)
          (2x1) /
              3 (1x1)

    Where 1 is left child of 0, 2 is right child of 0, 3 is left child of 2.

    Expected packing:
      0 at (0, 0), size 3x2  -> covers [0,3) x [0,2)
      1 (left of 0)  at (3, 0), size 2x1   -> covers [3,5) x [0,1)
      2 (right of 0) at (0, 2), size 1.5x4 -> covers [0,1.5) x [2,6)
      3 (left of 2)  at (1.5, ?). column [1.5, 2.5). top of contour there:
         contour after placing 0,1,2: segments with tops:
           [0, 1.5)  top=6   (from macro 2 at y=2 height 4)
           [1.5, 3)  top=2   (from macro 0)
           [3, 5)    top=1   (from macro 1)
         Over [1.5, 2.5): max top is max(2, 2)=2. So 3 sits at y=2.
         3 at (1.5, 2).
    """
    sizes = np.array(
        [
            [3.0, 2.0],
            [2.0, 1.0],
            [1.5, 4.0],
            [1.0, 1.0],
            [0.0, 0.0],  # extra unused macro to test orphan handling
        ]
    )
    t = BStarTree(sizes)
    t.root = 0
    t.left[0] = 1
    t.parent[1] = 0
    t.right[0] = 2
    t.parent[2] = 0
    t.left[2] = 3
    t.parent[3] = 2
    t.validate()

    ll = t.pack()
    expected = {
        0: (0.0, 0.0),
        1: (3.0, 0.0),
        2: (0.0, 2.0),
        3: (1.5, 2.0),
    }
    for idx, (x, y) in expected.items():
        assert np.isclose(ll[idx, 0], x), f"macro {idx}: x={ll[idx, 0]} expected {x}"
        assert np.isclose(ll[idx, 1], y), f"macro {idx}: y={ll[idx, 1]} expected {y}"
    # Macro 4 is orphan
    assert np.isnan(ll[4, 0])
    # Verify no overlap among placed macros
    assert not _has_overlap(ll[:4], sizes[:4])


def test_swap_nodes_preserves_topology():
    sizes = np.array([[2.0, 2.0], [1.0, 3.0], [3.0, 1.0], [1.5, 1.5]])
    t = BStarTree.from_left_skew(sizes, [0, 1, 2, 3])
    t.validate()
    t.swap_nodes(1, 2)
    t.validate()
    # After swap, the row order should be 0, 2, 1, 3 (positions are about
    # identities, so the second slot now holds macro 2 and third holds macro 1)
    ll = t.pack()
    # x positions are still cumulative widths in the new identity order.
    # Position-1 holds macro 2 (w=3), position-2 holds macro 1 (w=1).
    # Width sequence: 0(w=2), 2(w=3), 1(w=1), 3(w=1.5)
    expected_x = {0: 0.0, 2: 2.0, 1: 5.0, 3: 6.0}
    for idx, x in expected_x.items():
        assert np.isclose(ll[idx, 0], x)


def test_copy_is_deep():
    sizes = np.array([[1.0, 1.0], [1.0, 1.0]])
    a = BStarTree.from_left_skew(sizes, [0, 1])
    b = a.copy()
    b.swap_nodes(0, 1)
    # a unchanged
    assert a.root == 0
    assert a.left[0] == 1
    assert b.root == 1
    assert b.left[1] == 0


def test_detach_attach_roundtrip():
    sizes = np.array([[1.0, 1.0]] * 5)
    t = BStarTree.from_left_skew(sizes, [0, 1, 2, 3, 4])
    t.validate()
    # Detach node 2 (which has node 3 as left child); subtree {2, 3, 4}
    parent, side = t.detach(2)
    assert parent == 1 and side == 0
    assert t.left[1] == -1
    # Re-attach 2 back as left child of 1
    t.attach(2, 1, as_left=True)
    t.validate()


def test_pack_no_overlaps_random_trees():
    """Random valid trees produce overlap-free packings."""
    rng = np.random.default_rng(42)
    for trial in range(20):
        n = int(rng.integers(5, 30))
        sizes = rng.uniform(0.5, 3.0, size=(n, 2))
        order = list(range(n))
        rng.shuffle(order)
        t = BStarTree(sizes)
        t.root = order[0]
        # Random binary tree
        for k in range(1, n):
            child = order[k]
            # Pick a node from the already-inserted set with a free slot
            placed = order[:k]
            rng.shuffle(placed)
            for p in placed:
                free_left = t.left[p] == -1
                free_right = t.right[p] == -1
                if free_left and free_right:
                    side = bool(rng.integers(0, 2))
                elif free_left:
                    side = True
                elif free_right:
                    side = False
                else:
                    continue
                if side:
                    t.left[p] = child
                else:
                    t.right[p] = child
                t.parent[child] = p
                break
        t.validate()
        ll = t.pack()
        assert not _has_overlap(ll, sizes), f"overlap at trial {trial}"


# ---------------------------------------------------------------------------
# M2: coord -> tree conversion
# ---------------------------------------------------------------------------


def test_from_ll_recovers_known_tree():
    """Build a tree, pack it, then convert coords back; re-packed coords
    must match the original packing exactly (the recovered tree is
    equivalent up to free-tie choices, so we compare LL coords)."""
    sizes = np.array(
        [[3.0, 2.0], [2.0, 1.0], [1.5, 4.0], [1.0, 1.0]]
    )
    t = BStarTree(sizes)
    t.root = 0
    t.left[0] = 1
    t.parent[1] = 0
    t.right[0] = 2
    t.parent[2] = 0
    t.left[2] = 3
    t.parent[3] = 2

    ll_orig = t.pack()
    t2 = BStarTree.from_ll_coords(sizes, ll_orig)
    t2.validate()
    ll_new = t2.pack()
    # The LL coords should match exactly (not necessarily the tree topology,
    # but the layout is what matters for downstream cost).
    assert np.allclose(ll_new, ll_orig, atol=1e-9), (
        f"LL coords differ:\noriginal:\n{ll_orig}\nrecovered:\n{ll_new}"
    )


def test_from_ll_random_packed_layouts_roundtrip():
    """Generate random B*-trees, pack, then recover; LL coords must match."""
    rng = np.random.default_rng(7)
    failures = 0
    for trial in range(15):
        n = int(rng.integers(6, 25))
        sizes = rng.uniform(0.5, 3.0, size=(n, 2))
        order = list(range(n))
        rng.shuffle(order)
        t = BStarTree(sizes)
        t.root = order[0]
        for k in range(1, n):
            child = order[k]
            placed = order[:k]
            rng.shuffle(placed)
            for p in placed:
                free_left = t.left[p] == -1
                free_right = t.right[p] == -1
                if free_left and free_right:
                    side = bool(rng.integers(0, 2))
                elif free_left:
                    side = True
                elif free_right:
                    side = False
                else:
                    continue
                if side:
                    t.left[p] = child
                else:
                    t.right[p] = child
                t.parent[child] = p
                break
        ll_orig = t.pack()
        t2 = BStarTree.from_ll_coords(sizes, ll_orig)
        ll_new = t2.pack()
        if not np.allclose(ll_new, ll_orig, atol=1e-9):
            failures += 1
    # Allow rare ties to produce different but-still-valid trees with same
    # bounding box; require no overlaps and matching bbox even on those.
    # In practice the algorithm should hit zero failures here.
    assert failures == 0, f"{failures}/15 random roundtrips diverged"


# ---------------------------------------------------------------------------
# Smoke test on ibm01 (skipped if benchmark not present)
# ---------------------------------------------------------------------------


def test_ibm01_centers_to_btree_smoke():
    """Convert ibm01's hand-crafted .plc hard-macro layout to a B*-tree.

    We don't expect an exact recovery (the .plc is not B*-tree-packed),
    but the resulting tree must:
      * be structurally valid,
      * produce a zero-overlap packing,
      * fit inside a bounding box not pathologically larger than the canvas.
    """
    pt = Path("benchmarks/processed/public/ibm01.pt")
    if not pt.exists():
        pytest.skip("ibm01 .pt not present")
    from macro_place.benchmark import Benchmark

    b = Benchmark.load(str(pt))
    n_hard = b.num_hard_macros
    sizes = b.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
    centers = b.macro_positions[:n_hard].cpu().numpy().astype(np.float64)

    t = BStarTree.from_centers(sizes, centers)
    t.validate()
    ll = t.pack()
    assert not np.any(np.isnan(ll[:, 0])), "every hard macro must be in tree"

    # Brute-force overlap check is O(n^2) but n ~= 246 so fine.
    assert not _has_overlap(ll, sizes, tol=1e-9), "packing has overlaps"

    bw, bh = t.bounding_box(ll)
    canvas_w = float(b.canvas_width)
    canvas_h = float(b.canvas_height)
    # Compaction should be no larger than canvas in either dimension at this
    # utilization (~43%); allow 1.5x slack as a sanity bar (tight bound for
    # hard-macro-only packing is sqrt(util)).
    assert bw <= canvas_w * 1.5, f"bbox width {bw} >> canvas {canvas_w}"
    assert bh <= canvas_h * 1.5, f"bbox height {bh} >> canvas {canvas_h}"

    # Report recovery error for human inspection
    err = t.recovery_error(ll_from_centers(centers, sizes))
    print(
        f"ibm01 recovery: max_dx={err['max_dx']:.4f} max_dy={err['max_dy']:.4f} "
        f"mean_dx={err['mean_dx']:.4f} mean_dy={err['mean_dy']:.4f} "
        f"bbox_packed={err['bbox_packed']} canvas=({canvas_w:.3f}, {canvas_h:.3f})"
    )
