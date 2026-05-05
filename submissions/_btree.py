"""B*-tree data structure with contour packing.

A B*-tree (Chang et al., 2000) is an ordered binary tree representing a
non-slicing, overlap-free floorplan:

  * Every macro is one node. Indices are ``0..n-1`` and refer to rows of the
    ``sizes`` array passed at construction.
  * Left child of v: placed immediately to the right of v on the contour
    (``x_ll = x_ll(v) + w(v)``).
  * Right child of v: placed at the same x as v, above v on the contour
    (``x_ll = x_ll(v)``, ``y_ll >= y_ll(v) + h(v)``).
  * Root anchors at ``(0, 0)``.

Decoding (``pack``) walks the tree DFS while maintaining a horizontal contour
(skyline) of the current placed area; this guarantees a strictly overlap-free
layout in O(n) amortized time per pack.

Coordinates inside this module are **lower-left corners**. The competition
infrastructure stores macro positions as **centers**, so callers must convert
at the boundary; helpers ``ll_from_centers`` / ``centers_from_ll`` are
provided for that.

This module is pure numpy and has no torch / benchmark dependency, so it can
be unit-tested in isolation and used inside a tight SA inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


__all__ = [
    "BStarTree",
    "ll_from_centers",
    "centers_from_ll",
]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def ll_from_centers(centers: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Convert center coordinates to lower-left corners."""
    return centers - 0.5 * sizes


def centers_from_ll(ll: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Convert lower-left coordinates to centers."""
    return ll + 0.5 * sizes


# ---------------------------------------------------------------------------
# B*-tree
# ---------------------------------------------------------------------------


_NIL = -1


@dataclass
class _ContourSeg:
    """One horizontal skyline segment ``[x0, x1)`` with top y ``top``."""

    x0: float
    x1: float
    top: float


class BStarTree:
    """Mutable B*-tree over a fixed set of axis-aligned rectangles.

    Storage:
      * ``sizes``  -- (n, 2) float array of (width, height); never mutated.
      * ``parent`` -- (n,) int array; ``_NIL`` for the root.
      * ``left``   -- (n,) int array; ``_NIL`` if no left child.
      * ``right``  -- (n,) int array; ``_NIL`` if no right child.
      * ``root``   -- index of the root node, or ``_NIL`` for an empty tree.

    Mutators (``swap_nodes``, ``rotate_children``, ``detach``, ``attach``)
    keep the bookkeeping arrays consistent. Use ``validate`` after any
    sequence of moves to catch logic bugs in tests.
    """

    __slots__ = ("sizes", "parent", "left", "right", "root", "_n")

    # ---- construction ----------------------------------------------------

    def __init__(self, sizes: np.ndarray):
        sizes = np.asarray(sizes, dtype=np.float64)
        if sizes.ndim != 2 or sizes.shape[1] != 2:
            raise ValueError(f"sizes must be (n, 2); got {sizes.shape}")
        self.sizes = sizes
        n = sizes.shape[0]
        self._n = n
        self.parent = np.full(n, _NIL, dtype=np.int64)
        self.left = np.full(n, _NIL, dtype=np.int64)
        self.right = np.full(n, _NIL, dtype=np.int64)
        self.root: int = _NIL

    @property
    def n(self) -> int:
        return self._n

    @classmethod
    def from_left_skew(cls, sizes: np.ndarray, order: List[int]) -> "BStarTree":
        """Build a degenerate left-skew tree along ``order`` (a flat row).

        Useful as a deterministic starting point: every macro becomes the
        left child of its predecessor, so the packed layout is a single
        row at y=0 in the order specified.
        """
        t = cls(sizes)
        if not order:
            return t
        if len(order) != t.n or sorted(order) != list(range(t.n)):
            raise ValueError("order must be a permutation of 0..n-1")
        t.root = order[0]
        for prev, cur in zip(order[:-1], order[1:]):
            t.left[prev] = cur
            t.parent[cur] = prev
        return t

    def copy(self) -> "BStarTree":
        out = BStarTree.__new__(BStarTree)
        out.sizes = self.sizes  # shared (immutable)
        out._n = self._n
        out.parent = self.parent.copy()
        out.left = self.left.copy()
        out.right = self.right.copy()
        out.root = self.root
        return out

    # ---- structure mutation ---------------------------------------------

    def swap_nodes(self, a: int, b: int) -> None:
        """Swap the *identities* of two nodes (their positions in the tree
        stay; only which macro sits at each position changes).

        This is one of the canonical SA moves on a B*-tree. It keeps tree
        topology and only relabels the macros at two positions.
        """
        if a == b:
            return
        # Swap parent/left/right pointers AT positions a, b
        for arr in (self.parent, self.left, self.right):
            arr[a], arr[b] = arr[b], arr[a]
        # Fix up neighbors that referenced a or b
        # (their pointers get remapped a<->b)
        for i in range(self._n):
            if self.parent[i] == a:
                self.parent[i] = b
            elif self.parent[i] == b:
                self.parent[i] = a
            if self.left[i] == a:
                self.left[i] = b
            elif self.left[i] == b:
                self.left[i] = a
            if self.right[i] == a:
                self.right[i] = b
            elif self.right[i] == b:
                self.right[i] = a
        # The above also relabeled a<->b's own self-pointers if any existed,
        # which is fine because parent[a] etc. already swapped above.
        if self.root == a:
            self.root = b
        elif self.root == b:
            self.root = a

    def rotate_children(self, v: int) -> None:
        """Swap left and right children of ``v`` in place."""
        self.left[v], self.right[v] = self.right[v], self.left[v]

    def detach(self, v: int) -> Tuple[int, int]:
        """Detach the subtree rooted at ``v`` from its parent.

        Returns ``(former_parent, side)`` where ``side`` is 0 if v was a
        left child, 1 if right, and (-1, -1) if v was the root.
        After detaching, ``v.parent`` is ``_NIL``; the rest of the subtree
        below v stays intact (so the caller can re-attach it elsewhere).
        """
        p = int(self.parent[v])
        if p == _NIL:
            if self.root == v:
                self.root = _NIL
            return (_NIL, _NIL)
        side = 0 if self.left[p] == v else 1
        if side == 0:
            self.left[p] = _NIL
        else:
            self.right[p] = _NIL
        self.parent[v] = _NIL
        return (p, side)

    def attach(self, v: int, parent: int, as_left: bool) -> None:
        """Attach ``v`` (must currently have no parent) under ``parent``.

        Bumps the existing child of ``parent`` on that side down into v's
        own free slot, so this acts as an "insert above" operation that
        never destroys a subtree.
        """
        if self.parent[v] != _NIL:
            raise ValueError(f"node {v} already has parent {self.parent[v]}")
        if parent == _NIL:
            if self.root != _NIL:
                # New root: push old root underneath v as a child.
                old_root = self.root
                slot_left = self.left[v] == _NIL
                slot_right = self.right[v] == _NIL
                if slot_left:
                    self.left[v] = old_root
                    self.parent[old_root] = v
                elif slot_right:
                    self.right[v] = old_root
                    self.parent[old_root] = v
                else:
                    raise ValueError(
                        f"cannot promote {v} to root: both child slots full"
                    )
            self.root = v
            return

        existing = int(self.left[parent] if as_left else self.right[parent])
        if existing != _NIL:
            # Push the displaced child into a free slot on v.
            if self.left[v] == _NIL:
                self.left[v] = existing
            elif self.right[v] == _NIL:
                self.right[v] = existing
            else:
                raise ValueError(
                    f"cannot attach {v} under {parent}: displaced subtree has nowhere to go"
                )
            self.parent[existing] = v

        if as_left:
            self.left[parent] = v
        else:
            self.right[parent] = v
        self.parent[v] = parent

    def move_subtree(self, v: int, new_parent: int, as_left: bool) -> None:
        """Move the subtree rooted at ``v`` under ``new_parent``.

        Fails if the destination slot is occupied (the destructive variant
        ``attach`` does the displacement; here we want a strict move).
        """
        if new_parent == v or self._is_descendant(new_parent, v):
            raise ValueError("cannot move subtree under one of its own descendants")
        if new_parent != _NIL:
            existing = int(self.left[new_parent] if as_left else self.right[new_parent])
            if existing != _NIL:
                raise ValueError(
                    f"slot occupied: parent={new_parent} {'left' if as_left else 'right'}"
                )
        self.detach(v)
        if new_parent == _NIL:
            # Reroot: only valid if there is no current root.
            if self.root != _NIL:
                raise ValueError("cannot reroot while another root exists")
            self.root = v
        else:
            if as_left:
                self.left[new_parent] = v
            else:
                self.right[new_parent] = v
            self.parent[v] = new_parent

    def _is_descendant(self, candidate: int, root: int) -> bool:
        """True if ``candidate`` lies in the subtree rooted at ``root``."""
        if candidate == _NIL or root == _NIL:
            return False
        stack = [root]
        while stack:
            x = stack.pop()
            if x == candidate:
                return True
            if self.left[x] != _NIL:
                stack.append(int(self.left[x]))
            if self.right[x] != _NIL:
                stack.append(int(self.right[x]))
        return False

    # ---- packing ---------------------------------------------------------

    def pack(self) -> np.ndarray:
        """Compute lower-left coordinates for every node by contour walk.

        Returns ``ll`` with shape ``(n, 2)``. Macros that are not part of
        the tree (orphans) get ``(nan, nan)``. The layout is guaranteed
        overlap-free as long as the tree is well-formed.
        """
        n = self._n
        ll = np.full((n, 2), np.nan, dtype=np.float64)
        if self.root == _NIL:
            return ll

        # Doubly-linked list of contour segments, kept sorted by x0.
        # Segment 0 is the sentinel "ground": [0, +inf) at y=0.
        contour: List[_ContourSeg] = [_ContourSeg(0.0, float("inf"), 0.0)]

        sizes = self.sizes

        def place(node: int, x_ll: float) -> float:
            """Place ``node`` at the given x_ll using current contour; return y_ll."""
            w = float(sizes[node, 0])
            h = float(sizes[node, 1])
            x_end = x_ll + w
            # i = index of first segment that overlaps [x_ll, x_end)
            i = 0
            while i < len(contour) and contour[i].x1 <= x_ll:
                i += 1
            # j = index past the last overlapping segment
            j = i
            top = 0.0
            while j < len(contour) and contour[j].x0 < x_end:
                if contour[j].top > top:
                    top = contour[j].top
                j += 1
            new_top = top + h
            removed = contour[i:j]
            # Preserve the part of the leftmost overlapping segment that
            # lies strictly to the left of x_ll (segment straddles x_ll).
            head: List[_ContourSeg] = []
            if removed and removed[0].x0 < x_ll:
                head.append(_ContourSeg(removed[0].x0, x_ll, removed[0].top))
            # Preserve the part of the rightmost overlapping segment that
            # lies strictly to the right of x_end (segment straddles x_end).
            tail: List[_ContourSeg] = []
            if removed and removed[-1].x1 > x_end:
                tail.append(_ContourSeg(x_end, removed[-1].x1, removed[-1].top))
            new_seg = _ContourSeg(x_ll, x_end, new_top)
            contour[i:j] = [*head, new_seg, *tail]
            return top

        # DFS in B*-tree order: left child immediately to the right at same x range,
        # right child stays in same x column above. We use an explicit stack of
        # (node, parent_x_ll, parent_w, parent_y_ll, parent_h, side) frames.
        # Standard iterative DFS:
        ll[self.root] = (0.0, place(self.root, 0.0))
        stack = [self.root]
        while stack:
            v = stack.pop()
            x_v = float(ll[v, 0])
            y_v = float(ll[v, 1])
            w_v = float(sizes[v, 0])
            h_v = float(sizes[v, 1])
            r = int(self.right[v])
            l = int(self.left[v])
            # Visit right child first onto the stack so left is processed
            # first when popped (depth-first along left spine).
            if r != _NIL:
                # Right child sits in the same x column as v, above the
                # current contour at that column.
                y_r = place(r, x_v)
                ll[r] = (x_v, y_r)
                stack.append(r)
            if l != _NIL:
                x_l = x_v + w_v
                y_l = place(l, x_l)
                ll[l] = (x_l, y_l)
                stack.append(l)

        return ll

    def bounding_box(self, ll: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Return ``(width, height)`` of the packed layout's bounding box."""
        if ll is None:
            ll = self.pack()
        valid = ~np.isnan(ll[:, 0])
        if not np.any(valid):
            return (0.0, 0.0)
        ur = ll[valid] + self.sizes[valid]
        return (float(ur[:, 0].max()), float(ur[:, 1].max()))

    # ---- validation ------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` on structural inconsistency."""
        n = self._n
        # Every non-root has parent, root has no parent.
        if self.root != _NIL and self.parent[self.root] != _NIL:
            raise ValueError("root has a parent")
        for i in range(n):
            for arr_name, arr in (("left", self.left), ("right", self.right)):
                c = int(arr[i])
                if c != _NIL and self.parent[c] != i:
                    raise ValueError(
                        f"node {c} listed as {arr_name} child of {i} but parent={self.parent[c]}"
                    )
        # Reachability: every node with parent != _NIL must be reachable from root.
        reached = set()
        if self.root != _NIL:
            stack = [self.root]
            while stack:
                v = stack.pop()
                if v in reached:
                    raise ValueError(f"cycle detected at {v}")
                reached.add(v)
                if self.left[v] != _NIL:
                    stack.append(int(self.left[v]))
                if self.right[v] != _NIL:
                    stack.append(int(self.right[v]))
        for i in range(n):
            in_tree = self.parent[i] != _NIL or i == self.root
            if in_tree and i not in reached:
                raise ValueError(f"node {i} claims to be in tree but unreachable")

    # ---- coord -> tree (M2) ---------------------------------------------

    @classmethod
    def from_ll_coords(
        cls,
        sizes: np.ndarray,
        ll: np.ndarray,
        *,
        atol: float = 1e-6,
    ) -> "BStarTree":
        """Convert lower-left coordinates of an overlap-free layout to a tree.

        Algorithm (left-edge sweep):
          1. Sort macros by ``(x_ll, y_ll)`` ascending.
          2. Walk in order; for each macro m, identify a parent via:
             a. **Left-child slot**: an already-placed macro p with
                ``p.x_ll + p.w ≈ m.x_ll`` whose left child is empty and
                whose contour at column ``[m.x_ll, m.x_ll + m.w)`` would
                produce ``y_ll ≈ m.y_ll`` after packing. Pick the one
                whose induced y is closest to ``m.y_ll``.
             b. **Right-child slot**: failing (a), pick already-placed
                macro p with ``p.x_ll ≈ m.x_ll``, no right child, and
                largest ``p.y_ll`` not exceeding ``m.y_ll``.
             c. As a fallback (input not strictly compacted), attach to
                the closest viable predecessor by Manhattan distance.
          3. Root = first macro in sorted order; if it sits at ``(0, 0)``
             we report exact recovery.

        This is **best-effort**: for layouts that were already packed by a
        B*-tree it recovers the exact tree; for arbitrary legal layouts
        the resulting tree's ``pack()`` may differ from the input by the
        compaction shift. Use ``recovery_error`` to measure deviation.
        """
        sizes = np.asarray(sizes, dtype=np.float64)
        ll = np.asarray(ll, dtype=np.float64)
        if sizes.shape != ll.shape:
            raise ValueError(
                f"sizes {sizes.shape} and ll {ll.shape} must match"
            )
        n = sizes.shape[0]
        t = cls(sizes)
        if n == 0:
            return t

        x = ll[:, 0]
        y = ll[:, 1]
        w = sizes[:, 0]
        # Sort lex(x_ll, y_ll)
        order = np.lexsort((y, x))

        # Track which child slots are still free.
        for k, m in enumerate(order):
            m = int(m)
            if k == 0:
                t.root = m
                continue

            best_parent = _NIL
            best_side = 1  # right
            best_score = float("inf")

            xm, ym = float(x[m]), float(y[m])
            for prev_idx in order[:k]:
                p = int(prev_idx)
                xp, yp = float(x[p]), float(y[p])
                wp = float(w[p])
                # Left-child candidate: p's right edge == m.x_ll
                if abs((xp + wp) - xm) <= atol and t.left[p] == _NIL:
                    # Score by |y_p - y_m|; lower is better
                    score = abs(yp - ym)
                    if score < best_score - atol or (
                        score < best_score + atol and best_side == 1
                    ):
                        best_score = score
                        best_parent = p
                        best_side = 0
                # Right-child candidate: same column, p below m
                elif abs(xp - xm) <= atol and yp <= ym + atol and t.right[p] == _NIL:
                    score = abs(ym - yp)
                    if score < best_score:
                        best_score = score
                        best_parent = p
                        best_side = 1

            if best_parent == _NIL:
                # Fallback: nearest already-placed macro by Manhattan distance
                # with any free slot. This handles non-compacted input.
                for prev_idx in order[:k]:
                    p = int(prev_idx)
                    has_left = t.left[p] != _NIL
                    has_right = t.right[p] != _NIL
                    if has_left and has_right:
                        continue
                    d = abs(float(x[p]) - xm) + abs(float(y[p]) - ym)
                    if d < best_score:
                        best_score = d
                        best_parent = p
                        best_side = 0 if not has_left else 1
                if best_parent == _NIL:
                    raise ValueError(
                        f"cannot find parent for macro {m}; tree saturated"
                    )

            if best_side == 0:
                t.left[best_parent] = m
            else:
                t.right[best_parent] = m
            t.parent[m] = best_parent

        return t

    @classmethod
    def from_centers(
        cls,
        sizes: np.ndarray,
        centers: np.ndarray,
        *,
        atol: float = 1e-6,
    ) -> "BStarTree":
        """Convenience wrapper: convert center coords to a B*-tree."""
        sizes = np.asarray(sizes, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64)
        return cls.from_ll_coords(sizes, ll_from_centers(centers, sizes), atol=atol)

    # ---- diagnostics -----------------------------------------------------

    def recovery_error(self, ll_target: np.ndarray) -> dict:
        """Compare ``self.pack()`` to a target layout in lower-left coords.

        Returns dict with ``max_dx``, ``max_dy``, ``mean_dx``, ``mean_dy``,
        ``bbox_packed``, ``bbox_target``. NaN-guarded.
        """
        ll_target = np.asarray(ll_target, dtype=np.float64)
        ll_packed = self.pack()
        valid = ~np.isnan(ll_packed[:, 0])
        d = np.abs(ll_packed[valid] - ll_target[valid])
        bb_p = self.bounding_box(ll_packed)
        ur_t = ll_target[valid] + self.sizes[valid]
        bb_t = (float(ur_t[:, 0].max()), float(ur_t[:, 1].max()))
        return {
            "max_dx": float(d[:, 0].max()) if d.size else 0.0,
            "max_dy": float(d[:, 1].max()) if d.size else 0.0,
            "mean_dx": float(d[:, 0].mean()) if d.size else 0.0,
            "mean_dy": float(d[:, 1].mean()) if d.size else 0.0,
            "bbox_packed": bb_p,
            "bbox_target": bb_t,
        }
