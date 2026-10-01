"""
A sparse voxel index that does not spend its life in a Python loop.

Both grids — occupancy and coverage — need the same thing: given tens of
thousands of integer voxel coordinates, find the row each already occupies or
allocate a new one. The obvious implementation is a dict keyed by a coordinate
tuple and a `for` loop, and that is what both had.

Measured on a real keyframe at 960x540, half a million back-projected points
thinned to 130k: `volume.integrate` took 95 ms and `coverage.observe` 117 ms,
against 38 ms for the depth network they exist to consume. Fusion was 83% of the
cost of a keyframe and the scan ran at about four a second.

That number is not merely slow, it breaks *correctness* downstream. Frames are
selected by how much the picture has changed, so a scan that can only process
four a second while driving leaves consecutive keyframes metres apart — and
feature matching across that baseline fails. Tracking falls apart, and it looks
like a tracking problem rather than a throughput one.

So the loop goes. Coordinates pack into a single int64, and lookup is a
`searchsorted` over a sorted key array: fully vectorised, no per-voxel Python.
Insertion appends and re-sorts, which is O(n log n) once per frame rather than
O(n) interpreted operations per voxel.
"""

from __future__ import annotations

import numpy as np

#: Bits per axis in the packed key. 21 bits signed gives ±1,048,575 voxels per
#: axis — at quarter-metre voxels, ±262 km, which is past any game map.
_BITS = 21
_OFFSET = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1


def pack(keys: np.ndarray) -> np.ndarray:
    """(N,3) integer voxel coordinates to (N,) int64 keys."""
    keys = np.asarray(keys, dtype=np.int64)
    if keys.size and (np.abs(keys).max() >= _OFFSET):
        raise ValueError(
            f"voxel coordinate beyond ±{_OFFSET}; the scan is larger than this index supports"
        )
    shifted = keys + _OFFSET
    return (
        (shifted[:, 0] & _MASK)
        | ((shifted[:, 1] & _MASK) << _BITS)
        | ((shifted[:, 2] & _MASK) << (2 * _BITS))
    )


def unpack(packed: np.ndarray) -> np.ndarray:
    """The inverse, for reading positions back out."""
    packed = np.asarray(packed, dtype=np.int64)
    return (
        np.stack(
            [
                packed & _MASK,
                (packed >> _BITS) & _MASK,
                (packed >> (2 * _BITS)) & _MASK,
            ],
            axis=1,
        )
        - _OFFSET
    )


class VoxelIndex:
    """
    Maps packed voxel keys to dense row numbers, in bulk.

    Rows are handed out in first-seen order and never move, so the arrays a
    caller keeps alongside this — counts, weights, colours — can simply be
    appended to and indexed by row.
    """

    def __init__(self) -> None:
        #: Keys in row order: `_keys[row]` is the voxel that row stands for.
        self._keys = np.zeros(0, dtype=np.int64)
        #: The same keys sorted, with the row each came from. Rebuilt on insert.
        self._sorted = np.zeros(0, dtype=np.int64)
        self._rows = np.zeros(0, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._keys.size)

    @property
    def keys(self) -> np.ndarray:
        """Packed keys, in row order."""
        return self._keys

    def find(self, packed: np.ndarray) -> np.ndarray:
        """Row for each key, or -1 where it is not present. Vectorised."""
        packed = np.asarray(packed, dtype=np.int64)
        if self._sorted.size == 0 or packed.size == 0:
            return np.full(packed.shape, -1, dtype=np.int64)
        position = np.searchsorted(self._sorted, packed)
        # `searchsorted` can return one past the end for keys beyond the range.
        position = np.minimum(position, self._sorted.size - 1)
        found = self._sorted[position] == packed
        return np.where(found, self._rows[position], -1)

    def add(self, packed: np.ndarray) -> np.ndarray:
        """
        Row for each key, allocating rows for the ones not seen before.

        `packed` must already be unique — callers get that from `np.unique` on
        the frame's voxels, which they need anyway to group the points.
        """
        packed = np.asarray(packed, dtype=np.int64)
        if packed.size == 0:
            return np.zeros(0, dtype=np.int64)

        rows = self.find(packed)
        fresh = rows < 0
        count = int(fresh.sum())
        if count:
            start = self._keys.size
            rows[fresh] = np.arange(start, start + count, dtype=np.int64)
            self._keys = np.concatenate([self._keys, packed[fresh]])
            # One sort per frame, rather than a hash lookup per voxel. At the
            # sizes involved this is tens of milliseconds where the loop was
            # hundreds, and it stays flat as the scan grows.
            order = np.argsort(self._keys, kind="stable")
            self._sorted = self._keys[order]
            self._rows = order.astype(np.int64)
        return rows


def grow(array: np.ndarray, size: int, fill=0) -> np.ndarray:
    """Extend a row-indexed array to `size`, preserving what is there."""
    if array.shape[0] >= size:
        return array
    shape = (size,) + array.shape[1:]
    out = np.full(shape, fill, dtype=array.dtype)
    out[: array.shape[0]] = array
    return out
