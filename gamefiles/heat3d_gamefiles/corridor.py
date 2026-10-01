"""
The ground a course runs through: a band either side of the driven line.

A lap's bounding box is the right cut for a circuit and the wrong one for a
point-to-point race. A 14.7 km sprint runs corner to corner of a 6 by
3.6 km box, and the box is 22 square kilometres - twenty-five million
terrain triangles, nearly all of them hillside nobody drives past. The corridor
is the band that matters, and for a sprint it is a tenth of the box.

It is a raster: the driven line marked on a grid of `CELL`-metre squares and
widened by the margin in every direction. Asking whether a point, a tile or a
building is in it is then a lookup rather than a search along the line, which
is what makes it cheap enough to ask of every one of millions of triangles.
"""

from __future__ import annotations

import numpy as np

#: Grid resolution in metres. Finer than the margin by a factor of twenty, so
#: the band's edge is within a few metres of where the margin puts it.
CELL = 4.0

#: Consecutive samples closer than this are joined by a line; further apart,
#: they are not the same stretch of driving. A library course is dozens of lap
#: files end to end, and the last sample of one sprint and the first of the next
#: are the finish and the start - joined, they would mark a band across the map.
JOIN = 50.0


class Corridor:
    """Everything within `margin` metres of the driving, in the ground plane."""

    def __init__(self, route, margin: float, cell: float = CELL):
        route = np.asarray(route, dtype=np.float64)
        points = route[:, [0, 2]] if route.shape[1] >= 3 else route[:, :2]
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError("a corridor needs at least one point of driving")
        self.margin = float(margin)
        self.cell = float(cell)
        self.origin = points.min(axis=0) - margin - cell
        extent = points.max(axis=0) + margin + cell - self.origin
        self.shape = (int(np.ceil(extent[0] / cell)) + 1, int(np.ceil(extent[1] / cell)) + 1)
        self.grid = np.zeros(self.shape, dtype=bool)

        # Every cell the driving passes through, once, and a disc of the margin
        # around each. Samples are metres apart and the disc is eighty wide, so
        # the gaps between them are covered many times over.
        points = _densify(points, cell, JOIN)
        cells = np.unique(np.floor((points - self.origin) / cell).astype(np.int64), axis=0)
        reach = int(np.ceil(margin / cell))
        span = np.arange(-reach, reach + 1)
        dx, dz = np.meshgrid(span, span, indexing="ij")
        disc = (dx**2 + dz**2) * cell**2 <= (margin + cell) ** 2
        offsets = np.stack([dx[disc], dz[disc]], axis=1)
        for start in range(0, len(cells), 4096):
            marked = (cells[start : start + 4096, None, :] + offsets[None, :, :]).reshape(-1, 2)
            ok = (
                (marked[:, 0] >= 0)
                & (marked[:, 0] < self.shape[0])
                & (marked[:, 1] >= 0)
                & (marked[:, 1] < self.shape[1])
            )
            self.grid[marked[ok, 0], marked[ok, 1]] = True

    @property
    def low(self) -> tuple[float, float]:
        return float(self.origin[0]), float(self.origin[1])

    @property
    def high(self) -> tuple[float, float]:
        top = self.origin + np.array(self.shape) * self.cell
        return float(top[0]), float(top[1])

    @property
    def area(self) -> float:
        """Square metres inside the band."""
        return float(self.grid.sum()) * self.cell**2

    def contains(self, points) -> np.ndarray:
        """Which (x, z) points, or (x, y, z) points, lie in the band."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points[None, :]
        plane = points[:, [0, 2]] if points.shape[1] >= 3 else points[:, :2]
        index = np.floor((plane - self.origin) / self.cell).astype(np.int64)
        ok = (
            (index[:, 0] >= 0)
            & (index[:, 0] < self.shape[0])
            & (index[:, 1] >= 0)
            & (index[:, 1] < self.shape[1])
        )
        out = np.zeros(len(plane), dtype=bool)
        out[ok] = self.grid[index[ok, 0], index[ok, 1]]
        return out

    def touches(self, low, high) -> bool:
        """Whether any of the band lies in an (x, z) box."""
        a = np.floor((np.asarray(low, dtype=np.float64) - self.origin) / self.cell).astype(np.int64)
        b = np.floor((np.asarray(high, dtype=np.float64) - self.origin) / self.cell).astype(np.int64)
        a = np.clip(a, 0, np.array(self.shape) - 1)
        b = np.clip(b, 0, np.array(self.shape) - 1)
        if (np.asarray(high) < self.origin).any():
            return False
        top = self.origin + np.array(self.shape) * self.cell
        if (np.asarray(low) > top).any():
            return False
        return bool(self.grid[a[0] : b[0] + 1, a[1] : b[1] + 1].any())

    def triangles(self, points: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Which triangles reach into the band: any corner in it, or the middle of
        an edge, or the middle of the triangle.

        The middles catch a triangle larger than the band is wide lying across
        it with every corner outside - rare at the finest level of terrain, and
        a hole in the road where it happens.
        """
        if not len(faces):
            return np.zeros(0, dtype=bool)
        inside = self.contains(points)
        keep = inside[faces].any(axis=1)
        corners = points[faces]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            keep |= self.contains((corners[:, a] + corners[:, b]) / 2)
        keep |= self.contains(corners.mean(axis=1))
        return keep


def _densify(points: np.ndarray, step: float, join: float) -> np.ndarray:
    """The samples, with points added along every gap shorter than `join`."""
    if len(points) < 2:
        return points
    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    fill = (gaps > step) & (gaps < join)
    if not fill.any():
        return points
    extra = [points]
    for i in np.flatnonzero(fill):
        count = int(np.ceil(gaps[i] / step))
        t = np.arange(1, count)[:, None] / count
        extra.append(points[i] + t * (points[i + 1] - points[i]))
    return np.concatenate(extra)
