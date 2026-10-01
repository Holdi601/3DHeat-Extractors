"""
Depth maps into one surface.

Every keyframe contributes a few hundred thousand points, all of them noisy and
most of them redundant with points from the frames either side. Turning that into
geometry means deciding what the surface *is* where a dozen frames disagree by a
few centimetres, and doing it without keeping every point.

Occupancy rather than a signed distance field
---------------------------------------------
A proper TSDF carves along every pixel's ray, which is the better method and
needs the whole ray walked for each of some hundreds of millions of pixels. This
accumulates weighted occupancy per voxel instead and runs marching cubes on the
smoothed result.

The tradeoff is real and worth stating: a TSDF gets thin structures and clean
free space that this does not, because it knows a ray passed *through* empty
space on its way to a surface. Occupancy only knows where surfaces were seen. In
exchange it is fast enough to run while a scan is happening, on a machine that
may have no usable GPU, and the result is a closed surface rather than a point
cloud — which is what the viewer needs.

Weighting is by observation quality, not by count. A point measured from five
metres with good parallax should move the surface more than fifty points squinted
at from across a courtyard, and weighting by count alone gets that backwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .voxelindex import VoxelIndex, grow, unpack
from .voxelindex import pack as pack_keys

#: Occupancy level taken to be the surface. Marching cubes needs a crossing, and
#: the field runs from 0 in free space to 1 where many frames agree; halfway is
#: the natural place and keeps the surface between the noise on either side.
SURFACE_LEVEL = 0.5

#: A voxel seen fewer times than this is dropped before meshing. Single stray
#: points are almost always depth-network error at a silhouette edge, and left in
#: they become spikes hanging off the geometry.
MIN_WEIGHT = 0.35

#: Voxels per side of an extraction block.
#:
#: The surface is extracted a block at a time, because the alternative — one
#: dense array over the whole scan — is impossible at track scale: a kilometre
#: square at quarter-metre detail is 3.8 billion cells and 15 GB. 64 keeps each
#: block's array at a third of a megabyte while large enough that the halo below
#: is a small fraction of the work.
BLOCK = 64

#: Voxels of neighbouring data realised around each block.
#:
#: Without it every block boundary is a face with nothing beyond it, marching
#: cubes closes the surface against it, and a continuous road comes out as a grid
#: of boxes with walls between them. Three, because the smoothing kernel has
#: sigma 0.8 and reaches about three voxels — a halo narrower than the blur
#: leaves the field pulled down at the edges and a seam of thin surface instead.
HALO = 3

#: Field values sampled across the whole scan to set the surface level.
#: Two million is far more than a percentile needs and still a fixed few
#: megabytes, whatever the size of the scan.
SAMPLE_BUDGET = 2_000_000


def _empty_mesh() -> "Mesh":
    """A mesh with nothing in it, which several paths need to return."""
    return Mesh(
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.int32),
        np.zeros((0, 3), np.float32),
    )


@dataclass
class Mesh:
    """Triangles in world units, Y up."""

    vertices: np.ndarray  # (N,3) float32
    faces: np.ndarray  # (M,3) int32
    normals: np.ndarray  # (N,3) float32
    #: Per-vertex colour, 0..1, when frames contributed it.
    colours: np.ndarray | None = None

    @property
    def triangles(self) -> int:
        return int(self.faces.shape[0])

    def is_empty(self) -> bool:
        return self.faces.size == 0


class SurfaceVolume:
    """
    Weighted occupancy over a sparse voxel grid.

    Sparse because a walked route through a large map touches a tiny fraction of
    its bounding box, and a dense array over the box would be almost entirely
    zeros — at half-metre voxels a 400m map is half a billion cells, nearly none
    of them ever written.
    """

    def __init__(self, *, voxel: float = 0.25):
        if voxel <= 0:
            raise ValueError("voxel size must be positive")
        self.voxel = float(voxel)
        self._voxels = VoxelIndex()
        self._weight = np.zeros(0, dtype=np.float32)
        self._colour = np.zeros((0, 3), dtype=np.float32)
        self._has_colour = False

    def __len__(self) -> int:
        return len(self._voxels)

    def integrate(
        self,
        points: np.ndarray,
        *,
        weight: float = 1.0,
        colours: np.ndarray | None = None,
    ) -> int:
        """
        Add one frame's points. Returns the number of voxels touched.

        Points are not deduplicated across the frame before weighting, on
        purpose: a surface facing the camera squarely produces more pixels per
        square metre than one seen edge-on, and that density *is* evidence about
        how well it was measured.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if colours is not None:
            colours = np.asarray(colours, dtype=np.float32).reshape(-1, 3)[finite]
        if points.size == 0:
            return 0

        keys = np.floor(points / self.voxel).astype(np.int64)
        # Packed to int64 and made unique in one dimension, which is the whole
        # performance story here. `np.unique(keys, axis=0)` looks like the
        # natural call and builds a structured view to sort rows: measured on a
        # real keyframe's 129,600 points it takes **114 ms**, against **8 ms**
        # for the same answer via a packed scalar key. It appeared in both fusion
        # grids, so it alone was most of the 212 ms a keyframe spent fusing —
        # and that is what held the scan to four keyframes a second and, at
        # driving speed, pulled consecutive keyframes far enough apart that
        # feature matching failed. A throughput bug wearing a tracking bug's
        # clothes.
        unique, inverse, counts = np.unique(
            pack_keys(keys), return_inverse=True, return_counts=True
        )

        # Saturating rather than linear in the pixel count: past a point, more
        # pixels of the same surface from the same place stop being evidence.
        contribution = weight * np.sqrt(counts / counts.max())

        if colours is not None:
            summed = np.stack(
                [
                    np.bincount(inverse, weights=colours[:, c], minlength=len(unique))
                    for c in range(3)
                ],
                axis=1,
            )
            mean_colour = summed / counts[:, None]

        # Vectorised for the same reason coverage is: the loop this replaces
        # cost 95 ms a keyframe, which together with coverage made fusion 83% of
        # the work and left the scan too slow to track while driving.
        rows = self._voxels.add(unique)
        size = len(self._voxels)
        if size > self._weight.shape[0]:
            self._weight = grow(self._weight, size)
            self._colour = grow(self._colour, size)

        previous = self._weight[rows].astype(np.float64)
        added = contribution.astype(np.float64)
        self._weight[rows] = (previous + added).astype(np.float32)

        if colours is not None:
            self._has_colour = True
            total = previous + added
            safe = np.where(total > 1e-9, total, 1.0)[:, None]
            # A running weighted mean, so a voxel's colour follows the views that
            # actually contributed to its surface rather than the last one in.
            self._colour[rows] = (
                (self._colour[rows] * previous[:, None] + mean_colour * added[:, None]) / safe
            ).astype(np.float32)

        return len(unique)

    # -- meshing ---------------------------------------------------------

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if len(self._voxels) == 0:
            return None
        keys = unpack(self._voxels.keys)
        return keys.min(axis=0), keys.max(axis=0)

    def to_mesh(self, *, smooth: bool = True, min_weight: float = MIN_WEIGHT) -> Mesh:
        """
        Extract a surface, block by block over the parts that hold anything.

        The obvious implementation realises one dense array over the whole
        bounding box, and it is fine for a room. It cannot work for a racetrack,
        and the arithmetic is not close: a kilometre square at quarter-metre
        detail is 3.8 *billion* cells and 15 GB, and a real lap covers more. The
        previous code checked for that and returned an empty mesh — silently, so
        a scan holding three and a half million cells of good geometry reported
        "not enough geometry to build a surface" and lost the lot.

        Blocks work because the data is sparse: those cells sit in a few thousand
        occupied blocks out of billions of empty ones, so realising each occupied
        block costs a rounding error of the dense array.

        Two passes, and the second one is not optional
        ---------------------------------------------
        The field is normalised *after* smoothing, because a surface is a shell
        one voxel thick and a 3D Gaussian spreads it over its whole
        neighbourhood — an isolated occupied voxel falls to roughly an eighth of
        its peak, and without renormalising the whole field sits under the
        surface level and nothing extracts at all.

        But that normalisation has to be **global**. Done per block it rescales
        every block to its own maximum, which promotes a lone outlier — the
        stray point a depth network leaves at a silhouette edge — to a full
        surface in its own empty block, and hangs a blob forty metres off the
        geometry. So the first pass measures how strong the field actually gets
        across the whole scan, and the second extracts against that one number.
        """
        extent = self.bounds()
        if extent is None:
            return _empty_mesh()

        keys = unpack(self._voxels.keys)
        values = self._weight[: len(self._voxels)]
        keep = values >= min_weight
        if not keep.any():
            return _empty_mesh()
        keys, values = keys[keep], values[keep]

        # Scaled against the whole scan, so the surface level means the same
        # everywhere: a quiet corner and a busy one have to extract at the same
        # threshold or the two do not meet.
        scale = max(float(np.percentile(values, 75)), 1e-6)
        strength = np.clip(values / scale, 0.0, 1.0).astype(np.float32)

        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        keys, strength = keys[order], strength[order]

        blocks = sorted({tuple(b) for b in (keys // BLOCK)})

        # Pass one: how strong does the field get, once smoothed?
        #
        # Taken over the *values*, not over each block's maximum. The maxima
        # look like the cheaper summary and are not the same thing: a block
        # holding a sliver of road has a low maximum, and letting those into the
        # average drags the level down until the surface creeps outward. Measured
        # on the road fixture it cost 1.2 voxels of width — half a voxel at each
        # edge, which is exactly what a slightly wrong threshold on a blurred
        # field looks like.
        #
        # Sampled rather than collected whole: after blurring, a few million
        # occupied voxels become tens of millions of non-zero ones, and the
        # percentile does not need all of them.
        rng = np.random.default_rng(0)
        per_block = max(256, SAMPLE_BUDGET // max(len(blocks), 1))
        sample: list[np.ndarray] = []
        for block in blocks:
            field = self._block_field(block, keys, strength, smooth=smooth)
            if field is None:
                continue
            values_here = field[field > 1e-6]
            if values_here.size == 0:
                continue
            if values_here.size > per_block:
                values_here = rng.choice(values_here, per_block, replace=False)
            sample.append(values_here)
        if not sample:
            return _empty_mesh()
        level_scale = max(float(np.percentile(np.concatenate(sample), 99)), 1e-6)

        # Pass two: extract against that.
        parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        colours: list[np.ndarray] = []
        for block in blocks:
            field = self._block_field(block, keys, strength, smooth=smooth)
            if field is None:
                continue
            field = np.clip(field / level_scale, 0.0, 1.0)
            if field.max() < SURFACE_LEVEL:
                continue
            piece = self._extract_block(block, field)
            if piece is None:
                continue
            vertices, faces, normals = piece
            parts.append((vertices, faces, normals))
            if self._has_colour:
                colours.append(self._sample_colours(vertices))

        if not parts:
            return _empty_mesh()

        offsets = np.cumsum([0] + [len(v) for v, _f, _n in parts])
        vertices = np.concatenate([v for v, _f, _n in parts])
        faces = np.concatenate(
            [f + offset for (_v, f, _n), offset in zip(parts, offsets[:-1])]
        )
        normals = np.concatenate([n for _v, _f, n in parts])

        # Into world units. The half-voxel puts the result at cell centres
        # rather than corners.
        world = (vertices + 0.5) * self.voxel
        return Mesh(
            vertices=world.astype(np.float32),
            faces=faces.astype(np.int32),
            normals=normals.astype(np.float32),
            colours=np.concatenate(colours).astype(np.float32) if colours else None,
        )

    def _block_field(self, block, keys, strength, *, smooth: bool):
        """
        One block's occupancy, with a halo of its neighbours' voxels around it.

        The halo is what keeps the surface continuous. Without it every block
        boundary is a face with nothing beyond, marching cubes closes the surface
        against it, and a road comes out as a row of boxes with walls between
        them. It has to be wider than the smoothing kernel reaches, or the blur
        pulls the field down near the edges and leaves a seam of thin surface.
        """
        origin = np.array(block, dtype=np.int64) * BLOCK - HALO
        size = BLOCK + 2 * HALO
        high = origin + size - 1

        within = np.all(keys >= origin, axis=1) & np.all(keys <= high, axis=1)
        if not within.any():
            return None
        local = keys[within] - origin

        field = np.zeros((size, size, size), dtype=np.float32)
        field[local[:, 0], local[:, 1], local[:, 2]] = strength[within]

        if smooth:
            from scipy.ndimage import gaussian_filter

            # A gentle blur. Not for de-staircasing — the field is weighted
            # rather than binary, so marching cubes already has a gradient to
            # interpolate along. What this removes is the high-frequency wrinkle
            # that depth noise leaves on the surface.
            field = gaussian_filter(field, sigma=0.8)
        return field

    def _extract_block(self, block, field):
        """Triangles for one block, in voxel coordinates, halo trimmed away."""
        from skimage.measure import marching_cubes

        try:
            vertices, faces, normals, _ = marching_cubes(field, level=SURFACE_LEVEL)
        except (ValueError, RuntimeError):
            # A block whose field never crosses the level in a way marching
            # cubes can close. One block failing must not cost the scan.
            return None
        if len(vertices) == 0 or len(faces) == 0:
            return None

        # Each triangle belongs to the block its *centre* falls in, and that
        # rule is what makes the halo work rather than merely exist.
        #
        # The obvious rule — keep a triangle only if all three corners are
        # inside — loses every triangle that straddles a boundary: this block
        # rejects it for having a corner outside, and the neighbour rejects it
        # for having two, so it is emitted by nobody and the surface gets a gap
        # at every block seam. Measured on the road fixture that cost 0.6 m of a
        # 12 m width, which is a strip missing down both edges.
        #
        # A centroid falls in exactly one block, so every triangle is emitted
        # once: no gaps, and no coincident duplicates either.
        centres = vertices[faces].mean(axis=1)
        owned = np.all(
            (centres >= HALO) & (centres < HALO + BLOCK), axis=1
        )
        if not owned.any():
            return None
        faces = faces[owned]
        # Keep every vertex those triangles use, including ones sitting in the
        # halo — a triangle needs all three corners, wherever they are.
        kept = np.unique(faces)
        renumber = np.full(len(vertices), -1, dtype=np.int64)
        renumber[kept] = np.arange(len(kept))
        whole = np.ones(len(faces), dtype=bool)

        origin = np.array(block, dtype=np.int64) * BLOCK - HALO
        return (
            (vertices[kept] + origin).astype(np.float32),
            renumber[faces[whole]].astype(np.int64),
            normals[kept].astype(np.float32),
        )

    def _sample_colours(self, voxel_coords: np.ndarray) -> np.ndarray | None:
        keys = np.floor(voxel_coords).astype(np.int64)
        rows = self._voxels.find(pack_keys(keys))
        # Mid-grey where a vertex fell outside any occupied voxel, which happens
        # at the smoothed surface's outer skin. Black would read as a hole.
        out = np.full((len(keys), 3), 0.5, dtype=np.float32)
        known = rows >= 0
        out[known] = self._colour[rows[known]]
        return out


def classify(mesh: Mesh, *, ground_angle_deg: float = 35.0) -> np.ndarray:
    """
    Label each triangle `ground` or `structure` by which way it faces.

    The viewer draws the two differently — solid ground you read position
    against, near-transparent structures that would otherwise hide the heat
    inside them — and an engine export knows the answer from its own scene graph.
    A reconstruction has only the geometry, so the label comes from the surface
    normal: near-horizontal and facing up is ground, everything else is not.

    Water is never claimed. It is not distinguishable from flat ground by shape
    alone, and a wrong `water:` label would render a courtyard as a lake.
    """
    if mesh.is_empty():
        return np.zeros(0, dtype="<U9")
    corners = mesh.vertices[mesh.faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.divide(
        face_normals, lengths, out=np.zeros_like(face_normals), where=lengths > 1e-12
    )
    # Vertex normals decide the sign: a triangle's winding alone cannot say which
    # side is outside, but the extracted normals can.
    reference = mesh.normals[mesh.faces].mean(axis=1)
    flip = np.sum(face_normals * reference, axis=1) < 0
    face_normals[flip] *= -1

    upward = face_normals[:, 1] >= np.cos(np.radians(ground_angle_deg))
    return np.where(upward, "ground", "structure")


def mesh_to_parts(mesh: Mesh, name: str = "scan", *, colours: np.ndarray | None = None) -> list:
    """
    Split a mesh into the labelled parts the GLB writer wants.

    `colours` overrides the mesh's own voxel-averaged colour, for when the
    projection pass has produced something sharper.
    """
    from ..geometry.glb import Part

    labels = classify(mesh)
    vertex_colours = colours if colours is not None else mesh.colours
    parts = []
    for label in ("ground", "structure"):
        chosen = labels == label
        if not chosen.any():
            continue
        faces = mesh.faces[chosen]
        # Re-index to only the vertices this part uses, so each part carries its
        # own compact buffer instead of the whole scan's vertices.
        used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        parts.append(
            Part(
                cls=label,
                name=name,
                positions=mesh.vertices[used],
                indices=inverse.astype(np.uint32),
                normals=mesh.normals[used],
                colours=None if vertex_colours is None else vertex_colours[used],
            )
        )
    return parts
