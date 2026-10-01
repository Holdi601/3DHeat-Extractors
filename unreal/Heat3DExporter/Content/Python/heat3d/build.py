"""
Turning the kept actors into one mesh, without holding the whole map in memory.

Three ideas carry this:

  - **Read each unique mesh once.** A map places a few thousand distinct assets
    hundreds of thousands of times. Reading and reducing each asset once and
    reusing the result per instance turns the expensive part from
    per-instance to per-asset — on the test level, a sample of a couple of
    thousand actors used only a couple of hundred distinct meshes.

  - **Give each asset a triangle allowance from its size.** A 40m warehouse
    earns more than a 3m crate, because at the range this viewer is used from
    that is exactly how much of each you can see.

  - **Accumulate in tiles.** Simplifying each tile as it completes keeps peak
    memory flat instead of holding every triangle in the map at once, and it
    spends the triangle budget where the geometry actually is.

Everything is read through the *source model* rather than a render LOD. Render
LODs are a rendering resource and a commandlet with `-nullrhi` has none — reading
one asserts inside the engine. The source model is always there, so the
reduction happens here.
"""
import math
import os
import time

import unreal

# See MeshCache.get. Empty disables tracing. Comma-separated, so one run can
# trace something known-missing alongside something known-present — a trace that
# prints nothing looks the same whether the object never arrived or the
# instrument is broken.
_TRACE = [s for s in os.environ.get("HEAT3D_TRACE", "").split(",") if s]


# Nothing on a game map is a thousand kilometres across or a thousand kilometres
# from its own origin. Anything beyond this is not large, it is broken.
SANE_LIMIT_CM = 1.0e8


def sane_number(value):
    return math.isfinite(value) and abs(value) < SANE_LIMIT_CM


def sane_mesh(mesh):
    """Are this mesh's vertices somewhere a world could be?

    Asked of the bounding box, so it is one call whatever the triangle count.
    """
    try:
        box = mesh.get_bounding_box()
    except Exception:  # noqa: BLE001 - engine version differences
        return True
    return all(
        sane_number(v)
        for v in (box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z)
    )


def sane_transform(transform):
    """Is this placement somewhere a world could be?

    Worth its own function because an instanced component with one uninitialised
    placement is enough to ruin a whole export, silently and after twenty
    minutes: the vertices land at 1e34, the level's bounds become astronomical,
    and the viewer opens on an empty screen with the entire map a pixel across
    somewhere inside it. Every number the report prints looks correct.

    Scale is checked too. A zero scale collapses a mesh, which is ugly but
    harmless; a scale of 1e18 is the same catastrophe as a bad translation.
    """
    try:
        t = transform.translation
        s = transform.scale3d
    except Exception:  # noqa: BLE001 - engine version differences
        return True
    return all(
        sane_number(v) for v in (t.x, t.y, t.z, s.x, s.y, s.z)
    )


# A mesh with no more triangles than this, spanning more than MARKER_MIN_CM in
# every direction, is a marker volume rather than geometry. Twelve triangles is a
# closed box; sixteen leaves room for one that was authored slightly differently.
MARKER_TRIANGLES = 16
MARKER_MIN_CM = 5000.0


def _all_dimensions_over(mesh, limit):
    """Is this mesh larger than `limit` on all three axes?

    All three, not the diagonal: a road or a floor slab is enormous in two
    dimensions and thin in the third, and those are geometry players stand on.
    What this is looking for is the shell of a region — as wide as it is deep as
    it is tall.
    """
    try:
        box = mesh.get_bounding_box()
    except Exception:  # noqa: BLE001 - engine version differences
        return False
    return (
        (box.max.x - box.min.x) > limit
        and (box.max.y - box.min.y) > limit
        and (box.max.z - box.min.z) > limit
    )


def weld_options(tolerance=1.0):
    """Options for merging coincident vertices.

    One centimetre. Modular architecture is authored so its panels meet, and
    within a centimetre they do; welding there turns a pile of separate shells
    into a surface a simplifier can actually reduce. Wider than that and it
    starts welding a door to its frame.
    """
    options = unreal.GeometryScriptWeldEdgesOptions()
    try:
        options.tolerance = tolerance
    except Exception:  # noqa: BLE001 - engine version differences
        pass
    return options


def planar_options():
    """Options for the coplanar merge.

    Free detail removal: adjacent triangles lying in the same plane become one,
    which changes nothing about the shape. Architecture is mostly flat panels, so
    a wall built from two hundred triangles becomes two and the building is
    *identical*. This runs before anything that costs shape.
    """
    # `GeometryScriptPlanarSimplifyOptions` — no "Mesh" in the middle, unlike
    # every other options struct in this library.
    return unreal.GeometryScriptPlanarSimplifyOptions()


def simplify_options():
    """Options for a geometry-only reduction.

    The default method is `ATTRIBUTE_AWARE`, which refuses to collapse an edge
    where that would change what a triangle's material or attributes say. Within
    one asset that costs a little detail. Across an accumulated tile it is fatal:
    every instance arrives with its own material regions, so every shell boundary
    is untouchable and the simplifier returns the mesh unchanged — millions of
    triangles asked to shrink several times over and staying as they were,
    silently.

    `STANDARD_QEM` is plain quadric error minimisation over positions, which is
    all this export is about. Attributes are discarded on the way in for the same
    reason: the viewer reads positions and triangles and derives its own normals.
    """
    options = unreal.GeometryScriptSimplifyMeshOptions()
    try:
        options.set_editor_property(
            "method", unreal.GeometryScriptRemoveMeshSimplificationType.STANDARD_QEM
        )
    except Exception:  # noqa: BLE001 - enum name differs across engine versions
        pass
    return options


def strip_attributes(mesh):
    """Drop normals, UVs, material ids and polygroups.

    None of them reach the viewer, they make every append carry more data, and
    they are what stops the simplifier from working. Failure is not fatal — an
    older engine may not expose it — so this degrades to a no-op.
    """
    try:
        return mesh.discard_mesh_attributes()
    except Exception:  # noqa: BLE001 - engine version differences
        return mesh


class MeshCache(object):
    """Simplified geometry per (asset, allowance), built on demand."""

    def __init__(self, pool, log, use_boxes=False):
        self.pool = pool
        self.log = log
        self.use_boxes = use_boxes
        self.entries = {}
        self.read_seconds = 0.0
        self.simplify_seconds = 0.0
        self.source_triangles = 0
        self.planar_triangles = 0
        self.kept_triangles = 0
        self.count_limited = 0
        self.boxed = 0
        self.degenerate = 0
        # Which assets became boxes, so "what are all these cubes?" is answerable
        # from the report rather than by guessing at a screenshot. The first guess
        # was "trees", and it was wrong.
        self.boxed_by_asset = {}
        self.dropped_small = 0
        # What was left out for being too small to hold a shape. Named, because
        # "why is my fence missing" has to be answerable from the report.
        self.dropped_by_asset = {}
        self.box_fit_cache = {}
        self.boxes = {}
        self.failures = 0
        # Assets whose vertices came back as astronomical numbers; see `_sane`.
        self.insane = 0
        self.insane_examples = []
        # Assets rejected as marker volumes; see MARKER_TRIANGLES.
        self.markers = 0
        self.marker_examples = []

        self.options = unreal.GeometryScriptCopyMeshFromAssetOptions()
        self.lod = unreal.GeometryScriptMeshReadLOD()
        self.lod.lod_type = unreal.GeometryScriptLODType.SOURCE_MODEL
        self.lod.lod_index = 0
        self.simplify_options = simplify_options()

    def get(
        self,
        static_mesh,
        allowance,
        size_cm,
        strip_small=False,
        allow_box=True,
        tolerance_cap=None,
        count_limit=True,
    ):
        """A DynamicMesh for this asset, reduced without wrecking its shape.

        `tolerance_cap` and `count_limit` exist because the three stages below
        are not equally safe, and the parts a building is made of need the middle
        one on a tighter leash and the last one off entirely.

        Stage 3 — reduce to a triangle count — is the one that produces spikes.
        On a part that is itself an assembly of disconnected shells (a modular
        kit piece, a railing run, a window set) it bridges between them exactly
        the way it does across a whole building: it keeps a few far-apart
        vertices and joins them across the gap. `count_limit=False` turns it off.

        Stage 2 is bounded by its tolerance and cannot make a long spike, but it
        can still flatten a thing that is thinner than the tolerance — merging a
        panel's two faces into one sheet, which reads on screen as a big flat
        quad floating at an odd angle. The 15cm default is under a *wall*, and a
        railing, a window or a sign is thinner than that, so architecture passes a
        tighter cap.

        Measured on the test level: hundreds of assets reached stage 3 and around
        a thousand came out of stage 2 as sheets, and those were the shards left
        inside the buildings after the assembly-level reduction was removed.
        Turning stage 2 off as well is not the answer — with only the coplanar
        merge, parts cost several hundred triangles each and a building could
        afford a dozen of them.

        Three stages, cheapest and safest first:

        1. **Coplanar merge.** Free: a two-hundred-triangle wall becomes two and
           the building is identical. Architecture is mostly flat panels, so this
           alone does most of the work on the things that matter.
        2. **Reduce to a geometric tolerance**, derived from the object's own
           size. A tolerance says "stay within 15cm of the original"; a triangle
           count says "be 40 triangles, whatever that does to you". The first
           keeps a house a house. The second is what turned this map into a
           pixelated mess — asking a 40m building and a 4m crate each to become
           the same handful of triangles flattens the building into a wedge.
        3. **Only if it is still enormous**, fall back to a triangle count. By
           then the shape has already been through the two passes that respect
           it, so the fallback starts from something sane.

        Allowances are quantised so a hundred slightly different instance sizes
        share one cached reduction instead of each paying for its own.

        `strip_small` deletes disconnected scraps before reducing. It is for
        foliage, where the scraps are leaf cards, and it must stay off for
        everything else: a modular building *is* a pile of disconnected panels.

        `allow_box` is what happens when the allowance is too small to be worth
        spending on shape. For a wall, a shed or a container the answer is a box,
        which is both cheaper and more accurate than a reduction. For foliage it
        is not — a boxed tree is a crate — so foliage asks for the floor instead
        and is limited by the foliage cap rather than by substitution.
        """
        requested = _quantise(allowance)
        # Keyed apart from the reducing variants, or a part read exactly for a
        # building would be served from the cache to the structure pass — which
        # asked for a reduction and would silently get none.
        count_limit = bool(count_limit)
        tolerance_cap = (
            TOLERANCE_CEILING_CM if tolerance_cap is None else float(tolerance_cap)
        )
        # A small allowance means "spend little on this", never "delete it".
        #
        # This is the line that was hiding every building on the map, and it hid
        # them by way of a change made for a good reason. Box substitution was
        # turned off because a field of cyan cubes between the buildings was worse
        # than nothing — but the code below only ever had two answers for an
        # allowance under the floor, a box or a deletion, so switching boxes off
        # silently turned it into a deleter. Worse, it was *selective*: foliage
        # passes `allow_box=False` and was floored up to a usable 96 triangles,
        # while a wall panel passed `allow_box=True`, kept its 40-triangle
        # allowance, and was dropped. Measured on the test level: only a couple of
        # thousand unique assets reached the cache out of tens of thousands on the
        # map, vegetation came out ahead of architecture in every export, and
        # millions of triangles of building stood at the map's busiest spot with
        # barely a hundred of them in the file.
        #
        # So the floor now applies whenever a box is not going to be drawn. The
        # budget is not lost by this: the accumulator is what enforces it, and a
        # tile refusing its tail is a bound applied evenly and in priority order,
        # which a per-asset deletion is not.
        if not allow_box or not self.use_boxes:
            requested = max(MIN_OBJECT_TRIANGLES, requested)
        path = static_mesh.get_path_name()
        # `HEAT3D_TRACE=<substring>` narrates every decision taken about matching
        # assets.
        #
        # Worth its permanent place: "this asset is missing from the export" is
        # the hardest question this exporter gets asked, and answering it once
        # took comparing telemetry against geometry cell by cell, four probes into
        # the editor, and ten full exports. Every stage here can silently drop a
        # mesh — the allowance floor, the tolerance pass, the count ceiling, the
        # degeneracy check — and the report only shows totals.
        trace = any(want in path for want in _TRACE)
        if trace:
            self.log(
                "  trace {}: allowance {} -> requested {}, size {:.0f}cm".format(
                    path.rsplit("/", 1)[-1], allowance, requested, size_cm
                )
            )
        key = (path, requested, strip_small, tolerance_cap, count_limit)
        # `in`, not `.get() is not None`: an asset that resolved to nothing —
        # unreadable, empty, or foliage that reduced to splinters — has to stay
        # resolved, or every one of its instances pays to find out again.
        if key in self.entries:
            return self.entries[key]

        # Only reachable with `--boxes`: the floor above keeps everything else out
        # of here. A box is twelve triangles, it stands in the right place, and it
        # beats the spikes a quadric simplifier produces at that budget — but on a
        # dense map it produced a field of cyan cubes between the buildings that
        # hid the buildings, the ground and the heat, so it is off by default and
        # kept reachable for the sparser maps where the argument holds.
        if requested < MIN_OBJECT_TRIANGLES:
            if size_cm < BOX_MIN_SIZE and not self.box_fits(static_mesh):
                self.dropped_small += 1
                self.entries[key] = None
                return None
            box = self.box_for(static_mesh)
            if box is not None:
                self.boxed += 1
                self.boxed_by_asset[path] = self.boxed_by_asset.get(path, 0) + 1
                self.kept_triangles += box.get_triangle_count()
                self.entries[key] = box
                return box

        t0 = time.time()
        mesh = self.pool.request_mesh()
        try:
            mesh, _outcome = unreal.GeometryScript_AssetUtils.copy_mesh_from_static_mesh(
                static_mesh, mesh, self.options, self.lod
            )
        except Exception as err:  # noqa: BLE001 - one bad asset must not stop the export
            self.failures += 1
            self.log("  ! could not read {}: {}".format(path, err))
            self.entries[key] = None
            return None
        self.read_seconds += time.time() - t0

        count = mesh.get_triangle_count()
        if trace:
            self.log("  trace: read {} source triangles".format(count))
        if count == 0:
            self.entries[key] = None
            return None
        # Is what came back geometry, or numbers?
        #
        # A commandlet with `-nullrhi` reads the source model rather than render
        # data, and a handful of assets on a real map come back with vertices at
        # 1e34 — cloth, spline-deformed meshes, anything whose source geometry is
        # a placeholder the runtime replaces. One of them is enough to make a
        # several-hundred-MB export unopenable: the level's bounds become
        # astronomical and the whole map is a pixel inside them, so the viewer
        # shows an empty screen.
        #
        # Cheap to check and worth checking on every asset rather than on the ones
        # currently known to misbehave. This was found *after* an export looked
        # fine by every number the report prints.
        if not sane_mesh(mesh):
            self.insane += 1
            if len(self.insane_examples) < 25:
                self.insane_examples.append(path.rsplit("/", 1)[-1])
            if trace:
                self.log("  trace: rejected, vertices are not in any world")
            self.entries[key] = None
            return None
        # A district-sized box with twelve triangles in it is not a building.
        #
        # The name patterns above catch the two this map has, and names are a
        # convention rather than a fact — the next project will call them
        # something else. This is the same judgement made from the geometry: a
        # marker volume, a trigger shell or a blocking box is a handful of
        # triangles spanning a hundred metres in *every* direction, and nothing a
        # player walks on or shelters behind is. A floor slab or a road is thin,
        # so it fails the height test and is kept.
        if count <= MARKER_TRIANGLES and _all_dimensions_over(mesh, MARKER_MIN_CM):
            self.markers += 1
            if len(self.marker_examples) < 25:
                self.marker_examples.append(path.rsplit("/", 1)[-1])
            if trace:
                self.log("  trace: rejected as a marker volume")
            self.entries[key] = None
            return None
        self.source_triangles += count
        mesh = strip_attributes(mesh)

        t0 = time.time()
        if strip_small:
            try:
                # Drop the twigs and leaf cards before reducing: thousands of tiny
                # disconnected shells, none of which survives a reduction, each one
                # a shard in the result.
                #
                # By triangle count and area only — *never* by volume. A wall panel
                # is a flat shell with no volume at all, so a volume threshold
                # deletes half a modular building and leaves the other half hanging
                # in the air. Which is exactly what the first attempt did.
                small = unreal.GeometryScriptRemoveSmallComponentOptions()
                for name, value in (
                    ("min_triangle_count", 6),
                    ("min_area", max(1.0, size_cm * 0.005) ** 2),
                ):
                    try:
                        small.set_editor_property(name, value)
                    except Exception:  # noqa: BLE001 - field names vary by version
                        pass
                mesh = mesh.remove_small_components(small)
            except Exception:  # noqa: BLE001 - not present on every engine version
                pass
        try:
            mesh = mesh.apply_simplify_to_planar(planar_options())
            self.planar_triangles += mesh.get_triangle_count()
        except Exception:  # noqa: BLE001 - not present on every engine version
            self.planar_triangles += mesh.get_triangle_count()

        if mesh.get_triangle_count() > requested:
            # A fraction of the object's own size: the same 2% is a 20cm error on
            # a 10m shed and a 1.2m error on a 60m hangar, which is how much
            # detail each can lose before it stops being recognisable.
            #
            # Capped in absolute terms as well, because "how much detail" is not
            # the only thing a tolerance controls. A tolerance wider than a wall
            # is thick lets the simplifier merge the wall's inside face with its
            # outside one, and a building whose walls have all become single
            # surfaces collapses into a few floating quads. Walls are thinner
            # than this cap in every game ever made.
            tolerance = min(tolerance_cap, max(1.0, size_cm * TOLERANCE_FRACTION))
            try:
                mesh = mesh.apply_simplify_to_tolerance(tolerance, self.simplify_options)
            except Exception:  # noqa: BLE001 - fall through to the count-based path
                pass

        over = mesh.get_triangle_count()
        if count_limit and over > requested * HARD_CEILING:
            mesh = mesh.apply_simplify_to_triangle_count(
                int(requested * HARD_CEILING), self.simplify_options
            )
            self.count_limited += 1
            # Judged on the *result*, not on how much it lost.
            #
            # There used to be a second clause here — `over > requested * 40`,
            # meaning "lost more than 97.5% of itself, so it is debris". That is a
            # ratio, and a ratio punishes an object for being large. The test
            # level's centrepiece is a single landmark mesh kilometres across;
            # reduced to the tens of thousands of triangles it can afford it is a
            # perfectly good landmark, and the ratio test threw it away as debris.
            # The building the players fight over was missing from every export,
            # and the heat was left ringing a building that was not there.
            #
            # What actually matters is whether what came out is still a shape,
            # which is what the check below asks.
            if mesh.get_triangle_count() > requested * HARD_CEILING * 1.05:
                box = self.box_for(static_mesh) if self.use_boxes else None
                if box is not None:
                    self.boxed += 1
                    self.simplify_seconds += time.time() - t0
                    self.kept_triangles += box.get_triangle_count()
                    self.entries[key] = box
                    return box
        # A shape that came out of all that as a handful of triangles is not a
        # simplified object, it is a sheet: the tolerance was wider than the
        # thing was thick, so the two sides of it merged. On screen those read as
        # big flat panels floating at odd angles. Only meshes that actually had
        # detail to lose are judged this way — a mesh that arrived with twenty
        # triangles is allowed to stay at twenty.
        #
        # Dropped, not boxed, unless boxes are asked for. This path was the last
        # source of cubes after the allowance path stopped making them, and a
        # placeholder nobody wants is not improved by being rarer.
        final = mesh.get_triangle_count()
        if trace:
            self.log("  trace: after reduction {} triangles".format(final))
        if final < DEGENERATE_TRIANGLES and count >= DEGENERATE_SOURCE_TRIANGLES:
            self.degenerate += 1
            self.simplify_seconds += time.time() - t0
            box = self.box_for(static_mesh) if (allow_box and self.use_boxes) else None
            if box is not None:
                self.boxed += 1
                self.kept_triangles += box.get_triangle_count()
            self.entries[key] = box
            return box

        self.simplify_seconds += time.time() - t0
        self.kept_triangles += final

        self.entries[key] = mesh
        return mesh

    def box_fits(self, static_mesh):
        """Is this asset's bounding box a fair description of its shape?

        Asked so that the size rule above does not throw away the parts buildings
        are made of. A modular building is a stack of wall, floor and roof panels,
        each maybe eight metres across — under `BOX_MIN_SIZE`, and dropping them
        removes the building, which is the exact opposite of what the rule is for.
        But a wall *is* a box: its bounding box is the wall to within its own
        thickness, so substituting one loses nothing at all.

        A rock of the same size is not. It is lumpy and roughly equidimensional,
        its bounding box is half air, and at prop scale a cube standing where it
        stood is the clutter that hid the map.

        So: flat things are boxed at any size, blocky things have to earn it by
        being big. Decided from the bounding box alone, which costs nothing —
        no mesh is read.
        """
        path = static_mesh.get_path_name()
        hit = self.box_fit_cache.get(path)
        if hit is not None:
            return hit
        fits = False
        try:
            box = static_mesh.get_bounding_box()
            extent = sorted(
                (
                    abs(box.max.x - box.min.x),
                    abs(box.max.y - box.min.y),
                    abs(box.max.z - box.min.z),
                )
            )
            if extent[2] > 0:
                fits = (extent[0] / extent[2]) <= SLAB_ASPECT
        except Exception:  # noqa: BLE001 - engine version differences
            fits = False
        self.box_fit_cache[path] = fits
        return fits

    def box_for(self, static_mesh):
        """The asset's bounding box as a twelve-triangle mesh, cached per asset."""
        path = static_mesh.get_path_name()
        hit = self.boxes.get(path)
        if hit is not None:
            return hit
        try:
            bounds = static_mesh.get_bounding_box()
            size = bounds.max - bounds.min
            centre = (bounds.max + bounds.min) * 0.5
            transform = unreal.Transform()
            # `append_box` builds from the box's *base* by default, not its
            # centre — so the translation is the footprint centre at the bottom.
            transform.translation = unreal.Vector(centre.x, centre.y, bounds.min.z)
            mesh = self.pool.request_mesh()
            mesh = mesh.append_box(
                unreal.GeometryScriptPrimitiveOptions(),
                transform,
                max(1.0, size.x),
                max(1.0, size.y),
                max(1.0, size.z),
            )
            mesh = strip_attributes(mesh)
        except Exception as err:  # noqa: BLE001 - engine version differences
            self.log("  ! could not box {}: {}".format(path, err))
            self.boxes[path] = None
            return None
        self.boxes[path] = mesh
        return mesh


def _quantise(n):
    """Round an allowance up to the next power of two, floor 8.

    Without this every distinct instance scale produces its own reduction of the
    same asset, and the cache stops being a cache.
    """
    n = max(8, int(n))
    p = 8
    while p < n:
        p *= 2
    return p


def allowance_for(size_cm, settings):
    """Triangles worth spending on an object of this size.

    Linear in size, which is a deliberate under-weighting of big objects: area
    would give a warehouse a hundred times a crate's budget and spend the whole
    file on three buildings.
    """
    metres = size_cm / 100.0
    return max(8, min(4000, int(metres * 8)))


# The per-shell floor. A quadric simplifier collapses edges; it cannot merge two
# objects that do not touch, so a mesh made of N separate shells cannot go below
# roughly N times this however small a target you ask for. Measured on real
# tiles: a tile of over a hundred thousand triangles asked for 64 stopped at about
# 23 per instance, which is exactly this floor.
#
# It is the reason the triangle budget has to be spent by *choosing objects*
# rather than by shrinking all of them.
SHELL_FLOOR = 24

# What an object needs to still look like itself — and, below it, the point at
# which the bounding box is the better answer.
#
# The floor above is what a simplifier will *tolerate*; this is what a viewer
# needs. Below about a hundred triangles a house loses its roof line and a tree
# loses its canopy. A quadric simplifier given twenty triangles for a bush does
# not produce a small bush either; it produces a handful of long thin spikes,
# because the error metric is happy to keep a few far-apart vertices and
# collapse everything between them.
#
# So an object whose share comes to less than this is not reduced at all: it is
# replaced by its bounding box, which is twelve triangles, always reads as a
# solid object of about the right size, and is a more honest statement of what
# is known — something this big stands here. Subject to BOX_MIN_SIZE below.
MIN_OBJECT_TRIANGLES = 96

# ...but only for objects at least this big, in centimetres. Smaller ones are
# skipped instead.
#
# A box is an honest answer at building scale and clutter at prop scale, and the
# difference is not subtle from inside the viewer: a field of head-height cyan
# cubes scattered between the buildings hides the buildings, hides the ground,
# and hides the heat, while telling you nothing you wanted to know. Twelve metres
# is about the smallest thing whose footprint is worth knowing on a large map —
# a shed, a container, a bus. Below it, nothing is better than a cube.
#
# This is a deliberate reversal. Box substitution was introduced to stop a
# quadric simplifier turning small objects into spikes, and it did; the mistake
# was applying it at every scale, which traded a field of spikes for a field of
# cubes. The right answer at small scale is to leave the object out.
BOX_MIN_SIZE = 1_200.0

# Thinnest extent over longest, below which an asset counts as a slab and its
# bounding box is a faithful stand-in whatever its size. A wall panel is about
# 0.05; a boulder is about 0.8. See ox_fits.
SLAB_ASPECT = 0.25

# What an object costs *on average*, used only to decide how many of them the
# budget can hold.
#
# Not the same number as the threshold above, and using that one instead was a
# mistake worth naming: an allowance is what an asset is *offered*, and almost
# nothing takes all of it. Boxes cost twelve, and coplanar merging, the
# multi-instance damping and the per-shell floor bring the rest down — measured
# over the test level, tens of thousands of instances averaged about half the 96
# floor. Planning at 96 therefore held back half the objects the file had room
# for, and since the ranking is by size, the half it held back was every building
# smaller than about twenty metres: their roof slabs are large enough to survive
# on their own, so the export came out as floor plans hanging in mid-air over an
# empty map.
#
# Planning at the measured cost offers several times as many. The per-tile cap
# still enforces the real budget and drops from the bottom of the same ranking,
# so over-offering is safe in a way that under-offering is not.
PLANNED_OBJECT_TRIANGLES = 24

# How far the planned allowances over-subscribe the budget.
#
# The plan divides the budget across every actor a descriptor scan kept, and a
# large fraction of those never spend it: they turn out to be rock and are routed
# to the ground part, or they are foliage and hit the vegetation cap, or they are
# too small to be worth a box and are dropped. Measured on the test level, that
# left the structures spending about a third of their allowance — millions of
# triangles of headroom, while the buildings that were present had under a
# hundred triangles each and the ones that were not had been refused.
#
# Offering each object three times its strict share converts that headroom into
# detail. It is safe for the same reason the object *count* is over-offered: the
# per-tile ceiling is what enforces the budget, it is recomputed from what
# remains after each tile, and it refuses from the bottom of the size ranking.
PLAN_OVERSUBSCRIBE = 3.0

# Most triangles any one asset may be offered, before over-subscription.
#
# 4,000 was set when the largest thing on the map was assumed to be a warehouse.
# A landmark is a different scale: it can be a single mesh kilometres across, and
# at 4,000 (12,000 after over-subscription) it was reduced so far past
# recognition that the debris check threw it out. The per-actor tile ceiling still stops one
# object eating a district, so the cap only needs to be loose enough not to
# strangle the biggest thing a map has.
MAX_OBJECT_TRIANGLES = 20_000

# What a tree's allowance is multiplied by, having been asked to yield to the
# things that constrain movement. See `classify.VEGETATION_NAMES`.
#
# Not zero, and not the foliage cap doing the same job twice: the cap bounds
# vegetation placed through the foliage system, and this catches the trees placed
# as ordinary static meshes, which on the test level were several of the ten
# hungriest assets on the map. A tree at 40% still reads as a tree; a fence at
# nothing reads as an unexplained gap in the data.
VEGETATION_ALLOWANCE_WEIGHT = 0.4

# ...and what a building's is multiplied by, for the same reason from the other
# side.
#
# `allowance_for` is deliberately mean — eight triangles per metre — because it
# has to be safe for the ten thousand props on a map. A building is not a prop:
# at eight per metre a 40m warehouse gets 320 triangles, which draws its
# footprint and loses the thing that was actually asked for, the halls and
# walkways inside it. At 24 it gets 960 and the interior reads.
ARCH_ALLOWANCE_WEIGHT = 3.0

# A reduction that ends below this many triangles is treated as having failed,
# and the bounding box is used instead.
#
# The failure it catches: a geometric tolerance of, say, 40cm applied to a wall
# 20cm thick merges the wall's two faces into one, and repeated over a building
# that leaves a few large quads floating where the building was. It is the same
# problem MIN_OBJECT_TRIANGLES solves from the other end — there the allowance
# was too small to be worth trying, here the attempt was made and came out
# wrong.
#
# Only applied to meshes that arrived with real detail (below), so a fence plank
# that is genuinely eight triangles is left alone.
DEGENERATE_TRIANGLES = 20
DEGENERATE_SOURCE_TRIANGLES = 64

# Most of the structure budget any one actor may take inside a tile.
#
# The reason this exists: an InstancedFoliageActor's bounds span everything it
# places, so a forest is one "actor" the size of a district. Ranked by size it
# sorts to the very front, and — processing largest-first — it consumed entire
# tiles before a single building was reached. The export came out as trees.
#
# Raised from 6% once the foliage cap was doing that job properly. At 6% a tile
# containing one large building complex could not spend its own share on it: the
# complex is a single actor, so it hit this ceiling and thousands of instances
# were refused across the map while most of the structure budget sat unspent. A
# building that a heatmap has data inside of was among them.
PER_ACTOR_TILE_SHARE = 0.15

# ...and what a *building* may take, which is a different question.
#
# The cap above exists to stop one actor with thousands of placements eating a
# tile. A building complex is the opposite case: one actor, few placements, and
# the single most valuable thing in the tile. Held to 15% it cannot be built —
# and being large it is also the actor most likely to need more than that.
PER_ACTOR_TILE_SHARE_PRIORITY = 0.6

# How much of each tile's structure share only architecture may spend.
#
# This is the fix for the failure that survived four rounds of tuning every
# other number in this file. Components inside an actor were already offered
# architecture-first, but *actors* within a tile were offered largest-first, and
# the largest things on a map are rocks, foliage clusters and decorative
# hillsides. With `PER_ACTOR_TILE_SHARE` at 15%, seven such actors exhaust a
# tile — so by the time a 12m house was reached there was nothing left, in every
# tile, all the way across the map. Measured on the test level: architecture got
# less of the budget than vegetation did, and unclassified scenery got more than
# the two together. The report called all of it "kept".
#
# Reserving the band is what makes the priority list mean anything. It is the
# same mechanism as the vegetation cap, pointed the other way: scenery may spend
# up to `1 - ARCH_TILE_RESERVE` of a tile, and what is left is there when the
# walls, doors and stairs arrive however late in the tile they come. Anything
# architecture does not use is not wasted — `begin_tile` recomputes the next
# tile's share from the budget actually remaining.
ARCH_TILE_RESERVE = 0.5

# How far over its fair share a single tile may go.
#
# Tile shares are proportional to what the descriptor plan expected each tile to
# hold, and that expectation is blind to two things it cannot know until the
# actors are loaded: how much of a tile is rock that will be routed to the ground
# part instead, and how much of it is one packed actor. So a dense tile refuses
# geometry while the map-wide budget goes unspent.
#
# Letting a tile take up to twice its share fixes the common case without
# unbounding anything: `begin_tile` recomputes from the budget *remaining*, so a
# tile that overspends leaves less for the rest and the total still holds.
TILE_SHARE_SLACK = 2.0

# And a ceiling on vegetation overall, for the same reason at map scale.
#
# Cut from a quarter to a twelfth after measuring where a real export's budget
# went: foliage took about as many triangles as every building on the map put
# together. For a movement heatmap that is exactly inverted. Trees are scenery;
# the buildings are what a player was standing in when the datapoint was written,
# and a tree drawn in place of one is not a trade worth making.
FOLIAGE_BUDGET_SHARE = 0.08

# Share of the structure budget for landscape built out of meshes.
#
# Roads, paths and rock formations are not buildings, and on the test level they
# are a sizeable share of everything that used to be called a structure — mostly
# rock, the rest road. They are given a little under that, because a road needs
# less than a building does to be recognisable and a cliff needs less than either.
TERRAIN_MESH_SHARE = 0.35

# Share of the structure budget for the buildings that live in sub-levels.
#
# Nearly half, and it is the least arguable number in this file: on a World
# Partition map composed of Level Instances, this is *the* built environment —
# every house, shop, factory, school and office — and until the pass that
# reads it existed, the export contained none of it. See heat3d/sublevels.py.
#
# It buys more than the share suggests, too. The rest of the budget is spent per
# placement; this is spent per *building*, reduced once and stamped at every
# instance, so a house that occurs fifty times costs one house.
BUILDING_BUDGET_SHARE = 0.7

# How much of its bounds a scattered actor is ranked by. See plan().
SCATTERED_RANK_WEIGHT = 0.08

# How far an object may move from its original shape, as a fraction of its own
# size. 2% is 20cm on a 10m shed: enough to drop window mullions and roof tiles,
# not enough to round off a corner.
TOLERANCE_FRACTION = 0.02

# And never more than this, whatever the object's size. A tolerance has to stay
# under the thickness of a wall or the simplifier is free to merge a wall's two
# faces, which turns a building into a handful of floating panels. 15cm is under
# the thinnest wall in normal use and still large enough to erase trim, railings
# and window frames on a large building.
TOLERANCE_CEILING_CM = 15.0

# Tolerance-based reduction is shape-driven, so it can overshoot an allowance —
# a tree canopy has no flat panels and no cheap detail to give up. This is how
# far over its allowance an object may go before a triangle count is imposed
# after all.
HARD_CEILING = 3.0


def rank_weight(target):
    """How large an actor counts as, for every ordering in the exporter.

    Scattered actors — foliage, mostly — are ranked by a fraction of their
    bounds, because their bounds describe the *area they cover* rather than
    anything standing in it. Without this they occupy the whole front of the
    ranking, push the size threshold up past every building on the map, and the
    export comes out as trees.
    """
    return max(1.0, target.size) * (SCATTERED_RANK_WEIGHT if target.scattered else 1.0)


def plan(targets, budget, settings):
    """How many objects the budget can hold, and what each one gets.

    Two decisions, and they have to be made in this order:

    1. **How many objects.** The budget divided by what an object actually costs
       — `PLANNED_OBJECT_TRIANGLES`, measured, not the allowance floor. 150,000
       objects in 1.6M triangles is about eleven triangles each, less than a box,
       so the question is never whether to drop some but which.
    2. **How much each.** Proportional to size, normalised so the whole budget is
       spent. Spending it on the largest objects *by allowance* instead — the
       first attempt — kept the few hundred biggest things on the map and dropped
       every building, because the largest few absorbed 4,000 triangles apiece.

    The offer overshoots the budget, deliberately: the per-tile cap is what
    enforces it, and that cap drops from the bottom of this same ranking.

    Returns (kept, allowances, cut_size, planned).
    """
    weight = rank_weight
    ranked = sorted(targets, key=lambda t: -weight(t))
    affordable = max(1, int(budget // PLANNED_OBJECT_TRIANGLES))
    kept = ranked[:affordable]
    if not kept:
        return [], {}, 0.0, 0

    total_weight = float(sum(weight(t) for t in kept))
    allowances = {}
    planned = 0
    for target in kept:
        share = budget * (weight(target) / total_weight)
        # No floor here. A share below MIN_OBJECT_TRIANGLES is a real answer —
        # it means "this one gets a box" — and flooring it was what made the box
        # substitution unreachable: the cache floored the allowance again before
        # testing it against the threshold, so the test could never be true and
        # every object, however small its share, was given a full reduction it
        # had not been budgeted for.
        allowance = max(1, min(MAX_OBJECT_TRIANGLES, int(share * PLAN_OVERSUBSCRIBE)))
        allowances[id(target)] = allowance
        planned += allowance
    return kept, allowances, weight(kept[-1]), planned


class Accumulator(object):
    """One class of geometry (ground or structures), built tile by tile."""

    def __init__(self, name, pool, budget, log):
        self.name = name
        self.pool = pool
        self.budget = budget
        self.log = log
        self.total = pool.request_mesh()
        self.tile = pool.request_mesh()
        self.tile_instances = 0
        self.instances = 0
        self.tiles_flushed = 0
        self.append_seconds = 0.0
        self.simplify_seconds = 0.0
        self.raw_triangles = 0
        # Filled in by the caller before building, from the descriptor scan: the
        # total triangle allowance across every actor, and per tile. Shares are
        # allocated against these rather than against what a tile actually
        # produced, so they sum to the budget instead of to whatever the map
        # happened to contain.
        self.expected = 1
        self.expected_done = 0
        self.tile_share = budget
        self.skipped_instances = 0
        # Single objects admitted despite exceeding the per-actor share; see append.
        self.oversized_singles = 0
        # Tiles dropped for holding impossible coordinates; see flush_tile.
        self.insane_tiles = 0

    def begin_tile(self, tile_expected):
        """Set this tile's ceiling before anything is appended to it.

        The ceiling has to exist *before* the geometry arrives, not after. One
        actor can be a packed level containing a whole district — a few hundred
        actors on the test level turned into tens of thousands of instances and
        millions of triangles — so a plan made from descriptors is a guess, and
        reducing afterwards cannot help: a simplifier will not merge separate
        objects, so a tile of 25,000 shells has a floor of 600k triangles
        whatever it is asked for.
        """
        spent = self.total.get_triangle_count()
        remaining_budget = max(0, self.budget - spent)
        remaining_expected = max(1, self.expected - self.expected_done)
        weight = float(tile_expected) / remaining_expected if tile_expected else 1.0
        self.tile_share = max(
            SHELL_FLOOR, int(remaining_budget * min(1.0, weight * TILE_SHARE_SLACK))
        )
        self.expected_done += tile_expected

    def append(self, mesh, transforms, limit=None):
        """Append instances until this tile's ceiling is reached.

        Refusing here is what makes the budget a bound rather than a hope. What
        gets refused is the tail of a tile processed largest-object-first, so the
        things that go missing are the small ones.

        `limit` is an extra ceiling for this call alone — how much this one actor,
        or this one category, may still take. Without it a single
        InstancedFoliageActor, whose bounds span everything it plants and which
        therefore sorts first, took whole tiles before any building was reached.
        """
        if mesh is None or not transforms:
            return
        triangles = mesh.get_triangle_count()
        if triangles <= 0:
            return
        tile_room = self.tile_share - self.tile.get_triangle_count()
        room = tile_room if limit is None else min(tile_room, limit)
        if room <= 0:
            self.skipped_instances += len(transforms)
            return
        allowed = max(0, int(room // triangles))
        # One object is always worth one object, even if it is bigger than the
        # share any single actor is meant to take.
        #
        # `allowed = room // triangles` is integer division, so a mesh larger than
        # the per-actor room comes out as zero and is refused entirely — and the
        # bigger and more important the object, the more certainly that happens.
        # It is why the level's centrepiece, a landmark mesh, was missing from
        # every export: one mesh kilometres across, reduced to several times the
        # per-actor room and still over it. Refused, silently, every time, while
        # the report showed it as kept.
        #
        # The per-actor limit exists to stop one actor with *many* placements
        # taking a whole tile, which is a different thing. A lone instance is
        # bounded by the tile's own share instead, so the budget still holds.
        if allowed == 0 and len(transforms) == 1 and triangles <= tile_room:
            allowed = 1
            self.oversized_singles += 1
        if allowed < len(transforms):
            self.skipped_instances += len(transforms) - allowed
            transforms = transforms[:allowed]
            if not transforms:
                return

        t0 = time.time()
        before = self.tile.get_triangle_count()
        self.tile = self.tile.append_mesh_transformed(
            mesh, transforms, unreal.Transform(), False
        )
        self.append_seconds += time.time() - t0
        self.instances += len(transforms)
        self.tile_instances += len(transforms)
        return self.tile.get_triangle_count() - before

    def flush_tile(self):
        """Fold the finished tile into the total, trimming if it can be trimmed.

        The tile is already inside its ceiling — `append` enforced that — so this
        is mostly bookkeeping. It still asks for a reduction when the tile came in
        over its share, but never below what a simplifier can actually deliver:
        the floor is one shell's worth per instance, and asking for less spends
        minutes achieving nothing.
        """
        count = self.tile.get_triangle_count()
        if count == 0:
            return
        # A tile whose vertices are not in any world is dropped, loudly.
        #
        # This is the last line of defence and it earned its place: an export of
        # well over ten million triangles finished with every number in the report
        # correct and a structure part spanning 7.8e34 centimetres, which opens as
        # an empty screen because the map is a pixel inside its own bounds.
        # Somewhere in hundreds of thousands of actors one placement or one asset
        # was nonsense. Checking each
        # asset and each placement individually is right and is done, but a check
        # here is what makes the *file* safe rather than the paths currently known
        # to produce bad data — and it names the tile, which is where to look.
        if not sane_mesh(self.tile):
            self.insane_tiles += 1
            self.log(
                "  ! tile {} of {} has vertices outside any possible world; "
                "dropping its {} triangles".format(
                    self.tiles_flushed + 1, self.name, count
                )
            )
            self.tile.reset()
            self.tile_instances = 0
            return
        self.raw_triangles += count
        floor = SHELL_FLOOR * max(1, self.tile_instances)
        share = max(64, self.tile_share, min(count, floor))
        reduced = count
        if count > share:
            t0 = time.time()
            self.tile = self.tile.apply_simplify_to_triangle_count(
                share, simplify_options()
            )
            reduced = self.tile.get_triangle_count()
            self.simplify_seconds += time.time() - t0
        self.total = self.total.append_mesh(self.tile, unreal.Transform())
        self.tile.reset()
        # Every number the allocation depends on, once per tile. Cheap, and the
        # alternative is inferring the arithmetic from a running total that only
        # ever goes up: the first two attempts at this budget were debugged from
        # the outside and both diagnoses were wrong.
        self.log(
            "  tile {}: {} raw -> {} of a {} share (spent {}/{}, tile now {})".format(
                self.tiles_flushed + 1,
                count,
                reduced,
                share,
                self.total.get_triangle_count(),
                self.budget,
                self.tile.get_triangle_count(),
            )
        )
        self.tile_instances = 0
        self.tiles_flushed += 1

    def finish(self):
        """Last reduction, against the real budget rather than an estimate.

        A backstop, and it says so when it has work to do: with the allocation
        behaving, tiles land inside the budget and this trims a few percent.
        Anything more means the per-tile shares are wrong, and a quiet overshoot
        is how a 55MB export becomes a 280MB one.
        """
        self.flush_tile()
        count = self.total.get_triangle_count()
        if count > self.budget:
            t0 = time.time()
            self.total = self.total.apply_simplify_to_triangle_count(
                self.budget, simplify_options()
            )
            after = self.total.get_triangle_count()
            self.simplify_seconds += time.time() - t0
            self.log(
                "  final trim: {} -> {} triangles against a budget of {} ({:.0f}s)".format(
                    count, after, self.budget, time.time() - t0
                )
            )
            if after > self.budget * 1.05:
                self.log(
                    "  ! {:.1f}x over budget after the final trim. With {} separate "
                    "objects the floor is about {}k triangles whatever target is "
                    "asked for — raise --min-size or --budget.".format(
                        after / float(self.budget),
                        self.instances,
                        (self.instances * SHELL_FLOOR) // 1000,
                    )
                )
        return self.total

    def stats(self):
        return {
            "instances": self.instances,
            "instances_over_budget": self.skipped_instances,
            # Lone objects admitted despite exceeding the per-actor share. A
            # landmark is normally one of these, and before they were admitted
            # they were refused outright — see `append`.
            "oversized_singles": self.oversized_singles,
            "insane_tiles": self.insane_tiles,
            "tiles": self.tiles_flushed,
            "raw_triangles": self.raw_triangles,
            "triangles": self.total.get_triangle_count(),
            "append_seconds": round(self.append_seconds, 1),
            "simplify_seconds": round(self.simplify_seconds, 1),
        }


def read_mesh(mesh, scale, swap_yz=True, limit=None):
    """Pull a DynamicMesh into flat float/uint arrays in viewer space.

    `limit` drops triangles with any vertex further than that from the origin,
    and returning the count of them is the point: this is the last place before
    the bytes reach the file, so it is the only check that can promise the file
    is clean.

    It has to exist. Twenty-six vertex values out of millions in one export were
    at 1e26 and another few thousand were a hundred kilometres up, and every
    guard placed further upstream — on each asset as it was read, on each
    placement as it was used, on each tile as it was folded in — passed them
    without a word. The map's own bounds then span 1e26 centimetres, and a viewer
    that frames what it loads shows an empty screen with the whole level one pixel
    across somewhere inside it. Nothing in the report looks wrong.

    Unreal is Z-up and centimetres; the viewer's world is Y-up and takes its
    units from whatever the telemetry was in — which, for telemetry exported
    from the same game, is also centimetres. So the default is a pure axis
    relabel with no scaling, and the level lands exactly where the datapoints do.

    Swapping Y and Z mirrors the space, which reverses which way a triangle
    faces, so the winding is reversed to compensate. Getting that wrong leaves
    every surface facing inward: the viewer computes its normals from the
    winding, and the ground would come out lit from below and classified as
    something else entirely.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - Unreal ships numpy
        np = None

    mesh = mesh.compact_mesh()
    _m, position_list, pos_gaps = mesh.get_all_vertex_positions(True)
    _m2, triangle_list, tri_gaps = mesh.get_all_triangle_indices(True)
    vectors = position_list.convert_vector_list_to_array()
    triangles = triangle_list.convert_triangle_list_to_array()

    n = len(vectors)
    if np is not None:
        positions = np.empty((n, 3), dtype=np.float32)
        for i, v in enumerate(vectors):
            positions[i, 0] = v.x
            positions[i, 1] = v.y
            positions[i, 2] = v.z
        if swap_yz:
            positions = positions[:, [0, 2, 1]]
        if scale != 1.0:
            positions *= scale

        indices = np.empty((len(triangles), 3), dtype=np.uint32)
        for i, t in enumerate(triangles):
            indices[i, 0] = t.x
            indices[i, 1] = t.y
            indices[i, 2] = t.z
        if swap_yz:
            indices = indices[:, [0, 2, 1]]

        dropped = 0
        if limit:
            # A vertex is in range if every coordinate is finite and inside the
            # box; a triangle survives if all three of its vertices are. The
            # positions themselves are left alone — an unreferenced vertex costs
            # twelve bytes and nothing else, and rewriting the index map to remove
            # them would cost a pass over five million entries to save a megabyte.
            ok = np.isfinite(positions).all(axis=1) & (
                np.abs(positions) <= limit
            ).all(axis=1)
            if not ok.all():
                keep = ok[indices].all(axis=1)
                dropped = int((~keep).sum())
                indices = indices[keep]
                # And move the offending vertices to the origin.
                #
                # Dropping only the triangles is not enough, and finding that out
                # cost a whole export: a glTF accessor's `min`/`max` are computed
                # over every position in it, referenced or not, and a viewer frames
                # what it loads from those. So the file rendered correctly and
                # still reported bounds of 1e26, which looks exactly like the
                # original bug. Nothing references these vertices now, so where
                # they are does not matter — only that it is somewhere finite.
                positions[~ok] = 0.0
        return positions.reshape(-1), indices.reshape(-1), dropped

    positions = []
    in_range = []
    for v in vectors:
        if swap_yz:
            point = (v.x * scale, v.z * scale, v.y * scale)
        else:
            point = (v.x * scale, v.y * scale, v.z * scale)
        fine = not limit or all(
            math.isfinite(c) and abs(c) <= limit for c in point
        )
        # Moved to the origin rather than left where they are; see the numpy path.
        positions.extend(point if fine else (0.0, 0.0, 0.0))
        in_range.append(fine)
    indices = []
    dropped = 0
    for t in triangles:
        tri = (t.x, t.z, t.y) if swap_yz else (t.x, t.y, t.z)
        if limit and not all(in_range[i] for i in tri):
            dropped += 1
            continue
        indices.extend(tri)
    return positions, indices, dropped




