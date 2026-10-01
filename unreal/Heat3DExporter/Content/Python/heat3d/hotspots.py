"""Where the players actually went, if anyone has said.

Every other decision in this exporter is a guess about what matters, made by
looking at the level: how big an actor is, what its asset is called, how many
actors a sub-level holds. Those guesses are why the name lists in `classify` keep
growing, and they guess worst exactly where it matters most — a large building
that players spend an hour inside ranks below a hillside of rock, because the
hillside has more actors in it and a bigger bounding box.

The telemetry knows the answer, and it is the same telemetry the viewer is going
to draw on top of the export. `tools/heat-grid.mjs` buckets it into a coarse grid
in Unreal centimetres and writes two numbers per cell: how many datapoints landed
there, and how many distinct 3m height bands were stood on. The second is what
separates a building from a field — on open ground everyone is within a metre or
two of the surface, and inside a three-storey building they are spread over
fifteen.

Optional by construction. With no grid the exporter behaves exactly as it did,
because a first export of a new map has no telemetry to draw on yet — and the
point of this file is the *second* export, where it does.
"""
import json
import os


class Hotspots(object):
    """A traffic grid, and how to ask it about a place or a box."""

    def __init__(self, cell, cells, rows=0, source=""):
        self.cell = float(cell) if cell else 3200.0
        self.rows = rows
        self.source = source
        # (cx, cy) -> [datapoints, height bands used]
        self.grid = cells
        self.max_rows = max((v[0] for v in cells.values()), default=0)

    def __bool__(self):
        return bool(self.grid)

    # Python 2 style truth, in case this ever runs under an older embedded
    # interpreter than the one shipping with the engine now.
    __nonzero__ = __bool__

    def at(self, x, y):
        """`(datapoints, bands)` for the cell containing this point."""
        key = (int(x // self.cell), int(y // self.cell))
        return self.grid.get(key, (0, 0))

    def score_box(self, bmin, bmax):
        """Traffic across the cells a bounding box covers.

        Summed, not averaged. A building that covers nine busy cells is worth more
        than one that covers a single busy cell — it is a bigger place with more
        of the map's traffic inside it, and the budget is being divided between
        them.
        """
        x0 = int(bmin[0] // self.cell)
        x1 = int(bmax[0] // self.cell)
        y0 = int(bmin[1] // self.cell)
        y1 = int(bmax[1] // self.cell)
        rows = 0
        bands = 0
        for cx in range(min(x0, x1), max(x0, x1) + 1):
            for cy in range(min(y0, y1), max(y0, y1) + 1):
                got = self.grid.get((cx, cy))
                if got:
                    rows += got[0]
                    bands = max(bands, got[1])
        return rows, bands

    def score_points(self, points, reach=1):
        """Traffic near a set of positions, for scoring one instance.

        `reach` in cells, so a building is credited with the traffic just outside
        it as well: people walk up to a door before they walk through it, and a
        32m cell is smaller than a compound.
        """
        seen = set()
        rows = 0
        bands = 0
        for x, y in points:
            cx = int(x // self.cell)
            cy = int(y // self.cell)
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    key = (cx + dx, cy + dy)
                    if key in seen:
                        continue
                    seen.add(key)
                    got = self.grid.get(key)
                    if got:
                        rows += got[0]
                        bands = max(bands, got[1])
        return rows, bands


def load(path, log):
    """Read a grid written by `tools/heat-grid.mjs`, or return an empty one.

    Never raises. A missing or malformed grid means "nobody said where the
    players went", which is a normal state for a map being exported for the first
    time, and it must not be the difference between an export and no export.
    """
    if not path:
        return Hotspots(0, {})
    if not os.path.isfile(path):
        log("hotspots: no file at {}; ranking by size instead".format(path))
        return Hotspots(0, {})
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        cells = {}
        for row in data.get("cells", ()):
            if len(row) >= 3:
                cells[(int(row[0]), int(row[1]))] = (
                    int(row[2]),
                    int(row[3]) if len(row) > 3 else 0,
                )
        spots = Hotspots(
            data.get("cell", 3200),
            cells,
            rows=int(data.get("rows", 0)),
            source=str(data.get("source", "")),
        )
        log(
            "hotspots: {} cells of {:.0f}m from {} datapoints ({})".format(
                len(cells),
                spots.cell / 100.0,
                spots.rows,
                spots.source.rsplit("/", 1)[-1] or path,
            )
        )
        return spots
    except Exception as err:  # noqa: BLE001 - a bad grid must not stop the export
        log("hotspots: could not read {}: {}".format(path, err))
        return Hotspots(0, {})


def default_path(out_path):
    """`hotspots.json` beside the output file, which is where the tool writes it."""
    directory = os.path.dirname(os.path.abspath(out_path or "."))
    return os.path.join(directory, "hotspots.json")
