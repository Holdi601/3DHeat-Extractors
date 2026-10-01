"""
What a match is written as: one Parquet table for the heatmap and one JSON
document for the match analysis.

The table carries its own reading instructions in the Parquet footer, under
the key `heat3d`: which column is the position, the time, the player, the
side, the look direction, and which way round the world is. CS2's world is
right-handed with Z up, so it is drawn with the vertical as Y and the game's
Y negated - otherwise every map comes out mirrored, A site on the wrong side.
A viewer that knows the key needs no form filled in; anything else reads an
ordinary table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .match import NUMERIC, ROW_COLUMNS, Match
from .stats import summary

#: How a viewer should read the table. See the module docstring.
HINTS = {
    "version": 1,
    "kind": "cs2-demo",
    "axes": {"up": "z", "flipX": False, "flipY": True, "flipZ": False, "scale": 1},
    "columns": {
        "x": "x",
        "y": "y",
        "z": "z",
        "time": "time",
        "timeUnit": "seconds",
        "personId": "player",
        "teamId": "side",
        "sessionId": "roundId",
        "lookMode": "vector",
        "dirX": "dirX",
        "dirY": "dirY",
        "dirZ": "dirZ",
        "extra": ["event", "match", "round", "roundTime", "phase", "weapon", "place", "won", "buy", "opening", "team"],
    },
}

#: Columns stored as small integers or 32-bit floats; the rest are strings or float64.
INT16 = {"round", "health", "armor"}
INT32 = {"money", "equipment"}
FLOAT64 = {"time"}


def table(match: Match) -> pa.Table:
    rows = match.rows
    arrays = []
    fields = []
    for c in ROW_COLUMNS:
        col = rows[c]
        if c in NUMERIC:
            values = col.to_numpy(dtype=float)
            if c in INT16 or c in INT32:
                kind = pa.int16() if c in INT16 else pa.int32()
                mask = ~np.isfinite(values)
                ints = np.where(mask, 0, values).astype(np.int64)
                arrays.append(pa.array(ints, type=kind, mask=mask))
                fields.append(pa.field(c, kind))
            elif c in {"headshot", "opening", "traded", "won"}:
                mask = ~np.isfinite(values)
                arrays.append(pa.array(np.where(mask, 0, values).astype(np.int8), type=pa.int8(), mask=mask))
                fields.append(pa.field(c, pa.int8()))
            else:
                kind = pa.float64() if c in FLOAT64 else pa.float32()
                arrays.append(pa.array(values, type=kind, from_pandas=True))
                fields.append(pa.field(c, kind))
        else:
            arrays.append(pa.array(col.fillna("").astype(str).to_numpy(), type=pa.string()).dictionary_encode())
            fields.append(pa.field(c, pa.dictionary(pa.int32(), pa.string())))
    schema = pa.schema(fields, metadata={"heat3d": json.dumps(HINTS)})
    return pa.Table.from_arrays(arrays, schema=schema)


def write(match: Match, folder: str | Path) -> tuple[Path, Path]:
    """`<match>.parquet` and `<match>.match.json` in `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = folder / f"{match.name}.parquet"
    doc = folder / f"{match.name}.match.json"
    pq.write_table(table(match), data, compression="snappy", row_group_size=256_000)
    doc.write_text(json.dumps(summary(match), indent=1, ensure_ascii=False), encoding="utf-8")
    return data, doc
