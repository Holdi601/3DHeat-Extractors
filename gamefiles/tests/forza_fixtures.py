"""
Building ForzaTech containers, so the readers can be tested without Forza.

A shipped track is 97 GB across four archives and cannot be checked in, but the
formats are small enough to write. Everything here mirrors what a real file does
— including the parts that are easy to get wrong and that a hand-made fixture
would otherwise quietly avoid: the sixty-four bit segment base, the padding that
makes an entry's stored length differ from its compressed length, and the mixed
codecs.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from heat3d_gamefiles.minizip import DEFLATE, HEADER, LZ4, MAGIC, STORED


def pack(payload: bytes, codec: int) -> bytes:
    if codec == STORED:
        return payload
    if codec == DEFLATE:
        maker = zlib.compressobj(6, zlib.DEFLATED, -15)
        return maker.compress(payload) + maker.flush()
    if codec == LZ4:
        import lz4.block

        return lz4.block.compress(payload, store_size=False)
    raise ValueError(f"no such codec: {codec}")


def build_minizip(
    path: str | Path,
    payloads: list[bytes],
    *,
    codecs: list[int] | None = None,
    per_segment: int = 4,
    base_offset: int = 0,
) -> Path:
    """
    Write an archive holding `payloads`, one entry each.

    `base_offset` pads the file before the data so the segment bases can be
    pushed past four gigabytes without writing four gigabytes — the entries are
    then unreadable, but the index is exactly what a large archive's is, which
    is the part that needs testing.
    """
    path = Path(path)
    codecs = codecs or [LZ4] * len(payloads)
    count = len(payloads)
    segments = max(1, -(-count // per_segment))

    # One bundle per entry keeps the table honest without meaning anything.
    bundles = count
    head = struct.pack(
        "<4sIIIIIII", MAGIC, 101, HEADER, count, bundles, per_segment, segments, 0
    )
    table = struct.pack(f"<{bundles + 1}I", *range(bundles + 1))
    # The segment bases are eight-byte aligned, so a table with an odd number of
    # entries is followed by a padding word. Reproduced rather than avoided: a
    # fixture that only ever produced one parity would let a reader that ignores
    # the alignment pass everything here and fail on half the shipped archives.
    table += b"\x00" * ((-(len(head) + len(table))) % 8)
    per_last = count - per_segment * (segments - 1)
    index_bytes = (
        len(head)
        + len(table)
        + (segments - 1) * (8 + per_segment * 12)
        + 8
        + per_last * 12
    )
    data_start = index_bytes + base_offset

    # Lay the payloads out segment by segment, four-byte aligned, so each
    # entry's stored length is its compressed length plus its padding.
    records: list[bytes] = []
    bases: list[int] = []
    blob = bytearray()
    at = data_start
    for s in range(segments):
        bases.append(at)
        rows = []
        here = payloads[s * per_segment : (s + 1) * per_segment]
        kinds = codecs[s * per_segment : (s + 1) * per_segment]
        offset = 0
        for payload, codec in zip(here, kinds):
            packed = pack(payload, codec)
            padding = (-len(packed)) % 4
            rows.append(
                struct.pack(
                    "<3I", offset, len(payload), codec | (padding << 12)
                )
            )
            blob += packed + b"\x00" * padding
            offset += len(packed) + padding
        records.append(struct.pack("<Q", at) + b"".join(rows))
        at += offset

    path.write_bytes(
        head + table + b"".join(records) + b"\x00" * base_offset + bytes(blob)
    )
    return path


def build_contents(path: str | Path, names: list[str]) -> Path:
    """The manifest beside an archive: one line per entry, in entry order."""
    path = Path(path)
    body = "".join(
        f"<PREZIPPED>d:\\scratch\\p4\\forte_main\\zipcache\\pc\\{name}|{i}\n"
        for i, name in enumerate(names)
    )
    path.write_text(body, encoding="utf-8")
    return path


def _chunk(tag: str, payload: bytes) -> tuple[bytes, bytes]:
    return tag[::-1].encode("ascii"), payload


def build_modelbin(
    *,
    positions: np.ndarray,
    faces: np.ndarray,
    scale: tuple[float, float, float] = (256.0, 256.0, 256.0),
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
    position_format: int = 13,
    index_format: int = 57,
    extra_attribute: bool = False,
) -> bytes:
    """
    A `.modelbin` holding one mesh, quantised the way terrain is.

    `positions` are world metres; they are encoded into the box `scale` and
    `bias` describe, which is what the reader has to undo.
    """
    points = np.asarray(positions, dtype=np.float64)
    shorts = np.zeros((len(points), 4), dtype="<i2")
    shorts[:, :3] = np.round(
        (points - np.asarray(bias)) / np.asarray(scale) * 32767.0
    ).astype("<i2")

    names = ["POSITION"]
    elements = [(0, 0, 0, position_format)]
    if extra_attribute:
        names.append("TEXCOORD")
        elements += [(1, 0, 1, 35), (1, 1, 1, 35)]

    layout = struct.pack("<H", len(names))
    for name in names:
        layout += struct.pack("<I", len(name)) + name.encode("ascii")
    layout += struct.pack("<H", len(elements))
    for name, channel, buffer, fmt in elements:
        layout += struct.pack("<HHIIiI", name, channel, buffer, fmt, -1, 0)

    vertices = shorts.tobytes()
    vertex_chunk = struct.pack("<IIHHI", len(shorts), len(vertices), 8, 1, position_format) + vertices
    indices = np.asarray(faces, dtype="<u2").reshape(-1)
    index_chunk = (
        struct.pack("<IIHHI", len(indices), len(indices) * 2, 2, 1, index_format)
        + indices.tobytes()
    )

    # Only the tail matters to the reader: a scale and a bias, each padded out
    # to four floats.
    mesh_chunk = b"\x01\x00" + b"\x00" * 204 + struct.pack(
        "<8f", *scale, 0.0, *bias, 0.0
    )

    parts = [
        ("Mesh", mesh_chunk),
        ("IndB", index_chunk),
        ("VLay", layout),
        ("VerB", vertex_chunk),
    ]
    if extra_attribute:
        uv = np.zeros((len(shorts), 2), dtype="<i2")
        parts.append(
            (
                "VerB",
                struct.pack("<IIHHI", len(uv), uv.nbytes, 4, 1, 35) + uv.tobytes(),
            )
        )

    return build_container(parts)


def build_container(parts: list[tuple[str, bytes]]) -> bytes:
    """A `burG` chunk container holding `parts`, as (tag, payload) in order."""
    table_at = 20
    body_at = table_at + len(parts) * 24
    records = b""
    blob = bytearray()
    at = body_at
    for tag, payload in parts:
        records += tag[::-1].encode("ascii") + struct.pack(
            "<5I", 1, 0, at, len(payload), len(payload)
        )
        blob += payload
        pad = (-len(payload)) % 4
        blob += b"\x00" * pad
        at += len(payload) + pad

    total = body_at + len(blob)
    head = struct.pack("<4sIIII", b"burG", 1, 0, total, len(parts))
    return head + records + bytes(blob)


def build_texture(
    pixels: bytes,
    *,
    width: int,
    height: int,
    format_code: int,
    guid: bytes = bytes(range(16)),
    mips: int = 1,
) -> bytes:
    """
    A `.pb` swatch: one `TXCB` chunk of pixels behind the 96-byte `TXCH` header.

    Laid out as a shipped one is: the header sits between the chunk table and
    the pixels, and the chunk record's name field points at it. Byte 47 is 4,
    as it is in every real file, so a reader that takes it for the format is
    caught.
    """
    header = bytearray(96)
    header[0:4] = b"HCXT"
    header[16:32] = guid
    struct.pack_into("<II", header, 32, width, height)
    header[46] = mips
    header[47] = 4
    struct.pack_into("<I", header, 72, format_code)
    struct.pack_into("<I", header, 84, len(pixels))
    names_at = 20 + 24
    body_at = names_at + len(header)
    total = body_at + len(pixels)
    head = struct.pack("<4sIIII", b"burG", 0x101, body_at, total, 1)
    record = b"TXCB"[::-1] + struct.pack("<5I", 0x10000, names_at, body_at, len(pixels), len(pixels))
    return head + record + bytes(header) + pixels


def build_track(folder: str | Path, files: dict[str, bytes], *, chunk: int = 0) -> Path:
    """
    A track folder with one archive holding `files`, by their paths inside it.

    What `open_chunk` and everything built on it expects to find: the archive
    and its contents list, side by side.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = list(files)
    build_minizip(folder / f"GeoChunk{chunk}.minizip", [files[n] for n in names])
    build_contents(folder / f"ChunkContentsMiniZip{chunk}.txt", names)
    return folder


def _layout(names: list[str], elements: list[tuple[int, int, int, int, int]]) -> bytes:
    """A `VLay`: attribute names, then (name, channel, buffer, format, offset) elements."""
    out = struct.pack("<H", len(names))
    for name in names:
        out += struct.pack("<I", len(name)) + name.encode("ascii")
    out += struct.pack("<H", len(elements))
    for name, channel, buffer, fmt, offset in elements:
        out += struct.pack("<HHIIiI", name, channel, buffer, fmt, -1, offset)
    return out


def _mesh_chunk(*, variants, lods: int, first: int, triangles: int, layout: int,
                buffer: int, stride: int, scale, bias) -> bytes:
    """
    One submesh record. With one material variant it is the standard 238 bytes,
    the entry first and no count; with more it opens with `u32 count` and the
    entries, and every field after them moves back by the extra length.
    """
    entries = b"".join(struct.pack("<HHHH", 0xFFFF, summer, 0xFFFF, winter) for summer, winter in variants)
    head = entries if len(variants) == 1 else struct.pack("<I", len(variants)) + entries
    prefix = len(head) - 8
    raw = bytearray(238 + prefix)
    raw[: len(head)] = head
    struct.pack_into("<H", raw, 10 + prefix, lods)
    for offset, value in ((34, first), (42, triangles * 3), (46, triangles), (62, layout), (90, buffer), (98, stride)):
        struct.pack_into("<I", raw, offset + prefix, value)
    struct.pack_into("<8f", raw, len(raw) - 32, *scale, 0.0, *bias, 0.0)
    return bytes(raw)


def build_textured_model(
    *,
    positions: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    submeshes: list[tuple],
    materials: list[bytes] = (),
    scale=(8.0, 8.0, 8.0),
    bias=(0.0, 0.0, 0.0),
) -> bytes:
    """
    A model stored the way a placed prop is.

    One position buffer (four shorts, the fourth being the normal's x), then an
    attribute buffer holding the normal's y and z as two shorts and a UV set as
    two unsigned shorts, described by a second layout. `submeshes` are
    (material, lod mask, first triangle, triangles), where the material is a
    `MatI` index or a list of (summer, winter) variants.
    """
    points = np.asarray(positions, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    shorts = np.zeros((len(points), 4), dtype="<i2")
    shorts[:, :3] = np.round((points - np.asarray(bias)) / np.asarray(scale) * 32767.0)
    shorts[:, 3] = np.round(n[:, 0] * 32767.0)
    attributes = np.zeros((len(points), 4), dtype="<u2")
    attributes[:, 0:2] = np.round(n[:, 1:3] * 32767.0).astype("<i2").view("<u2")
    attributes[:, 2:4] = np.round(np.asarray(uvs) * 65535.0)

    POSITION, NORMAL, TEXCOORD = 0, 1, 2
    layouts = [
        _layout(["POSITION"], [(POSITION, 0, 0, 13, 0)]),
        _layout(["POSITION", "NORMAL", "TEXCOORD"],
                [(POSITION, 0, 0, 13, 0), (NORMAL, 0, 1, 37, 0), (TEXCOORD, 0, 1, 35, 4)]),
    ]
    indices = np.asarray(faces, dtype="<u2").reshape(-1)
    parts = [("MatI", m) for m in materials]
    for material, lods, first, triangles in submeshes:
        variants = [(material, 0xFFFF)] if isinstance(material, int) else list(material)
        parts.append(("Mesh", _mesh_chunk(variants=variants, lods=lods, first=first * 3,
                                          triangles=triangles, layout=1, buffer=1, stride=8,
                                          scale=scale, bias=bias)))
    parts.append(("IndB", struct.pack("<IIHHI", len(indices), len(indices) * 2, 2, 1, 57) + indices.tobytes()))
    parts += [("VLay", layout) for layout in layouts]
    parts.append(("VerB", struct.pack("<IIHHI", len(shorts), shorts.nbytes, 8, 1, 13) + shorts.tobytes()))
    parts.append(("VerB", struct.pack("<IIHHI", len(attributes), attributes.nbytes, 8, 2, 37) + attributes.tobytes()))
    return build_container(parts)
