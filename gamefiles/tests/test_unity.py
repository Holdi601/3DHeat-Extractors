"""
Tests for the Unity reader.

A Unity container is unusually good at checking itself, and both tests here lean
on that rather than on fixtures.

The object table has to *tile* the data section: every object's byte range inside
it, none overlapping, none separated by more than alignment padding, and the last
one ending exactly at the end of the file. Nothing in the header says so — it
falls out of the file being written by walking the objects in order — so a header
misread by four bytes, which still yields a plausible object count, cannot
produce a table that tiles.

A mesh checks itself too. A shipped build carries no type tree, so `read_mesh`
reads fields in an order known from the engine version and has no way to notice
it slipped. But Unity stores each mesh's bounding box as its own field, computed
from the real vertices, so the box of the decoded vertices has to agree with it.
That is what `aabb_matches()` is for, and it is the assertion that matters.

Both need a game installed and skip without one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from heat3d_gamefiles.unity import UnsupportedUnity, read_serialized
from heat3d_gamefiles.unity_mesh import read_mesh

from .installed import installed

def containers(limit: int = 6) -> list[Path]:
    """Unity containers on this machine: anything with a Unity data folder, two
    levels deep to catch launcher layouts."""
    found: list[Path] = []
    for folder in installed("*/*_Data", "*/*/*_Data"):
        for name in ("resources.assets", "sharedassets0.assets"):
            candidate = folder / name
            if candidate.is_file():
                found.append(candidate)
    return sorted(found)[:limit]


def readable() -> list[Path]:
    """Those the reader accepts — older formats are refused by design."""
    out = []
    for path in containers():
        try:
            read_serialized(path)
            out.append(path)
        except UnsupportedUnity:
            continue
    return out


INSTALLED = readable()
pytestmark = pytest.mark.skipif(not INSTALLED, reason="no Unity game on this machine")


@pytest.mark.parametrize("path", INSTALLED, ids=lambda p: p.parent.parent.name)
def test_object_table_tiles_the_data_section(path: Path):
    """
    Objects fill the data section exactly, in order and without overlap.

    The header field that decides this is a `uint32` sitting between two
    `int64`s; reading all three the same width moves the engine version string
    four bytes and everything after it. The count still parses, so the failure
    is silent until something like this looks at the result.
    """
    container = read_serialized(path)
    size = path.stat().st_size
    objects = sorted(container.objects, key=lambda o: o.offset)
    assert objects, "container holds no objects"

    # Objects are padded up to a boundary, and which boundary varies: 8 bytes in
    # some builds, 16 in others. So the check is that any gap is *only* padding —
    # smaller than the alignment and landing the next object on it — rather than
    # a fixed number of bytes.
    for before, after in zip(objects, objects[1:]):
        end = before.offset + before.size
        gap = after.offset - end
        assert gap >= 0, "objects overlap"
        assert gap < 16, f"{gap}-byte gap is too large to be padding"
        assert after.offset % 8 == 0, "object does not start on an alignment boundary"

    last = objects[-1]
    assert container.data_offset + last.offset + last.size == size
    assert len({o.path_id for o in objects}) == len(objects), "duplicate path ids"


@pytest.mark.parametrize("path", INSTALLED, ids=lambda p: p.parent.parent.name)
def test_engine_version_parses_as_a_version(path: Path):
    container = read_serialized(path)

    assert container.unity_version[0].isdigit(), container.unity_version
    assert "." in container.unity_version


def with_meshes() -> list[Path]:
    return [p for p in INSTALLED if read_serialized(p).of_class("Mesh")]


@pytest.mark.skipif(not with_meshes(), reason="no Unity meshes on this machine")
@pytest.mark.parametrize("path", with_meshes(), ids=lambda p: p.parent.parent.name)
def test_decoded_meshes_reproduce_their_recorded_bounds(path: Path):
    """
    The vertices decode to the bounding box Unity recorded for them.

    Reading a struct in the wrong order does not raise — it gives plausible
    floats. This is the check that tells the two apart, and it is why the mesh
    reader is willing to be pointed at a game nobody has tried.

    Compressed meshes are a different encoding and are refused rather than
    guessed at; those are allowed to raise, not to return something wrong.
    """
    container = read_serialized(path)
    meshes = container.of_class("Mesh")[:40]

    decoded = 0
    agreed = 0
    for obj in meshes:
        try:
            mesh = read_mesh(container, obj)
        except UnsupportedUnity:
            continue  # refused for a stated reason, which is the correct outcome
        decoded += 1
        agreed += mesh.aabb_matches()
        assert len(mesh.vertices) > 0
        if len(mesh.triangles):
            assert mesh.triangles.max() < len(mesh.vertices), mesh.name

    assert decoded, "no mesh in this container could be decoded at all"
    # Not every mesh: half-precision positions and normalised integers put a few
    # just outside the tolerance. A layout error puts *all* of them outside.
    assert agreed / decoded > 0.9, f"only {agreed} of {decoded} matched their bounds"


def test_refuses_a_bundle_where_a_container_is_expected(tmp_path: Path):
    fake = tmp_path / "level.bundle"
    fake.write_bytes(b"UnityFS\x00" + bytes(200))

    with pytest.raises(UnsupportedUnity, match="asset bundle"):
        read_serialized(fake)


def test_refuses_a_file_that_is_not_unity_at_all(tmp_path: Path):
    junk = tmp_path / "notes.txt"
    junk.write_bytes(b"x" * 500)

    with pytest.raises(UnsupportedUnity):
        read_serialized(junk)
