"""
Unreal Engine 5 cooked packages as IoStore stores them ("zen" packages).

A package is a header - names, the objects it imports, the objects it
exports - followed by each export's serialized data. Objects refer to one
another by `FPackageObjectIndex`, a 64-bit number whose top two bits say what
it is: an export of this package, a script object (a C++ class or its default
object, listed once for the whole game in `global.utoc`), an import from
another package (which package, and which of its exports by a hash of the
export's path), or nothing.

The summary grew between versions: 5.3 replaced the graph data with
dependency bundles and added the imported package names, 5.6 added two
offsets for "cells". Both layouts are tried and the one whose name table
parses is kept, so the reader does not need to be told the version.

Property data inside an export is read by `unversioned.py`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache

from .iostore import SCRIPT_OBJECTS, Store

_INDEX_MASK = (1 << 62) - 1
EXPORT, SCRIPT, PACKAGE, NULL = 0, 1, 2, 3


class ZenError(Exception):
    """A package that does not read as a zen package."""


def name_batch(data: bytes, at: int) -> tuple[list[str], int]:
    """A batch of names as `SaveNameBatch` writes them; the names and where it ends."""
    (count,) = struct.unpack_from("<i", data, at)
    at += 4
    if count == 0:
        return [], at
    if count < 0 or count > 10_000_000:
        raise ZenError("implausible name count")
    (size,) = struct.unpack_from("<I", data, at)
    at += 4 + 8 + count * 8  # string bytes, hash algorithm, hashes
    lengths = data[at : at + count * 2]
    at += count * 2
    names = []
    cursor = at
    for i in range(count):
        high, low = lengths[2 * i], lengths[2 * i + 1]
        length = ((high & 0x7F) << 8) | low
        if high & 0x80:
            if cursor % 2:
                cursor += 1
            names.append(data[cursor : cursor + length * 2].decode("utf-16-le"))
            cursor += length * 2
        else:
            names.append(data[cursor : cursor + length].decode("latin-1"))
            cursor += length
    return names, at + size


@dataclass
class Export:
    index: int
    name: str
    outer: int
    class_index: int
    template: int
    serial_offset: int
    serial_size: int
    public_hash: int
    flags: int
    #: Where its data starts in the package.
    at: int = 0


class Package:
    """One cooked package's header, and access to each export's data."""

    def __init__(self, data: bytes, path: str = "", scripts: "Scripts | None" = None) -> None:
        self.data = data
        self.path = path
        self.scripts = scripts
        (versioned, self.header_size) = struct.unpack_from("<II", data, 0)
        if versioned:
            raise ZenError(f"{path}: versioned packages are not read")
        (name_index, name_number, self.flags) = struct.unpack_from("<III", data, 8)
        offsets = struct.unpack_from("<10i", data, 24)
        # 5.3-5.5: nine fields then names at 52; 5.6: two more offsets, names at 60.
        for start, count in ((60, 10), (52, 8)):
            try:
                names, end = name_batch(data, start)
            except (ZenError, struct.error, UnicodeDecodeError, IndexError):
                continue
            if names and end <= self.header_size and name_index < len(names):
                break
        else:
            raise ZenError(f"{path}: the name table does not parse in either known layout")
        self.names = names
        # Imported export hashes, imports, exports, export bundles, two
        # dependency tables, imported package names (and in 5.6 two more).
        (self.hashes_at, self.imports_at, self.exports_at, self.bundles_at) = offsets[0:4]
        self.imported_names_at = offsets[6]
        self.name = self.mapped(name_index, name_number)
        self.public_hashes = [
            struct.unpack_from("<Q", data, self.hashes_at + 8 * i)[0]
            for i in range((self.imports_at - self.hashes_at) // 8)
        ]
        self.imports = [
            struct.unpack_from("<Q", data, self.imports_at + 8 * i)[0]
            for i in range((self.exports_at - self.imports_at) // 8)
        ]
        # The imported packages' names, then each name's number: a package
        # `SM_Tile_Terrain_0` is the name `SM_Tile_Terrain` numbered 1.
        try:
            bases, end = name_batch(data, self.imported_names_at)
            numbers = struct.unpack_from(f"<{len(bases)}i", data, end) if bases else ()
            self.imported_packages = [base if n == 0 else f"{base}_{n - 1}" for base, n in zip(bases, numbers)]
        except (ZenError, struct.error, UnicodeDecodeError):
            self.imported_packages = []
        self.exports: list[Export] = []
        for i in range((self.bundles_at - self.exports_at) // 72):
            (offset, size, ni, nn, outer, cls, _super, template, public, flags) = struct.unpack_from(
                "<QQIIQQQQQI", data, self.exports_at + 72 * i
            )
            # Export data follows the header in serial-offset order (checked:
            # the offsets run contiguously from zero).
            self.exports.append(
                Export(i, self.mapped(ni, nn), outer, cls, template, offset, size, public, flags, self.header_size + offset)
            )

    def mapped(self, index: int, number: int) -> str:
        name = self.names[index & 0x3FFFFFFF]
        return name if number == 0 else f"{name}_{number - 1}"

    def export_data(self, export: Export) -> bytes:
        return self.data[export.at : export.at + export.serial_size]

    def class_name(self, export: Export) -> str:
        """`/Script/Engine.StaticMeshComponent`, say; a package import's class as its package."""
        return self.describe(export.class_index)

    def describe(self, index: int) -> str:
        kind, value = index >> 62, index & _INDEX_MASK
        if kind == NULL:
            return ""
        if kind == EXPORT:
            return f"export:{value}"
        if kind == SCRIPT:
            return self.scripts.path(index) if self.scripts else f"script:{value:x}"
        package, _hash = self.package_import(index)
        return package or ""

    def package_import(self, index: int) -> tuple[str | None, int | None]:
        """For an import of another package's export: that package's name and the export's hash."""
        if index >> 62 != PACKAGE:
            return None, None
        value = index & _INDEX_MASK
        package, hash_index = value >> 32, value & 0xFFFFFFFF
        name = self.imported_packages[package] if package < len(self.imported_packages) else None
        public = self.public_hashes[hash_index] if hash_index < len(self.public_hashes) else None
        return name, public

    def reference(self, package_index: int) -> tuple[str, object]:
        """
        What an `FPackageIndex` in export data points at: `("export", Export)`,
        `("import", (package, hash))`, `("script", path)` or `("null", None)`.
        """
        if package_index == 0:
            return "null", None
        if package_index > 0:
            number = package_index - 1
            return ("export", self.exports[number]) if number < len(self.exports) else ("null", None)
        number = -package_index - 1
        if number >= len(self.imports):
            return "null", None
        index = self.imports[number]
        kind = index >> 62
        if kind == PACKAGE:
            return "import", self.package_import(index)
        if kind == SCRIPT:
            return "script", self.describe(index)
        return "null", None

    def outer_export(self, export: Export) -> Export | None:
        if export.outer >> 62 != EXPORT:
            return None
        number = export.outer & _INDEX_MASK
        return self.exports[number] if number < len(self.exports) else None


class Scripts:
    """The game's script objects - C++ classes and their defaults - from `global.utoc`."""

    def __init__(self, store: Store) -> None:
        self.objects: dict[int, tuple[str, int]] = {}
        container = store.global_container
        if container is None:
            return
        entry = next((i for i, (_c, _n, kind) in enumerate(container.chunk_ids) if kind == SCRIPT_OBJECTS), None)
        if entry is None:
            return
        data = container.read(entry)
        names, at = name_batch(data, 0)
        (count,) = struct.unpack_from("<i", data, at)
        at += 4
        for i in range(count):
            name_index, number, global_index, outer, _cdo = struct.unpack_from("<IIQQQ", data, at + 32 * i)
            name = names[name_index & 0x3FFFFFFF]
            self.objects[global_index] = (name if number == 0 else f"{name}_{number - 1}", outer)

    @lru_cache(maxsize=None)
    def path(self, index: int) -> str:
        parts = []
        seen = 0
        while index in self.objects and seen < 16:
            name, outer = self.objects[index]
            parts.append(name)
            if outer >> 62 == NULL:
                break
            index = outer
            seen += 1
        parts.reverse()
        if not parts:
            return ""
        return parts[0] + ("." + ".".join(parts[1:]) if len(parts) > 1 else "")


class Packages:
    """Packages of a store, read once each, and references followed across them."""

    def __init__(self, store: Store, *, mount: str = "") -> None:
        self.store = store
        self.scripts = Scripts(store)
        self._cache: dict[str, Package | None] = {}
        # `/Game/...` names map onto the project's Content folder.
        self.content = mount or next(
            (p.split("/content/")[0] + "/content/" for p in store.files if "/content/" in p), ""
        )

    def open(self, path: str) -> Package | None:
        key = path.lower()
        if key not in self._cache:
            try:
                self._cache[key] = Package(self.store.read_file(path), path, self.scripts)
            except (FileNotFoundError, ZenError, struct.error, IndexError):
                self._cache[key] = None
        return self._cache[key]

    def file_of(self, package_name: str) -> str | None:
        """`/Game/A/B` -> the `.uasset` or `.umap` file of that package."""
        if not package_name.startswith("/Game/"):
            return None
        stem = self.content + package_name[len("/Game/") :].lower()
        for suffix in (".uasset", ".umap"):
            if stem + suffix in self.store.files:
                return stem + suffix
        return None

    def resolve(self, package: Package, package_index: int) -> tuple[Package, Export] | None:
        """An object reference in `package`'s data, followed to the export it names."""
        kind, target = package.reference(package_index)
        if kind == "export":
            return package, target
        if kind != "import":
            return None
        name, public = target
        if not name or public is None:
            return None
        path = self.file_of(name)
        other = self.open(path) if path else None
        if other is None:
            return None
        for export in other.exports:
            if export.public_hash == public:
                return other, export
        return None
