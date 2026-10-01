"""
Reading Unreal `.pak` archives.

The third route to a level's geometry, alongside an engine export and
reconstruction from video. An engine export is exact but needs the project open;
reconstruction needs only a screen but is an estimate. This one needs the shipped
game and gives exact geometry — for anyone who has the build but not the project.

What this does and does not do
------------------------------
It **reads containers**. Parsing an archive format is ordinary file handling, and
it is what FModel, UModel and AssetRipper have done in the open for years.

It **contains no key material and no key recovery**. An encrypted archive opens
only if the caller supplies a key they already hold. Nothing here ships a key,
searches an executable for one, or reads another process's memory. That line is
deliberate: reading a file and defeating an access control are different acts,
and the second reaches the tool itself, not only what someone does with it.

What anyone points it at stays their own responsibility. A developer reading
their own studio's build is the unambiguous case; someone else's title carries
that publisher's terms, and no design choice here discharges them.

Layout
------
The footer is last and is read backwards from the end of the file. Its field
order matters and is easy to get wrong, because the encryption flag sits *before*
the magic number rather than after it::

    [encryption key GUID : 16]  [encrypted index : 1]
    [magic : 4]  [version : 4]  [index offset : 8]  [index size : 8]
    [index SHA-1 : 20]  [compression method names : 32 bytes each]

Taking the flag from after the magic instead lands on the first byte of the
string "Zlib", which is non-zero — so every archive reports itself encrypted, and
every method name loses its first letter and reads as "lib" or "odle". That
failure is quiet and plausible rather than loud, which is why the order is
spelled out here.

The index holds a mount point, two lookup structures, and a block of *encoded*
entries: a bit-packed form where a leading flags word says how wide each of the
numbers after it is, so a small file costs far less than its full 53-byte record.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

#: Little-endian `0x5A6F12E1`, near the end of the file.
PAK_MAGIC = 0x5A6F12E1

#: Versions this reader understands, covering Unreal 4.22 through 5.x. Earlier
#: ones store the index as a flat record list rather than the encoded form, and
#: claiming support with no file to test against would be worse than refusing.
SUPPORTED = (8, 9, 10, 11, 12)

#: Compression method names sit in a fixed-width table; at most this many.
MAX_METHODS = 5
METHOD_NAME_LEN = 32

#: Index 0 in that table is not a method — it means the entry is stored.
NO_COMPRESSION = 0

#: Encrypted payloads are padded to the AES block size, so a block's stored
#: length and the stride to the next block differ.
AES_BLOCK = 16


class UnsupportedPak(Exception):
    """Not a pak this reader can read; the message says why."""


class EncryptedPak(Exception):
    """Needs a key that was not supplied."""


@dataclass(frozen=True)
class PakInfo:
    """What the footer says."""

    version: int
    index_offset: int
    index_size: int
    encrypted_index: bool
    methods: tuple[str, ...]
    #: All zero unless the archive was built against a named encryption key.
    key_guid: bytes = b"\0" * 16

    def method_name(self, index: int) -> str:
        """Name for a method index. Index 0 is 'stored'."""
        if index == NO_COMPRESSION:
            return "none"
        if 0 < index <= len(self.methods):
            return self.methods[index - 1]
        return f"unknown({index})"


@dataclass(frozen=True)
class PakEntry:
    """One file inside the archive."""

    path: str
    offset: int
    size: int
    uncompressed_size: int
    method: int
    encrypted: bool
    block_size: int = 0
    #: (start, end) of each compressed block, relative to the entry's offset.
    blocks: tuple[tuple[int, int], ...] = ()

    @property
    def compressed(self) -> bool:
        return self.method != NO_COMPRESSION


def read_footer(path: str | Path) -> PakInfo:
    """Parse the footer, or raise `UnsupportedPak` rather than guess."""
    size = os.path.getsize(path)
    window = min(size, 1024)
    if window < 44:
        raise UnsupportedPak(f"{Path(path).name} is too small to be a pak")

    with open(path, "rb") as handle:
        handle.seek(size - window)
        tail = handle.read(window)

    at = tail.rfind(struct.pack("<I", PAK_MAGIC))
    if at < 0:
        raise UnsupportedPak(
            f"{Path(path).name} has no pak footer. A UE 4.26+ title may ship its "
            "content as IoStore (.utoc/.ucas) instead — see iostore.py."
        )

    version = struct.unpack_from("<I", tail, at + 4)[0]
    # 'Frozen index' builds set a high bit alongside the number. The version is
    # the low byte; the flag is not this reader's business.
    version &= 0xFF
    if version not in SUPPORTED:
        raise UnsupportedPak(
            f"pak version {version}; this reader handles "
            f"{min(SUPPORTED)}–{max(SUPPORTED)}"
        )

    index_offset, index_size = struct.unpack_from("<qq", tail, at + 8)

    # Before the magic, not after it. See the module docstring.
    encrypted = bool(tail[at - 1]) if at >= 1 else False
    key_guid = bytes(tail[at - 17 : at - 1]) if at >= 17 else b"\0" * 16

    methods: list[str] = []
    table = at + 24 + 20
    for i in range(MAX_METHODS):
        start = table + i * METHOD_NAME_LEN
        raw = tail[start : start + METHOD_NAME_LEN]
        if len(raw) < METHOD_NAME_LEN:
            break
        name = raw.split(b"\x00")[0].decode("ascii", "replace")
        if not name:
            break
        methods.append(name)

    if not (0 < index_offset < size) or not (0 < index_size <= size):
        raise UnsupportedPak(
            f"{Path(path).name}: the footer puts the index at {index_offset}+"
            f"{index_size}, outside a {size}-byte file, so it did not parse as claimed"
        )

    return PakInfo(
        version=version,
        index_offset=index_offset,
        index_size=index_size,
        encrypted_index=encrypted,
        methods=tuple(methods),
        key_guid=key_guid,
    )


class _Reader:
    """A little-endian cursor, because the index is densely packed."""

    def __init__(self, data: bytes):
        self.data = data
        self.at = 0

    def _take(self, fmt: str, width: int):
        if self.at + width > len(self.data):
            raise UnsupportedPak("index ended mid-record; it may be encrypted")
        value = struct.unpack_from(fmt, self.data, self.at)[0]
        self.at += width
        return value

    def u32(self) -> int:
        return self._take("<I", 4)

    def i32(self) -> int:
        return self._take("<i", 4)

    def i64(self) -> int:
        return self._take("<q", 8)

    def skip(self, count: int) -> None:
        self.at += count

    def string(self) -> str:
        """
        An FString: a length, then bytes. A negative length means UTF-16 and
        counts characters rather than bytes. Both include a trailing NUL.
        """
        length = self.i32()
        if length == 0:
            return ""
        if length < 0:
            count = -length * 2
            raw = self.data[self.at : self.at + count]
            self.at += count
            return raw.decode("utf-16-le", "replace").rstrip("\x00")
        if length > len(self.data):
            raise UnsupportedPak(
                f"index declares a {length}-byte string, which is longer than the "
                "index itself; it is almost certainly encrypted"
            )
        raw = self.data[self.at : self.at + length]
        self.at += length
        return raw.decode("utf-8", "replace").rstrip("\x00")


def _serialized_size(method: int, block_count: int) -> int:
    """
    Bytes of the record preceding an entry's payload in the data section.

    Every entry is written twice: bit-packed in the index, and in full
    immediately before its own bytes. The payload begins after this second copy,
    so a wrong answer here reads the data shifted by a few bytes — which fails
    decompression, or worse, succeeds into rubbish.
    """
    size = 8 + 8 + 8 + 20  # offset, size, uncompressed size, SHA-1
    size += 4  # compression method index
    size += 1 + 4  # flags, compression block size
    if method != NO_COMPRESSION:
        size += 4 + block_count * 16  # count, then an (int64, int64) pair each
    return size


def _decode_entry(blob: bytes, at: int, path: str) -> PakEntry:
    """
    Decode one bit-packed entry.

    The flags word says what follows: which numbers are 32 rather than 64 bit,
    the compression method, how many blocks there are, and the block size as a
    multiple of 2048. Every field has to be taken in order, because each one's
    width decides where the next begins.
    """
    (flags,) = struct.unpack_from("<I", blob, at)
    cursor = at + 4

    method = (flags >> 23) & 0x3F
    encrypted = bool(flags & (1 << 22))
    block_count = (flags >> 6) & 0xFFFF

    def number(is_32: bool) -> int:
        nonlocal cursor
        if is_32:
            (value,) = struct.unpack_from("<I", blob, cursor)
            cursor += 4
        else:
            (value,) = struct.unpack_from("<q", blob, cursor)
            cursor += 8
        return value

    # Six bits hold the block size in units of 2048, which tops out at 126 KiB.
    # All ones is an escape: the real size did not fit and follows as a plain
    # uint32 — and it comes *here*, before the offset, not after the sizes.
    #
    # Getting that position wrong is survivable in a small archive, because a
    # 256 KiB block size only appears in large ones, and every entry that does
    # fit in six bits still decodes. In a 27 GB pak it shifted 7% of entries by
    # four bytes and gave them offsets in the exabytes.
    packed = flags & 0x3F
    block_size = packed << 11
    if packed == 0x3F:
        (block_size,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4

    offset = number(bool(flags & (1 << 31)))
    uncompressed = number(bool(flags & (1 << 30)))
    size = number(bool(flags & (1 << 29))) if method != NO_COMPRESSION else uncompressed

    blocks: list[tuple[int, int]] = []
    if block_count:
        # Offsets run from the entry's own position and start after the full
        # record that precedes the payload.
        start = _serialized_size(method, block_count)
        stride = AES_BLOCK if encrypted else 1
        for _ in range(block_count):
            if block_count == 1:
                length = size
            else:
                (length,) = struct.unpack_from("<I", blob, cursor)
                cursor += 4
            blocks.append((start, start + length))
            start += -(-length // stride) * stride

    return PakEntry(
        path=path,
        offset=offset,
        size=size,
        uncompressed_size=uncompressed,
        method=method,
        encrypted=encrypted,
        block_size=block_size,
        blocks=tuple(blocks),
    )


class PakArchive:
    """An opened archive: what is in it, and the bytes of any one entry."""

    def __init__(self, path: str | Path, *, aes_key: bytes | None = None):
        self.path = Path(path)
        self.info = read_footer(self.path)
        #: Supplied by the caller, never derived. See the module docstring.
        self._key = aes_key
        self.mount = ""
        self.entries: dict[str, PakEntry] = {}
        self._load_index()

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, path: str) -> bool:
        return path in self.entries

    # -- index ------------------------------------------------------------

    def _read_at(self, offset: int, size: int, *, decrypt: bool) -> bytes:
        with open(self.path, "rb") as handle:
            handle.seek(offset)
            # AES works in whole blocks, and a declared size may stop part-way
            # through the last one, so read up to the block boundary and trim.
            want = -(-size // AES_BLOCK) * AES_BLOCK if decrypt else size
            raw = handle.read(want)
        if decrypt:
            raw = _decrypt(raw, self._key)
        return raw[:size]

    def _load_index(self) -> None:
        if self.info.encrypted_index and self._key is None:
            raise EncryptedPak(
                f"{self.path.name} has an encrypted index. Pass the AES key as "
                "`aes_key=` if you hold one; this tool does not recover keys."
            )

        raw = self._read_at(
            self.info.index_offset,
            self.info.index_size,
            decrypt=self.info.encrypted_index,
        )
        index = _Reader(raw)
        self.mount = index.string()
        declared = index.i32()

        index.skip(8)  # path hash seed
        if index.i32():  # has path hash index
            index.skip(8 + 8 + 20)
        if not index.i32():  # has full directory index
            raise UnsupportedPak(
                f"{self.path.name} carries only the path-hash index, so file names "
                "cannot be recovered from it — they were hashed away at build time"
            )
        directory_offset = index.i64()
        directory_size = index.i64()
        index.skip(20)

        encoded_size = index.i32()
        encoded = raw[index.at : index.at + encoded_size]

        directory_raw = self._read_at(
            directory_offset, directory_size, decrypt=self.info.encrypted_index
        )
        directory = _Reader(directory_raw)

        for _ in range(directory.i32()):
            folder = directory.string()
            for _ in range(directory.i32()):
                name = directory.string()
                where = directory.i32()
                # -1 marks an entry held in the index's overflow list rather than
                # the encoded block. Those are rare; they are skipped, not faked.
                if where < 0 or where + 4 > len(encoded):
                    continue
                full = f"{self.mount}{folder}{name}".replace("\\", "/")
                self.entries[full] = _decode_entry(encoded, where, full)

        if declared and not self.entries:
            raise UnsupportedPak(
                f"{self.path.name} declares {declared} entries but none decoded"
            )

    # -- reading ----------------------------------------------------------

    def list(self, pattern: str | None = None) -> list[str]:
        """Paths in the archive, optionally filtered by a case-insensitive substring."""
        paths = sorted(self.entries)
        if pattern:
            needle = pattern.lower()
            paths = [p for p in paths if needle in p.lower()]
        return paths

    def read(self, path: str) -> bytes:
        """The bytes of one entry, decrypted and decompressed."""
        entry = self.entries.get(path)
        if entry is None:
            raise KeyError(path)
        if entry.encrypted and self._key is None:
            raise EncryptedPak(f"{path} is encrypted and no key was supplied")

        if not entry.compressed:
            header = _serialized_size(entry.method, 0)
            raw = self._read_at(
                entry.offset + header, entry.uncompressed_size, decrypt=entry.encrypted
            )
            return raw[: entry.uncompressed_size]

        name = self.info.method_name(entry.method)
        out = bytearray()
        remaining = entry.uncompressed_size
        for start, end in entry.blocks:
            # Every block decompresses to the entry's block size except the last,
            # which holds the remainder. Oodle cannot infer either, so the length
            # has to be carried down from the index rather than discovered.
            expected = min(entry.block_size or remaining, remaining)
            chunk = self._read_at(
                entry.offset + start, end - start, decrypt=entry.encrypted
            )
            block = _decompress(chunk, name, expected)
            out += block
            remaining -= len(block)
        return bytes(out[: entry.uncompressed_size])


def _decompress(payload: bytes, method: str, expected: int) -> bytes:
    lowered = method.lower()
    if lowered == "zlib":
        import zlib

        return zlib.decompress(payload)
    if lowered == "gzip":
        import gzip

        return gzip.decompress(payload)
    if lowered == "oodle":
        from .oodle import decompress_oodle

        return decompress_oodle(payload, expected)
    raise UnsupportedPak(f"compression method {method!r} is not supported")


def _decrypt(payload: bytes, key: bytes | None) -> bytes:
    """
    AES-256-ECB with a key the caller supplied.

    Takes the key as an argument and has no path to obtaining one, deliberately.
    """
    if key is None:
        raise EncryptedPak("no key supplied")
    if len(key) != 32:
        raise EncryptedPak(f"an AES-256 key is 32 bytes; this one is {len(key)}")
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise UnsupportedPak(
            "reading an encrypted archive needs the `cryptography` package"
        ) from exc
    decryptor = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    return decryptor.update(payload) + decryptor.finalize()
