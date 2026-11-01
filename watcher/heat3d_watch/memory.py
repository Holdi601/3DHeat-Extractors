"""
Reading a single-player game's memory: read only, by a profile, for one build.

What is here can only read. The process is opened with `PROCESS_VM_READ |
PROCESS_QUERY_LIMITED_INFORMATION` and nothing else; there is no write, no
allocation in the other process, no thread started in it, nothing loaded into
it. `safety.py` decides whether a game may be opened at all, before it is.

A profile (`<data>/profiles/<game>.json`) says, for one game:

    {
      "schema": "heat3d.memprofile/1",
      "id": "mygame", "game": "My Game",
      "exe": "MyGame.exe",
      "online": false,                      # can it be played online at all?
      "versions": [ { "label": "1.06", "size": 123456789 } ],   # main module file size;
                                                                # or "sha256": "..."
      "anchors": { "World": { "aob": "48 8B 05 ?? ?? ?? ?? 48 85 C0", "rip": [3, 7] } },
      "values": {
        "pos": { "type": "vec3f", "base": "World", "chain": ["0x10", "0x68", "0x70"] },
        "hp":  { "type": "i32",   "base": "World", "chain": ["0x10", "0x138"] },
        "map": { "type": "u32",   "base": "World", "chain": ["0x10", "0x1C0"] }
      },
      "events": [ { "event": "death", "value": "hp", "when": "to_zero" },
                  { "event": "zone",  "value": "map", "when": "change" } ],
      "axes": { "up": "y", "units": "metres" },
      "rate": 10,
      "provenance": [ { "source": "https://...", "verified_on": "1.06" } ]
    }

An anchor is found by its byte pattern in the main module (`??` any byte); `rip`
turns the instruction it points at into the address it refers to (offset of the
32-bit displacement, length of the instruction). A value is read by following
its `chain` of offsets from its anchor: every offset but the last is a pointer
dereferenced. The version must match by file size or SHA-256 of the
executable before anything is read - an offset from one build reads garbage on
the next, and garbage looks just like numbers.

No profiles ship with the tools: offsets have to be checked on the build a
player has, and a profile that was right last patch is wrong this one. The
format is here so a player can write one for their game, from what they verify.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
#: The only access ever asked for.
READ_ACCESS = PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION

TYPES = {"i32": "<i", "u32": "<I", "f32": "<f", "f64": "<d", "i64": "<q", "u64": "<Q", "u8": "<B"}
VECTORS = {"vec3f": ("<3f", 12), "vec3d": ("<3d", 24)}


class ReadError(Exception):
    pass


class Reader:
    """A read-only view of another process's memory."""

    def __init__(self, pid: int):
        if not hasattr(ctypes, "windll"):
            raise ReadError("reading memory is only implemented on Windows")
        k32 = ctypes.windll.kernel32
        self._k32 = k32
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        k32.ReadProcessMemory.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = k32.OpenProcess(READ_ACCESS, False, pid)
        if not handle:
            raise ReadError(f"the game would not open for reading (error {ctypes.get_last_error() or k32.GetLastError()})")
        self.handle = handle
        self.pid = pid

    def read(self, address: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        got = ctypes.c_size_t(0)
        ok = self._k32.ReadProcessMemory(self.handle, ctypes.c_void_p(address), buf, size, ctypes.byref(got))
        if not ok or got.value != size:
            raise ReadError(f"could not read {size} bytes at {address:#x}")
        return buf.raw

    def pointer(self, address: int) -> int:
        return struct.unpack("<Q", self.read(address, 8))[0]

    def close(self) -> None:
        if getattr(self, "handle", None):
            self._k32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _ModuleEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("th32ModuleID", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("GlblcntUsage", ctypes.c_uint32),
        ("ProccntUsage", ctypes.c_uint32),
        ("modBaseAddr", ctypes.c_void_p),
        ("modBaseSize", ctypes.c_uint32),
        ("hModule", ctypes.c_void_p),
        ("szModule", ctypes.c_wchar * 256),
        ("szExePath", ctypes.c_wchar * 260),
    ]


@dataclass
class Module:
    name: str
    base: int
    size: int
    path: Path


def modules(pid: int) -> list[Module]:
    """
    The modules loaded in a process, from a Toolhelp snapshot.

    A snapshot rather than psutil's memory map: that opens the process with
    query-information rights of its own, and nothing here asks for more than
    `READ_ACCESS`. A protected process refuses the snapshot, which is the
    answer wanted.
    """
    if not hasattr(ctypes, "windll"):
        raise ReadError("only implemented on Windows")
    k32 = ctypes.windll.kernel32
    TH32CS_SNAPMODULE, TH32CS_SNAPMODULE32 = 0x8, 0x10
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    k32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    k32.Module32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ModuleEntry)]
    k32.Module32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ModuleEntry)]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    if not snap or snap == ctypes.c_void_p(-1).value:
        raise ReadError("the game's modules could not be listed")
    out: list[Module] = []
    try:
        entry = _ModuleEntry()
        entry.dwSize = ctypes.sizeof(_ModuleEntry)
        ok = k32.Module32FirstW(snap, ctypes.byref(entry))
        while ok:
            out.append(Module(entry.szModule, int(entry.modBaseAddr or 0), int(entry.modBaseSize), Path(entry.szExePath)))
            ok = k32.Module32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return out


def main_module(pid: int, exe: Path) -> Module:
    """The game's executable image among its modules."""
    for m in modules(pid):
        if m.name.lower() == exe.name.lower():
            return m
    raise ReadError(f"{exe.name} is not among the game's modules")


def read_image(reader: Reader, module: Module, block: int = 0x10000) -> bytes:
    """The whole image as mapped, a block at a time; a block that cannot be read is left as zeros."""
    out = bytearray(module.size)
    for at in range(0, module.size, block):
        size = min(block, module.size - at)
        try:
            out[at : at + size] = reader.read(module.base + at, size)
        except ReadError:
            continue
    return bytes(out)


def parse_pattern(text: str) -> re.Pattern[bytes]:
    """`48 8B 05 ?? ?? ?? ??` as a byte regex; `?` or `??` is any byte."""
    parts = []
    for tok in text.split():
        if tok in ("?", "??"):
            parts.append(b".")
        else:
            parts.append(re.escape(bytes([int(tok, 16)])))
    return re.compile(b"".join(parts), re.DOTALL)


def find(image: bytes, pattern: str) -> int:
    """Offset of the one place `pattern` occurs. Several places is as bad as none."""
    rx = parse_pattern(pattern)
    hits = [m.start() for m in rx.finditer(image)]
    if len(hits) != 1:
        raise ReadError(f"pattern found {len(hits)} times, wanted once: {pattern}")
    return hits[0]


def rip_target(image: bytes, at: int, disp: int, length: int) -> int:
    """The address a RIP-relative instruction at `at` refers to, as an offset into the image."""
    (rel,) = struct.unpack_from("<i", image, at + disp)
    return at + length + rel


@dataclass
class Profile:
    id: str
    game: str
    exe: str
    online: bool
    versions: list[dict[str, Any]]
    anchors: dict[str, dict[str, Any]]
    values: dict[str, dict[str, Any]]
    events: list[dict[str, Any]] = field(default_factory=list)
    axes: dict[str, Any] = field(default_factory=dict)
    rate: float = 10.0
    provenance: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Profile":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != "heat3d.memprofile/1":
            raise ValueError(f"{path.name}: not a heat3d.memprofile/1 profile")
        for need in ("id", "game", "exe", "versions", "anchors", "values"):
            if need not in raw:
                raise ValueError(f"{path.name}: no {need!r}")
        if "online" not in raw:
            raise ValueError(f"{path.name}: say whether the game can be played online (\"online\": true/false)")
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in raw.items() if k in known})

    def version_of(self, exe: Path) -> str | None:
        """The label of the build `exe` is, or None when no listed version matches."""
        size = exe.stat().st_size
        digest: str | None = None
        for v in self.versions:
            if "size" in v and int(v["size"]) != size:
                continue
            if "sha256" in v:
                if digest is None:
                    digest = hashlib.sha256(exe.read_bytes()).hexdigest()
                if digest.lower() != str(v["sha256"]).lower():
                    continue
            if "size" not in v and "sha256" not in v:
                continue
            return str(v.get("label", "?"))
        return None


class Resolver:
    """Anchors found once, values read each poll."""

    def __init__(self, reader: Reader, profile: Profile, base: int, image: bytes):
        self.reader = reader
        self.profile = profile
        self.base = base
        self.anchors: dict[str, int] = {}
        for name, a in profile.anchors.items():
            if "aob" in a:
                at = find(image, a["aob"])
                if "rip" in a:
                    disp, length = a["rip"]
                    at = rip_target(image, at, int(disp), int(length))
                at += int(str(a.get("offset", 0)), 0)
                self.anchors[name] = base + at
            elif "module_offset" in a:
                self.anchors[name] = base + int(str(a["module_offset"]), 0)
            else:
                raise ValueError(f"anchor {name!r} has neither aob nor module_offset")

    def address(self, value: dict[str, Any]) -> int:
        addr = self.anchors[value["base"]]
        chain = [int(str(c), 0) for c in value.get("chain", [])]
        if not chain:
            return addr
        # Every link but the last is a pointer to follow.
        addr = self.reader.pointer(addr)
        for off in chain[:-1]:
            addr = self.reader.pointer(addr + off)
            if addr == 0:
                raise ReadError("a null pointer on the way: the game is between levels or in a menu")
        return addr + chain[-1]

    def read(self, name: str) -> Any:
        value = self.profile.values[name]
        kind = value["type"]
        addr = self.address(value)
        if kind in VECTORS:
            fmt, size = VECTORS[kind]
            return struct.unpack(fmt, self.reader.read(addr, size))
        if kind in TYPES:
            fmt = TYPES[kind]
            return struct.unpack(fmt, self.reader.read(addr, struct.calcsize(fmt)))[0]
        raise ValueError(f"unknown type {kind!r}")
