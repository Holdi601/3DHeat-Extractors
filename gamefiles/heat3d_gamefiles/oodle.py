"""
Oodle decompression, through a library the machine already has.

Most shipped Unreal content is Oodle-compressed, so without this the pak reader
lists files it cannot open. Oodle is proprietary (RAD Game Tools, now Epic), and
there is no redistributable build and no Python implementation — so this module
**finds** a library rather than carrying one.

Why not bundle it
-----------------
The licence does not permit it. `oo2core_*.dll` is licensed to the product that
ships it, which is why FModel and UModel also ask the user to point at one. That
is not a workaround; it is the arrangement the licence contemplates, and it keeps
this exporter distributable on its own terms.

Where one comes from
--------------------
Any game on the machine that ships the DLL has one. Unreal 5 titles often link
Oodle statically and ship no DLL at all, so the one found may belong to a
different game than the archive being read — the format is the same either way.
The Unreal Engine install itself carries one under
`Engine/Binaries/ThirdParty/Oodle`, which is the tidiest source for a developer.

If none is found, that is reported plainly rather than guessed around: an entry
that cannot be decompressed is better as an error than as plausible rubbish.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

#: Newest first, so a modern archive gets a matching decoder. The format is
#: backward compatible, so an older library usually still works.
_DLL_NAMES = tuple(f"oo2core_{v}_win64.dll" for v in range(9, 2, -1))

#: Checked in order. Deliberately not a recursive scan of every drive: that
#: takes minutes and surprises the user. `HEAT3D_OODLE` short-circuits it.
_SEARCH_HINTS = (
    r"C:\Program Files\Epic Games",
    r"C:\Program Files (x86)\Steam\steamapps\common",
    r"C:\Program Files\Unreal Engine",
)

_library: ctypes.CDLL | None = None
_library_path: Path | None = None


class OodleUnavailable(Exception):
    """No Oodle library was found, and the message says how to supply one."""


def find_oodle(extra: str | os.PathLike[str] | None = None) -> Path | None:
    """
    Locate an Oodle library, or return None.

    Order: an explicit path, then `HEAT3D_OODLE`, then a shallow look through the
    usual game and engine directories.
    """
    candidates: list[Path] = []

    for given in (extra, os.environ.get("HEAT3D_OODLE")):
        if not given:
            continue
        path = Path(given)
        if path.is_dir():
            candidates.extend(path / name for name in _DLL_NAMES)
        else:
            candidates.append(path)

    # Every Steam library, not only the default one: a second drive is where
    # most games live, and the only copy on a machine is often there.
    hints = list(_SEARCH_HINTS)
    try:
        from .forzainstall import steam_libraries

        hints += [str(Path(library) / "steamapps" / "common") for library in steam_libraries()]
    except OSError:
        pass
    seen: set[str] = set()
    for hint in hints:
        root = Path(hint)
        if str(root).lower() in seen or not root.is_dir():
            continue
        seen.add(str(root).lower())
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            # A game's own folder, and the two places Unreal keeps it.
            for where in (
                child,
                child / "Engine" / "Binaries" / "ThirdParty" / "Oodle" / "Win64",
                child / "Engine" / "Binaries" / "Win64",
            ):
                candidates.extend(where / name for name in _DLL_NAMES)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_oodle(path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    """Load the library once and keep it; raise `OodleUnavailable` if there is none."""
    global _library, _library_path

    if _library is not None and path is None:
        return _library

    found = Path(path) if path else find_oodle()
    if found is None or not found.is_file():
        raise OodleUnavailable(
            "no Oodle library found. Most shipped Unreal content is "
            "Oodle-compressed and it cannot be decompressed without one.\n"
            "  Point at a copy with HEAT3D_OODLE=<path to oo2core_9_win64.dll>.\n"
            "  Any installed game or engine that ships the DLL has one; it is "
            "proprietary, so this tool does not include it."
        )

    library = ctypes.CDLL(str(found))
    decompress = library.OodleLZ_Decompress
    decompress.restype = ctypes.c_ssize_t
    decompress.argtypes = [
        ctypes.c_void_p,  # compressed bytes
        ctypes.c_ssize_t,  # their length
        ctypes.c_void_p,  # output buffer
        ctypes.c_ssize_t,  # its length
        ctypes.c_int,  # fuzz safe
        ctypes.c_int,  # check CRC
        ctypes.c_int,  # verbosity
        ctypes.c_void_p,  # decode buffer base
        ctypes.c_ssize_t,  # its size
        ctypes.c_void_p,  # progress callback
        ctypes.c_void_p,  # its user data
        ctypes.c_void_p,  # scratch memory
        ctypes.c_ssize_t,  # its size
        ctypes.c_int,  # thread phase
    ]

    _library, _library_path = library, found
    return library


def library_path() -> Path | None:
    """Which library is loaded, for the interface to report."""
    return _library_path


def decompress_oodle(payload: bytes, expected: int) -> bytes:
    """
    Decompress one block.

    Oodle cannot infer the output length, so the caller must supply it — the pak
    index knows it, which is why it is a parameter rather than a guess.
    """
    library = load_oodle()
    out = ctypes.create_string_buffer(expected)
    written = library.OodleLZ_Decompress(
        payload,
        len(payload),
        out,
        expected,
        1,  # fuzz safe: the input is untrusted, so bounds-check it
        0,  # no CRC check; the pak's own SHA-1 already covers the bytes
        0,  # silent
        None,
        0,
        None,
        None,
        None,
        0,
        3,  # unthreaded
    )
    if written != expected:
        raise OodleUnavailable(
            f"Oodle returned {written} bytes where {expected} were expected"
        )
    return out.raw[:expected]
