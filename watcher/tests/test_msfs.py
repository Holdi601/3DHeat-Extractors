"""MSFS: SimConnect messages built from SimConnect.h's layouts, decoded and turned into a flight.

The live part - loading SimConnect.dll and talking to the sim - is not tested
here: there is no sim to talk to. A stand-in client hands the adapter the bytes
SimConnect_GetNextDispatch would.
"""

from __future__ import annotations

import json
import struct
from dataclasses import replace

import pytest

from heat3d_watch.adapters.base import Context
from heat3d_watch.adapters.msfs import (
    ADAPTERS,
    DEFINE_ID,
    EVENTS,
    MAX_PATH,
    RECV_EVENT,
    RECV_EVENT_FILENAME,
    RECV_EXCEPTION,
    RECV_OPEN,
    RECV_QUIT,
    RECV_SIMOBJECT_DATA,
    SIMVARS,
    candidates,
    decode,
)
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

VERSION = 6  # dwVersion: whatever the sim's protocol is; not read


def recv(rid: int, body: bytes) -> bytes:
    """SIMCONNECT_RECV {dwSize, dwVersion, dwID} + the message's own fields, packed (pack 1)."""
    return struct.pack("<III", 12 + len(body), VERSION, rid) + body


def data(**v: float) -> bytes:
    values = [v.get(name, 0.0) for _, _, name in SIMVARS]
    # dwRequestID, dwObjectID, dwDefineID, dwFlags, dwentrynumber, dwoutof, dwDefineCount, then FLOAT64s
    head = struct.pack("<7I", 1, 0, DEFINE_ID, 0, 1, 1, len(values))
    return recv(RECV_SIMOBJECT_DATA, head + struct.pack(f"<{len(values)}d", *values))


def event(name: str) -> bytes:
    eid = next(k for k, n in EVENTS.items() if n == name)
    return recv(RECV_EVENT, struct.pack("<III", 0xFFFFFFFF, eid, 0))


def flight_loaded(path: str) -> bytes:
    eid = next(k for k, n in EVENTS.items() if n == "FlightLoaded")
    name = path.encode().ljust(MAX_PATH, b"\0")
    return recv(RECV_EVENT_FILENAME, struct.pack("<III", 0xFFFFFFFF, eid, 0) + name + struct.pack("<I", 0))


def test_messages_decode_per_the_sdk_layouts():
    m = decode(data(lat=47.5, lon=8.25, alt=500.0, g=1.5, on_ground=1.0))
    assert m["kind"] == "data" and m["define"] == DEFINE_ID
    assert m["vars"]["lat"] == 47.5 and m["vars"]["lon"] == 8.25 and m["vars"]["g"] == 1.5
    assert m["vars"]["on_ground"] == 1.0
    open_msg = recv(RECV_OPEN, b"KittyHawk".ljust(256, b"\0") + struct.pack("<8I", 11, 0, 62651, 3, 11, 0, 62651, 3) + b"\0" * 8)
    m = decode(open_msg)
    assert m["kind"] == "open" and m["application"] == "KittyHawk" and m["version_major"] == 11
    assert decode(event("Crashed"))["name"] == "Crashed"
    m = decode(flight_loaded(r"C:\some\folder\flight.FLT"))
    assert m["name"] == "FlightLoaded" and m["filename"].endswith("flight.FLT")
    m = decode(recv(RECV_EXCEPTION, struct.pack("<III", 7, 12, 3)))
    assert (m["kind"], m["exception"], m["send_id"], m["index"]) == ("exception", 7, 12, 3)
    assert decode(recv(RECV_QUIT, b""))["kind"] == "quit"
    with pytest.raises(ValueError):
        decode(b"\x00" * 8)
    with pytest.raises(ValueError):
        decode(struct.pack("<III", 400, 6, RECV_SIMOBJECT_DATA) + b"\0" * 30)


class FakeClient:
    """What SimConnect_GetNextDispatch would hand out, a poll at a time."""

    open = True

    def __init__(self):
        self.queue: list[list[bytes]] = []

    def messages(self):
        return self.queue.pop(0) if self.queue else []

    def close(self):
        self.open = False


def test_a_flight_from_simconnect_messages(tmp_path):
    a = replace(ADAPTERS[0])
    ctx = Context(Settings(data=str(tmp_path)), tmp_path)
    c = FakeClient()
    a._client = c
    lat, lon = 47.0, 8.0
    step = 0.0001
    polls = [[flight_loaded(r"C:\flights\a.FLT"), event("SimStart")]]
    for i in range(4):
        polls.append([data(lat=lat + i * step, lon=lon, alt=400, on_ground=1, g=1.0)])
    # A takeoff and a touchdown inside one poll's frames: both still seen.
    polls.append([
        data(lat=lat + 4 * step, lon=lon, alt=405, on_ground=0, g=1.3),
        data(lat=lat + 4.5 * step, lon=lon, alt=410, on_ground=0, g=1.8),
    ])
    polls.append([data(lat=lat + 5 * step, lon=lon, alt=420, on_ground=0, g=1.1)])
    polls.append([data(lat=lat + 6 * step, lon=lon, alt=400, on_ground=1, g=2.4), b"\x01\x02"])
    polls.append([event("Crashed"), event("SimStop")])
    polls.append([event("SimStart"), data(lat=lat, lon=lon, alt=400, on_ground=1)])
    polls.append([recv(RECV_QUIT, b"")])
    c.queue = polls
    folder = None
    for _ in range(len(polls)):
        a.poll(ctx)
        if a._session is not None:
            folder = a._session.folder
    assert a._session is None  # the sim quit: the session is closed
    rows = read_rows(folder)
    events = [r["event"] for r in rows if r["event"] != "position"]
    assert events == ["session_start", "takeoff", "landing", "crash", "session_end", "session_start", "session_end"]
    assert rows[0]["flight_file"] == "a.FLT"
    pos = [r for r in rows if r["event"] == "position"]
    assert pos[0]["x"] == pytest.approx(0, abs=1e-6) and pos[3]["z"] == pytest.approx(3 * 11.12, abs=0.3)
    assert [p["y"] for p in pos[4:7]] == [405, 410, 420]
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert st["takeoffs"] == 1 and st["landings"] == 1 and st["crashes"] == 1 and st["flights"] == 2
    assert st["max_g"] == pytest.approx(2.4) and st["bad_packets"] == 1
    assert meta["origin"] == {"lat": 47.0, "lon": 8.0}


def test_the_dll_is_looked_for_in_the_sim_and_its_sdk(tmp_path, monkeypatch):
    monkeypatch.setenv("MSFS2024_SDK", str(tmp_path / "sdk24"))
    monkeypatch.setenv("MSFS_SDK", str(tmp_path / "sdk20"))
    paths = [str(p) for p in candidates(True, tmp_path / "exe")]
    assert paths[0] == str(tmp_path / "exe" / "SimConnect.dll")
    i24 = paths.index(str(tmp_path / "sdk24" / "SimConnect SDK" / "lib" / "SimConnect.dll"))
    i20 = paths.index(str(tmp_path / "sdk20" / "SimConnect SDK" / "lib" / "SimConnect.dll"))
    assert i24 < i20  # the 2020 SDK only as a fallback for 2024
    assert not any(p.endswith("SimConnect_internal.dll") for p in paths)
    assert str(tmp_path / "sdk24" / "SimConnect SDK" / "lib" / "SimConnect.dll") not in [str(p) for p in candidates(False)]


def test_without_a_dll_the_adapter_says_why(monkeypatch):
    import heat3d_watch.adapters.msfs as msfs

    monkeypatch.setattr(msfs, "find_simconnect", lambda *a, **k: None)
    a = replace(ADAPTERS[1], _missing=(0.0, None))
    why = a.missing()
    assert why and "SimConnect.dll" in why and "MSFS 2024 SDK" in why
