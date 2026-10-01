"""
Forza "Data Out" UDP telemetry.

Forza Motorsport and Forza Horizon both ship an officially documented telemetry
feed: switch it on in the game's HUD/Gameplay settings, point it at a host and
port, and the game sends one UDP packet per simulation tick. Nothing is injected
and nothing is hooked — this is a feature, not a workaround, which is the whole
reason it is worth building on. It gives exact world position and orientation,
so for a Forza capture the camera path is *known* rather than estimated, and the
reconstruction inherits that accuracy.

Layout, and why it is discovered rather than declared
-----------------------------------------------------
The packet is a flat C struct in three known sizes:

    232  Motorsport "Sled"           — no position, telemetry only
    311  Motorsport "Dash"           — Sled plus the dash block
    324  Horizon (FH4/FH5) "Dash"    — same, with padding after the Sled block

The Sled block is stable and well established. The dash block's *start* in the
Horizon variant is not: the padding between the two sections is documented
inconsistently across every published description, and the arithmetic does not
close — a 79-byte dash block at offset 244 accounts for 323 of the 324 bytes.

Guessing that offset is the worst option available, because being wrong does not
fail. It yields finite, plausible-looking floats that are simply the wrong
fields, and the reconstruction is then built on a camera path that is quietly
garbage. So the offset is not assumed. Candidate bases are tried and checked
against a property the packet itself has to satisfy:

    the Sled block's velocity vector, and the dash block's scalar Speed,
    are the same quantity in metres per second

Those live in different sections, so they only agree when both are being read
correctly. One packet from a moving car identifies the layout with no ambiguity,
and `detect_layout()` reports which one it settled on and why. A stationary car
cannot distinguish them — every candidate reads zero — so detection holds off
until it sees motion rather than locking in a coin flip.
"""

from __future__ import annotations

import math
import socket
import struct
from dataclasses import dataclass
from typing import Iterator, Sequence

#: Packet sizes the game is known to emit.
SLED_LENGTH = 232
FM_DASH_LENGTH = 311
FH_DASH_LENGTH = 324

#: The dash block, from `PositionX` to the last byte of the packet.
DASH_BLOCK_LENGTH = 79

# --- Sled block, absolute offsets. Stable across all variants.
_IS_RACE_ON = 0
_TIMESTAMP_MS = 4
_ENGINE_MAX_RPM = 8
_ENGINE_IDLE_RPM = 12
_CURRENT_ENGINE_RPM = 16
_VELOCITY = 32  # three floats, world metres per second
_YAW = 56  # yaw, pitch, roll: three floats, radians

# --- Dash block, offsets relative to the block's own start.
_D_POSITION = 0  # three floats, world metres
_D_SPEED = 12
_D_POWER = 16
_D_TORQUE = 20
_D_BOOST = 40
_D_FUEL = 44
_D_DISTANCE = 48
_D_BEST_LAP = 52
_D_LAST_LAP = 56
_D_CURRENT_LAP = 60
_D_RACE_TIME = 64
_D_LAP_NUMBER = 68
_D_RACE_POSITION = 70
_D_ACCEL = 71
_D_BRAKE = 72
_D_GEAR = 75
_D_STEER = 76


@dataclass(frozen=True)
class ForzaFrame:
    """One simulation tick, in the terms the reconstruction needs."""

    is_race_on: bool
    timestamp_ms: int
    #: World position, metres.
    x: float
    y: float
    z: float
    #: Orientation, radians.
    yaw: float
    pitch: float
    roll: float
    #: Scalar speed, metres per second.
    speed: float
    #: World velocity, metres per second.
    vx: float
    vy: float
    vz: float
    distance_travelled: float
    race_time: float
    lap: int
    gear: int
    steer: float
    accel: float
    brake: float

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def moving(self) -> bool:
        return self.speed > 0.5


@dataclass(frozen=True)
class Layout:
    """Where the dash block starts, and how confident we are that it does."""

    packet_length: int
    dash_base: int
    #: Disagreement between |velocity| and Speed, in m/s, on the deciding packet.
    residual: float

    @property
    def name(self) -> str:
        if self.packet_length == FM_DASH_LENGTH:
            return "Motorsport dash"
        if self.packet_length == FH_DASH_LENGTH:
            return "Horizon dash"
        return f"{self.packet_length}-byte dash"


class LayoutUndetermined(Exception):
    """Raised when no candidate dash offset explains the packet."""


def _f32(buf: bytes, offset: int) -> float:
    return struct.unpack_from("<f", buf, offset)[0]


def _candidate_bases(length: int) -> list[int]:
    """
    Plausible starts for the dash block, best guess first.

    Anchoring to the end of the packet is first because it needs no knowledge of
    the padding at all: the dash block runs to the final byte in every documented
    variant, so its start is simply `length - 79`. The published fixed offsets
    follow as fallbacks, and a small sweep covers a variant nobody has written
    down. Every candidate still has to pass the velocity check before it is used.
    """
    bases = [length - DASH_BLOCK_LENGTH, SLED_LENGTH, SLED_LENGTH + 12]
    bases += [SLED_LENGTH + pad for pad in range(0, 32, 4)]
    seen: list[int] = []
    for b in bases:
        if b >= SLED_LENGTH and b + DASH_BLOCK_LENGTH <= length and b not in seen:
            seen.append(b)
    return seen


def _residual(buf: bytes, base: int) -> float:
    """
    How far `|velocity|` is from the dash block's `Speed`, in m/s.

    The two are the same physical quantity written into different sections of the
    packet, so a correct dash offset makes them agree to floating-point noise and
    an incorrect one leaves them unrelated.
    """
    vx, vy, vz = struct.unpack_from("<3f", buf, _VELOCITY)
    speed = _f32(buf, base + _D_SPEED)
    if not all(math.isfinite(v) for v in (vx, vy, vz, speed)):
        return math.inf
    return abs(math.hypot(vx, vy, vz) - speed)


def detect_layout(packets: Sequence[bytes]) -> Layout:
    """
    Work out the dash offset from real packets.

    Needs at least one packet from a car that is actually moving: while the car
    is stationary every candidate reads zero for both speed and velocity and they
    all agree perfectly, which is agreement that proves nothing. Rather than lock
    in whichever candidate was tried first, this refuses.
    """
    if not packets:
        raise LayoutUndetermined("no packets")

    length = len(packets[0])
    if any(len(p) != length for p in packets):
        raise LayoutUndetermined("packets are not all the same length")
    if length == SLED_LENGTH:
        raise LayoutUndetermined(
            "this is the Sled format, which carries no position. "
            "Set the game's data-out format to Dash (Forza Motorsport) or use "
            "Forza Horizon, which sends Dash packets."
        )
    if length < SLED_LENGTH + DASH_BLOCK_LENGTH:
        raise LayoutUndetermined(f"{length}-byte packet is too short to hold a dash block")

    moving = [p for p in packets if any(abs(v) > 1.0 for v in struct.unpack_from("<3f", p, _VELOCITY))]
    if not moving:
        raise LayoutUndetermined(
            "every packet has the car stationary, which cannot distinguish the "
            "layouts — drive for a moment and try again"
        )

    best: Layout | None = None
    for base in _candidate_bases(length):
        worst = max(_residual(p, base) for p in moving)
        if best is None or worst < best.residual:
            best = Layout(length, base, worst)

    # A metre per second of disagreement is far more than float noise and far
    # less than the nonsense a wrong offset produces, which is typically many
    # orders of magnitude out or not finite at all.
    if best is None or best.residual > 1.0:
        raise LayoutUndetermined(
            f"no dash offset made |velocity| agree with Speed "
            f"(best was {best.dash_base} at {best.residual:.1f} m/s off)"
            if best
            else "no candidate offsets fitted"
        )
    return best


def decode(packet: bytes, layout: Layout) -> ForzaFrame:
    """Decode one packet with an already-established layout."""
    if len(packet) != layout.packet_length:
        raise ValueError(
            f"packet is {len(packet)} bytes, layout is for {layout.packet_length}"
        )
    b = layout.dash_base
    yaw, pitch, roll = struct.unpack_from("<3f", packet, _YAW)
    vx, vy, vz = struct.unpack_from("<3f", packet, _VELOCITY)
    x, y, z = struct.unpack_from("<3f", packet, b + _D_POSITION)
    return ForzaFrame(
        is_race_on=struct.unpack_from("<i", packet, _IS_RACE_ON)[0] != 0,
        timestamp_ms=struct.unpack_from("<I", packet, _TIMESTAMP_MS)[0],
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        speed=_f32(packet, b + _D_SPEED),
        vx=vx,
        vy=vy,
        vz=vz,
        distance_travelled=_f32(packet, b + _D_DISTANCE),
        race_time=_f32(packet, b + _D_RACE_TIME),
        lap=struct.unpack_from("<H", packet, b + _D_LAP_NUMBER)[0],
        gear=struct.unpack_from("<B", packet, b + _D_GEAR)[0],
        steer=struct.unpack_from("<b", packet, b + _D_STEER)[0] / 127.0,
        accel=struct.unpack_from("<B", packet, b + _D_ACCEL)[0] / 255.0,
        brake=struct.unpack_from("<B", packet, b + _D_BRAKE)[0] / 255.0,
    )


class ForzaListener:
    """
    A UDP socket that yields decoded frames.

    Detection is deferred rather than done at construction: the layout cannot be
    established until the car moves, and refusing to start until then would mean
    the tool appears dead while someone is sitting in a menu. Frames before that
    point are buffered and dropped, since a capture cannot begin before the car
    does anyway.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5300, *, timeout: float = 1.0):
        self.host = host
        self.port = port
        self.layout: Layout | None = None
        self._pending: list[bytes] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(timeout)
        self._sock.bind((host, port))

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> "ForzaListener":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def frames(self) -> Iterator[ForzaFrame]:
        """Yield frames forever. Silent while waiting for the car to move."""
        while True:
            try:
                packet, _addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            frame = self.feed(packet)
            if frame is not None:
                yield frame

    def feed(self, packet: bytes) -> ForzaFrame | None:
        """
        Offer one packet. Returns a frame once the layout is known.

        Separate from `frames()` so the detection logic can be driven from a
        recorded capture in tests without a socket.
        """
        if self.layout is None:
            self._pending.append(packet)
            # Bounded: only enough history to catch the car moving.
            if len(self._pending) > 600:
                self._pending = self._pending[-300:]
            try:
                self.layout = detect_layout(self._pending)
            except LayoutUndetermined:
                return None
            self._pending.clear()
        return decode(packet, self.layout)
