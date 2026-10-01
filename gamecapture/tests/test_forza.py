"""
Tests for the Forza telemetry decoder.

The interesting property under test is not "can it read a struct" but "does it
work out *where* the struct is". Packets are synthesised here at a known dash
offset, and detection has to recover that offset without being told — which is
exactly the job it has to do against a real game whose padding is documented
inconsistently.
"""

from __future__ import annotations

import math
import struct

import pytest

from heat3d_capture.telemetry.forza import (
    DASH_BLOCK_LENGTH,
    FH_DASH_LENGTH,
    FM_DASH_LENGTH,
    SLED_LENGTH,
    ForzaListener,
    LayoutUndetermined,
    decode,
    detect_layout,
)


def make_packet(
    length: int,
    dash_base: int,
    *,
    velocity=(3.0, 0.0, 4.0),
    position=(100.0, 5.0, -200.0),
    yaw_pitch_roll=(0.5, -0.1, 0.02),
    lap: int = 2,
    gear: int = 4,
    race_on: bool = True,
    timestamp: int = 123456,
) -> bytes:
    """
    A packet whose Speed genuinely matches its velocity, as the game's would.

    Everything outside the two blocks is filled with a recognisable byte rather
    than zeros, so a decoder reading the wrong offset picks up obvious nonsense
    instead of a plausible-looking zero.
    """
    buf = bytearray(b"\xcd" * length)
    struct.pack_into("<i", buf, 0, 1 if race_on else 0)
    struct.pack_into("<I", buf, 4, timestamp)
    struct.pack_into("<f", buf, 8, 7000.0)  # EngineMaxRpm
    struct.pack_into("<f", buf, 12, 800.0)  # EngineIdleRpm
    struct.pack_into("<f", buf, 16, 3500.0)  # CurrentEngineRpm
    struct.pack_into("<3f", buf, 32, *velocity)
    struct.pack_into("<3f", buf, 56, *yaw_pitch_roll)

    speed = math.hypot(*velocity)
    struct.pack_into("<3f", buf, dash_base + 0, *position)
    struct.pack_into("<f", buf, dash_base + 12, speed)
    struct.pack_into("<f", buf, dash_base + 48, 1234.5)  # DistanceTraveled
    struct.pack_into("<f", buf, dash_base + 64, 98.5)  # CurrentRaceTime
    struct.pack_into("<H", buf, dash_base + 68, lap)
    struct.pack_into("<B", buf, dash_base + 71, 255)  # Accel
    struct.pack_into("<B", buf, dash_base + 72, 0)  # Brake
    struct.pack_into("<B", buf, dash_base + 75, gear)
    struct.pack_into("<b", buf, dash_base + 76, 64)  # Steer
    return bytes(buf)


#: The two layouts that matter, with the dash block ending at the final byte.
MOTORSPORT = (FM_DASH_LENGTH, FM_DASH_LENGTH - DASH_BLOCK_LENGTH)
HORIZON = (FH_DASH_LENGTH, FH_DASH_LENGTH - DASH_BLOCK_LENGTH)


class TestDetection:
    @pytest.mark.parametrize("length,base", [MOTORSPORT, HORIZON], ids=["motorsport", "horizon"])
    def test_recovers_the_dash_offset_it_was_not_told(self, length, base):
        packets = [make_packet(length, base, velocity=(10.0, 0.0, 5.0))]
        layout = detect_layout(packets)
        assert layout.dash_base == base
        assert layout.packet_length == length
        assert layout.residual < 1e-3

    def test_finds_an_offset_nobody_documented(self):
        # The point of searching rather than asserting: a variant with padding
        # that matches no published description still resolves, because the
        # velocity check does not care where the block is.
        length = SLED_LENGTH + 20 + DASH_BLOCK_LENGTH
        packets = [make_packet(length, SLED_LENGTH + 20, velocity=(0.0, 0.0, 22.0))]
        assert detect_layout(packets).dash_base == SLED_LENGTH + 20

    def test_refuses_when_the_car_never_moved(self):
        # Every candidate reads zero for speed and for velocity, so they all
        # agree. Agreement that proves nothing must not be taken as an answer.
        packets = [make_packet(*HORIZON, velocity=(0.0, 0.0, 0.0)) for _ in range(10)]
        with pytest.raises(LayoutUndetermined, match="stationary"):
            detect_layout(packets)

    def test_rejects_the_sled_format_with_a_usable_message(self):
        with pytest.raises(LayoutUndetermined, match="carries no position"):
            detect_layout([b"\x00" * SLED_LENGTH])

    def test_rejects_mixed_packet_lengths(self):
        with pytest.raises(LayoutUndetermined, match="same length"):
            detect_layout([make_packet(*HORIZON), make_packet(*MOTORSPORT)])

    def test_rejects_a_packet_that_is_not_forza(self):
        with pytest.raises(LayoutUndetermined):
            detect_layout([b"\x00" * 64])

    def test_does_not_accept_an_offset_that_only_nearly_fits(self):
        # A packet whose Speed field disagrees with its velocity is not a layout
        # problem, it is not Forza. Guessing an offset anyway would produce a
        # camera path built from unrelated floats.
        length, base = HORIZON
        buf = bytearray(make_packet(length, base, velocity=(10.0, 0.0, 0.0)))
        struct.pack_into("<f", buf, base + 12, 999.0)
        with pytest.raises(LayoutUndetermined, match="agree"):
            detect_layout([bytes(buf)])


class TestDecode:
    def test_reads_the_fields_the_reconstruction_needs(self):
        length, base = HORIZON
        packet = make_packet(
            length, base, velocity=(3.0, 0.0, 4.0), position=(100.0, 5.0, -200.0)
        )
        layout = detect_layout([packet])
        f = decode(packet, layout)

        assert f.position == pytest.approx((100.0, 5.0, -200.0))
        assert f.speed == pytest.approx(5.0)
        assert (f.vx, f.vy, f.vz) == pytest.approx((3.0, 0.0, 4.0))
        assert f.yaw == pytest.approx(0.5)
        assert f.pitch == pytest.approx(-0.1)
        assert f.roll == pytest.approx(0.02)
        assert f.is_race_on is True
        assert f.timestamp_ms == 123456
        assert f.lap == 2
        assert f.gear == 4
        assert f.moving is True

    def test_normalises_the_byte_ranged_inputs(self):
        length, base = HORIZON
        packet = make_packet(length, base, velocity=(10.0, 0.0, 0.0))
        f = decode(packet, detect_layout([packet]))
        assert f.accel == pytest.approx(1.0)
        assert f.brake == pytest.approx(0.0)
        assert f.steer == pytest.approx(64 / 127.0)

    def test_a_parked_car_is_not_moving(self):
        length, base = HORIZON
        # Layout from a moving packet, then decode a stationary one with it.
        layout = detect_layout([make_packet(length, base, velocity=(9.0, 0.0, 0.0))])
        f = decode(make_packet(length, base, velocity=(0.0, 0.0, 0.0)), layout)
        assert f.moving is False

    def test_refuses_a_packet_of_the_wrong_length(self):
        layout = detect_layout([make_packet(*HORIZON, velocity=(9.0, 0.0, 0.0))])
        with pytest.raises(ValueError, match="layout is for"):
            decode(make_packet(*MOTORSPORT), layout)


class TestListener:
    """Detection driven the way it happens live, one packet at a time."""

    def test_stays_quiet_until_the_car_moves_then_yields(self):
        listener = ForzaListener.__new__(ForzaListener)  # no socket
        listener.layout = None
        listener._pending = []

        for _ in range(5):
            assert listener.feed(make_packet(*HORIZON, velocity=(0.0, 0.0, 0.0))) is None
        assert listener.layout is None

        frame = listener.feed(make_packet(*HORIZON, velocity=(0.0, 0.0, 18.0)))
        assert frame is not None
        assert listener.layout is not None
        assert listener.layout.dash_base == HORIZON[1]
        assert frame.speed == pytest.approx(18.0)

    def test_does_not_grow_without_bound_while_parked(self):
        listener = ForzaListener.__new__(ForzaListener)
        listener.layout = None
        listener._pending = []
        parked = make_packet(*HORIZON, velocity=(0.0, 0.0, 0.0))
        for _ in range(2000):
            listener.feed(parked)
        assert len(listener._pending) <= 600
