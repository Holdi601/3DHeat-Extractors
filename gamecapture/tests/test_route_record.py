"""
Recording the route, and surviving whatever happens next.

This exists because of a lap that was driven and then lost. A real Forza run —
telemetry on, 55,579 packets, the whole track covered — was held in memory while
the surface extractor worked on it, the extractor failed, and the positions went
with it. Driving it again is cheap in principle and expensive in practice: the
game has to be running, the right track loaded, and someone has to drive.

So the property under test is not "it decodes telemetry" — `test_forza.py`
covers that. It is that the file on disk is complete at every moment, that a
stationary car does not fill it, and that the numbers written are the game's own
world coordinates rather than anything this project invented. Those coordinates
are the point: they say *where on the map* a route ran, which is the only handle
there is for finding the same stretch in the game's own files.
"""

from __future__ import annotations

import json
import socket

import numpy as np
import pytest

from heat3d_capture.telemetry.record import Recorder, Recording

from .test_forza import DASH_BLOCK_LENGTH, FH_DASH_LENGTH, make_packet

BASE = FH_DASH_LENGTH - DASH_BLOCK_LENGTH


@pytest.fixture
def port() -> int:
    """A free port, asked of the OS rather than guessed at."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    chosen = probe.getsockname()[1]
    probe.close()
    return chosen


def send(port: int, packets) -> None:
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for packet in packets:
        out.sendto(packet, ("127.0.0.1", port))
    out.close()


def driving(count: int, *, step: float = 2.0, moving: bool = True):
    """A car going somewhere, one packet per sample."""
    velocity = (30.0, 0.0, 0.0) if moving else (0.0, 0.0, 0.0)
    return [
        make_packet(
            FH_DASH_LENGTH,
            BASE,
            velocity=velocity,
            position=(1000.0 + i * step, 40.0, -2500.0 - i * step * 0.5),
        )
        for i in range(count)
    ]


def drain(recorder: Recorder, attempts: int) -> int:
    kept = 0
    for _ in range(attempts):
        if recorder.poll():
            kept += 1
    return kept


class TestItRecordsWhatWasDriven:
    def test_a_short_drive_lands_as_world_coordinates(self, port):
        with Recorder(port=port, min_step=0.5) as recorder:
            # Detection needs the car moving across a few packets before the
            # layout is known, so the first ones legitimately yield nothing.
            send(port, driving(40))
            drain(recorder, 40)
            route = recorder.finish()

        assert len(route) > 20
        # The game's own numbers, unshifted: a route two kilometres out from the
        # origin has to still be two kilometres out on disk, or it cannot be
        # matched against anything in the game's files.
        assert route.positions[0][0] == pytest.approx(1000.0, abs=5.0)
        assert route.positions[0][2] == pytest.approx(-2500.0, abs=5.0)

    def test_the_distance_driven_is_the_distance_reported(self, port):
        with Recorder(port=port, min_step=0.5) as recorder:
            send(port, driving(40, step=2.0))
            drain(recorder, 40)
            route = recorder.finish()

        # Each step is 2 m along x and 1 m along z: 2.236 m of travel.
        expected = (len(route) - 1) * np.hypot(2.0, 1.0)
        assert route.length == pytest.approx(expected, rel=0.05)

    def test_the_bounding_box_is_the_handle_into_the_map(self, port):
        with Recorder(port=port, min_step=0.5) as recorder:
            send(port, driving(40))
            drain(recorder, 40)
            route = recorder.finish()

        low, high = route.bounds
        assert np.all(low <= route.positions.min(axis=0) + 1e-9)
        assert np.all(high >= route.positions.max(axis=0) - 1e-9)
        assert high[0] - low[0] > 10.0


class TestItRefusesToFillUpWithNothing:
    def test_a_parked_car_records_no_route(self, port):
        """
        Forza sends sixty packets a second whether or not the car is going
        anywhere. Without this check a lap is mostly the start line, several
        thousand times over — and worse, the file looks full.
        """
        with Recorder(port=port) as recorder:
            send(port, driving(30, moving=False))
            drain(recorder, 30)
            route = recorder.finish()

        assert len(route) == 0

    def test_crawling_does_not_write_a_sample_per_packet(self, port):
        """
        The user drove this track at 10-20 km/h to keep the depth model happy.
        At 60 packets a second that is a sample every 5 cm; the route does not
        need that, and neither does anything reading it.
        """
        with Recorder(port=port, min_step=0.5) as recorder:
            send(port, driving(120, step=0.05))
            drain(recorder, 120)
            route = recorder.finish()

        assert len(route) < 40, f"{len(route)} samples from 6 m of crawling"

    def test_nothing_arriving_is_not_an_error(self, port):
        """A wrong port, or Data Out left off. Silence, not a crash."""
        with Recorder(port=port) as recorder:
            assert recorder.poll() is False
            assert len(recorder.finish()) == 0


class TestWhatIsOnDiskIsUsable:
    def test_it_round_trips(self, tmp_path, port):
        with Recorder(port=port) as recorder:
            send(port, driving(40))
            drain(recorder, 40)
            route = recorder.finish()

        written = route.save(tmp_path / "lap")
        back = Recording.load(written)

        assert np.allclose(back.positions, route.positions)

    def test_the_coordinates_are_readable_without_this_project(self, tmp_path, port):
        """
        The JSON is not a convenience. A route locked in a format that needs
        this code to open is useless to whatever tool ends up doing the lookup
        against the game's files.
        """
        with Recorder(port=port) as recorder:
            send(port, driving(40))
            drain(recorder, 40)
            route = recorder.finish()
        route.save(tmp_path / "lap")

        data = json.loads((tmp_path / "lap.json").read_text(encoding="utf-8"))

        assert data["samples"] == len(route)
        assert len(data["positions"]) == len(route)
        assert data["bounds"]["min"][0] == pytest.approx(route.positions[:, 0].min())

    def test_an_empty_recording_still_answers_rather_than_raising(self):
        empty = Recording(np.zeros((0, 3)), np.zeros(0), np.zeros(0))

        assert len(empty) == 0
        assert empty.length == 0.0
        low, high = empty.bounds
        assert np.array_equal(low, np.zeros(3))
        assert np.array_equal(high, np.zeros(3))
