"""
Press the buttons.

Three bugs shipped in a row that a user hit within seconds, and every one of them
got past a suite that was green:

- the window would not open at all, because two `QApplication`s were built;
- pressing Start died on `'str' object has no attribute 'keyframe_change'`,
  because Qt hands back a bare string for a `str` enum stored as item data;
- a bad source escaped as an unhandled exception on a worker thread, while the
  window sat saying "scanning".

They share one cause, and it is not in the code under test. Every UI test here
*built* a window and checked that it existed. None of them ever used it. A test
that constructs an object and inspects it cannot see a fault in the wiring
between that object and everything else — and the wiring is where a user lives.

So this drives the real window through a real scan: set a source, press Start,
let the event loop run, press Stop, and demand that nothing threw anywhere,
including on threads whose exceptions would otherwise only reach a console.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

from heat3d_capture.scan.session import Quality


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A short recording of the ground-truth room, as a real file on disk."""
    import cv2

    from heat3d_capture.pose.odometry import Intrinsics

    from .scene import survey

    path = tmp_path_factory.mktemp("clips") / "room.mp4"
    k = Intrinsics.from_fov(480, 270, 90.0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (480, 270))
    assert writer.isOpened()
    # Long enough that a few seconds of scanning does not finish it: the
    # behaviour under test is stopping *mid-scan*, which is what a user does.
    for frame in survey(k, steps=400):
        writer.write(cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


@pytest.fixture
def caught_thread_exceptions():
    """
    Collect anything raised on a worker thread.

    Without this the suite cannot see the class of fault that hit hardest: an
    exception on a background thread does not fail a test, it prints to stderr
    and the scan quietly stops doing anything.
    """
    seen: list[BaseException] = []
    previous = threading.excepthook

    def hook(args):
        seen.append(args.exc_value)
        previous(args)

    threading.excepthook = hook
    try:
        yield seen
    finally:
        threading.excepthook = previous


def pump(seconds: float) -> None:
    """Run the Qt event loop for a while, as a user sitting there would."""
    from PySide6 import QtWidgets

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.02)


@pytest.mark.skipif(
    not list(__import__("pathlib").Path(__file__).resolve().parents[1].glob("models/depth-*.onnx")),
    reason="no exported model; the window will not start a scan without one",
)
class TestDrivingTheWindow:
    @pytest.fixture
    def window(self):
        from heat3d_capture.ui import app as app_module

        app_module.application()
        window = app_module.build_window()()
        yield window
        if window.session is not None and window.session.is_running():
            window.session.stop()
        window.close()

    def test_pressing_start_actually_scans(self, window, clip, caught_thread_exceptions):
        # A video file, so the test needs no screen and no game.
        window.source_kind.setCurrentIndex(2)
        window.source_value.setText(str(clip))
        window.use_overlay.setChecked(False)

        assert window.start.isEnabled(), "the window refused to start at all"
        window._toggle()
        assert window.session is not None

        pump(8.0)

        progress = window.session.progress
        assert progress.frames_seen > 0, "no frames reached the scan"
        assert progress.keyframes > 0, "nothing was kept"
        assert progress.reconstruction is not None, "nothing was reconstructed"
        assert window.session.error is None, f"the scan failed: {window.session.error!r}"
        assert not caught_thread_exceptions, (
            f"something threw on a worker thread: {caught_thread_exceptions!r}"
        )

    def test_pressing_stop_mid_scan_stops_it(self, window, clip, caught_thread_exceptions):
        window.source_kind.setCurrentIndex(2)
        window.source_value.setText(str(clip))
        window.use_overlay.setChecked(False)

        window._toggle()
        pump(4.0)
        assert window.session.is_running(), "the clip is meant to outlast this wait"

        started = time.monotonic()
        window._toggle()
        elapsed = time.monotonic() - started

        assert not window.session.is_running(), "stop did nothing"
        # The join has a five second budget; taking all of it means the worker
        # is not checking often enough, which is what "the button does nothing"
        # feels like from the outside.
        assert elapsed < 3.0, f"stop blocked the window for {elapsed:.1f}s"
        assert window.session.progress.message == "stopped"
        assert window.session.progress.keyframes > 0, "the partial scan was discarded"
        assert not caught_thread_exceptions, (
            f"something threw on a worker thread: {caught_thread_exceptions!r}"
        )

    def test_a_finished_scan_resets_the_button_rather_than_restarting(
        self, window, tmp_path
    ):
        # `_toggle` is the same button either way. Pressing it after a scan has
        # ended by itself must start a *new* scan, not be mistaken for a stop —
        # and the label has to have changed back, or the user is pressing a
        # button that says the opposite of what it does.
        import cv2

        from heat3d_capture.pose.odometry import Intrinsics

        from .scene import survey

        short = tmp_path / "short.mp4"
        k = Intrinsics.from_fov(480, 270, 90.0)
        writer = cv2.VideoWriter(str(short), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (480, 270))
        for frame in survey(k, steps=12):
            writer.write(cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
        writer.release()

        window.source_kind.setCurrentIndex(2)
        window.source_value.setText(str(short))
        window.use_overlay.setChecked(False)
        window._toggle()
        pump(8.0)

        assert not window.session.is_running()
        # Not the whole string: the button also advertises its global hotkey,
        # and whether that registered depends on whether something else on the
        # machine already holds the key. What this is about is that the button
        # offers to start again rather than to stop.
        assert window.start.text().startswith("Start scanning")

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_every_quality_setting_survives_being_chosen(self, window, clip, index):
        # The exact shape of the bug: a value that round-trips through Qt and
        # comes back a different type. Reading it back must give the preset.
        window.quality.setCurrentIndex(index)
        chosen = window._quality()
        assert isinstance(chosen, Quality)
        # The attributes the scan will ask for, which a bare string does not have.
        assert isinstance(chosen.keyframe_change, float)
        assert isinstance(chosen.capture_side, int)
        assert window._settings().quality is chosen

    def test_a_source_that_cannot_be_opened_is_reported_not_thrown(
        self, window, caught_thread_exceptions
    ):
        # A mistyped path, a missing file, a monitor that is not there. This used
        # to escape the worker thread entirely, printing to a console nobody
        # reads while the window said "scanning" forever.
        window.source_kind.setCurrentIndex(2)
        window.source_value.setText("C:/definitely/not/a/real/file.mp4")
        window.use_overlay.setChecked(False)

        window._toggle()
        pump(4.0)

        assert not window.session.is_running(), "the scan should have stopped"
        assert window.session.error is not None, "the failure was swallowed"
        assert "fail" in window.session.progress.message.lower()
        assert not caught_thread_exceptions, (
            f"the failure escaped instead of being reported: {caught_thread_exceptions!r}"
        )


class TestSettingsCoercion:
    """The guard behind the window, for every other caller."""

    def test_a_bare_string_quality_is_coerced(self):
        from heat3d_capture.scan.session import ScanSettings

        # Exactly what Qt handed back before the window was fixed.
        settings = ScanSettings(quality="balanced")
        assert settings.quality is Quality.BALANCED
        assert settings.capture_side() == Quality.BALANCED.capture_side

    def test_capitalised_labels_are_accepted(self):
        from heat3d_capture.scan.session import ScanSettings

        assert ScanSettings(quality="Detailed").quality is Quality.DETAILED

    def test_nonsense_is_refused_loudly(self):
        from heat3d_capture.scan.session import ScanSettings

        with pytest.raises(ValueError):
            ScanSettings(quality="haphazard")


@pytest.mark.skipif(
    not list(__import__("pathlib").Path(__file__).resolve().parents[1].glob("models/depth-*.onnx")),
    reason="no exported model",
)
class TestClosingDuringAScan:
    """
    Shutting the window mid-scan.

    Found by running the suite twice: the second run died with a Windows access
    violation rather than a Python error. A background thread was computing a
    coverage map and emitting the result into a window Qt had already destroyed,
    which is not an exception anyone can catch — the process simply ends. A user
    closing the window while scanning would have hit exactly that.
    """

    def test_closing_mid_scan_does_not_take_the_process_with_it(self, clip):
        from heat3d_capture.ui import app as app_module

        app_module.application()
        window = app_module.build_window()()
        window.source_kind.setCurrentIndex(2)
        window.source_value.setText(str(clip))
        window.use_overlay.setChecked(False)

        window._toggle()
        pump(3.0)
        assert window.session.is_running()
        # Force map work to be in flight at the moment of closing, which is the
        # race rather than merely the sequence.
        window._refresh_map()

        window.close()

        assert window._closing is True
        assert not window.session.is_running(), "the scan outlived its window"
        # Pumping afterwards is where a stale queued signal would land.
        pump(1.5)

    def test_the_refresh_timer_stops_with_the_window(self, clip):
        from heat3d_capture.ui import app as app_module

        app_module.application()
        window = app_module.build_window()()
        window.close()
        assert not window._slow.isActive(), "the timer kept firing into a dead window"
