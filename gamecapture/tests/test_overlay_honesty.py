"""
What the overlay does when it cannot draw the truth.

This exists because of a specific failure, and it is worth stating plainly: a
race in the coverage map made `update_from` throw on the first large keyframe of
a real scan. Qt swallowed the exception into a console nobody was watching, the
panel went on painting whatever it had last drawn — which happened to be green —
and it reported a healthy scan for the rest of the lap.

A panel that shows nothing is a nuisance. A panel that shows *stale reassurance*
is worse than having none at all, because it is read as confirmation. So the
requirement is not "do not crash"; it is "when you cannot be right, say so
loudly, and do not keep the old picture on screen".
"""

from __future__ import annotations

import pytest

from heat3d_capture.ui.overlay import build_overlay


@pytest.fixture(scope="module")
def application():
    from heat3d_capture.ui.app import application as make

    return make()


@pytest.fixture
def overlay(application):
    made = build_overlay()()
    yield made
    made.close()


class TestSayingItStopped:
    def test_it_starts_out_fine(self, overlay):
        assert overlay._broken is None

    def test_reporting_a_failure_marks_it_broken(self, overlay):
        overlay.report_broken("ValueError: shapes do not match", 1)

        assert overlay._broken == ("ValueError: shapes do not match", 1)

    def test_the_stale_picture_is_not_painted_once_broken(self, overlay):
        """
        The old map must not survive alongside the warning.

        Leaving it visible is the whole failure: the driver reads the map, not
        the small print, and a green map next to a red banner still says green.
        """
        import inspect

        source = inspect.getsource(type(overlay).paintEvent)
        # The broken branch returns before anything else is drawn.
        before_return = source.split("return")[0]
        assert "_paint_broken" in before_return
        assert "self._map" not in before_return

    def test_the_message_says_the_scan_is_still_running(self, overlay):
        """
        Because the natural reaction to a red panel is to stop driving, and the
        scan is in fact fine — it is only the feedback that died.
        """
        import inspect

        source = inspect.getsource(type(overlay)._paint_broken)
        assert "still running" in source

    def test_the_message_warns_that_earlier_readings_were_stale(self, overlay):
        import inspect

        source = inspect.getsource(type(overlay)._paint_broken)
        assert "out of date" in source

    def test_repeated_failures_are_counted_not_just_flagged(self, overlay):
        """One failure is a glitch; a hundred is a panel that has been lying for
        a lap."""
        overlay.report_broken("ValueError: x", 1)
        overlay.report_broken("ValueError: x", 97)

        assert overlay._broken[1] == 97


class TestTheWindowKeepsGoing:
    def test_a_failing_overlay_does_not_stop_the_scan(self):
        """
        The handler catches around the overlay only, so a broken panel costs the
        panel and nothing else. Checked on the source because driving a real
        failure needs a live scan.
        """
        import inspect

        from heat3d_capture.ui.app import build_window

        source = inspect.getsource(build_window)
        # Anchored on the call itself: "if self.overlay is not None" also opens
        # `_open_overlay`, and matching that would test nothing.
        at = source.index("self.overlay.update_from(")
        handler = source[at - 900 : at + 900]

        assert "try:" in handler
        assert "report_broken" in handler
        # And the main window is named as the thing still worth reading.
        assert "still correct" in handler
