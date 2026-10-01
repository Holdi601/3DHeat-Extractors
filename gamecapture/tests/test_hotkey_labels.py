"""
Tests that the controls advertise the keys they actually have.

A button reading "Start scanning   Ctrl+Alt+F9" is only useful if that key works.
If the registration failed — another program holds it, or this is not Windows —
the label must say nothing, because the user finds out otherwise only by pressing
it mid-capture, which is the one moment the shortcut exists to protect.

So both directions are checked: the key appears when it was taken, and does not
appear when it was not.
"""

from __future__ import annotations

import pytest

from heat3d_capture.ui.app import build_window
from heat3d_capture.ui.hotkeys import Binding


@pytest.fixture(scope="module")
def application():
    from heat3d_capture.ui.app import application as make

    return make()


@pytest.fixture
def window(application, monkeypatch):
    """A window whose hotkeys are stubbed, so the test does not fight the machine."""
    Window = build_window()
    monkeypatch.setattr(Window, "_check_hardware", lambda self: None)
    monkeypatch.setattr(Window, "_start_hotkeys", lambda self: None)
    made = Window()
    yield made
    made.close()


def pretend_registered(window, names):
    window._hotkeys.bindings = [
        Binding(name, combination, lambda: None, name)
        for name, combination in names.items()
    ]


class TestWhenTheKeysWereTaken:
    def test_the_start_button_names_its_key(self, window):
        pretend_registered(window, {"toggle": "ctrl+alt+f9"})

        window._label_buttons()

        assert "Start scanning" in window.start.text()
        assert "Ctrl+Alt+F9" in window.start.text()

    def test_the_overlay_checkbox_names_its_key(self, window):
        pretend_registered(window, {"overlay": "ctrl+alt+f11"})

        window._label_buttons()

        assert "Ctrl+Alt+F11" in window.use_overlay.text()

    def test_the_key_survives_the_button_changing_to_stop(self, window):
        """
        The label is rebuilt whenever scanning starts or stops, and the shortcut
        has to come along — otherwise it appears on "Start" and vanishes on
        "Stop", which reads as the key having stopped working.
        """
        pretend_registered(window, {"toggle": "ctrl+alt+f9"})

        class Running:
            """Enough of a session to look like one, including to teardown."""

            def is_running(self):
                return True

            def stop(self):
                pass

        window.session = Running()
        label = window._start_label()

        assert label.startswith("Stop scanning")
        assert "Ctrl+Alt+F9" in label


class TestWhenTheyWereNot:
    def test_nothing_is_advertised(self, window):
        window._hotkeys.bindings = []

        window._label_buttons()

        assert window.start.text() == "Start scanning"
        assert window.use_overlay.text() == "Show feedback over the game"

    def test_a_key_that_failed_is_not_shown_while_its_neighbours_are(self, window):
        """One refused registration must not take the others' labels with it,
        nor gain one it does not have."""
        pretend_registered(window, {"toggle": "ctrl+alt+f9"})

        window._label_buttons()

        assert "Ctrl+Alt+F9" in window.start.text()
        assert "Ctrl+Alt+F11" not in window.use_overlay.text()
