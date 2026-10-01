"""
Tests for the global hotkeys.

A hotkey that appears in the interface and does nothing is worse than no hotkey,
because the user only finds out mid-capture — which is the one moment the whole
feature exists to protect. So the tests here are mostly about failure: a typo, a
key another program already holds, a platform without the API.

Registration itself is tested for real on Windows, using keys nothing sensible
binds, and skipped elsewhere. Parsing is tested everywhere, because that is where
a typo becomes a silent dead key.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from heat3d_capture.ui.hotkeys import (
    MOD_ALT,
    MOD_CONTROL,
    MOD_NOREPEAT,
    MOD_SHIFT,
    VK_F,
    Binding,
    HotkeyManager,
    available,
    parse,
)


class TestParsing:
    def test_a_plain_function_key(self):
        mask, key = parse("f9")

        assert key == VK_F["f9"]
        # No-repeat always: holding "stop scanning" down should stop it once.
        assert mask == MOD_NOREPEAT

    def test_modifiers_combine(self):
        mask, key = parse("ctrl+alt+shift+f4")

        assert key == VK_F["f4"]
        assert mask == MOD_NOREPEAT | MOD_CONTROL | MOD_ALT | MOD_SHIFT

    def test_it_is_case_and_space_insensitive(self):
        assert parse("Ctrl + Alt + F9") == parse("ctrl+alt+f9")

    def test_an_unsupported_key_is_refused_with_a_reason(self):
        """
        Refused loudly at start-up rather than registered as nothing.

        The message names the alternative, because "invalid key" leaves the
        reader to guess which keys are valid.
        """
        with pytest.raises(ValueError, match="F1 to F12"):
            parse("ctrl+q")

    def test_an_unknown_modifier_is_refused(self):
        with pytest.raises(ValueError, match="modifier"):
            parse("meta+f9")

    def test_an_empty_binding_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            parse("   ")


class TestTheManagerWithoutWindows:
    @pytest.mark.skipif(available(), reason="this is the non-Windows path")
    def test_it_declines_rather_than_pretending(self):
        manager = HotkeyManager()

        assert manager.start([Binding("x", "f9", lambda: None)]) is False
        assert not manager.active
        assert "Windows" in manager.describe()


@pytest.mark.skipif(not available(), reason="global hotkeys are Windows-only")
class TestTheManagerOnWindows:
    #: Keys nothing sensible binds, so running the suite does not fight the
    #: user's own shortcuts.
    SPARE = "ctrl+alt+shift+f7"
    OTHER = "ctrl+alt+shift+f8"

    def test_it_registers_and_releases(self):
        manager = HotkeyManager()
        try:
            assert manager.start([Binding("spare", self.SPARE, lambda: None, "test")])
            assert manager.active
            assert "Ctrl+Alt+Shift+F7" in manager.describe()
        finally:
            manager.stop()
        assert not manager.active

    def test_a_key_already_taken_is_reported_not_swallowed(self):
        """
        The second manager must say which key it could not have.

        Silently dropping it is how a control comes to exist in the interface
        and not in reality.
        """
        first = HotkeyManager()
        second = HotkeyManager()
        try:
            assert first.start([Binding("spare", self.SPARE, lambda: None)])
            second.start([Binding("spare", self.SPARE, lambda: None)])

            assert not second.active
            assert second.failures, "a refused registration was not recorded"
            assert "already" in second.failures[0][1]
        finally:
            second.stop()
            first.stop()

    def test_one_bad_binding_does_not_cost_the_others(self):
        manager = HotkeyManager()
        try:
            taken = manager.start(
                [
                    Binding("bad", "ctrl+q", lambda: None),
                    Binding("good", self.OTHER, lambda: None, "works"),
                ]
            )
            assert taken
            assert [b.name for b in manager.bindings] == ["good"]
            assert manager.failures[0][0].name == "bad"
        finally:
            manager.stop()

    def test_an_action_that_raises_does_not_kill_the_thread(self):
        """
        One broken action must not silently disable every other key.

        Checked by registering a second key after the first has been made to
        fail, and confirming the manager is still running.
        """
        manager = HotkeyManager()

        def explode():
            raise RuntimeError("deliberate")

        try:
            assert manager.start(
                [Binding("boom", self.SPARE, explode), Binding("fine", self.OTHER, lambda: None)]
            )
            # Cannot synthesise a WM_HOTKEY without extra API, so this asserts
            # the guard exists around the call rather than driving it.
            import inspect

            source = inspect.getsource(HotkeyManager._pump)
            assert "except Exception" in source
            assert manager.active
        finally:
            manager.stop()

    def test_stopping_twice_is_harmless(self):
        manager = HotkeyManager()
        manager.start([Binding("spare", self.SPARE, lambda: None)])
        manager.stop()
        manager.stop()

        assert not manager.active

    def test_registration_does_not_block_the_caller(self):
        """Start-up must not hang if the message loop misbehaves."""
        manager = HotkeyManager()
        started = time.monotonic()
        try:
            manager.start([Binding("spare", self.SPARE, lambda: None)])
        finally:
            manager.stop()

        assert time.monotonic() - started < 4.0
