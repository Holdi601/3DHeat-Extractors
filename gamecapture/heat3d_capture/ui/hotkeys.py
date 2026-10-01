"""
Hotkeys that work while the game has focus.

Starting and stopping a scan from the scanner's own window means alt-tabbing out
of the game, and alt-tabbing out of a full-screen game is exactly the thing that
ruins the capture you were about to start. So the keys have to be registered with
Windows rather than with the application, which is what `RegisterHotKey` is for:
it delivers the key to us no matter which window is in front.

That has a cost worth being explicit about. A global hotkey is taken away from
everything else on the machine — including the game — for as long as it is
registered. So the defaults use F-keys with a modifier, which games rarely bind,
the registration is released the moment scanning stops mattering, and a key the
system refuses to give us is reported rather than silently doing nothing. A
control that appears to exist and does not is worse than one that is absent.

Windows only. Elsewhere `available()` is false and the interface falls back to
its own buttons, which is the correct behaviour rather than a degraded one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

#: Modifier bits as `RegisterHotKey` wants them.
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
#: Stops the key auto-repeating while held, which would fire "stop scanning"
#: dozens of times.
MOD_NOREPEAT = 0x4000

#: Virtual key codes for F1..F12.
VK_F = {f"f{n}": 0x70 + n - 1 for n in range(1, 13)}

_MODS = {"alt": MOD_ALT, "ctrl": MOD_CONTROL, "control": MOD_CONTROL, "shift": MOD_SHIFT}


@dataclass(frozen=True)
class Binding:
    """One hotkey: what it is, and what it does."""

    name: str
    combination: str
    action: Callable[[], None]
    description: str = ""

    def pretty(self) -> str:
        return "+".join(part.capitalize() for part in self.combination.split("+"))


def parse(combination: str) -> tuple[int, int]:
    """
    Turn "ctrl+alt+f9" into the modifier mask and virtual key `RegisterHotKey` wants.

    Raises rather than guessing: a typo in a binding should be a message at
    start-up, not a key that never fires.
    """
    parts = [p.strip().lower() for p in combination.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    key = parts[-1]
    if key not in VK_F:
        raise ValueError(
            f"{key!r} is not a supported key; use F1 to F12, which games rarely bind"
        )
    mask = MOD_NOREPEAT
    for part in parts[:-1]:
        if part not in _MODS:
            raise ValueError(f"{part!r} is not a modifier; use ctrl, alt or shift")
        mask |= _MODS[part]
    return mask, VK_F[key]


def available() -> bool:
    return sys.platform == "win32"


class HotkeyManager:
    """
    Registers global hotkeys and runs their actions on a thread of its own.

    A thread, because `RegisterHotKey` delivers to the thread that registered and
    the messages have to be pumped; doing that on the interface thread would mean
    the interface can never be busy. Actions are therefore called off the GUI
    thread, and anything touching widgets must marshal — which is the caller's
    business, and is why `Binding.action` is a plain callable rather than
    anything Qt-aware.
    """

    def __init__(self) -> None:
        self.bindings: list[Binding] = []
        self.failures: list[tuple[Binding, str]] = []
        self._thread = None
        self._stop = None
        self._running = False

    @property
    def active(self) -> bool:
        return self._running

    def describe(self) -> str:
        """One line per binding, for the interface to show."""
        if not available():
            return "Global hotkeys need Windows; use the buttons."
        if not self._running:
            return "Hotkeys are off."
        lines = [f"{b.pretty()} — {b.description or b.name}" for b in self.bindings]
        lines += [f"{b.pretty()} unavailable: {why}" for b, why in self.failures]
        return "\n".join(lines)

    def start(self, bindings: list[Binding]) -> bool:
        """
        Register everything and begin pumping. Returns whether anything took.

        Failures are collected rather than raised: one key already taken by
        another program should not cost the user the others.
        """
        if not available() or self._running:
            return False
        import threading

        self.bindings = []
        self.failures = []
        self._stop = threading.Event()
        started = threading.Event()
        self._thread = threading.Thread(
            target=self._pump, args=(bindings, started), name="hotkeys", daemon=True
        )
        self._thread.start()
        started.wait(timeout=3.0)
        return self._running

    def stop(self) -> None:
        if not self._running or self._stop is None:
            return
        self._stop.set()
        # Nudge the message loop, which is otherwise blocked in GetMessage.
        try:
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(self._thread_id, 0x0400, 0, 0)
        except Exception:  # noqa: BLE001 - shutting down; a failure here is not fatal
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._running = False

    # -- the thread ---------------------------------------------------------

    def _pump(self, bindings, started) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()

        registered: dict[int, Binding] = {}
        for n, binding in enumerate(bindings, start=1):
            try:
                mask, key = parse(binding.combination)
            except ValueError as exc:
                self.failures.append((binding, str(exc)))
                continue
            if user32.RegisterHotKey(None, n, mask, key):
                registered[n] = binding
                self.bindings.append(binding)
            else:
                # Almost always because something else already holds it.
                self.failures.append((binding, "another program already has this key"))

        self._running = bool(registered)
        started.set()
        if not registered:
            return

        message = wintypes.MSG()
        WM_HOTKEY = 0x0312
        try:
            while not self._stop.is_set():
                if user32.GetMessageW(ctypes.byref(message), None, 0, 0) == 0:
                    break
                if message.message == WM_HOTKEY:
                    binding = registered.get(int(message.wParam))
                    if binding is not None:
                        try:
                            binding.action()
                        except Exception:  # noqa: BLE001
                            # A failing action must not take the hotkey thread
                            # with it, or every later key silently stops working.
                            pass
        finally:
            for n in registered:
                user32.UnregisterHotKey(None, n)
            self._running = False
