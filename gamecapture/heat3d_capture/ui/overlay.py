"""
Feedback drawn over the game.

The main window is the wrong place to read while scanning, and for a physical
reason: someone walking a level is looking at the level. Alt-tabbing to check
coverage means stopping, which means the frames either side of the check are of
a stationary camera, which is precisely the input the whole pipeline is worst at.
The feedback has to be where the eyes already are.

Deliberately small and quiet. It sits in a corner, shows the three things that
change what someone does next — where they have been, whether tracking is
holding, and how fast they are moving — and nothing else. A full dashboard over
a game is unreadable at a glance and covers the thing being scanned.

Click-through, so it cannot be hit by accident
----------------------------------------------
A window over a game that swallows a click is worse than no window: the click
that was meant to fire a weapon or steer a car goes nowhere, and the user has no
idea why. The overlay sets `WS_EX_TRANSPARENT`, which makes Windows deliver every
mouse event to whatever is underneath.

Exclusive fullscreen has no overlay
-----------------------------------
Nothing can draw over a game that has taken exclusive control of the display —
not this, not Steam's overlay, not Discord's. **Borderless windowed** is the mode
that works, and the panel says so rather than leaving someone to conclude the
feature is broken.
"""

from __future__ import annotations

import sys

import numpy as np

#: Where it sits, and how big. Small enough to ignore, large enough to read the
#: coverage map at a glance.
MARGIN = 18
MAP_SIZE = 190
PANEL_WIDTH = MAP_SIZE + 24


def make_click_through(window_id: int) -> bool:
    """
    Ask Windows to pass every mouse event straight through.

    Returns whether it worked. Failure is not fatal — the overlay is still
    useful, it just has to be moved out of the way if it gets in one — so this
    reports rather than raises.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_LAYERED = 0x00080000
        user32 = ctypes.windll.user32
        # The 64-bit forms, with the 32-bit ones as a fallback: the pointer-sized
        # variants do not exist on 32-bit Python.
        get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        current = get_long(int(window_id), GWL_EXSTYLE)
        set_long(int(window_id), GWL_EXSTYLE, current | WS_EX_TRANSPARENT | WS_EX_LAYERED)
        return True
    except Exception:
        return False


def build_overlay():  # pragma: no cover - constructed only with a display
    from PySide6 import QtCore, QtGui, QtWidgets

    from .preview import coverage_map, health_text, mark_weak_spots

    class Overlay(QtWidgets.QWidget):
        """A frameless, transparent, click-through panel pinned over the game."""

        def __init__(self, parent=None) -> None:
            super().__init__(
                None,
                QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
                # `Tool` rather than `Window`: it keeps the overlay out of the
                # taskbar and out of the alt-tab order, where an entry that
                # cannot be focused is only confusing.
                | QtCore.Qt.Tool
                | QtCore.Qt.WindowTransparentForInput,
            )
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
            # Taller than the map plus text: the advice line wraps to two, and
            # a panel that clips its own instruction is worse than no instruction.
            self.setFixedSize(PANEL_WIDTH, MAP_SIZE + 168)

            self._map: np.ndarray | None = None
            self._health = ("Waiting for the first frame.", "#8b949e")
            self._speed = 0.0
            self._sharpness = 1.0
            self._advice = ("", "ok")
            self._stats = ""
            self._hotkeys = ""
            self._broken: tuple[str, int] | None = None
            self._click_through = False

            self._position()

        # -- placement ---------------------------------------------------

        def _position(self) -> None:
            screen = QtWidgets.QApplication.primaryScreen()
            if screen is None:
                return
            area = screen.availableGeometry()
            # Top right: the corner least likely to hold a game's own HUD, which
            # tends to live at the bottom and in the centre.
            self.move(area.right() - self.width() - MARGIN, area.top() + MARGIN)

        def showEvent(self, event) -> None:
            super().showEvent(event)
            if not self._click_through:
                self._click_through = make_click_through(int(self.winId()))

        # -- content -----------------------------------------------------

        def update_from(self, reconstructor, progress) -> None:
            """Refresh from a live scan. Safe to call on the GUI thread only."""
            if reconstructor is not None:
                grid, extent = reconstructor.coverage.top_down(resolution=120)
                spots = reconstructor.state.weak_spots
                image = coverage_map(
                    grid,
                    trajectory=reconstructor.trajectory.positions,
                    extent=extent,
                    size=MAP_SIZE,
                )
                self._map = mark_weak_spots(image, spots, extent)
                self._health = health_text(reconstructor.state)
                state = reconstructor.state
                self._stats = (
                    f"{state.fused} placed · {len(spots)} to revisit · "
                    f"{state.trajectory_length:.0f} m"
                )
                self._sharpness = state.sharpness
                self._advice = (state.advice, state.advice_level)
            if progress is not None:
                self._speed = progress.change
            self.update()

        def report_broken(self, reason: str, count: int) -> None:
            """
            Say that this panel has stopped being true.

            Drawn over everything else and in red, because the failure mode it
            exists for is a panel that looks fine and is not. Whatever is on the
            map underneath is stale by definition once this is set.
            """
            self._broken = (reason, count)
            self.update()

        def set_hotkeys(self, text: str) -> None:
            """Shown once, because someone driving cannot look them up."""
            self._hotkeys = text
            self.update()

        # -- painting ----------------------------------------------------

        def paintEvent(self, event) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            if self._broken is not None:
                self._paint_broken(painter)
                painter.end()
                return

            # A dark panel rather than floating text: over a bright sky, white
            # text on nothing is unreadable, and over a dark interior black text
            # is. A backing plate is legible on both.
            painter.setBrush(QtGui.QColor(13, 17, 23, 205))
            painter.setPen(QtGui.QPen(QtGui.QColor(48, 54, 61, 220), 1))
            painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)

            y = 12
            if self._map is not None:
                h, w = self._map.shape[:2]
                image = QtGui.QImage(
                    np.ascontiguousarray(self._map).data, w, h, 3 * w, QtGui.QImage.Format_RGB888
                )
                painter.drawPixmap(12, y, QtGui.QPixmap.fromImage(image.copy()))
                y += h + 10
            else:
                painter.setPen(QtGui.QColor(139, 148, 158))
                painter.drawText(14, y + 20, "waiting for the scan…")
                y += 40

            painter.setPen(QtGui.QColor(self._health[1]))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            rect = QtCore.QRect(12, y, PANEL_WIDTH - 24, 34)
            painter.drawText(rect, QtCore.Qt.TextWordWrap, self._health[0])
            y += 36

            painter.setPen(QtGui.QColor(139, 148, 158))
            painter.drawText(12, y + 10, self._stats)
            y += 18

            # The walk-speed bar: how close the next frame is to being kept. The
            # only live control someone has while walking.
            self._bar(painter, y, self._speed, QtGui.QColor(63, 185, 80), "movement")
            y += 22

            # Sharpness, relative to this capture's own median. Full bar is
            # typical; a short bar is motion blur, which is the one thing the
            # driver can fix immediately.
            share = max(0.0, min(1.0, self._sharpness))
            colour = (
                QtGui.QColor(63, 185, 80)
                if share >= 0.75
                else QtGui.QColor(210, 153, 34)
                if share >= 0.45
                else QtGui.QColor(248, 81, 73)
            )
            self._bar(painter, y, share, colour, "sharpness")
            y += 26

            # The instruction. Largest thing on the panel after the map, because
            # it is the only part that asks the user to do something.
            text, level = self._advice
            if text:
                painter.setPen(
                    QtGui.QColor(
                        {"stop": "#f85149", "warn": "#d29922"}.get(level, "#3fb950")
                    )
                )
                font = painter.font()
                font.setPointSize(9)
                font.setBold(level != "ok")
                painter.setFont(font)
                painter.drawText(
                    QtCore.QRect(12, y, PANEL_WIDTH - 24, 40),
                    QtCore.Qt.TextWordWrap,
                    text,
                )
                font.setBold(False)
                font.setPointSize(8)
                painter.setFont(font)
            y += 42

            if self._hotkeys:
                painter.setPen(QtGui.QColor(110, 118, 129))
                painter.drawText(
                    QtCore.QRect(12, y, PANEL_WIDTH - 24, 30),
                    QtCore.Qt.TextWordWrap,
                    self._hotkeys,
                )
            painter.end()

        def _paint_broken(self, painter) -> None:
            reason, count = self._broken
            painter.setBrush(QtGui.QColor(40, 12, 12, 225))
            painter.setPen(QtGui.QPen(QtGui.QColor(248, 81, 73), 2))
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 8, 8)

            painter.setPen(QtGui.QColor(248, 81, 73))
            font = painter.font()
            font.setPointSize(11)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(16, 32, "Feedback stopped")

            font.setBold(False)
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(230, 200, 200))
            painter.drawText(
                QtCore.QRect(16, 44, PANEL_WIDTH - 32, self.height() - 60),
                QtCore.Qt.TextWordWrap,
                "Anything shown before this is out of date — do not read it as "
                f"the scan going well.\n\nThe scan itself is still running, and "
                f"the main window is still correct.\n\n{count}x  {reason}",
            )

        def _bar(self, painter, y, fraction, colour, label) -> None:
            painter.setPen(QtGui.QColor(110, 118, 129))
            painter.drawText(12, y + 6, label)
            left = 12 + 62
            width = PANEL_WIDTH - 24 - 62
            painter.setBrush(QtGui.QColor(33, 38, 45))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(left, y, width, 6, 3, 3)
            painter.setBrush(colour)
            painter.drawRoundedRect(left, y, max(2, int(width * min(1.0, fraction))), 6, 3, 3)

    return Overlay
