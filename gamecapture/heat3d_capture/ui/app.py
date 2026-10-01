"""
The scan window.

A thin view over `scan.session` and `scan.reconstruct`: it owns no scanning or
geometry logic, so a scan can be driven from a script and the parts worth testing
are testable without a display.

What the layout is for
----------------------
A scan is a *physical* act — someone walks a level for several minutes — and the
expensive failure is not a crash, it is finishing the walk and discovering it was
no good. Every prominent thing here exists to move that discovery earlier:

- The hardware verdict is shown **before** anything can start, in terms of what
  the scan will be like rather than in frames per second alone.
- The coverage map is the largest thing on screen while scanning, because it is
  the one panel that answers "where do I go next".
- Weak spots are listed with a reason and an instruction, numbered to match rings
  on the map, so "go back to number two and step sideways" is a thing the user
  can actually do while still standing in the level.
- Tracking health is called out separately, because it is the number that
  predicts whether a scan will be usable and the *only* one not visible in the
  pictures: frames keep arriving and depth keeps looking fine right up until the
  geometry turns out to be in pieces.

No jargon in the main flow. Someone who has never heard of a keyframe or an
execution provider should still be able to run this correctly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ..runtime.device import available_backends, describe_expectations, select_backend
from ..scan.session import Quality, ScanProgress, ScanSession, ScanSettings
from .hotkeys import Binding, HotkeyManager
from .hotkeys import available as hotkeys_available
from .preview import coverage_map, depth_to_preview, health_text, mark_weak_spots

#: Chosen as F-keys with a modifier because games almost never bind those,
#: and a global hotkey is taken away from the game for as long as it is held.
HOTKEY_TOGGLE = "ctrl+alt+f9"
HOTKEY_MARK = "ctrl+alt+f10"
HOTKEY_OVERLAY = "ctrl+alt+f11"

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

#: Tier to colour. Tiers are about the experience, so the colours are too: green
#: means walk normally, amber means adjust how you move, red means do not start.
TIER_COLOUR = {
    "realtime": "#3fb950",
    "workable": "#d29922",
    "slow": "#d29922",
    "impractical": "#f85149",
}

#: How often the expensive panels refresh. Clustering the coverage grid and
#: redrawing the map are cheap in absolute terms and not cheap enough to do on
#: every keyframe, and nothing about them changes meaningfully in half a second.
SLOW_REFRESH_MS = 900


def build_window():  # pragma: no cover - constructed only with a display
    from PySide6 import QtCore, QtGui, QtWidgets

    def pixmap(rgb: np.ndarray) -> QtGui.QPixmap:
        h, w = rgb.shape[:2]
        contiguous = np.ascontiguousarray(rgb)
        image = QtGui.QImage(contiguous.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(image.copy())

    class Window(QtWidgets.QMainWindow):
        progressed = QtCore.Signal(object)
        previewed = QtCore.Signal(object, object)
        mapped = QtCore.Signal(object, object)
        # Hotkeys arrive on their own thread; Qt widgets may only be touched
        # from the GUI thread, so the key crosses over as a signal.
        hotkeyPressed = QtCore.Signal(str)

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("3DHeat — Level scanner")
            self.resize(1420, 900)
            self.session: ScanSession | None = None
            self.estimator = None
            self._last_progress: ScanProgress | None = None
            #: Places the driver flagged by hand, which the coverage metric
            #: cannot see: a corner taken badly, a stretch where the view was
            #: blocked by something that has since moved.
            self._marked: list = []
            #: How many times the overlay has failed to update. Counted rather
            #: than flagged, because one failure is a glitch and a hundred is a
            #: panel that has not told the truth since.
            self._overlay_failures = 0

            # Worker threads must not touch widgets. These signals marshal both
            # callbacks onto the GUI thread, which is why the session takes plain
            # callables and knows nothing about Qt.
            self.progressed.connect(self._show_progress)
            self.previewed.connect(self._show_preview)
            self.mapped.connect(self._show_map)
            #: True while a coverage map is being computed off-thread, so the
            #: timer does not pile up work faster than it completes.
            self._mapping = False
            self._map_thread = None
            #: Set on close. A background thread that emits into a window Qt has
            #: already destroyed is not a Python exception — it is an access
            #: violation that takes the process with it.
            self._closing = False
            self._hotkeys = HotkeyManager()
            self.hotkeyPressed.connect(self._on_hotkey)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QHBoxLayout(central)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(14)
            layout.addWidget(self._controls(), 0)
            layout.addWidget(self._viewer(), 1)

            self.overlay = None

            self._slow = QtCore.QTimer(self)
            self._slow.timeout.connect(self._refresh_map)
            self._slow.start(SLOW_REFRESH_MS)

            self._check_hardware()
            # Last, because it takes keys away from the whole machine and should
            # only do so once the window is otherwise ready to use them.
            self._start_hotkeys()

        # -- left column -------------------------------------------------

        def _controls(self) -> QtWidgets.QWidget:
            panel = QtWidgets.QWidget()
            panel.setFixedWidth(390)
            scroll_body = QtWidgets.QVBoxLayout(panel)
            scroll_body.setSpacing(11)

            self.verdict = QtWidgets.QLabel("Checking your graphics card…")
            self.verdict.setWordWrap(True)
            self.verdict.setStyleSheet("padding:10px;border-radius:6px;border:1px solid #30363d;")
            scroll_body.addWidget(self.verdict)

            source_group = QtWidgets.QGroupBox("What to scan")
            source_form = QtWidgets.QFormLayout(source_group)
            self.source_kind = QtWidgets.QComboBox()
            self.source_kind.addItems(["Whole screen", "A window", "A video file"])
            self.source_kind.currentIndexChanged.connect(self._source_kind_changed)
            source_form.addRow("Source", self.source_kind)
            self.source_value = QtWidgets.QLineEdit("1")
            source_form.addRow("Which", self.source_value)
            self.browse = QtWidgets.QPushButton("Choose file…")
            self.browse.clicked.connect(self._browse)
            self.browse.setVisible(False)
            source_form.addRow("", self.browse)
            scroll_body.addWidget(source_group)

            quality_group = QtWidgets.QGroupBox("How carefully")
            quality_box = QtWidgets.QVBoxLayout(quality_group)
            self.quality = QtWidgets.QComboBox()
            for q in Quality:
                # The plain value, not the enum object. Qt stores item data as a
                # QVariant; `Quality` subclasses `str`, so Qt keeps the string
                # and `currentData()` hands back a bare one with the enum lost.
                self.quality.addItem(q.value.capitalize(), q.value)
            self.quality.setCurrentIndex(1)
            self.quality.currentIndexChanged.connect(self._quality_changed)
            quality_box.addWidget(self.quality)
            self.quality_note = QtWidgets.QLabel(Quality.BALANCED.summary)
            self.quality_note.setWordWrap(True)
            self.quality_note.setStyleSheet("color:#8b949e;")
            quality_box.addWidget(self.quality_note)

            fov_row = QtWidgets.QFormLayout()
            self.fov = QtWidgets.QDoubleSpinBox()
            self.fov.setRange(30, 140)
            self.fov.setValue(90)
            self.fov.setSuffix("°")
            self.fov.setToolTip(
                "The game's horizontal field of view. This is the one number the "
                "scanner has to be told rather than measure. Too narrow and the "
                "level reconstructs deeper than it is; too wide and it flattens. "
                "90 suits most first-person games; racing games are nearer 65."
            )
            fov_row.addRow("Game field of view", self.fov)
            self.voxel = QtWidgets.QDoubleSpinBox()
            self.voxel.setRange(0.05, 2.0)
            self.voxel.setSingleStep(0.05)
            self.voxel.setValue(0.25)
            self.voxel.setSuffix(" m")
            self.voxel.setToolTip("Detail of the reconstructed surface. Smaller is finer and heavier.")
            fov_row.addRow("Surface detail", self.voxel)
            quality_box.addLayout(fov_row)
            scroll_body.addWidget(quality_group)

            limits_group = QtWidgets.QGroupBox("Stop after")
            limits_form = QtWidgets.QFormLayout(limits_group)
            self.max_minutes = QtWidgets.QDoubleSpinBox()
            self.max_minutes.setRange(0, 120)
            self.max_minutes.setSuffix(" min")
            self.max_minutes.setSpecialValueText("no limit")
            limits_form.addRow("Time", self.max_minutes)
            self.max_keyframes = QtWidgets.QSpinBox()
            self.max_keyframes.setRange(0, 100_000)
            self.max_keyframes.setValue(3000)
            self.max_keyframes.setSpecialValueText("no limit")
            limits_form.addRow("Views kept", self.max_keyframes)
            scroll_body.addWidget(limits_group)

            clean_group = QtWidgets.QGroupBox("What to ignore")
            clean_box = QtWidgets.QVBoxLayout(clean_group)
            self.mask_hud = QtWidgets.QCheckBox("Ignore HUD and anything on the camera")
            self.mask_hud.setChecked(True)
            self.mask_hud.setToolTip(
                "A crosshair is the most repeatable feature in any frame — it "
                "matches perfectly between every pair and says the camera never "
                "moved, which compresses the whole reconstruction. A weapon or a "
                "car bonnet is real geometry in the wrong place, and fusing it "
                "smears it along the entire path walked.\n\n"
                "Both are found the same way: they are the parts of the picture "
                "that stay still while the world slides past."
            )
            clean_box.addWidget(self.mask_hud)
            self.mask_note = QtWidgets.QLabel("")
            self.mask_note.setWordWrap(True)
            self.mask_note.setStyleSheet("color:#8b949e;font-size:11px;")
            clean_box.addWidget(self.mask_note)
            scroll_body.addWidget(clean_group)

            overlay_group = QtWidgets.QGroupBox("While you play")
            overlay_box = QtWidgets.QVBoxLayout(overlay_group)
            self.use_overlay = QtWidgets.QCheckBox("Show feedback over the game")
            self.use_overlay.setChecked(True)
            self.use_overlay.setToolTip(
                "A small panel pinned over the game with the coverage map, "
                "tracking health and walk speed. Clicks pass straight through it.\n\n"
                "Needs the game in borderless windowed mode — nothing can draw "
                "over exclusive fullscreen, including Steam's and Discord's "
                "overlays."
            )
            self.use_overlay.toggled.connect(self._overlay_toggled)
            overlay_box.addWidget(self.use_overlay)
            hint = QtWidgets.QLabel(
                "Run the game in borderless windowed — exclusive fullscreen "
                "cannot be drawn over by anything."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#8b949e;font-size:11px;")
            overlay_box.addWidget(hint)
            scroll_body.addWidget(overlay_group)

            telemetry_group = QtWidgets.QGroupBox("Racing telemetry (Forza)")
            telemetry_box = QtWidgets.QVBoxLayout(telemetry_group)
            self.telemetry = QtWidgets.QCheckBox("Listen for position data")
            self.telemetry.setToolTip(
                "Forza publishes its own position feed. With it the camera path is "
                "exact instead of estimated. Turn Data Out on in the game, pointed "
                "at 127.0.0.1, and drive a few metres before expecting output."
            )
            telemetry_box.addWidget(self.telemetry)
            self.telemetry_port = QtWidgets.QSpinBox()
            self.telemetry_port.setRange(1, 65535)
            self.telemetry_port.setValue(5300)
            telemetry_box.addWidget(self.telemetry_port)
            self.telemetry_note = QtWidgets.QLabel("")
            self.telemetry_note.setWordWrap(True)
            self.telemetry_note.setStyleSheet("color:#8b949e;font-size:11px;")
            telemetry_box.addWidget(self.telemetry_note)
            scroll_body.addWidget(telemetry_group)

            self.start = QtWidgets.QPushButton("Start scanning")
            self.start.setMinimumHeight(42)
            self.start.clicked.connect(self._toggle)
            scroll_body.addWidget(self.start)

            self.build_model = QtWidgets.QPushButton("Set up the depth model")
            self.build_model.setMinimumHeight(38)
            self.build_model.setVisible(False)
            self.build_model.clicked.connect(self._build_model)
            scroll_body.addWidget(self.build_model)

            self.save = QtWidgets.QPushButton("Save level…")
            self.save.setEnabled(False)
            self.save.clicked.connect(self._save)
            scroll_body.addWidget(self.save)

            self.advanced = QtWidgets.QLabel()
            self.advanced.setWordWrap(True)
            self.advanced.setStyleSheet("color:#6e7681;font-size:11px;")
            scroll_body.addWidget(self.advanced)
            scroll_body.addStretch(1)

            holder = QtWidgets.QScrollArea()
            holder.setWidget(panel)
            holder.setWidgetResizable(True)
            holder.setFixedWidth(410)
            holder.setFrameShape(QtWidgets.QFrame.NoFrame)
            return holder

        # -- right column ------------------------------------------------

        def _viewer(self) -> QtWidgets.QWidget:
            panel = QtWidgets.QWidget()
            box = QtWidgets.QVBoxLayout(panel)
            box.setSpacing(9)

            top = QtWidgets.QHBoxLayout()
            self.view_rgb, self.label_rgb = self._pane("What the camera sees")
            self.view_depth, self.label_depth = self._pane("Shape it is reading")
            top.addWidget(self.view_rgb)
            top.addWidget(self.view_depth)
            box.addLayout(top, 3)

            middle = QtWidgets.QHBoxLayout()
            self.view_map, self.label_map = self._pane("Coverage — where you have been, and how well")
            middle.addWidget(self.view_map, 3)

            spots_group = QtWidgets.QGroupBox("Worth another look")
            spots_box = QtWidgets.QVBoxLayout(spots_group)
            self.spots = QtWidgets.QListWidget()
            self.spots.setStyleSheet("font-size:12px;")
            self.spots.setAlternatingRowColors(True)
            spots_box.addWidget(self.spots)
            self.spots_note = QtWidgets.QLabel("Nothing flagged yet.")
            self.spots_note.setWordWrap(True)
            self.spots_note.setStyleSheet("color:#8b949e;font-size:11px;")
            spots_box.addWidget(self.spots_note)
            middle.addWidget(spots_group, 2)
            box.addLayout(middle, 4)

            self.change_bar = QtWidgets.QProgressBar()
            self.change_bar.setRange(0, 100)
            self.change_bar.setFormat("walk speed — %p% of the way to the next view")
            box.addWidget(self.change_bar)

            self.health = QtWidgets.QLabel("Not scanning.")
            self.health.setWordWrap(True)
            box.addWidget(self.health)

            self.stats = QtWidgets.QLabel("")
            self.stats.setStyleSheet("font-family:Consolas,monospace;color:#c9d1d9;font-size:12px;")
            box.addWidget(self.stats)

            self.hotkey_note = QtWidgets.QLabel("")
            self.hotkey_note.setWordWrap(True)
            self.hotkey_note.setStyleSheet("color:#8b949e;font-size:11px;")
            box.addWidget(self.hotkey_note)

            self.hint = QtWidgets.QLabel("")
            self.hint.setWordWrap(True)
            self.hint.setStyleSheet("color:#d29922;")
            box.addWidget(self.hint)
            return panel

        def _pane(self, title: str):
            frame = QtWidgets.QGroupBox(title)
            inner = QtWidgets.QVBoxLayout(frame)
            label = QtWidgets.QLabel("—")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumSize(300, 190)
            label.setStyleSheet("background:#0d1117;border-radius:4px;color:#484f58;")
            inner.addWidget(label)
            return frame, label

        # -- hardware ----------------------------------------------------

        def _check_hardware(self) -> None:
            backend = select_backend()
            models = sorted(MODELS_DIR.glob("depth-*.onnx"))
            if not models:
                # Telling someone to run a command is not good enough here, and
                # the command in question is worse than most: converting the
                # model needs PyTorch, which is deliberately *not* in this
                # environment, so the instruction would have failed with a
                # ModuleNotFoundError at the next step. Offer to do it instead.
                self._verdict_style("impractical")
                self.verdict.setText(
                    "<b>One more thing to set up.</b><br>"
                    "The depth model has to be converted once before anything can "
                    "be scanned. It takes a few minutes and happens on this machine."
                )
                self.start.setEnabled(False)
                self.build_model.setVisible(True)
                return
            self.build_model.setVisible(False)

            from ..depth.onnx_estimator import OnnxDepthEstimator

            self.verdict.setText(f"Measuring {backend.device}…")
            QtWidgets.QApplication.processEvents()

            self.estimator = OnnxDepthEstimator(models[0], backend=backend)
            fps = self.estimator.measure_throughput(frames=6)
            expectation = describe_expectations(backend, fps)

            self._verdict_style(expectation.tier)
            self.verdict.setText(
                f"<b>{expectation.headline}</b><br>{backend.label} — {fps:.0f} views per second"
                f"<br><span style='color:#8b949e'>{expectation.notes[-1]}</span>"
            )
            self.start.setEnabled(expectation.tier != "impractical")
            others = " · ".join(b.api for b in available_backends())
            setup = expectation.notes[:-1]
            self.advanced.setText(
                f"Model: {models[0].name} · Available: {others}"
                + ("<br>" + "<br>".join(setup) if setup else "")
            )

        def _verdict_style(self, tier: str) -> None:
            colour = TIER_COLOUR[tier]
            self.verdict.setStyleSheet(
                f"padding:10px;border-radius:6px;border:1px solid {colour};color:{colour};"
            )

        # -- interaction -------------------------------------------------

        def _build_model(self) -> None:
            """
            Convert the depth model, fetching the one-off tools if agreed.

            PyTorch is needed for the conversion and for nothing else, which is
            why it is not a dependency of the scanner: it is a gigabyte, and it
            would make the app NVIDIA-shaped for a step that runs once. So it is
            asked for separately, here, at the only moment it is wanted.
            """
            from ..runtime.preflight import MODEL_EXPORT, ensure_graphical, missing

            if missing(MODEL_EXPORT) and not ensure_graphical(MODEL_EXPORT):
                self.verdict.setText(
                    "<b>Nothing was installed.</b><br>The model cannot be built "
                    "without those, and scanning cannot start without the model."
                )
                return

            self.build_model.setEnabled(False)
            self.build_model.setText("Converting… this takes a few minutes")
            self.verdict.setText("Converting the depth model. The window will not respond.")
            QtWidgets.QApplication.processEvents()

            import subprocess

            try:
                # A subprocess, not an import: the converter pulls PyTorch into
                # memory and leaves it there, and this process is about to spend
                # the rest of its life holding video frames instead.
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve().parents[2] / "tools" / "export_onnx.py"),
                        "--size",
                        "small",
                    ],
                    cwd=str(Path(__file__).resolve().parents[2]),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                    timeout=1800,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                result = None
                message = str(exc)

            self.build_model.setEnabled(True)
            self.build_model.setText("Set up the depth model")

            if result is not None and result.returncode == 0:
                self._check_hardware()
                return

            detail = (result.stdout + result.stderr)[-600:] if result is not None else message
            QtWidgets.QMessageBox.critical(
                self,
                "Could not build the model",
                f"The conversion did not finish.\n\n{detail}",
            )
            self.verdict.setText(
                "<b>The model could not be built.</b><br>"
                "Nothing was changed. The details are in the message above."
            )

        def _overlay_toggled(self, on: bool) -> None:
            if not on:
                self._close_overlay()
            elif self.session is not None and self.session.is_running():
                self._open_overlay()

        def _open_overlay(self) -> None:
            if self.overlay is not None:
                return
            from .overlay import build_overlay

            self.overlay = build_overlay()()
            # Shown on the overlay too, because someone driving cannot look them
            # up in a window that is behind the game.
            self.overlay.set_hotkeys(self._hotkeys.describe())
            self.overlay.show()

        def _close_overlay(self) -> None:
            if self.overlay is not None:
                self.overlay.close()
                self.overlay = None

        def _source_kind_changed(self, index: int) -> None:
            self.browse.setVisible(index == 2)
            self.source_value.setText({0: "1", 1: "", 2: ""}[index])
            self.source_value.setPlaceholderText(
                {0: "monitor number", 1: "window title", 2: "path to a video"}[index]
            )

        def _quality(self) -> Quality:
            """The chosen preset, rebuilt from what Qt actually stored."""
            return Quality(self.quality.currentData())

        def _quality_changed(self) -> None:
            self.quality_note.setText(self._quality().summary)

        def _browse(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Choose a recording", "", "Video (*.mp4 *.mkv *.avi *.mov);;All files (*)"
            )
            if path:
                self.source_value.setText(path)

        def _target(self) -> str:
            value = self.source_value.text().strip()
            return {0: f"monitor:{value or '1'}", 1: f"window:{value}", 2: value}[
                self.source_kind.currentIndex()
            ]

        def _settings(self) -> ScanSettings:
            return ScanSettings(
                source=self._target(),
                quality=self._quality(),
                max_seconds=self.max_minutes.value() * 60.0,
                max_keyframes=self.max_keyframes.value(),
                use_telemetry=self.telemetry.isChecked(),
                telemetry_port=self.telemetry_port.value(),
                field_of_view=self.fov.value(),
                voxel=self.voxel.value(),
                reconstruct=True,
                mask_hud=self.mask_hud.isChecked(),
            )

        # -- hotkeys ------------------------------------------------------

        def _shortcut(self, name: str) -> str:
            """
            The key for an action, as a suffix, or nothing.

            Nothing when the key could not be registered — a button that advertises
            a shortcut it does not have is worse than one that advertises none,
            because the user only discovers it mid-capture, which is the one moment
            the hotkey exists to protect.
            """
            for binding in self._hotkeys.bindings:
                if binding.name == name:
                    return f"   {binding.pretty()}"
            return ""

        def _label_buttons(self) -> None:
            """Put the keys on the controls they belong to, once they are known."""
            self.start.setText(self._start_label())
            self.use_overlay.setText(
                "Show feedback over the game" + self._shortcut("overlay")
            )

        def _start_label(self) -> str:
            running = bool(self.session and self.session.is_running())
            return ("Stop scanning" if running else "Start scanning") + self._shortcut(
                "toggle"
            )

        def _start_hotkeys(self) -> None:
            """
            Register the global keys, and say plainly if any could not be had.

            Registered once at start-up rather than around each scan: a key that
            appears in the interface has to work when it is read, not only while
            something is already running.
            """
            if not hotkeys_available():
                self.hotkey_note.setText("Global hotkeys need Windows; use the buttons.")
                return
            taken = self._hotkeys.start(
                [
                    Binding(
                        "toggle",
                        HOTKEY_TOGGLE,
                        lambda: self.hotkeyPressed.emit("toggle"),
                        "start / stop scanning",
                    ),
                    Binding(
                        "mark",
                        HOTKEY_MARK,
                        lambda: self.hotkeyPressed.emit("mark"),
                        "mark this spot to revisit",
                    ),
                    Binding(
                        "overlay",
                        HOTKEY_OVERLAY,
                        lambda: self.hotkeyPressed.emit("overlay"),
                        "show / hide the overlay",
                    ),
                ]
            )
            self.hotkey_note.setText(
                self._hotkeys.describe()
                if taken
                else "No hotkey could be registered — another program has them."
            )
            self._label_buttons()

        def _on_hotkey(self, name: str) -> None:
            """Runs on the GUI thread, because the signal crossed for that reason."""
            if self._closing:
                return
            if name == "toggle":
                self._toggle()
            elif name == "overlay":
                self.use_overlay.setChecked(not self.use_overlay.isChecked())
            elif name == "mark":
                self._mark_here()

        def _mark_here(self) -> None:
            """
            Note the current position as somewhere to come back to.

            The coverage map already marks what it thinks is thin. This is for
            what the *driver* noticed and the metric cannot see — a corner taken
            badly, a stretch where the view was blocked.
            """
            session = self.session
            if session is None or not session.is_running():
                return
            positions = getattr(session, "reconstructor", None)
            if positions is None:
                return
            path = positions.trajectory.positions
            if len(path) == 0:
                return
            self._marked.append(path[-1].copy())
            self.hint.setText(f"Marked {len(self._marked)} spot(s) to revisit.")

        def _toggle(self) -> None:
            if self.session and self.session.is_running():
                self.start.setEnabled(False)
                self.start.setText("Stopping…")
                QtWidgets.QApplication.processEvents()
                self.session.stop()
                self._close_overlay()
                self.start.setEnabled(True)
                self.start.setText(self._start_label())
                self._refresh_map()
                return

            self.spots.clear()
            self.session = ScanSession(
                self._settings(),
                estimator=self.estimator,
                on_progress=self.progressed.emit,
                on_preview=self.previewed.emit,
            )
            self.session.start()
            self.start.setText(self._start_label())
            self.save.setEnabled(False)
            self.hint.setText("")
            if self.use_overlay.isChecked():
                self._open_overlay()

        def _save(self) -> None:
            if not self.session:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save the level", "scan.glb", "glTF binary (*.glb)"
            )
            if not path:
                return
            written = self.session.export(Path(path), name=Path(path).stem)
            if written is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Nothing to save",
                    "The scan did not produce enough geometry to build a surface. "
                    "That usually means tracking was lost for most of it — the "
                    "coverage map shows how much was actually placed.",
                )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Saved",
                    f"Written to {written}\n\nLoad it in 3DHeat through the Level panel.",
                )

        # -- live display ------------------------------------------------

        def _show_progress(self, progress: ScanProgress) -> None:
            self._last_progress = progress
            threshold = self._quality().keyframe_change
            self.change_bar.setValue(int(min(1.0, progress.change / threshold) * 100))

            state = progress.reconstruction
            line = (
                f"{progress.keyframes:>5} views   {progress.frames_seen:>6} frames   "
                f"{progress.frames_dropped:>5} dropped   "
                f"{progress.fps:>5.1f}/s   {progress.elapsed:>6.1f}s"
            )
            if state is not None:
                line += (
                    f"   ·   {state.fused:>5} placed   {state.surface_voxels:>7} cells   "
                    f"{state.well_observed * 100:>4.0f}% well seen   "
                    f"{state.trajectory_length:>6.1f} m"
                )
                # Recoveries and closures are shown rather than absorbed: they
                # are the two things that explain a track which is nonetheless
                # not a straight line, and hiding them leaves drift unexplained.
                if state.relocalisations or state.loop_closures:
                    line += (
                        f"   ·   {state.relocalisations} recovered   "
                        f"{state.loop_closures} loops closed"
                    )
                self.mask_note.setText(state.mask_note)
            self.stats.setText(line)

            if progress.telemetry_frames:
                self.telemetry_note.setText(
                    f"{progress.telemetry_frames} packets — poses come from the game, "
                    "not from the picture"
                )
            elif self.telemetry.isChecked() and progress.running:
                self.telemetry_note.setText(
                    "no packets yet — check Data Out is on and pointed at 127.0.0.1"
                )

            text, colour = health_text(state)
            if state is not None and state.capture_note:
                # This outranks tracking health, and by a long way. Tracking can
                # report every frame placed while the capture carries no shape
                # information at all — which is exactly how a three-minute drive
                # turned into a handful of disconnected blobs.
                text = state.capture_note
                colour = "#f85149" if not state.reconstructable else "#d29922"
            self.health.setText(text)
            self.health.setStyleSheet(f"color:{colour};")

            if self.overlay is not None:
                # An overlay that throws must say so rather than freeze. It
                # keeps painting whatever it last drew, so a panel that died
                # while green goes on reporting green for the rest of the run —
                # which is worse than showing nothing, because the driver reads
                # it as confirmation. Seen for real: a race in the coverage map
                # killed the update on the first large keyframe and the overlay
                # stayed reassuring for a whole lap.
                try:
                    self.overlay.update_from(
                        self.session.reconstructor if self.session else None, progress
                    )
                except Exception as exc:  # noqa: BLE001 - shown, not swallowed
                    self._overlay_failures += 1
                    self.overlay.report_broken(
                        f"{type(exc).__name__}: {exc}", self._overlay_failures
                    )
                    self.hint.setText(
                        f"The overlay stopped updating ({type(exc).__name__}). "
                        "The scan is still running; this window is still correct."
                    )

            if not progress.running:
                self.start.setText(self._start_label())
                self.hint.setText(progress.message)
                self.save.setEnabled(state is not None and state.surface_voxels > 0)
                self._close_overlay()
                return

            if progress.frames_dropped > progress.frames_seen * 0.5 and progress.elapsed > 6:
                self.hint.setText(
                    f"Falling behind — {progress.frames_dropped} frames dropped. "
                    "Reconstruction is slower than capture, which pulls the kept "
                    "views apart until tracking cannot bridge them. Try Draft "
                    "quality, or a coarser surface detail."
                )
            elif progress.elapsed > 8 and progress.keyframe_rate < 0.3:
                self.hint.setText(
                    "Very little is being kept — you may be standing still, or looking at "
                    "something featureless. Move sideways rather than turning on the spot."
                )
            elif progress.elapsed > 8 and progress.keyframe_rate > 8:
                self.hint.setText("Keeping a lot. Slow down a little for cleaner geometry.")
            else:
                self.hint.setText("")

        def _show_preview(self, rgb: np.ndarray, depth: np.ndarray) -> None:
            for label, image in ((self.label_rgb, rgb), (self.label_depth, depth_to_preview(depth))):
                label.setPixmap(
                    pixmap(image).scaled(
                        label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                    )
                )

        def _refresh_map(self) -> None:
            """
            Kick off a coverage map, off the interface thread.

            This used to compute in place, on the timer, and it was the single
            worst bug in the program: clustering weak coverage over a driven
            route took **13.5 seconds** at 600 m, on a 900 ms timer, on the
            thread that paints and handles clicks. The window locked solid — the
            stop button did nothing, no progress was ever drawn, and the health
            line sat on "waiting for the first frame" because the queued updates
            were never processed. It looked like the scan had failed to start.
            The scan was fine; the interface could not tell anyone.

            The clustering is far cheaper now, but that is not the fix and could
            not be: any work proportional to a growing map will eventually
            outrun a fixed timer. Nothing that scales with the scan may run here.
            """
            import threading

            session = self.session
            if session is None or session.reconstructor is None or self._mapping:
                return
            reconstructor = session.reconstructor
            self._mapping = True

            def work() -> None:
                try:
                    grid, extent = reconstructor.coverage.top_down(resolution=200)
                    spots = reconstructor.refresh_weak_spots()
                    image = coverage_map(
                        grid,
                        trajectory=reconstructor.trajectory.positions,
                        extent=extent,
                        size=420,
                    )
                    result = (mark_weak_spots(image, spots, extent), spots)
                except Exception:
                    # A map that could not be drawn must not take the scan with
                    # it: the geometry is still being built correctly either way.
                    result = (None, [])
                # Checked as late as possible, and again by the receiving slot.
                # Emitting into a window that Qt has already torn down is an
                # access violation, not an exception — the process simply dies,
                # which is how closing the window mid-scan used to end.
                if not self._closing:
                    self.mapped.emit(*result)

            self._map_thread = threading.Thread(target=work, daemon=True)
            self._map_thread.start()

        def _show_map(self, image, spots) -> None:
            self._mapping = False
            if image is None or self._closing:
                return
            self.label_map.setPixmap(
                pixmap(image).scaled(
                    self.label_map.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )
            self.spots.clear()
            for i, spot in enumerate(spots, start=1):
                self.spots.addItem(f"{i}.  {spot.advice}")
            if spots:
                self.spots_note.setText(
                    "Numbered rings on the map. Worst first — the biggest patches of "
                    "unreliable geometry."
                )
            else:
                self.spots_note.setText(
                    "Nothing flagged yet."
                    + (
                        f"  Press{self._shortcut('mark')} while driving to mark a "
                        "spot yourself."
                        if self._shortcut("mark")
                        else ""
                    )
                )

        def closeEvent(self, event) -> None:
            # Order matters. The flag first, so anything already in flight knows
            # not to touch this window; then the timer, so nothing new starts;
            # then the waits.
            self._closing = True
            self._slow.stop()
            # Before anything else waits: a global hotkey left registered is
            # taken away from every other program on the machine.
            self._hotkeys.stop()
            if self.session and self.session.is_running():
                self.session.stop()
            if self._map_thread is not None and self._map_thread.is_alive():
                # Joined rather than abandoned. A daemon thread holding a
                # reference to a half-destroyed window is a crash looking for a
                # moment, and the map work is bounded anyway.
                self._map_thread.join(timeout=5.0)
                self._map_thread = None
            self._close_overlay()
            super().closeEvent(event)

    return Window


def application():
    """
    The process's one QApplication, built if it does not exist yet.

    Extracted and tested rather than inlined, because the fault it guards against
    only appears when something *else* built one first — and that only happens
    when a dependency was missing and the check put a dialog up. On a machine
    with everything installed the dangerous path is never taken, which is exactly
    why the first version of this shipped broken and passed every test here.

    Qt permits one QApplication per process. Constructing a second is a hard
    error, and `del` on the first does not help: it drops a Python reference
    while the C++ singleton lives on.
    """
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def main() -> int:  # pragma: no cover - entry point
    app = application()
    app.setStyle("Fusion")
    window = build_window()()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
