"""
Does the program actually start.

This exists because it did not, and nothing in the rest of the suite noticed.

Every other test that touches the interface builds a `QApplication` itself and
then calls `build_window()`. That is a convenient way to test a window and it
skips the one sequence a user performs: run the module, let the dependency check
put a dialog up, then let `main()` build the application. Qt permits exactly one
`QApplication` per process, the check was creating one and dropping the Python
reference — which does not end the C++ singleton — and `main()` then died on the
first line it ran. First launch, every time, before anything else had a chance.

The lesson generalises past Qt: a component test that constructs its own
environment cannot see a fault in how the environment is constructed. So this
runs the real entry point, as a subprocess, the way the batch file does.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Long enough to get through the dependency check, the backend probe and the
#: first model load; short enough not to stall the suite. It only has to reach
#: the event loop — staying there is the pass.
STARTUP_SECONDS = 25


def _launch(module: str) -> subprocess.Popen:
    environment = dict(os.environ)
    # No display on a test runner, and none needed: the faults this catches
    # happen before anything is painted.
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.mark.skipif(
    not list(ROOT.glob("models/depth-*.onnx")),
    reason="no exported model; the window refuses to start without one",
)
def test_the_scanner_starts_and_stays_up():
    process = _launch("heat3d_capture.ui")
    deadline = time.monotonic() + STARTUP_SECONDS
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                pytest.fail(
                    f"the scanner exited on its own with code {process.returncode}\n\n{output}"
                )
            time.sleep(0.25)
    finally:
        process.terminate()
        try:
            output = process.communicate(timeout=10)[0] or ""
        except subprocess.TimeoutExpired:
            process.kill()
            output = process.communicate()[0] or ""

    # Reaching the event loop is not enough on its own: Qt reports plenty of
    # fatal-in-practice problems on stderr while carrying on regardless.
    assert "Traceback" not in output, f"the scanner printed a traceback:\n\n{output}"
    assert "QApplication" not in output, (
        f"a QApplication problem reached the output:\n\n{output}"
    )


class TestFreshInstall:
    """
    The state a new machine is actually in: installed, nothing converted yet.

    The first version told the user to run `tools/export_onnx.py`, which would
    have failed on the next line with a missing PyTorch — the converter needs it
    and the scanner deliberately does not ship it. An instruction that cannot be
    followed is worse than an error, because it looks like the user's fault.
    """

    def test_it_offers_to_build_the_model_rather_than_naming_a_command(
        self, tmp_path, monkeypatch
    ):
        from PySide6 import QtWidgets

        from heat3d_capture.ui import app as app_module

        # Point the window at an empty directory instead of moving the real
        # model about, which would break a parallel run and leave the tree worse
        # off if this test failed.
        monkeypatch.setattr(app_module, "MODELS_DIR", tmp_path)
        app_module.application()

        window = app_module.build_window()()
        try:
            assert window.start.isEnabled() is False, "scanning must not start without a model"
            assert window.build_model.isVisibleTo(window), "no way offered to build it"
            # The dead-end instruction must be gone.
            assert "export_onnx" not in window.verdict.text()
        finally:
            window.close()

    def test_the_converter_is_asked_for_separately_from_the_scanner(self):
        # PyTorch is a gigabyte and NVIDIA-shaped, for a step that runs once. It
        # belongs in the conversion manifest and nowhere near the runtime one.
        from heat3d_capture.runtime.preflight import MODEL_EXPORT, SCANNER

        assert "torch" in {r.package for r in MODEL_EXPORT}
        assert "torch" not in {r.package for r in SCANNER}


class TestApplicationSingleton:
    """
    The exact fault that shipped, reproduced without needing a broken install.

    The subprocess test above cannot see this one: the dangerous path is only
    taken when a dependency is missing, so on a working machine it never runs.
    That is precisely how the bug got out, and testing the invariant directly is
    the answer — not a bigger integration test.
    """

    def test_a_second_call_adopts_the_first_application(self):
        from heat3d_capture.ui.app import application

        first = application()
        # As `main()` does after the dependency check has already built one. The
        # broken version constructed a new QApplication here and died with
        # "please destroy the QApplication singleton before creating a new one".
        second = application()
        assert second is first

    def test_the_preflight_leaves_a_usable_application_behind(self, monkeypatch):
        from PySide6 import QtWidgets

        from heat3d_capture.runtime import preflight
        from heat3d_capture.ui.app import application

        # The dialog is stubbed to decline. Left real it blocks forever under an
        # offscreen platform, because a modal box with nobody to click it is
        # exactly that — and a test that hangs teaches nothing.
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "exec", lambda self: QtWidgets.QMessageBox.Cancel
        )
        monkeypatch.setattr(preflight, "install", lambda *a, **k: (True, "not run"))

        absent = preflight.Requirement(
            "no-such-package-xyz", "no_such_module_xyz", "nothing", 1
        )
        assert preflight.ensure_graphical([absent]) is False
        # Whatever the check built or adopted, `main()` must still get one.
        assert application() is QtWidgets.QApplication.instance()
