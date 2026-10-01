"""
Check what is missing before anything needs it, and offer to fetch it.

The behaviour this replaces is a traceback. `ModuleNotFoundError: No module named
'cv2'` is a correct and useless thing to show someone who just wanted to scan a
level: it names one missing piece out of possibly several, says nothing about how
to get it, and looks like the program is broken rather than incomplete.

Three rules here, and they are the whole design:

- **Check everything first.** Fixing one missing package only to hit the next is
  the worst version of this. The full list is gathered before anything is said.
- **Ask, never assume.** Installing software on someone's machine without being
  told to is not acceptable regardless of how convenient it is, and the download
  can be hundreds of megabytes on a connection they are paying for. The size is
  shown, and nothing happens without a yes.
- **Explain in terms of what it is for.** "opencv-python" means nothing; "reading
  video files and camera frames" does.

Installation goes into the running interpreter's own environment, which for the
scanner is the private one next to `scan.bat`. It is deliberately not a system
install: the machine this was built on already had three conflicting copies of
onnxruntime in its global Python from unrelated projects, and adding a fourth
would have been the wrong kind of helpful.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    """One thing that has to be present, described for a human."""

    #: What `pip install` is given.
    package: str
    #: What `import` is given. Often not the same word.
    module: str
    #: What it is *for*, in the user's terms rather than the library's.
    purpose: str
    #: Rough download size in megabytes, for the prompt.
    megabytes: int
    #: False when the feature degrades rather than failing without it.
    essential: bool = True

    def present(self) -> bool:
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            # A half-removed package can leave a spec that raises on lookup,
            # which for our purposes is the same as being absent.
            return False


#: Everything the scanner needs, with the reason it needs it.
SCANNER: list[Requirement] = [
    Requirement("numpy", "numpy", "the arithmetic everything else is built on", 20),
    Requirement("opencv-python", "cv2", "reading video files and camera frames", 40),
    Requirement("onnxruntime-directml", "onnxruntime", "running the depth model on your graphics card", 120),
    Requirement("PySide6-Essentials", "PySide6", "the window and its controls", 90),
    Requirement("scipy", "scipy", "smoothing the reconstructed surface", 45),
    Requirement("scikit-image", "skimage", "turning measurements into a 3D surface", 30),
    Requirement("windows-capture", "windows_capture", "capturing a game window live", 5, essential=False),
]

#: Only needed to convert a checkpoint, once, and deliberately not a runtime
#: dependency — it is by far the largest thing here and the app never imports it.
MODEL_EXPORT: list[Requirement] = [
    Requirement("torch", "torch", "converting the depth model, one time only", 900),
    Requirement("transformers", "transformers", "loading the published depth model", 30),
    Requirement("onnxscript", "onnxscript", "writing the converted model out", 15),
]


def missing(requirements: list[Requirement], *, essential_only: bool = False) -> list[Requirement]:
    """Everything absent, in the order it was declared."""
    return [
        r for r in requirements if (r.essential or not essential_only) and not r.present()
    ]


def describe(absent: list[Requirement]) -> str:
    """The prompt text. Written to be read by someone who did not choose these."""
    if not absent:
        return "Everything needed is already installed."
    total = sum(r.megabytes for r in absent)
    lines = [
        f"{len(absent)} thing{'s' if len(absent) != 1 else ''} still needed "
        f"(about {total} MB to download):",
        "",
    ]
    lines += [f"  • {r.package} — {r.purpose}" for r in absent]
    lines += [
        "",
        "These install into this tool's own folder, not into your system Python, "
        "so nothing else on the machine is affected.",
    ]
    return "\n".join(lines)


def install(
    absent: list[Requirement],
    *,
    index_url: str | None = None,
    on_output=None,
) -> tuple[bool, str]:
    """
    Install them, reporting progress line by line. Returns (ok, last message).

    Run as a subprocess against this interpreter rather than through pip's
    internals, which are explicitly not an API and change between versions.
    """
    if not absent:
        return True, "nothing to install"

    command = [sys.executable, "-m", "pip", "install", *(r.package for r in absent)]
    if index_url:
        command += ["--index-url", index_url]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return False, f"could not start pip: {exc}"

    last = ""
    assert process.stdout is not None
    for line in process.stdout:
        last = line.rstrip()
        if on_output and last:
            on_output(last)
    code = process.wait()
    if code != 0:
        return False, last or f"pip exited with {code}"

    # Newly installed packages are not visible to an interpreter that already
    # failed to find them, because the failure is cached in the import system.
    importlib.invalidate_caches()
    return True, "installed"


def ensure_console(requirements: list[Requirement], *, assume_yes: bool = False) -> bool:
    """
    Check and, with consent, install — from a terminal.

    The fallback for when there is no display, or when the missing package *is*
    the one that would have drawn the dialog.
    """
    absent = missing(requirements)
    if not absent:
        return True

    print()
    print(describe(absent))
    print()
    if not assume_yes:
        try:
            answer = input("Install them now? [Y/n] ").strip().lower()
        except EOFError:
            # No terminal to ask at — a double-clicked shortcut, or a pipe.
            # Refusing is the safe reading of silence.
            print("No terminal to confirm at; not installing.")
            return False
        if answer not in ("", "y", "yes", "j", "ja"):
            print("Nothing installed.")
            return False

    ok, message = install(absent, on_output=lambda line: print("   ", line))
    print("Done." if ok else f"Install failed: {message}")
    return ok


def ensure_graphical(requirements: list[Requirement]) -> bool:
    """
    The same, as a dialog, falling back to the console when Qt is not there.

    Qt is itself one of the requirements, so this cannot assume it exists — which
    is exactly the case the console path is for.
    """
    absent = missing(requirements)
    if not absent:
        return True
    if importlib.util.find_spec("PySide6") is None:
        return ensure_console(requirements)

    from PySide6 import QtWidgets

    # Reused, never replaced. Qt allows exactly one QApplication per process and
    # `del` does not end it — it drops a Python reference while the C++ singleton
    # lives on. The first version did exactly that, and the next QApplication the
    # program tried to build died with "please destroy the QApplication singleton
    # before creating a new one", on the very first launch, before anything else
    # had a chance to run.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    del app  # noqa: F841 - kept alive by Qt itself; see above

    box = QtWidgets.QMessageBox()
    box.setWindowTitle("3DHeat scanner — setup")
    box.setIcon(QtWidgets.QMessageBox.Question)
    box.setText("Some things need downloading before the scanner can run.")
    box.setInformativeText(describe(absent))
    box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
    box.button(QtWidgets.QMessageBox.Yes).setText("Download and install")
    box.setDefaultButton(QtWidgets.QMessageBox.Yes)
    if box.exec() != QtWidgets.QMessageBox.Yes:
        return False

    progress = QtWidgets.QProgressDialog("Installing…", None, 0, 0)
    progress.setWindowTitle("3DHeat scanner — setup")
    progress.setMinimumWidth(520)
    progress.setCancelButton(None)
    progress.show()

    def report(line: str) -> None:
        progress.setLabelText(line[-110:])
        QtWidgets.QApplication.processEvents()

    ok, message = install(absent, on_output=report)
    progress.close()

    if not ok:
        QtWidgets.QMessageBox.critical(
            None,
            "3DHeat scanner — setup",
            f"The install did not finish.\n\n{message}\n\n"
            f"You can try by hand:\n{sys.executable} -m pip install "
            + " ".join(r.package for r in absent),
        )
    # Deliberately left alive for whatever runs next. See above.
    return ok
