"""
`python -m heat3d_capture.ui` — the scan window.

Everything here runs *before* the application imports anything heavy, which is
the point: the dependency check cannot live inside a module that already needs
numpy to be importable in order to be read.
"""

from ..runtime.preflight import SCANNER, ensure_graphical

if not ensure_graphical(SCANNER):
    raise SystemExit(
        "The scanner needs those packages to run. Nothing was installed and "
        "nothing was changed."
    )

from .app import main  # noqa: E402  - deliberately after the check

raise SystemExit(main())
