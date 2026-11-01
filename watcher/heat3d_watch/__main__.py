import sys
from pathlib import Path

# The updater and the recorders it shares code with live beside this package,
# in the same repository: make them importable when started from this folder.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "gamecapture", _ROOT / "cs2"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from .cli import main  # noqa: E402

raise SystemExit(main())
