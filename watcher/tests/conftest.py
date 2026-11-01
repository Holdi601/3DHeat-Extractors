"""Every test runs against a throwaway data folder, without asking GitHub for updates."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "watcher", ROOT / "gamecapture", ROOT / "cs2"):
    if str(p) not in sys.path:
        sys.path.append(str(p))


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HEAT3D_NO_UPDATE", "1")
    monkeypatch.setenv("HEAT3D_NO_TOAST", "1")
    monkeypatch.setenv("HEAT3D_WATCH_HOME", str(tmp_path / "home"))
    yield
