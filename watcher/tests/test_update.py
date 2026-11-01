"""The updater: a release archive laid over a downloaded copy, the way GitHub serves it."""

from __future__ import annotations

import io
import zipfile

import pytest

import heat3d_update as up


def _zipball(files: dict[str, str]) -> bytes:
    """An archive shaped like GitHub's source zip: everything under one top folder."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Holdi601-3DHeat-Extractors-abc1234/", "")
        for name, text in files.items():
            z.writestr(f"Holdi601-3DHeat-Extractors-abc1234/{name}", text)
    return buf.getvalue()


def _copy(tmp_path):
    root = tmp_path / "tools"
    (root / "watcher" / "heat3d_watch").mkdir(parents=True)
    (root / "VERSION").write_text("1.1.0\n", encoding="utf-8")
    (root / "watcher" / "heat3d_watch" / "old_module.py").write_text("# removed upstream\n", encoding="utf-8")
    (root / "watcher" / "heat3d_watch" / "cli.py").write_text("# 1.1.0\n", encoding="utf-8")
    (root / "watcher" / "settings.json").write_text("{}", encoding="utf-8")  # the user's, not the release's
    return root


def test_a_downloaded_copy_is_brought_to_the_release(tmp_path, monkeypatch):
    root = _copy(tmp_path)
    release = up.Release(version="1.2.0", tag="v1.2.0", notes="", url="", archive="https://example.invalid/zip")
    blob = _zipball({"VERSION": "1.2.0\n", "watcher/heat3d_watch/cli.py": "# 1.2.0\n", "watcher/heat3d_watch/new.py": "x = 1\n"})
    monkeypatch.setattr(up, "ROOT", root)
    monkeypatch.setattr(up, "_get", lambda url: blob)
    monkeypatch.setattr(up, "is_checkout", lambda: False)
    assert "1.2.0" in up.apply(release)
    assert (root / "VERSION").read_text(encoding="utf-8").strip() == "1.2.0"
    assert (root / "watcher" / "heat3d_watch" / "cli.py").read_text(encoding="utf-8") == "# 1.2.0\n"
    assert (root / "watcher" / "heat3d_watch" / "new.py").exists()
    # A module the release no longer has is gone, so it cannot shadow its replacement...
    assert not (root / "watcher" / "heat3d_watch" / "old_module.py").exists()
    # ...and what the user made is untouched.
    assert (root / "watcher" / "settings.json").exists()


def test_an_archive_that_is_not_the_version_it_claims_is_refused(tmp_path, monkeypatch):
    root = _copy(tmp_path)
    release = up.Release(version="1.2.0", tag="v1.2.0", notes="", url="", archive="https://example.invalid/zip")
    monkeypatch.setattr(up, "ROOT", root)
    monkeypatch.setattr(up, "_get", lambda url: _zipball({"VERSION": "1.1.5\n", "watcher/heat3d_watch/cli.py": "bad"}))
    monkeypatch.setattr(up, "is_checkout", lambda: False)
    with pytest.raises(RuntimeError, match="1.1.5"):
        up.apply(release)
    assert (root / "watcher" / "heat3d_watch" / "cli.py").read_text(encoding="utf-8") == "# 1.1.0\n"


def test_paths_that_climb_out_of_the_archive_are_ignored(tmp_path, monkeypatch):
    root = _copy(tmp_path)
    release = up.Release(version="1.2.0", tag="v1.2.0", notes="", url="", archive="https://example.invalid/zip")
    monkeypatch.setattr(up, "ROOT", root)
    monkeypatch.setattr(up, "_get", lambda url: _zipball({"VERSION": "1.2.0\n", "../escaped.txt": "no"}))
    monkeypatch.setattr(up, "is_checkout", lambda: False)
    up.apply(release)
    assert not (tmp_path / "escaped.txt").exists()


def test_a_tool_says_a_newer_release_is_out_and_carries_on_when_nobody_can_answer(monkeypatch, capsys):
    monkeypatch.delenv("HEAT3D_NO_UPDATE", raising=False)
    release = up.Release(version="9.0.0", tag="v9.0.0", notes="", url="https://example.invalid/r", archive="")
    monkeypatch.setattr(up, "latest", lambda use_cache=True: release)
    up.startup("heat3d_test", ask=False)
    err = capsys.readouterr().err
    assert "9.0.0 is out" in err and "heat3d_update" in err
