"""Following game logs: whole lines only, this sitting only, new and replaced files, both encodings."""

from __future__ import annotations

import os
import time

from heat3d_watch.adapters._logfollow import Follower, newest, read_shared


def test_a_file_made_after_the_game_started_is_read_from_its_first_line(tmp_path):
    since = time.time() - 1
    log = tmp_path / "game.log"
    f = Follower(lambda: log if log.exists() else None, since)
    assert f.lines() == []
    log.write_bytes(b"one\ntwo\nthr")
    assert f.lines() == ["one", "two"]
    with log.open("ab") as out:
        out.write(b"ee\r\nfour\n")
    assert f.lines() == ["three", "four"]
    assert f.lines() == []


def test_an_old_file_is_history_unless_its_lines_say_they_are_new(tmp_path):
    log = tmp_path / "old.log"
    log.write_text("100 before\n200 before too\n", encoding="utf-8")
    since = time.time() + 30  # the game "started" after the file was made
    plain = Follower(lambda: log, since)
    assert plain.lines() == []
    with log.open("a", encoding="utf-8") as out:
        out.write("later\n")
    assert plain.lines() == ["later"]

    log.write_text(f"{since - 100:.0f} last sitting\n{since + 1:.0f} this one\nno stamp, same sitting\n", encoding="utf-8")
    os.utime(log, (since + 2, since + 2))
    stamped = Follower(lambda: log, since, stamp=lambda line: float(line.split()[0]) if line[:1].isdigit() else None)
    assert stamped.lines() == [f"{since + 1:.0f} this one", "no stamp, same sitting"]


def test_a_newer_file_takes_over_after_the_old_one_is_read_out(tmp_path):
    since = time.time() - 1
    a, b = tmp_path / "Journal.1.log", tmp_path / "Journal.2.log"
    f = Follower(lambda: newest(tmp_path.glob("Journal.*.log")), since)
    a.write_text("a1\n", encoding="utf-8")
    assert f.lines() == ["a1"]
    with a.open("a", encoding="utf-8") as out:
        out.write("a2\n")
    b.write_text("b1\n", encoding="utf-8")
    os.utime(a, (time.time() - 0.5, time.time() - 0.5))
    assert f.lines() == ["a2", "b1"]


def test_a_log_replaced_in_place_is_read_again_from_the_start(tmp_path):
    log = tmp_path / "latest.log"
    f = Follower(lambda: log, time.time() - 1)
    log.write_text("first run, a long line\n", encoding="utf-8")
    assert f.lines() == ["first run, a long line"]
    log.write_text("new\n", encoding="utf-8")  # shorter: cut short and rewritten
    assert f.lines() == ["new"]


def test_utf16_logs_are_read_line_by_line(tmp_path):
    log = tmp_path / "chat.txt"
    f = Follower(lambda: log, time.time() - 1)
    log.write_bytes("﻿Ärger\r\nzwei\r\ndr".encode("utf-16-le"))
    assert f.lines() == ["Ärger", "zwei"]
    with log.open("ab") as out:
        out.write("ei\r\n".encode("utf-16-le"))
    assert f.lines() == ["drei"]


def test_the_file_is_not_held_open(tmp_path):
    log = tmp_path / "latest.log"
    log.write_bytes(b"x\n")
    assert read_shared(log) == b"x\n"
    f = Follower(lambda: log if log.exists() else None, time.time() - 1)
    assert f.lines() == ["x"]
    # The game renames its log at the next launch: nothing of ours stands in the way.
    log.rename(tmp_path / "old.log")
    assert f.lines() == []
