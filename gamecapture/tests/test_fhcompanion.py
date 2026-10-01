"""
Reading a library of recorded laps.

These laps are the best telemetry this project has: already driven, already
repeated, and in the game's own world coordinates — which is what lets a course
pick its own stretch of map out of the shipped archive. So what is tested here is
mostly about not corrupting that: the coordinates have to survive the trip to
disk at full precision, the course's extent has to be the extent of every lap
rather than one of them, and a single bad file among hundreds must not cost the
rest.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from heat3d_capture.telemetry.fhcompanion import (
    NoLapLibrary,
    courses,
    find_course,
    read_course,
    read_lap,
)


def lap_document(
    *,
    seconds: float = 47.737,
    positions=None,
    car_class: str = "A",
    ordinal: int = 382,
    recorded: str = "2026-09-20T12:48:59.3275462+02:00",
):
    points = positions if positions is not None else [
        (2784.9712, 461.6221, 4994.9365),
        (2790.1208, 462.07178, 4996.732),
        (2795.1418, 462.50894, 4998.4834),
    ]
    samples = []
    for i, (x, y, z) in enumerate(points):
        samples.append(
            {
                "Metres": i * 5.3,
                "Seconds": i * 0.085,
                "X": x,
                "Y": y,
                "Z": z,
                "Speed": 64.3 + i,
                "Throttle": 1,
                "Brake": 0,
                "Steer": 0,
                "LatG": -0.61296207,
                "LongG": 1.8548198,
                "Clutch": 0,
                "HandBrake": 0,
                "Gear": 5,
                "Puddle": 0,
            }
        )
    return {
        "Course": "course_2800_5000_to_2775_5000",
        "Tag": None,
        "Class": car_class,
        "Lap": {
            "lapSeconds": seconds,
            "lengthMetres": 1908.9011,
            "carOrdinal": ordinal,
            "performanceIndex": 700,
            "track": "Lakeside Circuit",
            "recordedAt": recorded,
            "StandingStart": False,
            "samples": samples,
        },
    }


def write_lap(folder, name="lap.json", **kwargs):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    # A byte-order mark, as the .NET tool that writes these produces.
    path.write_text(
        "﻿" + json.dumps(lap_document(**kwargs)), encoding="utf-8"
    )
    return path


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "laps"
    course = root / "Lakeside Circuit (course_2800_5000_to_2775_5000)"
    course.mkdir(parents=True)
    (course / "course.json").write_text(
        json.dumps(
            {
                "Name": "Lakeside Circuit",
                "StartX": 2788.0515,
                "StartZ": 4990.413,
                "FinishX": 2783.8538,
                "FinishZ": 4994.907,
                "Laps": 2,
            }
        ),
        encoding="utf-8",
    )
    write_lap(course / "A" / "car382" / "tune" / "untagged", "fast.json", seconds=47.737)
    write_lap(
        course / "B" / "car1655" / "tune" / "untagged",
        "slow.json",
        seconds=52.103,
        car_class="B",
        ordinal=1655,
        recorded="2026-09-21T12:48:59+02:00",
        positions=[
            (2418.7, 420.8, 4690.6),
            (2500.0, 440.0, 4800.0),
            (2973.2, 475.7, 5039.4),
        ],
    )
    (root / "Other Sprint (course_0_0_to_100_100)").mkdir()
    return root


class TestReadingALap:
    def test_the_byte_order_mark_does_not_stop_it(self, tmp_path):
        """
        The files are written by a .NET tool and begin with a BOM, which `json`
        refuses as plain utf-8. Cheap to handle and baffling to debug.
        """
        path = write_lap(tmp_path)

        lap = read_lap(path)

        assert len(lap) == 3

    def test_every_channel_comes_through(self, tmp_path):
        """
        Fifteen fields, and the fourteen that are not position are the analysis:
        where the braking happens, where the grip runs out, what gear it was in.
        Reading position only would make this a route recorder, which the project
        already has.
        """
        lap = read_lap(write_lap(tmp_path))

        for name in ("speed_ms", "throttle", "brake", "steer", "lat_g", "long_g", "gear"):
            assert name in lap.channels
            assert len(lap.channels[name]) == 3

    def test_the_positions_are_the_game_s_own(self, tmp_path):
        lap = read_lap(write_lap(tmp_path))

        assert lap.positions[0] == pytest.approx([2784.9712, 461.6221, 4994.9365])

    def test_a_lap_with_no_samples_is_refused(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"Lap": {"samples": []}}), encoding="utf-8")

        with pytest.raises(ValueError, match="no samples"):
            read_lap(path)


class TestReadingACourse:
    def test_it_finds_laps_however_deep_they_sit(self, library):
        """Class, then car, then tune, then tag - four levels, and the shape of
        that tree is the recording tool's business rather than this one's."""
        course = read_course(library / "Lakeside Circuit (course_2800_5000_to_2775_5000)")

        assert len(course) == 2

    def test_the_extent_is_every_lap_not_the_best_one(self, library):
        """
        The extent decides how much map gets cut. Taking it from one lap loses
        whatever the others did differently - and the interesting laps are
        exactly the ones that went somewhere the quick one did not.
        """
        course = read_course(library / "Lakeside Circuit (course_2800_5000_to_2775_5000)")

        low, high = course.bounds

        assert low[0] == pytest.approx(2418.7)
        assert high[0] == pytest.approx(2973.2)

    def test_the_start_and_finish_come_from_the_metadata(self, library):
        course = read_course(library / "Lakeside Circuit (course_2800_5000_to_2775_5000)")

        assert course.start == pytest.approx((2788.0515, 4990.413))
        assert course.closed, "start and finish a car's length apart is a circuit"

    def test_a_missing_metadata_file_falls_back_to_the_folder_name(self, tmp_path):
        """
        The coordinates are in the folder name too, rounded. A course driven
        once may not have the file yet, and the name is enough.
        """
        folder = tmp_path / "Hill Scramble (course_5900_1825_to_5900_1825)"
        write_lap(folder / "A" / "car1" / "t" / "untagged")

        course = read_course(folder)

        assert course.name == "Hill Scramble"
        assert course.start == (5900.0, 1825.0)

    def test_a_corrupt_metadata_file_does_not_cost_the_laps(self, tmp_path):
        folder = tmp_path / "Hill Scramble (course_5900_1825_to_5900_1825)"
        folder.mkdir()
        (folder / "course.json").write_text("{ this is not json", encoding="utf-8")
        write_lap(folder / "A" / "car1" / "t" / "untagged")

        course = read_course(folder)

        assert len(course) == 1
        assert course.start == (5900.0, 1825.0)

    def test_one_unreadable_lap_does_not_cost_the_course(self, library):
        """One file out of 859, and stopping on it would mean a library is all
        or nothing."""
        folder = library / "Lakeside Circuit (course_2800_5000_to_2775_5000)"
        (folder / "A" / "car382" / "tune" / "untagged" / "broken.json").write_text(
            "not json at all", encoding="utf-8"
        )

        course = read_course(folder)

        assert len(course) == 2

    def test_the_best_lap_is_the_fastest_one(self, library):
        course = read_course(library / "Lakeside Circuit (course_2800_5000_to_2775_5000)")

        assert course.best.seconds == pytest.approx(47.737)


class TestFindingACourseByName:
    def test_the_coordinates_glued_to_the_folder_name_can_be_ignored(self, library):
        """Nobody is going to type
        `Lakeside Circuit (course_2800_5000_to_2775_5000)`."""
        assert find_course("Lakeside Circuit", library).name.startswith("Lakeside Circuit")

    def test_case_and_partial_names_work(self, library):
        assert find_course("lakeside", library).name.startswith("Lakeside")

    def test_an_ambiguous_name_says_what_it_matched(self, tmp_path):
        root = tmp_path / "laps"
        (root / "Harbour Chase (course_0_0_to_1_1)").mkdir(parents=True)
        (root / "Harbour Circuit (course_0_0_to_0_0)").mkdir(parents=True)

        with pytest.raises(KeyError, match="matches 2 courses"):
            find_course("harbour", root)

    def test_a_name_that_matches_nothing(self, library):
        with pytest.raises(KeyError, match="no course matching"):
            find_course("Nurburgring", library)

    def test_a_library_that_is_not_there(self, tmp_path):
        with pytest.raises(NoLapLibrary, match="no lap library"):
            courses(tmp_path / "nowhere")


class TestWhatItWritesOut:
    def test_the_route_is_the_file_the_map_extractor_reads(self, library, tmp_path):
        """
        The same shape `heat3d_capture route` writes, so a course from the
        library and a lap driven a minute ago are interchangeable downstream.
        """
        course = read_course(library / "Lakeside Circuit (course_2800_5000_to_2775_5000)")

        path = course.save_route(tmp_path / "lakeside", margin=80.0)
        data = json.loads(path.read_text(encoding="utf-8"))

        assert set(data["bounds"]) == {"min", "max"}
        assert len(data["bounds"]["min"]) == 3
        assert data["positions"]
        # The margin widens the plane and leaves height alone: bounding altitude
        # and cutting to it would remove the ground under a hill the road climbs.
        assert data["bounds"]["min"][0] == pytest.approx(2418.7 - 80.0)
        assert data["bounds"]["min"][1] == pytest.approx(420.8)

    def test_the_coordinates_survive_the_csv_at_full_precision(self, library, tmp_path):
        """
        Four significant figures is plenty for a throttle position and ruinous
        for a coordinate: x=2784.97 writes as 2785, the course snaps to a metre
        grid, and the racing line comes out with stairs in it.
        """
        course = read_course(library / "Lakeside Circuit (course_2800_5000_to_2775_5000)")

        path = course.save_samples(tmp_path / "lakeside")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))

        first = next(r for r in rows if r["car_class"] == "A")
        assert float(first["x"]) == pytest.approx(2784.9712, abs=0.002)
        assert float(first["z"]) == pytest.approx(4994.9365, abs=0.002)

    def test_the_csv_holds_every_sample_of_every_lap(self, library, tmp_path):
        course = read_course(library / "Lakeside Circuit (course_2800_5000_to_2775_5000)")

        path = course.save_samples(tmp_path / "lakeside")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))

        assert len(rows) == course.samples == 6

    def test_the_columns_are_named_for_the_viewer_to_map_itself(self, library, tmp_path):
        """
        The viewer picks x, y and z out of a CSV by name and offers the other
        numeric columns as metrics. Naming them `X`, `Speed` and so on would
        work; naming them in lower case with the units spelled out means nobody
        has to ask what `Speed` is in.
        """
        course = read_course(library / "Lakeside Circuit (course_2800_5000_to_2775_5000)")

        path = course.save_samples(tmp_path / "lakeside")
        header = next(csv.reader(path.open(encoding="utf-8")))

        for wanted in ("x", "y", "z", "speed_kmh", "speed_ms", "throttle", "brake", "lat_g"):
            assert wanted in header

    def test_each_lap_is_identifiable_in_the_table(self, library, tmp_path):
        """Without this the whole course is one undifferentiated cloud, and
        comparing a fast lap with a slow one — the point of the exercise — is
        not possible."""
        course = read_course(library / "Lakeside Circuit (course_2800_5000_to_2775_5000)")

        path = course.save_samples(tmp_path / "lakeside")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))

        assert len({r["lap"] for r in rows}) == 2
        assert len({r["car_class"] for r in rows}) == 2
