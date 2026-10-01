"""
The match from its demo parts: rounds and how they were won, kills with
openings and trades, damage as health lost, clutches, the scoreboard, and the
table the heatmap reads - on a demo built by hand, so every answer is known.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from heat3d_cs2.match import build_match, chains
from heat3d_cs2.output import HINTS, write
from heat3d_cs2.stats import player_stats, round_summaries, scoreboard_text, summary

from .fixtures import two_rounds


@pytest.fixture(scope="module")
def match():
    return build_match([two_rounds()], "alpha-vs-bravo-m1-test")


def by_name(stats):
    return {p["name"]: p for p in stats}


class TestRounds:
    def test_each_round_its_winner_reason_and_score(self, match):
        assert [r.number for r in match.rounds] == [1, 2]
        assert [(r.winner, r.reason) for r in match.rounds] == [("T", "elimination"), ("T", "bomb")]
        assert match.rounds[0].score == {"Alpha": 1, "Bravo": 0}
        assert match.rounds[1].score == {"Alpha": 2, "Bravo": 0}
        assert match.rounds[0].sides == {"Alpha": "T", "Bravo": "CT"}
        assert match.teams == ["Alpha", "Bravo"]

    def test_the_clock_starts_at_the_match(self, match):
        r = match.rounds[0]
        # Freeze time ends at tick 640: 10 s in, decided at tick 1664.
        assert r.start == pytest.approx(10.0)
        assert r.end == pytest.approx(1664 / 64)

    def test_economy_and_the_plant(self, match):
        assert match.rounds[0].buy == {"T": "pistol", "CT": "pistol"}
        assert match.rounds[1].buy == {"T": "full", "CT": "full"}
        assert match.rounds[1].plant["site"] == "A" and match.rounds[1].plant["player"] == "ash"

    def test_teams_without_clan_names_are_named_by_their_first_side(self):
        m = build_match([two_rounds(clans=False)], "x")
        assert sorted(m.teams) == ["Started CT", "Started T"]
        assert m.rounds[0].score == {"Started T": 1, "Started CT": 0}


class TestKills:
    def test_opening_trades_and_flags(self, match):
        k = match.kills
        assert list(k["victim"]) == ["bob", "ace", "ben"]
        assert list(k["opening"]) == [True, False, False]
        # bob's death is avenged by ben 1.9 s later, ace's by ash 3.1 s later.
        assert list(k["traded"]) == [True, True, False]
        assert list(k["trade"]) == [False, True, True]
        assert bool(k["headshot"].iloc[0]) and k["weapon"].iloc[0] == "AK-47"

    def test_damage_is_health_lost(self, match):
        # 60 then a 137 on 40 health; 100 on ace; 120 on ben.
        assert list(match.damage["damage"]) == [60, 40, 100, 100]


class TestScoreboard:
    def test_the_numbers(self, match):
        p = by_name(player_stats(match))
        assert (p["ace"]["kills"], p["ace"]["deaths"], p["ace"]["headshotPct"]) == (1, 1, 100.0)
        assert p["ace"]["adr"] == pytest.approx(50.0)
        assert p["ash"]["adr"] == pytest.approx(50.0)
        assert p["bob"]["adr"] == 0
        # Everyone killed, survived or was traded in both rounds.
        assert {n: s["kast"] for n, s in p.items()} == {"ace": 100.0, "ash": 100.0, "bob": 100.0, "ben": 100.0}
        assert (p["ace"]["openingKills"], p["bob"]["openingDeaths"]) == (1, 1)
        assert (p["ben"]["tradeKills"], p["ash"]["tradeKills"]) == (1, 1)
        assert p["ash"]["plants"] == 1

    def test_clutches(self, match):
        r1 = round_summaries(match)[0]
        clutches = {c["player"]: (c["against"], c["won"]) for c in r1["clutches"]}
        # ben was alone against two after bob fell; ash alone against ben after ace.
        assert clutches == {"ben": (2, False), "ash": (1, True)}
        p = by_name(player_stats(match))
        assert (p["ash"]["clutches"], p["ash"]["clutchesWon"]) == (1, 1)

    def test_the_text_scoreboard(self, match):
        text = scoreboard_text(summary(match))
        assert "Alpha 2" in text and "rounds  1:TK 2:TB" in text


class TestTheTable:
    def test_events_where_they_happened(self, match):
        rows = match.rows
        counts = rows["event"].value_counts().to_dict()
        assert counts["kill"] == 3 and counts["death"] == 3 and counts["hurt"] == 4
        assert counts["shot"] == 1 and counts["throw"] == 1  # the knife is not a shot
        assert counts["smoke"] == 1 and counts["plant"] == 1 and counts["explode"] == 1
        smoke = rows[rows["event"] == "smoke"].iloc[0]
        assert (smoke["x"], smoke["y"], smoke["player"]) == (50.0, 60.0, "ash")
        kill = rows[rows["event"] == "kill"].iloc[0]
        assert (kill["player"], kill["other"], kill["opening"], kill["round"]) == ("ace", "bob", 1, 1)

    def test_positions_are_live_and_labelled(self, match):
        pos = match.rows[match.rows["event"] == "position"]
        assert len(pos) and (pos["roundTime"] >= 0).all()
        assert set(pos["side"]) == {"T", "CT"}
        assert set(pos["team"]) == {"Alpha", "Bravo"}
        assert set(pos.loc[pos["player"] == "ace", "weapon"]) == {"Knife"}
        # Looking along the yaw, pitched down: unit vectors.
        n = np.sqrt(pos["dirX"] ** 2 + pos["dirY"] ** 2 + pos["dirZ"] ** 2)
        assert np.allclose(n, 1.0, atol=1e-6)
        assert (pos["dirZ"] < 0).all()
        won = pos[pos["round"] == 1]
        assert set(won.loc[won["side"] == "T", "won"]) == {1.0}

    def test_written_with_its_reading_instructions(self, match, tmp_path):
        data, doc = write(match, tmp_path)
        table = pq.read_table(data)
        hints = json.loads(table.schema.metadata[b"heat3d"])
        assert hints == HINTS
        assert hints["axes"]["flipY"] is True
        assert table.num_rows == len(match.rows)
        summary_doc = json.loads(doc.read_text(encoding="utf-8"))
        assert summary_doc["format"] == "heat3d-cs2-match"
        assert summary_doc["score"] == {"Alpha": 2, "Bravo": 0}
        assert summary_doc["missing"] == []


class TestParts:
    def test_parts_in_order_whatever_their_names(self):
        whole = two_rounds()
        first = two_rounds()
        # A second file carrying on from the first: same rounds two and on,
        # its clock after the first's.
        later = two_rounds(clock0=1000.0 + 6000 / 64)
        later.samples["total_rounds_played"] += 2
        later.events["round_freeze_end"]["total_rounds_played"] += 2
        found = chains([later, first])
        assert len(found) == 1 and found[0][0] is first
        assert len(chains([whole, two_rounds()])) == 2

    def test_a_restarted_server_clock_still_follows_on(self):
        first = two_rounds(clock0=3000.0)
        later = two_rounds(clock0=100.0)
        later.samples["total_rounds_played"] += 2
        later.events["round_freeze_end"]["total_rounds_played"] += 2
        (chain,) = chains([later, first])
        m = build_match(chain, "x")
        assert [r.number for r in m.rounds] == [1, 2, 3, 4]
        starts = [r.start for r in m.rounds]
        assert starts == sorted(starts)


class TestRestoredRounds:
    def test_one_row_per_player_per_tick_the_one_that_leads_on(self):
        from heat3d_cs2.match import _one_per_tick

        raw = two_rounds()
        s = raw.samples
        # A restore: for two sampled ticks ace also shows up where the voided
        # round left him, far from where he goes on from.
        stale = s[(s["steamid"] == "1") & s["tick"].isin([800, 808])].copy()
        stale["X"] = 9000.0
        doubled = pd.concat([s, stale], ignore_index=True)
        once = _one_per_tick(doubled)
        assert not once.duplicated(["steamid", "tick"]).any()
        assert len(once) == len(s)
        assert (once.loc[once["steamid"] == "1", "X"] < 9000).all()
