"""
What a CS2 demo holds, read with demoparser2: the players sampled through the
match, and every event with a place on the map.

A demo is the server's own recording - SourceTV - so it has every player, every
tick (64 a second). Two things about the demos found in the wild decide how it
is read:

- A match can be split across files. The parts' names do not say their order
  (a "-p2" can be the earlier half), but the server's clock runs on across
  them, so parts are put in order by it, and so is everything in them.
- A recording can start before the match: warmup, a restart, a knife round.
  The match starts at the last "match start" announcement in a part, and
  warmup is left out everywhere.

Nothing here interprets the match; that is `match.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TICKRATE = 64

#: What is read for every sampled player.
PLAYER_FIELDS = [
    "X",
    "Y",
    "Z",
    "yaw",
    "pitch",
    "health",
    "armor_value",
    "team_num",
    "team_clan_name",
    "active_weapon_name",
    "is_alive",
    "last_place_name",
    "balance",
    "current_equip_value",
]

#: The game's own state, read alongside: the round count, who won it and how,
#: the score, the clock.
RULES_FIELDS = [
    "game_time",
    "total_rounds_played",
    "round_win_status",
    "round_win_reason",
    "team_rounds_total",
    "is_warmup_period",
    "is_freeze_period",
]

#: For each player in an event: where they were and on which side.
EVENT_PLAYER = ["X", "Y", "Z", "team_num", "last_place_name"]

#: Events read, where the demo has them.
EVENTS = [
    "player_death",
    "player_hurt",
    "player_blind",
    "weapon_fire",
    "smokegrenade_detonate",
    "flashbang_detonate",
    "hegrenade_detonate",
    "inferno_startburn",
    "decoy_detonate",
    "bomb_planted",
    "bomb_defused",
    "bomb_exploded",
    "round_freeze_end",
    "round_prestart",
    "round_announce_match_start",
    "cs_win_panel_match",
]

#: Far past the end of any match: the sampled ticks are a stride up to here,
#: and the parser stops at the demo's end.
LAST_TICK = 4_000_000


@dataclass
class RawPart:
    """One demo file as the parser gives it, untouched."""

    path: str
    map: str
    #: One row per player per sampled tick: PLAYER_FIELDS, RULES_FIELDS,
    #: tick, steamid, name.
    samples: pd.DataFrame
    #: Event name -> its rows, players' fields prefixed `user_`, `attacker_`,
    #: `assister_` as the parser names them.
    events: dict[str, pd.DataFrame] = field(default_factory=dict)


def read_part(path: str | Path, every: int = 8) -> RawPart:
    """Read one demo file: players every `every` ticks, and every event."""
    from demoparser2 import DemoParser

    parser = DemoParser(str(path))
    header = parser.parse_header()
    available = set(parser.list_game_events())
    samples = parser.parse_ticks(PLAYER_FIELDS + RULES_FIELDS, ticks=list(range(0, LAST_TICK, every)))
    events: dict[str, pd.DataFrame] = {}
    for name in EVENTS:
        if name in available:
            events[name] = parser.parse_event(name, player=EVENT_PLAYER, other=["total_rounds_played", "is_warmup_period"])
    return RawPart(str(path), header.get("map_name", ""), samples, events)
