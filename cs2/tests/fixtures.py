"""
A demo as the parser gives it, built by hand: two rounds, two players a side,
small enough to know every answer. The columns and their meaning are
demoparser2's, as `demo.read_part` asks for them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heat3d_cs2.demo import RawPart

#: steamid -> (name, team, team_num in the first half)
PLAYERS = {
    "1": ("ace", "Alpha", 2),
    "2": ("ash", "Alpha", 2),
    "3": ("bob", "Bravo", 3),
    "4": ("ben", "Bravo", 3),
}
POSITIONS = {"1": (100.0, 0.0), "2": (120.0, 10.0), "3": (-500.0, -300.0), "4": (-520.0, -320.0)}


def samples(
    rounds: list[dict],
    every: int = 8,
    clock0: float = 1000.0,
    last_tick: int | None = None,
    clans: bool = True,
) -> pd.DataFrame:
    """
    Every player every `every` ticks. `rounds`: dicts with `freeze` (tick the
    freeze time ends), `end` (tick decided), `winner` (2 or 3), `reason`, and
    `deaths` {steamid: tick}. Before a round's freeze ends the players are
    frozen; between rounds the next round's count is shown, as the game does.
    """
    last_tick = last_tick or rounds[-1]["end"] + 640
    rows = []
    score = {2: 0, 3: 0}
    for tick in range(0, last_tick + 1, every):
        played = sum(1 for r in rounds if tick >= r["end"])
        current = rounds[min(played, len(rounds) - 1)]
        decided = [r for r in rounds if r["end"] <= tick]
        score = {2: sum(1 for r in decided if r["winner"] == 2), 3: sum(1 for r in decided if r["winner"] == 3)}
        status = decided[-1]["winner"] if decided and tick < decided[-1]["end"] + 448 else 0
        reason = decided[-1]["reason"] if status else 0
        freeze = tick < current["freeze"] and played < len(rounds)
        for sid, (name, clan, num) in PLAYERS.items():
            died = current.get("deaths", {}).get(sid)
            alive = not (died is not None and tick >= died and tick < current["end"] + 448)
            x, y = POSITIONS[sid]
            rows.append(
                {
                    "tick": tick,
                    "steamid": sid,
                    "name": name,
                    "X": x + (tick % 64),
                    "Y": y,
                    "Z": -160.0,
                    "yaw": 90.0,
                    "pitch": 10.0,
                    "health": 100 if alive else 0,
                    "armor_value": 100,
                    "team_num": num,
                    "team_clan_name": clan if clans else None,
                    "active_weapon_name": "Karambit" if sid == "1" else "AK-47",
                    "is_alive": alive,
                    "last_place_name": "BombsiteA" if num == 2 else "CTSpawn",
                    "balance": 800,
                    "current_equip_value": 4800 if played else 800,
                    "game_time": clock0 + tick / 64,
                    "total_rounds_played": played,
                    "round_win_status": status,
                    "round_win_reason": reason,
                    "team_rounds_total": score[num],
                    "is_warmup_period": False,
                    "is_freeze_period": freeze,
                }
            )
    return pd.DataFrame(rows)


def player_columns(prefix: str, sid: str | None, tick: int) -> dict:
    if sid is None:
        return {f"{prefix}_{k}": None for k in ("name", "steamid", "X", "Y", "Z", "team_num", "last_place_name")}
    name, _, num = PLAYERS[sid]
    x, y = POSITIONS[sid]
    return {
        f"{prefix}_name": name,
        f"{prefix}_steamid": sid,
        f"{prefix}_X": x,
        f"{prefix}_Y": y,
        f"{prefix}_Z": -160.0,
        f"{prefix}_team_num": num,
        f"{prefix}_last_place_name": "BombsiteA" if num == 2 else "CTSpawn",
    }


def death(tick: int, attacker: str | None, victim: str, *, headshot=False, weapon="ak47", assister: str | None = None) -> dict:
    return {
        "tick": tick,
        **player_columns("attacker", attacker, tick),
        **player_columns("user", victim, tick),
        **player_columns("assister", assister, tick),
        "weapon": weapon,
        "headshot": headshot,
        "assistedflash": False,
        "penetrated": 0,
        "thrusmoke": False,
        "noscope": False,
        "attackerblind": False,
        "total_rounds_played": 0,
        "is_warmup_period": False,
    }


def hurt(tick: int, attacker: str, victim: str, dmg: int, health_after: int, weapon="ak47") -> dict:
    return {
        "tick": tick,
        **player_columns("attacker", attacker, tick),
        **player_columns("user", victim, tick),
        "weapon": weapon,
        "dmg_health": dmg,
        "health": health_after,
        "is_warmup_period": False,
    }


def two_rounds(clans: bool = True, clock0: float = 1000.0) -> RawPart:
    """
    Round 1 (T win by elimination): ace kills bob (the opening), ben kills ace
    1.9 s later (a trade: bob's death is traded), ash kills ben - alone
    against one, a clutch won. Round 2 (T win, the bomb): ash plants on A,
    nobody dies.
    """
    rounds = [
        {"freeze": 640, "end": 1664, "winner": 2, "reason": 9, "deaths": {"3": 1280, "1": 1400, "4": 1600}},
        {"freeze": 2560, "end": 4800, "winner": 2, "reason": 1, "deaths": {}},
    ]
    s = samples(rounds, clock0=clock0, clans=clans)
    events = {
        "round_freeze_end": pd.DataFrame(
            {"tick": [640, 2560], "total_rounds_played": [0, 1], "is_warmup_period": [False, False]}
        ),
        "round_prestart": pd.DataFrame({"tick": [0, 2112]}),
        "player_death": pd.DataFrame(
            [
                death(1280, "1", "3", headshot=True),
                death(1400, "4", "1"),
                death(1600, "2", "4"),
            ]
        ),
        "player_hurt": pd.DataFrame(
            [
                hurt(1270, "1", "3", 60, 40),
                hurt(1280, "1", "3", 137, 0),
                hurt(1300, "4", "1", 100, 0),
                hurt(1600, "2", "4", 120, 0),
            ]
        ),
        "bomb_planted": pd.DataFrame(
            [{"tick": 4000, **player_columns("user", "2", 4000), "site": 325, "is_warmup_period": False}]
        ),
        "weapon_fire": pd.DataFrame(
            [
                {"tick": 1270, **player_columns("user", "1", 1270), "weapon": "weapon_ak47", "is_warmup_period": False},
                {"tick": 1200, **player_columns("user", "2", 1200), "weapon": "weapon_smokegrenade", "is_warmup_period": False},
                {"tick": 1210, **player_columns("user", "2", 1210), "weapon": "weapon_knife_karambit", "is_warmup_period": False},
            ]
        ),
        "smokegrenade_detonate": pd.DataFrame(
            [{"tick": 1290, **player_columns("user", "2", 1290), "x": 50.0, "y": 60.0, "z": -150.0, "is_warmup_period": False}]
        ),
    }
    for df in events.values():
        df.columns = [str(c) for c in df.columns]
    return RawPart("synthetic.dem", "de_test", s, events)


def frame_series(n: int, value) -> np.ndarray:
    return np.full(n, value)
