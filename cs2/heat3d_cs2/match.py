"""
A match from its demo parts: the rounds and how each was won, every player's
position through them, every event where it happened, and what each player
did with it.

Everything is put on one clock, the match's: seconds from the start of the
live match, read from the server's own clock so a match split across demo
files runs on across the join. A round runs from the end of its freeze time
(players can move) to the next round's start; the part after it was decided
is its "after" phase, where the winners hunt and the losers save.

The conventions, where the game itself does not define them:

- A death is **traded** when its killer is killed by one of the victim's
  teammates within 5 s; that kill is a **trade kill**.
- The round's first kill is its **opening** duel.
- **KAST**: the share of rounds in which a player killed, assisted (a flash
  assist counts), survived, or was traded.
- **Damage** is what the victim actually lost: a 137 damage headshot on a
  player with 40 health is 40, as the scoreboard counts it.
- A **clutch** is being the last player alive on a side with enemies still
  standing; it is won when the side wins the round.
- **Buy**, per side, from the average equipment value at the end of freeze
  time: the first round of each half is a pistol round, then under $1,500 is
  an eco, under $4,000 a force buy, and anything more a full buy.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .demo import TICKRATE, RawPart

TRADE_SECONDS = 5.0

#: The game's round end reasons (CSRoundEndReason) that occur in a defusal match.
REASONS = {
    1: "bomb",
    7: "defuse",
    8: "elimination",
    9: "elimination",
    10: "draw",
    12: "time",
    17: "surrender",
    18: "surrender",
}

#: A death or damage event's weapon id -> the name the game shows, as the
#: players' own `active_weapon_name` has it.
WEAPONS = {
    "ak47": "AK-47",
    "aug": "AUG",
    "awp": "AWP",
    "bizon": "PP-Bizon",
    "c4": "C4 Explosive",
    "planted_c4": "C4 Explosive",
    "cz75a": "CZ75-Auto",
    "deagle": "Desert Eagle",
    "decoy": "Decoy Grenade",
    "elite": "Dual Berettas",
    "famas": "FAMAS",
    "fiveseven": "Five-SeveN",
    "flashbang": "Flashbang",
    "g3sg1": "G3SG1",
    "galilar": "Galil AR",
    "glock": "Glock-18",
    "hegrenade": "High Explosive Grenade",
    "hkp2000": "P2000",
    "incgrenade": "Incendiary Grenade",
    "inferno": "Molotov",
    "m249": "M249",
    "m4a1": "M4A4",
    "m4a1_silencer": "M4A1-S",
    "mac10": "MAC-10",
    "mag7": "MAG-7",
    "molotov": "Molotov",
    "mp5sd": "MP5-SD",
    "mp7": "MP7",
    "mp9": "MP9",
    "negev": "Negev",
    "nova": "Nova",
    "p250": "P250",
    "p90": "P90",
    "revolver": "R8 Revolver",
    "sawedoff": "Sawed-Off",
    "scar20": "SCAR-20",
    "sg556": "SG 553",
    "smokegrenade": "Smoke Grenade",
    "ssg08": "SSG 08",
    "taser": "Zeus x27",
    "tec9": "Tec-9",
    "ump45": "UMP-45",
    "usp_silencer": "USP-S",
    "world": "World",
    "xm1014": "XM1014",
}

SIDES = ("T", "CT")

#: Knife skins, as the players' active weapon names them.
KNIVES = ("Knife", "Bayonet", "Karambit", "Daggers")

GRENADES = {"smokegrenade", "flashbang", "hegrenade", "molotov", "incgrenade", "decoy"}
UTILITY_DAMAGE = {"hegrenade", "inferno", "molotov", "incgrenade"}

#: Detonation events -> the row's event name.
DETONATIONS = {
    "smokegrenade_detonate": "smoke",
    "flashbang_detonate": "flash",
    "hegrenade_detonate": "he",
    "inferno_startburn": "molotov",
    "decoy_detonate": "decoy",
}


def weapon_name(weapon: object) -> str:
    """A weapon id from an event (`ak47`, `weapon_ak47`, `knife_karambit`) as the game names it."""
    w = str(weapon or "").removeprefix("weapon_")
    if not w:
        return ""
    if "knife" in w or "bayonet" in w:
        return "Knife"
    return WEAPONS.get(w, w)


def held(name: object) -> str:
    """A player's active weapon by its name, every knife skin as `Knife`."""
    w = str(name or "")
    return "Knife" if any(k in w for k in KNIVES) else w


def side_of(team_num: object) -> str:
    try:
        n = int(team_num)
    except (TypeError, ValueError):
        return ""
    return "T" if n == 2 else "CT" if n == 3 else ""


def buy_type(average_equipment: float, pistol: bool) -> str:
    if pistol:
        return "pistol"
    if average_equipment < 1500:
        return "eco"
    if average_equipment < 4000:
        return "force"
    return "full"


@dataclass
class Round:
    #: The game's round number, 1-based.
    number: int
    #: Match seconds: freeze time over, the round decided, the next round started.
    start: float
    end: float
    until: float
    winner: str
    reason: str
    #: Team name -> rounds won after this one, and the side it played.
    score: dict[str, int]
    sides: dict[str, str]
    #: Side -> team equipment value at the end of freeze time, and the buy.
    equipment: dict[str, int] = field(default_factory=dict)
    buy: dict[str, str] = field(default_factory=dict)
    #: The bomb plant, if any: site, match seconds, planter, position.
    plant: dict | None = None


@dataclass
class Match:
    name: str
    map: str
    parts: list[str]
    rounds: list[Round]
    #: steamid -> name and team.
    players: dict[str, dict]
    #: Team names, in the order they first played T then CT... as first seen.
    teams: list[str]
    #: The long table: every position sample and every event, one row each.
    rows: pd.DataFrame
    #: One row per death, with who, how, opening, traded.
    kills: pd.DataFrame
    #: Effective damage between enemies, one row per hit.
    damage: pd.DataFrame
    #: Enemies blinded by a flash, one row per player blinded.
    blinds: pd.DataFrame
    #: Skipped rounds (cut off by the end of a demo), by the game's number.
    incomplete: list[int] = field(default_factory=list)


# ---------------------------------------------------------------- the clock


def _one_per_tick(samples: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player per tick.

    Where a round was restored from a backup, the demo carries the players'
    old bodies alongside the new ones for some seconds: two rows for one
    player at one tick, one where the voided round left them, one at the
    restored round's spawn. The one that leads on into the player's next
    single row is theirs - so each is chosen walking back from there.
    """
    dup = samples.duplicated(["steamid", "tick"], keep=False)
    if not dup.any():
        return samples
    s = samples.reset_index(drop=True)
    dup = dup.reset_index(drop=True)
    xyz = s[["X", "Y", "Z"]].to_numpy(dtype=float)
    drop: list[int] = []
    for sid, rows in s[dup].groupby("steamid"):
        ticks = sorted(rows["tick"].unique(), reverse=True)
        after = s.index[(s["steamid"] == sid) & ~dup & (s["tick"] > ticks[0])]
        ref = xyz[after.min()] if len(after) else None
        for tick in ticks:
            candidates = rows.index[rows["tick"] == tick]
            if ref is None or not np.isfinite(ref).all():
                best = candidates[-1]
            else:
                best = candidates[int(np.nanargmin(((xyz[candidates] - ref) ** 2).sum(axis=1)))]
            drop.extend(i for i in candidates if i != best)
            ref = xyz[best]
    return s.drop(index=drop)


class _Part:
    """One demo file on the match clock."""

    def __init__(self, raw: RawPart):
        raw.samples = _one_per_tick(raw.samples)
        self.raw = raw
        s = raw.samples
        clock = s[["tick", "game_time"]].dropna().drop_duplicates("tick").sort_values("tick")
        self.ticks = clock["tick"].to_numpy(dtype=float)
        self.clock = clock["game_time"].to_numpy(dtype=float)
        start = raw.events.get("round_announce_match_start")
        self.live_from = int(start["tick"].max()) if start is not None and len(start) else 0
        self.first_clock = float(self.at([self.live_from])[0]) if len(self.ticks) else math.inf
        self.last_clock = float(self.at([self.ticks[-1]])[0]) if len(self.ticks) else -math.inf
        # The game's index of the first and last round that starts here (its
        # freeze time ending), as the match's rounds count them.
        freezes = raw.events.get("round_freeze_end")
        index = (
            freezes.loc[(freezes["tick"] >= self.live_from) & ~_flag(freezes, "is_warmup_period"), "total_rounds_played"].dropna()
            if freezes is not None and "total_rounds_played" in freezes
            else pd.Series(dtype=float)
        )
        self.has_rounds = bool(len(index))
        self.rounds = (int(index.min()), int(index.max())) if len(index) else (0, 0)
        #: Server clock at the match's zero: set when the parts are put in order.
        self.offset = self.first_clock

    def at(self, ticks):
        """
        Seconds on the server's clock at `ticks`: from where the file starts,
        counted in ticks. The ticks never stop; the server's own clock can - it
        stands still through a pause while the ticks run on - and read through
        it two moments would share a time.
        """
        t = np.atleast_1d(np.asarray(ticks, dtype=float))
        if not len(self.ticks):
            return t / TICKRATE
        return self.clock[0] + (t - self.ticks[0]) / TICKRATE


def _flag(df: pd.DataFrame, column: str) -> pd.Series:
    """A boolean column, False where the demo does not have it."""
    if column not in df:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def _rules(samples: pd.DataFrame) -> pd.DataFrame:
    """The game state once per sampled tick, from the players' rows."""
    s = samples[samples["team_num"].isin([2, 3])]
    return s.drop_duplicates("tick").sort_values("tick")[
        ["tick", "total_rounds_played", "round_win_status", "round_win_reason", "is_warmup_period"]
    ]


@dataclass
class _Window:
    """Where a round lies in its part, and its score by the game's team numbers."""

    part: _Part
    freeze_tick: int
    end_tick: int
    #: team number (2 T, 3 CT) -> rounds won after this one, NaN where the demo had none.
    score: dict[int, float]


def _rounds_in(part: _Part, origin: float) -> tuple[dict[int, tuple[Round, _Window]], list[int]]:
    """Every round decided in one part, by the game's 0-based round index. Teams are named later."""
    raw = part.raw
    freezes = raw.events.get("round_freeze_end")
    if freezes is None or not len(freezes):
        return {}, []
    freezes = freezes[(freezes["tick"] >= part.live_from) & ~_flag(freezes, "is_warmup_period")]
    freezes = freezes.sort_values("tick")
    rules = _rules(raw.samples)
    prestarts = raw.events.get("round_prestart")
    prestart_ticks = np.sort(prestarts["tick"].to_numpy()) if prestarts is not None else np.array([])
    teams = raw.samples[raw.samples["team_num"].isin([2, 3])]
    last_tick = int(raw.samples["tick"].max())
    plants = raw.events.get("bomb_planted")
    out: dict[int, tuple[Round, _Window]] = {}
    cut: list[int] = []
    freeze_ticks = freezes["tick"].to_numpy()
    for k, (tick, index) in enumerate(zip(freeze_ticks, freezes["total_rounds_played"].to_numpy())):
        index = int(index)
        limit = freeze_ticks[k + 1] if k + 1 < len(freeze_ticks) else last_tick + 1
        done = rules[
            (rules["tick"] > tick)
            & (rules["tick"] < limit)
            & (rules["total_rounds_played"] == index + 1)
            & rules["round_win_status"].isin([2, 3])
        ]
        if not len(done):
            cut.append(index + 1)
            continue
        end_tick = int(done["tick"].iloc[0])
        status = int(done["round_win_status"].iloc[0])
        reason = REASONS.get(int(done["round_win_reason"].iloc[0]), "other")
        after = prestart_ticks[prestart_ticks > end_tick]
        until_tick = int(after[0]) if len(after) and after[0] < limit else min(limit, last_tick)
        at_end = teams[teams["tick"] == end_tick].drop_duplicates("team_num")
        score_by_num = {int(r.team_num): float(r.team_rounds_total) for r in at_end.itertuples()}
        # Equipment at the end of freeze time: the first sample from then, alive players.
        frozen = teams[(teams["tick"] >= tick) & teams["is_alive"].astype(bool)]
        first = frozen["tick"].min() if len(frozen) else None
        equipment: dict[str, int] = {}
        counts: dict[str, int] = {}
        if first is not None:
            at_start = frozen[frozen["tick"] == first]
            for team_num, group in at_start.groupby("team_num"):
                equipment[side_of(team_num)] = int(group["current_equip_value"].fillna(0).sum())
                counts[side_of(team_num)] = len(group)
        number = index + 1
        pistol = number in (1, 13)
        buy = {s: buy_type(v / max(1, counts.get(s, 1)), pistol) for s, v in equipment.items()}
        plant = None
        if plants is not None and len(plants):
            p = plants[(plants["tick"] > tick) & (plants["tick"] <= end_tick)]
            if len(p):
                row = p.iloc[0]
                place = str(row.get("user_last_place_name") or "")
                site = "A" if "BombsiteA" in place else "B" if "BombsiteB" in place else str(row.get("site", ""))
                plant = {
                    "site": site,
                    "time": float(part.at([row["tick"]])[0] - origin),
                    "player": str(row.get("user_name") or ""),
                    "x": float(row["user_X"]),
                    "y": float(row["user_Y"]),
                    "z": float(row["user_Z"]),
                }
        start, end, until = part.at([tick, end_tick, until_tick]) - origin
        out[index] = (
            Round(
                number=number,
                start=float(start),
                end=float(end),
                until=float(max(until, end)),
                winner="T" if status == 2 else "CT",
                reason=reason,
                score={},
                sides={},
                equipment=equipment,
                buy=buy,
                plant=plant,
            ),
            _Window(part, int(tick), end_tick, score_by_num),
        )
    return out, cut


def _players(parts: list[_Part]) -> dict[str, dict]:
    """
    steamid -> name and team, for everyone who played: alive at least once
    while the match was live. A coach or a caster on a team's side never is.

    A player's team is the clan name the game gives their side, the one seen
    most - the demo clears it at its very end. A match played without clan
    names has its teams named by the side they started on.
    """
    names: dict[str, str] = {}
    clans: dict[str, Counter] = {}
    first: dict[str, tuple[float, int]] = {}
    for part in parts:
        s = part.raw.samples
        s = s[
            s["team_num"].isin([2, 3])
            & _flag(s, "is_alive")
            & ~_flag(s, "is_freeze_period")
            & ~_flag(s, "is_warmup_period")
            & (s["tick"] >= part.live_from)
        ]
        for sid, group in s.groupby("steamid"):
            sid = str(sid)
            names[sid] = str(group["name"].iloc[-1])
            clans.setdefault(sid, Counter()).update(str(c) for c in group["team_clan_name"].dropna() if str(c))
            at = (part.first_clock, int(group["tick"].iloc[0]))
            if sid not in first or at < first[sid][0:2]:
                first[sid] = (*at, int(group["team_num"].iloc[0]))
    out = {}
    for sid, name in names.items():
        clan = clans[sid].most_common(1)
        team = clan[0][0] if clan else f"Started {side_of(first[sid][2])}"
        out[sid] = {"name": name, "team": team}
    return out


def _name_teams(rounds: list[Round], windows: list[_Window], players: dict[str, dict]) -> list[str]:
    """Each round's sides and score by team name, from who was on which side; the teams in order of appearance."""
    teams: list[str] = []
    previous: dict[str, int] = {}
    for r, w in zip(rounds, windows):
        s = w.part.raw.samples
        s = s[(s["tick"] >= w.freeze_tick) & (s["tick"] <= w.end_tick) & s["team_num"].isin([2, 3])]
        team_of: dict[int, str] = {}
        for num, group in s.groupby("team_num"):
            labels = Counter(players[str(sid)]["team"] for sid in group["steamid"].astype(str).unique() if str(sid) in players)
            if labels:
                team_of[int(num)] = labels.most_common(1)[0][0]
        r.sides = {team: side_of(num) for num, team in team_of.items()}
        for num in (2, 3):
            team = team_of.get(num)
            if team is None:
                continue
            if team not in teams:
                teams.append(team)
            won = w.score.get(num, math.nan)
            # The game's own count where the demo has it; else counted on.
            r.score[team] = int(won) if math.isfinite(won) else previous.get(team, 0) + int(r.winner == side_of(num))
        previous = dict(r.score)
    return teams


# ---------------------------------------------------------------- the rows


ROW_COLUMNS = [
    "match",
    "map",
    "round",
    "roundId",
    "phase",
    "time",
    "roundTime",
    "event",
    "player",
    "steamid",
    "side",
    "team",
    "x",
    "y",
    "z",
    "dirX",
    "dirY",
    "dirZ",
    "health",
    "armor",
    "money",
    "equipment",
    "weapon",
    "place",
    "other",
    "damage",
    "duration",
    "headshot",
    "opening",
    "traded",
    "won",
    "buy",
]


def _direction(yaw_deg, pitch_deg):
    """Where a player looks, as a unit vector in the game's frame (pitch down is positive)."""
    yaw = np.radians(np.asarray(yaw_deg, dtype=float))
    pitch = np.radians(np.asarray(pitch_deg, dtype=float))
    cp = np.cos(pitch)
    return cp * np.cos(yaw), cp * np.sin(yaw), -np.sin(pitch)


def _assign(times: np.ndarray, rounds: list[Round]) -> tuple[np.ndarray, np.ndarray]:
    """For each time: the index into `rounds` it falls in (-1 if none), and whether after the round was decided."""
    which = np.full(len(times), -1, dtype=int)
    after = np.zeros(len(times), dtype=bool)
    for k, r in enumerate(rounds):
        inside = (times >= r.start) & (times < r.until)
        which[inside] = k
        after[inside] = times[inside] > r.end
    return which, after


def _frame(columns: dict) -> pd.DataFrame:
    df = pd.DataFrame(columns)
    for c in ROW_COLUMNS:
        if c not in df:
            df[c] = np.nan if c in NUMERIC else ""
    return df[ROW_COLUMNS]


NUMERIC = {
    "round",
    "time",
    "roundTime",
    "x",
    "y",
    "z",
    "dirX",
    "dirY",
    "dirZ",
    "health",
    "armor",
    "money",
    "equipment",
    "damage",
    "duration",
    "headshot",
    "opening",
    "traded",
    "won",
}


def _event_rows(part: _Part, origin: float, prefix: str, df: pd.DataFrame, event: str, **extra) -> dict:
    """The common columns of an event's rows, for the player named by `prefix` (`user`, `attacker`)."""
    return {
        "time": part.at(df["tick"].to_numpy()) - origin,
        "event": event,
        "player": df[f"{prefix}_name"].fillna("").astype(str).to_numpy(),
        "steamid": df[f"{prefix}_steamid"].fillna("").astype(str).to_numpy(),
        "side": [side_of(v) for v in df[f"{prefix}_team_num"]],
        "x": df[f"{prefix}_X"].to_numpy(dtype=float),
        "y": df[f"{prefix}_Y"].to_numpy(dtype=float),
        "z": df[f"{prefix}_Z"].to_numpy(dtype=float),
        "place": df[f"{prefix}_last_place_name"].fillna("").astype(str).to_numpy(),
        **extra,
    }


def _in_order(parts: list[_Part]) -> list[_Part]:
    """Parts by the rounds they hold, then by the server's clock."""
    return sorted(parts, key=lambda p: (p.rounds[0], p.first_clock))


def chains(raws: list[RawPart]) -> list[list[RawPart]]:
    """
    Demo files named as one match, sorted into the matches they really are,
    the one with most rounds first.

    Parts of one match follow each other with the round count carrying on,
    and usually the server's clock too - but not always: a server restarted
    to restore a round starts its clock again, so the count decides and the
    clock only breaks a tie. A collection can also hold two different matches
    under one name - the same teams on the same map at two events - whose
    counts overlap, and those are kept apart. A file with no round in it
    (a pause, warmup) belongs to none.
    """
    out: list[list[_Part]] = []
    for part in _in_order([_Part(r) for r in raws]):
        if not part.has_rounds:
            continue
        for chain in out:
            last = chain[-1]
            clock_runs_on = part.first_clock >= last.last_clock - 5
            count_runs_on = last.rounds[1] <= part.rounds[0] <= last.rounds[1] + 1
            if part.rounds[0] >= last.rounds[1] and (clock_runs_on or count_runs_on):
                chain.append(part)
                break
        else:
            out.append([part])
    out.sort(key=lambda chain: -sum(p.rounds[1] - p.rounds[0] + 1 for p in chain))
    return [[p.raw for p in chain] for chain in out]


def _offsets(parts: list[_Part]) -> None:
    """
    Each part's place on the match clock: seconds from the start of the live
    match. Where the server's clock runs on between parts it is used as it
    is; where it was restarted, the part follows on a second after the last.
    """
    origin = parts[0].first_clock
    parts[0].offset = origin
    for prev, part in zip(parts, parts[1:]):
        if part.first_clock >= prev.last_clock - 5:
            part.offset = origin
        else:
            ended = prev.last_clock - prev.offset
            part.offset = part.first_clock - (ended + 1.0)


def build_match(raws: list[RawPart], name: str) -> Match:
    """One match from its demo parts, in whatever order they come."""
    parts = _in_order([_Part(r) for r in raws])
    if not parts:
        raise ValueError("no demo parts")
    _offsets(parts)
    by_index: dict[int, tuple[Round, _Window]] = {}
    incomplete: set[int] = set()
    for part in parts:
        found, cut = _rounds_in(part, part.offset)
        incomplete.update(cut)
        # A later part's account of a round replaces an earlier one's.
        by_index.update(found)
    rounds = [by_index[i][0] for i in sorted(by_index)]
    windows = [by_index[i][1] for i in sorted(by_index)]
    incomplete -= {r.number for r in rounds}
    map_name = next((p.raw.map for p in parts if p.raw.map), "")
    players = _players(parts)
    teams = _name_teams(rounds, windows, players)

    frames: list[pd.DataFrame] = []
    deaths: list[pd.DataFrame] = []
    hurts: list[pd.DataFrame] = []
    blinds: list[pd.DataFrame] = []
    for part in parts:
        s = part.raw.samples
        s = s[s["team_num"].isin([2, 3])]
        live = s[
            _flag(s, "is_alive")
            & ~_flag(s, "is_freeze_period")
            & ~_flag(s, "is_warmup_period")
            & (s["tick"] >= part.live_from)
        ]
        origin = part.offset
        times = part.at(live["tick"].to_numpy()) - origin
        dx, dy, dz = _direction(live["yaw"].to_numpy(), live["pitch"].to_numpy())
        frames.append(
            _frame(
                {
                    "time": times,
                    "event": "position",
                    "player": live["name"].astype(str).to_numpy(),
                    "steamid": live["steamid"].astype(str).to_numpy(),
                    "side": [side_of(v) for v in live["team_num"]],
                    "x": live["X"].to_numpy(dtype=float),
                    "y": live["Y"].to_numpy(dtype=float),
                    "z": live["Z"].to_numpy(dtype=float),
                    "dirX": dx,
                    "dirY": dy,
                    "dirZ": dz,
                    "health": live["health"].to_numpy(dtype=float),
                    "armor": live["armor_value"].to_numpy(dtype=float),
                    "money": live["balance"].to_numpy(dtype=float),
                    "equipment": live["current_equip_value"].to_numpy(dtype=float),
                    "weapon": [held(w) for w in live["active_weapon_name"]],
                    "place": live["last_place_name"].fillna("").astype(str).to_numpy(),
                }
            )
        )
        ev = part.raw.events

        def live_ev(df: pd.DataFrame, part: _Part = part) -> pd.DataFrame:
            return df[(df["tick"] >= part.live_from) & ~_flag(df, "is_warmup_period")]
        if "player_death" in ev:
            d = live_ev(ev["player_death"])
            d = d.assign(time=part.at(d["tick"].to_numpy()) - origin)
            deaths.append(d)
        if "player_hurt" in ev:
            h = live_ev(ev["player_hurt"])
            hurts.append(h.assign(time=part.at(h["tick"].to_numpy()) - origin))
        if "player_blind" in ev:
            b = live_ev(ev["player_blind"])
            blinds.append(b.assign(time=part.at(b["tick"].to_numpy()) - origin))
        if "weapon_fire" in ev:
            f = live_ev(ev["weapon_fire"])
            w = f["weapon"].fillna("").astype(str).str.removeprefix("weapon_")
            keep = ~w.str.contains("knife|bayonet") & (w != "c4")
            f, w = f[keep], w[keep]
            thrown = w.isin(GRENADES).to_numpy()
            frames.append(
                _frame(
                    _event_rows(
                        part,
                        origin,
                        "user",
                        f,
                        "shot",
                        weapon=[weapon_name(v) for v in w],
                    )
                ).assign(event=np.where(thrown, "throw", "shot"))
            )
        for event_name, kind in DETONATIONS.items():
            if event_name not in ev:
                continue
            g = live_ev(ev[event_name])
            if not len(g):
                continue
            rows = _event_rows(part, origin, "user", g, kind)
            rows.update(x=g["x"].to_numpy(dtype=float), y=g["y"].to_numpy(dtype=float), z=g["z"].to_numpy(dtype=float))
            rows["weapon"] = {"smoke": "Smoke Grenade", "flash": "Flashbang", "he": "High Explosive Grenade", "molotov": "Molotov", "decoy": "Decoy Grenade"}[kind]
            frames.append(_frame(rows))
        for event_name, kind in (("bomb_planted", "plant"), ("bomb_defused", "defuse")):
            if event_name in ev and len(ev[event_name]):
                b = live_ev(ev[event_name])
                frames.append(_frame(_event_rows(part, origin, "user", b, kind, weapon="C4 Explosive")))

    kills = _kills(pd.concat(deaths, ignore_index=True) if deaths else pd.DataFrame(), rounds)
    damage = _damage(pd.concat(hurts, ignore_index=True) if hurts else pd.DataFrame(), rounds)
    flashed = _blinds(pd.concat(blinds, ignore_index=True) if blinds else pd.DataFrame(), rounds)
    frames.extend(_kill_rows(kills))
    if len(damage):
        frames.append(
            _frame(
                {
                    "time": damage["time"].to_numpy(),
                    "event": "hurt",
                    "player": damage["victim"].to_numpy(),
                    "steamid": damage["victimId"].to_numpy(),
                    "side": damage["victimSide"].to_numpy(),
                    "x": damage["x"].to_numpy(),
                    "y": damage["y"].to_numpy(),
                    "z": damage["z"].to_numpy(),
                    "place": damage["place"].to_numpy(),
                    "other": damage["attacker"].to_numpy(),
                    "damage": damage["damage"].to_numpy(),
                    "weapon": damage["weapon"].to_numpy(),
                }
            )
        )
    if len(flashed):
        frames.append(
            _frame(
                {
                    "time": flashed["time"].to_numpy(),
                    "event": "blind",
                    "player": flashed["victim"].to_numpy(),
                    "steamid": flashed["victimId"].to_numpy(),
                    "side": flashed["victimSide"].to_numpy(),
                    "x": flashed["x"].to_numpy(),
                    "y": flashed["y"].to_numpy(),
                    "z": flashed["z"].to_numpy(),
                    "place": flashed["place"].to_numpy(),
                    "other": flashed["attacker"].to_numpy(),
                    "duration": flashed["duration"].to_numpy(),
                    "weapon": "Flashbang",
                }
            )
        )
    # The bomb going off, where it was planted.
    for r in rounds:
        if r.reason == "bomb" and r.plant:
            frames.append(
                _frame(
                    {
                        "time": [r.end],
                        "event": "explode",
                        "player": [r.plant["player"]],
                        "side": ["T"],
                        "x": [r.plant["x"]],
                        "y": [r.plant["y"]],
                        "z": [r.plant["z"]],
                        "place": [f"Bombsite{r.plant['site']}"],
                        "weapon": ["C4 Explosive"],
                    }
                )
            )
    filled = [f for f in frames if len(f)]
    rows = pd.concat(filled, ignore_index=True) if filled else _frame({})
    # Events name the player; the team is the player's.
    team_of = {pid: info["team"] for pid, info in players.items()}
    missing = rows["team"].fillna("") == ""
    rows.loc[missing, "team"] = rows.loc[missing, "steamid"].map(team_of).fillna("")
    rows = _label(rows, rounds, name, map_name)
    return Match(
        name=name,
        map=map_name,
        parts=[p.raw.path for p in parts],
        rounds=rounds,
        players=players,
        teams=teams,
        rows=rows,
        kills=kills,
        damage=damage,
        blinds=flashed,
        incomplete=sorted(incomplete),
    )


def _label(rows: pd.DataFrame, rounds: list[Round], name: str, map_name: str) -> pd.DataFrame:
    """Round, phase, round time, won and buy for every row; rows outside a round are dropped."""
    which, after = _assign(rows["time"].to_numpy(dtype=float), rounds)
    keep = which >= 0
    rows = rows[keep].copy()
    which, after = which[keep], after[keep]
    numbers = np.array([r.number for r in rounds])
    starts = np.array([r.start for r in rounds])
    winners = np.array([r.winner for r in rounds])
    rows["match"] = name
    rows["map"] = map_name
    rows["round"] = numbers[which]
    rows["roundId"] = [f"{name} #{n:02d}" for n in numbers[which]]
    rows["phase"] = np.where(after, "after", "live")
    rows["roundTime"] = rows["time"].to_numpy(dtype=float) - starts[which]
    sides = rows["side"].to_numpy()
    rows["won"] = np.where(sides == "", np.nan, (winners[which] == sides).astype(float))
    rows["buy"] = [rounds[k].buy.get(s, "") for k, s in zip(which, sides)]
    return rows.sort_values(["time", "event"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- kills


def _kills(deaths: pd.DataFrame, rounds: list[Round]) -> pd.DataFrame:
    """One row per death: who killed whom where, opening, traded, trade kill, assist."""
    columns = [
        "round", "time", "roundTime", "phase", "attacker", "attackerId", "attackerSide", "victim", "victimId",
        "victimSide", "assister", "assisterId", "flashAssist", "weapon", "headshot", "teamkill", "wallbang",
        "throughSmoke", "noscope", "attackerBlind", "opening", "traded", "trade",
        "ax", "ay", "az", "aplace", "vx", "vy", "vz", "vplace",
    ]
    if not len(deaths):
        return pd.DataFrame(columns=columns)
    d = deaths.sort_values("time").reset_index(drop=True)
    which, after = _assign(d["time"].to_numpy(dtype=float), rounds)
    d = d[which >= 0].reset_index(drop=True)
    after = after[which >= 0]
    which = which[which >= 0]
    out = pd.DataFrame(
        {
            "round": [rounds[k].number for k in which],
            "time": d["time"].to_numpy(dtype=float),
            "roundTime": d["time"].to_numpy(dtype=float) - np.array([rounds[k].start for k in which]),
            "phase": np.where(after, "after", "live"),
            "attacker": d["attacker_name"].fillna("").astype(str),
            "attackerId": d["attacker_steamid"].fillna("").astype(str),
            "attackerSide": [side_of(v) for v in d["attacker_team_num"]],
            "victim": d["user_name"].fillna("").astype(str),
            "victimId": d["user_steamid"].fillna("").astype(str),
            "victimSide": [side_of(v) for v in d["user_team_num"]],
            "assister": d.get("assister_name", pd.Series([""] * len(d))).fillna("").astype(str),
            "assisterId": d.get("assister_steamid", pd.Series([""] * len(d))).fillna("").astype(str),
            "flashAssist": d.get("assistedflash", pd.Series([False] * len(d))).fillna(False).astype(bool),
            "weapon": [weapon_name(w) for w in d["weapon"]],
            "headshot": d.get("headshot", pd.Series([False] * len(d))).fillna(False).astype(bool),
            "wallbang": d.get("penetrated", pd.Series([0] * len(d))).fillna(0).astype(int) > 0,
            "throughSmoke": d.get("thrusmoke", pd.Series([False] * len(d))).fillna(False).astype(bool),
            "noscope": d.get("noscope", pd.Series([False] * len(d))).fillna(False).astype(bool),
            "attackerBlind": d.get("attackerblind", pd.Series([False] * len(d))).fillna(False).astype(bool),
            "ax": d["attacker_X"].to_numpy(dtype=float),
            "ay": d["attacker_Y"].to_numpy(dtype=float),
            "az": d["attacker_Z"].to_numpy(dtype=float),
            "aplace": d["attacker_last_place_name"].fillna("").astype(str),
            "vx": d["user_X"].to_numpy(dtype=float),
            "vy": d["user_Y"].to_numpy(dtype=float),
            "vz": d["user_Z"].to_numpy(dtype=float),
            "vplace": d["user_last_place_name"].fillna("").astype(str),
        }
    )
    out["teamkill"] = (out["attackerSide"] == out["victimSide"]) & (out["attackerId"] != "")
    enemy = (out["attackerId"] != "") & ~out["teamkill"] & (out["attackerId"] != out["victimId"])
    # Opening: the first death of each round, while it was live.
    out["opening"] = False
    live = out[out["phase"] == "live"]
    out.loc[live.groupby("round").head(1).index, "opening"] = True
    out.loc[~enemy, "opening"] = False
    # Traded: the killer killed by a teammate of the victim within TRADE_SECONDS.
    out["traded"] = False
    out["trade"] = False
    for i in out.index[enemy]:
        r, t, killer, side = out.at[i, "round"], out.at[i, "time"], out.at[i, "attackerId"], out.at[i, "victimSide"]
        later = out[
            (out["round"] == r)
            & (out["time"] > t)
            & (out["time"] <= t + TRADE_SECONDS)
            & (out["victimId"] == killer)
            & (out["attackerSide"] == side)
        ]
        if len(later):
            out.at[i, "traded"] = True
            out.at[later.index[0], "trade"] = True
    return out[columns]


def _kill_rows(kills: pd.DataFrame) -> list[pd.DataFrame]:
    """A kill at the killer's position and a death at the victim's, for the heatmap."""
    if not len(kills):
        return []
    enemy = kills[(kills["attackerId"] != "") & ~kills["teamkill"]]
    kill = _frame(
        {
            "time": enemy["time"].to_numpy(),
            "event": "kill",
            "player": enemy["attacker"].to_numpy(),
            "steamid": enemy["attackerId"].to_numpy(),
            "side": enemy["attackerSide"].to_numpy(),
            "x": enemy["ax"].to_numpy(),
            "y": enemy["ay"].to_numpy(),
            "z": enemy["az"].to_numpy(),
            "place": enemy["aplace"].to_numpy(),
            "other": enemy["victim"].to_numpy(),
            "weapon": enemy["weapon"].to_numpy(),
            "headshot": enemy["headshot"].astype(float).to_numpy(),
            "opening": enemy["opening"].astype(float).to_numpy(),
            "traded": enemy["trade"].astype(float).to_numpy(),
        }
    )
    death = _frame(
        {
            "time": kills["time"].to_numpy(),
            "event": "death",
            "player": kills["victim"].to_numpy(),
            "steamid": kills["victimId"].to_numpy(),
            "side": kills["victimSide"].to_numpy(),
            "x": kills["vx"].to_numpy(),
            "y": kills["vy"].to_numpy(),
            "z": kills["vz"].to_numpy(),
            "place": kills["vplace"].to_numpy(),
            "other": kills["attacker"].to_numpy(),
            "weapon": kills["weapon"].to_numpy(),
            "headshot": kills["headshot"].astype(float).to_numpy(),
            "opening": kills["opening"].astype(float).to_numpy(),
            "traded": kills["traded"].astype(float).to_numpy(),
        }
    )
    return [kill, death]


def _damage(hurts: pd.DataFrame, rounds: list[Round]) -> pd.DataFrame:
    """Damage between enemies, as health actually lost."""
    columns = ["round", "time", "attacker", "attackerId", "attackerSide", "victim", "victimId", "victimSide", "weapon", "damage", "utility", "x", "y", "z", "place"]
    if not len(hurts):
        return pd.DataFrame(columns=columns)
    h = hurts.sort_values("time").reset_index(drop=True)
    which, _ = _assign(h["time"].to_numpy(dtype=float), rounds)
    h = h[which >= 0].reset_index(drop=True)
    which = which[which >= 0]
    remaining: dict[tuple[int, str], float] = {}
    lost = np.zeros(len(h))
    for i, (k, victim, dmg, after) in enumerate(zip(which, h["user_steamid"].astype(str), h["dmg_health"].fillna(0), h["health"].fillna(0))):
        key = (int(k), victim)
        before = remaining.get(key, 100.0)
        lost[i] = min(float(dmg), before)
        remaining[key] = float(after)
    raw_weapon = h["weapon"].fillna("").astype(str)
    out = pd.DataFrame(
        {
            "round": [rounds[k].number for k in which],
            "time": h["time"].to_numpy(dtype=float),
            "attacker": h["attacker_name"].fillna("").astype(str),
            "attackerId": h["attacker_steamid"].fillna("").astype(str),
            "attackerSide": [side_of(v) for v in h["attacker_team_num"]],
            "victim": h["user_name"].fillna("").astype(str),
            "victimId": h["user_steamid"].fillna("").astype(str),
            "victimSide": [side_of(v) for v in h["user_team_num"]],
            "weapon": [weapon_name(w) for w in raw_weapon],
            "damage": lost,
            "utility": raw_weapon.isin(UTILITY_DAMAGE).to_numpy(),
            "x": h["user_X"].to_numpy(dtype=float),
            "y": h["user_Y"].to_numpy(dtype=float),
            "z": h["user_Z"].to_numpy(dtype=float),
            "place": h["user_last_place_name"].fillna("").astype(str),
        }
    )
    enemy = out["attackerSide"].isin(SIDES) & out["victimSide"].isin(SIDES) & (out["attackerSide"] != out["victimSide"])
    return out[enemy].reset_index(drop=True)[columns]


def _blinds(blinds: pd.DataFrame, rounds: list[Round]) -> pd.DataFrame:
    """Enemies blinded by a flash."""
    columns = ["round", "time", "attacker", "attackerId", "victim", "victimId", "victimSide", "duration", "x", "y", "z", "place"]
    if not len(blinds):
        return pd.DataFrame(columns=columns)
    b = blinds.sort_values("time").reset_index(drop=True)
    which, _ = _assign(b["time"].to_numpy(dtype=float), rounds)
    b = b[which >= 0].reset_index(drop=True)
    which = which[which >= 0]
    out = pd.DataFrame(
        {
            "round": [rounds[k].number for k in which],
            "time": b["time"].to_numpy(dtype=float),
            "attacker": b["attacker_name"].fillna("").astype(str),
            "attackerId": b["attacker_steamid"].fillna("").astype(str),
            "attackerSide": [side_of(v) for v in b["attacker_team_num"]],
            "victim": b["user_name"].fillna("").astype(str),
            "victimId": b["user_steamid"].fillna("").astype(str),
            "victimSide": [side_of(v) for v in b["user_team_num"]],
            "duration": b["blind_duration"].fillna(0).to_numpy(dtype=float),
            "x": b["user_X"].to_numpy(dtype=float),
            "y": b["user_Y"].to_numpy(dtype=float),
            "z": b["user_Z"].to_numpy(dtype=float),
            "place": b["user_last_place_name"].fillna("").astype(str),
        }
    )
    # Enemies only: not teammates, and not the coach or a caster watching.
    enemy = out["attackerSide"].isin(SIDES) & out["victimSide"].isin(SIDES) & (out["attackerSide"] != out["victimSide"])
    return out[enemy].reset_index(drop=True)[columns]
