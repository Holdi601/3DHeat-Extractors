"""
Star Citizen: the client's own log, `<install>\\StarCitizen\\LIVE\\Game.log`.

The game writes Game.log afresh at each launch (the previous one goes to
`logbackups\\`). Only that file is read: Star Citizen runs Easy Anti-Cheat, and
nothing here goes near the game's process. Lines start with a UTC timestamp and
a level, then a tag in angle brackets:

    <2025-04-13T17:17:51.279Z> [Notice] <Actor Death> CActor::Kill: 'Victim' [200146297631] in zone
        'ANVL_Hornet_F7A_Mk2_2677329226210' killed by 'Killer' [202061381370] using
        'GATS_BallisticGatling_S3_2677329225797' [Class unknown] with damage type 'VehicleDestruction'
        from direction x: 0.000000, y: 0.000000, z: 0.000000 [Team_ActorTech][Actor]
    <...> [Notice] <Vehicle Destruction> CVehicle::OnAdvanceDestroyLevel: Vehicle 'ANVL_Arrow_123' [123] in zone
        'OOC_Stanton_1_Hurston' [pos x: 1.0, y: 2.0, z: 3.0 vel x: 0, y: 0, z: 0] driven by 'Driver' [456]
        advanced from destroy level 0 to 2 caused by 'Attacker' [789] with 'Combat' [Team_VehicleFeatures][Vehicle]
    <...> [Notice] <AccountLoginCharacterStatus_Character> Character: createdAt 1 - updatedAt 2 - geid 3 -
        accountId 4 - name Handle - state STATE_CURRENT [Team_GameServices][Login]
    <...> [Notice] <Quantum Navtarget> CSCItemQuantumDrive::RmMulticastOnQTToPoint : Local client user
        Handle[200000000001] received QT data for Entity:AEGS_Gladius_123[123] to Target Stanton1b ...
    <...> [Notice] <Changing Solar System> ... Client entity Handle ... changing system from Stanton to Pyro ...

(wrapped here; one line each in the file). The zone of a death is usually the
ship the victim sat in, not a place. The format is not documented by Cloud
Imperium and changes between patches; the patterns follow the lines community
kill trackers read (e.g. https://github.com/DimmaDont/all-slain, MIT - facts
only), and anything that does not match is skipped, not guessed. A vehicle's
`pos` is zone-relative and kept only as text: no coordinates are drawn.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..session import Axes, Session
from ._logfollow import Follower, game_started, newest
from .base import Adapter, Context

_STAMP = re.compile(r"^<(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z>")
_DEATH = re.compile(
    r"<Actor Death> CActor::Kill: '(?P<victim>[^']+)' \[(?P<vid>\d+)\] in zone '(?P<zone>[^']*)' "
    r"killed by '(?P<killer>[^']+)' \[(?P<kid>\d+)\] using '(?P<weapon>[^']*)' \[Class (?P<wclass>[^\]]*)\] "
    r"with damage type '(?P<damage>[^']*)'"
)
_VEHICLE = re.compile(
    r"<Vehicle Destruction> CVehicle::\w+: Vehicle '(?P<vehicle>[^']+)' \[\d+\] in zone '(?P<zone>[^']*)' "
    r"(?:\[pos (?P<pos>[^\]]*)\] )?driven by '(?P<driver>[^']*)' \[\d+\] advanced from destroy level "
    r"(?P<from>\d+) to (?P<to>\d+) caused by '(?P<cause>[^']*)' \[[\d_]+\] with '(?P<how>[^']*)'"
)
_CHARACTER = re.compile(r"<AccountLoginCharacterStatus_Character> Character: .*? - name (\S+) - state STATE_CURRENT")
_HANDLE = re.compile(r"User Login Success - Handle\[([^\]]+)\]")
_QUANTUM = re.compile(r"<Quantum Navtarget> .*?Local client user ([\w-]*)\[\d+\] received QT data for .*? to Target (\S+)")
_SYSTEM = re.compile(r"<Changing Solar System>.*? changing system from (\w+) to (\w+)")
_SPAWNED = re.compile(r"\[CSessionManager::OnClientSpawned\] Spawned!")
_RESPAWN = re.compile(r"Player '([\w-]+)' <(?:remote|local) client>: DoesLocationContainHospital: Searching landing zone "
                      r"location \"(.*?)\"")
_INCAP = re.compile(r"Logged an incap\.! nickname: ([\w-]+), causes: \[(.*)\]")
_END = re.compile(r"<CDisciplineServiceExternal::EndSession> Ending session")

#: Names of non-player characters: generated, with a long number on the end, or a known prefix.
_NPC = re.compile(r"^(PU_|NPC_|AIModule|Kopion|Quasigrazer|Marok|Stalker)|_\d{6,}$")


def line_time(line: str) -> float | None:
    m = _STAMP.match(line)
    if not m:
        return None
    ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    return ts + (float("0." + m.group(2)) if m.group(2) else 0.0)


def is_npc(name: str) -> bool:
    return bool(_NPC.search(name))


class GameLog:
    """Game.log lines into a session: deaths, kills, destroyed vehicles, quantum jumps."""

    def __init__(self, session_for):
        self.session_for = session_for
        self.me: str | None = None
        self.system: str | None = None

    def _who(self, name: str) -> str:
        return "me" if self.me and name == self.me else name

    def line(self, text: str) -> None:
        ts = line_time(text)
        if ts is None:
            return

        def ev(name: str, **kw) -> Session:
            s = self.session_for(ts)
            s.event(name, t=ts - s.started, ms=ts * 1000, **kw)
            return s

        if (m := _CHARACTER.search(text) or _HANDLE.search(text)) is not None:
            self.me = m.group(1)
            return
        if (m := _DEATH.search(text)) is not None:
            d = m.groupdict()
            victim, killer = d["victim"], d["killer"]
            s = ev("death", player=self._who(victim), killer=self._who(killer), victim=self._who(victim),
                   weapon=d["weapon"], weapon_class=d["wclass"], damage_type=d["damage"], zone=d["zone"],
                   npc=is_npc(victim), system=self.system)
            if self.me is not None:
                if victim == self.me:
                    s.stat("deaths", 1, "add")
                    if killer == self.me:
                        s.stat("suicides", 1, "add")
                elif killer == self.me:
                    s.stat("kills", 1, "add")
                    s.stat("npc_kills" if is_npc(victim) else "player_kills", 1, "add")
            return
        if (m := _VEHICLE.search(text)) is not None:
            d = m.groupdict()
            level = int(d["to"])
            s = ev("vehicle_destroyed" if level >= 2 else "vehicle_disabled", player=self._who(d["driver"] or "?"),
                   vehicle=d["vehicle"], zone=d["zone"], cause=self._who(d["cause"]), how=d["how"],
                   destroy_level=level, zone_pos=d["pos"], system=self.system)
            if self.me is not None and d["cause"] == self.me and d["driver"] != self.me and level >= 2:
                s.stat("vehicles_destroyed", 1, "add")
            if self.me is not None and d["driver"] == self.me and level >= 2:
                s.stat("vehicles_lost", 1, "add")
            return
        if (m := _QUANTUM.search(text)) is not None:
            s = ev("quantum_travel", player=self._who(m.group(1)) if m.group(1) else "me", target=m.group(2),
                   system=self.system)
            s.stat("quantum_jumps", 1, "add")
            return
        if (m := _SYSTEM.search(text)) is not None:
            self.system = m.group(2)
            s = ev("system_change", player="me", origin=m.group(1), destination=m.group(2))
            s.stat("system_jumps", 1, "add")
            return
        if _SPAWNED.search(text):
            ev("spawn", player="me", system=self.system)
            return
        if (m := _RESPAWN.search(text)) is not None:
            ev("respawn", player=self._who(m.group(1)), location=m.group(2), system=self.system)
            return
        if (m := _INCAP.search(text)) is not None:
            s = ev("incapacitated", player=self._who(m.group(1)), causes=m.group(2)[:300])
            if self.me is not None and m.group(1) == self.me:
                s.stat("incapacitations", 1, "add")
            return
        if _END.search(text):
            ev("session_end", player="me")


def log_candidates(exe: Path | None) -> list[Path]:
    """Game.log beside the running build's folder (LIVE, PTU, ...), else the launcher's default install."""
    if exe is not None:
        # <install>\StarCitizen\LIVE\Bin64\StarCitizen.exe
        return [exe.parent.parent / "Game.log"]
    out = []
    for base in (os.environ.get("ProgramFiles"), r"C:\Program Files"):
        if base:
            out.append(Path(base) / "Roberts Space Industries" / "StarCitizen" / "LIVE" / "Game.log")
    return out


@dataclass
class StarCitizenAdapter(Adapter):
    #: Game.log to follow; empty: found from the running game.
    path: Path | None = None
    _session: Session | None = field(default=None, repr=False)
    _follow: Follower | None = field(default=None, repr=False)
    _log: GameLog | None = field(default=None, repr=False)

    def start(self, ctx: Context) -> None:
        self._session = None
        paths = [Path(self.path)] if self.path else log_candidates(ctx.exe)

        def session_for(ts: float | None) -> Session:
            if self._session is None:
                self._session = ctx.open(self, when=ts if ts and abs(time.time() - ts) < 86400 else None)
            return self._session

        self._log = GameLog(session_for)
        self._follow = Follower(lambda: newest(p for p in paths if p.exists()), game_started(ctx.pid), line_time)

    def poll(self, ctx: Context) -> None:
        if self._follow is None or self._log is None:
            return
        for line in self._follow.lines():
            try:
                self._log.line(line)
            except (ValueError, IndexError):
                if self._session is not None:
                    self._session.stat("bad_lines", 1, "add")

    def stop(self, ctx: Context) -> None:
        self.poll(ctx)
        ctx.close(self._session)
        self._session = None
        self._follow = None
        self._log = None


ADAPTERS = [
    StarCitizenAdapter(
        id="starcitizen",
        game="Star Citizen",
        kind="log",
        exes=("StarCitizen.exe",),
        multiplayer=True,
        source=r"<install>\StarCitizen\LIVE\Game.log (the build folder of the running StarCitizen.exe), only the "
        r"log file the game writes - an Easy Anti-Cheat game: nothing touches its process",
        records=(
            "deaths with victim, killer, weapon, damage type and zone (players and NPCs)",
            "vehicles disabled and destroyed, with driver, cause and zone",
            "quantum travel targets, solar system changes, spawns, respawns, incapacitations",
            "kills (player and NPC), deaths, suicides, vehicles destroyed and lost, quantum jumps",
        ),
        enable="",
        interval=0.5,
        axes=Axes(units="no positions recorded"),
        verified="tested on synthetic Game.log lines in the format community trackers document; untested against "
        "the game",
    )
]
