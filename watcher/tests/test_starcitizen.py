"""Star Citizen: synthetic Game.log lines in the format community kill trackers document."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timezone

from heat3d_watch.adapters import starcitizen as sc
from heat3d_watch.adapters.base import Context
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows


def stamp(ts: float) -> str:
    return "<" + datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z>"


def death(victim: str, killer: str, weapon="GATS_BallisticGatling_S3_2677329225797", dmg="Bullet") -> str:
    return (
        f"[Notice] <Actor Death> CActor::Kill: '{victim}' [200146297631] in zone 'ANVL_Hornet_F7A_Mk2_2677329226210' "
        f"killed by '{killer}' [202061381370] using '{weapon}' [Class unknown] with damage type '{dmg}' "
        "from direction x: 0.000000, y: 0.000000, z: 0.000000 [Team_ActorTech][Actor]"
    )


def lines(t0: float) -> list[str]:
    body = [
        "[Notice] <AccountLoginCharacterStatus_Character> Character: createdAt 1700000000000 - updatedAt "
        "1700000000001 - geid 200000000001 - accountId 1234 - name Tester - state STATE_CURRENT "
        "[Team_GameServices][Login]",
        "[CSessionManager::OnClientSpawned] Spawned!",
        death("PU_Human_Enemy_GroundCombat_NPC_Pilot_2677329220001", "Tester"),
        death("Rival", "Tester"),
        death("Tester", "Rival", dmg="VehicleDestruction"),
        death("Tester", "Tester", weapon="unknown", dmg="Suicide"),
        "[Notice] <Vehicle Destruction> CVehicle::OnAdvanceDestroyLevel: Vehicle 'ANVL_Arrow_2677329221111' "
        "[2677329221111] in zone 'OOC_Stanton_1_Hurston' [pos x: 1.000000, y: 2.000000, z: 3.000000 vel x: 0.0, "
        "y: 0.0, z: 0.0] driven by 'Rival' [202061381370] advanced from destroy level 0 to 2 caused by 'Tester' "
        "[200000000001] with 'Combat' [Team_VehicleFeatures][Vehicle]",
        "[Notice] <Quantum Navtarget> CSCItemQuantumDrive::RmMulticastOnQTToPoint : Local client user "
        "Tester[200000000001] received QT data for Entity:AEGS_Gladius_2677329222222[2677329222222] to Target "
        "Stanton1b [Team_CGP2][QuantumDrive]",
        "[Notice] <Changing Solar System> CSCPlayerPUSpawningComponent: Client entity Tester [200000000001] is "
        "changing system from Stanton to Pyro [Team_Network][Spawning]",
        "[Notice] <CDisciplineServiceExternal::EndSession> Ending session [Team_GameServices][Session]",
        "a line without a timestamp",
    ]
    return [f"{stamp(t0 + i)} {b}" for i, b in enumerate(body)]


def test_the_adapter_records_kills_deaths_and_travel(tmp_path):
    live = tmp_path / "StarCitizen" / "LIVE"
    (live / "Bin64").mkdir(parents=True)
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data, exe=live / "Bin64" / "StarCitizen.exe")
    a = replace(sc.ADAPTERS[0])
    a.start(ctx)
    (live / "Game.log").write_text("\n".join(lines(time.time())) + "\n", encoding="utf-8")
    a.poll(ctx)
    a.stop(ctx)

    [session] = list((data / "sessions").glob("*/*"))
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert st["kills"] == 2 and st["npc_kills"] == 1 and st["player_kills"] == 1
    assert st["deaths"] == 2 and st["suicides"] == 1
    assert st["vehicles_destroyed"] == 1 and st["quantum_jumps"] == 1 and st["system_jumps"] == 1
    assert "nothing touches its process" in meta["source"]
    rows = read_rows(session)
    deaths = [r for r in rows if r["event"] == "death"]
    assert deaths[2]["player"] == "me" and deaths[2]["killer"] == "Rival" and deaths[2]["damage_type"] == "VehicleDestruction"
    assert deaths[0]["npc"] is True and deaths[1]["npc"] is False
    assert deaths[0]["zone"] == "ANVL_Hornet_F7A_Mk2_2677329226210"
    v = next(r for r in rows if r["event"] == "vehicle_destroyed")
    assert v["cause"] == "me" and v["player"] == "Rival" and v["destroy_level"] == 2
    q = next(r for r in rows if r["event"] == "quantum_travel")
    assert q["target"] == "Stanton1b" and q["player"] == "me"
    assert next(r for r in rows if r["event"] == "system_change")["destination"] == "Pyro"
    assert [r["event"] for r in rows][-1] == "session_end"
    assert all("x" not in r for r in rows)


def test_kills_are_not_counted_before_the_player_is_known():
    rows = []

    class S:
        started = 0.0
        stats: dict = {}

        def event(self, name, **kw):
            rows.append(name)

        def stat(self, *a):
            raise AssertionError("no stat without knowing who is playing")

    log = sc.GameLog(lambda ts: S())
    log.line(f"{stamp(time.time())} {death('A', 'B')}")
    assert rows == ["death"]


def test_the_log_is_found_beside_the_running_build():
    from pathlib import Path

    exe = Path(r"D:\RSI\StarCitizen\PTU\Bin64\StarCitizen.exe")
    assert sc.log_candidates(exe) == [Path(r"D:\RSI\StarCitizen\PTU\Game.log")]
