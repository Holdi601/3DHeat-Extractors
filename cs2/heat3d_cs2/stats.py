"""
What each player did in a match, and how each round went: the scoreboard and
the round list a match analysis starts from. The conventions are in
`match.py`'s docstring.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .match import Match, Round


def alive_at_start(match: Match) -> dict[int, dict[str, set[str]]]:
    """Round number -> side -> steamids alive when freeze time ended."""
    rows = match.rows
    pos = rows[(rows["event"] == "position") & (rows["phase"] == "live")]
    out: dict[int, dict[str, set[str]]] = {}
    for number, group in pos.groupby("round"):
        first = group[group["roundTime"] <= group["roundTime"].min() + 1.0]
        out[int(number)] = {side: set(g["steamid"]) for side, g in first.groupby("side")}
    return out


def round_summaries(match: Match) -> list[dict]:
    """Each round: result, economy, plant, kills in order, survivors, clutches."""
    starting = alive_at_start(match)
    out = []
    for r in match.rounds:
        kills = match.kills[match.kills["round"] == r.number]
        alive = {s: set(ids) for s, ids in starting.get(r.number, {}).items()}
        clutches = []
        clutched: set[str] = set()
        for k in kills[kills["phase"] == "live"].itertuples():
            alive.get(k.victimSide, set()).discard(k.victimId)
            for side, ids in alive.items():
                enemies = sum(len(v) for s, v in alive.items() if s != side)
                if len(ids) == 1 and enemies >= 1:
                    (last,) = tuple(ids)
                    if last not in clutched:
                        clutched.add(last)
                        clutches.append({"player": match.players.get(last, {}).get("name", last), "steamid": last, "side": side, "against": enemies, "won": side == r.winner})
        out.append(
            {
                "number": r.number,
                "start": round(r.start, 3),
                "end": round(r.end, 3),
                "until": round(r.until, 3),
                "winner": r.winner,
                "winnerTeam": next((t for t, s in r.sides.items() if s == r.winner), ""),
                "reason": r.reason,
                "score": r.score,
                "sides": r.sides,
                "equipment": r.equipment,
                "buy": r.buy,
                "plant": r.plant,
                "survivors": {s: sorted(match.players.get(i, {}).get("name", i) for i in ids) for s, ids in alive.items()},
                "clutches": clutches,
                "kills": [
                    {
                        "time": round(k.roundTime, 2),
                        "attacker": k.attacker,
                        "attackerSide": k.attackerSide,
                        "victim": k.victim,
                        "victimSide": k.victimSide,
                        "assister": k.assister or None,
                        "flashAssist": bool(k.flashAssist),
                        "weapon": k.weapon,
                        "headshot": bool(k.headshot),
                        "wallbang": bool(k.wallbang),
                        "throughSmoke": bool(k.throughSmoke),
                        "opening": bool(k.opening),
                        "traded": bool(k.traded),
                        "trade": bool(k.trade),
                        "teamkill": bool(k.teamkill),
                        "after": k.phase == "after",
                        "place": k.vplace,
                    }
                    for k in kills.itertuples()
                ],
            }
        )
    return out


def player_stats(match: Match, summaries: list[dict] | None = None) -> list[dict]:
    """The scoreboard: one entry per player, best first by kills minus deaths."""
    summaries = summaries if summaries is not None else round_summaries(match)
    rounds = len(match.rounds)
    kills = match.kills
    enemy = kills[(kills["attackerId"] != "") & ~kills["teamkill"] & (kills["attackerId"] != kills["victimId"])]
    stat: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    kast: dict[str, set[int]] = defaultdict(set)
    for k in enemy.itertuples():
        a = stat[k.attackerId]
        a["kills"] += 1
        a["headshots"] += int(k.headshot)
        a["openingKills"] += int(k.opening)
        a["tradeKills"] += int(k.trade)
        a["wallbangs"] += int(k.wallbang)
        kast[k.attackerId].add(k.round)
        if k.assisterId and k.assisterId != k.attackerId:
            s = stat[k.assisterId]
            s["flashAssists" if k.flashAssist else "assists"] += 1
            kast[k.assisterId].add(k.round)
    for k in kills.itertuples():
        v = stat[k.victimId]
        v["deaths"] += 1
        v["openingDeaths"] += int(k.opening)
        v["tradedDeaths"] += int(k.traded)
        if k.traded:
            kast[k.victimId].add(k.round)
        if k.teamkill:
            stat[k.attackerId]["teamkills"] += 1
    for d in match.damage.itertuples():
        s = stat[d.attackerId]
        s["damage"] += d.damage
        if d.utility:
            s["utilityDamage"] += d.damage
    for b in match.blinds.itertuples():
        if b.duration >= 1.0:
            stat[b.attackerId]["enemiesFlashed"] += 1
    per_round = enemy.groupby(["attackerId", "round"]).size()
    for (pid, _), n in per_round.items():
        if n >= 2:
            stat[pid][f"k{min(int(n), 5)}"] += 1
    died = set(zip(kills["victimId"], kills["round"]))
    starting = alive_at_start(match)
    for number, sides in starting.items():
        for ids in sides.values():
            for pid in ids:
                stat[pid]["roundsPlayed"] += 1
                if (pid, number) not in died:
                    kast[pid].add(number)
    for summary in summaries:
        for c in summary["clutches"]:
            stat[c["steamid"]]["clutches"] += 1
            stat[c["steamid"]]["clutchesWon"] += int(c["won"])
    rows = match.rows
    for kind in ("plant", "defuse"):
        for pid in rows.loc[rows["event"] == kind, "steamid"]:
            stat[pid][kind + "s"] += 1
    out = []
    for pid, info in match.players.items():
        s = stat.get(pid, {})
        played = int(s.get("roundsPlayed", 0)) or rounds
        k, d = int(s.get("kills", 0)), int(s.get("deaths", 0))
        out.append(
            {
                "name": info["name"],
                "steamid": pid,
                "team": info["team"],
                "kills": k,
                "deaths": d,
                "assists": int(s.get("assists", 0)),
                "flashAssists": int(s.get("flashAssists", 0)),
                "diff": k - d,
                "adr": round(s.get("damage", 0.0) / played, 1),
                "kast": round(100.0 * len(kast.get(pid, set())) / played, 1),
                "headshotPct": round(100.0 * s.get("headshots", 0) / k, 1) if k else 0.0,
                "openingKills": int(s.get("openingKills", 0)),
                "openingDeaths": int(s.get("openingDeaths", 0)),
                "tradeKills": int(s.get("tradeKills", 0)),
                "tradedDeaths": int(s.get("tradedDeaths", 0)),
                "multiKills": {f"{n}k": int(s.get(f"k{n}", 0)) for n in (2, 3, 4, 5)},
                "clutches": int(s.get("clutches", 0)),
                "clutchesWon": int(s.get("clutchesWon", 0)),
                "utilityDamage": round(s.get("utilityDamage", 0.0), 1),
                "enemiesFlashed": int(s.get("enemiesFlashed", 0)),
                "plants": int(s.get("plants", 0)),
                "defuses": int(s.get("defuses", 0)),
                "teamkills": int(s.get("teamkills", 0)),
                "roundsPlayed": played,
            }
        )
    out.sort(key=lambda p: (p["team"], -p["diff"], -p["kills"]))
    return out


def final_score(match: Match) -> dict[str, int]:
    return dict(match.rounds[-1].score) if match.rounds else {}


def summary(match: Match) -> dict:
    """The whole match as one document: teams, score, rounds, players."""
    rounds = round_summaries(match)
    return {
        "format": "heat3d-cs2-match",
        "version": 1,
        "match": match.name,
        "map": match.map,
        "demos": [p.replace("\\", "/").rsplit("/", 1)[-1] for p in match.parts],
        "teams": match.teams,
        "score": final_score(match),
        "rounds": rounds,
        "incomplete": match.incomplete,
        # Rounds before the first the demo has: the recording started late.
        "missing": list(range(1, match.rounds[0].number)) if match.rounds else [],
        "players": player_stats(match, rounds),
        "conventions": {
            "tradeSeconds": 5,
            "buy": "pistol: rounds 1 and 13; else average equipment under $1500 eco, under $4000 force, otherwise full",
            "damage": "health actually lost, enemies only",
            "kast": "rounds with a kill, an assist (flash assists count), survival or a traded death",
        },
    }


def scoreboard_text(doc: dict) -> str:
    """The match document as text, for the command line."""
    lines = [f"{doc['match']} - {doc['map']}  " + "  ".join(f"{t} {s}" for t, s in doc["score"].items())]
    header = f"{'player':<20}{'K':>4}{'D':>4}{'A':>4}{'+/-':>5}{'ADR':>7}{'KAST':>7}{'HS%':>6}{'OK-OD':>7}{'clutch':>8}"
    for team in doc["teams"]:
        lines.append("")
        lines.append(team)
        lines.append(header)
        for p in (p for p in doc["players"] if p["team"] == team):
            lines.append(
                f"{p['name'][:19]:<20}{p['kills']:>4}{p['deaths']:>4}{p['assists'] + p['flashAssists']:>4}{p['diff']:>+5}"
                f"{p['adr']:>7.1f}{p['kast']:>6.1f}%{p['headshotPct']:>5.0f}%{p['openingKills']:>4}-{p['openingDeaths']:<2}"
                f"{p['clutchesWon']:>5}/{p['clutches']:<2}"
            )
    lines.append("")
    strip = []
    for r in doc["rounds"]:
        mark = {"bomb": "B", "defuse": "D", "elimination": "K", "time": "T"}.get(r["reason"], "?")
        strip.append(f"{r['number']}:{r['winner']}{mark}")
    lines.append("rounds  " + " ".join(strip))
    return "\n".join(lines)
