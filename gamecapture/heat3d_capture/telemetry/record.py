"""
Recording a lap's positions, and nothing else.

A scan holds the route in memory and writes it out only if the whole run
succeeds. That is the wrong trade for the route itself: the positions are the
cheapest thing in the pipeline and the hardest to get again — they need the game
running, the right track loaded, and someone to drive it. Losing a lap's worth
because the surface extractor failed afterwards is exactly what happened, and it
is what this exists to prevent.

So this writes as it goes. No depth model, no GPU, no capture: it listens on the
telemetry port, appends each position to a file, and that file is complete at
every moment. Stop it whenever, and what you have is what you drove.

What the positions are good for beyond a scan
---------------------------------------------
They are the game's own world coordinates, so they say *where on the map* a route
runs. That is the handle for finding the same stretch in the game's files — a
lap's bounding box picks out which region of terrain to look at, out of a map
that is tens of gigabytes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Recording:
    """A lap, as it was driven."""

    positions: np.ndarray
    times: np.ndarray
    speeds: np.ndarray
    #: The game that produced it, as its packet layout identified it.
    layout: str = ""

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def length(self) -> float:
        """Distance driven, in metres."""
        if len(self.positions) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The box the route occupies, in the game's own world units."""
        if len(self.positions) == 0:
            return np.zeros(3), np.zeros(3)
        return self.positions.min(axis=0), self.positions.max(axis=0)

    def summary(self) -> str:
        low, high = self.bounds
        span = high - low
        return (
            f"{len(self)} positions over {self.times[-1] - self.times[0]:.0f}s, "
            f"{self.length:.0f} m driven\n"
            f"  x {low[0]:9.1f} .. {high[0]:9.1f}   ({span[0]:.0f} m)\n"
            f"  y {low[1]:9.1f} .. {high[1]:9.1f}   ({span[1]:.0f} m)\n"
            f"  z {low[2]:9.1f} .. {high[2]:9.1f}   ({span[2]:.0f} m)\n"
            f"  top speed {self.speeds.max() * 3.6:.0f} km/h"
        )

    def save(self, path: str | Path) -> Path:
        """
        Write as one `.npz`, plus a small `.json` beside it.

        The JSON is there so the route can be read by anything — the numbers are
        of no use locked in a format that needs this project to open.
        """
        path = Path(path)
        np.savez(
            path.with_suffix(".npz"),
            positions=self.positions,
            times=self.times,
            speeds=self.speeds,
        )
        low, high = self.bounds
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "layout": self.layout,
                    "samples": len(self),
                    "seconds": float(self.times[-1] - self.times[0]) if len(self) else 0.0,
                    "metres_driven": self.length,
                    "bounds": {"min": low.tolist(), "max": high.tolist()},
                    "positions": self.positions.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        data = np.load(Path(path).with_suffix(".npz"))
        return cls(
            positions=data["positions"], times=data["times"], speeds=data["speeds"]
        )


@dataclass
class Recorder:
    """
    Listens for telemetry and keeps every position it is sent.

    Deliberately not a thread: the caller drives the loop, so a command-line tool
    can print progress and a stop is a stop rather than a flag another thread
    might not read for a while.
    """

    port: int = 5300
    host: str = "127.0.0.1"
    #: Metres a car must move before a sample is kept. Forza sends 60 packets a
    #: second and a stationary car sends them too; without this, sitting in the
    #: menu fills the file.
    min_step: float = 0.5

    positions: list = field(default_factory=list)
    times: list = field(default_factory=list)
    speeds: list = field(default_factory=list)
    packets: int = 0
    layout: str = ""
    _listener: object = None

    def __enter__(self) -> "Recorder":
        from .forza import ForzaListener

        self._listener = ForzaListener(host=self.host, port=self.port, timeout=0.5)
        return self

    def __exit__(self, *_) -> None:
        if self._listener is not None:
            self._listener.close()

    def poll(self) -> bool:
        """
        Take whatever has arrived. Returns whether a position was kept.

        A moving car only. Forza sends sixty packets a second whether or not the
        car is going anywhere, so a lap recorded without this check is mostly the
        start line, several thousand times over.
        """
        import socket

        try:
            packet, _address = self._listener._sock.recvfrom(2048)
        except (socket.timeout, TimeoutError):
            return False
        frame = self._listener.feed(packet)
        if frame is None:
            return False

        self.packets += 1
        if not self.layout and self._listener.layout is not None:
            self.layout = getattr(self._listener.layout, "name", "") or "forza"
        if not frame.is_race_on or not frame.moving:
            return False

        position = np.array(frame.position, dtype=np.float64)
        if self.positions and np.linalg.norm(position - self.positions[-1]) < self.min_step:
            return False
        self.positions.append(position)
        self.times.append(frame.race_time if frame.race_time else time.time())
        self.speeds.append(float(frame.speed))
        return True

    def finish(self) -> Recording:
        if not self.positions:
            return Recording(
                np.zeros((0, 3)), np.zeros(0), np.zeros(0), layout=self.layout
            )
        return Recording(
            positions=np.stack(self.positions),
            times=np.asarray(self.times),
            speeds=np.asarray(self.speeds),
            layout=self.layout,
        )
