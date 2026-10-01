"""
Camera poses from a game that publishes them.

Where a game says where it is, guessing is absurd. Forza's telemetry feed gives
position and orientation every simulation tick, exactly, for free — and using it
removes the two things that limit visual odometry: drift, which accumulates
without bound over a lap, and scale, which monocular vision cannot recover at all.

A track reconstructed with telemetry poses is metrically correct end to end. One
reconstructed from vision alone is correct locally and slowly wrong globally.

Lining up two clocks
--------------------
The hard part is not the poses, it is *when* they happened. Telemetry arrives on
a UDP socket and frames arrive from the compositor, and the game's own timestamp
counts from something unrelated to either. Three clocks, no common origin.

So the game's timestamp is not used for alignment at all. Each packet is stamped
with the monotonic time it *arrived*, which is the same clock the capture stamps
frames with, and poses are interpolated to a frame's time from the samples either
side. That inherits the network's jitter — a millisecond or two on loopback —
which at 200 km/h is about six centimetres, and is the price of not having to
solve clock synchronisation. Visual odometry's error over a lap is metres.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .odometry import Pose

#: Refuse to interpolate across a gap longer than this. A pause, an alt-tab or a
#: dropped connection leaves two samples far apart, and a straight line between
#: them is not where the car went — it is a chord across whatever it actually did.
MAX_GAP = 0.35


def pose_from_forza(frame, *, y_up: bool = True) -> Pose:
    """
    Turn one telemetry frame into a camera pose.

    Forza reports yaw, pitch and roll of the *car*, and a position in metres.
    The camera is treated as being at the car — a chase camera's offset is not
    published and guessing it would put every reconstruction a few metres behind
    itself, which is worse than the error it would fix.

    `y_up` converts from the game's Z-up world into the viewer's Y-up one, which
    is the same swap the Unreal exporter does. Reversing a handedness without
    reversing the rotation with it would mirror the level, so the rotation is
    built in viewer space rather than converted after the fact.
    """
    x, y, z = frame.x, frame.y, frame.z
    if y_up:
        # Z-up to Y-up: the game's Z becomes the viewer's Y, and the game's Y
        # becomes the viewer's Z.
        position = np.array([x, z, y], dtype=np.float64)
        yaw, pitch, roll = frame.yaw, frame.pitch, frame.roll
    else:
        position = np.array([x, y, z], dtype=np.float64)
        yaw, pitch, roll = frame.yaw, frame.pitch, frame.roll

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # Yaw about up, then pitch, then roll — the order a vehicle's attitude is
    # reported in. Composed as world-from-camera to match `Pose`.
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Pose(rotation=ry @ rp @ rr, translation=position)


@dataclass
class _Sample:
    t: float  # monotonic arrival time
    pose: Pose
    speed: float


class TelemetryPoseTrack:
    """
    A rolling buffer of poses, queryable at a frame's timestamp.

    Deliberately a *buffer* and not a recording: a scan can run for half an hour
    at sixty packets a second, and nothing needs a pose once its frame has been
    fused. Old samples are dropped.
    """

    def __init__(self, *, window: float = 30.0, y_up: bool = True):
        self.window = window
        self.y_up = y_up
        self._samples: list[_Sample] = []
        self._lock = threading.Lock()
        self.received = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)

    def add(self, frame, *, now: float | None = None) -> None:
        """Record one telemetry frame, stamped with its arrival time."""
        t = time.monotonic() if now is None else now
        pose = pose_from_forza(frame, y_up=self.y_up)
        with self._lock:
            self.received += 1
            self._samples.append(_Sample(t, pose, frame.speed))
            cutoff = t - self.window
            # Trimmed from the front; samples arrive in order, so the first one
            # still inside the window ends the search.
            drop = 0
            for sample in self._samples:
                if sample.t >= cutoff:
                    break
                drop += 1
            if drop:
                del self._samples[:drop]

    def at(self, t: float) -> Pose | None:
        """
        The pose at time `t`, interpolated. None when it cannot be known.

        None rather than the nearest sample: a frame captured during a gap in the
        feed has no pose, and pretending otherwise puts its geometry somewhere
        plausible and wrong. The reconstructor treats that as an untracked frame,
        which is exactly what it is.
        """
        with self._lock:
            samples = list(self._samples)
        if len(samples) < 2:
            return None
        if t < samples[0].t or t > samples[-1].t:
            return None

        # Binary search for the bracketing pair.
        low, high = 0, len(samples) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if samples[middle].t <= t:
                low = middle
            else:
                high = middle
        before, after = samples[low], samples[high]
        gap = after.t - before.t
        if gap > MAX_GAP:
            return None
        if gap < 1e-9:
            return before.pose

        alpha = (t - before.t) / gap
        translation = before.pose.translation * (1 - alpha) + after.pose.translation * alpha
        rotation = _slerp(before.pose.rotation, after.pose.rotation, alpha)
        return Pose(rotation=rotation, translation=translation)

    @property
    def moving(self) -> bool:
        with self._lock:
            return bool(self._samples) and self._samples[-1].speed > 0.5


def _slerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate between two rotations.

    Through the rotation *between* them rather than by blending the matrices
    elementwise. A blended matrix is not a rotation — it shrinks toward the mean
    as the angle grows, and feeding a non-orthonormal matrix into the projection
    skews every point that frame contributes.
    """
    from scipy.spatial.transform import Rotation

    relative = Rotation.from_matrix(a.T @ b)
    # `as_rotvec` scales cleanly: a fraction of the axis-angle vector is the same
    # fraction of the turn.
    partial = Rotation.from_rotvec(relative.as_rotvec() * alpha)
    return a @ partial.as_matrix()


class ForzaPoseSource:
    """
    Listens for Forza telemetry on a thread and keeps the track fed.

    Started and stopped with the scan. Failure to bind the port is reported
    rather than raised: telemetry is an *upgrade*, and a scan that would have
    worked on vision alone must not be stopped because a port was busy.
    """

    def __init__(self, *, port: int = 5300, host: str = "127.0.0.1", y_up: bool = True):
        self.port = port
        self.host = host
        self.track = TelemetryPoseTrack(y_up=y_up)
        self.error: str | None = None
        self._listener = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def active(self) -> bool:
        """True once packets are arriving and the layout has been established."""
        return self._listener is not None and self._listener.layout is not None

    @property
    def status(self) -> str:
        if self.error:
            return self.error
        if self._thread is None:
            return "not started"
        if not self.track.received:
            return f"listening on {self.host}:{self.port} — no packets yet"
        if not self.active:
            return "packets arriving; drive a few metres so the layout can be identified"
        return f"{self.track.received} packets, layout {self._listener.layout.name}"

    def start(self) -> bool:
        from ..telemetry.forza import ForzaListener

        try:
            self._listener = ForzaListener(self.host, self.port, timeout=0.5)
        except OSError as exc:
            self.error = f"could not listen on {self.host}:{self.port} — {exc}"
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        import socket

        assert self._listener is not None
        while not self._stop.is_set():
            try:
                packet, _ = self._listener._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            frame = self._listener.feed(packet)
            if frame is not None and frame.is_race_on:
                self.track.add(frame)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._listener:
            self._listener.close()
            self._listener = None
