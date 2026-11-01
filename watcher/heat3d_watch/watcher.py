"""
The background loop: which games are running, and an adapter running for each.

Every few seconds the process list is read (names and paths only - nothing is
opened). An adapter whose game appeared is started in a thread of its own, one
whose game is gone is stopped, and its sessions are closed and counted. The
always-on adapters - platform statistics, highlights, replay folders - run the
whole time. A failing adapter is logged and retried later; it never takes the
others with it.

While no game is running the loop also looks for a newer release, and with
`auto_update` on installs it and starts itself again, so the watcher never
interrupts a game to update.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .adapters import Adapter, Context, load
from .config import Settings

log = logging.getLogger("heat3d_watch")

#: Consecutive failing polls before an adapter is stopped until its game starts again.
MAX_FAILURES = 20


def running_processes() -> dict[str, tuple[int, Path | None]]:
    """Lower-case executable name -> (pid, full path). Names and paths only."""
    import psutil

    out: dict[str, tuple[int, Path | None]] = {}
    for p in psutil.process_iter(["pid", "name", "exe"]):
        name = (p.info.get("name") or "").lower()
        if not name:
            continue
        exe = p.info.get("exe")
        out.setdefault(name, (p.info["pid"], Path(exe) if exe else None))
    return out


class Runner(threading.Thread):
    """One adapter, polled in its own thread until told to stop."""

    def __init__(self, adapter: Adapter, ctx: Context):
        super().__init__(name=f"watch-{adapter.id}", daemon=True)
        self.adapter = adapter
        self.ctx = ctx
        self.halt = threading.Event()
        self.failed = False

    def run(self) -> None:
        a = self.adapter
        try:
            a.start(self.ctx)
        except Exception:  # noqa: BLE001
            log.exception("%s: could not start", a.id)
            self.failed = True
            return
        failures = 0
        while not self.halt.is_set():
            try:
                a.poll(self.ctx)
                failures = 0
            except Exception:  # noqa: BLE001
                failures += 1
                log.exception("%s: poll failed (%d)", a.id, failures)
                if failures >= MAX_FAILURES:
                    self.failed = True
                    break
                self.halt.wait(min(5.0, a.interval * 10 + 0.5))
                continue
            self.halt.wait(a.interval)
        try:
            a.stop(self.ctx)
        except Exception:  # noqa: BLE001
            log.exception("%s: could not stop cleanly", a.id)

    def stop(self, wait: float = 10.0) -> None:
        self.halt.set()
        self.join(wait)


class Watcher:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.adapters = load()
        self.active: dict[str, Runner] = {}
        #: Adapters that gave up, by id -> the pid they gave up on, so they are not restarted at once.
        self.benched: dict[str, int | None] = {}
        self.last_update_check = 0.0

    def wanted(self, procs: dict[str, tuple[int, Path | None]]) -> dict[str, Context]:
        """Adapters that should be running now, with the context each is started with."""
        from .safety import memory_allowed

        s = self.settings
        out: dict[str, Context] = {}
        for a in self.adapters.values():
            if a.id in s.disabled or a.missing():
                continue
            if a.kind == "replay" and not s.replays:
                continue
            if a.kind == "memory" and not s.memory:
                continue
            if a.always:
                out[a.id] = Context(s, s.data_dir)
                continue
            hit = next((procs[e.lower()] for e in a.exes if e.lower() in procs), None)
            if hit is None:
                continue
            pid, exe = hit
            if a.kind == "memory":
                verdict = memory_allowed(a, pid, exe)
                if verdict:
                    log.info("%s: not reading memory: %s", a.id, verdict)
                    continue
            out[a.id] = Context(s, s.data_dir, pid=pid, exe=exe)
        return out

    def step(self) -> None:
        procs = running_processes()
        want = self.wanted(procs)
        for aid, runner in list(self.active.items()):
            if aid not in want or not runner.is_alive():
                if runner.is_alive():
                    log.info("%s: stopping", aid)
                runner.stop()
                if runner.failed:
                    self.benched[aid] = runner.ctx.pid
                del self.active[aid]
        for aid, ctx in want.items():
            if aid in self.active:
                continue
            if aid in self.benched and self.benched[aid] == ctx.pid:
                continue
            self.benched.pop(aid, None)
            log.info("%s: starting (%s)", aid, self.adapters[aid].game)
            r = Runner(self.adapters[aid], ctx)
            self.active[aid] = r
            r.start()

    def games_running(self) -> bool:
        return any(not self.adapters[a].always for a in self.active)

    def maybe_update(self) -> None:
        if not self.settings.auto_update or self.games_running():
            return
        if time.time() - self.last_update_check < 3600:
            return
        self.last_update_check = time.time()
        try:
            import heat3d_update as up
        except ImportError:
            return
        release = up.available()
        if release is None:
            return
        log.info("3DHeat Extractors %s is out (this is %s); updating", release.version, up.installed_version())
        from .notify import notify

        notify("3DHeat Watcher", f"Updating to {release.version}")
        try:
            log.info(up.apply(release))
        except RuntimeError as e:
            log.warning("update not applied: %s", e)
            notify("3DHeat Watcher", f"{release.version} is out; not updated: {e}")
            return
        self.shutdown()
        up.restart()

    def run(self, *, once: bool = False) -> None:
        log.info("watching; data in %s", self.settings.data_dir)
        self.recover()
        try:
            while True:
                self.step()
                if once:
                    break
                self.maybe_update()
                time.sleep(self.settings.scan_seconds)
        finally:
            if not once:
                self.shutdown()

    def recover(self) -> None:
        """Finish the sessions a watcher that was killed left open."""
        from . import db
        from .session import recover

        finished = recover(self.settings.data_dir)
        if not finished:
            return
        con = db.connect(self.settings.data_dir)
        try:
            for s in finished:
                db.record(con, s)
        finally:
            con.close()
        log.info("finished %d session(s) left open by an earlier run", len(finished))

    def shutdown(self) -> None:
        for runner in self.active.values():
            runner.stop()
        self.active.clear()
