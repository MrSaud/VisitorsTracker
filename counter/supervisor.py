"""Keeping the right cameras running, without anyone having to press Start.

Counting that depends on someone pressing a button is counting that will be
missing after a power cut, a reboot, or a Django restart -- and nobody
notices until the report is due. Two things can go wrong, and they need
different answers:

* the *source* drops out, which :class:`~counter.camera.CounterWorker`
  handles itself by reconnecting;
* the *process* goes away, taking every worker with it, which only
  something outside the worker can fix.

This module is the second one. It reconciles what is running against what
the database says should be running, and keeps doing so on a timer. Run it
as a thread inside the server (``COUNTER_AUTOSTART=1``) or as a process of
its own (``manage.py supervise``).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import namedtuple

from django.conf import settings
from django.db import close_old_connections

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 10.0

# A worker that dies the moment it starts must not be restarted in a tight
# loop. Reconnection already has its own backoff; this guards the case where
# the worker gives up and exits.
DEFAULT_COOLDOWN = 30.0

# Management commands that must never open a camera as a side effect of
# loading the app. Anything that counts does so deliberately, not by import.
NO_AUTOSTART_COMMANDS = frozenset({
    "camera_probe", "changepassword", "check", "check_health", "collectstatic",
    "createsuperuser", "dbshell", "dumpdata", "flush", "loaddata",
    "makemigrations", "migrate", "run_counter", "shell", "showmigrations",
    "sqlflush", "sqlmigrate", "squashmigrations", "startapp", "supervise",
    "test", "testserver",
})


# What one pass did. Named rather than a bare tuple: "finished" and
# "waiting" are easy to mix up, and they mean opposite things.
ReconcileResult = namedtuple(
    "ReconcileResult", "started stopped waiting finished")


class Supervisor:
    """Makes the running workers match the active cameras, and keeps them so."""

    def __init__(self, registry=None, interval=None, cooldown=DEFAULT_COOLDOWN):
        if registry is None:
            from .camera import REGISTRY as registry
        self.registry = registry
        self.interval = float(
            interval
            if interval is not None
            else getattr(settings, "COUNTER_SUPERVISE_INTERVAL", DEFAULT_INTERVAL)
        )
        self.cooldown = float(cooldown)
        self.passes = 0

        self._last_start: dict[int, float] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- one pass ---------------------------------------------------------

    def reconcile(self, now=None):
        """Start what should be running, stop what should not.

        Returns a :class:`ReconcileResult`. ``waiting`` is cameras that need
        restarting but are inside their cooldown; ``finished`` is file
        sources that played to their end, which is not a fault to fix.
        """
        from .health import FINISHED, camera_health
        from .models import Camera

        now = time.monotonic() if now is None else now
        close_old_connections()

        started: list[int] = []
        stopped: list[int] = []
        waiting: list[int] = []
        finished: list[int] = []

        cameras = list(Camera.objects.all())
        known = {camera.pk for camera in cameras}

        for camera in cameras:
            worker = self.registry.get(camera.pk)
            alive = worker is not None and worker.is_alive()

            if camera.is_active and not alive:
                if (worker is not None
                        and camera_health(camera, worker)["state"] == FINISHED):
                    # A clip that ran out has nothing left to give. Replaying
                    # it on a timer would invent visitors for ever.
                    finished.append(camera.pk)
                    continue
                previous = self._last_start.get(camera.pk)
                if previous is not None and now - previous < self.cooldown:
                    waiting.append(camera.pk)
                    continue
                self._last_start[camera.pk] = now
                self.registry.start(camera)
                started.append(camera.pk)
                logger.info("supervisor: started camera %s (%s)",
                            camera.pk, camera.name)
            elif not camera.is_active and alive:
                self.registry.stop(camera.pk)
                stopped.append(camera.pk)
                logger.info("supervisor: stopped camera %s (%s): not active",
                            camera.pk, camera.name)

        # A camera deleted from the database should not keep counting.
        for row in self.registry.stats():
            if row["camera_id"] not in known:
                self.registry.stop(row["camera_id"])
                stopped.append(row["camera_id"])
                logger.info("supervisor: stopped camera %s: no longer exists",
                            row["camera_id"])

        self.passes += 1
        return ReconcileResult(started, stopped, waiting, finished)

    # -- lifecycle --------------------------------------------------------

    def start(self):
        """Run reconciliation on a timer in a daemon thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="counter-supervisor", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self):
        logger.info("supervisor: watching cameras every %.0fs", self.interval)
        while not self._stop_event.is_set():
            try:
                self.reconcile()
            except Exception as exc:  # pragma: no cover - must never die
                # A migration not yet applied, a locked database, a camera
                # removed mid-pass: log it and try again next time.
                logger.warning("supervisor: pass failed: %s", exc)
            self._stop_event.wait(self.interval)


# ---------------------------------------------------------------------------
# autostart from AppConfig.ready()
# ---------------------------------------------------------------------------

_supervisor: Supervisor | None = None


def autostart_enabled() -> bool:
    """Whether this process should supervise cameras.

    Off unless asked for: opening a camera as a side effect of importing the
    app would surprise anyone running ``migrate``, and would fight with
    ``manage.py run_counter`` over the device.
    """
    if not getattr(settings, "COUNTER_AUTOSTART", False):
        return False

    argv = sys.argv
    command = argv[1] if len(argv) > 1 else ""
    if command in NO_AUTOSTART_COMMANDS:
        return False
    if os.path.basename(argv[0] or "").startswith("pytest"):
        return False

    # runserver's reloader runs the app twice: once in the watching parent,
    # once in the child that actually serves. Only the child should count.
    if command == "runserver" and "--noreload" not in argv:
        return os.environ.get("RUN_MAIN") == "true"

    return True


def maybe_autostart():
    """Start the supervisor thread if this process should have one."""
    global _supervisor

    if not autostart_enabled():
        return None
    if _supervisor is not None:
        return _supervisor

    _supervisor = Supervisor()
    _supervisor.start()
    return _supervisor


def current_supervisor():
    """The supervisor running in this process, if any."""
    return _supervisor
