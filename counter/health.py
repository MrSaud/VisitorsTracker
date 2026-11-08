"""Is each camera actually working?

The dangerous failure here is the quiet one. A camera that dies on a
Thursday night does not raise anything: the counts simply stop, and the
report that lands on Sunday is wrong without looking wrong. Worse, a frozen
stream keeps a worker alive and "running" while delivering nothing.

So health is judged on evidence of work -- recent frames, a live thread,
no recorded error -- rather than on whether something was started.
"""

import time

from django.conf import settings
from django.utils import timezone

# Per-camera states.
OK = "ok"              # alive and delivering frames
STALLED = "stalled"    # alive but no recent frame
RECONNECTING = "reconnecting"  # alive, lost the source, trying to get it back
FAILED = "failed"      # stopped with an error
OFFLINE = "offline"    # meant to be running, but nothing is
DISABLED = "disabled"  # deliberately not running
FINISHED = "finished"  # a file source played to the end, which is not a fault

# Rolled-up states.
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_DOWN = "down"

# Higher means worse. DISABLED is deliberate, so it is not a problem.
# Reconnecting is degraded rather than down: counting has stopped, but the
# worker is still there and may well recover on its own.
SEVERITY = {OK: 0, DISABLED: 0, FINISHED: 0,
            STALLED: 1, RECONNECTING: 1,
            FAILED: 2, OFFLINE: 2}

HEALTHY_STATES = (OK, DISABLED, FINISHED)


def stale_after() -> float:
    return float(getattr(settings, "COUNTER_STALE_FRAME_SECONDS", 30) or 30)


def camera_health(camera, worker=None, last_event_at=None):
    """Assess one camera. ``worker`` is None when nothing is running."""
    threshold = stale_after()
    detail = None

    if worker is None or not worker.is_alive():
        if not camera.is_active:
            # Deactivating a camera means "stop expecting this to run". A
            # stale error from an earlier run must not alarm forever.
            state = DISABLED
            detail = "Camera is not active, so nothing is expected to run."
        elif worker is not None and worker.error:
            state = FAILED
            detail = worker.error
        elif (worker is not None
              and getattr(worker, "is_file", False)
              and worker.frames_read > 0):
            # A clip that reached its end is finished, not broken. Live
            # cameras never end on their own, so this cannot mask a fault.
            state = FINISHED
            detail = f"Video file finished after {worker.frames_read} frames."
        else:
            state = OFFLINE
            detail = (
                "Camera is marked active but no worker is running. "
                "It was never started, or the process restarted without it."
            )
        seconds_since_frame = worker.seconds_since_frame if worker else None
    else:
        seconds_since_frame = worker.seconds_since_frame
        if getattr(worker, "reconnecting", False):
            # Checked before staleness: while reconnecting there is of course
            # no recent frame, and "reconnecting" is the more useful word.
            state = RECONNECTING
            reason = worker.error or "The source stopped delivering frames."
            retry_in = getattr(worker, "retry_in", None)
            detail = reason if reason.endswith(".") else reason + "."
            if retry_in:
                detail += f" Retrying in {retry_in:.0f}s."
            if worker.reconnects:
                detail += f" Reconnected {worker.reconnects} time(s) already."
        elif seconds_since_frame is None:
            # Alive but nothing delivered yet; only a problem once it has
            # had longer than the threshold to produce a first frame.
            # started_at is a monotonic reading, so compare like with like.
            started = worker.started_at
            waiting = None if started is None else time.monotonic() - started
            if waiting is not None and waiting > threshold:
                state = STALLED
                detail = (
                    f"Running for {waiting:.0f}s without delivering a frame."
                )
            else:
                state = OK
                detail = "Starting up."
        elif seconds_since_frame > threshold:
            state = STALLED
            detail = (
                f"No frame for {seconds_since_frame:.0f}s "
                f"(threshold {threshold:.0f}s). The source may have frozen."
            )
        else:
            state = OK

    row = {
        "camera_id": camera.pk,
        "camera_name": camera.name,
        "source": camera.source,
        "is_active": camera.is_active,
        "state": state,
        "healthy": state in HEALTHY_STATES,
        "detail": detail,
        "seconds_since_frame": (
            round(seconds_since_frame, 1) if seconds_since_frame is not None else None),
        "frames_read": worker.frames_read if worker else 0,
        "reconnects": getattr(worker, "reconnects", 0) if worker else 0,
        "fps": round(worker.fps, 2) if worker else 0.0,
        "last_event_at": last_event_at.isoformat() if last_event_at else None,
        "minutes_since_event": None,
    }

    if last_event_at is not None:
        delta = timezone.now() - last_event_at
        row["minutes_since_event"] = round(delta.total_seconds() / 60.0, 1)

    return row


def overall_status(rows):
    """Roll per-camera states into one word a monitor can alert on."""
    if not rows:
        return STATUS_OK
    worst = max(SEVERITY.get(row["state"], 0) for row in rows)
    if worst >= 2:
        return STATUS_DOWN
    if worst == 1:
        return STATUS_DEGRADED
    return STATUS_OK


def build_report(cameras, registry, last_events=None):
    """Full health report for a set of cameras."""
    last_events = last_events or {}
    rows = [
        camera_health(camera, registry.get(camera.pk), last_events.get(camera.pk))
        for camera in cameras
    ]
    status = overall_status(rows)
    return {
        "status": status,
        "healthy": status == STATUS_OK,
        "checked_at": timezone.now().isoformat(),
        "stale_after_seconds": stale_after(),
        "cameras": rows,
        "problems": [r for r in rows if not r["healthy"]],
    }
