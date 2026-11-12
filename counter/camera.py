"""Background workers that read cameras, count crossings and serve frames.

One :class:`CounterWorker` thread per camera owns the ``cv2.VideoCapture``,
runs detection/tracking, writes :class:`~counter.models.Crossing` rows and
keeps the most recent annotated frame as JPEG bytes. HTTP requests never touch
the capture device; they only read the cached frame, so any number of browser
tabs can watch one camera.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import uuid

import cv2
from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from .tracking import build_tracker
from .vision import LineCounter, VisitorCounter, build_detector

logger = logging.getLogger(__name__)

# How long a streaming client waits for a new frame before giving up.
FRAME_WAIT_TIMEOUT = 5.0

# Consecutive read failures tolerated before a live stream is called lost.
MAX_READ_FAILURES = 30

# A live source that drops is reconnected to, with the wait doubling after
# each failure. RTSP streams drop for ordinary reasons -- a network blip, a
# switch hiccup, the camera rebooting its own firmware -- and an event is
# over by the time anyone notices that counting stopped at 10:20.
RECONNECT_BACKOFF_START = 1.0
RECONNECT_BACKOFF_MAX = 30.0

# How a read loop ended.
STOPPED = "stopped"    # asked to stop
FINISHED = "finished"  # a file played to its end
LOST = "lost"          # a live source stopped delivering

# How long a resolved event is reused before being looked up again. Crossings
# are infrequent, but this keeps a busy doorway from querying per person while
# still noticing an event starting or ending within a few seconds.
EVENT_CACHE_SECONDS = 10.0


def is_file_source(source) -> bool:
    """Whether a source is a clip that ends, rather than a live feed.

    This decides two things that must not be got wrong: whether running out
    of frames is a normal finish or a fault, and whether to reconnect. An
    RTSP or HTTP camera is live -- it is the case reconnection exists for --
    so a URL is never a file, however much it looks like a path.
    """
    if isinstance(source, int):
        return False
    return "://" not in str(source)


def process_token() -> str:
    """Identifies this process to the lease, and stays put for its lifetime.

    Host and pid alone are not enough: pids are reused, and a stale lease
    from a crashed process could then be renewed by an unrelated one that
    happened to get the same pid.
    """
    global _PROCESS_TOKEN
    if _PROCESS_TOKEN is None:
        _PROCESS_TOKEN = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    return _PROCESS_TOKEN


_PROCESS_TOKEN: str | None = None


def lease_looks_stale(token) -> bool:
    """Whether a lease belongs to a process that is no longer running.

    Only decidable for leases taken on this machine, and only there is it
    checked: a pid on another host means nothing here. A process killed
    with SIGTERM or SIGKILL never gets to release its lease, so without
    this a restart is refused by its own corpse until the lease expires --
    45 seconds of a desktop app that will not count.
    """
    parts = str(token or "").split(":")
    if len(parts) != 3:
        return False
    host, pid, _unique = parts
    if host != socket.gethostname():
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError):
        return True
    except PermissionError:
        # Alive, owned by somebody else.
        return False
    return False


def capture_backends(source):
    """Capture backends to try for this source, best first.

    OpenCV's default differs by platform and the default is not always the
    one that works. On Windows it is Media Foundation, which can take
    several seconds to open a webcam and fails outright on some of them,
    while DirectShow opens immediately -- so Windows gets an explicit order
    with the default kept as a last resort. A URL or a file is left alone:
    those go through FFmpeg, and naming a camera backend would break them.

    macOS is deliberately left on the default. AVFoundation is already what
    it picks, and this is the platform the project is tested on.
    """
    if not isinstance(source, int):
        return [cv2.CAP_ANY]
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def camera_privacy_location() -> str:
    """Where this platform keeps the camera permission, by name."""
    if sys.platform.startswith("win"):
        return ("Settings > Privacy & security > Camera, with "
                "\"Let desktop apps access your camera\" turned on")
    if sys.platform.startswith("linux"):
        return "your user's access to the video device (often the 'video' group)"
    return "System Settings > Privacy & Security > Camera"


class CounterWorker(threading.Thread):
    """Reads one camera and counts people crossing its line."""

    daemon = True

    def __init__(self, camera):
        super().__init__(name=f"counter-cam{camera.pk}")
        self.camera_pk = camera.pk
        self.camera_name = camera.name
        self.source = camera.resolved_source()
        self.is_file = is_file_source(self.source)

        detector = build_detector(
            settings.COUNTER_DETECTOR,
            settings.COUNTER_MIN_CONFIDENCE,
            settings.COUNTER_ONNX_MODEL,
        )
        self.counter = VisitorCounter(
            detector=detector,
            tracker=build_tracker(
                settings.COUNTER_TRACKER,
                max_disappeared=settings.COUNTER_MAX_DISAPPEARED,
                detector=detector,
            ),
            line=LineCounter(
                position=camera.line_position,
                orientation=camera.line_orientation,
                invert=camera.invert_direction,
                commit_fraction=settings.COUNTER_LINE_COMMIT_FRACTION,
                jump_fraction=settings.COUNTER_LINE_JUMP_FRACTION,
            ),
            detect_every=settings.COUNTER_DETECT_EVERY,
            max_disappeared=settings.COUNTER_MAX_DISAPPEARED,
            frame_width=settings.COUNTER_FRAME_WIDTH,
        )

        self._event_id = None
        self._event_checked_at = 0.0

        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_seq = 0

        self.lease_owner = process_token()
        self._lease_renewed_at = 0.0

        self.error: str | None = None
        self.frames_read = 0
        self.started_at: float | None = None
        self.fps = 0.0
        # Wall-clock time of the most recent frame. A worker can be alive
        # and still be delivering nothing -- a frozen stream looks exactly
        # like a healthy one unless this is checked.
        self.last_frame_at: float | None = None
        self.finished_at: float | None = None
        # Reconnection state, so a dashboard can say "trying" rather than
        # leaving an operator to guess between frozen and gone.
        self.connected_at: float | None = None
        self.reconnects = 0
        self.reconnecting = False
        self.retry_in: float | None = None

    # -- public API -------------------------------------------------------

    @property
    def detector_name(self) -> str:
        return getattr(self.counter.detector, "name", "unknown")

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self.is_alive():
            self.join(timeout=timeout)

    def snapshot(self) -> bytes | None:
        with self._condition:
            return self._jpeg

    def wait_for_frame(self, last_seq: int, timeout: float = FRAME_WAIT_TIMEOUT):
        """Block until a frame newer than ``last_seq`` exists."""
        with self._condition:
            if self._frame_seq <= last_seq:
                self._condition.wait(timeout=timeout)
            return self._jpeg, self._frame_seq

    @property
    def seconds_since_frame(self):
        """Seconds since the last delivered frame, or None if none yet."""
        if self.last_frame_at is None:
            return None
        return max(0.0, time.time() - self.last_frame_at)

    def stats(self) -> dict:
        since_frame = self.seconds_since_frame
        return {
            "camera_id": self.camera_pk,
            "camera_name": self.camera_name,
            "running": self.is_alive(),
            "detector": self.detector_name,
            "error": self.error,
            "frames_read": self.frames_read,
            "seconds_since_frame": (
                round(since_frame, 1) if since_frame is not None else None),
            "fps": round(self.fps, 2),
            "lease_owner": self.lease_owner,
            "reconnecting": self.reconnecting,
            "reconnects": self.reconnects,
            "retry_in": self.retry_in,
            "session_in": self.counter.total_in,
            "session_out": self.counter.total_out,
            "session_inside": self.counter.occupancy,
            "active_tracks": self.counter.active_tracks,
        }

    # -- the counting lease -----------------------------------------------

    def _claim(self) -> bool:
        """Take the camera, or refuse to count it.

        Without this, a second process on the same camera counts everything
        a second time. An RTSP stream serves both readers happily, so there
        is no error anywhere -- the totals simply double.
        """
        from .models import Camera

        camera = Camera.objects.filter(pk=self.camera_pk).first()
        if camera is None:
            self.error = "Camera no longer exists."
            return False

        if camera.claim_lease(self.lease_owner):
            self._lease_renewed_at = time.monotonic()
            return True

        holder = camera.lease_held_by()

        if lease_looks_stale(holder):
            # The holder is a dead process on this machine. Take it over
            # rather than wait out a lease nobody is using.
            logger.warning(
                "camera %s: taking over the lease from %s, which is gone",
                self.camera_pk, holder)
            if camera.take_lease_from(holder, self.lease_owner):
                self._lease_renewed_at = time.monotonic()
                return True
            holder = camera.lease_held_by()

        self.error = (
            f"Another process is already counting this camera ({holder}). "
            f"Two would both write crossings and double the totals."
        )
        logger.error("camera %s: %s", self.camera_pk, self.error)
        return False

    def _renew(self):
        """Keep the lease alive while we are counting."""
        from .models import LEASE_RENEW_EVERY

        now = time.monotonic()
        if now - self._lease_renewed_at < LEASE_RENEW_EVERY:
            return
        self._lease_renewed_at = now

        from .models import Camera

        camera = Camera.objects.filter(pk=self.camera_pk).first()
        if camera is None or camera.renew_lease(self.lease_owner):
            return

        # Lost it: somebody decided we were dead and took over. Stop rather
        # than count alongside them.
        self.error = (
            f"Lost the counting lease to {camera.lease_held_by() or 'another process'}."
        )
        logger.error("camera %s: %s", self.camera_pk, self.error)
        self._stop_event.set()

    def _release(self):
        from .models import Camera

        camera = Camera.objects.filter(pk=self.camera_pk).first()
        if camera is not None:
            camera.release_lease(self.lease_owner)

    # -- thread body ------------------------------------------------------

    def run(self):
        """Stay connected to the source for as long as we are wanted.

        A live source is reconnected to after a drop; the counting session
        continues across the gap, so the totals keep running while the
        tracks are thrown away.
        """
        if not self._claim():
            return

        failures = 0                       # consecutive failures to get a stream
        backoff = RECONNECT_BACKOFF_START

        try:
            while not self._stop_event.is_set():
                capture = self._open()

                if capture is not None:
                    self._mark_connected()
                    failures = 0
                    backoff = RECONNECT_BACKOFF_START
                    try:
                        outcome = self._read_frames(capture)
                    finally:
                        capture.release()
                    if outcome in (STOPPED, FINISHED):
                        break
                elif self.is_file:
                    # A file that will not open is a mistake to report, not
                    # a blip to wait out.
                    break

                failures += 1
                if not self._may_retry(failures):
                    logger.error("camera %s: giving up after %d attempt(s)",
                                 self.camera_pk, failures)
                    break
                if not self._wait_to_retry(backoff):
                    break
                backoff = min(backoff * 2.0, RECONNECT_BACKOFF_MAX)
        finally:
            self.reconnecting = False
            self.retry_in = None
            self._release()
            self.finished_at = time.time()
            close_old_connections()
            with self._condition:
                self._condition.notify_all()

    # -- connection -------------------------------------------------------

    def _open(self):
        """Open the source, or record why it could not be opened."""
        for backend in capture_backends(self.source):
            capture = (cv2.VideoCapture(self.source) if backend == cv2.CAP_ANY
                       else cv2.VideoCapture(self.source, backend))
            if capture.isOpened():
                return capture
            capture.release()
        self.error = (
            f"Could not open video source {self.source!r}. " + self._open_hint()
        )
        logger.error("camera %s: %s", self.camera_pk, self.error)
        return None

    def _open_hint(self) -> str:
        """Advice that fits the kind of source, rather than all of them."""
        if isinstance(self.source, int):
            return (
                f"Check that the process running this has camera access in "
                f"{camera_privacy_location()}, and that no other app is "
                f"holding the device."
            )
        if self.is_file:
            return "Check that the file exists and is a format OpenCV can read."
        return (
            "Check that the camera is powered and reachable on the network, "
            "and that the URL, port and credentials are right."
        )

    def _mark_connected(self):
        reconnected = self.connected_at is not None
        self.connected_at = time.monotonic()
        if self.started_at is None:
            self.started_at = self.connected_at
        self.error = None
        self.reconnecting = False
        self.retry_in = None

        if reconnected:
            self.reconnects += 1
            # The scene moved on while we were away. A surviving track would
            # be matched to whoever now stands nearest, which reads as that
            # person crossing the line.
            self.counter.forget_tracks()
            logger.info("camera %s: reconnected (%d time(s) so far)",
                        self.camera_pk, self.reconnects)

    def _may_retry(self, failures: int) -> bool:
        """Whether to try again. Zero attempts configured means never give up."""
        limit = int(getattr(settings, "COUNTER_RECONNECT_ATTEMPTS", 0) or 0)
        return limit <= 0 or failures <= limit

    def _wait_to_retry(self, seconds: float) -> bool:
        """Back off before the next attempt. False if asked to stop meanwhile."""
        self.reconnecting = True
        self.retry_in = round(seconds, 1)
        logger.warning("camera %s: %s Retrying in %.0fs.",
                       self.camera_pk, self.error or "Stream lost.", seconds)
        interrupted = self._stop_event.wait(seconds)
        self.retry_in = None
        return not interrupted

    def _read_frames(self, capture) -> str:
        """Read until the source ends, breaks, or we are asked to stop."""
        # A file would otherwise be consumed as fast as the CPU allows, which
        # makes playback useless to watch and distorts fps. Pace it to the
        # clip's own rate. Live cameras already pace themselves.
        frame_interval = 0.0
        if self.is_file:
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            if 0.0 < source_fps < 240.0:
                frame_interval = 1.0 / source_fps
        next_frame_at = time.monotonic()
        failures = 0

        while not self._stop_event.is_set():
            self._renew()
            ok, frame = capture.read()

            if not ok or frame is None:
                if self.is_file:
                    # A video file simply ran out; that is a normal finish.
                    return FINISHED
                failures += 1
                if failures > MAX_READ_FAILURES:
                    self.error = "Camera stopped returning frames."
                    return LOST
                if self._stop_event.wait(0.1):
                    return STOPPED
                continue

            failures = 0
            self.frames_read += 1

            try:
                annotated, events = self.counter.process(frame)
            except cv2.error as exc:  # pragma: no cover - defensive
                self.error = f"OpenCV failure while processing a frame: {exc}"
                return LOST

            self._publish(annotated)

            if events:
                self._persist(events)

            elapsed = time.monotonic() - (self.started_at or time.monotonic())
            if elapsed > 0:
                self.fps = self.frames_read / elapsed

            if frame_interval:
                next_frame_at += frame_interval
                lag = next_frame_at - time.monotonic()
                if lag > 0:
                    # Wake early if someone asks us to stop mid-sleep.
                    self._stop_event.wait(lag)
                else:
                    # Processing is slower than the clip; don't accumulate debt.
                    next_frame_at = time.monotonic()

        return STOPPED

    # -- internals --------------------------------------------------------

    def _publish(self, frame):
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return
        now = time.time()
        with self._condition:
            self._jpeg = buffer.tobytes()
            self._frame_seq += 1
            self.last_frame_at = now
            self._condition.notify_all()

    def _current_event_id(self):
        """Which event, if any, is running for this camera right now."""
        now = time.monotonic()
        if now - self._event_checked_at < EVENT_CACHE_SECONDS:
            return self._event_id

        from .models import Camera, active_event_for

        self._event_checked_at = now
        try:
            close_old_connections()
            camera = Camera.objects.filter(pk=self.camera_pk).first()
            event = active_event_for(camera) if camera else None
            self._event_id = event.pk if event else None
        except Exception as exc:  # pragma: no cover - counting must continue
            logger.warning("camera %s: could not resolve the current event: %s",
                           self.camera_pk, exc)
            self._event_id = None
        return self._event_id

    def _persist(self, events):
        from .models import Crossing

        now = timezone.now()
        event_id = self._current_event_id()
        rows = [
            Crossing(
                camera_id=self.camera_pk,
                track_id=event.track_id,
                direction=event.direction,
                occurred_at=now,
                event_id=event_id,
                confidence=event.confidence,
                line_offset=event.offset,
                speed=event.speed,
                box_width=event.width,
                box_height=event.height,
            )
            for event in events
        ]
        try:
            close_old_connections()
            Crossing.objects.bulk_create(rows)
        except Exception as exc:  # pragma: no cover - keep the camera alive
            logger.warning("camera %s: could not save %d crossing(s): %s",
                           self.camera_pk, len(rows), exc)


class WorkerRegistry:
    """Process-wide map of camera id -> running worker."""

    def __init__(self):
        self._workers: dict[int, CounterWorker] = {}
        self._lock = threading.Lock()

    def start(self, camera) -> CounterWorker:
        with self._lock:
            existing = self._workers.get(camera.pk)
            if existing and existing.is_alive():
                return existing
            worker = CounterWorker(camera)
            self._workers[camera.pk] = worker
            worker.start()
            return worker

    def get(self, camera_pk: int) -> CounterWorker | None:
        with self._lock:
            return self._workers.get(int(camera_pk))

    def stop(self, camera_pk: int) -> bool:
        with self._lock:
            worker = self._workers.pop(int(camera_pk), None)
        if worker is None:
            return False
        worker.stop()
        return True

    def stop_all(self):
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()

    def stats(self) -> list:
        with self._lock:
            workers = list(self._workers.values())
        return [worker.stats() for worker in workers]


REGISTRY = WorkerRegistry()


def mjpeg_frames(worker: CounterWorker):
    """Yield a multipart MJPEG stream from a worker's cached frames."""
    boundary = b"--frame"
    last_seq = -1
    while worker.is_alive() or worker.snapshot() is not None:
        jpeg, seq = worker.wait_for_frame(last_seq)
        if jpeg is None:
            if not worker.is_alive():
                break
            continue
        if seq == last_seq:
            if not worker.is_alive():
                break
            continue
        last_seq = seq
        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            + jpeg + b"\r\n"
        )
