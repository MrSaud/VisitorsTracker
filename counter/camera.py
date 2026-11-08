"""Background workers that read cameras, count crossings and serve frames.

One :class:`CounterWorker` thread per camera owns the ``cv2.VideoCapture``,
runs detection/tracking, writes :class:`~counter.models.Crossing` rows and
keeps the most recent annotated frame as JPEG bytes. HTTP requests never touch
the capture device; they only read the cached frame, so any number of browser
tabs can watch one camera.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from .tracking import build_tracker
from .vision import LineCounter, VisitorCounter, build_detector

logger = logging.getLogger(__name__)

# How long a streaming client waits for a new frame before giving up.
FRAME_WAIT_TIMEOUT = 5.0

# Consecutive read failures tolerated before a live camera is abandoned.
MAX_READ_FAILURES = 30

# How long a resolved event is reused before being looked up again. Crossings
# are infrequent, but this keeps a busy doorway from querying per person while
# still noticing an event starting or ending within a few seconds.
EVENT_CACHE_SECONDS = 10.0


class CounterWorker(threading.Thread):
    """Reads one camera and counts people crossing its line."""

    daemon = True

    def __init__(self, camera):
        super().__init__(name=f"counter-cam{camera.pk}")
        self.camera_pk = camera.pk
        self.camera_name = camera.name
        self.source = camera.resolved_source()
        self.is_file = not isinstance(self.source, int)

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

        self.error: str | None = None
        self.frames_read = 0
        self.started_at: float | None = None
        self.fps = 0.0
        # Wall-clock time of the most recent frame. A worker can be alive
        # and still be delivering nothing -- a frozen stream looks exactly
        # like a healthy one unless this is checked.
        self.last_frame_at: float | None = None
        self.finished_at: float | None = None

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
            "session_in": self.counter.total_in,
            "session_out": self.counter.total_out,
            "session_inside": self.counter.occupancy,
            "active_tracks": self.counter.active_tracks,
        }

    # -- thread body ------------------------------------------------------

    def run(self):
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            self.error = (
                f"Could not open video source {self.source!r}. "
                "For a webcam, check that the terminal running this process has "
                "camera access in System Settings > Privacy & Security > Camera."
            )
            logger.error("camera %s: %s", self.camera_pk, self.error)
            self.finished_at = time.time()
            with self._condition:
                self._condition.notify_all()
            return

        self.started_at = time.monotonic()
        failures = 0

        # A file would otherwise be consumed as fast as the CPU allows, which
        # makes playback useless to watch and distorts fps. Pace it to the
        # clip's own rate. Live cameras already pace themselves.
        frame_interval = 0.0
        if self.is_file:
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            if 0.0 < source_fps < 240.0:
                frame_interval = 1.0 / source_fps
        next_frame_at = time.monotonic()

        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()

                if not ok or frame is None:
                    if self.is_file:
                        # A video file simply ran out; that is a normal finish.
                        break
                    failures += 1
                    if failures > MAX_READ_FAILURES:
                        self.error = "Camera stopped returning frames."
                        break
                    time.sleep(0.1)
                    continue

                failures = 0
                self.frames_read += 1

                try:
                    annotated, events = self.counter.process(frame)
                except cv2.error as exc:  # pragma: no cover - defensive
                    self.error = f"OpenCV failure while processing a frame: {exc}"
                    break

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
        finally:
            self.finished_at = time.time()
            capture.release()
            close_old_connections()
            with self._condition:
                self._condition.notify_all()

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
                track_id=track_id,
                direction=direction,
                occurred_at=now,
                event_id=event_id,
            )
            for track_id, direction in events
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
